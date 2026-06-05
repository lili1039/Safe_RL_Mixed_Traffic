"""Diagnose the acceleration spike at the IDM->RL handover (step 50).

For each CAV around the handover it prints the local state, the decomposition of
the action into the MLP (policy) part vs the CBF safety-layer correction, and the
reward terms that are active at that state, so we can attribute the spike.
"""
import numpy as np
import torch
import pandas as pd

from idm import idm_acceleration
from _eval import (load_agent, generate_NEDC_velocity_profile, _local_obs,
                   N_FOLLOW, N_GROUPS, WARMUP_STEPS, DT, S_STAR, V_STAR,
                   A_MAX, A_MIN, LEADER_LEN, _A, _B, _DELTA, _S0)


def mlp_nominal(agent, obs):
    """The MLP mean BEFORE the safety layer (tanh-bounded to +-max_action)."""
    actor = agent.actor
    s = torch.tensor(obs, dtype=torch.float).unsqueeze(0)
    with torch.no_grad():
        s_mlp = (s - actor.state_mean) / actor.state_std
        x = actor.activate_func(actor.fc1(s_mlp))
        x = actor.activate_func(actor.fc2(x))
        mean = actor.max_action * torch.tanh(actor.mean_layer(x))
    return float(mean.flatten()[0])


def reward_terms(s_cav, v_head, v_cav, v_fv1):
    """Per-step reward terms for one CAV sub-platoon (mirrors PlatoonEnv._get_reward)."""
    eps = 1e-6
    ttc = -s_cav / (v_head - v_cav + eps)
    safety = np.log(ttc / 4) if (0 <= ttc <= 4) else 0.0
    TG = s_cav / (v_cav + eps)
    R_eff = -1.0 if TG >= 2.5 else 0.0
    stability = -((v_cav - v_head) ** 2 + (v_fv1 - v_head) ** 2)
    return ttc, safety, TG, R_eff, stability


def main():
    import sys
    # Optional CLI: checkpoint episode to inspect (default 1000).
    episode = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    print("### inspecting checkpoint episode %d ###" % episode)
    agent = load_agent("model_parameters", episode)
    leader_vel = generate_NEDC_velocity_profile()
    veh_df = pd.read_csv("vehicle_parameters.csv").iloc[:N_FOLLOW].reset_index(drop=True)

    N = N_FOLLOW + 1
    is_cav = np.array([False] + [(j % 3 == 0) for j in range(N_FOLLOW)])
    length = np.array([LEADER_LEN] + veh_df["veh_len"].tolist(), dtype=float)
    Tgap = np.array([0.0] + veh_df["Tgap"].tolist(), dtype=float)
    v0 = np.array([0.0] + veh_df["v0_IDM"].tolist(), dtype=float)
    Tgap_rate = np.array([0.0] + veh_df["Tgap_rate"].tolist(), dtype=float)
    v0_rate = np.array([0.0] + veh_df["v0_rate"].tolist(), dtype=float)
    cav_globals = [i for i in range(1, N) if is_cav[i]]

    v_init = float(leader_vel[0])
    velocity = np.full(N, v_init)
    acceleration = np.zeros(N)
    spacing = np.zeros(N)
    spacing[0] = S_STAR
    for i in range(1, N):
        denom = np.sqrt(max(1.0 - (v_init / v0[i]) ** _DELTA, 1e-3))
        spacing[i] = (_S0 + v_init * Tgap[i]) / denom + length[i - 1]
    position = np.zeros(N)
    for i in range(1, N):
        position[i] = position[i - 1] - spacing[i]

    LOG_FROM, LOG_TO = WARMUP_STEPS - 1, WARMUP_STEPS + 4
    for k in range(LOG_TO + 1):
        rl_active = k >= WARMUP_STEPS
        v_lead = float(leader_vel[k - WARMUP_STEPS]) if rl_active else v_init

        control = np.zeros(N)
        log_rows = []
        qp_rows = []
        if rl_active:
            sl = agent.actor.safeLayer
            tau = float(sl.tau) if hasattr(sl, "tau") else 0.3
            gamma = sl.gamma.detach().cpu().numpy()
            k1 = float(sl.k1.detach().cpu().numpy().flatten()[0])
            for ci, g in enumerate(cav_globals, start=1):
                obs, loc_acc = _local_obs(g, spacing, velocity, acceleration)
                nominal = mlp_nominal(agent, obs)
                a_cmd, _ = agent.act(obs, evaluate=True, acceleration=loc_acc)
                final = float(a_cmd[0])
                control[g] = float(np.clip(final, A_MIN, A_MAX))
                if LOG_FROM <= k <= LOG_TO:
                    ttc, safety, TG, R_eff, stab = reward_terms(
                        spacing[g], velocity[g - 1], velocity[g], velocity[g + 1])
                    log_rows.append((ci, g, spacing[g], spacing[g] - length[g - 1],
                                     velocity[g], velocity[g - 1], velocity[g - 1] - velocity[g],
                                     nominal, final - nominal, control[g],
                                     ttc, TG, R_eff, stab))
                if k == WARMUP_STEPS:
                    # alpha_h_j = gamma_j * barrier_j  (own-front, FV1-vs-CAV, FV2-vs-CAV)
                    s_i, v_i = obs[1], obs[5]
                    s_f1, v_f1 = obs[2], obs[6]
                    s_f2, v_f2 = obs[3], obs[7]
                    a_own = gamma[0] * (s_i - tau * v_i)
                    a_f1 = gamma[1] * (s_f1 - s_i - tau * (v_f1 - v_i))
                    a_f2 = gamma[2] * (s_f2 - s_i - tau * (v_f2 - v_i))
                    # constraint right-hand sides actually solved (stored by the layer)
                    h = sl.h.detach().cpu().numpy().reshape(-1)  # [own, f1, f2, ctrl_bound]
                    qp_rows.append((ci, g, a_own, a_f1, a_f2, h[0], h[1], h[2], h[3], final - nominal))

        if LOG_FROM <= k <= LOG_TO:
            tag = "RL" if rl_active else "IDM(warmup)"
            print("\n=== step %d  (t=%.2fs, %s)  v_lead=%.2f ===" % (k, k * DT, tag, v_lead))
            if log_rows:
                print("CAV  g   spc   netgap   v_cav  v_prec  dv     MLP    CBFcorr  final |  ttc    TG    Reff  stab")
                for r in log_rows:
                    print("%2d  %2d  %5.1f  %5.1f   %5.2f  %5.2f  %+5.2f  %+5.2f  %+6.2f  %+5.2f | %5.2f  %4.2f  %+.0f  %6.2f" % r)
            if qp_rows:
                print("  -- QP barriers (alpha_h = gamma*barrier) and solved RHS h --  gamma=%s k1=%.2f"
                      % (np.round(gamma, 2), k1))
                print("  CAV  g | alpha_own  alpha_FV1  alpha_FV2 | h_own   h_FV1   h_FV2  h_ctrlbound | z0(CBFcorr)")
                for r in qp_rows:
                    print("  %2d  %2d | %8.2f  %8.2f  %8.2f | %6.2f %7.2f %7.2f %10.2f | %+8.2f" % r)

        # integrate (identical to run_simulation)
        new_v, new_s, new_p, new_a = velocity.copy(), spacing.copy(), position.copy(), acceleration.copy()
        for i in range(N - 1, 0, -1):
            gap_rate = velocity[i - 1] - velocity[i]
            if is_cav[i] and rl_active:
                acc = control[i]
            else:
                net_gap = spacing[i] - length[i - 1]
                dv = velocity[i] - velocity[i - 1]
                acc = float(np.clip(idm_acceleration(velocity[i], dv, net_gap,
                                    [v0[i], Tgap[i], _A, _B, _DELTA, _S0]), A_MIN, A_MAX))
            new_a[i] = acc
            new_v[i] = velocity[i] + acc * DT
            new_s[i] = spacing[i] + gap_rate * DT
            new_p[i] = position[i] + new_v[i] * DT
        new_a[0] = (v_lead - velocity[0]) / DT
        new_v[0] = v_lead
        new_p[0] = position[0] + v_lead * DT
        new_s[0] = spacing[0]
        velocity, spacing, position, acceleration = new_v, new_s, new_p, new_a
        for i in range(1, N):
            if not is_cav[i]:
                Tgap[i] *= (1.0 + Tgap_rate[i] / 100.0)
                v0[i] *= (1.0 + v0_rate[i] / 100.0)


if __name__ == "__main__":
    main()
