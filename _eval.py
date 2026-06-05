"""Evaluate the trained RL policy (episode 1000) on a 16-vehicle mixed-traffic
platoon and plot the closed-loop control results.

Platoon layout (16 vehicles, index 0 = leader):

    [Leader] [CAV HDV HDV] x 5            -> 1 leader + 15 followers

The 15 followers repeat the pattern (CAV, HDV, HDV) five times, so the CAV global
indices are 1, 4, 7, 10, 13 and each CAV has exactly two trailing HDVs. Every CAV
is driven by the SAME trained RL policy, which was trained on a local 4-vehicle
sub-platoon [head, CAV, FV1, FV2]; here each CAV sees its own local sub-platoon
[preceding vehicle, CAV, HDV, HDV].

Initial IDM parameters of the 15 followers come from `vehicle_parameters.csv`
(columns Tgap, v0_IDM, veh_len). For the first 50 steps every follower (CAVs
included) car-follows with the IDM; from step 51 the 5 CAVs are taken over by the
RL policy. Throughout the simulation each HDV's IDM parameters drift every step:

    Tgap <- Tgap * (1 + Tgap_rate/100)
    v0   <- v0   * (1 + v0_rate/100)

with Tgap_rate / v0_rate taken from the last two CSV columns.

Two leader scenarios are evaluated:
  1. NEDC-style accelerate/decelerate cruise profile.
  2. Emergency braking profile.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ppo_agent import PPOAgent
from idm import idm_acceleration, DEFAULT_IDM_PARAMS

# ------------------------------------------------------------------ constants
DT = 0.12                # simulation / leader-profile time step [s]
WARMUP_STEPS = 50        # pure-IDM warm-up before the RL policy takes over
A_MAX, A_MIN = 5.0, -5.0  # physical acceleration limits [m/s^2]
S_STAR, V_STAR = 42.0, 20.0   # equilibrium spacing / velocity (must match training)
LEADER_LEN = 4.75        # leader vehicle length [m]
N_GROUPS = 5             # number of (CAV, HDV, HDV) groups
N_FOLLOW = 3 * N_GROUPS  # 15 followers
# IDM params NOT provided by the csv keep their defaults: [a, b, delta, s0]
_A, _B, _DELTA, _S0 = (DEFAULT_IDM_PARAMS[2], DEFAULT_IDM_PARAMS[3],
                       DEFAULT_IDM_PARAMS[4], DEFAULT_IDM_PARAMS[5])


# --------------------------------------------------------------- leader profiles
def generate_NEDC_velocity_profile(Tstep=DT):
    """NEDC-style accelerate / decelerate cruise profile (m/s)."""
    segments = [10, 8, 40, 13, 40, 25, 40, 10, 80]
    total_time = sum(segments)
    time = np.arange(1, total_time, Tstep)
    vel = np.zeros_like(time)
    thresholds = np.cumsum(segments)
    for idx, t in enumerate(time):
        if t <= thresholds[0]:
            vel[idx] = 70
        elif t <= thresholds[1]:
            vel[idx] = 70 - 20 / 8 * (t - thresholds[0])
        elif t <= thresholds[2]:
            vel[idx] = 50
        elif t <= thresholds[3]:
            vel[idx] = 50 + 20 / 13 * (t - thresholds[2])
        elif t <= thresholds[4]:
            vel[idx] = 70
        elif t <= thresholds[5]:
            vel[idx] = 70 + 30 / 25 * (t - thresholds[4])
        elif t <= thresholds[6]:
            vel[idx] = 100
        elif t <= thresholds[7]:
            vel[idx] = 100 - 30 / 10 * (t - thresholds[6])
        elif t <= thresholds[8]:
            vel[idx] = 70
    return vel / 3.6


def generate_braking_velocity_profile(Tstep=DT):
    """Emergency braking profile (m/s)."""
    v_high = 20.0
    v_low = 8.0
    a_brake = -4.0
    a_accel = 4.0

    t_const1 = 1.0
    t_brake = (v_low - v_high) / a_brake
    t_const2 = 5.0
    t_accel = (v_high - v_low) / a_accel
    t_const3 = 100.0

    segments = [t_const1, t_brake, t_const2, t_accel, t_const3]
    total_time = sum(segments)
    thresholds = np.cumsum(segments)

    time = np.arange(0, total_time, Tstep)
    vel = np.zeros_like(time)
    for idx, t in enumerate(time):
        if t < thresholds[0]:
            vel[idx] = v_high
        elif t < thresholds[1]:
            vel[idx] = v_high + a_brake * (t - thresholds[0])
        elif t < thresholds[2]:
            vel[idx] = v_low
        elif t < thresholds[3]:
            vel[idx] = v_low + a_accel * (t - thresholds[2])
        else:
            vel[idx] = v_high
    return vel


# ------------------------------------------------------------------- the agent
def build_args():
    a = argparse.Namespace(
        max_train_steps=int(3e6), batch_size=2048, mini_batch_size=64, hidden_width=64,
        lr_a=3e-4, lr_c=3e-4, gamma=0.99, lamda=0.95, epsilon=0.2, K_epochs=10,
        is_adv_norm=True, is_state_norm=True, is_reward_norm=False, is_reward_scaling=True,
        entropy_coef=0.01, is_lr_decay=True, is_grad_clip=True, is_orthogonal_init=True,
        adam_eps=True, is_tanh=True, safety_layer_enabled=True, cbf_tau=0.3, CAV_idx=1,
        FV1_idx=2, FV2_idx=3, safety_layer_no_grad=False, car_following_parameters=[0.5, 0.5, 0.5],
        num_episodes=200, vehicle_num=4, SIDE_update=False, lr_cf=1e-4, lr_de=1e-4,
        batch_size_SIDE=256, buffer_size_SIDE=10000, SIDE_enabled=True, SIDE_load=False,
        s_star=42, v_star=20, max_action=1.0, velocity_noise=1.0, safety_double_apply=False,
        cbf_cav_alpha=1.0, cbf_follower_alpha=0.5, cbf_min_gap=5.0,
    )
    a.device = "cpu"
    # The policy operates on a local 4-vehicle sub-platoon: obs = 4*N - 2 = 14, action = 1.
    a.state_dim = 4 * a.vehicle_num - 2
    a.action_dim = 1
    a.max_episode_steps = 400
    return a


def load_agent(ckpt_dir, episode):
    args = build_args()
    agent = PPOAgent(args)
    agent.load(ckpt_dir, episode)
    agent.SIDE_FV1.load_model(os.path.join(ckpt_dir, "SIDE_FV1_"))
    agent.SIDE_FV2.load_model(os.path.join(ckpt_dir, "SIDE_FV2_"))
    agent.FW1_parameters = agent.SIDE_FV1.car_following_model_parameters()
    agent.FW2_parameters = agent.SIDE_FV2.car_following_model_parameters()
    agent.actor.safeLayer.FW1_parameters = agent.FW1_parameters
    agent.actor.safeLayer.FW2_parameters = agent.FW2_parameters
    return agent


# ------------------------------------------------------------------- simulator
def _local_obs(g, spacing, velocity, acceleration):
    """Build the 14-dim observation + 4-dim acceleration for the CAV at global
    index `g`, from its local sub-platoon [g-1 (head), g (CAV), g+1, g+2].

    During training the head vehicle's spacing was held constant at S_STAR (the
    lead vehicle has no real predecessor), so the local head spacing is fixed to
    S_STAR to keep the MLP input in-distribution; every other quantity is the
    true physical state of the sub-platoon.
    """
    idx = (g - 1, g, g + 1, g + 2)
    s = np.array([spacing[j] for j in idx], dtype=np.float64)
    s[0] = S_STAR
    v = np.array([velocity[j] for j in idx], dtype=np.float64)
    s_diff = s[:-1] - s[1:]
    v_diff = v[:-1] - v[1:]
    obs = np.concatenate((s, v, s_diff, v_diff)).astype(np.float32)
    loc_acc = np.array([acceleration[j] for j in idx], dtype=np.float32)
    return obs, loc_acc


def run_simulation(agent, leader_vel, veh_df):
    """Roll out the 16-vehicle platoon. Returns a history dict and metadata."""
    N = N_FOLLOW + 1  # include the leader at index 0

    # Per-vehicle static / mutable IDM parameters (index 0 = leader placeholder).
    is_cav = np.array([False] + [(j % 3 == 0) for j in range(N_FOLLOW)])
    length = np.array([LEADER_LEN] + veh_df["veh_len"].tolist(), dtype=float)
    Tgap = np.array([0.0] + veh_df["Tgap"].tolist(), dtype=float)
    v0 = np.array([0.0] + veh_df["v0_IDM"].tolist(), dtype=float)
    Tgap_rate = np.array([0.0] + veh_df["Tgap_rate"].tolist(), dtype=float)
    v0_rate = np.array([0.0] + veh_df["v0_rate"].tolist(), dtype=float)
    cav_globals = [i for i in range(1, N) if is_cav[i]]

    # ---- initial conditions: every follower at its IDM equilibrium for v_init.
    v_init = float(leader_vel[0])
    velocity = np.full(N, v_init)
    acceleration = np.zeros(N)
    spacing = np.zeros(N)
    spacing[0] = S_STAR  # leader head spacing (kept constant, matches training)
    for i in range(1, N):
        denom = np.sqrt(max(1.0 - (v_init / v0[i]) ** _DELTA, 1e-3))
        net_eq = (_S0 + v_init * Tgap[i]) / denom
        spacing[i] = net_eq + length[i - 1]
    position = np.zeros(N)
    for i in range(1, N):
        position[i] = position[i - 1] - spacing[i]

    total_steps = WARMUP_STEPS + len(leader_vel)
    hist = {k: np.zeros((total_steps, N)) for k in ("vel", "acc", "gap", "pos")}

    for k in range(total_steps):
        rl_active = k >= WARMUP_STEPS
        v_lead = float(leader_vel[k - WARMUP_STEPS]) if rl_active else v_init

        # CAV control inputs from the RL policy (based on the current state).
        control = np.zeros(N)
        if rl_active:
            for g in cav_globals:
                obs, loc_acc = _local_obs(g, spacing, velocity, acceleration)
                a_cmd, _ = agent.act(obs, evaluate=True, acceleration=loc_acc)
                control[g] = float(np.clip(a_cmd[0], A_MIN, A_MAX))

        # Integrate followers rear-to-front (preceding velocity still 'old'),
        # then move the leader. Mirrors idm.IDMModel.update.
        new_v = velocity.copy()
        new_s = spacing.copy()
        new_p = position.copy()
        new_a = acceleration.copy()
        for i in range(N - 1, 0, -1):
            gap_rate = velocity[i - 1] - velocity[i]
            if is_cav[i] and rl_active:
                acc = control[i]
            else:
                net_gap = spacing[i] - length[i - 1]
                dv = velocity[i] - velocity[i - 1]
                acc = idm_acceleration(velocity[i], dv, net_gap,
                                       [v0[i], Tgap[i], _A, _B, _DELTA, _S0])
                acc = float(np.clip(acc, A_MIN, A_MAX))
            new_a[i] = acc
            new_v[i] = velocity[i] + acc * DT
            new_s[i] = spacing[i] + gap_rate * DT
            new_p[i] = position[i] + new_v[i] * DT
        new_a[0] = (v_lead - velocity[0]) / DT
        new_v[0] = v_lead
        new_p[0] = position[0] + v_lead * DT
        new_s[0] = spacing[0]

        velocity, spacing, position, acceleration = new_v, new_s, new_p, new_a

        # Record (net / bumper-to-bumper gap for the followers).
        hist["vel"][k] = velocity
        hist["acc"][k] = acceleration
        hist["pos"][k] = position
        net_gap = spacing - np.concatenate(([0.0], length[:-1]))
        net_gap[0] = np.nan
        hist["gap"][k] = net_gap

        # Drift the HDV IDM parameters (time-varying heterogeneity).
        for i in range(1, N):
            if not is_cav[i]:
                Tgap[i] *= (1.0 + Tgap_rate[i] / 100.0)
                v0[i] *= (1.0 + v0_rate[i] / 100.0)

    return hist, cav_globals, total_steps


# -------------------------------------------------------------------- plotting
# Distinct colour per CAV (group 1..5) + dark->light grey for HDVs head->tail.
CAV_COLORS = ["#0066CC", "#E35A58", "#009999", "#7A42F4", "#FF8000"]


def _vehicle_colors(N):
    """Per-vehicle colours: leader black; CAV_g a distinct colour; HDVs shaded
    dark (head) -> light (tail) grey."""
    greys = plt.cm.Greys(np.linspace(0.6, 0.3, N - 1))
    colors = ["black"]
    for g in range(N_GROUPS):
        colors.append(CAV_COLORS[g])               # CAV of group g
        for h in range(2):                         # the group's two HDVs
            colors.append(greys[3 * g + h])
    return colors


def plot_results(hist, cav_globals, total_steps, scenario_name, out_path):
    t = np.arange(total_steps) * DT
    N = hist["vel"].shape[1]
    is_cav = np.zeros(N, dtype=bool)
    is_cav[cav_globals] = True
    veh_color = _vehicle_colors(N)

    panels = [("vel", "Velocity (m/s)"),
              ("gap", "Inter-vehicle gap (m)"),
              ("acc", r"Acceleration (m/s$^2$)")]
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    for ax, (key, ylabel) in zip(axes, panels):
        data = hist[key]
        for i in range(N):
            if i == 0:
                if key == "gap":
                    continue
                ax.plot(t, data[:, i], color="black", lw=2.0, ls="--", zorder=5)
            elif is_cav[i]:
                ax.plot(t, data[:, i], color=veh_color[i], lw=1.8, zorder=4)
            else:
                ax.plot(t, data[:, i], color=veh_color[i], lw=1.0, zorder=2)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.axvline(WARMUP_STEPS * DT, color="gray", ls=":", lw=1.2)
    axes[1].axhline(0.0, color="red", lw=1.0, alpha=0.6)  # collision threshold
    axes[-1].set_xlabel("Time (s)")

    handles = [Line2D([0], [0], color="black", ls="--", lw=2, label="Leader")]
    for g in range(N_GROUPS):
        handles.append(Line2D([0], [0], color=CAV_COLORS[g], lw=1.8, label="CAV_%d" % (g + 1)))
    handles.append(Line2D([0], [0], color=plt.cm.Greys(0.5), lw=1.0,
                          label="HDV (head→tail: dark→light)"))
    handles.append(Line2D([0], [0], color="gray", ls=":", lw=1.2,
                          label="RL takeover (step %d)" % WARMUP_STEPS))
    axes[0].legend(handles=handles, loc="best", fontsize=9, ncol=2)
    axes[0].set_title("Mixed-traffic platoon control — %s" % scenario_name, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def summarize(hist, cav_globals):
    gap = hist["gap"][:, 1:]            # follower net gaps
    min_gap = float(np.nanmin(gap))
    collided = bool(np.nanmin(gap) <= 0.0)
    return min_gap, collided


# ------------------------------------------------------------------------ main
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="model_parameters")
    p.add_argument("--episode", type=int, default=1000)
    p.add_argument("--csv", default="vehicle_parameters.csv")
    p.add_argument("--out_dir", default="figures")
    cli = p.parse_args()

    os.makedirs(cli.out_dir, exist_ok=True)
    veh_df = pd.read_csv(cli.csv).iloc[:N_FOLLOW].reset_index(drop=True)
    assert len(veh_df) == N_FOLLOW, "expected %d follower rows, got %d" % (N_FOLLOW, len(veh_df))

    agent = load_agent(cli.ckpt_dir, cli.episode)

    scenarios = [
        ("NEDC cruise", generate_NEDC_velocity_profile(), "eval_NEDC.png"),
        ("Emergency braking", generate_braking_velocity_profile(), "eval_braking.png"),
    ]

    print("Evaluating policy ep%d on a %d-vehicle platoon (1 leader + %d followers, %d CAVs)\n"
          % (cli.episode, N_FOLLOW + 1, N_FOLLOW, N_GROUPS))
    for name, profile, fname in scenarios:
        hist, cav_globals, total_steps = run_simulation(agent, profile, veh_df)
        out = plot_results(hist, cav_globals, total_steps, name,
                           os.path.join(cli.out_dir, fname))
        min_gap, collided = summarize(hist, cav_globals)
        print("  %-18s steps=%4d  duration=%6.1fs  min_gap=%6.2fm  collision=%s"
              % (name, total_steps, total_steps * DT, min_gap, collided))
        print("                     figure -> %s" % out)
