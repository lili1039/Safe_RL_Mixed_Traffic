"""Generate qualitative string-stability figures from the trained Run A policy.

Loads the backed-up Run A actor/critic (episode 300, safety on, learnable CBF)
plus the pretrained SIDE, runs one deterministic episode per scenario on a fixed
seed, and saves velocity/spacing plots to figures/.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from platoon_env import PlatoonEnv
from ppo_agent import PPOAgent

CKPT_DIR = "model_parameters/runA_final"
OUT_DIR = "figures"
DT = 0.12


def build_args():
    a = argparse.Namespace(
        max_train_steps=int(3e6), batch_size=2048, mini_batch_size=64, hidden_width=64,
        lr_a=3e-4, lr_c=3e-4, gamma=0.99, lamda=0.95, epsilon=0.2, K_epochs=10,
        is_adv_norm=True, is_state_norm=True, is_reward_norm=False, is_reward_scaling=True,
        entropy_coef=0.01, is_lr_decay=True, is_grad_clip=True, is_orthogonal_init=True,
        adam_eps=True, is_tanh=True, safety_layer_enabled=True, cbf_tau=0.3, CAV_idx=1,
        FV1_idx=2, FV2_idx=3, safety_layer_no_grad=False, car_following_parameters=[0.5, 0.5, 0.5],
        num_episodes=300, vehicle_num=4, SIDE_update=False, lr_cf=1e-4, lr_de=1e-4,
        batch_size_SIDE=256, buffer_size_SIDE=10000, SIDE_enabled=True, SIDE_load=False,
        s_star=42, v_star=20, max_action=1.0, velocity_noise=1.0, safety_double_apply=False,
    )
    a.device = "cpu"
    return a


def rollout(agent, env, scenario, seed):
    np.random.seed(seed)
    state, acc = env.reset()
    env.select_scenario = scenario
    vel, spc, ctrl = [], [], []
    done = False
    while not done:
        action, _ = agent.act(state, evaluate=True, acceleration=acc)
        state, _, acc, done, _ = env.step(action)
        vel.append(env.get_velocity())
        spc.append(env.get_spacing())
        ctrl.append(float(action[0]))
    return np.array(vel), np.array(spc), np.array(ctrl)


def plot(vel, spc, ctrl, title, path):
    t = np.arange(len(vel)) * DT
    labels = ["Head (0)", "CAV (1)", "HDV FV1 (2)", "HDV FV2 (3)"]
    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for i in range(vel.shape[1]):
        ax[0].plot(t, vel[:, i], label=labels[i], linewidth=1.2)
    ax[0].set_ylabel("Velocity (m/s)")
    ax[0].set_title(title)
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].grid(alpha=0.3)
    for i in range(1, spc.shape[1]):  # skip head spacing (meaningless)
        ax[1].plot(t, spc[:, i], label=f"gap veh {i}", linewidth=1.2)
    ax[1].axhline(4.75, color="r", ls="--", lw=0.8, label="collision (gap=L=4.75)")
    ax[1].set_ylabel("Spacing (m)")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].grid(alpha=0.3)
    ax[2].plot(t, ctrl, color="k", linewidth=1.0)
    ax[2].set_ylabel("CAV action (m/s²)")
    ax[2].set_xlabel("Time (s)")
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("saved", path, "| min net gap =", round(float(spc[:, 1:].min()) - 4.75, 2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=CKPT_DIR)
    p.add_argument("--episode", type=int, default=300)
    p.add_argument("--prefix", default="runA")
    cli = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    args = build_args()
    init_params = {"v0": args.v_star, "s0": args.s_star, "velocity_noise": args.velocity_noise, "a_max": 5, "a_min": -5}
    env = PlatoonEnv(num_vehicles=args.vehicle_num, init_params=init_params, vehicle_length=4.75)
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_episode_steps = env.max_steps

    agent = PPOAgent(args)
    agent.load(cli.ckpt_dir, cli.episode)
    agent.SIDE_FV1.load_model(os.path.join(cli.ckpt_dir, "SIDE_FV1_"))
    agent.SIDE_FV2.load_model(os.path.join(cli.ckpt_dir, "SIDE_FV2_"))
    agent.FW1_parameters = agent.SIDE_FV1.car_following_model_parameters()
    agent.FW2_parameters = agent.SIDE_FV2.car_following_model_parameters()
    agent.actor.safeLayer.FW1_parameters = agent.FW1_parameters
    agent.actor.safeLayer.FW2_parameters = agent.FW2_parameters

    for sc, name in [(0, "scenario0_headnoise"), (1, "scenario1_braking")]:
        vel, spc, ctrl = rollout(agent, env, sc, seed=12345)
        plot(vel, spc, ctrl, f"{cli.prefix} policy — {name}", os.path.join(OUT_DIR, f"{cli.prefix}_{name}.png"))
    print("DONE")
