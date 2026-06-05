"""Evaluate a saved checkpoint per-scenario (eval_return + collision rate)."""
import argparse, os
import numpy as np
from platoon_env import PlatoonEnv
from ppo_agent import PPOAgent
from main import evaluate_policy


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
    )
    a.device = "cpu"
    return a


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="model_parameters")
    p.add_argument("--episode", type=int, default=200)
    p.add_argument("--eval_episodes", type=int, default=6)
    p.add_argument("--reward_clip", type=float, default=15.0)
    p.add_argument("--domain_randomize", action="store_true", default=True)
    cli = p.parse_args()

    args = build_args()
    init_params = {"v0": args.v_star, "s0": args.s_star, "velocity_noise": args.velocity_noise, "a_max": 5, "a_min": -5}
    env = PlatoonEnv(num_vehicles=args.vehicle_num, init_params=init_params,
                     reward_clip=cli.reward_clip, vehicle_length=4.75, domain_randomize=cli.domain_randomize)
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

    m = evaluate_policy(agent, env, seeds=list(range(cli.eval_episodes)), scenarios=(0, 1, 2, 3))
    print("ckpt:", cli.ckpt_dir, "ep", cli.episode, "(reward_clip=%.0f, DR=%s)" % (cli.reward_clip, cli.domain_randomize))
    for sc in (0, 1, 2, 3):
        print("  scenario %d: eval_return=%8.1f  collision_rate=%.2f" % (
            sc, m["eval_return_sc%d" % sc], m["eval_collision_sc%d" % sc]))
    print("  MEAN     : eval_return=%8.1f  collision_rate=%.2f" % (m["eval_return"], m["eval_collision_rate"]))
