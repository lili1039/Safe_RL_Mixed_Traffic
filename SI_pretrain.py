import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data

from NN_SI import NN_SI_DE_Module
from utils import str2bool


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "SI_pretrain"
MODEL_DIR = BASE_DIR / "model_parameters"


def build_ppo_args(env, device, side_load=True):
    args = argparse.Namespace()
    args.max_train_steps = int(3e6)
    args.evaluate_freq = 5e3
    args.save_freq = 20
    args.batch_size = 2048
    args.mini_batch_size = 64
    args.hidden_width = 64
    args.lr_a = 3e-4
    args.lr_c = 3e-4
    args.gamma = 0.99
    args.lamda = 0.95
    args.epsilon = 0.2
    args.K_epochs = 10
    args.is_adv_norm = True
    args.is_state_norm = True
    args.is_reward_norm = False
    args.is_reward_scaling = True
    args.entropy_coef = 0.01
    args.is_lr_decay = True
    args.is_grad_clip = True
    args.is_orthogonal_init = True
    args.adam_eps = True
    args.is_tanh = True
    args.safety_layer_enabled = True
    args.cbf_tau = 0.3
    args.CAV_idx = 1
    args.FV1_idx = 2
    args.FV2_idx = 3
    args.safety_layer_no_grad = False
    args.car_following_parameters = [0.5, 0.5, 0.5]
    args.num_episodes = 500
    args.vehicle_num = 4
    args.SIDE_update = False
    args.lr_cf = 1e-4
    args.lr_de = 1e-4
    args.batch_size_SIDE = 256
    args.buffer_size_SIDE = 10000
    args.SIDE_enabled = True
    args.SIDE_load = side_load
    args.device = device
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = 1.0
    args.s_star = 42
    args.v_star = 20
    args.safety_double_apply = False
    args.max_episode_steps = env.max_steps
    return args


def load_pretrain_model_and_test(ppo_episode=500, max_timesteps=10000, device="cpu", side_load=True):
    from platoon_env import PlatoonEnv
    from ppo_agent import PPOAgent

    env = PlatoonEnv(max_steps=max_timesteps)
    args = build_ppo_args(env, device, side_load=side_load)
    agent = PPOAgent(args)
    agent.load(str(MODEL_DIR), ppo_episode)
    return agent, env


def test(agent, env, train_type, num_episodes=1):
    velocity_data = []
    spacing_data = []
    acceleration_data = []

    for _ in range(num_episodes):
        state, acceleration = env.reset()

        if train_type == 0:
            env.select_scenario = 0
        else:
            env_select = random.random()
            if env_select < 0.8:
                env.select_scenario = 0
            elif env_select < 0.9:
                env.select_scenario = 1
            elif env_select < 0.95:
                env.select_scenario = 2
            else:
                env.select_scenario = 3

        done = False
        while not done:
            action, _ = agent.act(state, acceleration=acceleration)
            next_state, _, next_acceleration, done, _ = env.step(action)

            state = next_state
            acceleration = next_acceleration
            velocity_data.append(env.get_velocity())
            spacing_data.append(env.get_spacing())
            acceleration_data.append(env.get_acceleration())

    return np.array(velocity_data), np.array(spacing_data), np.array(acceleration_data)


def save_rollout_data(ppo_episode, num_episodes, train_type, device, side_load=True):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    agent, env = load_pretrain_model_and_test(ppo_episode=ppo_episode, device=device, side_load=side_load)
    velocity_data, spacing_data, acceleration_data = test(agent, env, train_type, num_episodes)
    np.save(DATA_DIR / "velocity_data.npy", velocity_data)
    np.save(DATA_DIR / "spacing_data.npy", spacing_data)
    np.save(DATA_DIR / "acceleration_data.npy", acceleration_data)


def save_pure_idm_rollout_data(num_episodes=20, max_steps=400, scenarios=(0, 1), seed=0,
                               s_star=42, v_star=20, velocity_noise=1.0, num_vehicles=4,
                               domain_randomize=True, dr_range=0.15, vehicle_length=4.75):
    """Collect SIDE pretraining data from a PURE car-following (IDM) platoon.

    No PPO policy is involved: ``pure_car_following=True`` makes the CAV follow IDM
    too, so every vehicle obeys the human IDM dynamics we want SIDE to approximate.
    Only head-disturbance scenarios (0=random walk, 1=braking) are used, because
    scenarios 2/3 force a follower's acceleration (non-IDM) and would corrupt the
    car-following targets.
    """
    from platoon_env import PlatoonEnv

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    init_params = {'v0': v_star, 's0': s_star, 'velocity_noise': velocity_noise, 'a_max': 5, 'a_min': -5}
    env = PlatoonEnv(num_vehicles=num_vehicles, init_params=init_params,
                     max_steps=max_steps, pure_car_following=True,
                     vehicle_length=vehicle_length, domain_randomize=domain_randomize, dr_range=dr_range)

    np.random.seed(seed)
    random.seed(seed)
    velocity_data, spacing_data, acceleration_data = [], [], []
    for ep in range(num_episodes):
        env.reset()
        env.select_scenario = scenarios[ep % len(scenarios)]
        done = False
        while not done:
            # Action is ignored under pure_car_following; pass a dummy value.
            _, _, _, done, _ = env.step([0.0])
            velocity_data.append(env.get_velocity())
            spacing_data.append(env.get_spacing())
            acceleration_data.append(env.get_acceleration())

    np.save(DATA_DIR / "velocity_data.npy", np.array(velocity_data))
    np.save(DATA_DIR / "spacing_data.npy", np.array(spacing_data))
    np.save(DATA_DIR / "acceleration_data.npy", np.array(acceleration_data))
    print(f"Saved {len(velocity_data)} pure-IDM samples to {DATA_DIR}")


def load_rollout_data():
    required = ["velocity_data.npy", "spacing_data.npy", "acceleration_data.npy"]
    missing = [f for f in required if not (DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"SIDE pretraining data not found in {DATA_DIR} (missing: {missing}). "
            "Collect fresh rollouts from the CURRENT environment first:\n"
            "    python SI_pretrain.py --collect_data --ppo_episode 500 --num_episodes 1\n"
            "The stale 5-vehicle data has been archived under SI_pretrain/legacy_5veh/ "
            "(see its NOTE.md)."
        )
    velocity_data = np.load(DATA_DIR / "velocity_data.npy")
    spacing_data = np.load(DATA_DIR / "spacing_data.npy")
    acceleration_data = np.load(DATA_DIR / "acceleration_data.npy")
    return velocity_data, spacing_data, acceleration_data


def build_side_dataset(spacing_data, velocity_data, veh_idx, s_star=42, v_star=20):
    if veh_idx <= 0:
        raise ValueError("veh_idx must be a follower vehicle index.")
    if veh_idx >= velocity_data.shape[1]:
        raise ValueError(
            f"veh_idx={veh_idx} is out of bounds for data with "
            f"{velocity_data.shape[1]} vehicles."
        )

    state = np.stack(
        (
            spacing_data[:-1, veh_idx] - s_star,
            -(velocity_data[:-1, veh_idx] - v_star),
            velocity_data[:-1, veh_idx - 1] - v_star,
        ),
        axis=1,
    )
    next_state = np.stack(
        (
            spacing_data[1:, veh_idx] - s_star,
            -(velocity_data[1:, veh_idx] - v_star),
            velocity_data[1:, veh_idx - 1] - v_star,
        ),
        axis=1,
    )

    return Data.TensorDataset(
        torch.as_tensor(state, dtype=torch.float32),
        torch.as_tensor(next_state, dtype=torch.float32),
    )


def train_side_module(module, dataset, epochs, batch_size, device, print_every=50):
    train_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        last_loss_cf = None
        last_loss_de = None

        for state, next_state in train_loader:
            state = state.to(device=device, dtype=torch.float32)
            next_state = next_state.to(device=device, dtype=torch.float32)
            target_next_velocity = -next_state[:, 1]

            module.optimizer_cf.zero_grad()
            action_pred = module.car_following_estimator(state)
            next_state_pred_wo_de = module._get_next_state(state, action_pred)
            loss_cf = F.mse_loss(next_state_pred_wo_de, target_next_velocity)
            loss_cf.backward()
            module.optimizer_cf.step()

            module.optimizer_de.zero_grad()
            action_pred = module.car_following_estimator(state).detach()
            action_disturbance = module.disturbance_estimator(torch.cat((state, action_pred), dim=1))
            next_state_pred_w_de = module._get_next_state(state, action_pred + action_disturbance)
            loss_de = F.mse_loss(next_state_pred_w_de, target_next_velocity)
            loss_de.backward()
            module.optimizer_de.step()

            last_loss_cf = loss_cf.item()
            last_loss_de = loss_de.item()

        module.loss_cf_lst.append(last_loss_cf)
        module.loss_de_lst.append(last_loss_de)

        if epoch % print_every == 0 or epoch == epochs - 1:
            print(
                f"epoch {epoch}: loss_cf={last_loss_cf:.6f}, "
                f"loss_de={last_loss_de:.6f}, "
                f"parameters={module.car_following_model_parameters()}"
            )


def pretrain_side(args):
    velocity_data, spacing_data, _ = load_rollout_data()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = BASE_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    side_fv1 = NN_SI_DE_Module(
        3,
        1,
        args.lr_cf,
        args.lr_de,
        args.batch_size,
        args.buffer_size,
        args.device,
        args.FV1_idx,
        num_vehicles=args.vehicle_num,
    )
    side_fv2 = NN_SI_DE_Module(
        3,
        1,
        args.lr_cf,
        args.lr_de,
        args.batch_size,
        args.buffer_size,
        args.device,
        args.FV2_idx,
        num_vehicles=args.vehicle_num,
    )

    fv1_dataset = build_side_dataset(spacing_data, velocity_data, args.FV1_idx, s_star=args.s_star, v_star=args.v_star)
    fv2_dataset = build_side_dataset(spacing_data, velocity_data, args.FV2_idx, s_star=args.s_star, v_star=args.v_star)

    print("Pretraining SIDE_FV1")
    train_side_module(side_fv1, fv1_dataset, args.epochs, args.batch_size, args.device, args.print_every)
    print("Pretraining SIDE_FV2")
    train_side_module(side_fv2, fv2_dataset, args.epochs, args.batch_size, args.device, args.print_every)

    side_fv1.save_model(str(output_dir / "SIDE_FV1_"))
    side_fv2.save_model(str(output_dir / "SIDE_FV2_"))
    print(f"Saved SIDE weights to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser("Pretrain SIDE models for PPO")
    parser.add_argument("--collect_data", action="store_true", help="Collect rollout data via a trained PPO policy (legacy)")
    parser.add_argument("--collect_pure_idm", action="store_true", help="Collect data from a pure car-following (IDM) platoon (recommended; no PPO needed)")
    parser.add_argument("--collect_episodes", type=int, default=20, help="Pure-IDM collection episodes")
    parser.add_argument("--max_steps", type=int, default=400, help="Steps per collection episode")
    parser.add_argument("--scenarios", type=str, default="0,1", help="Comma-separated head-disturbance scenarios for pure-IDM collection")
    parser.add_argument("--s_star", type=float, default=42.0, help="Equilibrium spacing centre-to-centre (must match training)")
    parser.add_argument("--v_star", type=float, default=20.0, help="Equilibrium velocity (must match training)")
    parser.add_argument("--velocity_noise", type=float, default=1.0, help="Head accel noise std for collection")
    parser.add_argument("--domain_randomize", type=str2bool, nargs="?", const=True, default=True, help="Randomize IDM params during collection (match training)")
    parser.add_argument("--dr_range", type=float, default=0.15, help="Domain randomization fraction (+/-)")
    parser.add_argument("--vehicle_length", type=float, default=4.75, help="Vehicle length (IDM net gap = spacing - length)")
    parser.add_argument("--ppo_episode", type=int, default=500, help="PPO checkpoint episode used for legacy data collection")
    parser.add_argument("--num_episodes", type=int, default=1, help="Legacy PPO rollout episodes used for data collection")
    parser.add_argument("--train_type", type=int, default=0, help="0 uses scenario 0; other values sample scenarios")
    parser.add_argument("--device", type=str, default="cpu", help="Training device")
    parser.add_argument("--epochs", type=int, default=500, help="Pretraining epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="SIDE pretraining batch size")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Kept for PPO SIDE module compatibility")
    parser.add_argument("--lr_cf", type=float, default=1e-3, help="Car-following estimator learning rate")
    parser.add_argument("--lr_de", type=float, default=1e-3, help="Disturbance estimator learning rate")
    parser.add_argument("--vehicle_num", type=int, default=4, help="Number of vehicles in the platoon")
    parser.add_argument("--FV1_idx", type=int, default=2, help="First HDV follower index")
    parser.add_argument("--FV2_idx", type=int, default=3, help="Second HDV follower index")
    parser.add_argument("--output_dir", type=str, default="model_parameters", help="Directory for PPO-loadable SIDE weights")
    parser.add_argument("--print_every", type=int, default=50, help="Logging interval in epochs")
    parser.add_argument("--SIDE_load", type=str2bool, nargs="?", const=True, default=True,
                        help="During --collect_data, whether the rollout agent loads trained SIDE weights (true/false)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.collect_pure_idm:
        scenarios = tuple(int(x) for x in str(args.scenarios).split(","))
        save_pure_idm_rollout_data(num_episodes=args.collect_episodes, max_steps=args.max_steps,
                                   scenarios=scenarios, seed=0, s_star=args.s_star, v_star=args.v_star,
                                   velocity_noise=args.velocity_noise, num_vehicles=args.vehicle_num,
                                   domain_randomize=args.domain_randomize, dr_range=args.dr_range,
                                   vehicle_length=args.vehicle_length)
    elif args.collect_data:
        save_rollout_data(args.ppo_episode, args.num_episodes, args.train_type, args.device, side_load=args.SIDE_load)
    pretrain_side(args)
