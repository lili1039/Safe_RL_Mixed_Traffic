import numpy as np
from ppo_agent import PPOAgent
from platoon_env import PlatoonEnv
from visualize import plot_rewards, plot_velocity_and_spacing
from utils import str2bool, RewardScaling
import pandas as pd
import torch
import matplotlib.pyplot as plt
import argparse
import wandb
import random
import os

def evaluate_policy(agent, env, seeds, scenarios=(0,)):
    '''Deterministic evaluation on FIXED seeds, per scenario (primary metric).

    Collision = net gap (= spacing - vehicle_length) of any follower <= 0.
    The global numpy RNG is saved/restored so evaluation does not perturb the
    training disturbance stream. Returns a dict with per-scenario and mean metrics.
    '''
    L = getattr(env.sim, 'vehicle_length', 0.0)
    rng_state = np.random.get_state()
    out = {}
    all_returns, all_coll = [], []
    for sc in scenarios:
        rets, colls = [], 0
        for seed in seeds:
            np.random.seed(int(seed))
            state, acceleration = env.reset()
            env.select_scenario = sc
            done = False
            ep_r = 0.0
            collided = False
            while not done:
                action, _ = agent.act(state, evaluate=True, acceleration=acceleration)
                state, reward, acceleration, done, _ = env.step(action)
                ep_r += reward
                if float(np.min(env.get_spacing()[1:])) <= L:  # follower net gap <= 0
                    collided = True
            rets.append(ep_r)
            colls += int(collided)
        out[f'eval_return_sc{sc}'] = float(np.mean(rets))
        out[f'eval_collision_sc{sc}'] = colls / max(len(seeds), 1)
        all_returns.append(out[f'eval_return_sc{sc}'])
        all_coll.append(out[f'eval_collision_sc{sc}'])
    np.random.set_state(rng_state)
    out['eval_return'] = float(np.mean(all_returns))
    out['eval_collision_rate'] = float(np.mean(all_coll))
    return out


def train(agent, env, args, reward_scaling=None, eval_env=None):
    '''
    Train the PPO agent.
    '''
    num_episodes = args.num_episodes
    SIDE_update = args.SIDE_update
    eval_seeds = list(range(args.eval_episodes))
    episode_rewards = []
    velocity_data = []
    spacing_data = []
    # Cumulative environment step counter across episodes (drives PPO lr decay)
    total_steps = 0
    # Train the agent
    for episode in range(num_episodes+1):
        # Reset the environment
        state, acceleration = env.reset()
        # Scenario selection: fixed args.train_scenario, or mixed sampling if < 0
        if args.train_scenario < 0:
            es = random.random()
            env.select_scenario = 0 if es < 0.5 else 1 if es < 0.75 else 2 if es < 0.875 else 3
        else:
            env.select_scenario = args.train_scenario

        done = False
        episode_reward = 0
        min_spacing = np.inf
        collided = False
        while not done:
            action, action_prob = agent.act(state, acceleration=acceleration)  # 每步调用.act得到动作
            next_state, reward, next_acceleration, done, _ = env.step(action)   # 环境执行
            episode_reward += reward  # log the RAW (unscaled) reward
            # Scale only the reward fed to PPO (divide by running std of returns)
            train_reward = float(reward_scaling(reward)[0]) if reward_scaling is not None else reward
            agent.step(state, action, action_prob, train_reward, next_state, done, total_steps, acceleration)  # 更新PPO并更新辨识器

            # Update the state
            state = next_state
            acceleration = next_acceleration
            total_steps += 1

            # Episode-level safety metrics (net gap of the followers)
            L = getattr(env.sim, 'vehicle_length', 0.0)
            sp = env.get_spacing()[1:]
            min_spacing = min(min_spacing, float(np.min(sp)))
            if float(np.min(sp)) <= L:
                collided = True

            # Collect data for visualization
            velocity_data.append(env.get_velocity())
            spacing_data.append(env.get_spacing())

        # Reset the discounted-return accumulator at the end of each episode
        if reward_scaling is not None:
            reward_scaling.reset()

        # Save the rewards
        episode_rewards.append(episode_reward)
        # Log to wandb: episode reward + safety + PPO health diagnostics + SIDE params
        log_dict = {'episode_reward': episode_reward,
                    'min_spacing': min_spacing,
                    'collision': int(collided)}
        log_dict.update(agent.train_metrics)  # actor/critic loss, entropy, approx_kl, clip_frac, ...
        try:
            log_dict.update({
                'FW1_alpha1': float(agent.FW1_parameters[0]),
                'FW1_alpha2': float(agent.FW1_parameters[1]),
                'FW1_alpha3': float(agent.FW1_parameters[2]),
                'FW2_alpha1': float(agent.FW2_parameters[0]),
                'FW2_alpha2': float(agent.FW2_parameters[1]),
                'FW2_alpha3': float(agent.FW2_parameters[2]),
            })
        except (TypeError, IndexError):
            pass
        side_ready = SIDE_update and len(agent.SIDE_FV1.loss_cf_lst) > 0
        if side_ready:
            log_dict.update({
                'SIDE_FV1_loss_cf': agent.SIDE_FV1.loss_cf_lst[-1],  # car-following model loss
                'SIDE_FV2_loss_cf': agent.SIDE_FV2.loss_cf_lst[-1],
                'SIDE_FV1_loss_de': agent.SIDE_FV1.loss_de_lst[-1],  # disturbance estimator loss
                'SIDE_FV2_loss_de': agent.SIDE_FV2.loss_de_lst[-1],
            })

        # Periodic deterministic evaluation on fixed seeds (PRIMARY success metric).
        # Mixed training (train_scenario<0) is evaluated on ALL scenarios 0-3 (per-scenario
        # robustness); otherwise just the configured eval scenario.
        if eval_env is not None and episode % args.eval_freq == 0:
            eval_scenarios = (0, 1, 2, 3) if args.train_scenario < 0 else (args.eval_scenario,)
            log_dict.update(evaluate_policy(agent, eval_env, eval_seeds, scenarios=eval_scenarios))

        wandb.log(log_dict, step=episode)

        # Periodically checkpoint the SIDE modules and the PPO networks
        if SIDE_update and episode % 10 == 0:
            agent.SIDE_FV1.save_model('model_parameters/SIDE_FV1_')
            agent.SIDE_FV2.save_model('model_parameters/SIDE_FV2_')
        if episode % 50 == 0:
            agent.save("model_parameters", episode)

        # Always print per-episode progress (+ per-scenario eval when available)
        sc_str = " ".join("{}={:.0f}".format(k.replace('eval_return_', ''), v)
                          for k, v in sorted(log_dict.items()) if k.startswith('eval_return_sc'))
        eval_mean = round(log_dict['eval_return'], 1) if 'eval_return' in log_dict else 'NA'
        print("Episode: {}, Reward: {:.1f}, eval(mean)={} [{}], min_gap: {:.1f}".format(
            episode, episode_reward, eval_mean, sc_str, min_spacing))

    # Convert data to NumPy arrays
    velocity_data = np.array(velocity_data)
    spacing_data = np.array(spacing_data)

    if SIDE_update:
        agent.SIDE_FV1.save_model('model_parameters/SIDE_FV1_')
        agent.SIDE_FV2.save_model('model_parameters/SIDE_FV2_')

    print('Following vehicle 1 weights:', agent.FW1_parameters, ', Following vehicle 2 weights:', agent.FW2_parameters)

    return episode_rewards, velocity_data, spacing_data


def test(agent, env, num_episodes=50):
    '''
    Test the PPO agent.
    '''
    episode_rewards = []
    velocity_data = []
    spacing_data = []
    for episode in range(num_episodes):
        # Reset the environment
        state, acceleration = env.reset()
        done = False
        episode_reward = 0

        # Run the episode
        while not done:
            action, _ = agent.act(state, evaluate=True, acceleration=acceleration)
            # Take the action
            next_state, reward, next_acceleration, done, _ = env.step(action)

            # Update the state
            state = next_state
            acceleration = next_acceleration
            episode_reward += reward
            # Collect data for visualization
            velocity_data.append(env.get_velocity())
            spacing_data.append(env.get_spacing())

        # Print the test result
        print("Test Episode: {}".format(episode))

        # Save the rewards
        episode_rewards.append(episode_reward)

    # Convert data to NumPy arrays
    velocity_data = np.array(velocity_data)
    spacing_data = np.array(spacing_data)

    return episode_rewards, velocity_data, spacing_data

if __name__ == "__main__":
    # Set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Set the safety layer
    safety_layer_enabled = True
    # 控制 safety layer 里面的 CBF 参数是否参与梯度训练
    # safety_layer_no_grad = False CBF 参数 gamma, k1 是 nn.Parameter，会跟 actor 一起被优化。
    safety_layer_no_grad = False 

    # Select the agent
    agent_select = 'ppo'
    # Set if train the agent
    agent_train = True
    # Set if update the SIDE (online human-driver identification)
    SIDE_update = True
    SIDE_enabled = True

    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--is_adv_norm", type=str2bool, nargs="?", const=True, default=True, help="Advantage normalization")
    parser.add_argument("--is_state_norm", type=str2bool, nargs="?", const=True, default=True, help="State normalization")
    parser.add_argument("--is_reward_norm", type=str2bool, nargs="?", const=True, default=False, help="Reward normalization")
    parser.add_argument("--is_reward_scaling", type=str2bool, nargs="?", const=True, default=True, help="Reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Policy entropy")
    parser.add_argument("--is_lr_decay", type=bool, default=True, help="Learning rate Decay")
    parser.add_argument("--is_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--is_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--adam_eps", type=float, default=True, help="Set Adam epsilon=1e-5")
    parser.add_argument("--is_tanh", type=float, default=True, help="Tanh activation function")
    parser.add_argument("--safety_layer_enabled", type=str2bool, nargs="?", const=True, default=safety_layer_enabled, help="Safety layer enabled or not")
    parser.add_argument("--cbf_tau", type=float, default=0.3, help="CBF time headway tau")
    parser.add_argument("--cbf_cav_alpha", type=float, default=1.0, help="CBF class-K gain for the CAV's own front barrier (fixed, full strength)")
    parser.add_argument("--cbf_follower_alpha", type=float, default=0.5, help="CBF class-K gain for the follower barriers (fixed, <1.0 = gentler gap-consistency restoration)")
    parser.add_argument("--cbf_min_gap", type=float, default=5.0, help="Minimum spacing margin in the CAV-front CBF barrier (s>=min_gap, accounts for vehicle length)")
    parser.add_argument("--CAV_idx", type=float, default=1, help="CAV index in the platoon")
    parser.add_argument("--FV1_idx", type=float, default=2, help="First HDV follower index")
    parser.add_argument("--FV2_idx", type=float, default=3, help="Second HDV follower index")
    parser.add_argument("--safety_layer_no_grad", type=str2bool, nargs="?", const=True, default=safety_layer_no_grad, help="Freeze CBF parameters (no gradient)")
    parser.add_argument("--car_following_parameters", type=list, default=[0.5,0.5,0.5], help="car following parameters initialized") #[1.2566, 1.5000, 0.9000]
    parser.add_argument("--num_episodes",type=int, default = 1000, help="number of training episodes")
    parser.add_argument("--vehicle_num",type=int, default = 4, help="number of vehicles in the platoon")
    parser.add_argument("--SIDE_update", type=str2bool, nargs="?", const=True, default=SIDE_update, help="SIDE update enabled or not")
    parser.add_argument("--lr_cf", type=float, default=1e-4, help="SI learning rate")
    parser.add_argument("--lr_de", type=float, default=1e-4, help="DE learning rate")
    parser.add_argument("--batch_size_SIDE", type=int, default=256, help="SIDE batch size")
    parser.add_argument("--buffer_size_SIDE", type=int, default=10000, help="SIDE buffer size")
    parser.add_argument("--SIDE_enabled", type=str2bool, nargs="?", const=True, default=SIDE_enabled, help="SIDE enabled or not")
    parser.add_argument("--SIDE_load", type=str2bool, nargs="?", const=True, default=True,
                        help="true: load pretrained SIDE weights as init then keep learning online; "
                             "false: learn SIDE online from scratch (ignore saved weights)")
    # --- operating point / disturbance / scenarios / evaluation / safety-fix ---
    parser.add_argument("--s_star", type=float, default=42.0, help="Equilibrium spacing centre-to-centre (reset, CBF, SI); ~IDM steady state at v=20")
    parser.add_argument("--v_star", type=float, default=20.0, help="Equilibrium (cruise) velocity (reset, CBF, SI)")
    parser.add_argument("--max_action", type=float, default=1.0, help="Nominal RL action bound (CAV accel)")
    parser.add_argument("--velocity_noise", type=float, default=1.0, help="Head accel noise std (scenario 0)")
    parser.add_argument("--train_scenario", type=int, default=0, help="Training scenario id; <0 = mixed sampling")
    parser.add_argument("--eval_scenario", type=int, default=0, help="Evaluation scenario id (single-scenario training)")
    parser.add_argument("--eval_freq", type=int, default=10, help="Evaluate every N episodes")
    parser.add_argument("--eval_episodes", type=int, default=8, help="Number of fixed-seed eval episodes")
    parser.add_argument("--safety_double_apply", type=str2bool, nargs="?", const=True, default=False,
                        help="Legacy: apply safety filter twice (biases the PPO ratio); default False")
    # --- reward clip + HDV (IDM) realism / heterogeneity / scenario magnitudes ---
    parser.add_argument("--reward_clip", type=float, default=0.0, help="Per-step reward floor magnitude (reward=max(r,-clip)); 0 disables")
    parser.add_argument("--vehicle_length", type=float, default=4.75, help="Vehicle length; IDM net gap = spacing - length")
    parser.add_argument("--domain_randomize", type=str2bool, nargs="?", const=True, default=False, help="Per-episode per-vehicle IDM parameter randomization")
    parser.add_argument("--dr_range", type=float, default=0.15, help="Domain randomization fraction (+/-) on IDM params")
    parser.add_argument("--brake_accel", type=float, default=5.0, help="Head emergency-braking magnitude (scenario 1)")
    parser.add_argument("--accel_mag", type=float, default=1.0, help="Follower forced-acceleration magnitude (scenario 2/3)")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed")
    parser.add_argument("--run_name", type=str, default="", help="Optional wandb run-name suffix")
    args = parser.parse_args()

    # Reproducibility (training disturbance stream is seeded so runs are comparable)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build the IDM environment(s) using the configured operating point / disturbance.
    init_params = {'v0': args.v_star, 's0': args.s_star,
                   'velocity_noise': args.velocity_noise, 'a_max': 5, 'a_min': -5}
    env_kwargs = dict(num_vehicles=args.vehicle_num, init_params=init_params,
                      reward_clip=args.reward_clip, vehicle_length=args.vehicle_length,
                      domain_randomize=args.domain_randomize, dr_range=args.dr_range,
                      brake_accel=args.brake_accel, accel_mag=args.accel_mag)
    env = PlatoonEnv(**env_kwargs)
    eval_env = PlatoonEnv(**env_kwargs)

    args.device = device
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_episode_steps = env.max_steps
    agent = PPOAgent(args)

    # Train new model or load pretrained model
    if agent_train:
        # Initialize wandb. Defaults to offline; set WANDB_MODE=online to upload.
        run_name = 'platoon_' + agent_select + '_sl_' + str(safety_layer_enabled)
        if args.run_name:
            run_name += '_' + args.run_name
        wandb.init(project="safe-rl-mixed-traffic",
                   name=run_name,
                   config=vars(args),
                   mode=os.getenv("WANDB_MODE", "offline"))
        # Reward scaling: divide PPO rewards by the running std of discounted returns
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma) if args.is_reward_scaling else None
        # Train the agent
        episode_rewards, velocity_data, spacing_data = train(agent, env, args, reward_scaling=reward_scaling, eval_env=eval_env)
        wandb.finish()
        # Save the training data
        os.makedirs('training_traj', exist_ok=True)
        if safety_layer_enabled:
            episode_rewards_pd = pd.DataFrame(episode_rewards).to_csv('training_traj/episode_rewards_' + agent_select + '.csv')
            velocity_data_pd = pd.DataFrame(velocity_data).to_csv('training_traj/velocity_data_' + agent_select + '.csv')
            spacing_data_pd = pd.DataFrame(spacing_data).to_csv('training_traj/spacing_data_' + agent_select + '.csv')
        else:
            episode_rewards_pd = pd.DataFrame(episode_rewards).to_csv('training_traj/episode_rewards_no_safety_' + agent_select + '.csv')
            velocity_data_pd = pd.DataFrame(velocity_data).to_csv('training_traj/velocity_data_no_safety_' + agent_select + '.csv')
            spacing_data_pd = pd.DataFrame(spacing_data).to_csv('training_traj/spacing_data_no_safety_' + agent_select + '.csv')
        # Visualize the collected data
        plot_rewards(episode_rewards, episode=args.num_episodes)
        plot_velocity_and_spacing(velocity_data, spacing_data)
        plt.show()
    else:
        agent.load("model_parameters", 500)
        # Test the agent
        episode_rewards, velocity_data, spacing_data = test(agent, env)
        # Visualize the collected data
        plot_rewards(episode_rewards, episode=50)
        plot_velocity_and_spacing(velocity_data, spacing_data)
        plt.show()


