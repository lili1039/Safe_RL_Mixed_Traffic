import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Normal
from utils import ReplayBuffer_PPO, equilibrium_state_stats
import os
from BarrierNet import BarrierLayer
from NN_SI import NN_SI_DE_Module
import numpy as np

# Orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        # Initialize the parameters of the network
        self.max_action = args.max_action
        # Physical actuator limit (m/s^2); the safety-corrected mean may exceed the
        # nominal [-max_action, max_action] range, so the executed action is clamped to
        # this physical bound (matching the IDM internal clip) rather than to max_action.
        self.a_phys = getattr(args, 'a_phys', 5.0)
        # When True keep the legacy (buggy) behaviour of applying the safety filter a
        # second time to the sampled action; default False applies it once (in the mean),
        # so the stored action matches the sampling distribution and the PPO ratio is consistent.
        self.safety_double_apply = getattr(args, 'safety_double_apply', False)
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.is_tanh]  #  use tanh
        self.safety_layer_enabled = args.safety_layer_enabled
        self.tau = args.cbf_tau
        self.CAV_index = int(args.CAV_idx)
        self.safety_layer_no_grad = args.safety_layer_no_grad
        self.car_following_parameters = args.car_following_parameters
        # Fixed state normalization applied ONLY to the MLP input (the safety layer below
        # still receives the raw physical state). Centres spacing/velocity at the
        # equilibrium so the tanh layers do not saturate. Disabled (identity) when
        # is_state_norm is False, for ablation.
        if getattr(args, 'is_state_norm', True):
            s_mean, s_std = equilibrium_state_stats(args.vehicle_num, getattr(args, 's_star', 25), getattr(args, 'v_star', 20))
        else:
            s_mean = np.zeros(args.state_dim, dtype=np.float32)
            s_std = np.ones(args.state_dim, dtype=np.float32)
        self.register_buffer('state_mean', torch.tensor(s_mean))
        self.register_buffer('state_std', torch.tensor(s_std))
        if self.safety_layer_enabled or self.safety_layer_no_grad:
            self.safeLayer = BarrierLayer(args.state_dim, self.car_following_parameters, self.safety_layer_no_grad, SIDE_enabled=args.SIDE_enabled, num_vehicles=args.vehicle_num, s_star=getattr(args, 's_star', 25), v_star=getattr(args, 'v_star', 20),
                                          cav_alpha=getattr(args, 'cbf_cav_alpha', 1.0), follower_alpha=getattr(args, 'cbf_follower_alpha', 0.5), min_gap=getattr(args, 'cbf_min_gap', 5.0))
        else:
            self.safeLayer = None
        # Use orthogonal initialization
        if args.is_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s, acceleration = None, cf_saturation_FW1 = None, cf_saturation_FW2 = None):
        # Get the mean of the Gaussian distribution based on the current state.
        # The MLP sees the NORMALIZED state; the safety layer sees the RAW state.
        s_mlp = (s - self.state_mean) / self.state_std
        x = self.activate_func(self.fc1(s_mlp))
        x = self.activate_func(self.fc2(x))
        mean = self.max_action * torch.tanh(self.mean_layer(x))  # [-1,1]->[-max_action,max_action]
        if self.safety_layer_enabled or self.safety_layer_no_grad:
            mean_safe = self.safeLayer(mean, s, self.tau, self.CAV_index, acceleration, cf_saturation_FW1, cf_saturation_FW2)
            mean = mean + mean_safe
        return mean

    def get_dist(self, s, acceleration = None, cf_saturation_FW1 = None, cf_saturation_FW2 = None):
        # Get the Gaussian distribution based on the current state
        mean = self.forward(s, acceleration = acceleration, cf_saturation_FW1 = cf_saturation_FW1, cf_saturation_FW2 = cf_saturation_FW2)
        log_std = self.log_std.expand_as(mean)  # To expand 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Generate the Gaussian distribution based on mean and std
        return dist
    
    def get_act_from_dist(self, s, acceleration = None, cf_saturation_FW1 = None, cf_saturation_FW2 = None):
        dist = self.get_dist(s, acceleration, cf_saturation_FW1, cf_saturation_FW2)
        # Sample the action according to the probability distribution (reparameterization trick).
        # The distribution mean already contains the CBF correction (applied once in forward),
        # so gamma/k1 remain in the graph (learnable) and the executed/stored action matches
        # the sampling distribution -> the PPO importance ratio is consistent.
        a = dist.rsample()
        if self.safety_double_apply and (self.safety_layer_enabled or self.safety_layer_no_grad):
            # Legacy behaviour (kept behind a flag for ablation): clamp the nominal sample,
            # then re-apply the safety filter. NOTE: this makes the stored action differ from
            # the sampling distribution, biasing the PPO ratio.
            a = torch.clamp(a, -self.max_action, self.max_action)
            a_logprob = dist.log_prob(a)
            a_safe = self.safeLayer(a, s, self.tau, self.CAV_index, acceleration, cf_saturation_FW1, cf_saturation_FW2)
            a = a + a_safe
        else:
            # Clamp to the physical actuator limit (not max_action) so the safety correction
            # baked into the mean is not stripped away; execute and store this action.
            a = torch.clamp(a, -self.a_phys, self.a_phys)
            a_logprob = dist.log_prob(a)
        return a, a_logprob


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        # Initialize the parameters of the network
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.is_tanh]  # use tanh activation function

        # Fixed state normalization (same as the actor) to keep tanh layers unsaturated.
        if getattr(args, 'is_state_norm', True):
            s_mean, s_std = equilibrium_state_stats(args.vehicle_num, getattr(args, 's_star', 25), getattr(args, 'v_star', 20))
        else:
            s_mean = np.zeros(args.state_dim, dtype=np.float32)
            s_std = np.ones(args.state_dim, dtype=np.float32)
        self.register_buffer('state_mean', torch.tensor(s_mean))
        self.register_buffer('state_std', torch.tensor(s_std))

        # Use orthogonal initialization
        if args.is_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        # Get the value of the current state (normalized input)
        s = (s - self.state_mean) / self.state_std
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class PPOAgent():
    # 初始化所有 PPO 超参数、Actor、Critic、优化器、ReplayBuffer、RLS 滤波器和 SIDE 模块。
    def __init__(self, args): 
        # Initialize the parameters of the agent
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.adam_eps = args.adam_eps
        self.is_grad_clip = args.is_grad_clip
        self.is_lr_decay = args.is_lr_decay
        self.is_adv_norm = args.is_adv_norm
        self.safety_layer_enabled = args.safety_layer_enabled
        self.device = args.device
        self.safety_layer_no_grad = args.safety_layer_no_grad
        self.replay_buffer = ReplayBuffer_PPO(args)
        self.FV1_idx = int(args.FV1_idx)
        self.FV2_idx = int(args.FV2_idx)
        self.SIDE_update = args.SIDE_update
        self.SIDE_enabled = args.SIDE_enabled
        self.SIDE_load = getattr(args, 'SIDE_load', False)
        self.num_vehicles = args.vehicle_num

        self.FW1_parameters = args.car_following_parameters
        self.FW2_parameters = args.car_following_parameters
        self.s_star = getattr(args, 's_star', 25)
        self.v_star = getattr(args, 'v_star', 20)
        # Latest PPO-update diagnostics (logged to wandb each episode, carried forward
        # between updates since an update only happens once the rollout buffer fills).
        self.train_metrics = {}
        # Initialize the actor and critic networks
        self.actor = Actor(args).to(self.device)
        self.critic = Critic(args).to(self.device)

        if self.adam_eps:  # Set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.car_following_parameters = args.car_following_parameters #[1.2566, 1.5000, 0.9000]

        self.SIDE_FV1 = NN_SI_DE_Module(3, 1, args.lr_cf, args.lr_de, args.batch_size_SIDE, args.buffer_size_SIDE, args.device, args.FV1_idx, num_vehicles=self.num_vehicles, s_star=self.s_star, v_star=self.v_star)
        self.SIDE_FV2 = NN_SI_DE_Module(3, 1, args.lr_cf, args.lr_de, args.batch_size_SIDE, args.buffer_size_SIDE, args.device, args.FV2_idx, num_vehicles=self.num_vehicles, s_star=self.s_star, v_star=self.v_star)
        if self.SIDE_enabled:
            # When SIDE_load is False, SIDE starts fresh and learns the (IDM) human
            # driving behaviour online via SIDE_update; stale OVM-era weights are skipped.
            if self.SIDE_load:
                self.SIDE_FV1.load_model('model_parameters/SIDE_FV1_')
                #self.SIDE_FV1.load_model('SI_pretrain/')
                self.SIDE_FV2.load_model('model_parameters/SIDE_FV2_')
            self.FW1_parameters = self.SIDE_FV1.car_following_model_parameters() # [alpha1, alpha2, alpha3]
            self.FW2_parameters = self.SIDE_FV2.car_following_model_parameters() # [alpha1, alpha2, alpha3]
            self.actor.safeLayer.FW1_parameters = self.FW1_parameters # 把这两套参数传给 actor 里的 safety layer
            self.actor.safeLayer.FW2_parameters = self.FW2_parameters
            self.car_following_parameters = self.FW1_parameters
        self.num_episodes = args.num_episodes
        self.input_blending_weight = np.arange(self.num_episodes) / (self.num_episodes - 1)
        self.episode_cnt = 0

    def evaluate(self, s, acceleration=None):
        # When evaluating the policy, we only use the mean action.
        if acceleration is None:
            acceleration = np.zeros(self.num_vehicles)
        action, _ = self.act(s, evaluate=True, acceleration=acceleration)
        return action

    def act(self, s, add_noise=False, evaluate = False, acceleration = None, cf_saturation = None):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        acceleration = torch.unsqueeze(torch.tensor(acceleration, dtype=torch.float), 0).to(self.device)

        if self.SIDE_enabled:
            cf_saturation_FW1 = self.SIDE_FV1._get_disturbance_estimation(s)
            cf_saturation_FW2 = self.SIDE_FV2._get_disturbance_estimation(s)
        else:
            cf_saturation_FW1 = None
            cf_saturation_FW2 = None
            
        if not evaluate:
            # Get the Gaussian distribution based on the current state
            # dist = self.actor.get_dist(s)
            # Sample the action according to the probability distribution (reparameterization trick)
            # a = dist.rsample() 
            # a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
            # a_logprob = dist.log_prob(a)  # The log probability density of the action

            a, a_logprob = self.actor.get_act_from_dist(s, acceleration, cf_saturation_FW1, cf_saturation_FW2)
        else:
            # If evaluating, we only use the mean
            a = self.actor(s, acceleration, cf_saturation_FW1, cf_saturation_FW2)

            return a.cpu().detach().numpy().flatten(), None
        
        # Return the action and the log probability density of the action
        with torch.no_grad():
           return a.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten()
        # return a, a_logprob

    def step(self, s, a, a_logprob, r, s_, done, total_steps, acceleration):
        if self.SIDE_update:
            self.parameter_estimation(s, s_, acceleration)

        self.replay_buffer.store(s, a, a_logprob, r, s_, done, acceleration)  # Store the transition in the replay buffer
        
        if self.replay_buffer.count == self.batch_size:
            s, a, a_logprob, r, s_, done, acceleration = self.replay_buffer.numpy_to_tensor()  # Get training data
            s, a, a_logprob, r, s_, done, acceleration = s.to(self.device), a.to(self.device), a_logprob.to(self.device), r.to(self.device), s_.to(self.device), done.to(self.device), acceleration.to(self.device)

            if self.SIDE_enabled:
                cf_saturation_FW1 = self.SIDE_FV1._get_disturbance_estimation(s)
                cf_saturation_FW2 = self.SIDE_FV2._get_disturbance_estimation(s)
            else:
                cf_saturation_FW1 = None
                cf_saturation_FW2 = None
            advantages = []
            gae = 0
            with torch.no_grad():  # advantages and v_target have no gradient
                vs = self.critic(s)
                vs_ = self.critic(s_)
                deltas = r.cpu() + self.gamma * vs_.cpu()- vs.cpu()
                for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.cpu().flatten().numpy())):
                    gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                    advantages.insert(0, gae)
                advantages = torch.tensor(advantages, dtype=torch.float).view(-1, 1)
                v_target = advantages + vs.cpu()
                if self.is_adv_norm:  # Advantage normalization
                    advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-5))

            # Health-metric accumulators (averaged over all minibatch updates).
            m = {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0,
                 'approx_kl': 0.0, 'clip_frac': 0.0,
                 'actor_grad_norm': 0.0, 'critic_grad_norm': 0.0, 'n': 0}

            # Optimize policy for K epochs:
            for _ in range(self.K_epochs):
                # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
                for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                    if self.SIDE_enabled:
                        dist_current = self.actor.get_dist(s[index], acceleration[index], cf_saturation_FW1 = cf_saturation_FW1[index], cf_saturation_FW2 = cf_saturation_FW2[index])
                    else:
                        dist_current = self.actor.get_dist(s[index], acceleration[index])
                    dist_entropy = dist_current.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                    a_logprob_current = dist_current.log_prob(a[index])
                    # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                    logratio = a_logprob_current.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True)
                    ratios = torch.exp(logratio)  # shape(mini_batch_size X 1)

                    surr1 = ratios * advantages[index].to(self.device)  # Only calculate the gradient of 'a_logprob_current' in ratios
                    surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages[index].to(self.device)
                    actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Policy entropy
                    # Update actor
                    self.optimizer_actor.zero_grad()
                    actor_loss.mean().backward()
                    if self.is_grad_clip:  # Gradient clip
                        a_gn = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    else:
                        a_gn = torch.tensor(0.0)
                    self.optimizer_actor.step()
                    if self.safety_layer_enabled:
                        # gamma is now a fixed buffer (not learned); only k1 is clamped.
                        self.actor.safeLayer.k1.data = torch.clamp(self.actor.safeLayer.k1.data, 0, 10)

                    v_s = self.critic(s[index])
                    critic_loss = F.mse_loss(v_target[index].to(self.device), v_s)
                    # Update critic
                    self.optimizer_critic.zero_grad()
                    critic_loss.backward()
                    if self.is_grad_clip:  # Gradient clip
                        c_gn = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    else:
                        c_gn = torch.tensor(0.0)
                    self.optimizer_critic.step()

                    # Accumulate diagnostics (no gradient)
                    with torch.no_grad():
                        m['actor_loss'] += actor_loss.mean().item()
                        m['critic_loss'] += critic_loss.item()
                        m['entropy'] += dist_entropy.mean().item()
                        # Schulman's low-variance approx KL estimator
                        m['approx_kl'] += ((ratios - 1) - logratio).mean().item()
                        m['clip_frac'] += (ratios.sub(1).abs() > self.epsilon).float().mean().item()
                        m['actor_grad_norm'] += float(a_gn)
                        m['critic_grad_norm'] += float(c_gn)
                        m['n'] += 1

            self.replay_buffer.count = 0
            if self.is_lr_decay:  # Learning rate Decay
                self.lr_decay(total_steps)

            # Finalize per-update diagnostics for wandb logging.
            n = max(m.pop('n'), 1)
            self.train_metrics = {f'train/{k}': v / n for k, v in m.items()}
            with torch.no_grad():
                self.train_metrics['train/policy_std'] = float(self.actor.log_std.exp().mean())
                self.train_metrics['train/value_mean'] = float(vs.mean())
                self.train_metrics['train/adv_abs_mean'] = float(advantages.abs().mean())
                self.train_metrics['train/lr_a'] = self.optimizer_actor.param_groups[0]['lr']
                if self.safety_layer_enabled or self.safety_layer_no_grad:
                    g = self.actor.safeLayer.gamma.detach().cpu().numpy()
                    self.train_metrics['train/cbf_gamma_mean'] = float(g.mean())
                    self.train_metrics['train/cbf_k1'] = float(self.actor.safeLayer.k1.detach().cpu().mean())

    def parameter_estimation(self, state, next_state, acceleration):
        off = self.num_vehicles
        state_FW1 = state[[self.FV1_idx, self.FV1_idx + off, self.FV1_idx + off - 1]]
        state_FW2 = state[[self.FV2_idx, self.FV2_idx + off, self.FV2_idx + off - 1]]
        state_FW1[0] = state_FW1[0] - self.s_star
        state_FW2[0] = state_FW2[0] - self.s_star
        state_FW1[1] = - (state_FW1[1] - self.v_star)
        state_FW2[1] = - (state_FW2[1] - self.v_star)
        state_FW1[2] = state_FW1[2] - self.v_star
        state_FW2[2] = state_FW2[2] - self.v_star

        if self.SIDE_update:
            next_state_FW1 = next_state[[self.FV1_idx, self.FV1_idx + off, self.FV1_idx + off - 1]]
            next_state_FW2 = next_state[[self.FV2_idx, self.FV2_idx + off, self.FV2_idx + off - 1]]
            next_state_FW1[0] = next_state_FW1[0] - self.s_star
            next_state_FW2[0] = next_state_FW2[0] - self.s_star
            next_state_FW1[1] = - (next_state_FW1[1] - self.v_star)
            next_state_FW2[1] = - (next_state_FW2[1] - self.v_star)
            next_state_FW1[2] = next_state_FW1[2] - self.v_star
            next_state_FW2[2] = next_state_FW2[2] - self.v_star
            
            
            self.SIDE_FV1.step(state_FW1, acceleration[self.FV1_idx], next_state_FW1)
            self.SIDE_FV2.step(state_FW2, acceleration[self.FV2_idx], next_state_FW2)

            self.FW1_parameters = self.SIDE_FV1.car_following_model_parameters()
            self.FW2_parameters = self.SIDE_FV2.car_following_model_parameters()

        self.actor.safeLayer.FW1_parameters = self.FW1_parameters
        self.actor.safeLayer.FW2_parameters = self.FW2_parameters


    def lr_decay(self, total_steps):
        # Learning rate decay
        lr_a_current = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_current = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_current
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_current

    def save(self, checkpoint_path, epsilon_number):
        # Save checkpoint
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        if self.safety_layer_enabled or self.safety_layer_no_grad:
            torch.save(self.actor.state_dict(), os.path.join(checkpoint_path, 'ppo_actor_episode_' + str(epsilon_number) + '.pth'))
            torch.save(self.critic.state_dict(), os.path.join(checkpoint_path, 'ppo_critic_episode_' + str(epsilon_number) + '.pth'))
        else:
            torch.save(self.actor.state_dict(), os.path.join(checkpoint_path, 'ppo_actor_episode_' + str(epsilon_number) + '_no_safety.pth'))
            torch.save(self.critic.state_dict(), os.path.join(checkpoint_path, 'ppo_critic_episode_' + str(epsilon_number) + '_no_safety.pth'))

    def load(self, checkpoint_path, epsilon_number):
        # Load checkpoint
        if self.safety_layer_enabled or self.safety_layer_no_grad:
            self.actor.load_state_dict(torch.load(os.path.join(checkpoint_path, 'ppo_actor_episode_' + str(epsilon_number) + '.pth')))
            self.critic.load_state_dict(torch.load(os.path.join(checkpoint_path, 'ppo_critic_episode_' + str(epsilon_number) + '.pth')))
        else:
            self.actor.load_state_dict(torch.load(os.path.join(checkpoint_path, 'ppo_actor_episode_' + str(epsilon_number) + '_no_safety.pth')))
            self.critic.load_state_dict(torch.load(os.path.join(checkpoint_path, 'ppo_critic_episode_' + str(epsilon_number) + '_no_safety.pth')))
