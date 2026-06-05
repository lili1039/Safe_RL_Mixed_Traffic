import argparse
import numpy as np
import random
from collections import deque, namedtuple
import torch


def str2bool(v):
    """argparse type that actually parses booleans.

    The plain `type=bool` argument is a trap: bool("False") is True, so
    `--flag False` silently enables the flag. Use this instead, e.g.
    `parser.add_argument("--flag", type=str2bool, nargs='?', const=True, default=...)`.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "f", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


class RunningMeanStd:
    """Online (Welford) mean/std estimator."""

    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


def equilibrium_state_stats(num_vehicles, s_star, v_star,
                            spacing_scale=10.0, velocity_scale=5.0,
                            spacing_diff_scale=5.0, velocity_diff_scale=5.0):
    """Fixed mean/std vectors for normalizing the platoon observation.

    The observation (head-inclusive layout) is
    ``[spacing(N), velocity(N), spacing_diff(N-1), velocity_diff(N-1)]``.
    Raw spacing (~25-50) and velocity (~20) saturate the actor/critic ``tanh``
    first layer, killing gradients. We centre spacing at ``s_star`` and velocity
    at ``v_star`` (differences at 0) and divide by a fixed scale so the network
    sees ~unit-magnitude inputs.

    A *fixed* affine transform is used (not running stats) because: (1) the
    operating point is known, so centring is principled; (2) it is stateless, so
    collection / PPO updates / evaluation all see the identical mapping; and
    (3) the safety layer still needs the RAW state, so normalization must be
    applied only to the MLP path inside the network, where a fixed buffer is the
    cleanest fit.

    Returns ``(mean, std)`` float32 arrays of length ``4*N - 2``.
    """
    n = num_vehicles
    mean = np.concatenate([
        np.full(n, s_star), np.full(n, v_star),
        np.zeros(n - 1), np.zeros(n - 1),
    ]).astype(np.float32)
    std = np.concatenate([
        np.full(n, spacing_scale), np.full(n, velocity_scale),
        np.full(n - 1, spacing_diff_scale), np.full(n - 1, velocity_diff_scale),
    ]).astype(np.float32)
    return mean, std


class RewardScaling:
    """PPO reward scaling: divide reward by the running std of the discounted
    return (no mean subtraction). Keeps value targets / advantages at ~unit scale
    so occasional huge-magnitude episodes don't destabilise the critic.

    Call once per environment step; call reset() at the end of each episode.
    """

    def __init__(self, shape, gamma):
        self.shape = shape
        self.gamma = gamma
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        return x / (self.running_ms.std + 1e-8)  # only divide by std

    def reset(self):
        self.R = np.zeros(self.shape)


class ReplayBuffer_PPO:
    def __init__(self, args):
        # Initialize a ReplayBuffer_PPO object.
        self.state = np.zeros((args.batch_size, args.state_dim))
        self.action = np.zeros((args.batch_size, args.action_dim))
        self.action_logprob = np.zeros((args.batch_size, args.action_dim))
        self.reward = np.zeros((args.batch_size, 1))
        self.state_ = np.zeros((args.batch_size, args.state_dim))
        self.done = np.zeros((args.batch_size, 1))
        self.acceleration = np.zeros((args.batch_size, args.vehicle_num))
        self.count = 0

    def store(self, state, action, action_logprob, reward, state_, done, acceleration):
        # Store the transition in the replay buffer
        self.state[self.count] = state
        self.action[self.count] = action
        self.action_logprob[self.count] = action_logprob
        self.reward[self.count] = reward
        self.state_[self.count] = state_
        self.done[self.count] = done
        self.acceleration[self.count] = acceleration
        self.count += 1

    def numpy_to_tensor(self):
        # Convert numpy array to torch tensor and return
        state = torch.tensor(self.state, dtype=torch.float)
        action = torch.tensor(self.action, dtype=torch.float)
        action_logprob = torch.tensor(self.action_logprob, dtype=torch.float)
        reward = torch.tensor(self.reward, dtype=torch.float)
        state_ = torch.tensor(self.state_, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)
        acceleration = torch.tensor(self.acceleration, dtype=torch.float)
        return state, action, action_logprob, reward, state_, done, acceleration


class ReplayBuffer_SIDE:
    def __init__(self, buffer_size=int(1e5), batch_size=64):
        # Initialize a ReplayBuffer object (for SIDE).
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "next_state"])
        self.count = 0

    def store(self, state, action, next_state):
        # Add a new experience to memory.
        e = self.experience(state, action, next_state)
        self.memory.append(e)
        self.count += 1

    def sample(self, batch_size=None):
        # Randomly sample a batch of experiences from memory.
        if batch_size is None:
            batch_size = self.batch_size
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        return (states, actions, next_states)
