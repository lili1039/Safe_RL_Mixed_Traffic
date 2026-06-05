import gymnasium as gym
import numpy as np
from gymnasium import spaces
from idm import IDMModel

class PlatoonEnv(gym.Env):
    def __init__(self, num_vehicles=4, init_params=None, dt=0.12, max_steps=400, select_scenario=0, pure_car_following=False, set_disturbance = False,
                 reward_clip=0.0, vehicle_length=4.75, domain_randomize=False, dr_range=0.15, brake_accel=5.0, accel_mag=1.0):
        super(PlatoonEnv, self).__init__()

        # Set the default parameters
        self.num_vehicles = num_vehicles
        self.dt = dt
        self.max_steps = max_steps
        self.steps = 0
        # Per-step reward floor: reward = max(reward, -reward_clip) when reward_clip > 0.
        # Bounds the magnitude of catastrophic (e.g. hard-braking) episodes so reward
        # scaling / gradients are not dominated by them. 0 disables clipping.
        self.reward_clip = reward_clip
        # 0: random head velocity, 1: head emergency braking,
        # 2: vehicle 2 acceleration, 3: vehicle 3 acceleration
        self.select_scenario = select_scenario
        self.pure_car_following = pure_car_following

        # Set up action and observation spaces
        # Observation includes ALL vehicles (head included): spacing(N) + velocity(N)
        # + spacing_diff(N-1) + velocity_diff(N-1) = 4*N - 2
        # Nominal RL action range [-1, 1] (original design). The executed action (after the
        # safety correction) is limited to the physical +-5 m/s^2 by the IDM internal clip.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4 * num_vehicles - 2,), dtype=np.float32)

        # Initialize the IDM-based platoon simulator. Platoon layout: [head(0), CAV(1), HDV(2), HDV(3)]
        self.cav_index = [1]
        self.sim = IDMModel(num_vehicles=num_vehicles, params=init_params, cav_index=self.cav_index,
                            vehicle_length=vehicle_length, domain_randomize=domain_randomize, dr_range=dr_range,
                            brake_accel=brake_accel, accel_mag=accel_mag)
        self.hx_ls = []

        self.set_disturbance = set_disturbance

    def reset(self, disturbance = None):
        self.steps = 0
        if self.set_disturbance:
            self.sim.reset(disturbance)
        else:
            self.sim.reset()

        return self._get_obs(), self.get_acceleration()

    def step(self, action):
        # Apply action to the CAV (vehicle 1)
        self.sim.set_control_input(1, action[0])

        # Update the simulator for all vehicles
        self.sim.update(self.dt, self.select_scenario, self.pure_car_following)

        obs = self._get_obs()
        reward = self._get_reward()
        acceleration = self.get_acceleration()
        done = self.steps >= self.max_steps
        self.steps += 1

        return obs, reward, acceleration, done, {}

    def _get_obs(self):
        # The observation is the spacing, velocity, differential spacing, and differential velocity.
        # The head vehicle is included so that observation index == physical vehicle index.
        spacing_diff = self.sim.spacing[:-1] - self.sim.spacing[1:]
        velocity_diff = self.sim.velocity[:-1] - self.sim.velocity[1:]
        obs = np.concatenate((self.sim.spacing, self.sim.velocity, spacing_diff, velocity_diff))
        return obs

    def get_velocity(self):
        # Return the velocity of all vehicles
        return self.sim.velocity.copy()

    def get_spacing(self):
        # Return the spacing of all vehicles
        return self.sim.spacing.copy()

    def get_position(self):
        # Return the position of all vehicles
        return self.sim.position.copy()

    def get_acceleration(self):
        # Return the acceleration of all vehicles
        return self.sim.acceleration.copy()

    def _get_reward(self):
        # Platoon layout: [head(0), CAV(1), FV1/HDV(2), FV2/HDV(3)]
        # Energy consumption (not considered in this work)
        energy_consumption = 0

        eps = 1e-6
        # Safety (CAV w.r.t. head)
        ttc = - self.sim.spacing[1] / (self.sim.velocity[0] - self.sim.velocity[1] + eps)
        if ttc >= 0 and ttc <= 4:
            safety = np.log(ttc / 4)
        else:
            safety = 0

        ttc_FW1 = - self.sim.spacing[2] / (self.sim.velocity[1] - self.sim.velocity[2] + eps)
        if ttc_FW1 >= 0 and ttc_FW1 <= 4:
            safety_FW1 = 0.5*np.log(ttc_FW1 / 4)
        else:
            safety_FW1 = 0

        ttc_FW2 = - self.sim.spacing[3] / (self.sim.velocity[2] - self.sim.velocity[3] + eps)
        if ttc_FW2 >= 0 and ttc_FW2 <= 4:
            safety_FW2 = 0.1*np.log(ttc_FW2 / 4)  # weight 0.1 (matches upstream)
        else:
            safety_FW2 = 0

        # Traffic Efficiency: penalize an over-large time gap. Threshold relaxed to 3.5
        # (was 2.5) so the CAV is not pushed to aggressively close moderate gaps.
        TG = self.sim.spacing[1] / (self.sim.velocity[1] + eps)
        if TG >= 3.5:
            R_eff = -1
        else:
            R_eff = 0

        # Stability
        v_star = self.sim.v0
        s_star = self.sim.s0
        stability = -np.sum(np.square((self.sim.velocity[1:3] - self.sim.velocity[0])))

        # Return the sum of the rewards (with optional floor clip to bound catastrophic episodes)
        reward = energy_consumption + safety + stability + R_eff + safety_FW1 + safety_FW2
        if self.reward_clip and self.reward_clip > 0:
            reward = max(reward, -self.reward_clip)
        return reward
