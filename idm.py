import numpy as np

# Default IDM car-following parameters: [v0, Tgap, a, b, delta, s0]
#   v0    desired (free-flow) velocity        [m/s]
#   Tgap  safe time headway                   [s]
#   a     maximum acceleration                [m/s^2]
#   b     comfortable deceleration            [m/s^2]
#   delta acceleration exponent (standard IDM = 4)
#   s0    minimum NET gap at standstill       [m]
DEFAULT_IDM_PARAMS = [
    40.0,    # v0
    1.4,     # Tgap
    1.13,    # a
    4.0,     # b
    4.0,     # delta
    8.16,    # s0 (net jam gap)
]

# Physical vehicle length [m]. The IDM spacing `s` is the NET (bumper-to-bumper)
# gap = centre-to-centre distance - vehicle length.
VEHICLE_LENGTH = 4.75


def idm_acceleration(v, dv, s, params):
    """
    IDM acceleration. `s` is the NET (bumper-to-bumper) gap.
    params = [v0, Tgap, a, b, delta, s0]
    dv is the approach rate (own velocity - preceding velocity).
    """
    v0, Tgap, a, b, delta, s0 = params

    v = max(float(v), 0.0)
    s = max(float(s), 1e-2)

    s_star = s0 + v * Tgap + (v * dv) / (2 * np.sqrt(a * b))
    acc = a * (1 - (v / v0) ** delta - (s_star / s) ** 2)
    return float(np.clip(acc, -5.0, 5.0))


class IDMModel:
    def __init__(self, num_vehicles=4, params=None, cav_index=None, idm_params=None,
                 vehicle_length=VEHICLE_LENGTH, domain_randomize=False, dr_range=0.15,
                 brake_accel=5.0, accel_mag=1.0):
        # Set the default parameters used by the simulator (equilibrium / head scenarios)
        if params is None:
            params = {
                'v0': 20,             # Equilibrium / reset (cruise) velocity
                's0': 42,             # Equilibrium / reset spacing (centre-to-centre, ~IDM steady state at v=20)
                'velocity_noise': 1.0,  # Head acceleration noise std (scenario 0 disturbance)
                'a_max': 5,
                'a_min': -5
            }
        self.num_vehicles = num_vehicles
        self.v0 = params['v0']
        self.s0 = params['s0']
        self.velocity_noise = params['velocity_noise']
        self.a_max = params['a_max']
        self.a_min = params['a_min']

        # HDV (IDM) car-following parameters + heterogeneity / robustness knobs
        self.nominal_idm_params = list(idm_params) if idm_params is not None else list(DEFAULT_IDM_PARAMS)
        self.vehicle_length = vehicle_length
        self.domain_randomize = domain_randomize
        self.dr_range = dr_range
        # Per-vehicle IDM parameters (resampled each reset when domain_randomize=True)
        self.veh_idm_params = [list(self.nominal_idm_params) for _ in range(num_vehicles)]

        # Scenario disturbance magnitudes (configurable for experiments)
        self.brake_accel = brake_accel   # head emergency-braking magnitude (scenario 1)
        self.accel_mag = accel_mag       # follower forced-acceleration magnitude (scenario 2/3)

        self.spacing = np.zeros(num_vehicles)
        self.velocity = np.zeros(num_vehicles)
        self.position = np.zeros(num_vehicles)
        self.control_input = np.zeros(num_vehicles)
        self.cav_index = cav_index
        self.acceleration = np.zeros(num_vehicles)
        self.t = 0

        self.disturbance = None

    def _sample_idm_params(self):
        """Sample one vehicle's IDM params, optionally domain-randomized within +/-dr_range.
        v0, Tgap, a, b, s0 are randomized; delta and vehicle length are kept fixed."""
        p = np.array(self.nominal_idm_params, dtype=float)
        if self.domain_randomize:
            factors = np.ones(6)
            for idx in (0, 1, 2, 3, 5):  # v0, Tgap, a, b, s0 (skip delta=idx 4)
                factors[idx] = np.random.uniform(1.0 - self.dr_range, 1.0 + self.dr_range)
            p = p * factors
        return p.tolist()

    def reset(self, disturbance=None):
        # The spacing between the vehicles (centre-to-centre)
        self.spacing.fill(self.s0)
        # The velocity of the vehicles
        self.velocity.fill(self.v0)
        self.t = 0
        # The position of the vehicles
        for i in range(self.num_vehicles):
            self.position[i] = (self.num_vehicles - i - 1) * self.s0
        # The control input of the vehicles
        self.control_input.fill(0)
        # The acceleration of the vehicles
        self.acceleration.fill(0)
        # (Re)sample per-vehicle IDM parameters for this episode (heterogeneous HDVs)
        self.veh_idm_params = [self._sample_idm_params() for _ in range(self.num_vehicles)]

        self.disturbance = disturbance
        # disturbance[0] = 扰动结束时间（从0开始到disturbance[0]停止） disturbance[1] = 扰动加速度

    def set_control_input(self, vehicle_idx, control_input):
        # Set the control input of the vehicle
        self.control_input[vehicle_idx] = control_input

    def update(self, dt, select_scenario, pure_car_following):
        # Update the time
        # scenario 0: 头车 vehicle 0 随机速度扰动。
        # scenario 1: 头车 vehicle 0 紧急刹车。前半段加速度 -brake_accel，后半段 +brake_accel。
        # scenario 2: HDV vehicle 2 加速。前半段加速度 +accel_mag，后半段 0。
        # scenario 3: HDV vehicle 3 加速。前半段加速度 +accel_mag，后半段 0。

        self.t += dt
        scenario_id = []
        if select_scenario == 0 or select_scenario == 1:
            duration = [0, 5]
        elif select_scenario == 2:
            scenario_id = [2]
            duration = [0, 8]
        elif select_scenario == 3:
            scenario_id = [3]
            duration = [0, 8]
        else:
            raise ValueError(f"Unsupported scenario: {select_scenario}")

        if self.disturbance is not None:
            duration = [0, self.disturbance[0]]

        # Update all the vehicles
        for i in range(self.num_vehicles - 1, 0, -1):
            # Update the velocity of CAV
            if i in self.cav_index and not pure_car_following:
                # The autonomous vehicle uses the provided control input
                dv = self.velocity[i-1] - self.velocity[i]
                if self.control_input[i] > self.a_max:
                    self.control_input[i] = self.a_max
                elif self.control_input[i] < self.a_min:
                    self.control_input[i] = self.a_min

                self.velocity[i] += self.control_input[i] * dt
                # Update the spacing
                self.spacing[i] += dv * dt
                self.position[i] += self.velocity[i] * dt

                self.acceleration[i] = self.control_input[i]

            elif i in scenario_id and self.t >= duration[0] and self.t < duration[1]:  # and not pure_car_following
                # Forced-acceleration disturbance on a following vehicle
                dv = self.velocity[i-1] - self.velocity[i]
                if self.disturbance is not None:
                    emergent_acc = self.disturbance[1]
                else:
                    emergent_acc = self.accel_mag
                if self.t >= duration[0] and self.t < duration[1]/2:
                    self.velocity[i] += emergent_acc * dt
                    self.acceleration[i] = emergent_acc
                if self.t >= duration[1]/2 and self.t < duration[1]:
                    self.velocity[i] += 0
                    self.acceleration[i] = 0
                self.position[i] += self.velocity[i] * dt
                self.spacing[i] += dv * dt

            else:
                # The human-driven followers use the IDM model (on the NET gap)
                gap_rate = self.velocity[i-1] - self.velocity[i]   # closing rate for spacing update
                dv_idm = self.velocity[i] - self.velocity[i-1]     # IDM approach rate (own - preceding)

                net_gap = self.spacing[i] - self.vehicle_length
                acceleration = idm_acceleration(self.velocity[i], dv_idm, net_gap, self.veh_idm_params[i])
                if acceleration > self.a_max:
                    acceleration = self.a_max
                elif acceleration < self.a_min:
                    acceleration = self.a_min

                self.acceleration[i] = acceleration

                self.velocity[i] += acceleration * dt

                # Update the spacing
                self.spacing[i] += gap_rate * dt
                self.position[i] += self.velocity[i] * dt

        if select_scenario == 0:
            # Random noise scenario
            head_acceleration = self.control_input[0] + np.random.normal(0, self.velocity_noise)
            self.velocity[0] += head_acceleration * dt
            self.spacing[0] += 0
            self.position[0] += self.velocity[0] * dt
            self.acceleration[0] = head_acceleration
        elif select_scenario == 1:
            # Emergency braking scenario. When a disturbance is injected (e.g. by the
            # safe-region sweep), disturbance[1] sets the braking magnitude so the
            # acceleration axis is meaningful; otherwise fall back to brake_accel.
            braking_acc = -abs(self.brake_accel)
            if self.disturbance is not None:
                braking_acc = -abs(self.disturbance[1])
            self.acceleration[0] = 0
            if self.t >= duration[0] and self.t < duration[1]/2:
                self.velocity[0] += braking_acc * dt
                self.acceleration[0] = braking_acc
            elif self.t >= duration[1]/2 and self.t < duration[1]:
                self.velocity[0] += -braking_acc * dt
                self.acceleration[0] = -braking_acc

            self.position[0] += self.velocity[0] * dt
        else:
            # In scenarios 2 and 3, the head vehicle is not disturbed.
            self.acceleration[0] = 0
            self.position[0] += self.velocity[0] * dt
