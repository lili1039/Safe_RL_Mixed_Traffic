import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ReplayBuffer_SIDE


class OVM_Estimator(nn.Module):
    # Linear car-following estimator. Despite the historical "OVM" name, it is used
    # as a linear approximation of the (IDM) human-driver acceleration around the
    # equilibrium point, with state x = [s - s*, -(v - v*), v_prev - v*].
    def __init__(self):
        super(OVM_Estimator, self).__init__()
        self.alpha1 = nn.Parameter(torch.tensor(1., requires_grad=True))
        self.alpha2 = nn.Parameter(torch.tensor(1., requires_grad=True))
        self.alpha3 = nn.Parameter(torch.tensor(1., requires_grad=True))

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        a = self.alpha1*x[:, 0] + self.alpha2*x[:, 1] + self.alpha3*x[:, 2]
        return a.unsqueeze(1)


class Disturbance_Estimator(nn.Module):
    def __init__(self, state_num, action_num):
        super(Disturbance_Estimator, self).__init__()
        self.fc1 = nn.Linear(state_num + action_num, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class NN_SI_DE_Module():
    def __init__(self, state_num, action_num, lr_cf, lr_de, batch_size, buffer_size, device, veh_idx, num_vehicles=4, s_star=25, v_star=20):
        self.state_num = state_num
        self.action_num = action_num
        self.lr_cf = lr_cf
        self.lr_de = lr_de
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device
        self.veh_idx = int(veh_idx)
        # Velocity-block offset in the observation (head-inclusive layout): velocity of
        # vehicle k is at index k + num_vehicles.
        self.vel_offset = num_vehicles

        self.car_following_estimator = OVM_Estimator().to(self.device)
        self.disturbance_estimator = Disturbance_Estimator(self.state_num, self.action_num).to(self.device)

        self.optimizer_cf = torch.optim.Adam(self.car_following_estimator.parameters(), lr=self.lr_cf)
        self.optimizer_de = torch.optim.Adam(self.disturbance_estimator.parameters(), lr=self.lr_de)

        self.replay_buffer = ReplayBuffer_SIDE(self.buffer_size, self.batch_size)

        self.loss_de_lst = []
        self.loss_cf_lst = []
        self.dt = 0.12

        self.s_star = s_star
        self.v_star = v_star

    def step(self, state, action, next_state):
        self.replay_buffer.store(state, action, next_state)
        if self.replay_buffer.count > self.batch_size:
            self.learn()

    def learn(self):
        state, action, next_state = self.replay_buffer.sample()
        # sample() already returns float tensors; convert dtype/device without re-constructing
        state = state.to(device=self.device, dtype=torch.float)
        action = action.to(device=self.device, dtype=torch.float)
        next_state = next_state.to(device=self.device, dtype=torch.float)
        self.optimizer_cf.zero_grad()
        self.optimizer_de.zero_grad()

        action_pred = self.car_following_estimator(state)
        # Keep all tensors on self.device (no .cpu() here, which broke CUDA training)
        action_disturbance = self.disturbance_estimator(torch.cat((state, action_pred.detach()), 1))

        next_state_pred_wo_de = self._get_next_state(state, action_pred)
        next_state_pred_w_de = self._get_next_state(state, action_pred.detach() + action_disturbance)

        loss_cf = F.mse_loss(next_state_pred_wo_de, -next_state[:, 1])
        loss_cf.backward()
        self.optimizer_cf.step()

        loss_de = F.mse_loss(next_state_pred_w_de, -next_state[:, 1])
        loss_de.backward()
        self.optimizer_de.step()

        self.loss_cf_lst.append(loss_cf.item())
        self.loss_de_lst.append(loss_de.item())

    def _get_disturbance_estimation(self, state):
        off = self.vel_offset
        state_FW = state[:, [self.veh_idx, self.veh_idx + off, self.veh_idx + off - 1]]

        state_FW[:, 0] = state_FW[:, 0] - self.s_star
        state_FW[:, 1] = - (state_FW[:, 1] - self.v_star)
        state_FW[:, 2] = state_FW[:, 2] - self.v_star

        # state_FW is already a float tensor slice; convert dtype/device without re-constructing
        state = state_FW.to(device=self.device, dtype=torch.float)
        action_pred = self.car_following_estimator(state).detach()
        disturbance = self.disturbance_estimator(torch.cat((state, action_pred), 1))
        return disturbance.detach().cpu().squeeze()

    def _get_next_state(self, state, action):
        next_state = -state[:, 1] + self.dt * (action.squeeze(1))
        return next_state

    def car_following_model_parameters(self):
        return [self.car_following_estimator.alpha1.cpu().detach().numpy().tolist(),
                self.car_following_estimator.alpha2.cpu().detach().numpy().tolist(),
                self.car_following_estimator.alpha3.cpu().detach().numpy().tolist()]

    def save_model(self, path):
        torch.save(self.car_following_estimator.state_dict(), path + 'car_following_estimator.pth')
        torch.save(self.disturbance_estimator.state_dict(), path + 'disturbance_estimator.pth')

    def load_model(self, path):
        try:
            self.car_following_estimator.load_state_dict(torch.load(path + 'car_following_estimator.pth'))
            self.disturbance_estimator.load_state_dict(torch.load(path + 'disturbance_estimator.pth'))
        except Exception:
            print('error loading')
