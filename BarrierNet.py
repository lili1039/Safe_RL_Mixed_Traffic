import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from utils import *
from qpth.qp import QPFunction, QPSolvers

eps = 1e-7

class BarrierLayer(nn.Module):
    def __init__(self, states_size, car_following_parameters = [1.2566, 1.5000, 0.9000], safety_layer_no_grad = False, SIDE_enabled = False, num_vehicles = 4, s_star = 25, v_star = 20,
                 cav_alpha = 1.0, follower_alpha = 0.5, min_gap = 5.0):
        super(BarrierLayer, self).__init__()
        # Equilibrium operating point (must match env reset spacing, reward & SI normalization)
        self.s_star = s_star
        self.v_star = v_star
        # Empty equality-constraint placeholder. Double dtype to match the QP solve below,
        # which runs in float64 for numerical accuracy (the QP is always feasible, so the
        # float32 solver's "inaccurate/residual large" warnings were purely numerical).
        self.e = Variable(torch.DoubleTensor())
        self.states_size = states_size
        self.following_veh = 2
        # Velocity-block offset in the observation. The observation is laid out as
        # [spacing(N), velocity(N), spacing_diff(N-1), velocity_diff(N-1)] with the head
        # included, so observation index == physical vehicle index and the velocity of
        # vehicle k lives at index k + num_vehicles.
        self.vel_offset = num_vehicles
        # CBF class-K gains gamma = [CAV-front, follower-1, follower-2]. These are now
        # FIXED (not learned): the CAV's own front-safety barrier stays at full strength
        # (cav_alpha=1.0) so collision avoidance is uncompromised, while the follower
        # barriers use a smaller gain (follower_alpha<1.0) so the gap-consistency
        # restoration is gentle and does not over-react (no large accel spike when a CAV
        # starts much farther back than its followers, e.g. at an IDM->RL handover).
        gamma_init = torch.tensor([float(cav_alpha), float(follower_alpha), float(follower_alpha)])
        if safety_layer_no_grad:
            self.gamma = gamma_init
            self.k1 = torch.tensor([1.0])
        else:
            # gamma is a non-trainable buffer; only k1 (feasibility-constraint gain) is learned.
            self.register_buffer('gamma', gamma_init)
            self.k1 = nn.Parameter(torch.tensor([10.0]), requires_grad=True)
        # Minimum spacing margin enforced by the CAV-front barrier (accounts for vehicle
        # length): the safe set is s_i - tau*v_i - min_gap >= 0, i.e. spacing >= min_gap
        # (+ the time-headway term) rather than >= 0.
        self.min_gap = float(min_gap)
        # self.car_following_parameters = car_following_parameters

        self.FW1_parameters = car_following_parameters
        self.FW2_parameters = car_following_parameters

        self.SIDE_enabled = SIDE_enabled

    def forward(self, u_nominal, states, tau, CAV_index, acceleration = None, cf_saturation_FW1 = None, cf_saturation_FW2 = None):
        
        # Safety ahead constraint

        v_star   = self.v_star
        s_star   = self.s_star
        min_gap  = self.min_gap
        if_batch = states.dim() > 1 # check if batch

        if if_batch:
            
            batch_size = states.size(dim=0)
            # define the parameters of the CBF constraint

            # bacth Q matrix
            self.Q = torch.zeros(batch_size, self.following_veh + 1 , self.following_veh + 1)
            Q_weight = [1, 1, 1]
            for i in range(0, self.following_veh + 1):
                self.Q [:,i,i] = Q_weight[i]

            # batch p vector
            self.p = torch.zeros(batch_size, self.following_veh + 1)
            #self.p [:,0] = -u_nominal.squeeze(1)

            # batch states
            off = self.vel_offset
            s_i, v_i = states[:, CAV_index], states[:, CAV_index+off]                               # CAV's spacing and velocity
            s_im, v_im = states[:, CAV_index-1], states[:, CAV_index+off-1]           # CAV front vehicle's spacing and velocity
            s_f_1, v_f_1 = states[:, CAV_index + 1], states[:, CAV_index+off+1]                               # following vehicle 1's spacing and velocity
            s_f_im1, v_f_im1 = states[:, CAV_index], states[:, CAV_index+off]           # front vehicle's spacing and velocity
            s_f_2, v_f_2 = states[:, CAV_index + 2], states[:, CAV_index+off+2]                               # following vehicle 2's spacing and velocity
            s_f_im2, v_f_im2 = states[:, CAV_index + 1], states[:, CAV_index+off+1]           # front vehicle's spacing and velocity
            
            s_i_ls = [s_i, s_f_1, s_f_2]
            v_i_ls = [v_i, v_f_1, v_f_2]
            v_im_ls = [v_im, v_f_im1, v_f_im2]

            if self.SIDE_enabled == False:
                cf_saturation_FW1 = torch.zeros(batch_size)
                cf_saturation_FW2 = torch.zeros(batch_size)

            Lfh1 = (v_im_ls[0] - v_i_ls[0]).unsqueeze(1) #+ La_CAV.detach()
            Lfh2 =(- v_im_ls[0] +  v_i_ls[0] + v_im_ls[1] - v_i_ls[1] - tau*(self.FW1_parameters[0]*(s_i_ls[1]-s_star) - self.FW1_parameters[1]*(v_i_ls[1]-v_star) + self.FW1_parameters[2]*(v_im_ls[1]-v_star) + cf_saturation_FW1)).unsqueeze(1)
            Lfh3 =(- v_im_ls[0] +  v_i_ls[0] + v_im_ls[2] - v_i_ls[2] - tau*(self.FW2_parameters[0]*(s_i_ls[2]-s_star) - self.FW2_parameters[1]*(v_i_ls[2]-v_star) + self.FW2_parameters[2]*(v_im_ls[2]-v_star) + cf_saturation_FW2)).unsqueeze(1)
            Lfh_ls = torch.hstack([Lfh1, Lfh2, Lfh3])
            Lgh_ls = torch.hstack([-tau * torch.ones(batch_size,1), tau * torch.ones(batch_size,1), tau * torch.ones(batch_size,1)]) #Lb_CAV.detach()
            #Lfh = (v_im - v_i).unsqueeze(1) #+ La.detach()
            #Lgh = -tau #+ Lb.detach()

            alpha_h_1 = (self.gamma[0]*(s_i - tau * v_i - min_gap).pow(1)).unsqueeze(1)
            alpha_h_2 = (self.gamma[1]*(s_f_1 - s_i - tau * (v_f_1 - v_i)).pow(1)).unsqueeze(1)
            alpha_h_3 = (self.gamma[2]*(s_f_2 - s_i - tau * (v_f_2 - v_i)).pow(1)).unsqueeze(1)
            alpha_h_ls = torch.hstack([alpha_h_1, alpha_h_2, alpha_h_3])
            nominal_part = u_nominal.mul(Lgh_ls)

            # Feasible acceleration constraint
            if acceleration is not None:
                u_min = -5
                control_bound = acceleration[:, CAV_index - 1] + self.k1*(v_im_ls[0] - v_i_ls[0] - tau*u_min) - u_nominal.squeeze()
                # control_bound = 10000*torch.ones(batch_size)

            # batch G matrix
            self.G = torch.zeros(batch_size, self.following_veh+2, self.following_veh+1) # batch size, number of constraints, number of variables
            #for i in range(0, self.following_veh+1):

            self.G [:,0:self.following_veh+1,0] = - Lgh_ls
            self.G [:,1,1] = - 1
            self.G [:,2,2] = - 1
            # feasibility
            self.G [:,self.following_veh+1,0] = 1
            
            # batch h vector
            self.h = torch.zeros(batch_size, self.following_veh+2)
            self.cbf_h = (alpha_h_ls+Lfh_ls+nominal_part)
            self.h[:,0:self.following_veh+1] = self.cbf_h
            self.h[:,self.following_veh+1] = control_bound
            
            # print(self.Q, self.p, self.G, self.h)
            # calculate the CBF constraint with QP solvers in batch (solve in float64,
            # cast the solution back to float32 for the rest of the network).
            # eps=1e-8 is an achievable, sensible tolerance: it is far below the float32
            # output resolution (~1e-7) so the solution is numerically identical, while
            # avoiding qpth's unreachable default eps=1e-12 that produced spurious
            # "inaccurate solution / residual large" warnings on ill-conditioned instances.
            u_ = QPFunction(eps=1e-8, maxIter=50, verbose=-1)(self.Q.double(), self.p.double(),
                            self.G.double(), self.h.double(), self.e, self.e).float()

            # return the first column of the solution
            return u_[:,0].unsqueeze(1)
        
        else:
            # define the parameters of the CBF constraint

            # Q matrix
            self.Q = torch.zeros(self.following_veh + 2 , self.following_veh + 2)
            Q_weight = [1, 1, 1]
            for i in range(0, self.following_veh + 1):
                self.Q [i,i] = Q_weight[i]
            
            # p vector
            self.p = torch.zeros(self.following_veh + 1)
            # self.p [0] = -u_nominal[0]

            # states
            off = self.vel_offset
            s_i, v_i = states[CAV_index], states[CAV_index+off]                               # CAV's spacing and velocity
            s_im, v_im = states[CAV_index-1], states[CAV_index+off-1]           # CAV front vehicle's spacing and velocity
            s_f_1, v_f_1 = states[CAV_index + 1], states[CAV_index+off+1]                               # following vehicle 1's spacing and velocity
            s_f_im1, v_f_im1 = states[CAV_index], states[CAV_index+off]           # front vehicle's spacing and velocity
            s_f_2, v_f_2 = states[CAV_index + 2], states[CAV_index+off+2]                               # following vehicle 2's spacing and velocity
            s_f_im2, v_f_im2 = states[CAV_index + 1], states[CAV_index+off+1]           # front vehicle's spacing and velocity

            s_i_ls = [s_i, s_f_1, s_f_2]
            v_i_ls = [v_i, v_f_1, v_f_2]
            v_im_ls = [v_im, v_f_im1, v_f_im2]

            if self.SIDE_enabled == False:
                cf_saturation_FW1 = torch.zeros(0)
                cf_saturation_FW2 = torch.zeros(0)

            # lie derivatives
            Lfh1 = (v_im_ls[0] - v_i_ls[0])
            Lfh2 =(- v_im_ls[0] +  v_i_ls[0] + v_im_ls[1] - v_i_ls[1] - tau*(self.FW1_parameters[0]*(s_i_ls[1]-s_star) - self.FW1_parameters[1]*(v_i_ls[1]-v_star) + self.FW1_parameters[2]*(v_im_ls[1]-v_star) + cf_saturation_FW1))
            Lfh3 =(- v_im_ls[0] +  v_i_ls[0] + v_im_ls[2] - v_i_ls[2] - tau*(self.FW2_parameters[0]*(s_i_ls[2]-s_star) - self.FW2_parameters[1]*(v_i_ls[2]-v_star) + self.FW2_parameters[2]*(v_im_ls[2]-v_star) + cf_saturation_FW2))
            Lfh_ls = torch.hstack([Lfh1, Lfh2, Lfh3])
            Lgh_ls = torch.tensor([-tau, tau, tau])

            # Feasible acceleration constraint
            if acceleration is not None:
                u_min = -5
                control_bound = acceleration[CAV_index - 1] + self.k1*(v_im_ls[0] - v_i_ls[0] - tau*u_min) - u_nominal
                # control_bound = 10000*torch.ones(1)

            alpha_h_1 = (self.gamma[0]*(s_i - tau * v_i - min_gap).pow(1))
            alpha_h_2 = (self.gamma[1]*(s_f_1 - s_i - tau * (v_f_1 - v_i)).pow(1))
            alpha_h_3 = (self.gamma[2]*(s_f_2 - s_i - tau * (v_f_2 - v_i)).pow(1))
            alpha_h_ls = torch.hstack([alpha_h_1, alpha_h_2, alpha_h_3])
            nominal_part = u_nominal.mul(Lgh_ls)
            # batch G matrix
            self.G = torch.zeros(self.following_veh+2, self.following_veh+1) # number of constraints, number of variables
            self.G [0:self.following_veh+1,0] = - Lgh_ls
            self.G [1,1] = - 1
            self.G [2,2] = - 1
            # feasibility
            self.G [self.following_veh+1,0] = 1
            
            # h vector
            self.h = torch.zeros(self.following_veh+2)
            self.cbf_h = (alpha_h_ls+Lfh_ls+nominal_part)
            self.h[0:self.following_veh+1] = self.cbf_h
            self.h[self.following_veh+1] = control_bound

            # calculate the CBF constraint with QP solvers (solve in float64,
            # cast the solution back to float32 for the rest of the network).
            # eps=1e-8: achievable tolerance below the float32 output resolution; avoids
            # qpth's unreachable default eps=1e-12 (spurious "residual large" warnings).
            u_ = QPFunction(eps=1e-8, maxIter=50, verbose=-1)(self.Q.double(), self.p.double(),
                            self.G.double(), self.h.double(), self.e, self.e).float()
            # return the first column of the solution
            return u_[:,0]

if __name__ == "__main__":
    # test
    x = torch.tensor([[0, 10, 0, 0, 0, 10, 0, 0, 5, 10, 0, 0, 15, 20, 0, 0],[0, 10, 0, 0, 0, 10, 0, 0, 5, 10, 0, 0, 15, 20 ,0 ,0]])
    x = torch.tensor([0, 10, 0, 0, 0, 10, 0, 0, 5, 10, 0, 0, 15, 20, 0, 0])
    CAV_index = 1
    u_nominal = torch.tensor([[3],[4]])
    u_nominal = torch.tensor([4])
    tau = 0.3
    BN_model = BarrierLayer(8)
    output = BN_model(u_nominal, x, tau, CAV_index)
    print(output.size())
    print(u_nominal.size())
    print(output + u_nominal)
