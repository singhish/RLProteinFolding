from typing import Tuple
import torch
from torch import nn, optim
from torch.nn import functional as F

from .replay_buffer import ReplayBuffer
from .ou_process import OrnsteinUhlenbeckProcess as OUProcess
from .actor_critic import Actor, Critic
from protein import ProteinState


def update_target_nnet(nnet_target: nn.Module, nnet: nn.Module, soft=True, tau=0.001):
    for param_t, param in zip(nnet_target.parameters(), nnet.parameters()):
            param_t.data = (tau * param.data + (1 - tau) * param_t.data) if soft else param.data

class Agent:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 action_bounds: Tuple[float, float],
                 actor_lr=0.0001,
                 critic_lr=0.001,
                 gamma=0.99):

        self.actor = Actor(state_dim, action_dim, action_bounds)
        self.actor_t = Actor(state_dim, action_dim, action_bounds)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim)
        self.critic_t = Critic(state_dim, action_dim)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.action_bounds = action_bounds
        self.gamma = gamma
        self.iter = 0
        self.N = OUProcess(action_dim)

        update_target_nnet(self.actor_t, self.actor, soft=False)
        update_target_nnet(self.critic_t, self.critic, soft=False)

    def get_action(self, state: ProteinState):
        state_tensor = torch.from_numpy(state.angles().flatten()).float()

        action = self.actor(state_tensor).detach().data.numpy()

        action_range_diff = self.action_bounds[1] - self.action_bounds[0]
        action_offset = (self.action_bounds[0] + self.action_bounds[1]) / 2
        noised_action = action + ((self.N.sample() * action_range_diff) + action_offset)
        noised_action %= 360
        return noised_action.reshape(state.angles().shape)

    def update(self, buffer: ReplayBuffer, batch_size=100):
        s, a, r, s_p = tuple(buffer.sample(batch_size))

        s_tensor = torch.from_numpy(s).float()
        a_tensor = torch.from_numpy(a).float()
        r_tensor = torch.from_numpy(r).float()
        s_p_tensor = torch.from_numpy(s_p).float()

        # optimize critic
        a_p_pred_tensor = self.actor_t(s_p_tensor).detach()
        q_next = torch.squeeze(self.critic_t(s_p_tensor, a_p_pred_tensor).detach())
        y_exp = r_tensor + self.gamma * q_next
        y_pred = torch.squeeze(self.critic(s_tensor, a_tensor))
        critic_loss = F.mse_loss(y_exp, y_pred)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # optimize actor
        a_pred_tensor = self.actor(s_tensor)
        actor_loss = -torch.sum(self.critic(s_tensor, a_pred_tensor))
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        update_target_nnet(self.actor_t, self.actor)
        update_target_nnet(self.critic_t, self.critic)
