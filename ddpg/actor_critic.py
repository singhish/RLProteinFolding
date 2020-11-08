import torch
import numpy as np
from torch import nn
from typing import Tuple

from protein import ProteinState


def fanin_init(size):
    return torch.Tensor(size).uniform_(-1. / np.sqrt(size[0]), 1. / np.sqrt(size[0]))

class Actor(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 action_bounds: Tuple[float, float],
                 epsilon=0.003):

        super(Actor, self).__init__()

        assert action_bounds[1] > action_bounds[0]
        self.action_bounds = action_bounds

        self.fc1 = nn.Linear(state_dim, 400)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(400, 300)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(300, action_dim)
        self.fc3.weight.data.uniform_(-epsilon, epsilon)
        self.tanh = nn.Tanh()

    def forward(self, state: torch.Tensor):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))

        action_range_diff = self.action_bounds[1] - self.action_bounds[0]
        action_offset = (self.action_bounds[0] + self.action_bounds[1]) / 2

        action = (x * action_range_diff) + action_offset

        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, epsilon=0.003):
        super(Critic, self).__init__()

        self.s_fc1 = nn.Linear(state_dim, 400)
        self.s_fc1.weight.data = fanin_init(self.s_fc1.weight.data.size())
        self.relu = nn.ReLU()

        self.s_fc2 = nn.Linear(400, 300)
        self.s_fc2.weight.data = fanin_init(self.s_fc2.weight.data.size())

        self.a_fc2 = nn.Linear(action_dim, 300)
        self.a_fc2.weight.data = fanin_init(self.a_fc2.weight.data.size())

        self.fc3 = nn.Linear(600, 200)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

        self.fc4 = nn.Linear(200, 1)
        self.fc4.weight.data.uniform_(-epsilon, epsilon)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        s = self.relu(self.s_fc1(state))
        s = self.relu(self.s_fc2(s))

        a = self.relu(self.a_fc2(action))

        x = torch.cat((s, a), dim=1)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x
