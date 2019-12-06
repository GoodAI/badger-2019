from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class PolicyNetwork(nn.Module):
    """
    implements both actor and critic in one model
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 128, num_actions: int = 2):
        super(PolicyNetwork, self).__init__()
        self.affine1 = nn.Linear(input_size, hidden_size)
        self.affine2 = nn.Linear(hidden_size, hidden_size)

        self.cell = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)
        self.mem = torch.zeros((1, hidden_size), dtype=torch.float32)

        # actor's layer
        # self.actor_cell = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)
        # self.actor_mem = torch.zeros((1, hidden_size), dtype=torch.float32)
        self.action_head = nn.Linear(hidden_size, num_actions)

        # critic's layer
        # self.value_cell = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)
        # self.value_mem = torch.zeros((1, hidden_size), dtype=torch.float32)
        self.value_head = nn.Linear(hidden_size, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = x.unsqueeze(0)

        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))

        # self.mem = self.cell(x, self.mem)
        # x = self.mem

        # actor: choses action to take from state s_t by returning probability of each action
        # a_x = self.actor_cell(x, self.actor_mem)
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        # s_x = self.value_cell(x, self.value_mem)
        state_values = self.value_head(x)
        # TODO maybe tanh?

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob.squeeze(0), state_values.squeeze(0)

    def reset(self):
        # self.actor_mem.zero_()
        # self.actor_mem.detach_()
        # self.value_mem.zero_()
        # self.value_mem.detach_()
        self.mem.zero_()
        self.mem.detach_()

