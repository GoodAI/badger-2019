from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TestNet(nn.Module):

    def __init__(self, input_size: int = 4, hidden_size: int = 128, num_actions: int = 2):
        super(TestNet, self).__init__()
        self.affine1 = nn.Linear(input_size, hidden_size)

        self.cell = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)
        self.mem = torch.zeros((1, hidden_size), dtype=torch.float32)

        # actor's layer
        self.action_head = nn.Linear(hidden_size, num_actions)

        # critic's layer
        self.value_head = nn.Linear(hidden_size, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # self.mem = self.cell(x, self.mem)
        # x = self.mem

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)
        # TODO maybe tanh?

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


if __name__ == '__main__':

    input_size = 2
    hidden_size = 16
    num_actions = 5

    bs = 1

    net = TestNet(input_size, hidden_size, num_actions)

    input = torch.zeros((bs, input_size))

    output = net.forward(input)

    print('done')
