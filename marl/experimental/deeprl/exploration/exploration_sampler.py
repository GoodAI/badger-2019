import random
from argparse import Namespace

import torch


class ExplorationSampler:
    epsilon: float
    num_actions: int

    def __init__(self, action_dim: int, config: Namespace):
        # TODO determinism here
        self.num_actions = action_dim
        epsilon = config.epsilon
        assert 0 <= epsilon <= 1, 'epsilon value out of bounds'
        self.epsilon = epsilon

    def pick_action(self, q_values: torch.tensor) -> int:
        return self.pick_randomized_action(q_values, self.num_actions, self.epsilon)

    def set_epsilon(self, epsilon: float):
        self.epsilon = epsilon

    @staticmethod
    def pick_randomized_action(q_values: torch.tensor, num_actions: int, epsilon: float) -> int:
        """Pick a discrete action given vector of q-values"""
        assert q_values.numel() == num_actions, f'incompatible num. actions: {q_values.shape}'
        assert 0 <= epsilon <= 1, f'epsilon out of bounds (value: {epsilon})'

        if random.random() < epsilon:
            return random.randint(0, num_actions - 1)

        # dims: [sequence, minibatch, data]
        max_val, max_action = torch.max(q_values, dim=-1)
        # print(f'max_val {max_val} from {q_values}, ind: {max_action.item()}')
        return max_action.item()
