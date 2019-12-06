from argparse import Namespace

import torch

from marl.experimental.deeprl.exploration.exploration_sampler import ExplorationSampler
from marl.experimental.deeprl.exploration.exponential_decay_scheduler import ExponentialDecayScheduler


class ExplorationSamplerExponential:

    decay_scheduler: ExponentialDecayScheduler

    epsilon: float

    def __init__(self,
                 action_dim: int,
                 config: Namespace):
        """
        Goes exponentially from 1 to 0 according to the: epsilon = e^(-decay_slope * step / total_steps)

        This decay is repeated num_annealings times through the expected_num_steps.
        """

        self.decay_scheduler = ExponentialDecayScheduler(expected_num_steps=config.num_steps,
                                                         num_annealings=config.eps_num_annealings,
                                                         decay_slope=config.eps_decay_slope)

        assert action_dim > 0
        self.num_actions = action_dim
        self.epsilon = 1.0

    def pick_action(self, q_values: torch.tensor) -> int:
        """Pick randomized action from given q_values, epsilon decays exponentially with a given rate"""

        if self.decay_scheduler.is_scheduling_enabled:
            self.epsilon = self.decay_scheduler.compute_exponential_decay()

        return ExplorationSampler.pick_randomized_action(q_values, self.num_actions, self.epsilon)

    def set_epsilon(self, epsilon: float):
        self.decay_scheduler.enable_scheduling(False)
        self.epsilon = epsilon

