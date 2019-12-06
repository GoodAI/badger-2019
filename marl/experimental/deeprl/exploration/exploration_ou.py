from argparse import Namespace
import numpy as np
import random
import copy

import torch

from marl.experimental.deeprl.policies.networks import my_device
from marl.experimental.deeprl.exploration.exploration_sampler_exponential import ExponentialDecayScheduler


class OU_Noise:
    """Ornstein-Uhlenbeck process.

    Class taken from:
    https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/utilities/OU_Noise.py
    """
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.array([np.random.normal() for _ in range(len(self.state))])
        self.state += dx
        return self.state


class ExplorationOu:
    """OU noise exploration"""

    st_dev: float
    action_dim: int

    decay_scheduler: ExponentialDecayScheduler

    def __init__(self,
                 action_dim: int,
                 config: Namespace):

        self.st_dev = 1.0

        assert action_dim > 0
        self.action_dim = action_dim

        self.decay_scheduler = ExponentialDecayScheduler(expected_num_steps=config.num_steps,
                                                         num_annealings=config.eps_num_annealings,
                                                         decay_slope=config.eps_decay_slope)

        self.noise = OU_Noise(size=action_dim, seed=0, mu=config.mu, theta=config.theta, sigma=config.sigma)

    def pick_action(self, action: torch.tensor) -> torch.tensor:
        if self.decay_scheduler.is_scheduling_enabled:
            self.st_dev = self.decay_scheduler.compute_exponential_decay()

        noise = self.noise.sample() * self.st_dev

        noisy = action + torch.tensor(noise, device=my_device(), dtype=torch.float32)
        return noisy

    def set_epsilon(self, st_dev: float):
        """Actually set the scale of the OU noise"""
        self.decay_scheduler.enable_scheduling(False)
        assert 0 <= st_dev <= 1
        self.st_dev = st_dev

