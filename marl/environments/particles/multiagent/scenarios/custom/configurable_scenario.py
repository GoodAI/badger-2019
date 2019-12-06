from abc import ABC, abstractmethod

import numpy as np
from marl.environments.particles.multiagent.core import World, Agent, Landmark
from marl.environments.particles.multiagent.scenarios.base_scenario import BaseScenario


def has_setup_method(scenario):
    setup_op = getattr(scenario, "setup", None)
    if callable(setup_op):
        return True
    return False


class ConfigurableScenario(BaseScenario, ABC):

    configured: bool

    def __init__(self, num_agents: int, episode_length: int = 16):
        super().__init__(num_agents=num_agents, episode_length=episode_length)
        self.configured = False

    @abstractmethod
    def setup(self, **kwargs):
        """
        The environment can be configured by calling the setup method
        Args:
            **kwargs: a dictionary with a custom setup
        """
        self.configured = True

    @abstractmethod
    def reset_world(self, world):
        super().reset_world(world)
        assert self.configured, 'setup not called'
