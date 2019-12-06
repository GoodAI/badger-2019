from abc import ABC, abstractmethod
from typing import List, Union


class BaseScenario(ABC):
    """ Defines scenario upon which the world is built
    """
    steps_from_last_reset: int
    num_agents: int
    episode_length: int

    def __init__(self,
                 num_agents: int,
                 episode_length: int = 16):

        self.episode_length = episode_length
        self.num_agents = num_agents
        self.steps_from_last_reset = 0

    @abstractmethod
    def make_world(self):
        """ Create elements of the world """
        raise NotImplementedError()

    def reset_world(self, world):
        """ create initial conditions of the world """
        self.steps_from_last_reset = 0

    def is_done(self, agent, world) -> bool:
        """ Environment should be able to say that it is done (for each agent ideally).

        Step is called num_agent times """
        if self.steps_from_last_reset / self.num_agents > self.episode_length:
            return True
        return False

    def observation(self, agent, world):
        """ Returns the observation for the agent passed in the parameter"""
        self.steps_from_last_reset += 1


