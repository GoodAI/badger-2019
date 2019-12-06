from argparse import Namespace
from pydoc import locate
from typing import Dict, Any, List, Optional, Union

import gym
import numpy as np

from marl.experimental.deeprl.policies.ddpg_policy_batched import DDPGPolicyBatched
from marl.experimental.deeprl.policies.policy_base import PolicyBase

from marl.experimental.deeprl.utils.utils import get_gym_box_dimensionality, concatenate_spaces


class GlobalPolicy(PolicyBase):
    """Contains 1 (**)DDPG policy working in multiagent environment in a way that it controls all the agents.

    Should serve just as a baseline without MA-*** improvements.
    """
    policy: DDPGPolicyBatched
    num_agents: int
    run_id: int

    action_sizes: List[int]

    def __init__(self,
                 observation_space: List[Union[gym.Space, gym.spaces.Tuple]],
                 action_space: List[Union[gym.Space, gym.spaces.Tuple]],
                 config: Namespace,
                 _run=None,
                 run_id: int = 0):
        super().__init__()

        self._run = _run
        self.run_id = run_id

        assert isinstance(observation_space, List)
        assert isinstance(action_space, List)
        assert len(action_space) == len(observation_space)

        num_agents = len(observation_space)
        assert num_agents > 0, 'not enough agents to control'

        policy_class = locate(config.independent_policy)  # what policy are we going to use?

        obs_space = concatenate_spaces(observation_space)
        act_space = concatenate_spaces(action_space)

        self.action_sizes = [get_gym_box_dimensionality(act) for act in action_space]

        self.policy = policy_class(observation_space=obs_space,
                                   action_space=act_space,
                                   config=config,
                                   _run=_run,
                                   run_id=run_id)

    def pick_action(self, observation: List[np.ndarray]) -> List[np.array]:
        assert isinstance(observation, List)

        global_observation = np.concatenate(observation)
        action = self.policy.pick_action(global_observation).reshape(-1)

        results = []
        pos = 0
        for size in self.action_sizes:
            act = action[pos:pos+size]
            results.append(act)
            pos += size

        return results

    def remember(self, new_observation: List[np.array], reward: List[float], done: List[bool]):
        """ Done if all agents done, reward is sum of agent rewards, observations are concatenated
        """

        new_global_obs = np.concatenate(new_observation)
        global_reward = sum(reward)
        global_done = all(done)

        self.policy.remember(new_global_obs, global_reward, global_done)

    def learn(self, batch_size: Optional[int] = None):
        self.policy.learn(batch_size)

    def set_epsilon(self, epsilon: float):
        self.policy.set_epsilon(epsilon)

    @property
    def epsilon(self) -> float:
        return self.policy.epsilon

    def reset(self, batch_size: int = 1):
        self.policy.reset(batch_size)

    def log(self, step: int):

        if self._run is not None:
            # common for all policies
            self._run.log_scalar(f'{self.run_id} exploration', self.policy.exploration.st_dev)
            self._run.log_scalar(f'{self.run_id} buff_num_written', self.policy.buffer.num_total_written_items)
            self._run.log_scalar(f'{self.run_id} buff_num_deleted', self.policy.buffer.num_deleted_items)

            self._run.log_scalar(f'{self.run_id} step', step)
            self._run.log_scalar(f'{self.run_id} num_learns', self.policy.num_learns)

            self._run.log_scalar(f'{self.run_id} actor_loss', self.policy.last_actor_loss)
            self._run.log_scalar(f'{self.run_id} critic_loss', self.policy.last_critic_loss)

            avg_reward = self.policy.get_avg_reward()
            self._run.log_scalar(f'{self.run_id} avg_reward', avg_reward)

    def serialize(self) -> Dict[str, Any]:
        return self.policy.serialize()

    def deserialize(self, data: Dict[str, Any]):
        self.policy.deserialize(data)

