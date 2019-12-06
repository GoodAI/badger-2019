from argparse import Namespace
from pydoc import locate
from typing import Dict, Any, List, Optional, Union

import gym
import numpy as np

from marl.experimental.deeprl.policies.ddpg_policy_batched import DDPGPolicyBatched
from marl.experimental.deeprl.policies.policy_base import PolicyBase


class IndependentPolicies(PolicyBase):
    """Contains N independent DDPG policies working in multiagent environment.

    Should serve just as a baseline without MA-*** improvements.
    """
    policies: List[DDPGPolicyBatched]  # theoretically anything single-agent from PolicyBase
    num_agents: int
    run_id: int

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

        self.policies = []

        policy_class = locate(config.independent_policy)  # what policy are we going to use?

        for obs_space, act_space in zip(observation_space, action_space):
            policy = policy_class(observation_space=obs_space,
                                  action_space=act_space,
                                  config=config,
                                  _run=_run,
                                  run_id=run_id)
            self.policies.append(policy)

    def pick_action(self, observation: List[np.ndarray]) -> List[np.array]:
        assert isinstance(observation, List)
        assert len(observation) == len(self.policies)

        actions = []

        for policy, obs in zip(self.policies, observation):
            action = policy.pick_action(obs).reshape(-1)
            actions.append(action)

        return actions

    def remember(self, new_observation: List[np.array], reward: List[float], done: List[bool]):
        for policy, obs, rew, dn in zip(self.policies, new_observation, reward, done):
            policy.remember(obs, rew, dn)

    def learn(self, batch_size: Optional[int] = None):
        for policy in self.policies:
            policy.learn(batch_size)

    def set_epsilon(self, epsilon: float):
        for policy in self.policies:
            policy.set_epsilon(epsilon)

    @property
    def epsilon(self) -> float:
        return self.policies[0].epsilon

    def reset(self, batch_size: int = 1):
        for policy in self.policies:
            policy.reset(batch_size)

    def log(self, step: int):

        if self._run is not None:
            # common for all policies
            self._run.log_scalar(f'{self.run_id} exploration', self.policies[0].exploration.st_dev)
            self._run.log_scalar(f'{self.run_id} buff_num_written', self.policies[0].buffer.num_total_written_items)
            self._run.log_scalar(f'{self.run_id} buff_num_deleted', self.policies[0].buffer.num_deleted_items)

            self._run.log_scalar(f'{self.run_id} step', step)
            self._run.log_scalar(f'{self.run_id} num_learns', self.policies[0].num_learns)

            total_avg_rew = 0

            # separate for each policy
            for pol_id, policy in enumerate(self.policies):
                self._run.log_scalar(f'{self.run_id} pol_{pol_id} actor_loss', policy.last_actor_loss)
                self._run.log_scalar(f'{self.run_id} pol_{pol_id} critic_loss', policy.last_critic_loss)

                avg_reward = policy.get_avg_reward()
                self._run.log_scalar(f'{self.run_id} pol_{pol_id} avg_reward', avg_reward)
                total_avg_rew += avg_reward

            total_avg_rew /= len(self.policies)
            self._run.log_scalar(f'{self.run_id} total_avg_reward', total_avg_rew)

    def serialize(self) -> Dict[str, Any]:
        result = {}
        for pol_id, policy in enumerate(self.policies):
            result[f'policy_{pol_id}'] = policy.serialize()

        return result

    def deserialize(self, data: Dict[str, Any]):
        for pol_id, policy in enumerate(self.policies):
            policy.deserialize(data[f'policy_{pol_id}'])
