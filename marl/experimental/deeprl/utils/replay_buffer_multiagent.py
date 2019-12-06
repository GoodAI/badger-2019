from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch

from marl.experimental.deeprl.utils.replay_buffer import ReplayBuffer


@dataclass
class MultiagentTransition:
    """Transition for N parallel agents"""

    state: List[torch.Tensor]
    action: List[torch.Tensor]
    new_state: List[torch.Tensor]
    reward: List[float]
    done: List[bool]


class ReplayBufferMultiagent(ReplayBuffer):
    """The same as ReplayBuffer, but works for multiple agents at the same time"""

    buffer: List[MultiagentTransition]
    input_sizes: List[int]
    action_sizes: List[int]

    def __init__(self,
                 buffer_size: int,
                 input_sizes: List[int],
                 action_sizes: List[int],
                 _run=None):
        super().__init__(
            buffer_size=buffer_size,
            state_size=-1,
            num_actions=-1,
            _run=_run)

        assert len(input_sizes) == len(action_sizes)
        self.input_sizes = input_sizes
        self.action_sizes = action_sizes

    @staticmethod
    def _sanitize_inputs(states: List[torch.Tensor],
                         actions: List[Union[int, torch.Tensor]],
                         new_states: List[torch.Tensor],
                         rewards: List[Union[float, np.ndarray]],
                         dones: List[bool]):
        res_states = []
        res_actions = []
        res_new_states = []
        res_rewards = []
        res_dones = []

        for state, action, new_state, reward, done in zip(states, actions, new_states, rewards, dones):
            r_state, r_action, r_new_state, r_reward, r_done = ReplayBuffer._sanitize_inputs(state,
                                                                                             action,
                                                                                             new_state,
                                                                                             reward,
                                                                                             done)
            res_states.append(r_state)
            res_actions.append(r_action)
            res_new_states.append(r_new_state)
            res_rewards.append(r_reward)
            res_dones.append(r_done)

        return res_states, res_actions, res_new_states, res_rewards, res_dones

    def _check_inputs(self,
                      states: List[torch.Tensor],
                      actions: List[torch.Tensor],
                      new_states: List[torch.Tensor],
                      rewards: List[float],
                      dones: List[bool]):

        assert isinstance(states, List)
        assert isinstance(actions, List)
        assert isinstance(new_states, List)
        assert isinstance(rewards, List)
        assert isinstance(dones, List)

        assert len(states) == len(actions) == len(new_states) == len(rewards) == len(dones),\
            'inconsistent list lengths!'

        for state, action, new_state, reward, done, input_size, action_size in \
                zip(states, actions, new_states, rewards, dones, self.input_sizes, self.action_sizes):

            assert state.numel() == input_size
            assert new_state.numel() == input_size
            assert action.numel() == action_size

            assert isinstance(reward, float)
            assert isinstance(done, bool)

    def remember(self,
                 states: List[torch.Tensor],
                 actions: List[torch.Tensor],
                 new_states: List[torch.Tensor],
                 rewards: List[float],
                 dones: List[bool]):
        """
        Remember one transition for all agents separately in MultiagentTransition.
        """

        states, actions, new_states, rewards, dones = self._sanitize_inputs(states, actions, new_states, rewards, dones)

        self._check_inputs(states, actions, new_states, rewards, dones)

        transition = MultiagentTransition(states, actions, new_states, rewards, dones)

        self.push_to_buffer(transition)

