import random
from dataclasses import dataclass
from typing import List, Union, Tuple

import torch

import numpy as np

from marl.experimental.deeprl.utils.buffer import Buffer


@dataclass
class Transition:

    state: torch.Tensor
    action: Union[int, torch.Tensor]
    new_state: torch.Tensor
    reward: float
    done: bool


class ReplayBuffer(Buffer):

    state_size: int
    num_actions: int

    buffer: List[Transition]

    def __init__(self,
                 buffer_size: int,
                 state_size: int,
                 num_actions: int,
                 _run=None):
        super().__init__(buffer_size=buffer_size)

        self.num_actions = num_actions
        self.state_size = state_size
        self._run = _run

    @staticmethod
    def sanitize_reward(reward: Union[float, np.ndarray]) -> float:

        # turns out that reward can be multidimensional array..
        if isinstance(reward, np.ndarray):
            assert reward.size == 1
            while isinstance(reward, np.ndarray):
                reward = reward[0]
            reward = float(reward)

        reward = float(reward)
        return reward

    @staticmethod
    def _sanitize_inputs(state: torch.Tensor,
                         action: Union[int, torch.Tensor],
                         new_state: torch.Tensor,
                         reward: float,
                         done: bool):
        """Expected IO is 3D: [batch_s=1, seq_len=1, size], reshape if necessary."""

        # expected IO is 3D: [batch_s, seq_len, size]
        state = state.view(1, 1, -1)
        new_state = new_state.view(1, 1, -1)

        reward = ReplayBuffer.sanitize_reward(reward)

        if isinstance(action, torch.Tensor):
            action = action.view(1, 1, -1)

        return state, action, new_state, reward, done

    def _check_inputs(self,
                      state: torch.Tensor,
                      action: Union[int, torch.Tensor],
                      new_state: torch.Tensor,
                      reward: float,
                      done: bool):

        assert state.numel() == self.state_size
        assert new_state.numel() == self.state_size

        if isinstance(action, int):
            assert 0 <= action < self.num_actions
        elif isinstance(action, torch.Tensor):
            assert action.numel() == self.num_actions
        else:
            assert False, 'unsupported action type'

        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def remember(self,
                 state: torch.Tensor,
                 action: Union[int, torch.Tensor],
                 new_state: torch.Tensor,
                 reward: float,
                 done: bool):
        """
        Remember one transition - place it to the buffer.
        """

        state, action, new_state, reward, done = self._sanitize_inputs(state, action, new_state, reward, done)

        self._check_inputs(state=state, action=action, new_state=new_state, reward=reward, done=done)

        transition = Transition(state, action, new_state, reward, done)

        self.push_to_buffer(transition)

    def sample(self, batch_size: int):
        """ Sample the memories (transitions or episodes) from the buffer

        Returns: list of memories (transitions, episodes..)
        """
        assert batch_size > 0

        batch = []

        for sample in range(batch_size):
            choice = random.randint(0, self.num_items - 1)
            batch.append(self.buffer[choice])

        return batch

