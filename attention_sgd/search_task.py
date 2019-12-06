import pickle
import random
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from torch import Tensor
import torch

import numpy as np
import gym

from attention_sgd.agents.agent import DeviceAware


class FillTask(gym.Env, ABC):
    _buffer: np.ndarray

    def __init__(self, size: int):
        self._size = size
        self.reset()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, dict]:
        new_cells_hit = 0

        # fill the buffer
        for i in action:
            if self._buffer[i] == 0:
                new_cells_hit += 1
            self._buffer[i] = 1

        # compute reward
        reward = 0 if action.shape[0] == 0 else new_cells_hit / action.shape[0]

        is_done = np.all(self._buffer)

        return self._buffer, np.array([reward], dtype=np.float32), is_done, {}

    def reset(self) -> np.ndarray:
        self._buffer = np.zeros((self._size,), dtype=np.int)
        return self._buffer

    def render(self, mode='human'):
        pass


class TorchTask(ABC):
    @abstractmethod
    def step(self, action: Tensor) -> Tuple[Tensor, Tensor, bool, dict]:
        pass

    @abstractmethod
    def reset(self) -> Tensor:
        pass


class FillTorchTask(TorchTask, DeviceAware):
    _buffer: Tensor

    def __init__(self, size: int, device: Optional[str] = None):
        super().__init__(device)
        self._size = size
        self.reset()

    def step(self, action: Tensor) -> Tuple[Tensor, Tensor, bool, dict]:

        new_cells_hit = torch.tensor([0.0], device=self.device)

        # fill the buffer
        for i in action:
            # TODO - if is not propagating gradients - rewrite to multiplication?
            if self._buffer[i].item() == 0:
                new_cells_hit[0] += 1
            self._buffer[i] = True

        # compute reward
        reward = 0 if action.shape[0] == 0 else new_cells_hit / action.shape[0]

        is_done = torch.all(self._buffer).item()

        return self._buffer, torch.tensor(reward), is_done, {}

    def reset(self) -> Tensor:
        self._buffer = torch.zeros((self._size,), dtype=torch.uint8, device=self.device)
        return self._buffer


class ToZeroTorchTask(TorchTask, DeviceAware):
    def __init__(self, size: int, device: Optional[str] = None):
        super().__init__(device)
        self._size = size
        self._observations = torch.ones((size, ), device=self.device)
        self.reset()

    def step(self, action: Tensor) -> Tuple[Tensor, Tensor, bool, dict]:
        """

        Args:
            action: [n_experts, self._size]

        Returns:

        """
        # targets = torch.zeros((self._size,), device=self.device)
        # targets are zero, so no need to subtract them
        result_diff = action[:, 0:1]  # use just the first value of results vector
        loss = torch.mean(result_diff ** 2)#.view(1)
        is_done = False
        return self._observations, loss, is_done, {}

    def reset(self) -> Tensor:
        return self._observations

    # def _empty_tensor(self):
    #     return torch.tensor(0, device=self.device)
