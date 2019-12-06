from dataclasses import dataclass
from typing import List, Tuple

import torch
import random

from marl.experimental.deeprl.utils.available_device import my_device
from marl.experimental.deeprl.utils.buffer import Buffer


@dataclass
class ATOCClassifierSample:
    thought: torch.Tensor  # input to the classifier
    q_delta: torch.Tensor  # delta_q - used for training


class ATOCClassifierBuffer(Buffer):

    thought_size: int
    num_actions: int

    buffer: List[ATOCClassifierSample]

    def __init__(self,
                 buffer_size: int,
                 thought_size: int,
                 _run=None):
        super().__init__(buffer_size=buffer_size)

        assert thought_size > 0

        self.thought_size = thought_size
        self._run = _run
        self._num_deleted = 0

    def _check_inputs(self,
                      thought: torch.Tensor,
                      q_delta: torch.Tensor):

        assert thought.numel() == self.thought_size
        assert q_delta.numel() == 1

    def remember(self,
                 thought: torch.Tensor,
                 q_delta: torch.Tensor):
        """ Remember the sample for the ATOC classifier training (q_delta is the ground-truth)
        """

        self._check_inputs(thought, q_delta)
        transition = ATOCClassifierSample(thought, q_delta)
        self.push_to_buffer(transition)

    def sample_normalized_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: thoughts, q_deltas.. where q_deltas are normalized to <0,1> interval
        """
        assert batch_size > 0

        thoughts = torch.zeros((batch_size, self.thought_size), device=my_device())
        deltas = torch.zeros((batch_size, 1), device=my_device())

        for sample in range(batch_size):
            choice = random.randint(0, self.num_items - 1)
            thoughts[sample] = self.buffer[choice].thought
            deltas[sample] = self.buffer[choice].q_delta

        # normalize to <-0,1>
        min_value = torch.min(deltas)
        val_range = torch.max(deltas) - min_value
        norm_deltas = (deltas - min_value) / val_range

        return thoughts, norm_deltas

