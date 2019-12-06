from typing import Optional
import numpy as np


class ExponentialDecayScheduler:

    expected_num_steps: int
    num_annealings: int
    decay_slope: float
    anneal_length: int

    current_step: int

    _is_scheduling_enabled: bool

    def __init__(self,
                 expected_num_steps: int,
                 num_annealings: Optional[int] = 1,
                 decay_slope: Optional[int] = 5):
        """
        Goes exponentially from 1 to 0 according to the: epsilon = e^(-decay_slope * step / total_steps)

        This decay is repeated num_annealings times through the expected_num_steps.

        Args:
            expected_num_steps:
            num_annealings:
            decay_slope:
        """

        assert num_annealings > 0
        assert num_annealings < expected_num_steps
        assert decay_slope >= 1

        self.expected_num_steps = expected_num_steps
        self.num_annealings = num_annealings
        self.decay_slope = decay_slope
        self.anneal_length = expected_num_steps // num_annealings

        self.current_step = 1
        self._is_scheduling_enabled = True

    @property
    def is_scheduling_enabled(self) -> bool:
        return self._is_scheduling_enabled

    def enable_scheduling(self, enable: bool):
        self._is_scheduling_enabled = enable

    def compute_exponential_decay(self) -> float:
        self.current_step += 1
        return np.exp(-self.decay_slope * (self.current_step % self.anneal_length) / self.anneal_length)
