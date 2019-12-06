from functools import reduce
from typing import Callable, Tuple

import torch
from torch import Tensor

from attention.search_task import TorchTask


class BatchedTask(TorchTask):
    def __init__(self, batch_size: int, task_factory: Callable[[], TorchTask]):
        # self._batch_size = batch_size
        # self._task_factory = task_factory

        self._tasks = [task_factory() for i in range(batch_size)]

    def step(self, action: Tensor) -> Tuple[Tensor, Tensor, bool, dict]:
        results = [task.step(action[i]) for i, task in enumerate(self._tasks)]
        return (
            torch.stack([r[0] for r in results]),
            torch.stack([r[1] for r in results]),
            reduce(lambda a, b: a or b, [r[2] for r in results], False),  # TODO maybe just return array of booleans?
            {}  # ignore the dict
        )

    def reset(self) -> Tensor:
        return torch.stack([task.reset() for task in self._tasks])
