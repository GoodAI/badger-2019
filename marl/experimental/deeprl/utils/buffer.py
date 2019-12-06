from abc import ABC
from typing import List, Any


class Buffer(ABC):

    buffer_size: int
    buffer: List[Any]

    _num_deleted: int

    def __init__(self,
                 buffer_size: int):
        self.buffer_size = buffer_size

        self.buffer = []
        self._num_deleted = 0

    @property
    def num_items(self) -> int:
        return len(self.buffer)

    @property
    def num_deleted_items(self) -> int:
        return self._num_deleted

    @property
    def num_total_written_items(self) -> int:
        total_written = self.num_items
        if self.num_items == self.buffer_size:
            total_written = self._num_deleted + self.buffer_size

        return total_written

    def push_to_buffer(self, sample):
        """Not meant to be used from the outside"""
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
            self._num_deleted += 1

        self.buffer.append(sample)
