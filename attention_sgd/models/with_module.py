from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn as nn


class WithModule(ABC):

    @property
    @abstractmethod
    def module(self) -> Optional[nn.Module]:
        pass
