from abc import abstractmethod, ABC
from typing import Optional, Any, Dict

from badger_utils.sacred import Serializable
from torch import nn as nn

from attention_sgd.utils.torch_utils import default_device


class DeviceAware(ABC):
    device: str = 'cpu'

    def __init__(self, device: Optional[str]):
        if device is None:
            device = default_device()
        self.device = device


class DeviceAwareModule(nn.Module):
    device: str = 'cpu'

    def __init__(self, device: Optional[str]):
        super().__init__()
        if device is None:
            device = default_device()
        self.device = device

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        result.device = args[0]  # hack to extract device
        return result


class RolloutAware:
    @abstractmethod
    def init_rollout(self, batch_size: int, n_experts: int):
        pass


class Agent(DeviceAwareModule, RolloutAware, Serializable):

    def __init__(self, device: Optional[str]):
        super().__init__(device)

    @abstractmethod
    def init_rollout(self, batch_size: int, n_experts: int):
        pass

    def serialize(self) -> Dict[str, Any]:
        return {
            'model': self.state_dict(),
            'optim': self.optim.state_dict()
        }

    def deserialize(self, data: Dict[str, Any]):
        self.load_state_dict(data['model'])
        self.optim.load_state_dict(data['optim'])
