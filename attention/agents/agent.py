from abc import abstractmethod, ABC
from typing import Optional, Any, Dict, Generic, TypeVar

from badger_utils.sacred import Serializable
from torch import nn as nn

from attention.utils.torch_utils import default_device


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


T_INIT_ROLLOUT_PARAMS = TypeVar('T_INIT_ROLLOUT_PARAMS')


class RolloutAware(ABC, Generic[T_INIT_ROLLOUT_PARAMS]):
    @abstractmethod
    def init_rollout(self, batch_size: int, n_experts: int, params: T_INIT_ROLLOUT_PARAMS):
        pass


class Agent(DeviceAwareModule, RolloutAware[T_INIT_ROLLOUT_PARAMS], Serializable, Generic[T_INIT_ROLLOUT_PARAMS]):

    def __init__(self, device: Optional[str]):
        super().__init__(device)

    @abstractmethod
    def init_rollout(self, batch_size: int, n_experts: int, params: T_INIT_ROLLOUT_PARAMS):
        pass

    def serialize(self) -> Dict[str, Any]:
        return {
            'model': self.state_dict(),
            'optim': self.optim.state_dict()
        }

    def deserialize(self, data: Dict[str, Any]):
        self.load_state_dict(data['model'])
        self.optim.load_state_dict(data['optim'])
