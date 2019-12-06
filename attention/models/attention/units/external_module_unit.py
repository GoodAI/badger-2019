from abc import abstractmethod
from typing import Generic, Optional, TypeVar, Callable

from torch import Tensor, nn as nn

from attention.agents.agent import T_INIT_ROLLOUT_PARAMS
from attention.models.attention.multi_unit_attention import AttentionUnit, AttentionMessages
from attention.models.with_module import WithModule
from attention.utils.torch_utils import expand_to_batch_size

T_P = TypeVar('T_P')
T_R = TypeVar('T_R')


class ExternalModuleUnit(AttentionUnit[T_P, Tensor, T_INIT_ROLLOUT_PARAMS], WithModule, Generic[T_P, T_INIT_ROLLOUT_PARAMS]):
    _m: nn.Module

    def __init__(self, module: nn.Module):
        self._m = module

    @abstractmethod
    def generate_messages(self, params: T_P) -> AttentionMessages:
        pass

    @abstractmethod
    def receive_attention(self, result: Tensor, params: T_P) -> Tensor:
        pass

    @property
    def module(self) -> Optional[nn.Module]:
        return self._m


class FixedKeyUnit(ExternalModuleUnit[T_P, T_INIT_ROLLOUT_PARAMS], Generic[T_P, T_INIT_ROLLOUT_PARAMS]):
    def __init__(self, key: Tensor, module: nn.Module, param_extractor: Callable[[T_P], Tensor]):
        """
        Args:
            key: float[n_outputs, key_size]
            module:

        """
        super().__init__(module)
        self._key = key.unsqueeze(0)
        self._param_extractor = param_extractor

    def generate_messages(self, params: T_P) -> AttentionMessages:
        return AttentionMessages(
            expand_to_batch_size(self._key, self.batch_size),
            self.module(self._param_extractor(params)),
            None)

    def receive_attention(self, result: Tensor, params: T_P) -> Tensor:
        return result


class FixedQueryUnit(ExternalModuleUnit[T_P, T_INIT_ROLLOUT_PARAMS], Generic[T_P, T_INIT_ROLLOUT_PARAMS]):

    def __init__(self, query: Tensor, module: nn.Module):
        """
        Args:
            query: float[n_outputs, key_size]
            module
        """
        super().__init__(module)
        self._query = query.unsqueeze(0)

    def generate_messages(self, params: T_P) -> AttentionMessages:
        return AttentionMessages(None, None, expand_to_batch_size(self._query, self.batch_size))

    def receive_attention(self, result: Tensor, params: T_P) -> Tensor:
        # noinspection PyTypeChecker
        return self.module(result)


class ExternalModuleWriteUnit(ExternalModuleUnit[T_P, T_INIT_ROLLOUT_PARAMS], Generic[T_P, T_INIT_ROLLOUT_PARAMS]):
    def __init__(self, module: nn.ModuleDict, param_extractor: Callable[[T_P], Tensor]):
        super().__init__(module)
        assert 'key' in dir(module)
        assert 'value' in dir(module)
        self._param_extractor = param_extractor

    def generate_messages(self, params: T_P) -> AttentionMessages:
        return AttentionMessages(
            self.module.key(self._param_extractor(params)),
            self.module.value(self._param_extractor(params)),
            None)

    def receive_attention(self, result: Tensor, params: T_P) -> Tensor:
        return result


class ExternalModuleReadUnit(ExternalModuleUnit[T_P, T_INIT_ROLLOUT_PARAMS], Generic[T_P, T_INIT_ROLLOUT_PARAMS]):
    def __init__(self, module: nn.Module, param_extractor: Callable[[T_P], Tensor]):
        super().__init__(module)
        self._param_extractor = param_extractor

    def generate_messages(self, params: T_P) -> AttentionMessages:
        return AttentionMessages(None, None, self.module(self._param_extractor(params)))

    def receive_attention(self, result: Tensor, params: T_P) -> Tensor:
        return result


class FilterAttentionUnit(AttentionUnit[T_P, T_R, T_INIT_ROLLOUT_PARAMS], Generic[T_P, T_R, T_INIT_ROLLOUT_PARAMS]):
    def __init__(self, unit: AttentionUnit[T_P, T_R, T_INIT_ROLLOUT_PARAMS]):
        self.unit = unit

    def generate_messages(self, params: T_P) -> AttentionMessages:
        return self.unit.generate_messages(params)

    def receive_attention(self, result: Tensor, params: T_P) -> T_R:
        return self.unit.receive_attention(result, params)


class OutputOnlyFilterAttentionUnit(FilterAttentionUnit[T_P, T_R, T_INIT_ROLLOUT_PARAMS], Generic[T_P, T_R, T_INIT_ROLLOUT_PARAMS]):

    def generate_messages(self, params: T_P) -> AttentionMessages:
        result = self.unit.generate_messages(params)
        return AttentionMessages(result.key, result.value, None)

    def receive_attention(self, result: Tensor, params: T_P) -> T_R:
        return result
