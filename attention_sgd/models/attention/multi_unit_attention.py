from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, NamedTuple, Callable, Optional

# class UnitParams(ABC):
#     pass
import torch
from torch import Tensor

from attention_sgd.agents.agent import RolloutAware

T_PARAM = TypeVar('T_PARAM')
T_RESULT = TypeVar('T_RESULT')


class AttentionMessages(NamedTuple):
    key: Optional[Tensor]
    value: Optional[Tensor]
    query: Optional[Tensor]


class RolloutAwareAndStore(RolloutAware):
    n_experts: int
    batch_size: int

    def init_rollout(self, batch_size: int, n_experts: int):
        self.batch_size = batch_size
        self.n_experts = n_experts


class AttentionUnit(Generic[T_PARAM, T_RESULT], RolloutAwareAndStore, ABC):

    @abstractmethod
    def generate_messages(self, params: T_PARAM) -> AttentionMessages:
        """

        Returns:
            (key, query, value) where
                key: float[batch_size, n_experts * out_heads, key_size]
                query: float[batch_size, n_experts * in_heads, key_size]
                value: float[batch_size, n_experts * out_heads, hidden_state_size]
            Note, in dimension 1, experts are packed together for each head, i.e.: e1h1,e2h1,e3h1,e1h2,e2h2,e3h2
        """
        pass

    @abstractmethod
    def receive_attention(self, result: Tensor, params: T_PARAM) -> T_RESULT:
        """
        Compute new hidden state from input values. These are the results of attention computation.

        Args:
            params:
            result: float[batch_size, n_experts * in_heads, hidden_state_size]

        Returns:

        """
        pass


class MultiUnitAttention(RolloutAwareAndStore, Generic[T_PARAM]):
    _queries_per_unit: List[int] = None

    def __init__(self, units: List[AttentionUnit[T_PARAM, Tensor]], result_callback: Callable[[List[Tensor]], None],
                 unit_params: Callable[[], T_PARAM]):
        self.units = units
        self._result_callback = result_callback
        self._unit_params = unit_params
        super().__init__()

    def init_rollout(self, batch_size: int, n_experts: int):
        for unit in self.units:
            unit.init_rollout(batch_size, n_experts)

    def generate_messages(self) -> AttentionMessages:
        keys = []
        values = []
        queries = []
        self._queries_per_unit = []
        for unit in self.units:
            k, v, q = unit.generate_messages(self._unit_params())
            k_set, v_set, q_set = [i is not None for i in [k, v, q]]
            # Sanity check
            assert k_set == v_set, "key and values must be both present or missing"

            if k_set:
                assert k.size(1) == v.size(1), "number of keys and values must match"
                assert k.size(0) == v.size(0), "batch size must match"

            if k_set and q_set:
                assert k.size(0) == v.size(0) == q.size(0), "batch size must match"
                assert k.size(2) == q.size(2), "key_size and query_size must match"
            if k_set:
                keys.append(k)
            if v_set:
                values.append(v)
            if q_set:
                queries.append(q)

            self._queries_per_unit.append(q.size(1) if q_set else 0)

        return AttentionMessages(MultiUnitAttention._join_heads(keys),
                                 MultiUnitAttention._join_heads(values),
                                 MultiUnitAttention._join_heads(queries))

    def receive_attention(self, result: Tensor) -> List[Tensor]:
        """
        Compute new hidden state from input values. These are the results of attention computation.

        Args:
            result: float[batch_size, n_experts * in_heads, hidden_state_size]
        """
        tensors = [unit.receive_attention(result_part, self._unit_params())
                   for unit, result_part in zip(self.units, self._split_attention_by_units(result))]
        self._result_callback(tensors)
        return tensors

    @staticmethod
    def _join_heads(head_data: List[Tensor]) -> Tensor:
        return torch.cat(head_data, 1)

    def _split_attention_by_units(self, attention_result: Tensor) -> List[Tensor]:
        i = 0
        result = []
        assert len(self._queries_per_unit) == len(self.units), \
            'receive_attention must be called after generate_messages'
        for in_heads in self._queries_per_unit:
            result.append(attention_result[:, i:i + in_heads, :])
            i = i + in_heads
        return result
