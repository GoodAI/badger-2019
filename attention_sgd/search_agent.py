import dataclasses
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, TypeVar

import torch
from torch import Tensor
from torch import nn as nn

from attention_sgd.agents.agent import Agent, DeviceAwareModule, DeviceAware
from attention_sgd.experts.expert_group import ExpertGroup
from attention_sgd.models.attention.attention import Attention, normalize_tensor, AttentionOperation

from attention_sgd.models.attention.multi_unit_attention import MultiUnitAttention, AttentionUnit, AttentionMessages
from attention_sgd.models.attention.units.external_module_unit import OutputOnlyFilterAttentionUnit
from attention_sgd.utils.func_utils import apply
from attention_sgd.utils.observer_utils import Observer, MultiObserver
from attention_sgd.utils.torch_utils import id_to_one_hot, add_modules, with_prefix, LambdaModule, expand_to_batch_size
from attention_sgd.search_expert_group import SearchExpertGroup


@dataclass
class SearchAgentParams:
    input: Tensor
    error: Tensor


SearchAttentionUnit = AttentionUnit[SearchAgentParams, Tensor]


@dataclass
class SearchAgentUnits:
    inputs: SearchAttentionUnit
    experts: SearchExpertGroup


class InputsAndOutputUnit(AttentionUnit[SearchAgentParams, Tensor], DeviceAware):
    key_size: int
    value_size: int
    n_inputs: int
    _keys: Tensor = None
    _values: Tensor = None

    def __init__(self, n_inputs: int, key_size: int, value_size: int, device: Optional[str] = None):
        super().__init__(device)
        self.n_inputs = n_inputs
        self.key_size = key_size
        self.value_size = value_size

    def init_rollout(self, batch_size: int, n_experts: int):
        super().init_rollout(batch_size, n_experts)
        if self._values is None:
            self._generate_keys_and_values(batch_size)

    def _generate_keys_and_values(self, batch_size: int):
        values = torch.tensor([1.0] + [0.0] * (self.value_size - 1), device=self.device)
        values = values[None, None].expand(batch_size, self.n_inputs, self.value_size)  # unsqueeze and expand
        self._values = values

        keys = torch.rand(self.n_inputs, self.key_size, device=self.device) * 2 - 1
        self._keys = expand_to_batch_size(keys.unsqueeze(0), batch_size)

    def generate_messages(self, params: SearchAgentParams) -> AttentionMessages:
        return AttentionMessages(self._keys, self._values, self._keys)

    def receive_attention(self, result: Tensor, params: SearchAgentParams) -> Tensor:
        return result


class SearchAgent(Agent):
    _attention_beta: float
    units: SearchAgentUnits
    expert_prefix: Tensor
    output_prefix: Tensor
    error_prefix: Tensor
    input_prefix: Tensor

    last_output: Tensor
    last_input: Tensor
    last_error: Tensor

    def __init__(self, hidden_state_size: int, key_size: int, id_size: int, value_size: int, input_size: int,
                 n_inputs: int, learning_rate: float, attention_beta: float, attention_operation: AttentionOperation,
                 device: Optional[str] = None):
        super().__init__(device)
        self.hidden_state_size = hidden_state_size
        self.key_size = key_size
        self.id_size = id_size
        self.value_size = value_size
        self.input_size = input_size
        self.n_inputs = n_inputs
        self.attn = Attention(key_size, value_size, attention_operation)
        self._attention_beta = attention_beta

        self.units = self.create_units()
        self._attention_pass_1 = self.create_attention_pass_1(
            [
                self.units.inputs,
                self.units.experts
            ])
        self._attention_pass_2 = self.create_attention_pass_2(
            [
                self.units.inputs,
                OutputOnlyFilterAttentionUnit(self.units.experts)
            ])

        add_modules(self, [getattr(self.units, f.name) for f in dataclasses.fields(self.units)])

        self.optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # self.optim = torch.optim.SGD(self.parameters(), lr=1e-3)

    def init_rollout(self, batch_size: int, n_experts: int):
        self._attention_pass_1.init_rollout(batch_size, n_experts)
        self._attention_pass_2.init_rollout(batch_size, n_experts)

    def create_attention_params(self):
        return SearchAgentParams(self.last_input, self.last_error.unsqueeze(-1))

    def create_attention_pass_1(self, units: List[SearchAttentionUnit]) -> MultiUnitAttention[SearchAgentParams]:
        return MultiUnitAttention[SearchAgentParams](units, lambda _: None, self.create_attention_params)

    def create_attention_pass_2(self, units: List[SearchAttentionUnit]) -> MultiUnitAttention[SearchAgentParams]:
        def process_attention_result(tensors: List[Tensor]):
            self.last_output = tensors[0].squeeze(-1)

        return MultiUnitAttention[SearchAgentParams](units, process_attention_result, self.create_attention_params)

    def create_units(self) -> SearchAgentUnits:
        inputs = InputsAndOutputUnit(self.n_inputs, self.key_size, self.value_size, self.device)
        experts = SearchExpertGroup(self.hidden_state_size, self.key_size, self.id_size, self.value_size,
                                    device=self.device)
        return SearchAgentUnits(inputs, experts)

    def _create_prefix(self, prefix_size: int, one_hot_position: int, one_hot_value: float,
                       fill_value: float) -> Tensor:
        prefix = torch.tensor([fill_value] * prefix_size, device=self.device)
        prefix[one_hot_position] = one_hot_value
        return prefix

    def forward(self, inp: Tensor, err, drop=False, observer: Optional[MultiObserver] = None):
        """

        Args:
            inp: float[batch_size, n_input, input_size]
            err: float[batch_size, 1]
            drop:

        Returns:

        """
        self.experts: ExpertGroup
        batch_size = inp.size(0)
        self.last_input = inp

        self.last_error = err
        # pass 1
        messages = self._attention_pass_1.generate_messages()
        # messages = super().generate_messages(SearchAgentParams(inp))
        result, w, idx = self.attn.compute(messages.key, messages.value, messages.query, drop=drop,
                                           beta=self._attention_beta,
                                           normalize=False,
                                           observer=apply(observer, lambda o: o.current.with_suffix('_1')))
        self._attention_pass_1.receive_attention(result)
        self._update_observers('_1', observer, messages, result, w)

        # pass 2
        messages = self._attention_pass_2.generate_messages()
        # messages = super().generate_messages(SearchAgentParams(inp))
        result, w, idx = self.attn.compute(messages.key, messages.value, messages.query, drop=drop,
                                           beta=self._attention_beta,
                                           normalize=False,
                                           observer=apply(observer, lambda o: o.current.with_suffix('_2')))
        self._attention_pass_2.receive_attention(result)
        self._update_observers('_2', observer, messages, result, w)

        # super().receive_attention(result, None)

        return self.last_output, w, idx

    def _update_observers(self, name: str, observer: Optional[MultiObserver], messages: AttentionMessages,
                          result: Tensor,
                          w: Tensor):
        def with_name(text: str):
            return f'{text}{name}'

        if observer is not None:
            # observer.add_plot('err_key', plot_matrix(err_keys[0][0]))
            # observer.add_plot('expert_keys', plot_matrix(expert_keys[0][0]))
            # observer.add_plot('input_keys', plot_matrix(torch.cat(inp_keys, 1)[0]))
            batch = 0
            o = observer.current
            o.add_tensor(with_name('keys'), messages.key[batch])
            o.add_tensor(with_name('queries'), messages.query[batch])
            o.add_tensor(with_name('values'), messages.value[batch])
            o.add_tensor(with_name('experts-hid'), self.units.experts.hidden_state[batch])
            o.add_tensor(with_name('experts-id'), self.units.experts.expert_id[batch])
            o.add_tensor(with_name('weights'), w[batch])
            o.add_tensor(with_name('attn-result'), result[batch])
