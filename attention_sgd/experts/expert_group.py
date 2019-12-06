from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Any

import torch
from torch import Tensor, nn
from torch.nn import Parameter

from attention_sgd.agents.agent import DeviceAwareModule
from attention_sgd.models.attention.multi_unit_attention import AttentionUnit, MultiUnitAttention, AttentionMessages

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from attention_sgd.experts.units.expert_unit import ExpertUnit


@dataclass
class ExpertHiddenStateParams:
    hidden_state: Tensor
    expert_id: Tensor


class ExpertGroup(DeviceAwareModule, AttentionUnit[None, List[Tensor]]):
    attn: MultiUnitAttention[ExpertHiddenStateParams]
    hidden_state: Tensor  # float[batch_size, n_experts, hidden_state_size]
    expert_id: Tensor  # float[batch_size, n_experts, id_size]
    initial_hidden_state: Parameter  # float[1, hidden_state_size, 1]

    def __init__(self, hidden_state_size: int, key_size: int, id_size: int, device: Optional[str] = None):
        super().__init__(device)

        # Initial hidden state value
        # noinspection PyArgumentList
        self.initial_hidden_state = nn.Parameter(torch.randn(1, 1, hidden_state_size, device=self.device))

        self.hidden_state_size = hidden_state_size
        self.key_size = key_size
        self.id_size = id_size

        self.units = self.create_units()
        for i, unit in enumerate(self.units):
            if unit.module is not None:
                self.add_module(f'unit_{i}', unit.module)

        self.attn = MultiUnitAttention[ExpertHiddenStateParams](
            self.units,
            self.update_hidden_state,
            lambda: ExpertHiddenStateParams(self.hidden_state, self.expert_id)
        )

    def init_rollout(self, batch_size: int, n_experts: int, reset_static=True):
        super().init_rollout(batch_size, n_experts)
        self.attn.init_rollout(batch_size, n_experts)
        if reset_static:
            self.expert_id = torch.randn(1, n_experts, self.id_size, device=self.device).expand(batch_size, n_experts,
                                                                                                self.id_size)
        self.hidden_state = self.initial_hidden_state.expand(batch_size, n_experts, self.hidden_state_size)

    @abstractmethod
    def update_hidden_state(self, tensors: List[Tensor]):
        pass

    @abstractmethod
    def create_units(self) -> List['ExpertUnit']:
        pass

    def generate_messages(self, params: Any) -> AttentionMessages:
        return self.attn.generate_messages()

    def receive_attention(self, result: Tensor, params: None) -> List[Tensor]:
        return self.attn.receive_attention(result)
