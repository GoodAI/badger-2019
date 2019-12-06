from typing import List, Optional

import torch
from torch import Tensor, nn as nn
from torch.nn import functional as F

from attention_sgd.agents.agent import DeviceAwareModule, DeviceAware
from attention_sgd.experts.expert_group import ExpertGroup, ExpertHiddenStateParams
from attention_sgd.models.attention.multi_unit_attention import AttentionMessages
from attention_sgd.experts.units.expert_unit import ExpertUnit, ExternalModuleWriteHead, ExternalModuleReadHead

from attention_sgd.utils.torch_utils import WithPrefixModule, id_to_one_hot, tile_tensor, expand_to_batch_size
from functools import reduce


class SearchExpertUnit(ExpertUnit, DeviceAware):
    def __init__(self, hidden_state_size: int, id_size: int, key_size: int, value_size: int, device: str):
        super().__init__(device)
        self.hidden_state_size = hidden_state_size
        self.value_size = value_size
        self._m = nn.Linear(self.hidden_state_size + id_size, key_size)

    @property
    def module(self) -> Optional[nn.Module]:
        return self._m

    def generate_messages(self, params: ExpertHiddenStateParams) -> AttentionMessages:
        k = self._m(torch.cat([params.hidden_state, params.expert_id], 2))
        k_expanded = expand_to_batch_size(k, self.batch_size)
        # note expert_id size must be value_size - 1
        fixed_value = torch.empty((*params.expert_id.shape[:-1], 1), device=self.device).fill_(-1.0)
        values = torch.cat([fixed_value, params.expert_id], -1)
        assert values.shape[-1] == self.value_size
        return AttentionMessages(
            k_expanded,
            values,
            k_expanded
        )

    def receive_attention(self, result: Tensor, params: ExpertHiddenStateParams) -> Tensor:
        return result


class SearchExpertGroup(ExpertGroup):

    def __init__(self, hidden_state_size: int, key_size: int, id_size: int, value_size: int, device: Optional[str] = None):
        self.value_size = value_size
        super().__init__(hidden_state_size, key_size, id_size, device)
        self.read_heads = 1

        # Computes new hidden state
        self.update_hidden_state_value = nn.Conv1d(self.value_size * self.read_heads + id_size, hidden_state_size, 1)
        # Computes weight the new hidden state is applied with
        self.update_hidden_state_gate = nn.Conv1d(self.value_size * self.read_heads + id_size, hidden_state_size, 1)

    def update_hidden_state(self, tensors: List[Tensor]):
        # [batch_size, n_inputs, value_size]
        # inputs = tensors[0]
        nonempty_tensors = reduce(lambda count, t: count + (1 if t.numel() > 0 else 0), tensors, 0)
        assert self.read_heads == nonempty_tensors, f'expected {self.read_heads} read heads, but received {nonempty_tensors} non-empty results'

        z = torch.cat(tensors[0:self.read_heads] + [self.expert_id], dim=2)
        # Update hidden state
        # Weight for new hidden state
        w = F.sigmoid(self.update_hidden_state_gate(z.transpose(1, 2))).transpose(1, 2)
        new_hid = self.update_hidden_state_value(z.transpose(1, 2)).transpose(1, 2)
        try:
            self.hidden_state = self._checked_tensor(new_hid * w + self.hidden_state * (1 - w))
        except ValueError as e:
            print(f'Hidden state exploded (and was reset to random values):')
            # print(f'z={z}')
            # print(f'w={w}')
            # print(f'new_hid={new_hid}')
            self.hidden_state = torch.rand_like(self.hidden_state)

    @staticmethod
    def _checked_tensor(tensor: Tensor) -> Tensor:
        if torch.sum(~torch.isfinite(tensor)) > 0:
            raise ValueError('Tensor contains inf or nan value')
        return tensor

    def create_units(self) -> List[ExpertUnit]:
        return [
            SearchExpertUnit(self.hidden_state_size, self.id_size, self.key_size, self.value_size, self.device)
        ]

    def init_rollout(self, batch_size: int, n_experts: int, reset_static=True):
        super().init_rollout(batch_size, n_experts, reset_static)
        ids = tile_tensor(torch.eye(self.id_size, device=self.device), 0, n_experts)
        self.expert_id = expand_to_batch_size(ids.unsqueeze(0), self.batch_size)
