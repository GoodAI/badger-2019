from typing import List, Optional

import torch
from torch import Tensor, nn as nn
from torch.nn import functional as F

from attention.agents.agent import DeviceAwareModule, DeviceAware
from attention.experts.expert_group import ExpertGroup, ExpertHiddenStateParams, ExpertInitRolloutParams
from attention.models.attention.multi_unit_attention import AttentionMessages
from attention.experts.units.expert_unit import ExpertUnit, ExternalModuleWriteHead, ExternalModuleReadHead

from attention.utils.torch_utils import WithPrefixModule, id_to_one_hot, tile_tensor, expand_to_batch_size
from functools import reduce


# https://stackoverflow.com/questions/50817916/how-do-i-add-lstm-gru-or-other-recurrent-layers-to-a-sequential-in-pytorch
class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]


class SearchExpertUnit(ExpertUnit, DeviceAware):
    _delta: Tensor

    def __init__(self, hidden_state_size: int, id_size: int, key_size: int, value_size: int, device: str,
                 model_name: str):
        super().__init__(device)
        self.hidden_state_size = hidden_state_size
        self.value_size = value_size
        self.key_size = key_size
        self._m = self._create_model(model_name, id_size, key_size)
        self._value_model = nn.Linear(self.hidden_state_size + id_size, value_size - 1)

        self._delta = self._create_delta()

    def _create_model(self, model_name: str, id_size: int, key_size: int):
        model_switcher = {
            '1': self._create_model_1,
            '2': self._create_model_2,
            '3': self._create_model_3,
        }
        return model_switcher[model_name](id_size, key_size)

    def _create_model_1(self, id_size: int, key_size: int) -> nn.Linear:
        return nn.Linear(self.hidden_state_size + id_size, key_size)

    def _create_model_2(self, id_size: int, key_size: int) -> nn.Sequential:
        hidden_size_first = self.hidden_state_size + id_size
        return nn.Sequential(nn.Linear(hidden_size_first, hidden_size_first),  # nn.BatchNorm1d(hidden_size_first),
                             nn.ReLU(), nn.Linear(hidden_size_first, key_size))

    def _create_model_3(self, id_size: int, key_size: int) -> nn.Sequential:
        hidden_size_first = self.hidden_state_size + id_size
        return nn.Sequential(nn.LSTM(hidden_size_first, hidden_size_first, 2), SelectItem(0),
                             nn.Linear(hidden_size_first, key_size))

    def _create_delta(self) -> Tensor:
        result = torch.zeros((self.n_queries, self.key_size), dtype=torch.float, device=self.device)
        for i in range(self.key_size):
            result[2 * i + 1, i] = -1
            result[2 * i + 2, i] = 1
        return result

    @property
    def module(self) -> Optional[nn.Module]:
        return nn.ModuleDict({'key': self._m, 'value': self._value_model})

    def generate_messages(self, params: ExpertHiddenStateParams) -> AttentionMessages:
        # [batch_size, n_keys, key_size]
        z = torch.cat([params.hidden_state, params.expert_id], 2)
        keys = self._m(z)
        # note expert_id size must be value_size - 1
        fixed_value = torch.empty((*params.expert_id.shape[:-1], 1), device=self.device).fill_(-1.0)
        values = torch.cat([fixed_value, self._value_model(z)], -1)
        # values = torch.cat([fixed_value, params.expert_id], -1)
        queries = self._enhance_queries(keys)
        assert values.shape[-1] == self.value_size
        return AttentionMessages(
            keys,
            values,
            queries
        )

    def receive_attention(self, result: Tensor, params: ExpertHiddenStateParams) -> Tensor:
        return result.view(self.batch_size, self.n_experts, -1)

    def _enhance_queries(self, query: Tensor) -> Tensor:
        result = query.unsqueeze(2).expand(-1, -1, self.n_queries, -1).contiguous().view(self.batch_size,
                                                                                self.n_queries * self.n_experts, -1)
        # result = query.expand(-1, self.key_size * 2 + 1, -1)
        coef = 1e-3
        return result + expand_to_batch_size(self._delta.repeat(self.n_experts, 1).unsqueeze(0) * coef, query.shape[0])

    @property
    def n_queries(self) -> int:
        return self.key_size * 2 + 1


class SearchExpertGroup(ExpertGroup):

    def __init__(self, hidden_state_size: int, key_size: int, id_size: int, value_size: int, onehot_ids: bool,
                 model_name: str, device: Optional[str] = None):
        self.value_size = value_size
        self.model_name = model_name
        super().__init__(hidden_state_size, key_size, id_size, device)
        self._onehot_ids = onehot_ids
        self.read_heads = 1

        # Computes new hidden state
        # result_size = (2 * key_size + 1) * value_size + id_size
        result_size = sum([u.n_queries * value_size for u in self.units]) + id_size

        self.update_hidden_state_value = nn.Conv1d(result_size, hidden_state_size, 1)
        # Computes weight the new hidden state is applied with
        self.update_hidden_state_gate = nn.Conv1d(result_size, hidden_state_size, 1)
        # self.id_transform = nn.Linear(self.id_size, self.value_size)

    def update_hidden_state(self, tensors: List[Tensor]):
        # [batch_size, n_inputs, value_size]
        # inputs = tensors[0]
        nonempty_tensors = reduce(lambda count, t: count + (1 if t.numel() > 0 else 0), tensors, 0)
        assert self.read_heads == nonempty_tensors, f'expected {self.read_heads} read heads, but received {nonempty_tensors} non-empty results'

        # z = torch.cat(tensors[0:self.read_heads] + [self.id_transform(self.expert_id)], dim=1)
        z = torch.cat(tensors[0:self.read_heads] + [self.expert_id], dim=2)
        # Update hidden state
        # Weight for new hidden state
        w = torch.sigmoid(self.update_hidden_state_gate(z.transpose(1, 2))).transpose(1, 2)
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
            SearchExpertUnit(self.hidden_state_size, self.id_size, self.key_size, self.value_size, self.device,
                             self.model_name)
        ]

    def init_rollout(self, batch_size: int, n_experts: int, params: ExpertInitRolloutParams):
        super().init_rollout(batch_size, n_experts, params)
        if self._onehot_ids and params.reset_expert_id:
            ids = tile_tensor(torch.eye(self.id_size, device=self.device), 0, n_experts)
            self.expert_id = expand_to_batch_size(ids.unsqueeze(0), self.batch_size)
