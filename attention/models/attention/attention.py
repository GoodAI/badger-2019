from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from math import *

from attention.utils.cache import SingleCachedTensor
from attention.utils.func_utils import apply
from attention.utils.torch_utils import norm
from badger_utils.view.observer_utils import MultiObserver, Observer


def normalize_tensor(t: Tensor) -> Tensor:
    return t / t.abs().sum(dim=-1).unsqueeze(-1)


class AttentionOperation(Enum):
    DOT_PRODUCT = 1,
    NORMALIZED_DOT_PRODUCT = 2,
    EUCLIDEAN_DISTANCE = 3

    @staticmethod
    def from_string(value: str) -> 'AttentionOperation':
        if value == 'dot_product':
            return AttentionOperation.DOT_PRODUCT
        elif value == 'normalized_dot_product':
            return AttentionOperation.NORMALIZED_DOT_PRODUCT
        elif value == 'euclidean_distance':
            return AttentionOperation.EUCLIDEAN_DISTANCE
        else:
            raise ValueError(f'Unrecognized value "{value}"')

class Attention:
    key_size: int
    value_size: int

    def __init__(self, key_size: int, value_size: int, operation: AttentionOperation = AttentionOperation.DOT_PRODUCT):
        super().__init__()

        self.value_size = value_size
        self.key_size = key_size
        self.operation = operation

    def compute(self, key: Tensor, value: Tensor, query: Tensor, drop: bool = False, beta: float = 1,
                normalize: bool = False, topk: Optional[int] = None, observer: Optional[Observer] = None):
        """
        Compute attention, note values have transposed dims 1 and 2 compared with AttentionLayer
        Args:
            key: [ batch_size, n_inputs, key_size]
            value: [ batch_size, n_inputs, value_size]
            query: [ batch_size, n_outputs, key_size]
            drop:
            beta: weight multiplier
            normalize: normalize values over batch x n_outputs

        Returns:
        values float[batch_size, n_outputs, value_size]
        weights float[bach_size, n_inputs, n_outputs]
        idx dummy
        """
        batch_size = key.size(0)
        n_outputs = query.size(1)

        # weights: [batch_size, n_inputs, n_outputs]
        if self.operation == AttentionOperation.DOT_PRODUCT:
            key = key.unsqueeze(2)
            query = query.unsqueeze(1)
            weights = self._dot_product(key, query) / sqrt(self.key_size)
        elif self.operation == AttentionOperation.NORMALIZED_DOT_PRODUCT:
            key = normalize_tensor(key).unsqueeze(2)
            query = normalize_tensor(query).unsqueeze(1)
            weights = self._dot_product(key, query) / sqrt(self.key_size)
        elif self.operation == AttentionOperation.EUCLIDEAN_DISTANCE:
            key = key.unsqueeze(2)
            query = query.unsqueeze(1)
            clamp_range = 1e4
            eps = 1e-4
            distance = self._euclidean_distance_squared(key, query) + eps
            # (distance.log_()).exp_()
            # distance = (eps + distance).pow(0.9)
            weights = (distance.pow(0.5) + 1).reciprocal().clamp(-clamp_range, clamp_range)
            # weights = (self._euclidean_distance(key, query).pow(2) +1).reciprocal().clamp(-clamp_range, clamp_range)

            apply(observer, lambda o: o.add_tensor('weights_before_softmax', weights[0]))

        else:
            raise ValueError(f'Unrecognized operation: {self.operation}')

        if drop:
            mask = torch.le(torch.rand(batch_size, 1, n_outputs).cuda(), 0.25).float()
            weights = weights - 40 * mask

        if topk is not None:
            topk_weights, idx = torch.topk(weights, topk, dim=1)
            weights.zero_()
            topk_weights = F.softmax(topk_weights * beta, dim=1)
            weights.scatter_(1, idx, topk_weights)
        else:
            weights = F.softmax(weights * beta, dim=1)
            # weights = normalize_tensor(weights)

        values = torch.sum(weights.unsqueeze(3) * value.unsqueeze(2), 1).view(batch_size * n_outputs, self.value_size)
        if normalize:
            values = norm(values)
        values = values.view(batch_size, n_outputs, self.value_size)  # .transpose(1, 2).contiguous()

        return values, weights, 0

    def _dot_product(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.sum(a * b, 3)

    def _euclidean_distance_squared(self, k: Tensor, q: Tensor) -> Tensor:
        # return (self._dot_product(k, k) - 2 * self._dot_product(k, q) + self._dot_product(q, q)).sqrt()
        return self._dot_product(k, k) - 2 * self._dot_product(k, q) + self._dot_product(q, q)

    def _euclidean_distance(self, k: Tensor, q: Tensor) -> Tensor:
        return (self._dot_product(k, k) - 2 * self._dot_product(k, q) + self._dot_product(q, q)).sqrt()
        # return self._dot_product(k, k) - 2 * self._dot_product(k, q) + self._dot_product(q, q)


# Inputs are Memory, Attender -> Batch size, Features, Number of sources/Number of receivers
class AttentionLayer(nn.Module):
    def __init__(self, key_size, value_size):
        super().__init__()

        self.value_size = value_size
        self.key_size = key_size

    def forward(self, key: Tensor, query: Tensor, value: Tensor, drop: bool = False, beta: float = 1):
        """

        Args:
            key: [ batch_size, n_inputs, 1, key_size]
            query: [ batch_size, 1, n_outputs, key_size]
            value: [ batch_size, n_inputs, value_size]
            drop:
            beta: weight multiplier

        Returns:
        values float[batch_size, value_size, n_outputs]
        weights float[bach_size, n_inputs, n_outputs]
        idx dummy
        """
        batch_size = key.size(0)
        n_outputs = query.size(2)
        # weights: [batch_size, n_inputs, n_outputs]
        weights = torch.sum(key * query, 3) / sqrt(self.key_size)
        if drop:
            mask = torch.le(torch.rand(batch_size, 1, n_outputs).cuda(), 0.25).float()
            weights = weights - 40 * mask
        weights = F.softmax(weights * beta, dim=1)

        values = torch.sum(weights.unsqueeze(3) * value.unsqueeze(2), 1).view(batch_size * n_outputs, self.value_size)
        values = norm(values)
        values = values.view(batch_size, n_outputs, self.value_size).transpose(1, 2).contiguous()

        return values, weights, 0


# This layer only uses the top k weights for attention
class NarrowAttentionLayer(nn.Module):

    def __init__(self, N1, N2, NK, NV, topK=3):
        super(NarrowAttentionLayer, self).__init__()
        self.device = 'cpu'
        self.N1 = N1
        self.N2 = N2
        self.NV = NV
        self.NK = NK
        self.topK = topK
        # precompute tensors
        self._cached_index = SingleCachedTensor(lambda size: torch.arange(size).to(self.device))
        self._cached_drop_rand = SingleCachedTensor(lambda sizes: torch.FloatTensor(size=sizes).to(self.device))

    def forward(self, key, query, value, drop=False):
        """

        :param key: [ batch_size, n_inputs, 1, key_size]
        :param query: [ batch_size, 1, n_outputs, key_size]
        :param value: [ batch_size, n_inputs, value_size]
        :param drop:
        :return:
        values float[batch_size, value_size, n_outputs]
        weights float[bach_size, top_k, n_outputs]
        idx int[bach_size, top_k, n_outputs]
        """
        BS = key.size(0)
        NA = key.size(2)  # number of outputs
        NB = query.size(2)
        NV = self.NV

        # print(f'k: {key.shape}, q: {query.shape}, v: {value.shape}')
        weights = torch.sum(key * query, 3) / sqrt(self.NK)

        if drop:
            # mask = torch.le(torch.rand(BS,NA,NB).cuda(),0.25).float()
            rand = self._cached_drop_rand.tensor([BS, NA, NB])
            rand.uniform_()
            mask = torch.le(rand, 0.25).float()
            weights = weights - 40 * mask

        best_w, best_idx = torch.topk(weights, self.topK, 1)

        std = torch.std(best_w, 1, keepdims=True) + 1e-8
        std = 1 / (1 / std + 1)
        best_w = (best_w - torch.mean(best_w, 1, keepdims=True)) / std
        best_w = F.softmax(best_w, dim=1)

        # best_w and best_idx are BS x topK x nExperts
        # value is BS x nExperts x Features

        values = []

        # idx = torch.arange(BS).cuda()
        idx = self._cached_index.tensor(BS)
        for i in range(NB):
            row = 0
            for j in range(self.topK):
                k = best_idx[:, j, i]
                row = row + best_w[:, j, i].unsqueeze(1) * value[idx, k[idx], :]
            values.append(row.unsqueeze(1))

        values = torch.cat(values, 1).view(BS * NB, NV)
        values = norm(values)
        values = values.view(BS, NB, NV).transpose(1, 2).contiguous()

        return values, best_w, best_idx

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.device = args[0]  # hack to extract device
        return self
