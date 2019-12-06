import math
from typing import Union, List, Callable

import torch
import torch.nn as nn
from torch import Tensor, nn as nn

from attention.models.with_module import WithModule


def default_device() -> str:
    # return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def norm(x):
    mu = x.mean(1, keepdims=True)
    std = x.std(1, keepdims=True)

    return (x - mu) / (std + 1e-8)


def expand_to_batch_size(tensor: Tensor, batch_size: int):
    return tensor.expand([batch_size] + list(tensor.size())[1:])


def id_to_one_hot(data: torch.Tensor, vector_len: int):
    """Converts ID to one-hot representation.
    Each element in `data` is converted into a one-hot-representation - a vector of
    length vector_len having all zeros and one 1 on the position of value of the element.
    Args:
        data: ID of a class, it must hold for each ID that 0 <= ID < vector_len
        vector_len: length of the output tensor for each one-hot-representation
        dtype: data type of the output tensor
    Returns:
        Tensor of size [data.shape[0], vector_len] with one-hot encoding.
        For example, it converts the integer cluster indices of size [flock_size, batch_size] into
        one hot representation [flock_size, batch_size, n_cluster_centers].
    """
    device = data.device
    data_a = data.view(-1, 1)
    n_samples = data_a.shape[0]
    output = torch.zeros(n_samples, vector_len, device=device)
    output.scatter_(1, data_a, 1)
    output_dims = data.size() + (vector_len,)
    return output.view(output_dims)


def add_modules(target_module: nn.Module, module_list: List[Union[nn.Module, WithModule]], prefix: str = 'unit_'):
    """
    Adds modules to target modules. WithModule type is supported
    Args:
        target_module: Module to be injected
        module_list: List of modules to be added
        prefix: Prefix to module names
    """
    for i, unit in enumerate(module_list):
        name = f'{prefix}{i}'
        if isinstance(unit, nn.Module):
            target_module.add_module(name, unit)
        elif 'module' in dir(unit) and unit.module is not None:
            target_module.add_module(name, unit.module)


class LambdaModule(nn.Module):
    def __init__(self, func: Callable[[Tensor], Tensor]):
        super().__init__()
        self._func = func

    def forward(self, tensor: Tensor) -> Tensor:
        return self._func(tensor)


def with_prefix(prefix: Tensor, data: Tensor) -> Tensor:
    """
    Adds prefix to last dimension of data
    Args:
        prefix: float[prefix_size]
        data: float[*]

    Returns:

    """
    data_size = data.size()
    prefix_size = prefix.size(-1)
    p = prefix.view([1] * (len(data_size) - 1) + [prefix_size])
    p = p.expand(data_size[:-1] + (prefix_size,))
    return torch.cat([p, data], dim=-1)


class WithPrefixModule(nn.Module):
    def __init__(self, prefix: Tensor):
        super().__init__()
        self._prefix = prefix

    def forward(self, x: Tensor) -> Tensor:
        return with_prefix(self._prefix, x)


def tile_tensor(tensor: Tensor, dim: int, size: int) -> Tensor:
    """
    Tile tensor in dimension dim
    Args:
        tensor: arbitrary tensor
        dim: dimension in which to tile
        size: desired size in tiled dimension

    Returns:

    """
    tensor_size = tensor.size(dim)
    assert tensor_size > 0
    if tensor_size < size:
        repeat_shape = [1] * tensor.dim()
        repeat_shape[dim] = math.ceil(size / tensor_size)
        tensor = tensor.repeat(repeat_shape)
    return tensor.narrow(dim, 0, size)
