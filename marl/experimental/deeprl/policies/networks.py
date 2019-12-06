from abc import abstractmethod, ABC
from typing import Union, Tuple, Optional, List

import torch
from torch import nn

from marl.experimental.deeprl.utils.available_device import my_device


class NetworkBase(nn.Module, ABC):
    hidden_sizes: List[int]
    hidden_layers: nn.ModuleList

    output_size: int
    output_activation: nn.Module
    output_rescale: Union[float, None]

    softmaxed_parts: Optional[Tuple[List[int], List[bool]]]
    temperature: float

    def __init__(self,
                 output_size: int,
                 output_activation: Optional[str],
                 output_rescale: Optional[float],
                 hidden_sizes: Union[int, None, List[int]] = 20,
                 softmaxed_parts: Optional[Tuple[List[int], List[bool]]] = None,
                 temperature: float = 1.0):
        """

        Args:
            output_size:
            output_activation:
            output_rescale:
            hidden_sizes:
            softmaxed_parts: Gumbel-softmax parts of the output (if not set, the standard output activation is used)
            temperature: Gumbel-softmax temperature (positive value, can be above 0)
        """

        super(NetworkBase, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = []
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        assert output_size > 0

        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.output_rescale = output_rescale
        self.output_activation = self.get_activation(output_activation)

        assert temperature > 0
        self.temperature = temperature

        self.softmaxed_parts = softmaxed_parts
        if self.softmaxed_parts is not None:
            assert sum(self.softmaxed_parts[0]) == self.output_size, \
                'incompatible partitioning of the output'

    @staticmethod
    def get_activation(activation_type: Union[None, str]) -> nn.Module:
        if activation_type is None:
            return nn.Sequential()
        if activation_type == 'leaky_relu':
            return nn.LeakyReLU()
        if activation_type == 'relu':
            return nn.ReLU()
        if activation_type == 'sigmoid':
            return nn.Sigmoid()
        if activation_type == 'tanh':
            return nn.Tanh()

        raise Exception(f'Unsupported type of activation function: {activation_type}')

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def reset(self, batch_size: int = 1):
        pass

    def _apply_output_activation(self, data: torch.Tensor) -> torch.Tensor:
        output = self.output_activation(data)
        if self.output_rescale is not None:
            output = output * self.output_rescale
        return output

    def apply_output_activation(self, data: torch.Tensor) -> torch.Tensor:
        if self.softmaxed_parts is None:
            return self._apply_output_activation(data)
        else:
            chunks = self.softmaxed_parts[0]
            softmaxed = self.softmaxed_parts[1]

            results = []
            output_chunks = torch.split(data, chunks, dim=-1)
            for chunk, soft in zip(output_chunks, softmaxed):
                if soft:
                    res = torch.nn.functional.gumbel_softmax(chunk, tau=self.temperature, hard=False)
                    results.append(res)
                else:
                    res = self._apply_output_activation(chunk)
                    results.append(res)

            output = torch.cat(results, dim=-1)

        return output


class SimpleFF(NetworkBase):
    """
    FF network with LeakyRelu in hidden layers, no activation on the output layer.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_sizes: Union[None, int, List[int]] = 20,
                 output_activation: Optional[str] = None,
                 output_rescale: Optional[float] = None,
                 softmaxed_parts: Optional[Tuple[List[int], List[bool]]] = None,
                 temperature: float = 1.0):

        super(SimpleFF, self).__init__(output_size,
                                       output_activation,
                                       output_rescale,
                                       hidden_sizes,
                                       softmaxed_parts,
                                       temperature)

        hidden_layers = []
        previous_size = input_size

        for layer_size in self.hidden_sizes:
            layer = nn.Linear(previous_size, layer_size)
            hidden_layers.append(layer)
            previous_size = layer_size

        self.hidden_layers = nn.ModuleList(hidden_layers)

        self.output_layer = nn.Linear(previous_size, output_size)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        for layer_id in range(len(self.hidden_sizes)):
            x = self.hidden_layers[layer_id](x)
            x = self.relu(x)

        output = self.output_layer(x)
        output = self.apply_output_activation(output)
        return output

    def reset(self, batch_size: Optional[int] = 1):
        pass


class SimpleLSTM(NetworkBase):
    """
    LSTM network with LeakyRelu activations on LSTM layers, no activation on the linear output.
    """

    zero_hidden_state: bool
    hidden: List[Tuple[torch.tensor, torch.tensor]]  # state
    bidirectional: bool
    disable_output_layer: bool

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_sizes: Union[None, int, List[int]] = 20,
                 zero_hidden_state: Optional[bool] = True,
                 output_activation: Optional[str] = None,
                 output_rescale: Optional[float] = None,
                 softmaxed_parts: Optional[Tuple[List[int], List[bool]]] = None,
                 temperature: float = 1.0,
                 bidirectional: Optional[bool] = False,
                 disable_output_layer: Optional[bool] = False):
        super(SimpleLSTM, self).__init__(output_size,
                                         output_activation,
                                         output_rescale,
                                         hidden_sizes,
                                         softmaxed_parts,
                                         temperature)

        self.zero_hidden_state = zero_hidden_state
        self.bidirectional = bidirectional
        self.disable_output_layer = disable_output_layer
        if self.disable_output_layer:
            assert self.hidden_sizes[-1] == output_size, \
                f'if the output layer is disabled, last hidden has to have correct size'

        hidden_layers = []
        previous_size = input_size

        for layer_size in self.hidden_sizes:
            layer = nn.LSTM(previous_size, layer_size, batch_first=True, bidirectional=self.bidirectional)
            hidden_layers.append(layer)
            previous_size = layer_size

        self.hidden_layers = nn.ModuleList(hidden_layers)

        if not self.disable_output_layer:
            self.output_layer = nn.Linear(previous_size, output_size)

        self.reset()

    @property
    def num_directions(self) -> int:
        return 2 if self.bidirectional else 1

    def forward(self, x):
        assert len(x.shape) == 3, '3D input expected'

        for layer_id in range(len(self.hidden_sizes)):
            x, self.hidden[layer_id] = self.hidden_layers[layer_id](x, self.hidden[layer_id])

        if self.disable_output_layer:
            return x

        output = self.output_layer(x)
        output = self.apply_output_activation(output)
        return output

    def get_hidden(self, batch_size: int) -> List[Tuple[torch.tensor, torch.tensor]]:
        return [self._get_one_hidden(hidden_size, batch_size) for hidden_size in self.hidden_sizes]

    def _get_one_hidden(self, size: int, batch_size: int):
        layers = 1  # num layers, for now it is done sequentially.. (list of layers)
        if self.zero_hidden_state:
            return (
                torch.zeros(layers * self.num_directions, batch_size, size, device=my_device()),
                torch.zeros(layers * self.num_directions, batch_size, size, device=my_device())
            )
        return (
            torch.randn(layers * self.num_directions, batch_size, size, device=my_device()),
            torch.randn(layers * self.num_directions, batch_size, size, device=my_device())
        )

    def reset(self, batch_size: Optional[int] = 1):
        """ Reset the network, this means create new hidden state.

        Batch_size has to be different for inference (1) and training (self.batch_size).
        """
        self.hidden = self.get_hidden(batch_size)


class BidirectionalLSTM(NetworkBase):
    """
    LSTM network with LeakyRelu activations on LSTM layers, no activation on the linear output.
    """

    zero_hidden_state: bool
    hidden: Tuple[torch.tensor, torch.tensor]  # state
    disable_output_layer: bool

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: int = 20,
                 num_layers: int = 1,
                 zero_hidden_state: Optional[bool] = True,
                 output_activation: Optional[str] = None,
                 output_rescale: Optional[float] = None,
                 softmaxed_parts: Optional[Tuple[List[int], List[bool]]] = None,
                 temperature: float = 1.0,
                 disable_output_layer: Optional[bool] = False):
        super().__init__(output_size,
                         output_activation,
                         output_rescale,
                         hidden_size,
                         softmaxed_parts,
                         temperature)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        assert self.hidden_size == input_size, "Hidden size for the comm should be the same as the input size, add more layers if you need more power"
        self.zero_hidden_state = zero_hidden_state
        self.disable_output_layer = disable_output_layer
        if self.disable_output_layer:
            assert self.hidden_sizes[-1] == output_size, \
                f'if the output layer is disabled, last hidden has to have correct size'

        self.hidden_layer = nn.LSTM(input_size, self.hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

        if not self.disable_output_layer:
            self.output_layer = nn.Linear(self.hidden_size * 2, output_size)

        self.reset()

    @property
    def num_directions(self) -> int:
        return 2

    def forward(self, x):
        assert len(x.shape) == 3, '3D input expected'

        x, self.hidden = self.hidden_layer(x, self.hidden)

        forward_res = x[:, :, :self.hidden_size]
        reverse_res = x[:, :, self.hidden_size:]

        if self.disable_output_layer:
            # Sum them together?
            return forward_res + reverse_res

        output = self.output_layer(torch.cat((forward_res, reverse_res), dim=-1))
        output = self.apply_output_activation(output)
        return output

    def get_hidden(self, batch_size: int):
        if self.zero_hidden_state:
            return (
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=my_device()),
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=my_device())
            )
        return (
            torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=my_device()),
            torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=my_device())
        )

    def reset(self, batch_size: Optional[int] = 1):
        """ Reset the network, this means create new hidden state.

        Batch_size has to be different for inference (1) and training (self.batch_size).
        """
        self.hidden = self.get_hidden(batch_size)