from abc import abstractmethod
from typing import Tuple, List, Dict, Optional

import torch
from bokeh.model import Model
from torch import Tensor
from bokeh.layouts import column, row, gridplot
from bokeh.models import Select, Slider

from attention_sgd.utils.bokeh_utils import plot_tensor, update_figure_by_tensor, sanitize_tensor
from attention_sgd.utils.observer_utils import MultiObserver
from itertools import *


class TensorData:
    @property
    @abstractmethod
    def tensor_names(self) -> List[Tuple[int, str]]:
        pass

    @abstractmethod
    def tensor(self, step: int, tensor_id: int) -> Tensor:
        pass

    @property
    @abstractmethod
    def step_count(self) -> int:
        pass

    def tensor_name_to_id(self, name: str) -> int:
        return {name: id for id, name in self.tensor_names}[name]


class TensorDataListDict(TensorData):

    def __init__(self, data: List[Dict[str, Tensor]]):
        self._data = data
        self._tensor_map = {i: name for i, name in enumerate(self._data[-1].keys())}

    @property
    def tensor_map(self) -> Dict[int, str]:
        return self._tensor_map

    @property
    def tensor_names(self) -> List[Tuple[int, str]]:
        return [(i, name) for i, name in self._tensor_map.items()]

    @abstractmethod
    def tensor(self, step: int, tensor_id: int) -> Tensor:
        return sanitize_tensor(self._data[step][self._tensor_map[tensor_id]])

    @property
    def step_count(self) -> int:
        return len(self._data)


class TensorDataMultiObserver(TensorData):
    _observer: MultiObserver

    def __init__(self, observer: MultiObserver):
        self._observer = observer

    @property
    def tensor_names(self) -> List[Tuple[int, str]]:
        return [(i, o.name) for i, o in enumerate(self._observer.observers[0].tensors)]

    def tensor(self, step: int, tensor_id: int) -> Tensor:
        return self._observer.observers[step].tensors[tensor_id].value

    @property
    def step_count(self) -> int:
        # Just hack to ignore last empty observer
        return len(self._observer.observers) - 1


class TensorViewer:
    class PlotView:
        tensor_id: int = 0

        def __init__(self, parent: 'TensorViewer', default_tensor_id: int = 0):
            self.parent = parent
            self.tensor_id = default_tensor_id
            self.plot = plot_tensor(self.parent._tensor(self.tensor_id))

            self.tensor_select = Select(title="tensor: ", value=f'{self.tensor_id}', options=self.parent.tensors)

            def tensor_select_updated(attrname, old, new):
                self.tensor_id = int(self.tensor_select.value)
                self.update()

            self.tensor_select.on_change('value', tensor_select_updated)

        def update(self):
            update_figure_by_tensor(self.parent._tensor(self.tensor_id), self.plot)

        @property
        def view(self):
            return column(self.tensor_select, self.plot)

    rollout_step: int = 0

    def __init__(self, tensor_data: TensorData, plot_count: int = 3, displayed_tensors: Optional[List[str]] = None):
        self.tensors = tensor_data.tensor_names
        self.tensor_data = tensor_data
        self.rollout_step = self.n_rollouts - 1
        tensor_ids = list(map(tensor_data.tensor_name_to_id, [] if displayed_tensors is None else displayed_tensors))
        self.plots = [self.PlotView(self, tensor_id) for tensor_id in
                      chain(tensor_ids, [0] * (plot_count - len(tensor_ids)))]

    def _tensor(self, tensor_id: int) -> torch.Tensor:
        return self.tensor_data.tensor(self.rollout_step, tensor_id)

    @property
    def n_rollouts(self) -> int:
        return self.tensor_data.step_count

    def update(self):
        for p in self.plots:
            p.update()

    def create_figure(self, doc):
        doc.add_root(self.create_layout())
        self.update()

    def create_layout(self) -> Model:
        use_slider = self.n_rollouts > 1

        def update_rollout_slider(attrname, old, new):
            self.rollout_step = int(rollout_slider.value)
            self.update()

        if use_slider:
            rollout_slider = Slider(title="rollout step: ", value=self.n_rollouts - 1, start=0, end=self.n_rollouts - 1,
                                    step=1)
            rollout_slider.on_change('value', update_rollout_slider)

        # layout = column(row(rollout_slider), row(*[p.view for p in self.plots]))
        slider_row = row(rollout_slider) if use_slider else row()
        return column(slider_row, gridplot([p.view for p in self.plots], ncols=3))
