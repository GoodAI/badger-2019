from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Dict

from matplotlib.figure import Figure
from torch import Tensor


@dataclass
class ObserverPlot:
    name: str
    figure: Figure


@dataclass
class ObserverScalar:
    name: str
    value: float


@dataclass
class ObserverTensor:
    name: str
    value: Tensor


class Observer:
    plots: List[ObserverPlot]
    scalars: List[ObserverScalar]
    tensors: List[ObserverTensor]

    def __init__(self):
        self.plots = []
        self.scalars = []
        self.tensors = []

    def add_plot(self, name: str, figure: Figure):
        self.plots.append(ObserverPlot(name, figure))

    def add_scalar(self, name: str, value: float):
        self.scalars.append(ObserverScalar(name, value))

    def add_tensor(self, name: str, tensor: Tensor):
        self.tensors.append(ObserverTensor(name, tensor))

    def tensors_as_dict(self) -> Dict[str, Tensor]:
        return {t.name: t.value for t in self.tensors}

    def with_suffix(self, suffix: str) -> 'Observer':
        return SuffixObserver(self, suffix)


class ObserverWrapper(Observer):
    _observer: Observer

    def __init__(self, observer: Observer):
        super().__init__()
        self._observer = observer

    def add_plot(self, name: str, figure: Figure):
        self._observer.add_plot(self._process_name(name), figure)

    def add_scalar(self, name: str, value: float):
        self._observer.add_scalar(self._process_name(name), value)

    def add_tensor(self, name: str, tensor: Tensor):
        self._observer.add_tensor(self._process_name(name), tensor)

    @abstractmethod
    def _process_name(self, name: str) -> str:
        pass


class SuffixObserver(ObserverWrapper):
    def __init__(self, observer: Observer, suffix: str):
        super().__init__(observer)
        self._suffix = suffix

    def _process_name(self, name: str) -> str:
        return name + self._suffix


class MultiObserver:
    observers: List[Observer]

    def __init__(self):
        self.observers = []
        self.add_observer()

    @property
    def current(self) -> Observer:
        return self.observers[-1]

    def add_observer(self) -> Observer:
        observer = Observer()
        self.observers.append(observer)
        return observer
