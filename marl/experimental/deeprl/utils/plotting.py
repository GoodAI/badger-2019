import math
import os
from typing import List, Optional, Union

import pyformulas as pf
import matplotlib.pyplot as plt
import numpy as np
import time


class RunningPlot:
    """ Non-blocking matplotlib plotting.

    Inspired here:
    https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib
    """

    colors = ['blue', 'green', 'red']
    folder_name = 'data'

    def __init__(self, plot_name: Optional[str] = 'loss'):
        self.fig = plt.figure()
        canvas = np.zeros((480,640))
        self.screen = pf.screen(canvas, plot_name)

    @property
    def xlabel(self):
        return ''

    @property
    def ylabel(self):
        return ''

    @property
    def title(self):
        return ''

    def _get_color(self, series_id: int) -> str:
        # colors not accurate and different in the live plotting and saved fig.
        if series_id == 0:
            return self.colors[0]
        return self.colors[series_id % len(self.colors)]

    def plot(self,
             values: Union[List[float], List[List[float]]],
             legend: Optional[Union[str, List[str]]] = None):
        """Draws one or multiple lines in a non-blocking way"""

        if not isinstance(values[0], List):
            values = [values]

        if legend is not None and not isinstance(legend, List):
            legend = [legend]

        if legend is None:
            legend = [f'id {id}' for id in range(len(values))]

        plt.clf()
        for s_id, series in enumerate(values):
            plt.plot(series, color=self._get_color(s_id), label=legend[s_id])

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)

        plt.legend(loc='upper left')

        # If we haven't already shown or saved the plot, then we need to draw the figure first...
        self.fig.canvas.draw()

        image = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        self.screen.update(image)

    def save(self):
        timestr = time.strftime("%Y-%m-%d--%H-%M-%S")
        if not os.path.exists(self.folder_name):
            os.mkdir(self.folder_name)
        name = os.path.join(self.folder_name, f'Loss_{timestr}.svg')
        plt.savefig(name, format='svg')

    def close(self):
        self.save()
        self.screen.close()


class LossPlot(RunningPlot):

    loss_values: List[float]

    @property
    def xlabel(self):
        return 'Learning calls'

    @property
    def ylabel(self):
        return 'Loss'

    @property
    def title(self):
        return 'Loss during training'

    def __init__(self):
        super().__init__('Training loss')
        self.loss_values = []

    def add_value(self, value: float):
        self.loss_values.append(value)
        self.plot(self.loss_values)

        max_val = max(self.loss_values)
        print(f'max_val is {max(self.loss_values)}')

        # top: probably another bug in plotting
        plt.ylim(bottom=0, top=max_val + max_val / 10)


class LossRewardPlot(RunningPlot):

    loss_values: List[float]
    reward_values: List[float]

    @property
    def xlabel(self):
        return 'Learning calls'

    @property
    def ylabel(self):
        return 'Loss & reward'

    @property
    def title(self):
        return 'Loss and reward during training'

    def __init__(self):
        super().__init__('Training loss & reward')
        self.loss_values = []
        self.reward_values = []

    def add_value(self, loss: float, reward: float):
        self.loss_values.append(loss)
        self.reward_values.append(reward)

        self.plot([self.loss_values, self.reward_values],
                  ['loss', 'avg reward'])

        max_val = max(self.loss_values)
        max_val_r = max(self.reward_values)
        max_val = max(max_val, max_val_r)

        # top: probably another bug in plotting
        plt.ylim(bottom=0, top=max_val + max_val / 10)
