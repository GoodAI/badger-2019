from typing import Optional, Dict, Any

import pandas as pd
from bokeh.models import ColumnDataSource
from bokeh.plotting import Figure
from torch import Tensor


class Tensor2DPlot:
    data_source: ColumnDataSource

    def _create_2d_plot_data(self, task_size: int, data: Optional[Tensor]) -> pd.DataFrame:
        if data is None:
            return pd.DataFrame([], columns=['x', 'y', 'color'])

        t = data[:, 0:2]
        return pd.DataFrame(
            [[t[i, 0].item(), t[i, 1].item(), 'red' if i < task_size else 'blue'] for i in range(t.shape[0])],
            columns=['x', 'y', 'color'])

    def update(self, data: Tensor, config: Dict[str, Any]):
        task_size = config['task_size']
        self.data_source.data = self._create_2d_plot_data(task_size, data)

    def create_2d_plot(self) -> Figure:
        self.data_source = ColumnDataSource(data=self._create_2d_plot_data(None, None))
        fig = Figure(width=300, height=300, x_range=(-2, 2), y_range=(-2, 2))
        fig.circle(x='x', y='y', color='color', source=self.data_source)
        return fig