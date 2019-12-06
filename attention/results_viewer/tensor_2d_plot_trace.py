from typing import Optional, Dict, Any

import pandas as pd
from bokeh import colors
from bokeh.models import ColumnDataSource
from bokeh.plotting import Figure
from torch import Tensor


class Tensor2DPlotTrace:
    data_source: ColumnDataSource

    def __init__(self):
        self.data_source = ColumnDataSource(data=self._create_2d_plot_data(None, {}))

    def _create_2d_plot_data(self, data: Optional[Tensor], config:[Dict[str, Any]]) -> pd.DataFrame:
        if data is None:
            return pd.DataFrame([], columns=['x', 'y', 'xn', 'yn', 'color'])

        task_size, original_rollout_size = config['task_size'], config['rollout_size']
        t = data[:, :, 0:2] # convert to 2D
        rollout_size, point_count = t.shape[:2]

        def data_color(rollout_idx:int, point_idx:int, task_size:int):
            rollout_alpha = ((original_rollout_size - rollout_idx) / original_rollout_size) * 0.5
            index_color = colors.RGB(255, 0, 0)
            point_color = colors.RGB(0, 0, 255).lighten(rollout_alpha) if rollout_idx < original_rollout_size else colors.RGB(255,200,255)
            return index_color if point_idx < task_size else point_color

        def data_row(t, rollout_idx, point_idx, task_size):
            next_rollout_idx = rollout_idx if rollout_idx == 0 else rollout_idx - 1
            return [
                t[rollout_idx, point_idx, 0].item(),
                t[rollout_idx, point_idx, 1].item(),
                t[next_rollout_idx, point_idx, 0].item(),
                t[next_rollout_idx, point_idx, 1].item(),
                data_color(rollout_idx, point_idx, task_size)
            ]

        return pd.DataFrame(
            [data_row(t, rollout_idx, point_idx,task_size) for point_idx in range(point_count) for rollout_idx in range(rollout_size)],
            columns=['x', 'y', 'xn', 'yn', 'color'])

    def update(self, data: Optional[Tensor], config: Dict[str, Any]):
        self.data_source.data = self._create_2d_plot_data(data, config)

    def create_2d_plot(self) -> Figure:
        fig = Figure(width=300, height=300, x_range=(-2, 2), y_range=(-2, 2))
        fig.circle(x='x', y='y', color='color', source=self.data_source)
        fig.segment(x0='x', y0='y', x1='xn', y1='yn', color='color', line_dash='dotted', source=self.data_source)
        return fig