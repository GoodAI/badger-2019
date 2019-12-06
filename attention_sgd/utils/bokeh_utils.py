import math
from typing import Dict, Optional

import bokeh
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from torch import Tensor
import numpy as np
import pandas as pd


def bokeh_reg_green_cmap(n=256):
    middle = n / 2 - 0.5
    result = []
    for i in range(n):
        a = math.floor(max(i - middle, 0) / middle * 255)
        b = math.floor(max(middle - i, 0) / middle * 255)
        result.append(f'#{b:02x}{a:02x}00')
    return result


def create_tensor_source(t: Tensor) -> Dict:
    height, width = t.size()
    xs = np.tile(np.arange(width), height)
    ys = np.arange(height - 1, -1, -1).repeat(width)
    values = t.contiguous().view(-1).detach().cpu().numpy()
    return {'x': xs + 0.5, 'y': ys + 0.5, 'row': np.flip(ys), 'column': xs, 'val': values,
            'text': [f'{i:0.1f}' for i in values],
            'text_color': ['#ffffff' if i < 0.7 else '#000000' for i in values]}


def update_figure_by_tensor(t: Tensor, fig: bokeh.plotting.Figure, update_data: bool = True):
    """
    Update Figure size, axes and datasource to display passed tensor
    Args:
        t: Tensor to be shown (2D)
        fig: Figure to be updated
        update_data: When True, datasource is also updated
    """

    def to_str(a):
        return list(map(str, a))

    height, width = t.size()
    margin_x = 20  # + 30
    margin_y = 30
    cell_size = 25
    fig_width = width * cell_size + margin_x
    fig_height = height * cell_size + margin_y

    fig.x_range.factors = to_str(range(width))
    fig.y_range.factors = to_str(reversed(range(height)))
    fig.width = fig_width
    fig.height = fig_height

    if update_data:
        data = create_tensor_source(t)
        fig.select_one({'name': 'data_rect'}).data_source.data = data


def plot_tensor(t: Tensor, src: Optional[ColumnDataSource] = None) -> bokeh.plotting.Figure:
    if src is None:
        src = ColumnDataSource(data=create_tensor_source(t))

    tooltips = [
        ("row", "@row"),
        ("column", "@column"),
        ('value', '@val')
    ]
    fig = figure(tools="", toolbar_location=None, tooltips=tooltips, x_range=[''], y_range=[''])
    fig.rect('x', 'y', name='data_rect', width=1, height=1, source=src,
             fill_color=linear_cmap('val', bokeh_reg_green_cmap(), -1, 1))
    fig.text('x', 'y', name='data_text', text='text', source=src, text_color='text_color', text_font_size='8pt',
             x_offset=-8, y_offset=6)
    update_figure_by_tensor(t, fig, False)
    return fig


# def plot_tensor(t: Tensor) -> bokeh.plotting.Figure:
#     return plot_tensor_from_datasource(t, ColumnDataSource(data=create_tensor_source(t)))

def sanitize_tensor(t: Tensor):
    if t.dim() < 2:
        return t.unsqueeze(0)
    elif t.dim() > 2:
        return t.view(-1, t.size(-1))
    return t


def plot_dataframe(df: pd.DataFrame):
    fig = figure(y_axis_type="log")
    fig.line(df.index, df.iloc[:, 1])
    return fig
