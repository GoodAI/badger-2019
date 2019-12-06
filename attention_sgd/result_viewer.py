import itertools
import os
from dataclasses import dataclass, field
from functools import partial, reduce
from typing import List, Dict

from badger_utils.sacred import SacredReader
from badger_utils.sacred.sacred_config import SacredConfig, SacredConfigFactory
from bokeh.layouts import column, row, layout
from bokeh.models import Button, Text, TextInput, Select, Div, ColumnDataSource, HoverTool
from bokeh.plotting import figure, curdoc, Figure
from torch import Tensor
from bokeh.palettes import Dark2_5 as palette
from bokeh.events import ButtonClick, DoubleTap
import pandas as pd
from attention_sgd.utils.bokeh.tensor_viewer import TensorViewer, TensorDataListDict
from attention_sgd.utils.bokeh_utils import plot_tensor, sanitize_tensor
from attention_sgd.utils.sacred_utils import SacredUtils


@dataclass
class AppState:
    experiment_id: int = 0
    sacred_reader: SacredReader = None
    epoch: int = 0
    epochs: List[int] = field(default_factory=list)


class App:

    def __init__(self):
        self._sacred_config = SacredConfigFactory.local()
        self._sacred_utils = SacredUtils(self._sacred_config)
        self.state = AppState

    def update_experiment(self):
        print('Update experiment')
        try:
            self.widget_config_div.text = 'loading'
            self.state.experiment_id = int(self.widget_text_experiment_id.value)
            sr = SacredReader(self.state.experiment_id, self._sacred_config)
            self.state.sacred_reader = sr
            # print(f'Experiment: {self.state}')
            epochs = sr.get_epochs()
            self.state.epochs = epochs
            # print(f'Epochs: {epochs}')
            self.widget_button.label = f'Experiment Id: {self.state.experiment_id}'
            self.widget_epoch_select.options = list(map(str, epochs))
            self.widget_epoch_select.value = str(epochs[-1])

            formatted_config = '<br/>'.join([f'{k}: {v}' for k, v in sr.config.items()])
            self.widget_config_div.text = f'<pre>{formatted_config}</pre>'
            # update epochs figure
            self.widget_loss_pane.children.clear()
            self.widget_loss_pane.children.append(
                self._create_loss_figure(self._sacred_utils.load_metrics([self.state.experiment_id])))

            self._update()
        except ValueError as e:
            print(f'Error: {e}')

        # TensorViewer(observer)

    def update_epoch(self, attr, old, new):
        try:
            self.state.epoch = int(self.widget_epoch_select.value)
            print(f'Update epoch: {self.state.epoch}')
            self._update()
        except ValueError as e:
            pass

    def _update(self):
        sr = self.state.sacred_reader
        tensors: List[Dict[str, Tensor]] = sr.load_tensor('tensors.pt', self.state.epoch)
        self.widget_pane.children.clear()
        try:
            tensors_data = TensorDataListDict(tensors)
            viewer = TensorViewer(tensors_data, 6, displayed_tensors=['keys_2', 'weights_2', 'attn-result_2',
                                                                      'weights_before_softmax_2'])
            self.widget_pane.children.append(viewer.create_layout())

            # Plot grads
            # param_tensors = {k: v for k, v in tensors_data.tensor_map.items() if v.startswith('param')}
            # for tensor_id, tensor_name in param_tensors.items():
            #     self.widget_pane.children.append(Div(text=f'T: {tensor_name}'))
            #     fig = plot_tensor(tensors_data.tensor(2, tensor_id))
            #     self.widget_pane.children.append(fig)

        except Exception as e:
            print(f'ERR, {e}')

    def _create_loss_figure(self, df: pd.DataFrame):
        colors = itertools.cycle(palette)
        df.columns = [str(i) for i in df.columns]
        ds = ColumnDataSource(df)
        fig = Figure(y_axis_type="log", width=1000, height=300)
        for column, color in zip(df.columns, colors):
            glyph = fig.line(x='index', y=column, source=ds, color=color)
            fig.add_tools(
                HoverTool(tooltips=[("epoch", "@index")] + [(f"loss_{column}", f"@{column}")],
                          mode='vline', renderers=[glyph]))

        def update_epoch(self: App, event):
            epoch = reduce(lambda a, b: a if a[0] < b[0] else b, [(abs(e - event.x), e) for e in self.state.epochs])[1]
            self.widget_epoch_select.value = str(epoch)

        fig.on_event(DoubleTap, partial(update_epoch, self))
        return fig

    # def update_experiment_id(self, attr, old, new):
    #     self.update_experiment()

    def on_last_run(self):
        id = self._sacred_utils.get_last_run().id
        self.widget_text_experiment_id.value = str(id)

    def run(self):

        self.widget_text_experiment_id = TextInput(value='170')
        self.widget_text_experiment_id.on_change('value', lambda a, o, n: self.update_experiment())
        self.widget_button_last_run = Button(label="Last run")
        self.widget_button_last_run.on_click(self.on_last_run)
        self.widget_text_experiment_id.on_change('value', lambda a, o, n: self.update_experiment())
        self.widget_epoch_select = Select(title='epoch', options=[])
        self.widget_button = Button(label="Read experiment")
        self.widget_button.on_click(self.update_experiment)
        self.widget_epoch_select.on_change('value', self.update_epoch)
        self.widget_pane = column()
        self.widget_loss_pane = column()
        self.widget_config_div = Div(text="")
        curdoc().add_root(
            column(
                row(
                    column(
                        row(Div(text="Experiment ID:"), self.widget_text_experiment_id, self.widget_button,
                            self.widget_button_last_run),
                        self.widget_loss_pane
                    ),
                    self.widget_config_div
                ),
                self.widget_epoch_select, self.widget_pane))


if __name__ == '__main__':
    cmd = f'bokeh serve --show {__file__}'
    print(f'Running bokeh server: {cmd}')
    os.system(cmd)
else:
    App().run()
