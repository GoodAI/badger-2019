import dataclasses
import itertools
import os
from dataclasses import dataclass, field
from functools import partial, reduce
from typing import List, Dict, Optional, Any

import torch
from badger_utils.sacred import SacredReader, SacredUtils
from badger_utils.sacred.sacred_config import SacredConfigFactory
from bokeh.layouts import column, row
from bokeh.models import Button, TextInput, Select, Div, ColumnDataSource, HoverTool
from bokeh.plotting import curdoc, Figure
from torch import Tensor
from bokeh.palettes import Dark2_5 as palette
from bokeh.events import DoubleTap
import pandas as pd
from badger_utils.view import TensorViewer, TensorDataListDict, TensorDataMultiObserver
from badger_utils.view.observer_utils import MultiObserver
from attention.results_viewer.sacred_runs_config_analyzer import SacredRunsConfigAnalyzer
from attention.results_viewer.tensor_2d_plot import Tensor2DPlot
from attention.results_viewer.tensor_2d_plot_trace import Tensor2DPlotTrace
from attention.search_experiment import run_inference, create_agent, Params


@dataclass
class AppState:
    experiment_id: int = 0
    sacred_reader: SacredReader = None
    epoch: int = 0
    rollout_step: int = 0
    epochs: List[int] = field(default_factory=list)
    tensors_data: Optional[TensorDataListDict] = None
    inference_config: Dict[str, Any] = field(default_factory=dict)


class App:

    def __init__(self):
        self._sacred_config = SacredConfigFactory.local()
        self._sacred_utils = SacredUtils(self._sacred_config)
        self.state = AppState
        self.tensor_2d_plot = Tensor2DPlot()
        self.tensor_2d_plot_trace = Tensor2DPlotTrace()
        self.config_analyzer = SacredRunsConfigAnalyzer(self._sacred_config.create_mongo_observer())

    def _run_inference(self) -> MultiObserver:
        try:
            n_inputs = int(self.widget_text_n_inputs.value)
            n_experts = int(self.widget_text_n_experts.value)
            n_rollouts = int(self.widget_text_n_rollouts.value)

            observer = MultiObserver()
            params = Params(**self.state.sacred_reader.config)
            params.n_experts = n_experts
            params.task_size = n_inputs
            self.state.inference_config = dataclasses.asdict(params)
            agent = create_agent(params)
            self.state.sacred_reader.load_model(agent, 'agent', self.state.epoch)

            rollout_size = n_rollouts  # 15 + params.rollout_size
            run_inference(params, agent, observer, rollout_size)
            return observer

        except ValueError as e:
            print(f'ValueError: {e}')

    def update_experiment(self):
        print('Update experiment')
        try:
            # parse experiment id and load experiment from sacred
            self.widget_config_div.text = 'loading'
            self.state.experiment_id = int(self.widget_text_experiment_id.value)
            sr = SacredReader(self.state.experiment_id, self._sacred_config)
            self.state.sacred_reader = sr

            # update epochs select
            epochs = sr.get_epochs()
            self.state.epochs = epochs
            self.widget_button.label = f'Experiment Id: {self.state.experiment_id}'
            self.widget_epoch_select.options = list(map(str, epochs))
            self.widget_epoch_select.value = str(epochs[-1])

            # update config
            formatted_config = '<br/>'.join([f'{k}: {v}' for k, v in sr.config.items()])
            self.widget_config_div.text = f'<pre>{formatted_config}</pre>'

            # update inference params
            params = Params(**sr.config)
            self.widget_text_n_rollouts.value = str(params.rollout_size)
            self.widget_text_n_experts.value = str(params.n_experts)
            self.widget_text_n_inputs.value = str(params.task_size)

            self.update_loss_plot()
            self._update()
        except Exception as e:
            print(f'Error: {e}')

    def update_loss_plot(self):
        try:
            loss_average_window = int(self.widget_loss_smooth_text.value)
            # update epochs figure
            self.widget_loss_pane.children.clear()
            self.widget_loss_pane.children.append(
                self._create_loss_figure(
                    self._sacred_utils.load_metrics([self.state.experiment_id], loss_average_window)))
        except ValueError as e:
            print(f'Error: {e}')

    def update_epoch(self, attr, old, new):
        try:
            self.state.epoch = int(self.widget_epoch_select.value)
            print(f'Update epoch: {self.state.epoch}')
            self._update()
        except ValueError as e:
            print(f'Error: {e}')

    def _update(self):
        sr = self.state.sacred_reader
        use_training_inference_data = False
        try:
            self.widget_pane.children.clear()
            if use_training_inference_data:
                # load training data
                tensors: List[Dict[str, Tensor]] = sr.load_tensor('tensors.pt', self.state.epoch)
                self.state.tensors_data = TensorDataListDict(tensors)
            else:
                observer = self._run_inference()
                self.state.tensors_data = TensorDataMultiObserver(observer)
            viewer = TensorViewer(self.state.tensors_data, 6, displayed_tensors=['keys_2', 'weights_2', 'attn-result_2',
                                                                                 'weights_before_softmax_2'],
                                  rollout_on_change=self.rollout_step_on_change)
            self.widget_pane.children.append(viewer.create_layout())

            joined_data = torch.stack(
                [self.state.tensors_data.tensor_by_name(rollout_step, 'keys_2') for rollout_step in
                 range(self.state.tensors_data.step_count)])
            self.tensor_2d_plot_trace.update(joined_data, self.state.inference_config)

            self.widget_loss_pane_inference.children.clear()
            inf_loss_data = torch.stack(
                [self.state.tensors_data.tensor_by_name(rollout_step, 'error_per_inf_step') for rollout_step in
                 range(self.state.tensors_data.step_count)])
            self.widget_loss_pane_inference.children.append(
                self._create_loss_figure(pd.DataFrame(inf_loss_data.tolist()), width=500))

            # Plot grads
            # param_tensors = {k: v for k, v in tensors_data.tensor_map.items() if v.startswith('param')}
            # for tensor_id, tensor_name in param_tensors.items():
            #     self.widget_pane.children.append(Div(text=f'T: {tensor_name}'))
            #     fig = plot_tensor(tensors_data.tensor(2, tensor_id))
            #     self.widget_pane.children.append(fig)

        except Exception as e:
            print(f'ERR, {e}')

    def _create_loss_figure(self, df: pd.DataFrame, width: int = 1000, height: int = 300):
        colors = itertools.cycle(palette)
        df.columns = [str(i) for i in df.columns]
        ds = ColumnDataSource(df)

        fig = Figure(y_axis_type="log", width=width, height=height)
        fig.below[0].formatter.use_scientific = False
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

    def on_last_run(self):
        id = self._sacred_utils.get_last_run().id
        self.widget_text_experiment_id.value = str(id)

    def rollout_step_on_change(self, rollout_step: int):
        self.state.rollout_step = rollout_step
        tensor = self.state.tensors_data.tensor_by_name(self.state.rollout_step, 'keys_2')
        self.tensor_2d_plot.update(tensor, self.state.inference_config)
        # self._2d_plot_source.data = self._create_2d_plot_data()

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
        self.widget_loss_pane_inference = column()
        self.widget_loss_smooth_text = TextInput(value='100')
        self.widget_loss_smooth_text.on_change('value', lambda a, o, n: self.update_experiment())
        self.widget_config_div = Div(text="")
        self.widget_2d_plot = self.tensor_2d_plot.create_2d_plot()
        self.widget_2d_plot_trace = self.tensor_2d_plot_trace.create_2d_plot()

        self.widget_run_inference_button = Button(label='Run inference')
        self.widget_run_inference_button.on_click(self._update)
        self.widget_text_n_inputs = TextInput(title='n_inputs', value='5')
        self.widget_text_n_inputs.on_change('value', lambda a, o, n: self._update())
        self.widget_text_n_experts = TextInput(title='n_experts', value='2')
        self.widget_text_n_experts.on_change('value', lambda a, o, n: self._update())
        self.widget_text_n_rollouts = TextInput(title='n_rollouts', value='15')
        self.widget_text_n_rollouts.on_change('value', lambda a, o, n: self._update())

        self.widget_inference_pane = column(
            self.widget_text_n_inputs,
            self.widget_text_n_experts,
            self.widget_text_n_rollouts,
            self.widget_run_inference_button
        )

        curdoc().add_root(
            row(
                column(
                    row(
                        column(
                            row(Div(text="Experiment ID:"), self.widget_text_experiment_id, self.widget_button,
                                self.widget_button_last_run),
                            column(
                                self.widget_loss_pane,
                                row(Div(text='Smooth window:'), self.widget_loss_smooth_text)
                            )
                        ),
                        self.widget_config_div,
                    ),
                    row(
                        column(
                            self.widget_epoch_select,
                            self.widget_inference_pane
                        ),
                        self.widget_2d_plot,
                        self.widget_2d_plot_trace,
                        self.widget_loss_pane_inference
                    ),
                    self.widget_pane
                ),
                self.config_analyzer.create_layout()
            )
        )


if __name__ == '__main__':
    cmd = f'bokeh serve --show {__file__}'
    print(f'Running bokeh server: {cmd}')
    os.system(cmd)
else:
    App().run()
