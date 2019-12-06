# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
import sys
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from IPython import display
import time
import numpy as np
from badger_utils.sacred import SacredReader, SacredConfigFactory
from tqdm import tqdm
from dataclasses import dataclass
import dataclasses

from bokeh.layouts import column, row
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, Select, TextInput, Slider


output_notebook()

project_path = os.path.abspath(os.path.join('..', '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

from badger_utils.view.observer_utils import Observer, MultiObserver
from badger_utils.view import TensorViewer, TensorDataMultiObserver
from attention.learning_loop import LearningLoop
from attention.search_experiment import Params, load_agent, create_agent, create_task, run_inference
from attention.results_viewer.tensor_2d_plot import Tensor2DPlot
from attention.results_viewer.tensor_2d_plot_trace import Tensor2DPlotTrace


def pt(t, name):
    print(f'{name}: {t.shape}')
    print(t)


# %%
experiment_id, epoch = 1939, 300000 # 297400

sr = SacredReader(experiment_id, SacredConfigFactory.local())
p = Params(**sr.config)

# p.batch_size = 1
# p.task_size = 5
# p.n_experts = 5
p.exp_decay = 0.0
p.onehot_expert_ids = True
# p.rollout_size

with torch.no_grad():
    agent = create_agent(p)
    task = create_task(p)
    sr.load_model(agent, 'agent', epoch)
    inner_loop = LearningLoop()

    observer = MultiObserver()
    agent.init_rollout(p.batch_size, p.n_experts)
    err = inner_loop.train_fixed_steps(agent, task, p.rollout_size * 2, 0.0, p.learning_rollout_steps_clip, observer)
    # task.reset(True, True)
#     err = torch.mean(err)


print(f'Params: {p}')

# %%

data = TensorDataMultiObserver(observer)
config = dataclasses.asdict(p)
print(config['task_size'])
def app(doc):
    plot = Tensor2DPlot()
    plot_trace = Tensor2DPlotTrace()
    joined_data = torch.stack([data.tensor(rollout_step, data.tensor_name_to_id('keys_2')) for rollout_step in range(data.step_count)])
    plot_trace.update(joined_data, config)
    
    def rollout_on_change(rollout_step):
        plot.update(data.tensor(rollout_step, data.tensor_name_to_id('keys_2')), config)
    
    doc.add_root(
        column(
            row(plot.create_2d_plot(), plot_trace.create_2d_plot()),
            TensorViewer(TensorDataMultiObserver(observer), 4, rollout_on_change=rollout_on_change).create_layout()
        )
    )
    
show(app)


# %%
