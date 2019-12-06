import torch
from torch import Tensor
import numpy as np
from badger_utils.sacred import SacredReader, SacredConfigFactory
from bokeh.plotting import figure, show, output_notebook

from badger_utils.view.observer_utils import Observer, MultiObserver
from attention.learning_loop import LearningLoop
from attention.search_experiment import Params, load_agent, create_agent, create_task

experiment_id, epoch = 1923, 100000

sr = SacredReader(experiment_id, SacredConfigFactory.local())
p = Params(**sr.config)

p.batch_size = 1
p.task_size = 5
p.n_experts = 1

with torch.no_grad():
    agent = create_agent(p)
    task = create_task(p)
    sr.load_model(agent, 'agent', epoch)
    inner_loop = LearningLoop()

    observer = MultiObserver()
    agent.init_rollout(p.batch_size, p.n_experts)
    err = inner_loop.train_fixed_steps(agent, task, p.rollout_size, observer)
    # task.reset(True, True)
    err = torch.mean(err)
