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

# %% [markdown]
# # Badger communication through attention
#
# Badger is an architecture for multi-agent learning with sharing policies between agents, as described in [BADGER: Learning to (Learn \[Learning Algorithms\] through Multi-Agent Communication)](https://arxiv.org/abs/1912.01513). The Badger architecture calls these agents "experts". 
#
# The core feature of Badger architecture is the *shared policy*: all experts share the same code and parameters. Through communication (and randomized starting initialization) the experts assume different roles in processing inputs and apply the shared policy towards solving the task at hand.
#
# The shared policy allows for faster training, similarly as filters in convolutional neural networks allow for a faster acquisition of features. There are at least two more benefits that we expect from the shared policy. The first is an easier development of communication between experts, since they know each other intimately, and the second is a good scalability.
#
# ## The "covering" task
# For a proof-of concept, we chose the task of covering a number of landmarks by experts, exemplified below.
#
# ![OpenAI particles env](https://openai.com/content/images/2017/06/spread_maddpg_notag.gif)
#
# Example of a 3-target, 3-expert task; image credits [OpenAI](https://openai.com/blog/learning-to-cooperate-compete-and-communicate/).
#
# There can be a variable number of landmarks and a variable number of experts. To enable fast iterations, we chose relatively small numbers of experts and landmarks (2-10) for training. 
#
# ## Communication and training
# There are different possibilities how to allow the experts to communicate with each other and also with the environment.  We provided them with neural attention interface - keys, queries and values, as described in the reference [Transformer paper](https://arxiv.org/abs/1706.03762). Each expert in our setting had a single read/write head that allowed it to publish a key, query and value and receive a value on input computed by attention. Additionally, we used "input" experts that published information about landmarks as values.
#
# The trained experts perform inference in a series of steps we refer to as "inner loop". Let's say the number of steps in the inner loop is *N*. We used supervised learning for adjusting the expert policy. During each cycle of training, the experts would operate for *N* steps and at the end, we would calculate loss for the experts as approximately the sum of distances between experts and their closest targets. The loss was subsequently used for optimizing the expert policy using Adam.
#
# We modified the interface attention uses for communication between experts by using euclidean distance instead of dot product as a similarity measure. We also reduced the dimensionality of the attention keys to only 2-D to allow interpretation of the attention keys as positions in 2-D space We verified using SGD that the loss landscape was still navigable even if there were 20 experts and 20 targets. (At the same time, we are aware that this modification could have been over-restrictive for attention, so our next batch of experiments will use higher-dimensional keys).
#
# The architecture of each expert was based on an LSTM cell paired with a 2-layer feed-forward network for transforming the hidden state vector into output keys, queries and values. Each expert had a single read/write attention head and its keys were restricted to 2-D. We enriched information obtained by each query the experts made by adding 4 more read heads, each outputting a slightly shifted original query to allow the expert to "sense" the value landscape.
#
# A parallel stream of our research tried communication and training through a variant of ATOC ([Learning Attentional Communication for Multi-Agent Cooperation](https://arxiv.org/abs/1805.07733)) using reinforcement learning. [](http://and_it_will_be_described_here)
#
# ## Experiments
# In the full setting, our experts were expected to solve the task in a few steps of inference performed in the inner loop. We verified the validity of our setup on an overfitted setting where we trained the same number of experts reach a number of targets on fixed positions. [The code in this notebook describes the setting](https://github.com/GoodAI/badger-2019/blob/master/attention_sgd/experiment.ipynb). When this setup was converging well, we made the next step and trained the more general setting with variable positions.
#
# We first tried a more restricted version, where the experts could communicate using very constrained values. The values were restricted to a fixed 1-D signal (+1 emitted by targets, -1 emitted by experts) and an identifier of the communicating expert. The results were only partially satisfactory, so we modified the communication to include a vector freely set by each expert and we excluded the identifiers. The performance improved, probably due to increased opportunity for communication. After roughly 100k training steps, the experts were able to reach the targets reliably when performing inference in the inner loop rollout. 
#
# ![inner rollout](https://raw.githubusercontent.com/GoodAI/badger-2019/master/attention/data/rollout.gif)
#
# Behaviour in inner loop after convergence in training.
#
# We tested the expert policy using a setting with more experts and targets. On a policy trained with 3 experts and 3 targets, we observed a modest generalization to the *k*-experts/*k*-targets setting for *k=4,5*. Modest here means that the experts could reach error levels seen on smaller scale training in 25-50% of cases for k=5 and 4, respectively. The policy did not scale well to significantly higher values of *k* (e.g. 20). 
#
#

# %% {"pycharm": {"is_executing": false}}
import random
import os
import sys
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from IPython import display
import time
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
import dataclasses

from bokeh.layouts import column, row
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, Select, TextInput, Slider


output_notebook()

project_path = os.path.abspath(os.path.join('..'))
if project_path not in sys.path:
    sys.path.append(project_path)

from attention.utils.observer_utils import Observer, MultiObserver
from attention.utils.tensor_viewer import TensorViewer, TensorDataMultiObserver
from attention.learning_loop import LearningLoop
from attention.search_experiment import Params, create_agent, create_task, run_inference
from attention.results_viewer.tensor_2d_plot import Tensor2DPlot
from attention.results_viewer.tensor_2d_plot_trace import Tensor2DPlotTrace

from attention.utils.torch_utils import default_device

from attention.utils.plot_utils import plot_forwarding_task_targets, plot_matrix
from attention.utils.bokeh_utils import plot_tensor, update_figure_by_tensor
from attention.search_agent import SearchAgent, SearchAgentInitRolloutParams
from attention.models.attention.attention import AttentionOperation
from attention.batched_task import BatchedTask
from attention.learning_loop import LearningLoop
from attention.search_agent import SearchAgent, SearchAgentInitRolloutParams
from attention.search_task import FillTask, FillTorchTask, ToZeroTorchTask, TorchTask



def pt(t, name):
    print(f'{name}: {t.shape}')
    print(t)


# %%
@dataclass
class Params:
    seed: int
    batch_size: int = 32
    hidden_state_size: int = 16
    id_size: int = 8  # must be value_size - 1
    key_size: int = 2
    value_size: int = 17
    onehot_expert_ids: bool = False

    epochs: int = 300000
    rollout_size: int = 8

    # ignore_error_start_steps: int = 5
    # training_reset_targets_period: int = -1
    learning_rate: float = 1e-3
    learning_exp_decay: float = 2.0
    learning_rollout_steps_clip: int = 3

    task_size: int = 5
    n_experts: int = 2

    save_period: int = 1000
    image_save_period: int = 10000
    attention_beta: float = 5
    attention_operation: str = 'euclidean_distance'
    description: str = 'randomized experts and task_size'

    model_name: str = '2'

        
def create_agent(p: Params) -> SearchAgent:
    return SearchAgent(hidden_state_size=p.hidden_state_size, key_size=p.key_size, id_size=p.id_size,
                       value_size=p.value_size, input_size=1, n_inputs=p.task_size,
                       onehot_expert_ids=p.onehot_expert_ids,
                       attention_beta=p.attention_beta,
                       attention_operation=AttentionOperation.from_string(p.attention_operation),
                       learning_rate=p.learning_rate, model_name=p.model_name).to(default_device())


def create_task(p: Params, min_weight: float = 1.0):
    return BatchedTask(p.batch_size, lambda: ToZeroTorchTask(p.task_size, min_weight))


def run_inference(p: Params, agent: SearchAgent, observer: MultiObserver, rollout_size: int) -> float:
    with torch.no_grad():
        task = create_task(p)
        inner_loop = LearningLoop()
        agent.init_rollout(p.batch_size, p.n_experts, SearchAgentInitRolloutParams(True))
        err = inner_loop.train_fixed_steps(agent, task, rollout_size, 0.0, p.learning_rollout_steps_clip, observer)
        return err.item()
    
def load_agent(p: Params, file: str):
    agent = create_agent(p)
    agent.load_state_dict(torch.load(file))
    return agent

p = Params(random.randint(0, 2 ** 32))

agent = create_agent(p)
task = create_task(p)
inner_loop = LearningLoop()

loss = []    


# %%
# for epoch in tqdm(range(1, p.epochs + 1)):
for epoch in range(1, p.epochs + 1):
    observer = None
    agent.optim.zero_grad()
    agent.init_rollout(p.batch_size, p.n_experts, SearchAgentInitRolloutParams(True))
    rollout_size = p.rollout_size
    err = inner_loop.train_fixed_steps(agent, task, rollout_size, p.learning_exp_decay,
                                       p.learning_rollout_steps_clip, None)
    err.backward()
    agent.optim.step()

    loss.append(err.cpu().detach().item())

    # log targets
    if epoch % 50 == 0:
        # plot loss
        plt.gcf().set_size_inches((12, 6))
        display.clear_output(wait=True)
        display.display(plt.gcf())
        time.sleep(0.01)
        
        # Average loss
        avg_window = 10 if epoch < 200 else 100
        cs = np.cumsum(loss)
        cs = (cs[avg_window:] - cs[:-avg_window]) / avg_window
        plt.clf()
        plt.plot(cs)
        plt.yscale('log')


# %%
# Run inference

should_load_agent = True
if should_load_agent:
    p = Params(**torch.load('data/agent.config'))

# Uncomment to change parameters to other values than the model was trained on
# p.task_size = 4
# p.n_experts = 4
# p.rollout_size = 10

if should_load_agent:
    agent = load_agent(p, 'data/agent.model')
observer = MultiObserver()
run_inference(p, agent, observer, p.rollout_size)


def app(doc):
    config = dataclasses.asdict(p)
    data = TensorDataMultiObserver(observer)
    plot = Tensor2DPlot()
    plot_trace = Tensor2DPlotTrace()
    joined_data = torch.stack([data.tensor(rollout_step, data.tensor_name_to_id('keys_2')) for rollout_step in range(data.step_count)])
    plot_trace.update(joined_data, config)
    
    def rollout_on_change(rollout_step):
        plot.update(data.tensor(rollout_step, data.tensor_name_to_id('keys_2')), config)
    
    doc.add_root(
        column(
            row(plot.create_2d_plot(), plot_trace.create_2d_plot()),
            TensorViewer(TensorDataMultiObserver(observer), 2, rollout_on_change=rollout_on_change, 
                         displayed_tensors=['keys_2', 'attn-result_2']).create_layout()
        )
    )
    
show(app)


# %% [markdown]
# ## Next Steps 
# We are now preparing experiments that will provide different and local information to each expert. Our plan is to allow each expert to see the ids and positions of its *k* nearest neighbors (experts and targets) to allow it to negotiate more efficiently on what its desired target is, to allow it to move quicker to it, and to allow for improved scalability. At the same time, we would also like to create an experimental setup with a less restricted attention (high-dimensional keys, multi-head attention) more similar to the original Transformer paper. Lastly, we're considering replacing supervised learning with a carefully designed loss function with reinforcement learning.
#
#

# %%
