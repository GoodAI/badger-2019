import dataclasses
import random
from dataclasses import dataclass

import torch
import numpy as np
from badger_utils.sacred import SacredWriter, SacredConfigFactory, SacredReader
from sacred.run import Run
from tqdm import tqdm

from attention.models.attention.attention import AttentionOperation
from badger_utils.view.observer_utils import Observer, MultiObserver
from attention.utils.torch_utils import default_device

from sacred import Experiment

from attention.batched_task import BatchedTask
from attention.learning_loop import LearningLoop
from attention.search_agent import SearchAgent, SearchAgentInitRolloutParams
from attention.search_task import FillTask, FillTorchTask, ToZeroTorchTask, TorchTask

ex = Experiment("Search task")
sacred_writer = SacredWriter(ex, SacredConfigFactory.local())


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


ex.add_config(dataclasses.asdict(Params(random.randint(0, 2 ** 32))))


def load_agent(experiment_id, epoch, agent):
    reader = SacredReader(experiment_id, SacredConfigFactory.local())
    reader.load_model(agent, 'agent', epoch)


def create_agent(p: Params) -> SearchAgent:
    return SearchAgent(hidden_state_size=p.hidden_state_size, key_size=p.key_size, id_size=p.id_size,
                       value_size=p.value_size, input_size=1, n_inputs=p.task_size,
                       onehot_expert_ids=p.onehot_expert_ids,
                       attention_beta=p.attention_beta,
                       attention_operation=AttentionOperation.from_string(p.attention_operation),
                       learning_rate=p.learning_rate, model_name=p.model_name).to(default_device())


def create_task(p: Params, min_weight: float = 1.0) -> TorchTask:
    return BatchedTask(p.batch_size, lambda: ToZeroTorchTask(p.task_size, min_weight))


def run_inference(p: Params, agent: SearchAgent, observer: MultiObserver, rollout_size: int) -> float:
    with torch.no_grad():
        task = create_task(p)
        inner_loop = LearningLoop()
        agent.init_rollout(p.batch_size, p.n_experts, SearchAgentInitRolloutParams(True))
        err = inner_loop.train_fixed_steps(agent, task, rollout_size, 0.0, p.learning_rollout_steps_clip, observer)
        return err.item()


@ex.automain
def main(_run: Run, _config):
    p = Params(**_config)

    agent = create_agent(p)
    # load_agent(761, 200, agent)
    task = create_task(p)
    inner_loop = LearningLoop()

    n_experts = p.n_experts

    for epoch in tqdm(range(1, p.epochs + 1)):
        # with torch.autograd.detect_anomaly():
        should_save_images = epoch % p.image_save_period == 0
        observer = MultiObserver() if should_save_images else None

        # if epoch == 10000:
        #     task = create_task(p, 0.5)
        #
        # if epoch == 20000:
        #     task = create_task(p, 0.01)

        # if epoch % 100 == 0:
        #     task_size = random.randint(n_experts, n_experts + 10)
        #     p.task_size = task_size
        #     task = create_task(p)

        # reset_ids = epoch > 10000 or (epoch - 1) % 2000 == 0
        reset_ids = True
        agent.optim.zero_grad()
        agent.init_rollout(p.batch_size, n_experts, SearchAgentInitRolloutParams(reset_ids))

        rollout_size = p.rollout_size
        # if epoch > 10000:
        #     rollout_size = rollout_size + random.randint(-5, 5)

        err = inner_loop.train_fixed_steps(agent, task, rollout_size, p.learning_exp_decay,
                                           p.learning_rollout_steps_clip, observer)
        err.backward()

        _run.log_scalar('loss', err.cpu().detach().item())

        # save grads
        # if observer is not None:
        #     params = [p for p in agent.parameters() if p.grad is not None]
        #     for i, param in enumerate(params):
        #         observer.current.add_tensor(f'params_{i}', param.detach().cpu())
        #         observer.current.add_tensor(f'params-grads_{i}', param.grad.detach().cpu())

        agent.optim.step()

        if observer is not None:
            # print(f'Saving plots ep: {epoch}')
            tensors = [o.tensors_as_dict() for o in observer.observers]
            sacred_writer.save_tensor(tensors, 'tensors', epoch)

        # log targets
        if epoch % p.save_period == 0:
            sacred_writer.save_model(agent, 'agent', epoch)
