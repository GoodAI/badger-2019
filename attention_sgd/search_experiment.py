import dataclasses
import random
from dataclasses import dataclass

import torch
import numpy as np
from badger_utils.sacred import SacredWriter, SacredConfigFactory, SacredReader
from sacred.run import Run
from tqdm import tqdm

from attention_sgd.models.attention.attention import AttentionOperation
from attention_sgd.utils.observer_utils import Observer, MultiObserver
from attention_sgd.utils.torch_utils import default_device

from sacred import Experiment

from attention_sgd.batched_task import BatchedTask
from attention_sgd.learning_loop import LearningLoop
from attention_sgd.search_agent import SearchAgent
from attention_sgd.search_task import FillTask, FillTorchTask, ToZeroTorchTask

ex = Experiment("Search task")
sacred_writer = SacredWriter(ex, SacredConfigFactory.local())


@dataclass
class Params:
    seed: int
    batch_size: int = 32
    hidden_state_size: int = 16
    id_size: int = 31  # must be value_size - 1
    key_size: int = 2
    value_size: int = 32

    epochs: int = 3000
    rollout_size: int = 2

    # ignore_error_start_steps: int = 5
    # training_reset_targets_period: int = -1
    learning_rate: float = 1e-3

    task_size: int = 5
    n_experts: int = 5

    save_period: int = 100
    image_save_period: int = 100
    attention_beta: float = 8
    attention_operation: str = 'euclidean_distance'


ex.add_config(dataclasses.asdict(Params(random.randint(0, 2 ** 32))))


def load_agent(experiment_id, epoch, agent):
    reader = SacredReader(experiment_id, SacredConfigFactory.local())
    reader.load_model(agent, 'agent', epoch)


@ex.automain
def main(_run: Run, _config):
    p = Params(**_config)

    agent = SearchAgent(hidden_state_size=p.hidden_state_size, key_size=p.key_size, id_size=p.id_size,
                        value_size=p.value_size, input_size=1, n_inputs=p.task_size,
                        attention_beta=p.attention_beta,
                        attention_operation=AttentionOperation.from_string(p.attention_operation),
                        learning_rate=p.learning_rate).to(default_device())
    # load_agent(761, 200, agent)

    task = BatchedTask(p.batch_size, lambda: ToZeroTorchTask(p.task_size))
    inner_loop = LearningLoop()

    for epoch in tqdm(range(1, p.epochs + 1)):
        # with torch.autograd.detect_anomaly():
        should_save_images = epoch % p.image_save_period == 0
        observer = MultiObserver() if should_save_images else None

        agent.optim.zero_grad()
        agent.init_rollout(p.batch_size, p.n_experts)
        err = inner_loop.train_fixed_steps(agent, task, p.rollout_size, observer)
        # task.reset(True, True)
        err = torch.mean(err)
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
