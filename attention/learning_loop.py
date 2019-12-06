from typing import Optional
from torch import Tensor
import torch

from attention.agents.agent import Agent, DeviceAware
from badger_utils.view.observer_utils import MultiObserver
from attention.search_task import TorchTask


class LearningLoop(DeviceAware):
    def __init__(self, device: Optional[str] = None):
        super().__init__(device)

    # def _convert_to_torch(self, data: np.array) -> Tensor:
    #     return torch.from_numpy(data).to(self.device)

    def train_fixed_steps(self, agent: Agent, task: TorchTask, steps: int, exp_decay: float, error_clip_steps: int,
                          observer: Optional[MultiObserver] = None) -> Tensor:
        """
        Train agent for fixed number of steps on task
        Args:
            agent:
            task:
            steps:
            observer:

        Returns: error of shape [] - a single value

        """
        observations = task.reset()
        errors = torch.tensor(0.0, device=self.device)
        err_counter = 0
        for step in range(steps):
            outputs, w, idx = agent.forward(observations, errors, observer=observer)
            obs, reward, is_done, _ = task.step(outputs)
            decay = (exp_decay ** (steps - step - 1))  # exp decay
            weight = 1 if decay == 0.0 else 1 / decay
            error = torch.mean(reward) * weight
            if step >= steps - error_clip_steps:
                err_counter += 1
                errors += error
            if observer is not None:
                observer.current.add_tensor("error_per_inf_step", error.clone().detach())
                if step < steps - 1:
                    observer.add_observer()

        return errors / err_counter
