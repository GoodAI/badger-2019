from typing import Optional
from torch import Tensor
import torch

from attention_sgd.agents.agent import Agent, DeviceAware
from attention_sgd.utils.observer_utils import MultiObserver
from attention_sgd.search_task import TorchTask


class LearningLoop(DeviceAware):
    def __init__(self, device: Optional[str] = None):
        super().__init__(device)

    # def _convert_to_torch(self, data: np.array) -> Tensor:
    #     return torch.from_numpy(data).to(self.device)

    def train_fixed_steps(self, agent: Agent, task: TorchTask, steps: int,
                          observer: Optional[MultiObserver] = None) -> Tensor:
        """
        Train agent for fixed number of steps on task
        Args:
            agent:
            task:
            steps:
            observer:

        Returns: error of shape [batch_size]

        """
        observations = task.reset()
        errors = torch.zeros((0,), device=self.device)
        for step in range(steps):
            outputs, w, idx = agent.forward(observations, errors, observer=observer)
            obs, reward, is_done, _ = task.step(outputs)
            errors = reward
            if observer is not None and step < steps - 1:
                observer.add_observer()

        # return error just from the last step
        return errors
