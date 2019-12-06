import io
import os
from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np
import torch
from badger_utils.sacred import SacredWriter, SacredReader, Serializable


class PolicyBase(ABC, Serializable):

    tau: float
    collected_rewards: List[float]

    def __init__(self):
        self.collected_rewards = []

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def pick_action(self, observation: np.ndarray) -> np.array:
        pass

    @abstractmethod
    def remember(self,
                 new_observation: np.array,
                 reward: float,
                 done: bool):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def set_epsilon(self, epsilon: float):
        pass

    @property
    def epsilon(self) -> float:
        raise NotImplementedError('epsilon property not implemented')

    @abstractmethod
    def reset(self, batch_size: int = 1):
        pass

    @abstractmethod
    def log(self, step: int):
        """log anything (e.g. into sacred) here"""
        pass

    @property
    def is_tracking_used(self) -> bool:
        return self.tau != 0.0

    @staticmethod
    def track_network_weights(source, target, tau: float):
        if tau == 0:
            return
        assert 0 < tau <= 1

        for target_param, local_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, name: str, epoch: int):

        def to_bytes(data) -> bytes:
            writer = io.BytesIO()
            torch.save(data, writer)
            return writer.getvalue()

        dictionary = self.serialize()
        blob = to_bytes({key: to_bytes(value) for (key, value) in dictionary.items()})

        name = f"data/models/{name}_{epoch}.model"

        # create temp directory, write blob to the file, push file to the sacred, delete directory
        if not os.path.exists('data/models/'):
            os.makedirs('data/models')
        with open(name, 'wb') as f:
            f.write(blob)

    def load(self, model_file: str):

        def from_bytes(my_blob: bytes) -> Dict[str, bytes]:
            data_buff = io.BytesIO(my_blob)
            dictionary = torch.load(data_buff)
            return dictionary

        with open(f'data/models/{model_file}.model', 'rb') as f:
            blob = f.read()

        top_dict = from_bytes(blob)
        dictionary = {key: from_bytes(value) for (key, value) in top_dict.items()}
        self.deserialize(dictionary)

    def append_reward(self, reward: float):
        """Collects stats about the average reward received"""
        self.collected_rewards.append(reward)

    def get_avg_reward(self):
        """Average reward from the last logging, clears the collected rewards during the call"""
        avg_rew = sum(self.collected_rewards) / len(self.collected_rewards) if len(self.collected_rewards) > 0 else 0
        self.collected_rewards = []
        return avg_rew
