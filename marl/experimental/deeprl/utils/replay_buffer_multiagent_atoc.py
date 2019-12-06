from dataclasses import dataclass
from typing import List

import torch

from marl.experimental.deeprl.utils.replay_buffer_multiagent import ReplayBufferMultiagent


@dataclass
class ATOCMultiagentTransition:
    """Transition for N parallel agents"""

    state: List[torch.Tensor]
    action: List[torch.Tensor]
    new_state: List[torch.Tensor]
    reward: List[float]
    done: List[bool]
    comm_matrix: torch.Tensor  # matrix [num_agents, num_agents] indicating communication groups (line~initiator)


class ReplayBufferMultiagentATOC(ReplayBufferMultiagent):
    """The same as ReplayBuffer, but works for multiple agents at the same time"""

    buffer: List[ATOCMultiagentTransition]
    num_agents: int

    def __init__(self,
                 buffer_size: int,
                 input_sizes: List[int],
                 action_sizes: List[int],
                 _run=None):
        super().__init__(
            buffer_size=buffer_size,
            input_sizes=input_sizes,
            action_sizes=action_sizes,
            _run=_run)

        self.num_agents = len(input_sizes)

    def remember(self,
                 states: List[torch.Tensor],
                 actions: List[torch.Tensor],
                 new_states: List[torch.Tensor],
                 rewards: List[float],
                 dones: List[bool]):
        raise NotImplementedError('the method remember should not be used, use the remember_atoc instead')

    def remember_atoc(self,
                      states: List[torch.Tensor],
                      actions: List[torch.Tensor],
                      new_states: List[torch.Tensor],
                      rewards: List[float],
                      dones: List[bool],
                      comm_matrix: torch.Tensor):
        """
        Remember one transition for all agents separately in ATOCMultiagentTransition.
        """

        states, actions, new_states, rewards, dones = self._sanitize_inputs(states, actions, new_states, rewards, dones)

        self._check_inputs(states, actions, new_states, rewards, dones)
        assert isinstance(comm_matrix, torch.Tensor)
        assert comm_matrix.shape == (self.num_agents, self.num_agents)

        transition = ATOCMultiagentTransition(states, actions, new_states, rewards, dones, comm_matrix)

        self.push_to_buffer(transition)
