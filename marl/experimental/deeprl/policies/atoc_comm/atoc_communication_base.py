from abc import ABC, abstractmethod
from typing import Optional

import torch

# from marl.experimental.deeprl.policies.atoc_policy import ATOCPolicy
from marl.experimental.deeprl.policies.networks import NetworkBase
from marl.experimental.deeprl.utils.available_device import my_device


class ATOCCommunicationBase(ABC):

    owner: 'ATOCPolicy'
    current_step: int
    force_communication: bool

    def __init__(self,
                 owner: 'ATOCPolicy',
                 force_communication: Optional[bool] = False):

        self.current_step = -1
        self.owner = owner
        self.force_communication = force_communication

        self.ones_for_setting_b = torch.ones((1, self.owner.comm_bandwidth,), dtype=torch.bool, device=my_device())
        self.ones_for_setting_l = torch.ones((1, self.owner.comm_bandwidth,), dtype=torch.int64, device=my_device())

    @abstractmethod
    def build_network(self) -> NetworkBase:
        """Build a network and don't store the reference (we need target as well)"""
        pass

    @abstractmethod
    def reset(self):
        pass

    def determine_initiators(self, thoughts: torch.Tensor) -> torch.Tensor:
        """ Find the communication initiators

        -run the classifier on each thought vector, compute the P(communication)
        -sample the probabilities

        Args:
            thoughts: [batch_size,  num_agents, thought_size]

        Returns: [batch_size, num_agents] binary indicator whether the agent wants to initiate the communication

        """
        if self.force_communication:
            return torch.ones((thoughts.shape[0], thoughts.shape[1]), dtype=torch.long, device=my_device())

        probabilities = self.owner.classifier.forward(thoughts)
        # TODO store these probabilities to the buffer
        random_uniform = torch.rand(size=probabilities.shape, device=my_device())
        decisions = random_uniform <= probabilities

        return decisions.long().view(-1, self.owner.num_agents)

    def determine_initiators_random(self, _: torch.Tensor) -> torch.Tensor:
        # TODO obsolete
        initiators = torch.randint(low=0, high=2, size=(self.owner.num_agents, 1), dtype=torch.bool,
                                   device=my_device())
        return initiators

    def determine_communication_pools(self,
                                      nearest_agent_ids: torch.Tensor,
                                      comm_matrices: torch.Tensor,
                                      thoughts: torch.Tensor) -> torch.Tensor:
        """Changes the last_comm matrix and sets the self._current_initiators tensor values

        Returns: communication matrices of sizes [batch_size, num_agents, num_agents]
        """
        assert len(nearest_agent_ids.shape) == 3, 'expected size [batch_size, num_agents, num_perceived_agents]'
        assert len(comm_matrices.shape) == 3, 'expected size [batch_size, num_agents, num_agents]'
        assert len(thoughts.shape) == 3, 'expected size [batch_size, num_agents, thought_size]'
        assert nearest_agent_ids.shape[0] == comm_matrices.shape[0] == thoughts.shape[0]
        assert nearest_agent_ids.shape[1] == comm_matrices.shape[1] == thoughts.shape[1]

        if self.current_step % self.owner.comm_decision_period == 0:

            batch_size = comm_matrices.size(0)
            batch_index = torch.arange(batch_size, device=my_device())
            ones_for_setting_b = self.ones_for_setting_b.expand(batch_size, self.owner.comm_bandwidth, )
            ones_for_setting_l = self.ones_for_setting_l.expand(batch_size, self.owner.comm_bandwidth, )

            # which agents initiate the communication now?
            # initiators = self.determine_initiators_random(thoughts).view(-1)
            initiators = self.determine_initiators(thoughts)

            # Indices of nearest experts collected, now cycle through and pick those to communicate to
            comm_matrices = torch.zeros_like(comm_matrices)
            communicating_with_others = torch.zeros((batch_size, self.owner.num_agents), dtype=torch.int64,
                                                    device=my_device())

            for i_expert in range(self.owner.num_agents):
                # Get list of experts which are actually going to do communication, but treat all experts in batch as
                # if they are going to communicate and gat out the ones that aren't at the end
                actually_initiating = initiators[:, i_expert]
                n_indices = nearest_agent_ids[:, i_expert]
                modified_indices = n_indices.clone()

                # Go though potential experts and multiply them if they are being communicated with or are initiators
                # themselves
                for comm_idx in range(modified_indices.size(1)):
                    potentials = modified_indices[:, comm_idx]
                    potentials_recieving = communicating_with_others[batch_index, potentials] * 1000
                    potentials_initiating = initiators[batch_index, potentials] * 100000

                    modified_indices[:, comm_idx] = (potentials_recieving + potentials_initiating)

                _, selected_indices = torch.topk(modified_indices, self.owner.comm_bandwidth, largest=False, dim=1)
                experts_to_communicate_with = n_indices[batch_index.view(-1, 1), selected_indices]

                # Gate the ones that are not actually communicating out of the comm matrix update, and out of the
                # tracker for if an expert is being communicated with
                comm_matrices[batch_index.view(-1,
                                               1), i_expert, experts_to_communicate_with] = ones_for_setting_b * actually_initiating.view(
                    -1, 1).bool()
                communicating_with_others[batch_index.view(-1,
                                                           1), experts_to_communicate_with] += ones_for_setting_l * actually_initiating.view(
                    -1, 1)

        return comm_matrices

    @abstractmethod
    def communicate(self,
                    thoughts: torch.Tensor,
                    agent_ids: torch.Tensor,
                    comm_matrix: torch.Tensor,
                    generate_comm: bool = True):
        pass

    @abstractmethod
    def communicate_batched(self,
                            thoughts: torch.Tensor,
                            agent_ids: torch.Tensor,
                            comm_matrix: torch.Tensor) -> torch.Tensor:
        pass

