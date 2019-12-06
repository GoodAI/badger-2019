from abc import ABC, abstractmethod
from argparse import Namespace

import torch

# from marl.experimental.deeprl.policies.atoc_policy import ATOCPolicy
from marl.experimental.deeprl.policies.atoc_comm.atoc_communication_base import ATOCCommunicationBase
from marl.experimental.deeprl.policies.networks import SimpleLSTM, NetworkBase
from marl.experimental.deeprl.utils.available_device import my_device


class BidirectionalLSTMSequential(ATOCCommunicationBase):
    current_step: int
    config: Namespace

    def __init__(self,
                 owner: 'ATOCPolicy',
                 config: Namespace):
        super().__init__(owner, config.force_communication)
        self.current_step = -1
        self.config = config

    def build_network(self) -> NetworkBase:
        # communication channel - bidirectional LSTM which integrates thoughts in time
        return SimpleLSTM(input_size=self.owner.thought_size,
                          output_size=self.owner.thought_size,
                          hidden_sizes=self.config.communication_channel_sizes,
                          bidirectional=True,
                          disable_output_layer=True).to(my_device())

    def reset(self):
        self.current_step = -1

    def communicate(self,
                    thoughts: torch.Tensor,
                    agent_ids: torch.Tensor,
                    comm_matrix: torch.Tensor,
                    generate_comm: bool = True):
        """ Communication for the current step

        Args:
            thoughts: [num_agents, thought_size]
            agent_ids: [num_agents, num_perceived_agents]
            comm_matrix: [num_agents, num_agents]
            generate_comm:

        Returns: new thought vectors and new comm_matrix
        """
        self.current_step += 1

        if generate_comm:
            # This generation method assumes a batch of inputs
            comm_matrix = self.determine_communication_pools(agent_ids.unsqueeze(0), comm_matrix.unsqueeze(0), thoughts.unsqueeze(0))
            comm_matrix = comm_matrix.squeeze(0)

        # Reset the communication network hidden state
        self.owner.communication_channel.reset()

        return_thoughts = torch.zeros_like(thoughts)
        return_thoughts.copy_(thoughts)

        # Communication phase
        current_initiators = torch.nonzero(comm_matrix.sum(1).view(-1)).view(-1)

        comms = comm_matrix[current_initiators]
        for init_id, collaborators in zip(current_initiators, comms):
            thoughts_indices = torch.cat((init_id.unsqueeze(0), torch.nonzero(collaborators).squeeze(1)))
            input_thoughts = return_thoughts[thoughts_indices].unsqueeze(0)

            new_thoughts = self.owner.communication_channel(input_thoughts)
            return_thoughts[thoughts_indices] = new_thoughts

        # mask out any thoughts that haven't been changed (i.e the expert isn't in a comm pool)
        return_thoughts = return_thoughts * (
                torch.any(comm_matrix.transpose(1, 0), dim=1) + torch.any(comm_matrix, dim=1)).float().unsqueeze(1)
        return return_thoughts, comm_matrix

    def communicate_batched(self,
                            thoughts: torch.Tensor,
                            agent_ids: torch.Tensor,
                            comm_matrix: torch.Tensor) -> torch.Tensor:
        """ Communication in the batch by sequential processing
        Args:
            thoughts: [batch_size, num_agents, thought_size]
            agent_ids: [batch_size, num_agents, num_perceived_agents]
            comm_matrix: [batch_size, num_agents, num_agents]

        Returns: integrated_thoughts [batch_size, num_agents, thought_size]
        """
        integrated_thoughts = []
        for t, a, c in zip(thoughts, agent_ids, comm_matrix):
            new_thought, _ = self.communicate(t, a, c, generate_comm=False)
            integrated_thoughts.append(new_thought)

        return torch.stack(integrated_thoughts)
