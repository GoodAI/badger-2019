from abc import ABC, abstractmethod
from argparse import Namespace

import torch

# from marl.experimental.deeprl.policies.atoc_policy import ATOCPolicy
from marl.experimental.deeprl.policies.atoc_comm.atoc_communication_base import ATOCCommunicationBase
from marl.experimental.deeprl.policies.networks import SimpleLSTM, NetworkBase, BidirectionalLSTM
from marl.experimental.deeprl.utils.available_device import my_device


class BidirectionalLSTMBatched(ATOCCommunicationBase):
    config: Namespace

    def __init__(self,
                 owner: 'ATOCPolicy',
                 config: Namespace):
        super().__init__(owner, config.force_communication)
        self.current_initiators = None
        self.config = config

        self.index_tensor = torch.arange(self.owner.num_agents, device=my_device()).view(1, -1)

    def build_network(self) -> NetworkBase:
        # communication channel - bidirectional LSTM which integrates thoughts in time
        return BidirectionalLSTM(input_size=self.owner.thought_size,
                                 output_size=self.owner.thought_size,
                                 hidden_size=self.config.communication_channel_sizes[0],
                                 num_layers=2,
                                 disable_output_layer=True).to(my_device())

    def reset(self):
        self.current_step = -1

    def determine_initiators_random(self, _: torch.Tensor) -> torch.Tensor:
        # TODO obsolete
        initiators = torch.randint(low=0, high=2, size=(self.owner.num_agents, 1), dtype=torch.bool,
                                   device=my_device())
        return initiators

    def communicate(self,
                    thoughts: torch.Tensor,
                    agent_ids: torch.Tensor,
                    comm_matrix: torch.Tensor,
                    generate_comm: bool = True):

        new_thoughts, comm_matrix = self._communicate(thoughts.unsqueeze(0), agent_ids.unsqueeze(0),
                                                      comm_matrix.unsqueeze(0), generate_comm)
        return new_thoughts.squeeze(0), comm_matrix.squeeze(0)

    def _communicate(self,
                     thoughts: torch.Tensor,
                     agent_ids: torch.Tensor,
                     comm_matrices: torch.Tensor,
                     generate_comm: bool = True):
        """ Communication for the current step

        Args:
            thoughts: [batch_size, num_agents, thought_size]
            agent_ids: [batch_size, num_agents, num_perceived_agents]
            comm_matrices: [batch_size, num_agents, num_agents]
            generate_comm:

        Returns: new thought vectors and new comm_matrix
        """
        self.current_step += 1

        if generate_comm:
            comm_matrices = self.determine_communication_pools(agent_ids, comm_matrices, thoughts)

        batch_size = comm_matrices.size(0)

        # Work out the initiators from the comm matrices
        return_thoughts = torch.zeros_like(thoughts)
        return_thoughts.copy_(thoughts.detach())

        index_tensor = self.index_tensor.expand(batch_size, -1)
        batch_index_tensor = torch.arange(batch_size, device=my_device()).view(batch_size, 1)

        # Communication phase
        for init_id in torch.arange(self.owner.num_agents, device=my_device()):
            collaborators = comm_matrices[:, init_id].clone()
            update_thoughts = collaborators.sum(1) > 0
            # Because there are no collaborators for experts that are not updating, make some up!
            not_update = update_thoughts == 0
            collaborators[not_update, :self.owner.comm_bandwidth] = 1

            # Retrieve the thoughts batch that will be communicated to the
            thoughts_indices = index_tensor[collaborators].view(batch_size, self.owner.comm_bandwidth)
            thoughts_indices = torch.cat((init_id.view(1, 1).expand(batch_size, 1), thoughts_indices), dim=1)
            input_thoughts = return_thoughts[batch_index_tensor, thoughts_indices]

            # The communication channel should be reset each run, use it to get the new thoughts
            self.owner.communication_channel.reset(batch_size)
            new_thoughts = self.owner.communication_channel(input_thoughts)#[:, :, self.owner.thought_size:]

            # Write the new thoughts back to input thoughts based on whether the batch is fake or not
            not_updated = not_update.view(batch_size, 1, 1).float()
            updated = update_thoughts.view(batch_size, 1, 1).float()
            return_thoughts[batch_index_tensor, thoughts_indices] = new_thoughts * updated + input_thoughts * not_updated

        # mask out any thoughts that haven't been changed (i.e the expert isn't in a comm pool)
        should_be_updated = torch.clamp(comm_matrices.sum(1) + comm_matrices.sum(2), max=1).unsqueeze(2).expand(
            return_thoughts.shape)
        return_thoughts = return_thoughts * should_be_updated.float()
        return return_thoughts, comm_matrices

    def communicate_batched(self,
                            thoughts: torch.Tensor,
                            agent_ids: torch.Tensor,
                            comm_matrices: torch.Tensor) -> torch.Tensor:
        """ Communication in the batch by sequential processing
        Args:
            thoughts: [batch_size, num_agents, thought_size]
            agent_ids: [batch_size, num_agents, num_perceived_agents]
            comm_matrices: [batch_size, num_agents, num_agents]

        Returns: integrated_thoughts [batch_size, num_agents, thought_size]
        """
        t, m = self._communicate(thoughts, agent_ids, comm_matrices, generate_comm=False)
        return t

