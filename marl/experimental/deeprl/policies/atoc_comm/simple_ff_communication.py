from argparse import Namespace

import torch

# from marl.experimental.deeprl.policies.atoc_policy import ATOCPolicy
from marl.experimental.deeprl.policies.atoc_comm.atoc_communication_base import ATOCCommunicationBase
from marl.experimental.deeprl.policies.networks import NetworkBase, SimpleFF
from marl.experimental.deeprl.utils.available_device import my_device


class SimpleFFCommunication(ATOCCommunicationBase):
    current_step: int
    config: Namespace

    def __init__(self,
                 owner: 'ATOCPolicy',
                 config: Namespace):
        super().__init__(owner, config.force_communication)
        self.current_step = -1
        self.config = config

        assert config.comm_bandwidth == 1, 'just 1to1 communication supported for now'

    def build_network(self) -> NetworkBase:
        return SimpleFF(input_size=self.owner.thought_size,
                        output_size=self.owner.thought_size,
                        hidden_sizes=self.config.communication_channel_sizes,
                        output_activation='tanh').to(my_device())

    def reset(self):
        self.current_step = -1

    def communicate(self,
                    thoughts: torch.Tensor,
                    agent_ids: torch.Tensor,
                    comm_matrix: torch.Tensor,
                    generate_comm: bool = True):
        """ Communication for the current step.

        Args:
            thoughts: [num_agents, thought_size]
            agent_ids: [num_agents, num_perceived_agents]
            comm_matrix: [num_agents, num_agents]
            generate_comm: whether to re-initialize the communication (used only during inference)

        Returns: new thought vectors and (possibly unchanged) comm_matrix
        """
        assert len(thoughts.shape) == len(agent_ids.shape) == len(comm_matrix.shape) == 2
        # is_batched = len(thoughts.shape) == 3

        # if not is_batched:
        #     thoughts = thoughts.unsqueeze(0)
        #     agent_ids = agent_ids.unsqueeze(0)
        #     comm_matrix = comm_matrix.unsqueeze(0)

        self.current_step += 1

        """Determine the initiators in the comm_matrix"""
        if generate_comm:
            comm_matrix = self.determine_communication_pools(agent_ids.unsqueeze(0),
                                                             comm_matrix.unsqueeze(0),
                                                             thoughts.unsqueeze(0)).squeeze(0)
        current_initiators = torch.nonzero(comm_matrix.sum(1).view(-1)).view(-1)

        return_thoughts = torch.zeros_like(thoughts)
        # return_thoughts.copy_(thoughts)

        """No initiators? done"""
        if current_initiators.numel() == 0:
            # if is_batched:
            #     return return_thoughts.squeeze(0), comm_matrix.squeeze(0)
            return return_thoughts, comm_matrix

        """Run the thoughts of initiators through the comm. channel"""
        comm_channel_inputs = thoughts[current_initiators]
        comm_channel_outputs = self.owner.communication_channel.forward(comm_channel_inputs)

        """Deliver the messages to the correct places"""
        addressees = comm_matrix[current_initiators]  # [num_initiators, num_agents]

        for current_initiator, initiator_addressees in zip(current_initiators, addressees):
            ids = initiator_addressees.nonzero()
            ids = ids.view(-1).squeeze()  # the only one target id should be here

            return_thoughts[ids] = comm_channel_outputs[current_initiator]
            # print(f'done')

        # for initiator_id, collaborators in zip(current_initiators, initiator_targets):
        #
        #     thoughts_indices = torch.cat((initiator_id.unsqueeze(0), torch.nonzero(collaborators).squeeze(1)))
        #     input_thoughts = return_thoughts[thoughts_indices].unsqueeze(0)
        #
        #     new_thoughts = self.owner.communication_channel(input_thoughts)
        #     return_thoughts[thoughts_indices] = new_thoughts
        #
        # mask out any thoughts that haven't been changed (i.e the expert isn't in a comm pool)
        # return_thoughts = return_thoughts * (
        #         torch.any(comm_matrix.transpose(1, 0), dim=1) + torch.any(comm_matrix, dim=1)).clamp(
        #     max=1).float().unsqueeze(1)
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
        # integrated_thoughts, _ = self.communicate(thoughts.unsqueeze, agent_ids, comm_matrix, generate_comm=False)

        integrated_thoughts = []
        for t, a, c in zip(thoughts, agent_ids, comm_matrix):
            new_thought, _ = self.communicate(t, a, c, generate_comm=False)
            integrated_thoughts.append(new_thought)

        return torch.stack(integrated_thoughts)
        # return integrated_thoughts
