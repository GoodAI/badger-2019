from argparse import Namespace

import torch

# from marl.experimental.deeprl.policies.atoc_policy import ATOCPolicy
from marl.experimental.deeprl.policies.atoc_comm.atoc_communication_base import ATOCCommunicationBase
from marl.experimental.deeprl.policies.networks import NetworkBase, SimpleFF
from marl.experimental.deeprl.utils.available_device import my_device


class FixedZeroToOne(ATOCCommunicationBase):
    """Fixed communication which sends the information only from the agent 0 to agent 1"""

    current_step: int
    config: Namespace

    def __init__(self,
                 owner: 'ATOCPolicy',
                 config: Namespace):
        super().__init__(owner, config.force_communication)
        self.current_step = -1
        self.config = config

        assert config.comm_bandwidth == 1, 'just 1to1 communication supported for now'
        assert config.num_perceived_agents == 1, 'just a comm. from agent 0 to agent 1 supported'

    def build_network(self) -> NetworkBase:
        # not used here
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
        integrated_thoughts = self.communicate_batched(thoughts.unsqueeze(0), agent_ids.unsqueeze(0), comm_matrix.unsqueeze(0))
        return integrated_thoughts.squeeze(0), comm_matrix

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

        integrated_thoughts = torch.zeros_like(thoughts, device=my_device())
        integrated_thoughts[:, 1] = thoughts[:, 0]

        return integrated_thoughts
