from typing import List

import torch
import torch.nn as nn
from torch.autograd import Variable

from agents.emergent_language.modules.processing import ProcessingModule
from agents.emergent_language.modules.goal_predicting import GoalPredictingProcessingModule
from agents.emergent_language.modules.action import ActionModule
from agents.emergent_language.modules.word_counting import WordCountingModule


class MyAgent(nn.Module):
    """

    """

    device: str = 'cpu'
    # batch_size: int
    # num_agents: int
    # num_entities: int

    def __init__(self, config):
        super(MyAgent, self).__init__()

        # parse config
        self.training = True
        self.using_utterances = config.use_utterances
        self.penalizing_words = config.penalize_words
        # self.using_cuda = config.use_cuda
        self.time_horizon = config.time_horizon
        self.movement_dim_size = config.movement_dim_size
        self.vocab_size = config.vocab_size
        self.goal_size = config.goal_size
        self.processing_hidden_size = config.physical_processor.hidden_size

        # self.batch_size = batch_size
        # self.num_agents = num_agents
        # self.num_entities = num_entities

        # tensors
        # self.total_cost = Variable(self.Tensor(1).zero_())
        self.total_cost = torch.zeros(1, dtype=torch.float32, requires_grad=True).to(self.device)

        # physical processing
        self.physical_processor = ProcessingModule(config.physical_processor)
        self.physical_pooling = nn.AdaptiveMaxPool2d((1, config.feat_vec_size))

        # doing actions
        self.action_processor = ActionModule(config.action_processor)

        # utterance processing
        if self.using_utterances:
            self.utterance_processor = GoalPredictingProcessingModule(config.utterance_processor)
            self.utterance_pooling = nn.AdaptiveMaxPool2d((1,config.feat_vec_size))

            if self.penalizing_words:
                self.word_counter = WordCountingModule(config.word_counter)

    def reset(self):
        self.total_cost = torch.zeros_like(self.total_cost)

        if self.using_utterances and self.penalizing_words:
            self.word_counter.word_counts = torch.zeros_like(self.word_counter.word_counts)

    def train(self, mode=True):
        super(MyAgent, self).train(mode)
        self.training = mode

    # ----------------- physical

    def process_physical(self,
                         agent: int,
                         other_entity: int,
                         physical_processes,
                         observations):
        physical_processed, new_mem = self.physical_processor(
            torch.cat((observations[:, agent, other_entity], game.physical[:, other_entity]),
                      1),
            game.memories["physical"][:, agent, other_entity]
        )

        self.update_mem(game, "physical", new_mem,agent, other_entity)
        physical_processes[:, other_entity, :] = physical_processed

    def get_physical_feat(self,
                          agent: int,
                          batch_size: int,
                          num_entities: int,
                          processing_hidden_size: int,
                          observations: torch.Tensor):
        # physical_processes = Variable(self.Tensor(game.batch_size, game.num_entities, self.processing_hidden_size))
        # results of the physical networks
        physical_processes = torch.tensor((batch_size, num_entities, processing_hidden_size),
                                          dtype=torch.float32, requires_grad=True).to(self.device)

        # go through each of the network and compute the result
        for entity in range(num_entities):
            self.process_physical(agent, entity, physical_processes, observations)

        # pool the results
        return self.physical_pooling(physical_processes)

    def make_step(self, batch_size: int, num_agents: int, num_entities: int, observations: List[np.array]):

        # movements = self.Tensor(batch_size, num_entities, movement_dim_size).zero_())
        movements = torch.zeros((batch_size, num_entities, self.movement_dim_size),
                                dtype=torch.float32, requires_grad=True).to(self.device)
        utterances = None
        goal_predictions = None

        if self.using_utterances:
            # utterances = Variable(self.Tensor(batch_size, num_agents, self.vocab_size))
            utterances = torch.tensor((batch_size, num_agents, self.vocab_size),
                                      dtype=torch.float32, requires_grad=True).to(self.device)
            # goal_predictions = Variable(self.Tensor(batch_size, num_agents, num_agents, self.goal_size))
            goal_predictions = torch.tensor((batch_size, num_agents, num_agents, self.goal_size),
                                            dtype=torch.float32, requires_grad=True).to(self.device)

        # TODO check the type of observations and convert ?

        # TODO this
        # for agent in range(num_agents):
            # physical_feat = self.get_physical_feat(agent,
            #                                        batch_size,
            #                                        num_entities,
            #                                        self.processing_hidden_size,
            #                                        observations)

            # utterance_feat = self.get_utterance_feat(game, agent, goal_predictions)
            # self.get_action(game, agent, physical_feat, utterance_feat, movements, utterances)

    def forward(self, *input):
        # TODO
        print("TODO")
        pass
