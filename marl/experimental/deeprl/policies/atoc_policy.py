from argparse import Namespace
from pydoc import locate
from typing import Dict, Any, Optional, List, Union, Tuple

import gym
import numpy as np
import torch

from marl.experimental.deeprl.policies.atoc_comm.bidirectional_lstm_sequential import BidirectionalLSTMSequential, ATOCCommunicationBase
from marl.experimental.deeprl.policies.ddpg_policy import DDPGPolicy
from marl.experimental.deeprl.policies.networks import NetworkBase, my_device, SimpleFF, SimpleLSTM
from marl.experimental.deeprl.policies.policy_base import PolicyBase
from marl.experimental.deeprl.utils.atoc_comm_classifier_buffer import ATOCClassifierBuffer
from marl.experimental.deeprl.utils.replay_buffer_multiagent_atoc import ReplayBufferMultiagentATOC, ATOCMultiagentTransition
from marl.experimental.deeprl.utils.utils import get_gym_box_dimensionality, get_total_space_size


class ATOCAgent(PolicyBase):
    """"
    Each ATOC agent should:
        -have own exploration (self-correlated noise generator),
        -compute own reward statistics
    """

    agent_id: int

    def __init__(self,
                 num_actions: int,
                 config: Namespace,
                 agent_id: int):
        super().__init__()

        exploration_class = locate(config.exploration)
        self.exploration = exploration_class(action_dim=num_actions, config=config)

        self.agent_id = agent_id

    def pick_action(self, observation: np.ndarray) -> np.array:
        print(f'TODO')

    def remember(self, new_observation: np.array, reward: float, done: Optional[bool]):
        raise NotImplementedError('should not be used')

    def learn(self):
        raise NotImplementedError('should not be used')

    def set_epsilon(self, epsilon: float):
        self.exploration.set_epsilon(epsilon)

    @property
    def epsilon(self) -> float:
        return self.exploration.st_dev

    def log(self, step: int):
        raise NotImplementedError('should not be used')

    def reset(self, batch_size: int = 1):
        raise NotImplementedError('should not be used')

    def serialize(self) -> Dict[str, Any]:
        raise NotImplementedError('should not be used')

    def deserialize(self, data: Dict[str, Any]):
        raise NotImplementedError('should not be used')


class ATOCPolicy(PolicyBase):
    total_steps: Optional[int]
    num_learns: int
    num_learning_iterations: int

    buffer: ReplayBufferMultiagentATOC
    batch_size: int

    classifier_buffer: ATOCClassifierBuffer
    classifier_buffer_size: int
    classifier_batch_size: int
    classifier_lr: float

    last_observations: List[torch.tensor]
    last_actions: List[torch.tensor]
    last_comm_matrix: torch.Tensor
    last_thoughts: torch.Tensor

    """Training"""
    run_id: int
    num_learns: int
    last_actor_loss: float
    last_critic_loss: 0
    last_classifier_loss: 0

    last_observations: List[torch.Tensor]
    last_actions: List[torch.Tensor]

    gamma: float
    tau: float

    agents: List[ATOCAgent]
    num_agents: int
    num_perceived_agents: int

    thought_size: int
    disable_communication: bool
    force_communication: bool

    """Main networks"""
    actor_a: NetworkBase
    actor_b: NetworkBase
    critic: NetworkBase
    classifier: SimpleFF
    communication_channel: torch.nn.Module

    """Main networks"""
    target_actor_a: NetworkBase
    target_actor_b: NetworkBase
    target_critic: NetworkBase
    target_classifier: SimpleFF
    target_communication_channel: torch.nn.Module

    """Communication infrastructure"""
    communication: ATOCCommunicationBase

    def __init__(self,
                 observation_space: Union[List, gym.Space],
                 action_space: Union[gym.Space, List],
                 config: Namespace,
                 _run=None,
                 run_id: int = 0):
        super().__init__()

        """Params"""
        self._run = _run
        self.run_id = run_id
        self.gamma = config.gamma

        assert config.tau > 0, 'target networks need to be used'
        self.tau = config.tau

        assert config.batch_size > 0
        self.batch_size = config.batch_size

        assert config.num_learning_iterations > 0
        self.num_learning_iterations = config.num_learning_iterations

        self.comm_decision_period = config.comm_decision_period
        self.comm_bandwidth = config.comm_bandwidth
        self.disable_communication = config.disable_communication

        """IO sizes"""
        self.input_size, self.output_size = self.read_io_size(observation_space, action_space)
        self.num_agents = len(observation_space)
        self.num_perceived_agents = config.num_perceived_agents
        assert self.num_perceived_agents <= self.num_agents
        self.last_comm_matrix = torch.zeros(self.num_agents, self.num_agents, dtype=torch.bool, device=my_device())

        """Build networks"""
        self.actor_a, self.actor_b, self.thought_size, self.critic, self.classifier = \
            self.build_networks(config=config, _run=_run, observation_space=observation_space)

        self.target_actor_a, self.target_actor_b, _, self.target_critic, self.target_classifier = \
            self.build_networks(config=config, _run=_run, observation_space=observation_space)  # TODO use the target classifier

        """Build communication channel networks"""
        atoc_comm_class = locate(config.atoc_comm)
        self.communication = atoc_comm_class(self, config)
        self.communication_channel = self.communication.build_network()
        self.target_communication_channel = self.communication.build_network()  # TODO use the target comm channel
        self.force_communication = config.force_communication

        self.track_targets(tau=1.0)

        """Build optimizers"""
        self.actor_a_optim = torch.optim.Adam(self.actor_a.parameters(), lr=config.lr)
        self.actor_b_optim = torch.optim.Adam(self.actor_b.parameters(), lr=config.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.classifier_optim = torch.optim.Adam(self.classifier.parameters(), lr=config.classifier_lr)
        self.communication_channel_optim = torch.optim.Adam(self.communication_channel.parameters(), lr=config.lr)

        """Build the buffers"""
        self.buffer = ReplayBufferMultiagentATOC(buffer_size=config.buffer_size,
                                                 input_sizes=self.num_agents * [self.input_size],
                                                 action_sizes=self.num_agents * [self.output_size],
                                                 _run=_run)

        self.classifier_batch_size = config.classifier_batch_size
        self.classifier_buffer_size = config.classifier_buffer_size
        self.classifier_lr = config.classifier_lr
        self.classifier_buffer = ATOCClassifierBuffer(buffer_size=self.classifier_buffer_size,
                                                      thought_size=self.thought_size)

        """Criterion"""
        self.criterion = torch.nn.SmoothL1Loss()
        self.classifier_criterion = torch.nn.BCELoss()

        """Agents"""
        self.agents = [
            ATOCAgent(num_actions=self.output_size, config=config, agent_id=aid) for aid in range(self.num_agents)
        ]

        self.num_learns = 0
        self.last_critic_loss = 0
        self.last_actor_loss = 0
        self.last_classifier_loss = 0

    def build_networks(self, config: Namespace, _run: int, observation_space) \
            -> Tuple[NetworkBase, NetworkBase, int, NetworkBase, SimpleFF]:
        # both parts of the actor
        actor_a, actor_b, thought_size = self.build_actor_nets(
            input_size=self.input_size,
            num_actions=self.output_size,
            config=config
        )

        # standard critic network
        critic = DDPGPolicy.build_critic(input_size=self.input_size + self.output_size,
                                         network=config.network,
                                         hidden_sizes=config.atoc_critic_sizes)

        # binary classifier for deciding the communication
        classifier = SimpleFF(input_size=thought_size,
                              output_size=1,
                              hidden_sizes=config.classifier_hidden_sizes,
                              output_activation='sigmoid').to(my_device())

        return actor_a, actor_b, thought_size, critic, classifier

    @staticmethod
    def read_io_size(observation_space: Union[List, gym.Space], action_space: Union[gym.Space, List]) \
            -> Tuple[int, int]:
        """ Checks correctness of the input/output spaces.

        Returns: input size, total output size
        """
        assert isinstance(observation_space, List), 'for now, only multi-agent environments supported'
        assert isinstance(observation_space[0], gym.spaces.Box), 'only continuous observations supported'

        assert isinstance(action_space, List), 'for now, only multi-agent environments supported'
        assert isinstance(action_space[0], gym.spaces.Box) or \
               (isinstance(action_space[0], gym.spaces.Tuple) and isinstance(action_space[0][0], gym.spaces.Box)), \
            'only continuous actions supported'

        assert len(observation_space) == len(action_space), 'incompatible length of action and observation spaces'

        for obs_space, act_space in zip(observation_space[1:], action_space[1:]):
            assert obs_space == observation_space[0]
            assert act_space == action_space[0]

        return get_gym_box_dimensionality(observation_space[0]), get_total_space_size(action_space[0])

    @staticmethod
    def build_actor_nets(input_size: int,
                         num_actions: int,
                         config: Namespace) -> Tuple[NetworkBase, NetworkBase, int]:
        """ Build two actor networks that are connected into series
        """

        hidden_sizes = config.hidden_sizes
        assert len(hidden_sizes) >= 4 and len(hidden_sizes) % 2 == 0, \
            'hidden sizes should be at least 4 and should be divisible by two (actor_a and actor_b)'

        hidden_a = hidden_sizes[:len(hidden_sizes) // 2 - 1]
        thought_size = hidden_sizes[len(hidden_sizes) // 2 - 1]
        hidden_b = hidden_sizes[len(hidden_sizes) // 2 + 1:]

        network = locate(config.network)

        actor_a = network(input_size=input_size,
                          output_size=thought_size,
                          hidden_sizes=hidden_a,
                          output_activation=config.actor_a_output_act,  # activation on the thought vector h_t
                          output_rescale=None).to(my_device())

        actor_b = network(input_size=thought_size * 2,  # assuming the integrated thought is stacked here
                          output_size=num_actions,
                          hidden_sizes=hidden_b,
                          output_activation=config.output_activation,
                          output_rescale=config.output_rescale).to(my_device())

        return actor_a, actor_b, thought_size

    def _env_observation_to_tensor(self, env_observation: List[np.ndarray]) -> torch.Tensor:
        """Get the observation from the environment, convert to torch.Tensor.

        Returns: tensor of sizes [batch_size=1, num_agents, input_size]
        """

        assert isinstance(env_observation, List), 'observation should be list of arrays'
        assert len(env_observation) == self.num_agents, 'the length of observation list incompatible with num agents!'
        for obs in env_observation:
            assert isinstance(obs, np.ndarray), 'list of arrays expected'
            assert obs.size == env_observation[0].size, 'inconsistent observation sizes'

        obs_array = np.concatenate(env_observation)
        result = torch.tensor(obs_array, dtype=torch.float32, device=my_device()).view(1, self.num_agents, -1)
        return result

    def _split_observation_tensor(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split the observation tensor to data and agent_ids
        Args:
            observation: complete observation from the environment in the tensor [batch_size, num_agents, input_size]

        Returns:
            -data for the actor: [batch_size, num_agents, input_size] (agent_ids positions zeroed)
            -observed_agent ids (for the communication): [batch_size, num_agents, num_perceivable_agents]
        """
        assert len(observation.shape) == 3
        assert observation.shape[-1] == self.input_size

        split_index = self.input_size - self.num_perceived_agents

        agent_ids = observation[:, :, split_index:].long()
        data = observation.clone()
        data[:, :, split_index:] = 0
        return data, agent_ids

    def pick_action(self, observation_input: List[np.ndarray]) -> List[np.array]:

        """ Convert the data """
        observations_tensor = self._env_observation_to_tensor(observation_input)
        observations, agent_ids = self._split_observation_tensor(observations_tensor)
        observations = observations.view(self.num_agents, self.input_size)
        agent_ids = agent_ids.view(self.num_agents, self.num_perceived_agents)

        """ actor_a -> comm -> actor_b """
        with torch.no_grad():
            thoughts = self.actor_a.forward(observations)

            # actions without the communication (either used or a baseline for the classifier)
            total_no_comm_thoughts = torch.cat([thoughts, torch.zeros_like(thoughts)], dim=-1)
            no_comm_actions = self.actor_b.forward(total_no_comm_thoughts)

            if self.disable_communication:
                self.last_comm_matrix.zero_()
                actions = no_comm_actions
                self.last_thougts = total_no_comm_thoughts
            else:
                integrated_thoughts, self.last_comm_matrix = \
                    self.communication.communicate(thoughts, agent_ids, self.last_comm_matrix)
                total_thought = torch.cat([thoughts, integrated_thoughts], dim=-1)
                actions = self.actor_b.forward(total_thought)
                self.last_thougts = total_thought

                if not self.force_communication:
                    self._compute_classifier_targets(observations, no_comm_actions, actions, thoughts)

        """ Explore """
        explored_actions = [
            agent.exploration.pick_action(actions[agent_id]) for agent_id, agent in enumerate(self.agents)
        ]

        """ Store to buffers and return actions"""
        self.last_observations = list(torch.split(observations_tensor,
                                                  split_size_or_sections=[1] * self.num_agents,
                                                  dim=1))
        self.last_actions = explored_actions
        return [action.to('cpu').numpy().reshape(-1) for action in explored_actions]

    def render(self):
        if hasattr(self, 'last_thoughts') and self.last_thoughts is not None:
            lt = self.last_thougts.to('cpu').numpy()
            print(f'last thoughts: {np.round(lt, 3)}')

    def _compute_classifier_targets(self,
                                    observations: torch.Tensor,
                                    no_comm_actions: torch.Tensor,
                                    actions: torch.Tensor,
                                    thoughts: torch.Tensor):
        """ Computes the estimated communication advantage for each of the communication initiators.
        Stores the [thought, q_detla] to the classifier buffer
        """
        # TODO remove this check after this works
        # for agent in range(self.num_agents):
        #     if self.last_comm_matrix[agent, agent] != 0:
        #         print(f'WARNING: agents should not communicate with themselves')

        current_initiators = torch.sum(self.last_comm_matrix, dim=1).nonzero().view(-1)

        # compose the critic inputs and evaluate q(s,a) with communication and without it
        with torch.no_grad():
            q_no_comm = self.target_critic.forward(torch.cat([observations, no_comm_actions], dim=-1))
            q_comm = self.target_critic.forward(torch.cat([observations, actions], dim=-1))

        # compute the [num_agents, 1] Q value advantages of communication for each agent
        q_delta = q_comm - q_no_comm
        q_delta_repeated = q_delta.view(1, self.num_agents).expand(self.num_agents, -1)
        sum_deltas = torch.sum(q_delta_repeated * self.last_comm_matrix.float(), dim=1)  # sum deltas for everyone

        # store delta for each of the initiators to the buffer
        for initiator_id in current_initiators:
            thought = thoughts[initiator_id]
            delta = sum_deltas[initiator_id]
            self.classifier_buffer.remember(thought, delta)

    def _numpy_to_tensors(self, data: List[np.ndarray]) -> List[torch.Tensor]:
        return [
            torch.tensor(d.flatten(), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(my_device()) \
            for d in data
        ]

    def remember(self, new_observation: List[np.ndarray], reward: List[float], done: List[bool]):
        # TODO collect data for the classifier

        assert isinstance(new_observation, List)
        new_observations = self._numpy_to_tensors(new_observation)

        self.buffer.remember_atoc(
            states=self.last_observations,
            actions=self.last_actions,
            new_states=new_observations,
            rewards=reward,
            dones=done,
            comm_matrix=self.last_comm_matrix
        )

        # agents remember just own (average) rewards
        for agent, rew in zip(self.agents, reward):
            agent.append_reward(rew)

    def _batch_to_tensors(self, batch: List[ATOCMultiagentTransition])\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Get the batch and convert to expected format for learning, usually: [batch_size, num_agents, data_size] for now

        Returns: states, agent_ids, actions, new_states, new_agent_ids, rewards, comm_matrix_batch
        """
        assert isinstance(batch, List)
        assert isinstance(batch[0], ATOCMultiagentTransition)
        assert isinstance(batch[0].action, List)
        assert isinstance(batch[0].state[0], torch.Tensor)

        # Create a list of all states, then stack and reshape accordingly
        """ States, new states, actions, and rewards """
        states = []
        new_states = []
        actions = []
        rewards = []

        for transition in batch:
            states.extend(transition.state)
            new_states.extend(transition.new_state)
            actions.extend(transition.action)
            rewards.extend(transition.reward)

        # Stack all the data we need
        complete_observations = torch.stack(states).view(self.batch_size, self.num_agents, self.input_size)
        new_complete_observations = torch.stack(new_states).view(self.batch_size, self.num_agents, self.input_size)
        actions = torch.stack(actions).view(self.batch_size, self.num_agents, self.output_size)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=my_device()).view(self.batch_size, self.num_agents, 1)

        states, agent_ids = self._split_observation_tensor(complete_observations)
        new_states, new_agent_ids = self._split_observation_tensor(new_complete_observations)

        comm_matrix_batch = torch.stack([transition.comm_matrix for transition in batch])
        return states, agent_ids, actions, new_states, new_agent_ids, rewards, comm_matrix_batch

    def _update_network(self, batch: List[ATOCMultiagentTransition]):

        """ Convert batch to expected tensor format"""
        states, agent_ids, actions, new_states, new_agent_ids, rewards, comm_matrix = \
            self._batch_to_tensors(batch)

        self.reset()  # no batch size here

        """Evaluate the actor"""
        thoughts = self.actor_a.forward(states)

        if self.disable_communication:
            integrated_thoughts = torch.zeros_like(thoughts)
        else:
            integrated_thoughts = self.communication.communicate_batched(thoughts, agent_ids, comm_matrix)
        complete_thoughts = torch.cat([thoughts, integrated_thoughts], dim=-1)

        actions_actor = self.actor_b.forward(complete_thoughts)

        q_values = self.target_critic.forward(torch.cat([states, actions_actor], dim=-1))
        actor_losses = -q_values.mean() / self.num_agents

        """Evaluate the critic"""
        critic_orig_output = self.critic.forward(torch.cat([states, actions], dim=-1))

        with torch.no_grad():

            # make action from the s'
            new_thoughts = self.target_actor_a.forward(new_states)

            if self.disable_communication:
                new_integrated_thoughts = torch.zeros_like(new_thoughts)
            else:
                new_integrated_thoughts = self.communication.communicate_batched(new_thoughts, new_agent_ids, comm_matrix)
            new_complete_thoughts = torch.cat([new_thoughts, new_integrated_thoughts], dim=-1)

            new_actions = self.target_actor_b.forward(new_complete_thoughts)

            # compute the Q(s',a')
            new_q_values = self.target_critic.forward(torch.cat([new_states, new_actions], dim=-1))

        critic_targets = rewards + self.gamma * new_q_values
        critic_losses = self.criterion(critic_orig_output, critic_targets).mean() / self.num_agents

        """Update actors and comm. channel if used"""
        # actor training here
        self.actor_a.zero_grad()
        if not self.disable_communication:
            self.communication_channel.zero_grad()
        self.actor_b.zero_grad()
        actor_losses.backward()
        self.actor_a_optim.step()
        if not self.disable_communication:
            self.communication_channel_optim.step()
        self.actor_b_optim.step()

        """Update the critic"""
        self.critic.zero_grad()
        critic_losses.backward()
        self.critic_optim.step()

        """Update the classifier"""
        if not self.disable_communication and not self.force_communication:
            thoughts, deltas = self.classifier_buffer.sample_normalized_batch(self.classifier_batch_size)
            outputs = self.classifier.forward(thoughts)
            classifier_loss = self.classifier_criterion(outputs, deltas)
            self.classifier.zero_grad()
            classifier_loss.backward()
            self.classifier_optim.step()
        else:
            classifier_loss = torch.zeros(1, device=my_device())

        """Pos-training"""
        self.num_learns += 1
        self.last_actor_loss = actor_losses.item()
        self.last_critic_loss = critic_losses.item()
        self.last_classifier_loss = classifier_loss.item()
        self.track_targets(self.tau)

    def learn(self, batch_size: Optional[int] = None):
        batch_s = self.batch_size if batch_size is None else batch_size

        if self.buffer.num_items < batch_s:
            return

        for iteration in range(self.num_learning_iterations):
            batch = self.buffer.sample(batch_s)
            self._update_network(batch)

    def reset(self, batch_size: int = 1):
        self.actor_a.reset(batch_size)
        self.actor_b.reset(batch_size)

        self.target_actor_a.reset(batch_size)
        self.target_actor_b.reset(batch_size)

        self.critic.reset(batch_size)
        self.target_critic.reset(batch_size)

        self.classifier.reset(batch_size)
        self.target_classifier.reset(batch_size)

        self.communication_channel.reset(batch_size)
        self.target_communication_channel.reset(batch_size)

        self.communication.reset()

    def set_epsilon(self, epsilon: float):
        for agent in self.agents:
            agent.set_epsilon(epsilon)

    @property
    def epsilon(self) -> float:
        return self.agents[0].epsilon

    def log(self, step: int):

        if self._run is not None:
            # common for all policies
            self._run.log_scalar(f'{self.run_id} exploration', self.agents[0].exploration.st_dev)
            self._run.log_scalar(f'{self.run_id} buff_num_written', self.buffer.num_total_written_items)
            self._run.log_scalar(f'{self.run_id} buff_num_deleted', self.buffer.num_deleted_items)

            self._run.log_scalar(f'{self.run_id} class_b_num_written', self.classifier_buffer.num_total_written_items)
            self._run.log_scalar(f'{self.run_id} class_b_num_deleted', self.classifier_buffer.num_deleted_items)

            self._run.log_scalar(f'{self.run_id} step', step)
            self._run.log_scalar(f'{self.run_id} num_learns', self.num_learns)

            self._run.log_scalar(f'{self.run_id} actor_loss', self.last_actor_loss)
            self._run.log_scalar(f'{self.run_id} critic_loss', self.last_critic_loss)
            self._run.log_scalar(f'{self.run_id} classifier_loss', self.last_classifier_loss)
            total_avg_rew = 0

            # separate for each policy
            for agent_id, agent in enumerate(self.agents):
                avg_reward = agent.get_avg_reward()
                self._run.log_scalar(f'{self.run_id} ag._{agent_id} avg_reward', avg_reward)
                total_avg_rew += avg_reward

            total_avg_rew /= len(self.agents)
            self._run.log_scalar(f'{self.run_id} total_avg_reward', total_avg_rew)

    def track_targets(self, tau: float):
        self.track_network_weights(self.actor_a, self.target_actor_a, tau)
        self.track_network_weights(self.actor_b, self.target_actor_b, tau)
        self.track_network_weights(self.critic, self.target_critic, tau)
        self.track_network_weights(self.classifier, self.target_classifier, tau)
        self.track_network_weights(self.communication_channel, self.target_communication_channel, tau)

    def serialize(self) -> Dict[str, object]:
        result = {
            'actor_a': self.actor_a.state_dict(),
            'actor_b': self.actor_b.state_dict(),
            'critic': self.critic.state_dict(),
            'classifier': self.classifier.state_dict(),
            'communication_channel': self.communication_channel.state_dict(),

            'actor_a_optim': self.actor_a_optim.state_dict(),
            'actor_b_optim': self.actor_b_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'classifier_optim': self.classifier_optim.state_dict(),
            'communication_channel_optim': self.communication_channel_optim.state_dict(),

            'target_actor_a': self.target_actor_a.state_dict(),
            'target_actor_b': self.target_actor_b.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'target_classifier': self.target_classifier.state_dict(),
            'target_communication_channel': self.target_communication_channel.state_dict(),
        }
        return result

    def deserialize(self, data: Dict[str, object]):
        data: Dict[str, Dict[str, torch.Tensor]]

        self.actor_a.load_state_dict(data['actor_a'])
        self.actor_b.load_state_dict(data['actor_b'])
        self.critic.load_state_dict(data['critic'])
        self.classifier.load_state_dict(data['classifier'])
        self.communication_channel.load_state_dict(data['communication_channel'])

        self.target_actor_a.load_state_dict(data['actor_a'])
        self.target_actor_b.load_state_dict(data['actor_b'])
        self.target_critic.load_state_dict(data['critic'])
        self.target_classifier.load_state_dict(data['classifier'])
        self.target_communication_channel.load_state_dict(data['communication_channel'])

        self.actor_a_optim.load_state_dict(data['actor_a_optim'])
        self.actor_b_optim.load_state_dict(data['actor_b_optim'])
        self.critic_optim.load_state_dict(data['critic_optim'])
        self.classifier_optim.load_state_dict(data['classifier_optim'])
        self.communication_channel_optim.load_state_dict(data['communication_channel_optim'])

        print(f'ATOC agents deserialized correctly')
