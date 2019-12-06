from argparse import Namespace
from pydoc import locate
from typing import List, Union, Optional, Tuple, Dict

import gym
import numpy as np
import torch

from marl.experimental.deeprl.exploration.exploration_ou import ExplorationOu
from marl.experimental.deeprl.policies.networks import NetworkBase, my_device
from marl.experimental.deeprl.policies.policy_base import PolicyBase
from marl.experimental.deeprl.utils.replay_buffer import ReplayBuffer, Transition
from marl.experimental.deeprl.utils.utils import get_total_space_size


class DDPGPolicy(PolicyBase):
    """
    Continuous action spaces.
    """
    input_size: int
    num_actions: int
    gamma: float
    total_steps: Optional[int]

    actor: NetworkBase
    critic: NetworkBase
    target_actor: Optional[NetworkBase]
    target_critic: Optional[NetworkBase]

    exploration: ExplorationOu

    buffer: ReplayBuffer

    last_observation: torch.tensor
    last_action: torch.tensor

    num_learns: int
    num_learning_iterations: int

    collected_rewards: List[float]

    batch_size: int

    last_actor_loss: float
    last_critic_loss: float

    def __init__(self,
                 observation_space: Union[gym.Space, gym.spaces.Tuple],
                 action_space: Union[gym.Space, gym.spaces.Tuple],
                 config: Namespace,
                 _run=None,
                 run_id: int = 0):  # TODO remove run id?
        super().__init__()

        self._run = _run
        self.run_id = run_id

        assert config.batch_size > 0
        self.batch_size = config.batch_size

        self.num_actions = get_total_space_size(action_space)
        assert self.num_actions > 0, 'invalid num actions'

        self.input_size = get_total_space_size(observation_space)
        assert self.input_size > 0, 'invalid input size'

        assert 0 <= config.gamma <= 1, 'gamma out of range'
        self.gamma = config.gamma

        assert config.hidden_sizes is None or \
               isinstance(config.hidden_sizes, int) and config.hidden_sizes > 0 or \
               len(config.hidden_sizes) > 0, 'hidden size is None or list of sizes'

        self.tau = config.tau

        assert config.num_learning_iterations > 0
        self.num_learning_iterations = config.num_learning_iterations

        self.num_learns = 0

        self.actor, self.critic = self._build_actor_critic(config)

        if self.is_tracking_used:
            self.target_actor, self.target_critic = self._build_actor_critic(config)
            self.track_targets(tau=1.0)  # copy the params
        else:
            self.target_actor = None
            self.target_critic = None

        exploration_class = locate(config.exploration)
        self.exploration = exploration_class(action_dim=self.num_actions, config=config)

        self.buffer = ReplayBuffer(config.buffer_size, self.input_size, self.num_actions, _run)

        self.criterion = torch.nn.SmoothL1Loss()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.collected_rewards = []

        self.last_actor_loss = 0
        self.last_critic_loss = 0

    def _build_actor_critic(self,
                            config: Namespace,
                            ) -> Tuple[NetworkBase, NetworkBase]:

        actor = self.build_actor(input_size=self.input_size,
                                 num_actions=self.num_actions,
                                 config=config)

        critic = self.build_critic(input_size=self.input_size + self.num_actions,
                                   network=config.network,
                                   hidden_sizes=config.hidden_sizes)

        return actor, critic

    @staticmethod
    def build_actor(input_size: int,
                    num_actions: int,
                    config: Namespace) -> NetworkBase:

        network = locate(config.network)
        actor = network(input_size=input_size,
                        output_size=num_actions,
                        hidden_sizes=config.hidden_sizes,
                        output_activation=config.output_activation,
                        output_rescale=config.output_rescale).to(my_device())
        return actor

    @staticmethod
    def build_critic(input_size: int, network: str, hidden_sizes: Union[int, List[int]]) -> NetworkBase:

        network = locate(network)
        critic = network(input_size=input_size,
                         output_size=1,
                         hidden_sizes=hidden_sizes,
                         output_activation=None,
                         output_rescale=None,
                         softmaxed_parts=None).to(my_device())
        return critic

    def serialize(self) -> Dict[str, object]:
        result = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optim': self.actor_optimizer.state_dict(),
            'critic_optim': self.critic_optimizer.state_dict()
        }
        if self.is_tracking_used:
            result['target_actor'] = self.target_actor.state_dict()
            result['target_critic'] = self.target_critic.state_dict()

        return result

    def deserialize(self, data: Dict[str, object]):
        data: Dict[str, Dict[str, torch.Tensor]]

        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])
        self.actor_optimizer.load_state_dict(data['actor_optim'])
        self.critic_optimizer.load_state_dict(data['critic_optim'])

        if self.is_tracking_used:
            self.target_actor.load_state_dict(data['target_actor'])
            self.target_critic.load_state_dict(data['target_critic'])

        print(f'DDPG deserialized correctly')

    def track_targets(self, tau: float):
        if not self.is_tracking_used:
            if tau != 0.0:
                print(f'WARNING: tracking not used but still trying to track targets with a custom tau')
            return
        # the target tracks source (slowly)
        self.track_network_weights(self.actor, self.target_actor, tau)
        self.track_network_weights(self.critic, self.target_critic, tau)

    def pick_action(self, observation: np.ndarray):
        obs = torch.tensor(observation.flatten(), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(my_device())

        with torch.no_grad():
            actions = self.actor.forward(obs)

        action = self.exploration.pick_action(actions)

        self.last_observation = obs
        self.last_action = action
        return action.to('cpu').numpy()

    def remember(self, new_observation: np.array, reward: float, done: bool):
        """No change"""
        obs = torch.tensor(new_observation, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(my_device())

        self.buffer.remember(self.last_observation,
                             self.last_action,
                             obs,
                             reward,
                             done)

        self.append_reward(ReplayBuffer.sanitize_reward(reward))

    def set_epsilon(self, st_dev: float):
        """Set the standard deviation of the additive noise applied to actions"""
        self.exploration.set_epsilon(st_dev)

    @property
    def epsilon(self) -> float:
        return self.exploration.st_dev

    def reset(self, batch_size: int = 1):
        self.actor.reset(batch_size)
        self.critic.reset(batch_size)
        if self.is_tracking_used:
            self.actor_t.reset(batch_size)
            self.critic_t.reset(batch_size)

    def log(self, step: int):
        """ Note, logs the following:
            -average reward from the last logging
            -last critic loss
            -last actor loss
        """
        avg_reward = self.get_avg_reward()

        if self._run is not None:
            self._run.log_scalar(f'{self.run_id} actor_loss', self.last_actor_loss)
            self._run.log_scalar(f'{self.run_id} critic_loss', self.last_critic_loss)

            self._run.log_scalar(f'{self.run_id} avg_reward', avg_reward)
            self._run.log_scalar(f'{self.run_id} exploration', self.exploration.st_dev)

            self._run.log_scalar(f'{self.run_id} buff_num_written', self.buffer.num_total_written_items)
            self._run.log_scalar(f'{self.run_id} buff_num_deleted', self.buffer.num_deleted_items)

            self._run.log_scalar(f'{self.run_id} step', step)
            self._run.log_scalar(f'{self.run_id} num_learns', self.num_learns)
        else:
            # legacy
            print(f'num_learns {self.num_learns},' +
                  f'\tactor loss: {self.last_actor_loss},' +
                  f'\tcritic loss: {self.last_critic_loss},' +
                  f'\tst_dev: {self.exploration.st_dev},' +
                  f'\tavg_reward: {avg_reward}')

    def learn(self, batch_size: Optional[int] = None):
        batch_s = self.batch_size if batch_size is None else batch_size

        if self.buffer.num_items < batch_s:
            return

        for iteration in range(self.num_learning_iterations):
            batch = self.buffer.sample(batch_s)
            self._update_network(batch)

    @property
    def critic_t(self) -> NetworkBase:
        return self.target_critic if self.is_tracking_used else self.critic

    @property
    def actor_t(self) -> NetworkBase:
        return self.target_actor if self.is_tracking_used else self.actor

    def _update_network(self, batch: List[Transition]):

        assert isinstance(batch, List)
        assert isinstance(batch[0], Transition)

        tr = batch[0]
        assert isinstance(tr.state, torch.Tensor)
        assert isinstance(tr.new_state, torch.Tensor)
        assert isinstance(tr.action, torch.Tensor)

        assert len(tr.state.shape) == 3
        assert len(tr.new_state.shape) == 3
        assert len(tr.action.shape) == 3
        assert tr.action.shape[2] == self.num_actions

        all_actor_losses = torch.zeros((1, 1, 1), dtype=torch.float, device=my_device())
        all_critic_losses = torch.zeros(1, dtype=torch.float, device=my_device())

        for tr in batch:
            # TODO the done flag not used
            self.reset()

            # DDPG learning: you made a state-action-next_state transition
            #
            # q1: how good was the action?
            #   -use the critic
            #
            # q2: how good was the critic estimate?
            #   -use the actor to make next_action from next_state (we cannot get the argmax_a Q(s,a))
            #   -use the critic to evaluate that Q(next_state, next_action)
            #   -using the bellman e.q. to compute the loss based on TD between Q(state, action) and Q(next_s, next_a)

            # q1
            action = self.actor.forward(tr.state)
            action_val = self.critic_t.forward(torch.cat((tr.state, action), dim=2))
            # suppress the bad actions: higher action_val => lower loss
            # note: we are differentiating through the critic and then actor,
            #   the actor_optimizer has just the actor params, so no update of the critic here
            all_actor_losses += -action_val

            # q2:
            with torch.no_grad():
                next_action = self.actor_t.forward(tr.new_state)
                next_value = self.critic_t.forward(torch.cat((tr.new_state, next_action), dim=2))
                # note: compared to the previous case, now we used the actor and critic in the future
                #   to compute the targets for the critic.
                #   Therefore no gradients in this part

            action_val_target = tr.reward + self.gamma * next_value  # bellman equation..

            remembered_action_val = self.critic.forward(torch.cat((tr.state, tr.action), dim=2))  # compute the output

            all_critic_losses += self.criterion(remembered_action_val, action_val_target)

        all_actor_losses = all_actor_losses / self.batch_size

        # take the optimization steps
        self.actor_optimizer.zero_grad()
        all_actor_losses.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        all_critic_losses.backward()
        self.critic_optimizer.step()

        # post-training
        self.num_learns += 1
        self.last_actor_loss = all_actor_losses.item()
        self.last_critic_loss = all_critic_losses.item()
        self.track_targets(self.tau)
