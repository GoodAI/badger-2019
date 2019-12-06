from torch import optim
from torch.distributions import Categorical

from multiagent import PolicyNetwork, SavedAction
from multiagent import BasePolicy
import numpy as np
import torch
import torch.nn.functional as F


class A2CPolicy(BasePolicy):
    """
    Actor-critic policy adapted from:

    https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
    """

    model: PolicyNetwork

    observation_size: int
    goal_size: int
    num_actions: int

    input_size: int
    hidden_size: int

    gamma: float
    lr: float

    # only_x: bool

    def __init__(self,
                 env,
                 observation_size: int,
                 goal_size: int,
                 agent_index: int = 0,
                 num_actions: int = 5,
                 hidden_size: int = 256,
                 gamma: float = 0.01,
                 lr: float = 0.002):
                 # only_x: bool = False):

        super(A2CPolicy, self).__init__()

        self.observation_size = observation_size
        self.goal_size = goal_size
        self.num_actions = num_actions
        self.input_size = self.observation_size + self.goal_size
        self.hidden_size = hidden_size
        self.lr = lr
        # self.only_x = only_x

        assert gamma < 1
        self.gamma = gamma

        assert agent_index == 0, 'only one agent supported for now'

        self.model = PolicyNetwork(input_size=self.input_size, hidden_size=self.hidden_size, num_actions=num_actions)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.eps = np.finfo(np.float32).eps.item()

        self.env = env

    def action(self, obs):
        action_index = self._select_action(obs, None)

        if self.env.discrete_action_input:
            raise NotImplementedError("not supported")
        else:
            # one-hot action vector, 5D because of no-move action at u[0]
            u = np.zeros(self.num_actions)  # 5-d because of no-move action
            u[action_index] = 1

        # TODO utterance is silent for now
        # stack with the empty communication channel
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

    def remember(self, new_state, reward, done):
        self.model.rewards.append(reward)
        # ep_reward += reward

    def _select_action(self, observation: np.array, goal: np.array):

        # TODO goal not used yet

        # forward though the network(s)
        state = torch.from_numpy(observation).float()
        probs, state_value = self.model(state)

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)
        # and sample an action using the distribution
        action = m.sample()

        # save to action buffer and return action
        self.model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()

    def finish_episode(self) -> float:
        """
        Training code. Calculates actor and critic loss and performs backprop.
        """
        R = 0
        saved_actions = self.model.saved_actions
        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss
        returns = []  # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.model.rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            # abs advantage added, according to the: https://danieltakeshi.github.io/assets/ml/vpg.png
            policy_losses.append(-log_prob * torch.abs(advantage))

            if policy_losses[-1] < 0:
                print(f'WARNING: policy: -log_prob: {-log_prob}, advantage: {advantage}')

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

            if value_losses[-1] < 0:
                print(f'WARNING: value: value {value}, R {R}')

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        self.optimizer.step()
        self.model.reset()

        # reset rewards and action buffer
        del self.model.rewards[:]
        del self.model.saved_actions[:]

        return loss.item()

