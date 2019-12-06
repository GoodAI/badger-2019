import os
import time
from typing import List, Optional

import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from marl.environments.simple_envs.envs.simple_env_base import SimpleEnvBase
from marl.experimental.deeprl.policies.networks import my_device
from marl.experimental.deeprl.policies.policy_base import PolicyBase


def _collect_action_values(step: float,
                           env_size: int,
                           policy: PolicyBase,
                           num_actions: int,
                           one_hot_obs: bool) -> [List[List[float]], List[float]]:
    """ Collect values for states in range <0, env_size) with step size step

    Args:
        step:
        env_size:
        policy:
        num_actions:
    Returns: for each action, list of Q(s,a) values; list of states
    """

    # for each action, there is a list of Q(s,a) values, one for every state
    q_values = [[] for _ in range(num_actions)]

    states = []

    state = 0.0

    while state < env_size:
        with torch.no_grad():
            # TODO using RNN in a non-sequential manner, not a good idea
            state_tensor = _make_state(state, one_hot_obs, env_size)
            values = policy.network.forward(state_tensor)

        for action in range(num_actions):
            q_values[action].append(values[0, 0, action].item())

        states.append(state)
        state += step

    return q_values, states


def _make_state(position: float, one_hot_obs, size: int) -> torch.tensor:
    if one_hot_obs:
        state = SimpleEnvBase.to_one_hot(np.math.floor(position), size)
    else:
        state = np.array(position).reshape(1)
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(my_device())
    return state


def has_one_hot_obs(env: gym.Env):
    if not hasattr(env, 'one_hot_obs'):
        return False
    return env.one_hot_obs


def visualize(policy: PolicyBase, env: gym.Env, show: Optional[bool] = True):
    """ Visualize the dependency of Q(s,a) values on the state
    """

    assert hasattr(env, 'env_size'), 'environment has no attribute env_size'
    one_hot = has_one_hot_obs(env)
    env_size = env.env_size
    num_actions = env.action_space.n
    plt.figure()

    # collect and plot the Q(s,a) values
    q_values, states = _collect_action_values(1, env_size, policy, num_actions, one_hot)

    for id, action_values in enumerate(q_values):
        dir = 'left' if id == 0 else 'right'
        plt.plot(states, action_values, label=f'action {id} {dir}')

    # plot the same, but with smaller steps (NN not taught on these states, should still work)
    q_values, states = _collect_action_values(0.05, env_size, policy, num_actions, one_hot)
    for id, action_values in enumerate(q_values):
        dir = 'left' if id == 0 else 'right'
        plt.plot(states, action_values, ':', label=f'.. action {id} {dir}')

    plt.legend()
    plt.title('Dependency of Q(s,a) values on state')
    plt.ylabel('Q(s,a)')
    plt.xlabel('states')
    plt.grid()

    # save
    timestr = time.strftime("%Y-%m-%d--%H-%M-%S")
    if not os.path.exists('data'):
        os.mkdir('data')
    plt.savefig(f'data/Qsa_{timestr}.svg', format='svg')

    if show:
        plt.show()


