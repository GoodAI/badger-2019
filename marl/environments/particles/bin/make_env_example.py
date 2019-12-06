import argparse
from time import sleep
import numpy as np
import gym
# from bin.standard_gym_example import random_actions
from marl.environments.particles.make_env import make_env_configured
# import marl.environments.particles.multiagent.entry_point


def parse_args():
    parser = argparse.ArgumentParser(description="Configure the environment with a random policy")
    parser.add_argument('--num-agents', type=int, default=1, help='num gents')
    parser.add_argument('--num-landmarks', type=int, default=1, help='num landmarks')
    # parser.add_argument('--single-agent-compatible', action='store_true',
    #                     help='if True, the environment behaves as a normal single-agent one (no arrays..)')

    dictionary = vars(parser.parse_args())
    return dictionary


# def random_actions()

def random_discrete_actions(action_spaces):
    chosen_actions = []

    for space in action_spaces:
        action = space.sample()

        num_actions = space.n
        act = [0] * num_actions
        act[action] = 1
        chosen_actions.append(np.array(act))

    return chosen_actions


def random_continuous_actions(action_size: int, num_agents: int):
    """TODO"""
    result = []
    for agent in range(num_agents):
        result.append(np.random.uniform(-1, 1, size=action_size))
    return result


if __name__ == '__main__':
    args = parse_args()
    args['num_agents'] = 4
    args['num_landmarks'] = 3
    args['add_obstacles'] = False
    args['episode_length'] = 100
    # env = make_env_configured(scenario_name='custom/custom_no_comm', kwargs=args, shared_viewer=False)
    env = make_env_configured(scenario_name='custom/custom_no_comm_goals',
                              kwargs=args, shared_viewer=True)  # shared & egocentric viewer

    t = 0
    num_actions = env.action_space[0].n
    observation = env.reset()

    while True:
        t += 1
        env.render()
        print(observation)
        chosen = random_discrete_actions(env.action_space)
        sleep(0.05)

        observation, reward, done, info = env.step(chosen)
        if done[0]:
            print("Episode finished after {} timesteps".format(t + 1))
            observation = env.reset()

    env.close()

