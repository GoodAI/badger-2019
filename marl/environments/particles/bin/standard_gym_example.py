from time import sleep

import gym
from make_env import make_env
import marl.environments.particles.multiagent.entry_point


def random_actions(action_spaces):
    chosen_actions = []

    for space in action_spaces:
        action = space.sample()

        num_actions = space.n
        act = [0] * num_actions
        act[action] = 1
        chosen_actions.append(act)

    return chosen_actions


if __name__ == '__main__':
    # env = gym.make('MountainCar-v0')  # normal gym env
    env = gym.make('MultiagentCustom-v0')
    # env = make_env('simple')  # an alternative, environment-specific, initialization

    t = 0
    num_actions = env.action_space[0].n

    while True:
        t += 1
        env.render()
        observation = env.reset()
        print(observation)
        chosen = random_actions(env.action_space)
        sleep(0.1)

        observation, reward, done, info = env.step(chosen)
        if done[0]:
            print("Episode finished after {} timesteps".format(t + 1))
            break

    env.close()

