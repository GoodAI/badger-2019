#!/usr/bin/env python
import os
import sys
import time

import numpy as np

# from multiagent import InteractivePolicy
from marl.environments.particles.multiagent.entry_point.environment import MultiAgentEnv
from marl.environments.particles.multiagent.policies.interactive_policy import InteractivePolicy

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
# from multiagent import MultiAgentEnv
from marl.environments.particles.multiagent import scenarios as scenarios

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    scenario.setup(num_landmarks=2, num_agents=1, data_scale=1, show_abs_pos=False)
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                        shared_viewer=True)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env, i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    while True:
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        # time.sleep(0.1)
        # render all agent views
        env.render()
        # print("hello")
        # display rewards
        for agent in env.world.agents:
            print(agent.name + " reward: %0.3f" % env._get_reward(agent))

            print(f'{agent.name} observation:'
                  f'{np.array2string(env._get_obs(agent), precision=3, suppress_small=True)}')


