from time import sleep
from typing import List

import torch

from agents.emergent_language import configs
from agents.emergent_language.train import get_parser
from multiagent import scenarios
from multiagent import MultiAgentEnv
from multiagent import A2CPolicy

import os
import datetime

import pyformulas as pf
import matplotlib.pyplot as plt
import numpy as np


def save_fig():
    if not os.path.exists('data'):
        os.makedirs('data')
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H-%M-%S")
    plt.savefig(os.path.join('data', 'loss_'+timestamp+'.png'))


class LossPrinter:

    def __init__(self):
        self.fig = plt.figure()
        canvas = np.zeros((480, 640))
        self.screen = pf.screen(canvas, 'Running loss')

    def show_running_loss(self, losses: List[float]):
        plt.ylim(min(losses), max(losses))
        plt.xlim(0, len(losses))
        plt.plot(losses, c='black')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Running loss')

        self.fig.canvas.draw()
        image = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        self.screen.update(image)

    @staticmethod
    def save():
        save_fig()

    def close_fig(self):
        self.screen.close()


def print_losses(losses: List[float]):
    """Print losses in a non-blocking way"""

    plt.figure()
    plt.plot(list(range(len(losses))), losses)

    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.title('loss during training')

    save_fig()
    plt.ion()
    plt.show()
    plt.pause(0.001)


def show_behavior(env, policies, num_epochs, time_horizon):
    """ Showcase what it learned
    """
    for epoch in range(num_epochs):

        # randomly place the agent(s)
        obs_n = env.reset()

        for step in range(time_horizon):
            # collect actions from the agents
            act_n = []
            for i, policy in enumerate(policies):
                act_n.append(policy.action(obs_n[i]))

            # step environment
            obs_n, reward_n, done_n, _ = env.step(act_n)
            env.render()
            sleep(0.05)
            print(f'step: {step}, reward: {reward_n[0]}')


def main():

    parser = get_parser()
    args = vars(parser.parse_args())
    agent_config = configs.get_agent_config(args)
    game_config = configs.get_game_config(args)
    training_config = configs.get_training_config(args)

    print(f'\n training_conf: \t{training_config}')
    print(f'\n game conf: \t\t{game_config}')
    print(f'\n agent conf: \t\t{agent_config}\n')

    # an agent composed of modules (processing, goal_predicting, word_counting, action)
    # agent = AgentAdapted(agent_config)
    # agent = MyAgent(agent_config)

    scenario = scenarios.load('custom/custom_no_comm.py').Scenario()
    scenario.setup(num_agents=1, num_landmarks=1, are_positions_fixed=False, scale=0.5, only_x=True)

    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                        info_callback=None, shared_viewer=False)

    if training_config.render:
        env.render()
        sleep(1)

    # TODO is this correct?
    observation_size = env.observation_space[0].shape[0]
    goal_size = 0

    # create the policy
    policies = [A2CPolicy(env,
                          observation_size=observation_size,
                          goal_size=goal_size,
                          lr=training_config.learning_rate,
                          gamma=0)]

    load = False
    if load:
        print("loading model")
        policies[0] = torch.load('model.pkl')

    # create interactive policies for each agent
    # policies = [RandomPolicy(env, i) for i in range(env.n)]

    losses = []
    running_rewards = []
    running_reward = 0

    runs_per_epoch = 512

    printer = LossPrinter()

    # execution loop
    for epoch in range(training_config.num_epochs):

        for run in range(runs_per_epoch):

            ep_reward = 0
            # randomly place the agent(s)
            obs_n = env.reset()

            for step in range(agent_config.time_horizon):

                # collect actions from the agents
                act_n = []
                for i, policy in enumerate(policies):
                    act_n.append(policy.action(obs_n[i]))

                # step environment
                obs_n, reward_n, done_n, _ = env.step(act_n)

                # stats
                ep_reward += reward_n[0]
                running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
                running_rewards.append(running_reward)

                # update the replay memories
                for policy, obs, reward, done in zip(policies, obs_n, reward_n, done_n):
                    # rew = reward if step == agent_config.time_horizon - 2 else 0 # rew only the last step
                    rew = reward
                    policy.remember(new_state=obs, reward=rew, done=done)

                if training_config.render:
                    env.render()

        for policy_id, policy in enumerate(policies):
            loss = policy.finish_episode()
            if policy_id == 0:
                losses.append(loss)
                print(f'learning, epoch no: {len(losses)}, loss: {losses[-1]}, \trunning_r: {running_reward}')
                printer.show_running_loss(losses)

    # torch.save(policies[0], 'model.pkl')
    printer.save()
    # policies = [RandomPolicy(env, i) for i in range(env.n)]
    # print_losses(losses)
    show_behavior(env, policies, 500, agent_config.time_horizon)
    # plt.show()


if __name__ == '__main__':
    main()
