import os
import pickle
import time
from argparse import Namespace
from pydoc import locate
from typing import Optional, List

import gym
import numpy as np
from badger_utils.sacred import SacredWriter
from tqdm import tqdm

from marl.environments.particles.make_env import make_env_configured_conf
from marl.experimental.deeprl.policies.policy_base import PolicyBase
from marl.experimental.deeprl.utils.available_device import my_device, choose_device
from marl.experimental.deeprl.utils.replay_buffer import ReplayBuffer


def all_done(done: List[bool]) -> bool:
    for d in done:
        if not d:
            return False
    return True


def simulate_multiagent(env: gym.Env,
                        policy: PolicyBase,
                        num_steps: int,
                        learning_period: Optional[int] = None,
                        num_logs: Optional[int] = None,  # number of data points on X during the num_steps
                        num_serializations: Optional[int] = None,  # number of serialized models during the num_steps
                        render: bool = False,
                        sleep: float = 0.0,
                        precision: int = 3,
                        name: str = "",
                        benchmark: bool = False):
    """Used both for training and rendering"""
    episode = 0
    done = [True]
    observation = None

    # compute how often to log/serialize
    logging_period = np.floor(num_steps / num_logs) if num_logs is not None else None
    serialization_period = np.floor(num_steps / num_serializations) if num_serializations is not None else None

    if render:  # pybullet compatibility (render before reset), other environments need reset before render..
        np.set_printoptions(precision=precision)
        try:
            env.render()
        except:
            pass

    b_rewards = []
    b_collisions = []
    b_min_dists = []
    b_occupied = []

    all_step_rewards = []

    if name == "":
        datetime = time.strftime("%y-%m-%d_%H%M", time.localtime())
        f_name = f"Run_{datetime}"
    else:
        f_name = name

    for step in tqdm(range(num_steps)):

        # reset between episodes
        if all_done(done):

            if benchmark and len(b_rewards) > 0:
                bench_name = f"data/stats/{f_name}.bench"
                # Save the benchmark stats
                with open(bench_name, 'ab') as f:
                    pickle.dump((b_rewards, b_collisions, b_min_dists, b_occupied), f)

            b_rewards = []
            b_collisions = []
            b_min_dists = []
            b_occupied = []

            episode += 1
            observation = env.reset()
            policy.reset()
            if render:
                print(f'\n xxxxxxxxxxxxxxxxxx reset after episode {episode}!\n')

        # The main simulation loop
        action = policy.pick_action(observation)
        observation, reward, done, info = env.step(action)
        policy.remember(observation, reward, done)
        all_step_rewards.append(reward[0])

        if benchmark:
            # Store the info here
            info = info['n'][0]
            b_rewards.append(info[0])
            b_collisions.append(info[1])
            b_min_dists.append(info[2])
            b_occupied.append(info[3])

        if render:
            for agent_id, (a_obs, a_rew, a_act, a_done) in enumerate(zip(observation, reward, action, done)):
                r = ReplayBuffer.sanitize_reward(a_rew if a_rew is not None else 0)
                print(f'AGENT: {agent_id}: '
                      f'OBS: {np.round(a_obs.reshape(-1), precision)},\t'
                      f'A: {np.round(a_act, precision)},\t'
                      f'R: {r},\t'
                      f'D: {a_done}')
            if hasattr(policy, 'render'):
                policy.render()
            print(f'\n')
            env.render()
            if sleep > 0.0:
                time.sleep(sleep)

        if learning_period is not None and step % learning_period == 0:
            policy.learn()

        if logging_period is not None and step % logging_period == 0:
            policy.log(step)
            if not os.path.exists('data/stats/'):
                os.makedirs('data/stats')
            with open(f"data/stats/{f_name}.rewards", "a") as f:
                f.write(str(all_step_rewards)[1:-1] + ", ")
                all_step_rewards = []

        if step == num_steps - 1 or \
                (serialization_period is not None and name != "" and step % serialization_period == 0):
            print(f'Step: {step}; model serialization now, device is: {my_device()}')
            policy.save(name=name, epoch=step)

    # Write the reward info:


def get_env_multiagent(conf: Namespace, benchmark=False):

    return make_env_configured_conf(conf.scenario_name, conf, benchmark=benchmark)


def deserialize_policy(policy: PolicyBase, name: str):
    """Deserialize the weights of a given policy from a given exp_id (the last epoch now)."""

    policy.load(name)
    print(f'Policy {name} deserialized')


def run_experiment_multiagent(_run, _config, sacred_writer: SacredWriter):
    """
    Runs the experiment in sacred, multiagent setting
    """

    conf = Namespace(**_config)  # parse the dict into a Namespace
    choose_device(conf)

    for run_id in range(conf.num_runs):
        env = get_env_multiagent(conf)

        policy = locate(conf.policy)(env.observation_space,
                                     env.action_space,
                                     conf,
                                     _run=_run,
                                     run_id=run_id)

        if conf.deserialize_from is not None:
            deserialize_policy(policy, conf.deserialize_from)

        # training
        simulate_multiagent(env=env,
                            policy=policy,
                            num_steps=conf.num_steps,
                            learning_period=conf.learning_period,
                            num_logs=conf.num_logs,
                            num_serializations=conf.num_serializations,
                            render=conf.render,
                            sleep=conf.sleep)
