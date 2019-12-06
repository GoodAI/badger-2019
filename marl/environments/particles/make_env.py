"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all emergent_comm. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""
import inspect
from argparse import Namespace

from marl.environments.particles.multiagent.scenarios.custom.configurable_scenario import has_setup_method
import numpy as np


def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    # from multiagent import MultiAgentEnv
    from marl.environments.particles.multiagent.entry_point.environment import MultiAgentEnv
    from marl.environments.particles.multiagent import scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def make_env_configured_conf(scenario_name: str, conf: Namespace, benchmark=False, shared_viewer=True):
    """Make_env_configured version which just takes the config and filters-out the relevant parts to call the setup"""
    from marl.environments.particles.multiagent import scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()

    env_size = get_env_size(conf)

    if has_setup_method(scenario):
        specs = inspect.getfullargspec(scenario.setup)
        arguments = specs.args

        env_kwargs = {}
        for conf_key, conf_value in vars(conf).items():
            if conf_key in arguments:
                env_kwargs[conf_key] = conf_value

        if 'env_size' not in arguments:
            if env_size != 1.0:
                print(f'WARNING: the env_size != 1, but the env_size param is not in the scenario.setup() params')
        else:
            env_kwargs['env_size'] = env_size

        scenario.setup(**env_kwargs)

    return _make_world(scenario, benchmark, shared_viewer, camera_range=env_size)


def get_env_size(config: Namespace) -> float:
    if config.agent_density is None:
        return 1.0

    return np.sqrt(config.num_agents / config.agent_density)


def make_env_configured(scenario_name, benchmark=False, shared_viewer=True, kwargs=None):
    """
    The same as make_env, but this calls the scenario.setup(**kwargs)
    """
    from marl.environments.particles.multiagent import scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()

    if has_setup_method(scenario):
        scenario.setup(**kwargs)

    return _make_world(scenario, benchmark, shared_viewer)


def _make_world(scenario, benchmark: bool, shared_viewer: bool, camera_range: float = 1.0):
    from marl.environments.particles.multiagent.entry_point.environment import MultiAgentEnv

    # create world
    world = scenario.make_world()

    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world=world,
                            reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            done_callback=scenario.is_done,
                            info_callback=scenario.benchmark_data,
                            shared_viewer=shared_viewer,
                            camera_range=camera_range)
    else:
        env = MultiAgentEnv(world=world,
                            reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            done_callback=scenario.is_done,
                            shared_viewer=shared_viewer,
                            camera_range=camera_range)
    return env


