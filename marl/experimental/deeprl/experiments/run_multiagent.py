import argparse
from pydoc import locate
import json

from marl.experimental.deeprl.experiments.simulate_multiagent import get_env_multiagent, simulate_multiagent


def parse_args():
    parser = argparse.ArgumentParser(description="Configure the environment with a random policy")
    parser.add_argument('--sleep', type=float, default=0.0, help='sleep between steps')
    parser.add_argument('--type', type=str, default='DDPG', help='Which algorithm to run [DDPG|GLOBAL|ATOC]')
    parser.add_argument('--epsilon', type=float, default=0.0, help='randomization of the policy, -1 means no change!')
    parser.add_argument('--deserialize', type=str, default="", help='saved weights for the policy')
    parser.add_argument('--num-agents', type=int, default=5, help="How many agents?")
    parser.add_argument('--num-landmarks', type=int, default=5, help="How many landmarks?")
    parser.add_argument('--num_perceived_agents', type=int, default=3, help="Number of nearest agents that an agent can see.")
    parser.add_argument('--num_perceived_landmarks', type=int, default=3, help="Number of nearest landmarks that an agent can see.")
    parser.add_argument("--name", type=str, default="", help="file to where you want the model weights saved")
    parser.add_argument("--benchmark", action='store_true', help="run the environment in benchmark mode to collect stats")
    parser.add_argument("--render", action='store_true', help="Show the environment running")
    parser.add_argument("--num-steps", type=int, default=100000, help="Number of env steps to learn/benchmark")
    return parser.parse_args()


if __name__ == "__main__":

    # parse arguments, get the core configuration
    args = parse_args()

    json_loc = "marl/experimental/deeprl/experiments/data/json/"

    if args.type == 'DDPG':
        json_loc += "ddpg.json"
    elif args.type == 'GLOBAL':
        json_loc += "global.json"
    elif args.type == 'ATOC':
        json_loc += "atoc.json"
    else:
        raise ValueError(f'Algorithm type not found! Is {args.type}, should be in [DDPG|GLOBAL|ATOC]')

    # obtain the core config
    conf_data = json.load(open(json_loc, 'r'))
    config = argparse.Namespace(**conf_data)

    # and update it with the params
    config.epsilon = args.epsilon
    config.num_agents = args.num_agents
    config.num_landmarks = args.num_landmarks
    config.num_perceived_agents = args.num_perceived_agents
    config.num_perceived_landmarks = args.num_perceived_landmarks
    config.render = args.render
    config.num_steps = args.num_steps

    # build the experiment
    env = get_env_multiagent(config, benchmark=args.benchmark)
    policy = locate(config.policy)(env.observation_space, env.action_space, config)

    # deserialize the model parameters if any
    if args.deserialize != "":
        policy.load(args.deserialize)

    # test time
    if args.epsilon != -1.0:
        policy.set_epsilon(args.epsilon)

    # run it
    simulate_multiagent(env=env,
                        policy=policy,
                        num_steps=config.num_steps,
                        learning_period=config.learning_period,
                        num_logs=config.num_logs,
                        num_serializations=config.num_serializations,
                        render=config.render,
                        sleep=config.sleep,
                        name=args.name,
                        benchmark=args.benchmark)
