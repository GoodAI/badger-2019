from time import sleep

from agents.emergent_language import configs
from agents.emergent_language.train import get_parser
from multiagent import scenarios
from multiagent import MultiAgentEnv
from multiagent import MyAgent
from multiagent import RandomPolicy


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
    agent = MyAgent(agent_config)

    scenario = scenarios.load('custom/custom_no_comm.py').Scenario()
    scenario.setup(num_agents=2, num_landmarks=3)
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                        info_callback=None, shared_viewer=False)

    env.render()
    sleep(1)

    # create interactive policies for each agent
    policies = [RandomPolicy(env, i) for i in range(env.n)]

    # execution loop
    for epoch in range(training_config.num_epochs):

        # randomly place the agent(s)
        obs_n = env.reset()

        for step in range(agent_config.time_horizon):

            # query for action from each agent's policy
            act_n = []
            for i, policy in enumerate(policies):
                act_n.append(policy.action(obs_n[i]))
            # step environment
            obs_n, reward_n, done_n, _ = env.step(act_n)

            # -----------------------------------
            # all agents make step here
            agent.make_step(game_config.batch_size,
                            num_agents=env.n,
                            num_entities=len(env.world.entities),
                            observations=obs_n)

            env.render()

    print('done')


if __name__ == '__main__':
    main()
