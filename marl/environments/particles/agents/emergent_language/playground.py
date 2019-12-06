import torch

from agents.emergent_language import configs
from agents.emergent_language.modules.game import GameModule
from agents.emergent_language.configs import default_game_config, get_game_config
import code

from agents.emergent_language.train import get_parser


def run_default_config():
    config = {
        'batch_size': default_game_config.batch_size,
        'world_dim': default_game_config.world_dim,
        'max_agents': default_game_config.max_agents,
        'max_landmarks': default_game_config.max_landmarks,
        'min_agents': default_game_config.min_agents,
        'min_landmarks': default_game_config.min_landmarks,
        'num_shapes': default_game_config.num_shapes,
        'num_colors': default_game_config.num_colors,
        'no_utterances': not default_game_config.use_utterances,
        'vocab_size': default_game_config.vocab_size,
        'memory_size': default_game_config.memory_size
    }

    agent = torch.load('weights.pkl')
    agent.reset()
    agent.train(False)
    code.interact(local=locals())


if __name__ == '__main__':
    # run_default_config()

    parser = get_parser()
    args = vars(parser.parse_args())
    agent_config = configs.get_agent_config(args)
    game_config = configs.get_game_config(args)
    training_config = configs.get_training_config(args)

    agent = torch.load('nice_weights.pkl')
    agent.reset()
    agent.train(False)

    num_agents = game_config.min_agents
    num_landmarks = game_config.min_landmarks

    game = GameModule(game_config, num_agents, num_landmarks)
    total_loss, _ = agent(game)

    code.interact(local=locals())

