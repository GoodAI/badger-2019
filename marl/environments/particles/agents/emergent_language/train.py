import argparse
from typing import List, Dict, Tuple

import numpy as np
import pandas
import torch
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
import agents.emergent_language.configs
from agents.emergent_language import configs
from agents.emergent_language.modules.agent import AgentModule
from agents.emergent_language.modules.game import GameModule
from collections import defaultdict

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt


def get_parser():
    parser = argparse.ArgumentParser(description="Trains the agents for cooperative communication task")
    parser.add_argument('--no-utterances', action='store_true', help='if specified disables the communications '
                                                                     'channel ( '
                                                                     'default enabled)')
    parser.add_argument('--render', action='store_true', help='render environment?')
    parser.add_argument('--penalize-words', action='store_true', help='if specified penalizes uncommon word usage ('
                                                                      'default disabled)')
    parser.add_argument('--n-epochs', '-e', type=int, help='if specified sets number of training epochs (default 5000)')
    parser.add_argument('--learning-rate', type=float, help='if specified sets learning rate (default 1e-3)')
    parser.add_argument('--batch-size', type=int, help='if specified sets batch size(default 256)')
    parser.add_argument('--n-timesteps', '-t', type=int, help='if specified sets timestep length of each episode (default '
                                                              '32)')
    parser.add_argument('--num-shapes', '-s', type=int, help='if specified sets number of colors (default 3)')
    parser.add_argument('--num-colors', '-c', type=int, help='if specified sets number of shapes (default 3)')
    parser.add_argument('--max-agents', type=int, help='if specified sets maximum number of agents in each episode ('
                                                       'default 3)')
    parser.add_argument('--min-agents', type=int, help='if specified sets minimum number of agents in each episode ('
                                                       'default 1)')
    parser.add_argument('--max-landmarks', type=int, help='if specified sets maximum number of landmarks in each episode '
                                                          '(default 3)')
    parser.add_argument('--min-landmarks', type=int, help='if specified sets minimum number of landmarks in each episode '
                                                          '(default 1)')
    parser.add_argument('--vocab-size', '-v', type=int, help='if specified sets maximum vocab size in each episode ('
                                                             'default 6)')
    parser.add_argument('--world-dim', '-w', type=int, help='if specified sets the side length of the square grid where '
                                                            'all agents and landmarks spawn(default 16)')
    parser.add_argument('--oov-prob', '-o', type=int, help='higher value penalize uncommon words less when penalizing '
                                                           'words (default 6)')
    parser.add_argument('--load-model-weights', type=str, help='if specified start with saved model weights saved at file '
                                                               'given by this argument')
    parser.add_argument('--save-model-weights', type=str, help='if specified save the model weights at file given by this '
                                                               'argument')
    parser.add_argument('--use-cuda', action='store_true', help='if specified enables training on CUDA (default disabled)')
    return parser


class LossPrinter:
    all_losses: Dict[int, Dict[int, List[Tuple[float, float]]]]

    def __init__(self):
        self.all_losses = {}

    def print_losses(self, epoch, losses, dists, game_config):

        for a in range(game_config.min_agents, game_config.max_agents + 1):
            for l in range(game_config.min_landmarks, game_config.max_landmarks + 1):

                loss = losses[a][l][-1] if len(losses[a][l]) > 0 else 0
                min_loss = min(losses[a][l]) if len(losses[a][l]) > 0 else 0

                dist = dists[a][l][-1] if len(dists[a][l]) > 0 else 0
                min_dist = min(dists[a][l]) if len(dists[a][l]) > 0 else 0

                print("[epoch %d][%d agents, %d landmarks][%d batches][last loss: %f][min loss: %f][last dist: %f][min "
                      "dist: %f]" % (epoch, a, l, len(losses[a][l]), loss, min_loss, dist, min_dist))

                self._add_to_dicts(a, l, loss, dist)

        print("_________________________")

    def _add_to_dicts(self, a: int, l: int, loss, dist):

        if a not in self.all_losses:
            self.all_losses[a] = {}

        if l not in self.all_losses[a]:
            self.all_losses[a][l] = []

        if type(dist) == torch.Tensor:
            self.all_losses[a][l].append((loss.item(), dist.item()))
        else:
            self.all_losses[a][l].append((float(loss), float(dist)))

    def show_graphs(self):

        # fig = go.Figure()
        # fig = plt.figure()

        fig, (ax_loss, ax_dist) = plt.subplots(2, 1)
        for num_agents, dicts in self.all_losses.items():
            for num_landmarks, values in self.all_losses[num_agents].items():
                # convert to pandas dataframe
                losses = [vl[0] for vl in values]
                dists = [vl[1] for vl in values]
                epochs = list(range(len(losses)))
                # df = pandas.DataFrame({'loss': losses, 'steps': steps, 'dists': dists})
                # fig = px.line(df, x="steps", y='loss', title='Loss vs epochs')
                # fig.add_trace(go.Scatter(x=steps, y=losses,
                #                          mode='lines',
                #                          name=f'agents: {num_agents}, landmarks: {num_landmarks} - losses'))
                #
                # fig.add_trace(go.Scatter(x=steps, y=dists,
                #                          mode='lines',
                #                          name=f'agents: {num_agents}, landmarks: {num_landmarks} - dists'))
                ax_loss.plot(epochs, losses, label=f'agents: {num_agents}, landmarks: {num_landmarks} - losses')
                ax_dist.plot(epochs, dists, label=f'agents: {num_agents}, landmarks: {num_landmarks} - dists')

        ax_loss.legend()
        ax_loss.set_title('loss')
        ax_dist.legend()
        ax_dist.set_title('distance to goal')

        # fig = px.line(df, x="steps", y='dists', title='Loss vs epochs')
        # fig.show()
        plt.show()


def main():
    parser = get_parser()
    args = vars(parser.parse_args())
    agent_config = configs.get_agent_config(args)
    game_config = configs.get_game_config(args)
    training_config = configs.get_training_config(args)

    print("Training with config:")
    print(training_config)
    print(game_config)
    print(agent_config)

    agent = AgentModule(agent_config)
    if training_config.use_cuda:
        agent.cuda()

    optimizer = RMSprop(agent.parameters(), lr=training_config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, cooldown=5)
    losses = defaultdict(lambda: defaultdict(list))
    dists = defaultdict(lambda: defaultdict(list))

    lp = LossPrinter()

    for epoch in range(training_config.num_epochs):

        # each epoch generate random num of agents and landmarks
        num_agents = np.random.randint(game_config.min_agents, game_config.max_agents + 1)
        num_landmarks = np.random.randint(game_config.min_landmarks, game_config.max_landmarks + 1)
        agent.reset()  # clear the total cost (of the epoch) and the word_counts

        # create new game instance (!)
        game = GameModule(game_config, num_agents, num_landmarks)
        if training_config.use_cuda:
            game.cuda()
        optimizer.zero_grad()

        # run given num steps in the game and collect gradients
        total_loss, _ = agent(game)
        per_agent_loss = total_loss.data[0] / num_agents / game_config.batch_size
        losses[num_agents][num_landmarks].append(per_agent_loss)

        # average dist of agents to goal
        dist = game.get_avg_agent_to_goal_distance()
        avg_dist = dist.data / num_agents / game_config.batch_size
        dists[num_agents][num_landmarks].append(avg_dist)

        lp.print_losses(epoch, losses, dists, game_config)

        total_loss.backward()
        optimizer.step()

        if num_agents == game_config.max_agents and num_landmarks == game_config.max_landmarks:
            scheduler.step(losses[game_config.max_agents][game_config.max_landmarks][-1])

    if training_config.save_model:
        torch.save(agent, training_config.save_model_file)
        print("Saved agent model weights at %s" % training_config.save_model_file)
    """
    import code
    code.interact(local=locals())
    """

    lp.show_graphs()


if __name__ == "__main__":
    main()




























