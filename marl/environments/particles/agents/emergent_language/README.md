# emergent-language
An implementation of Emergence of Grounded Compositional Language in Multi-Agent Populations by Igor Mordatch and Pieter Abbeel

To run, invoke `python3 train.py` in environment with PyTorch installed. To experiment with parameters, invoke `python3 train.py --help` to get a list of command line arguments that modify parameters. Currently training just prints out the loss of each game episode run, without any further analysis, and the model weights are not saved at the end. These features are coming soon.

* `game.py` provides a non-tensor based implementation of the game mechanics (used for game behavior exploration and random game generation during training
* `model.py` provides the full computational model including agent and game dynamics through an entire episode
* `train.py` provides the training harness that runs many games and trains the agents
* `configs.py` provides the data structures that are passed as configuration to various modules in the computational graph as well as the default values used in training now
* `constants.py` provides constant factors that shouldn't need modification during regular running of the model
* `visualize.py` provides a computational graph visualization tool taken from [here](https://github.com/szagoruyko/functional-zoo/blob/master/visualize.py)
* `simple_model.py` provides a simple model that doesn't communicate and only moves based on its own goal (used for testing other components)
* `comp-graph.pdf` is a pdf visualization of the computational graph of the game-agent mechanics

## modifications

Source repo [here](https://github.com/bkgoksel/emergent-language)

Presentation slides [here](https://pdfs.semanticscholar.org/e1c5/26518bc525f4437851e6d5196c34e859a351.pdf)

Paper [here](https://arxiv.org/pdf/1703.04908.pdf)

Blogpost [here](https://openai.com/blog/learning-to-communicate/)



### From the presentation:

- A population of agents is situated as moving particles in a continuous 2-D environment,
possessing properties such as color and shape.

- The goals of the population are such as moving to a location, and use of language in order to
coordinate on those goals

- A reinforcement learning problem - Agents perform some action ’a’ and communication
utterances ’c’ according to identical policy for all agents

- The language - assigns symbols to separately refer to ENVIRONMENTAL LANDMARKS,
ACTION VERBS and AGENTS.

- Non-verbal communication such as pointing and guiding when language communication is
unavailable.

- training on a variety of tasks and environment configurations simultaneously

### Implementation details

- all fully-connected modules with 256 hidden units and 2 layers
each are used in all the experiments
- Size is feature vectors φ is 256 and
- size of each memory module is 32. 

### Tuning details

- maximum vocabulary size K = 20 in all experiments
- For small maximum size, the policy optimization became stuck in a local minima

### Practical notes

- It seems to train all num_agents (e.g. from 2 to 5 agents)
- And then id backprops though that
- Config like 2-7 agents kills it on memory (32gigs)
