import random

from multiagent import BasePolicy
import numpy as np


class RandomPolicy(BasePolicy):
    """Random actions"""

    def __init__(self, env, agent_index):

        super(RandomPolicy, self).__init__()
        self.env = env
        # hard-coded keyboard events
        self.move = [False for _ in range(4)]
        self.comm = [False for _ in range(env.world.dim_c)]

    def action(self, obs):
        if self.env.discrete_action_input:
            # movements in 4 directions (not used by default)
            u = random.randrange(0, 4, 1)
        else:
            # one-hot action vector, 5D because of no-move action at u[0]
            u = np.zeros(5)  # 5-d because of no-move action
            u[random.randrange(0, 5, 1)] = 1

        # stack with the empty communication channel
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])
