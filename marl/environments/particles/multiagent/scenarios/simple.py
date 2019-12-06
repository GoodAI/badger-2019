import numpy as np
from marl.environments.particles.multiagent.core import World, Agent, Landmark
# from multiagent.scenarios import BaseScenario
from marl.environments.particles.multiagent.scenarios.base_scenario import BaseScenario


class Scenario(BaseScenario):
    """
    1 agent, 1 landmark

    """

    def __init__(self):
        super().__init__(num_agents=1)

    def make_world(self):
        world = World()

        # add emergent_comm
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True

        # add landmarks
        world.landmarks = [Landmark() for i in range(1)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        super().reset_world(world)

        # random properties for emergent_comm
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.75, 0.75])
        world.landmarks[0].color = np.array([0.75, 0.25, 0.25])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        return -dist2

    def observation(self, agent, world):
        super().observation(agent, world)

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)
