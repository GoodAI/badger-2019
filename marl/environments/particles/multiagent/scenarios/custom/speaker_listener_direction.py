from typing import Union, List

import numpy as np
from marl.environments.particles.multiagent.core import World, Agent, Landmark
# from multiagent.scenarios import BaseScenario
from marl.environments.particles.multiagent.scenarios.base_scenario import BaseScenario
from marl.environments.particles.multiagent.scenarios.custom.configurable_scenario import ConfigurableScenario


class Scenario(ConfigurableScenario):
    """
    Simplified version of the speaker_listener.

    Here the speaker gets 1of3 goal landmarks, should communicate this to the listener.

    There is 1 direction associated with each landmark.
    The listener is rewarded by moving in the direction corresponding to the landmark.
    """

    zero_speaker_obs: bool

    def __init__(self):
        super().__init__(num_agents=2)

    def setup(self,
              rollout_length: int,
              zero_speaker_obs: bool = False):
        super().setup()
        self.zero_speaker_obs = zero_speaker_obs

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 3
        num_landmarks = 3
        world.collaborative = True
        # add emergent_comm
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.size = 0.075
        # speaker
        world.agents[0].movable = True
        world.agents[0].silent = True  # comm not done through the env..
        # listener
        world.agents[1].silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        super().reset_world(world)

        # assign goals to emergent_comm
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want listener to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)
        # random properties for emergent_comm
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.65, 0.15, 0.15])
        world.landmarks[1].color = np.array([0.15, 0.65, 0.15])
        world.landmarks[2].color = np.array([0.15, 0.15, 0.65])

        # tag each landmark with a ground-truth direction
        world.landmarks[0].listener_direction = np.array([0, 1])
        world.landmarks[1].listener_direction = np.array([1, -1])
        world.landmarks[2].listener_direction = np.array([-1, -1])

        # special colors for goals
        world.agents[0].goal_a.color = world.agents[0].goal_b.color + np.array([0.45, 0.45, 0.45])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        return self.reward(agent, reward)

    def reward(self, agent, world):
        # squared distance between
        # the velocity (direction) of the listener and the ground-truth direction
        a = world.agents[0]
        dist2 = np.sum(np.square(a.goal_a.state.p_vel - a.goal_b.listener_direction))
        return -dist2

    def observation(self, agent, world):
        super().observation(agent, world)

        # goal color
        goal_color = np.zeros(world.dim_color)
        if agent.goal_b is not None:
            goal_color = agent.goal_b.color

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # speaker
        if agent == world.agents[0]:
            # RGB of the goal
            vel = agent.state.p_vel
            entity_positions = entity_pos
            if self.zero_speaker_obs:
                vel *= 0
                entity_positions = [ep * 0 for ep in entity_pos]

            return np.concatenate([vel] +  # my velocity
                                  entity_positions +  # other entity positions
                                  [goal_color] +  # color of the goal
                                  [np.ones(1, )])  # id of other agent
        # listener
        if agent == world.agents[1]:
            # goal hidden
            return np.concatenate([agent.state.p_vel] +  # my velocity
                                  entity_pos +  # other entity positions
                                  [np.zeros(3)] +  # zeroed-out goal
                                  [np.zeros(1,)])  # id of other agent

        raise Exception('Unexpected agent')

