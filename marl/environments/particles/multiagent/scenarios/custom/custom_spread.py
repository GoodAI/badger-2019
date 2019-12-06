from operator import itemgetter
from typing import List, Tuple

import numpy as np

from marl.environments.particles.multiagent.core import World, Agent, Landmark, Entity
from marl.environments.particles.multiagent.scenarios.custom.configurable_scenario import ConfigurableScenario


class Scenario(ConfigurableScenario):
    """ Cooperative Navigation task in the https://arxiv.org/pdf/1706.02275.pdf

    Mostly the same as scenarios.simple_spread (used in the paper), here, the observation is formatted in a way that
    is required by the ATOC paper: https://arxiv.org/abs/1805.07733

    N agents, N (or M) landmarks, each agent should sit on one landmark, implicit/explicit communication.
    """

    num_landmarks: int

    num_perceived_landmarks: int
    num_perceived_agents: int
    show_agent_velocities: bool

    env_size: float

    def __init__(self):
        # parameters overriden by the setup
        super().__init__(num_agents=3)
        self.num_landmarks = 3

    def setup(self,
              num_agents: int,
              num_landmarks: int,
              rollout_length: int,
              num_perceived_agents: int,
              num_perceived_landmarks: int,
              show_agent_velocities: bool,
              env_size: float):
        """
        Args:
            env_size: size of the environment (default 1), in what range to generate the landmarks/agents
            num_agents: num agents in the environment
            num_landmarks: num_lanrmadks in the environment
            rollout_length: num steps of one epoch
            num_perceived_agents: number of perceived agents, excluding the current agent (myself)
            num_perceived_landmarks: number of landmakrs perceived by the agent
            show_agent_velocities: show velocities of other agents as well? (my velocity shown always, my pos. never)
        """
        super().setup()

        self.env_size = env_size
        self.episode_length = rollout_length
        self.show_agent_velocities = show_agent_velocities

        # num agents
        self.num_agents = num_agents
        assert num_perceived_agents < num_agents, 'num_perceived agents has to be smaller than num_agents'
        self.num_perceived_agents = num_perceived_agents

        # num landmarks
        self.num_landmarks = num_landmarks
        if num_perceived_landmarks > num_landmarks:
            print(f'WARNING: num_perceived_lanrmarks should be <= num_landmarks, decreasing the number automatically')
            self.num_perceived_landmarks = num_landmarks
        else:
            self.num_perceived_landmarks = num_perceived_landmarks

    def make_world(self):

        world = World()
        # set any world properties first
        world.dim_c = 2  # TODO dim_c??
        world.collaborative = True
        # add emergent_comm
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            agent.index = i
        # add landmarks
        world.landmarks = [Landmark() for i in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.index = i
            landmark.size = 0.05

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        """Default for now"""
        super().reset_world(world)

        # random properties for emergent_comm
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1 * self.env_size, +1 * self.env_size, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1 * self.env_size, +1 * self.env_size, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        """Not used"""
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        # TODO this is not scaled
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0

        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def _compute_sorted_distances_to(self, agent: Entity, entities: List[Entity]) \
            -> List[Tuple[float, Entity, np.ndarray]]:
        """
        Args:
            agent: me
            entities: list of other entities to measure distance to

        Returns: list of Tuples[distance to me, index of the entity]
        """
        my_pos = agent.state.p_pos
        results = []

        # measure distance and store id
        for entity in entities:
            rel_pos = my_pos - entity.state.p_pos
            dist = float(np.sum(np.abs(rel_pos)))
            results.append((dist, entity, rel_pos))

        # sort by distance
        sorted_results = sorted(results, key=itemgetter(0))
        return sorted_results

    def observation(self, agent, world) -> np.ndarray:
        """ Composes the Observation.

        The agent perceives K nearest other agents and L nearest landmarks.

        observation:
                [agent positions and velocities, landmark positions, agent_ids]

        Communication through the environment not used here.
        """
        super().observation(agent, world)

        # sorted nearest agents
        sorted_agents = self._compute_sorted_distances_to(agent, world.agents)

        # stack info about K nearest agents (including me)
        agents = []
        agent_ids = []
        for agent_id, (dist, other_agent, other_agent_rel_pos) in enumerate(sorted_agents):
            if agent_id == self.num_perceived_agents + 1:
                break

            # optional velocities of other agents, my velocity shown always
            if self.show_agent_velocities or agent_id == 0:
                agents.append(other_agent.state.p_vel)

            # don't add my id, don't show my velocity
            if agent_id > 0:
                agents.append(other_agent_rel_pos)
                agent_ids.append(other_agent.index)

        sorted_landmarks = self._compute_sorted_distances_to(agent, world.landmarks)
        # stack info about L nearest landmarks
        landmarks = []
        # landmark_ids = []
        for landmark_id, (dist, landmark, landmark_rel_pos) in enumerate(sorted_landmarks):
            if landmark_id == self.num_perceived_landmarks:
                break
            landmarks.append(landmark_rel_pos)
        #     landmark_ids.append(landmark.index)
        observation = np.concatenate(agents + landmarks + [np.array(agent_ids)])
        return observation
