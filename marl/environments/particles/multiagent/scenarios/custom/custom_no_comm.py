import numpy as np
from marl.environments.particles.multiagent.core import World, Agent, Landmark
from marl.environments.particles.multiagent.scenarios.custom.configurable_scenario import ConfigurableScenario


class Scenario(ConfigurableScenario):
    """
    Custom scenario without communication

    """

    num_landmarks: int

    are_positions_fixed: bool  # do not change positions of objects/agents?

    scale: float

    only_x: bool

    # single_agent_compatible: bool

    episode_length: int

    data_scale: float

    def __init__(self):
        super().__init__(num_agents=1)

    def setup(self,
              num_agents: int,
              num_landmarks: int,
              are_positions_fixed: bool = False,
              scale: float = 1,
              only_x: bool = False,
              rollout_length: int = 16,
              data_scale: float = 1.0):
        """
        Args:
            reward_type:
            data_scale:
            num_agents: num controllable agents in the environment
            num_landmarks: num landmarks
            are_positions_fixed: initial positions (after reset) are fixed or random every reset?
            scale: how far from the origin the things should be spawned? lower number ~ closer
            only_x:
            rollout_length: how many steps to simulate before setting done to True?
        """
        super().setup()

        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.are_positions_fixed = are_positions_fixed
        assert scale > 0 and scale <= 1
        self.scale = scale
        self.data_scale = data_scale

        assert rollout_length > 0
        self.episode_length = rollout_length

        self.init_agent_pos = []
        self.init_landmarks_pos = []
        self.only_x = only_x

        # self.single_agent_compatible = single_agent_compatible
        # if self.single_agent_compatible:
        #     assert self.num_agents == 1, 'if single_agent_compatible is set, there has to be just one agent'

    def make_world(self):

        world = World()

        # add emergent_comm
        world.agents = [Agent() for _ in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True

        # add landmarks
        world.landmarks = [Landmark() for _ in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        super().reset_world(world)

        # TODO random colors

        # random properties for emergent_comm
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.75, 0.75])
        world.landmarks[0].color = np.array([0.75, 0.25, 0.25])

        # set random initial states
        if not self.are_positions_fixed or len(self.init_agent_pos) == 0:
            for agent in world.agents:
                agent.state.p_pos = np.random.uniform(-1*self.scale, +1*self.scale, world.dim_p)
                self.init_agent_pos.append(agent.state.p_pos)
        else:
            for agent, pos in zip(world.agents, self.init_agent_pos):
                agent.state.p_pos = pos
                print(f'setting agent to {pos}')

        for agent in world.agents:
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        if not self.are_positions_fixed or len(self.init_landmarks_pos) == 0:
            for i, landmark in enumerate(world.landmarks):
                landmark.state.p_pos = np.random.uniform(-1*self.scale, +1*self.scale, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
                self.init_landmarks_pos.append(landmark.state.p_pos)
        else:
            for landmark, pos in zip(world.landmarks, self.init_landmarks_pos):
                landmark.state.p_pos = pos
                print(f'setting landmark to {pos}')

    def reward(self, agent, world):
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))  # l2 loss
        return -dist2

    def observation(self, agent, world):
        super().observation(agent, world)
        # TODO stack also colors

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # result = np.concatenate(entity_pos)
        result = np.concatenate([agent.state.p_vel] + entity_pos)
        result *= self.data_scale
        # return np.concatenate([agent.state.p_vel] + entity_pos)
        return result

    def is_done(self, agent, world) -> bool:
        """
        is called num_agent times
        Args:
            agent:
            world:

        Returns: indication of episode end
        """
        if self.steps_from_last_reset / self.num_agents > self.episode_length:
            return True
        return False




