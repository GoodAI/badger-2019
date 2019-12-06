import copy
from typing import Tuple, List, Optional, Union

import numpy as np
from marl.environments.particles.multiagent.core import World, Agent, Landmark
from marl.environments.particles.multiagent.scenarios.custom.configurable_scenario import ConfigurableScenario


class Scenario(ConfigurableScenario):
    """
    Custom scenario without communication:

    -N agents,
    -M landmarks,
    -each receives randomly chosen goal from M landmarks
    """

    def __init__(self):
        # note: num agents is overwritten in the setup..
        super().__init__(num_agents=1)

    num_landmarks: int

    landmarks: List[Landmark]  # landmarks which the agents care about
    obstacles: List[Landmark]  # an optional fence about the environment

    add_obstacles: bool

    data_scale: float
    landmarks_hidden: bool

    discrete_action_space: bool

    just_x: bool

    reward_last_n_steps: int
    show_abs_pos: bool

    reward_scale: float

    show_velocity: bool

    fence_size = 0.3
    spawn_radius: float = 1 - fence_size

    max_agent_speed: Union[float, None]

    goal_by_color: bool
    fixed_colors: bool
    color_palette = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # just RGB now

    agent_size: float = 0.07

    reward_type: str
    distance_type: str
    max_distance: Optional[float]
    max_reward_distance: Optional[float]
    use_distance: bool

    def setup(self,
              num_agents: int,
              num_landmarks: int,
              rollout_length: int = 16,
              data_scale: float = 1.0,
              landmarks_hidden: bool = False,
              just_x: Optional[bool] = False,
              discrete_action_space: Optional[bool] = True,
              reward_last_n_steps: Optional[int] = -1,
              show_abs_pos: Optional[bool] = False,
              reward_scale: float = 1.0,
              add_obstacles: bool = False,
              show_velocity: bool = False,
              max_speed: Optional[float] = None,
              goal_by_color: Optional[bool] = False,
              fixed_colors: Optional[bool] = False,
              reward_type: str = 'l2',
              max_reward_distance: Optional[float] = None,
              use_distance: bool = True,
              distance_type: str = 'l2',
              max_distance: Optional[float] = None):
        """
        Args:
            fixed_colors: if True, the colors are taken from a fixed "palette", always the same set of colors..
            goal_by_color: if False, the goal is specified by relative position, if True, color of landmark is shown
            max_speed:
            show_velocity:
            add_obstacles: surround the environment with obstacles? (so that the agent cannot escape too far)
            reward_last_n_steps:
            show_abs_pos:
            reward_scale:
            reward_type:
            discrete_action_space:
            just_x: place objects just on the Y=0? (movement not constrained for now)
            landmarks_hidden:
            data_scale:
            num_agents: num controllable agents in the environment
            num_landmarks: num landmarks
            rollout_length: how many steps to simulate before setting done to True?
            max_reward_distance: limit the distance from which the agent can see the reward (sparsify rewards)
            distance_type: distance type perceived by the agent
            max_distance: limit the maximum distance perceived by the agent (limits the FOV of the agent)
            use_distance: use the distance? if false, the 2D direction is used
        """
        super().setup()

        self.max_distance = max_distance
        self.max_reward_distance = max_reward_distance
        self.distance_type = distance_type
        self.reward_type = reward_type
        self.use_distance = use_distance

        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.just_x = just_x
        self.discrete_action_space = discrete_action_space
        self.show_abs_pos = show_abs_pos

        assert rollout_length > 0
        self.episode_length = rollout_length
        self.data_scale = data_scale
        self.landmarks_hidden = landmarks_hidden
        self.reward_scale = reward_scale

        self.reward_last_n_steps = reward_last_n_steps
        if self.reward_last_n_steps > self.episode_length:
            self.reward_last_n_steps = -1

        self.add_obstacles = add_obstacles
        self.show_velocity = show_velocity

        self.max_agent_speed = max_speed
        self.goal_by_color = goal_by_color
        self.fixed_colors = fixed_colors
        if fixed_colors:
            assert self.num_landmarks <= 3, 'in case the fixed_colors is used, the num_landmarks has to be <= 3 now'

        self.landmarks = []
        self.obstacles = []

    def make_world(self):
        world = World()
        # world.damping = 0.75

        # actions continuous or discrete? (see the MultiAgentEnv constructor)
        world.discrete_action_space = self.discrete_action_space
        if self.discrete_action_space:
            world.discrete_action = True

        # add emergent_comm
        world.agents = [Agent() for _ in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = self.add_obstacles  # agent does not collide if there are no obstacles
            agent.silent = True
            agent.size = self.agent_size  # agents are bigger than landmarks
            if self.max_agent_speed is not None:
                agent.max_speed = self.max_agent_speed

        # add landmarks
        self.landmarks = []
        world.landmarks = [Landmark() for _ in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            self.landmarks.append(landmark)

        self.create_fence(world)

        # make initial conditions
        self.reset_world(world)
        return world

    def create_fence(self, world):
        """Create the fence around the visible part of the environment, so that the agent(s) cannot escape"""

        self.obstacles = []

        if self.add_obstacles:

            width = 1
            size = self.fence_size
            no_obstacles = np.math.floor((width * 2 + size) / size)

            x_positions = [x * size - width + 0.5 * size for x in range(no_obstacles)]
            y_positions = x_positions

            # draw a rectangle fence composed of obstacles
            for x_pos in x_positions:
                for y_pos in y_positions:
                    if x_pos == x_positions[0] or \
                            x_pos == x_positions[-1] or \
                            y_pos == y_positions[0] or \
                            y_pos == y_positions[-1]:
                        self._add_obstacle_at(x_pos, y_pos, size, world)

    def _add_obstacle_at(self, x: float, y: float, size: float, world):
        landmark = Landmark()
        landmark.name = f'fence_part_{x}_{y}'
        landmark.collide = True
        landmark.size = size
        landmark.movable = False
        landmark.color = np.array([0, 1, 0])
        landmark.state.p_pos = np.array([x, y])
        world.landmarks.append(landmark)
        self.obstacles.append(landmark)

    @staticmethod
    def _random_color(interval: Tuple[float, float]):
        return np.array(np.random.uniform(interval[0], interval[1], 3))

    def _pick_color(self, position: int):
        return self.color_palette[position]

    def _random_colors(self, world):
        # random properties for color
        for i, agent in enumerate(world.agents):
            # agent.color = Scenario._random_color((0, 0.25))
            agent.color = Scenario._random_color((0, 1))

        # random properties for landmarks
        for i, landmark in enumerate(self.landmarks):
            # landmark.color = Scenario._random_color((0.75, 1))
            if self.fixed_colors:
                landmark.color = self._pick_color(i)
            else:
                landmark.color = Scenario._random_color((0, 1))

    def _random_positions(self, world):
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1 * self.spawn_radius, +1 * self.spawn_radius, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            if self.just_x:
                agent.state.p_pos[1] = 0

        for i, landmark in enumerate(self.landmarks):
            # for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1 * self.spawn_radius, +1 * self.spawn_radius, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            if self.just_x:
                landmark.state.p_pos[1] = 0

    def _random_goals(self, world):
        for agent in world.agents:
            # pick a random goal and assign to each agent
            goal = world.landmarks[np.random.randint(0, len(self.landmarks))]
            # (agent has alpha=0.5!)
            agent.color = np.copy(goal.color)  # agent obtains color of the landmark
            agent.goal = goal  # append directly to the agent

    def reset_world(self, world):
        self.steps_from_last_reset = 0
        self._random_colors(world)
        self._random_positions(world)
        self._random_goals(world)

    @staticmethod
    def _compute_dist_to_goal(agent, dist_type: str, max_dist: Optional[float]) -> float:
        """Return distance between the agent and its goal"""

        if dist_type == 'l1':
            dist2 = np.sum(np.abs(agent.state.p_pos - agent.goal.state.p_pos))  # l1 loss
        elif dist_type == 'smoothl1':
            print(f'TODO this is not tested, check results carefully!')
            true = agent.goal.state.p_pos
            pred = agent.state.p_pos
            delta = 0.5
            # https://pytorch.org/docs/master/nn.html?highlight=smoothl1#torch.nn.SmoothL1Loss
            # https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0
            loss = np.where(np.abs(true - pred) < delta, 0.5 * ((true - pred) ** 2),
                            delta * np.abs(true - pred) - 0.5 * (delta ** 2))
            dist2 = np.sum(loss)
        elif dist_type == 'l2':
            dist2 = np.sum(np.square(agent.state.p_pos - agent.goal.state.p_pos))  # l2 loss
        elif dist_type == 'l4':
            dist2 = np.sum(np.power(agent.state.p_pos - agent.goal.state.p_pos, 4))
        elif dist_type == 'l0.5':
            dist2 = np.sum(np.pow(np.abs(agent.state.p_pos - agent.goal.state.p_pos), 0.5))
        elif dist_type == 'discrete':
            # done in the same way as distance: distance 0 is perfect, distance 1 otherwise
            l1_dist = np.sum(np.abs(agent.state.p_pos - agent.goal.state.p_pos))  # l1 loss
            dist2 = 1 if l1_dist > 0.1 else 0
        else:
            raise AttributeError('unknown reward type')

        if max_dist is not None and dist2 > max_dist:
            dist2 = max_dist

        return dist2

    def reward(self, agent, world):
        dist2 = 0

        if self.reward_last_n_steps < 0 or self.episode_length - self.steps_from_last_reset < self.reward_last_n_steps:
            dist2 = self._compute_dist_to_goal(agent=agent, dist_type=self.reward_type, max_dist=self.max_reward_distance)

        # reward is negative distance
        return -dist2 * self.reward_scale

    def observation(self, agent, world):
        """ Produces an observation
        Args:
            agent: agent that requests the observation
            world:

        Returns: observation is in the format [rel_goal_x, rel_goal_y, vel_x, vel_y, relative coordinates of others]
        """
        super().observation(agent, world)
        # TODO stack also colors?

        # get positions of all entities in this agent's reference frame
        entity_observations = []
        if not self.landmarks_hidden:
            for entity in self.landmarks:
                entity_observations.append(entity.state.p_pos - agent.state.p_pos)
                # show colors only in case the goal by color is chosen
                if self.goal_by_color:
                    entity_observations.append(entity.color)

        vel = []
        if self.show_velocity:
            vel = [agent.state.p_vel]

        if self.goal_by_color:
            goal_definition = agent.goal.color
        else:
            if self.use_distance:
                # just the distance
                dist = self._compute_dist_to_goal(agent=agent, dist_type=self.distance_type, max_dist=self.max_distance)
                goal_definition = np.array(dist).reshape(1)
            else:
                # 2D direction to the goal
                goal_definition = agent.goal.state.p_pos - agent.state.p_pos

        result = np.concatenate([goal_definition] +
                                vel +
                                ([agent.state.p_pos] if self.show_abs_pos else []) +
                                entity_observations)

        return result * self.data_scale
