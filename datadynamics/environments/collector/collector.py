import math
from itertools import groupby

import gymnasium
import numpy as np
import pygame
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import AECEnv

from datadynamics.utils.objects import Collector, Point

FPS = 120
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
# Rendering sizes.
POINT_SIZE = 8
PATH_SIZE = 2
COLLECTOR_SIZE = 4
COLLECTOR_LEN = 10
FONT_SIZE = 12


def env(**kwargs):
    """Creates a collector environment.

    Returns:
        pettingzoo.utils.env.AECEnv: Created environment.
    """
    if "n_points" in kwargs:
        env = SamplingWrapperEnv(**kwargs)
    else:
        env = raw_env(**kwargs)
    return env


class raw_env(AECEnv):
    """Raw collector environment.

    This environment is based on a 2D plane. Points are given by their (x, y)
    coordinates and each agent defines a `collector` that can move around the
    plane to collect points for a reward. Collectors may only move between
    points and they always collect the point that they move to.
    The cost of moving is defined as the Euclidean distance traveled.
    Agents may also cheat and collect an already collected point. Cheating as
    well as rewards are defined by a user-given function.

    Attributes:
        See AECEnv.
    """

    metadata = {
        "name": "collector",
        "render_modes": ["rgb_array", "human"],
        "is_parrallelizable": False,
        "render_fps": FPS,
    }

    def __init__(
        self,
        point_positions,
        init_agent_positions,
        max_collect,
        cheating_cost=lambda point: 500 * 0.5,
        collection_reward=lambda point: 100,
        reveal_cheating_cost=True,
        reveal_collection_reward=True,
        seed=None,
        render_mode=None,
    ):
        """Initialize environment.

        Args:
            point_positions (np.ndarray): Positions of collectable points as a
                numpy array with shape (n, 2) representing n (x, y)
                coordinates.
            init_agent_positions (np.ndarray): Initial positions of n agents
                as a numpy array with shape (n, 2) representing n (x, y)
                coordinates.
            max_collect (list): List of maximum number of points that each
                agent can collect. Index i corresponds to agent i given by
                init_agent_positions.
            cheating_cost (function, optional): Function that takes a point
                and returns the cost of cheating by collecting that point.
                Defaults to lambda point: 500 * 0.5.
            collection_reward (function, optional): Function that takes a
                point and returns the reward of collecting that point.
                Defaults to lambda point: 100.
            reveal_cheating_cost (bool, optional): Whether to reveal the
                cheating costs to the agent in observations. Defaults to True.
            reveal_collection_reward (bool, optional): Whether to reveal the
                collection rewards to the agent in observations. Defaults to
                True.
            seed (int, optional): Seed for random number generator. Defaults
                to None.
            render_mode (str, optional): Render mode. Supported modes are
                specified in environment's metadata["render_modes"] dict.
                Defaults to None.
        """
        assert (
            render_mode in self.metadata["render_modes"] or render_mode is None
        ), (
            f"render_mode: {render_mode} is not supported. "
            f"Supported modes: {self.metadata['render_modes']}"
        )

        self.seed(seed)

        self.point_positions = point_positions
        self.agent_positions = init_agent_positions
        self.render_mode = render_mode
        self.cheating_cost = cheating_cost
        self.collection_reward = collection_reward
        self.reveal_cheating_cost = reveal_cheating_cost
        self.reveal_collection_reward = reveal_collection_reward

        self.reward_range = (-np.inf, 0)

        self.agents = [f"agent_{i}" for i in range(len(self.agent_positions))]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = {
            agent: i for i, agent in enumerate(self.agents)
        }
        self._agent_selector = agent_selector(self.agents)
        self.max_collect = {
            agent: max_collect[i] for i, agent in enumerate(self.agents)
        }

        self.scaling, self.translation = self._get_scaling_translation(
            self.point_positions,
            self.agent_positions,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
        )

        self.action_spaces = self._get_action_spaces(
            self.agents, len(self.point_positions)
        )
        self.observation_spaces = self._get_observation_spaces(
            self.agents,
            self.agent_positions,
            self.point_positions,
            self.reveal_cheating_cost,
            self.reveal_collection_reward,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
        )
        self.state_space = self._get_state_space(
            self.agent_positions,
            self.point_positions,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
        )

        # The following are set in reset().
        self.iteration = 0
        self.total_points_collected = 0
        self.points = None
        self.agent_selection = None
        self.has_reset = False
        self.terminate = False
        self.truncate = False
        # Dicts with agent as key.
        self.rewards = None
        self._cumulative_rewards = None
        self.terminations = None
        self.truncations = None
        self.infos = None
        self.collectors = None
        self.cumulative_rewards = None

        # pygame
        self.screen = None
        self.clock = None
        self.surf = None
        self.isopen = False

    def _get_boundary_arrays(self, array_2d, shape):
        """Creates arrays with minimum and maximum with same shape as input.

        Args:
            array_2d (np.ndarray): Input array to find minimum and maximum.
            shape (_type_): Tuple with shape of output arrays.

        Returns:
            np.ndarray: Boundary arrays with minimum and maximum.
        """
        boundary_low = np.full(
            shape, np.min(array_2d, axis=0), dtype=np.float64
        )
        boundary_high = np.full(
            shape, np.max(array_2d, axis=0), dtype=np.float64
        )
        return boundary_low, boundary_high

    def _get_obs_state_space(
        self,
        agent_positions,
        point_positions,
        reveal_cheating_cost,
        reveal_collection_reward,
        screen_width,
        screen_height,
    ):
        """Retrieves global observation/state space.

        Args:
            agent_positions (np.ndarray): Agent positions.
            point_positions (np.ndarray): Point positions.
            reveal_cheating_cost (bool): Whether to include cheating cost.
            reveal_collection_reward (bool): Whether to include collection
                rewards.
            screen_width (int): Width of display screen.
            screen_height (int): Height of display screen.

        Returns:
            gymnasium.spaces.Dict: Dict space with global observation/state.
        """
        n_points = point_positions.shape[0]
        point_boundary_low, point_boundary_high = self._get_boundary_arrays(
            point_positions, shape=(n_points, 2)
        )
        boundary_low, boundary_high = self._get_boundary_arrays(
            np.concatenate((agent_positions, point_positions)),
            shape=(len(agent_positions), 2),
        )

        spaces = {
            "point_positions": gymnasium.spaces.Box(
                low=point_boundary_low,
                high=point_boundary_high,
                dtype=np.float64,
            ),
            "collected": gymnasium.spaces.Box(
                low=0, high=np.inf, shape=(n_points,), dtype=int
            ),
            "collector_positions": gymnasium.spaces.Box(
                low=boundary_low, high=boundary_high, dtype=np.float64
            ),
            "image": gymnasium.spaces.Box(
                low=0,
                high=255,
                shape=(screen_width, screen_height, 3),
                dtype=np.uint8,
            ),
        }

        if reveal_cheating_cost:
            spaces["cheating_cost"] = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(n_points,), dtype=np.float64
            )
        if reveal_collection_reward:
            spaces["collection_reward"] = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(n_points,), dtype=np.float64
            )

        space = gymnasium.spaces.Dict(spaces)

        return space

    def _get_action_spaces(self, agents, n_points):
        """Retrieves action spaces for all agents.

        Each action is a point to collect (by index).

        Args:
            agents (list[str]): List of agent names.
            n_points (int): Number of points.

        Returns:
            dict: Dictionary of discrete action spaces.
        """
        action_spaces = {
            agent: gymnasium.spaces.Discrete(n_points) for agent in agents
        }
        return action_spaces

    def _get_observation_spaces(
        self,
        agents,
        agent_positions,
        point_positions,
        reveal_cheating_cost,
        reveal_collection_reward,
        screen_width,
        screen_height,
    ):
        """Retrieves observation spaces for all agents.

        Each observation consist of the point positions, points collected,
        agent (incl. inactive) positions, and an image of the environment.

        Note:
            These are identical for all agents.

        Args:
            agents (list[str]): List of agent names.
            agent_positions (np.ndarray): Agent positions.
            point_positions (np.ndarray): Point positions.
            reveal_cheating_cost (bool): Whether to include cheating cost.
            reveal_collection_reward (bool): Whether to include collection
                rewards.
            screen_width (int): Width of display screen.
            screen_height (int): Height of display screen.

        Returns:
            dict: Dictionary of observation spaces keyed by agent name.
        """
        observation_spaces = {
            agent: self._get_obs_state_space(
                agent_positions,
                point_positions,
                reveal_cheating_cost,
                reveal_collection_reward,
                screen_width,
                screen_height,
            )
            for agent in agents
        }
        return observation_spaces

    def _get_state_space(
        self, agent_positions, point_positions, screen_width, screen_height
    ):
        """Retrieves state space.

        Args:
            agent_positions (np.ndarray): Agent positions.
            point_positions (np.ndarray): Point positions.
            screen_width (int): Width of display screen.
            screen_height (int): Height of display screen.

        Returns:
            gymnasium.spaces.Dict: State space.
        """
        state_space = self._get_obs_state_space(
            agent_positions,
            point_positions,
            True,
            True,
            screen_width,
            screen_height,
        )
        return state_space

    def _get_scaling_translation(
        self,
        point_positions,
        agent_positions,
        screen_width,
        screen_height,
        relative_padding=0.1,
    ):
        """Returns scaling and translation factors for x- and y-axis.

        Fits data on display while preserving aspect ratio.

        Args:
            point_positions (np.ndarray): Point positions.
            agent_positions (np.ndarray): Agent positions.
            screen_width (int): Width of display screen.
            screen_height (int): Height of display screen.
            relative_padding (float, optional): Outside padding.
                Defaults to 0.1.

        Returns:
            tuple: Tuple of scaling and translation factors for x and y-axis.
                ((scale_x, scale_y), (translate_x, translate_y)).
        """
        assert 0 <= relative_padding < 1, "Relative padding must be in [0,1)."
        pos = np.concatenate((point_positions, agent_positions), axis=0)
        minimum = np.min(pos, axis=0)
        maximum = np.max(pos, axis=0)
        x_min, y_min, x_max, y_max = (
            minimum[0],
            minimum[1],
            maximum[0],
            maximum[1],
        )
        x_range = x_max - x_min
        y_range = y_max - y_min
        max_range = np.max([x_range, y_range, 1])

        upper_x = screen_width * (1 - relative_padding)
        lower_x = screen_width * relative_padding
        scaling_x = (upper_x - lower_x) / max_range
        translation_x = -x_min * scaling_x + lower_x

        upper_y = screen_height * (1 - relative_padding)
        lower_y = screen_height * relative_padding
        scaling_y = (upper_y - lower_y) / max_range
        translation_y = -y_min * scaling_y + lower_y

        return (scaling_x, scaling_y), (translation_x, translation_y)

    def _scale_position(self, position):
        """Scale a position using stored scaling and translation factors.

        Args:
            position (tuple): (x, y) tuple of position.

        Returns:
            tuple: Scaled (x, y) tuple of position.
        """
        x, y = position
        x = x * self.scaling[0] + self.translation[0]
        y = y * self.scaling[1] + self.translation[1]
        return x, y

    def _create_collectors(
        self, agent_positions, agents, scaling, translation
    ):
        """Creates collector for each agent as a dict.

        Args:
            agent_positions (np.ndarray): Agent positions.
            agents (list[str]): List of agent names.
            scaling (tuple): Scaling factors for displaying.
            translation (tuple): Translation factors for displaying.

        Returns:
            dict: Dictionary of collectors keyed by agent name.
        """
        collectors = {
            agent: Collector(
                pos=position,
                scaling=scaling,
                translation=translation,
                id=f"collector_{agent}",
            )
            for agent, position in zip(agents, agent_positions)
        }
        return collectors

    def _create_points(self, point_positions, scaling, translation):
        """Creates points for all given positions.

        Args:
            point_positions (np.ndarray): Point positions.
            scaling (tuple): Scaling factors for displaying.
            translation (tuple): Translation factors for displaying.

        Returns:
            list[Point]: List of points.
        """
        points = [
            Point(
                pos=position,
                scaling=scaling,
                translation=translation,
                id=f"point_{i}",
            )
            for i, position in enumerate(point_positions)
        ]
        return points

    def _create_image_array(self, surf, size):
        """Returns image array from pygame surface.

        Args:
            surf (pygame.Surface): Surface to convert to image array.
            size (tuple): Tuple of (width, height) to scale surface to.

        Returns:
            np.ndarray: Image array.
        """
        scaled_surf = pygame.transform.smoothscale(surf, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_surf)), axes=(1, 0, 2)
        )

    def reward(self, collector, point):
        """Returns reward for collecting a given point.

        Collecting a point triggers a reward, but if the point has already
        been collected, we add a penalty for cheating. The cost of moving
        is the Euclidean distance.

        Args:
            collector (Collector): Collector that collected the point.
            point (Point): Point that is collected.

        Returns:
            float: Reward.
        """
        reward = -np.linalg.norm(collector.position - point.position)
        reward += self.collection_reward(point)
        if point.is_collected():
            reward -= self.cheating_cost(point)
        return reward

    def _state(
        self,
        points,
        collectors,
        reveal_cheating_cost,
        reveal_collection_reward,
    ):
        """Retrieves state of the current global environment.

        Args:
            points (list[Point]): List of points in the environment.
            collectors (dict): Dictionary of collectors.
            reveal_cheating_cost (bool): Whether to reveal cheating cost.
            reveal_collection_reward (bool): Whether to reveal collection
                reward.

        Returns:
            dict: Current global state.
        """
        state = {
            "point_positions": np.array(
                [point.position for point in points], dtype=np.float64
            ),
            "collected": np.array(
                [point.get_collect_counter() for point in points], dtype=int
            ),
            "collector_positions": np.array(
                [collector.position for collector in collectors.values()],
                dtype=np.float64,
            ),
            "image": self._render(render_mode="rgb_array"),
        }
        if reveal_cheating_cost:
            state["cheating_cost"] = np.array(
                [self.cheating_cost(point) for point in points],
                dtype=np.float64,
            )
        if reveal_collection_reward:
            state["collection_reward"] = np.array(
                [self.collection_reward(point) for point in points],
                dtype=np.float64,
            )
        return state

    def observe(self, agent):
        # FIXME: Warning for api_test /Users/lfwa/Library/Caches/pypoetry/
        # virtualenvs/collector-gjPrMD7k-py3.10/lib/python3.10/site-packages/
        # pettingzoo/test/api_test.py:60: UserWarning: Observation is not
        # NumPy array
        # warnings.warn("Observation is not NumPy array")
        return self._state(
            self.points,
            self.collectors,
            self.reveal_cheating_cost,
            self.reveal_collection_reward,
        )

    def state(self):
        return self._state(self.points, self.collectors, True, True)

    def reset(self, seed=None, return_info=False, options=None):
        if seed is not None:
            self.seed(seed)

        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.collectors = self._create_collectors(
            self.agent_positions, self.agents, self.scaling, self.translation
        )
        self.points = self._create_points(
            self.point_positions, self.scaling, self.translation
        )

        self.iteration = 0
        self.total_points_collected = 0
        self.has_reset = True
        self.terminate = False
        self.truncate = False
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.cumulative_rewards = {agent: 0 for agent in self.agents}

        observations = {agent: self.observe(agent) for agent in self.agents}

        if not return_info:
            return observations
        else:
            return observations, self.infos

    def step(self, action):
        assert (
            self.has_reset
        ), "Environment has not been reset yet. Call env.reset() first."

        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            # Guard against first agent dying first since _was_dead_step()
            # does not update agent_selection when that happens.
            if self.agent_selection not in self.agents and self.agents:
                self.agent_selection = self._agent_selector.next()
            return

        if (
            not self.action_space(agent).contains(action)
            and action is not None
        ):
            raise ValueError(f"Action {action} is invalid for agent {agent}.")

        if action is not None:
            point_to_collect = self.points[action]
            collector = self.collectors[agent]
            reward = self.reward(collector, point_to_collect)
            # Move collector to point position.
            collector.move(point_to_collect.position)
            # Only collect point after reward has been calculated.
            collector.collect(point_to_collect, self.total_points_collected)
            self.total_points_collected += 1
        else:
            reward = 0

        # Update termination and truncation for agent.
        if (
            self.collectors[agent].total_points_collected
            >= self.max_collect[agent]
        ):
            self.terminations[agent] = True

        self.terminate = all(self.terminations.values())
        self.truncate = all(self.truncations.values())
        self.iteration += 1

        self.rewards[agent] = reward
        self.cumulative_rewards[agent] += reward
        # Cumulative reward since agent has last acted.
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()

        self.agent_selection = self._agent_selector.next()

        if self.render_mode == "human":
            self.render()

    def render(self):
        assert (
            self.has_reset
        ), "Environment has not been reset yet. Call env.reset() first."

        if self.render_mode is None:
            gymnasium.logger.warn(
                f"No render mode specified, skipping render. Please "
                "specify render_mode as one of the supported modes "
                f"{self.metadata['render_modes']} at initialization."
            )
        else:
            return self._render(render_mode=self.render_mode)

    def _render(self, render_mode):
        """Renders the environment.

        Args:
            render_mode (str): One of the supported render modes.

        Returns:
            np.ndarray or None: Returns the rendered image if render_mode is
                `rgb_array`, otherwise returns None.
        """
        pygame.font.init()
        if self.screen is None and render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (SCREEN_WIDTH, SCREEN_HEIGHT)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        # Add white background.
        self.surf.fill((255, 255, 255))

        self._render_points(self.surf, self.points, POINT_SIZE)
        self._render_paths(self.surf, self.collectors, PATH_SIZE)
        self._render_collectors(
            self.surf, self.collectors, COLLECTOR_LEN, COLLECTOR_SIZE
        )
        # Flip y-axis since pygame has origin at top left.
        self.surf = pygame.transform.flip(self.surf, flip_x=False, flip_y=True)
        self._render_text(self.surf)

        if render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.display.update()
        elif render_mode == "rgb_array":
            return self._create_image_array(
                self.surf, (SCREEN_WIDTH, SCREEN_HEIGHT)
            )

    def _render_text(self, surf):
        """Renders information text, e.g. stats about environment and actions.

        Args:
            surf (pygame.Surface): Surface to render text on.
        """
        (
            stats,
            overall_total_points_collected,
            overall_unique_points_collected,
            overall_cheated,
        ) = self._get_stats()
        total_reward = sum(self.cumulative_rewards.values())
        font = pygame.font.Font(pygame.font.get_default_font(), FONT_SIZE)
        text1 = font.render(
            (
                f"Iteration: {self.iteration} | "
                f"Total points collected: {overall_total_points_collected} | "
                "Unique points collected: "
                f"{overall_unique_points_collected} / {len(self.points)} | "
                f"Cheated: {overall_cheated}"
            ),
            True,
            (0, 0, 255),
        )
        text2 = font.render(
            f"Total cumulative reward: {total_reward}",
            True,
            (0, 0, 255),
        )
        surf.blit(text1, (10, 10))
        surf.blit(text2, (10, 40))

    def _get_stats(self):
        """Retrieves stats for all collectors.

        Returns:
            tuple: Tuple of stats.
        """
        stats = {}
        overall_total_points_collected = 0
        overall_unique_points_collected = 0
        overall_cheated = 0
        for agent in self.collectors:
            collector = self.collectors[agent]
            stats[agent] = {
                "total_points_collected": collector.total_points_collected,
                "unique_points_collected": collector.unique_points_collected,
                "cheated": collector.cheated,
            }
            overall_total_points_collected += collector.total_points_collected
            overall_unique_points_collected += (
                collector.unique_points_collected
            )
            overall_cheated += collector.cheated
        return (
            stats,
            overall_total_points_collected,
            overall_unique_points_collected,
            overall_cheated,
        )

    def _render_points(self, surf, points, point_size):
        """Renders all points as circles.

        Points are colored according to their collector as a pie chart.

        Args:
            surf (pygame.Surface): Surface to render points on.
            points (list[Points]): List of points to render.
            point_size (int): Render size of points.
        """
        for point in points:
            x, y = tuple(point.scaled_position - (point_size / 2))
            bounding_box = pygame.Rect(x, y, point_size, point_size)
            total_collections = point.get_collect_counter()
            start_angle = 0

            if total_collections == 0:
                pygame.draw.circle(
                    surf,
                    point.color,
                    tuple(point.scaled_position),
                    point_size / 2,
                )
            else:
                for (
                    collector_id,
                    collections,
                ) in point.collector_tracker.items():
                    if collections == 0:
                        continue
                    arc_length = collections / total_collections * 2 * math.pi
                    pygame.draw.arc(
                        surf,
                        self.collectors[collector_id[10:]].color,
                        bounding_box,
                        start_angle,
                        start_angle + arc_length,
                        point_size,
                    )
                    start_angle += arc_length

    def _render_paths(self, surf, collectors, path_size):
        """Renders paths taken by collectors.

        Colors are assigned to paths based on the collector that took it.
        If paths overlap then they are colored in segments.

        Args:
            surf (pygame.Surface): Surface to render paths on.
            collectors (dict): Dict of collectors.
            path_size (int): Render size of paths.
        """
        path_pairs = {}
        for collector in collectors.values():
            path_pos_len = len(collector.path_positions)
            if path_pos_len < 2:
                continue
            for i in range(1, path_pos_len):
                key = (
                    tuple(collector.path_positions[i - 1]),
                    tuple(collector.path_positions[i]),
                )
                reverse_key = (key[1], key[0])
                # We should not care whether it is (a, b) or (b, a).
                if key in path_pairs:
                    path_pairs[key] += [collector]
                elif reverse_key in path_pairs:
                    path_pairs[reverse_key] += [collector]
                else:
                    path_pairs[key] = [collector]

        for path, collectors in path_pairs.items():
            total_collectors = len(collectors)
            prev_x, prev_y = self._scale_position(path[0])
            x, y = self._scale_position(path[1])
            segment_x = (x - prev_x) / total_collectors
            segment_y = (y - prev_y) / total_collectors

            for collector in collectors:
                pygame.draw.line(
                    surf,
                    collector.color,
                    (prev_x, prev_y),
                    (prev_x + segment_x, prev_y + segment_y),
                    path_size,
                )
                prev_x += segment_x
                prev_y += segment_y

    def _render_collectors(
        self, surf, collectors, collector_len, collector_size
    ):
        """Renders all collectors as crosses.

        Collectors are rotated when stacked to avoid overlapping.
        Black borders are added to crosses.

        Args:
            surf (pygame.Surface): Surface to render collectors on.
            collectors (dict): Dict of collectors.
            collector_len (int): Length of collector cross.
            collector_size (int): Size of collector cross.
        """
        for position, colls in groupby(
            collectors.values(), lambda col: tuple(col.position)
        ):
            colls = list(colls)
            position = colls[0].scaled_position
            total_collectors = len(colls)
            shift_increment = collector_len / total_collectors
            shift = collector_len / 2

            for i, collector in enumerate(colls):
                cross_rotate_shift = i * shift_increment
                # Add black border to cross.
                border_size = math.ceil(collector_size * 1.7)
                pygame.draw.line(
                    surf,
                    (0, 0, 0),
                    start_pos=(
                        position[0] + cross_rotate_shift - shift,
                        position[1] - shift,
                    ),
                    end_pos=(
                        position[0] + shift - cross_rotate_shift,
                        position[1] + shift,
                    ),
                    width=border_size,
                )
                pygame.draw.line(
                    surf,
                    (0, 0, 0),
                    start_pos=(
                        position[0] + shift,
                        position[1] + cross_rotate_shift - shift,
                    ),
                    end_pos=(
                        position[0] - shift,
                        position[1] + shift - cross_rotate_shift,
                    ),
                    width=border_size,
                )
                # Draw cross.
                pygame.draw.line(
                    surf,
                    collector.color,
                    start_pos=(
                        position[0] + cross_rotate_shift - shift,
                        position[1] - shift,
                    ),
                    end_pos=(
                        position[0] + shift - cross_rotate_shift,
                        position[1] + shift,
                    ),
                    width=collector_size,
                )
                pygame.draw.line(
                    surf,
                    collector.color,
                    start_pos=(
                        position[0] + shift,
                        position[1] + cross_rotate_shift - shift,
                    ),
                    end_pos=(
                        position[0] - shift,
                        position[1] + shift - cross_rotate_shift,
                    ),
                    width=collector_size,
                )

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        self.rng, seed = gymnasium.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()


class SamplingWrapperEnv(raw_env):
    """Wrapper that creates point and agent positions from a sampler.

    Attributes:
        See AECEnv.
    """

    def __init__(
        self,
        n_agents,
        max_collect,
        n_points,
        sampler=lambda rng, n: rng.multivariate_normal(
            np.array([0, 0]), np.array([[1, 0], [0, 1]]), n
        ),
        **kwargs,
    ):
        """Initialize wrapper environment.

        Args:
            n_agents (int): Number of agents to create.
            max_collect (list[int]): List of maximum number of points to
                collect for each agent.
            n_points (int): Number of points to generate.
            sampler (lambda, optional): Lambda function from which n points
                are generated. Should accept an argument `rng` representing a
                random generator (rng, e.g. numpy default_rng) and `n`
                representing number of points to generate. Defaults to a 2D
                standard Normal distribution.
        """
        super().seed()
        assert n_agents > 0, "n_agents must be greater than 0"
        assert n_points > 0, "n_points must be greater than 0"
        self.n_agents = n_agents
        self.n_points = n_points
        self.sampler = sampler
        point_positions = self.sampler(self.rng, self.n_points)
        point_mean = np.mean(point_positions, axis=0)
        init_agent_positions = np.array(
            [point_mean for _ in range(self.n_agents)]
        )
        super().__init__(
            point_positions=point_positions,
            init_agent_positions=init_agent_positions,
            max_collect=max_collect,
            **kwargs,
        )

    def reset(self, seed=None, return_info=False, options=None):
        """Resets the environment to a starting state.

        Args:
            seed (int, optional): Random seed to use for resetting. Defaults
                to None.
            return_info (bool, optional): Whether to return infos. Defaults to
                False.
            options (dict, optional): Additional options. Defaults to None.

        Returns:
            dict: Dictionary of observations for each agent. Infos are
                returned if `return_info` is True.
        """
        return super().reset(
            seed=seed, return_info=return_info, options=options
        )
