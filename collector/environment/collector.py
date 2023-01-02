import gymnasium
import numpy as np
import pygame
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import AECEnv

from collector.utils.objects import Collector, Point

FPS = 120
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
# Rendering sizes.
POINT_SIZE = 4
PATH_SIZE = 2
COLLECTOR_SIZE = 4
COLLECTOR_LEN = 5
FONT_SIZE = 20


def env(**kwargs):
    if "n_points" in kwargs:
        env = SamplingWrapperEnv(**kwargs)
    else:
        env = raw_env(**kwargs)
    return env


class raw_env(AECEnv):
    metadata = {
        "name": "collector",
        "render_modes": ["rgb_array", "human"],
        "is_parrallelizable": False,
        "render_fps": FPS,
    }

    def __init__(
        self,
        point_positions,
        agent_positions,
        max_collect,
        cheat_cost=500,
        caught_probability=0.5,
        render_mode=None,
    ):
        assert (
            render_mode in self.metadata["render_modes"] or render_mode is None
        ), (
            f"render_mode: {render_mode} is not supported. "
            f"Supported modes: {self.metadata['render_modes']}"
        )

        self.seed()

        # Positions should be a np.ndarray w. shape (n, 2) representing
        # n (x, y) coordinates.
        self.point_positions = point_positions
        self.agent_positions = agent_positions
        self.render_mode = render_mode
        self.cheat_cost = cheat_cost
        self.caught_probability = caught_probability

        self.reward_range = (-np.inf, 0)

        self.agents = [f"agent_{i}" for i in range(len(agent_positions))]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = {
            agent: i for i, agent in enumerate(self.agents)
        }
        self._agent_selector = agent_selector(self.agents)
        self.max_collect = {
            agent: max_collect[i] for i, agent in enumerate(self.agents)
        }

        self.scaling, self.translation = self._compute_scaling_and_translation(
            point_positions, agent_positions, SCREEN_WIDTH, SCREEN_HEIGHT
        )

        self.state_space = self._get_state_space(
            agent_positions, point_positions, SCREEN_WIDTH, SCREEN_HEIGHT
        )
        self.action_spaces = self._get_action_spaces(
            self.agents, len(point_positions)
        )
        self.observation_spaces = self._get_observation_spaces(
            self.agents,
            agent_positions,
            point_positions,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
        )

        # The following are set in reset().
        self.iteration = 0
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

    def _create_boundary_arrays(self, array_2d, shape):
        """Create arrays with minimum and maximum with same shape as input."""
        boundary_low = np.full(shape, np.min(array_2d, axis=0))
        boundary_high = np.full(shape, np.max(array_2d, axis=0))
        return boundary_low, boundary_high

    def _get_state_space(
        self, agent_positions, point_positions, screen_width, screen_height
    ):
        """Retrieve state space."""
        n_points = point_positions.shape[0]
        point_boundary_low, point_boundary_high = self._create_boundary_arrays(
            point_positions, shape=(n_points, 2)
        )
        boundary_low, boundary_high = self._create_boundary_arrays(
            np.concatenate((agent_positions, point_positions)),
            shape=(len(agent_positions), 2),
        )

        state_space = gymnasium.spaces.Dict(
            {
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
        )
        return state_space

    def _get_action_spaces(self, agents, n_points):
        """Retrieve action spaces for all agents.

        Each action is a point to collect (by index).
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
        screen_width,
        screen_height,
    ):
        """Retrieve observation spaces for all agents.

        Each observation consist of the point positions, points collected,
        agent (incl. inactive) positions, and an image of the environment.
        Note that these are identical for all agents.
        """
        n_points = point_positions.shape[0]
        point_boundary_low, point_boundary_high = self._create_boundary_arrays(
            point_positions, shape=(n_points, 2)
        )
        boundary_low, boundary_high = self._create_boundary_arrays(
            np.concatenate((agent_positions, point_positions)),
            shape=(len(agents), 2),
        )

        observation_spaces = {
            agent: gymnasium.spaces.Dict(
                {
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
            )
            for agent in agents
        }
        return observation_spaces

    def _compute_scaling_and_translation(
        self,
        point_positions,
        agent_positions,
        screen_width,
        screen_height,
        relative_padding=0.1,
    ):
        """Compute scaling and translation to fit all points and agents on
        screen while preserving aspect ratio of data."""
        assert 0 <= relative_padding < 1, "Relative padding must be in [0,1[."
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
        # Scale to fit within [a,b] while preserving aspect ratio.
        if x_range > y_range and x_range > 0:
            b = screen_width * (1 - relative_padding)
            a = screen_width * relative_padding
            scaling = (b - a) / x_range
            translation = -x_min * scaling + a
        elif y_range > 0:
            b = screen_height * (1 - relative_padding)
            a = screen_height * relative_padding
            scaling = (b - a) / y_range
            translation = -y_min * scaling + a
        else:
            scaling = 1
            translation = 0
        return scaling, translation

    def _create_collectors(
        self, agent_positions, agents, scaling, translation
    ):
        """Create collectors for all agents as a dict."""
        collectors = {
            agent: Collector(position, scaling, translation)
            for agent, position in zip(agents, agent_positions)
        }
        return collectors

    def _create_points(self, point_positions, scaling, translation):
        """Create points for all given positions."""
        points = [
            Point(position, scaling, translation)
            for position in point_positions
        ]
        return points

    def _create_image_array(self, surf, size):
        """Create image (numpy) array from pygame surface."""
        scaled_surf = pygame.transform.smoothscale(surf, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_surf)), axes=(1, 0, 2)
        )

    def cheating_cost(self, point):
        """Returns cost of cheating for a given point."""
        return self.cheat_cost * self.caught_probability

    def reward(self, collector, point):
        """Returns reward for collecting a given point."""
        # Use Euclidean distance as initial cost.
        cost = np.linalg.norm(collector.position - point.position)
        if point.is_collected():
            cost += self.cheating_cost(point)
        # Return negated cost as reward since we are using a cost-based model.
        return -cost

    def _state(self, points, collectors):
        """Retrieve observation of the global environment."""
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
        return state

    def observe(self, agent):
        """Returns the observation an agent currently can make.

        This is identical for all agents.
        """
        # TODO: Warning for api_test /Users/lfwa/Library/Caches/pypoetry/virtualenvs/collector-gjPrMD7k-py3.10/lib/python3.10/site-packages/pettingzoo/test/api_test.py:60: UserWarning: Observation is not NumPy array
        # warnings.warn("Observation is not NumPy array")
        return self._state(self.points, self.collectors)

    def state(self):
        """Returns a global view of the environment."""
        return self._state(self.points, self.collectors)

    def reset(self, seed=None, return_info=False, options=None):
        """Resets the environment to a starting state."""
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
        self.has_reset = True
        self.terminate = False
        self.truncate = False
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.cumulative_rewards = {agent: 0 for agent in self.agents}

        obs = self._state(self.points, self.collectors)
        observations = {agent: obs for agent in self.agents}

        if not return_info:
            return observations
        else:
            return observations, self.infos

    def step(self, action):
        """Advances the environment by one step for the current agent.

        Automatically switches control to the next agent."""
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

        if not self.action_space(agent).contains(action):
            raise ValueError(f"Action {action} is invalid for agent {agent}.")

        point_to_collect = self.points[action]
        collector = self.collectors[agent]
        reward = self.reward(collector, point_to_collect)
        # Move collector to point position.
        collector.position = point_to_collect.position
        # Only collect point after reward has been calculated.
        collector.collect(point_to_collect)

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
        """Renders the environment as specified by self.render_mode."""
        if self.render_mode is None:
            gymnasium.logger.warn(
                (
                    f"No render mode specified, skipping render. Please "
                    "specify render_mode as one of the supported modes "
                    f"{self.metadata['render_modes']} at initialization."
                )
            )
        else:
            return self._render(render_mode=self.render_mode)

    def _render(self, render_mode):
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

        # TODO: All collectors and paths are rendered even for inactive
        # agents. Should this happen? Or should we render inactive agents
        # differently?
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
        """Render info text."""
        # TODO: Render each text by itself since whole string will move around due to size differences in character length.
        (
            stats,
            overall_total_points_collected,
            overall_unique_points_collected,
            overall_cheated,
        ) = self._get_stats()
        total_reward = sum(self.cumulative_rewards.values())
        font = pygame.font.Font(pygame.font.get_default_font(), FONT_SIZE)
        text1 = font.render(
            f"Iteration: {self.iteration} | Total points collected: {overall_total_points_collected} | Unique points collected: {overall_unique_points_collected} / {len(self.points)} | Cheated: {overall_cheated}",
            True,
            (0, 0, 0),
        )
        text2 = font.render(
            f"Total cumulative reward: {total_reward}",
            True,
            (0, 0, 0),
        )
        surf.blit(text1, (10, 10))
        surf.blit(text2, (10, 40))

    def _get_stats(self):
        """Get stats for all collectors."""
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
        """Render all points as circles"""
        # TODO: Problem: Multiple collectors have taken the same path, only latest will be rendered!
        for point in points:
            pygame.draw.circle(
                surf,
                point.color,
                tuple(point.scaled_position),
                point_size,
            )

    def _render_paths(self, surf, collectors, path_size):
        """Render paths taken between collections of points."""
        # TODO: Problem: Multiple collectors have taken the same path, only latest will be rendered!
        for collector in collectors.values():
            for i in range(1, len(collector.points)):
                pygame.draw.line(
                    surf,
                    collector.color,
                    collector.points[i - 1].scaled_position,
                    collector.points[i].scaled_position,
                    path_size,
                )

    def _render_collectors(
        self, surf, collectors, collector_len, collector_size
    ):
        """Render all collectors as crosses."""
        for collector in collectors.values():
            pygame.draw.line(
                surf,
                collector.color,
                start_pos=tuple(collector.scaled_position - collector_len),
                end_pos=tuple(collector.scaled_position + collector_len),
                width=collector_size,
            )
            pygame.draw.line(
                surf,
                collector.color,
                start_pos=(
                    collector.scaled_position[0] - collector_len,
                    collector.scaled_position[1] + collector_len,
                ),
                end_pos=(
                    collector.scaled_position[0] + collector_len,
                    collector.scaled_position[1] - collector_len,
                ),
                width=collector_size,
            )

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        """Set random seed."""
        self.rng, seed = gymnasium.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        """Close pygame display if exists."""
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()


class SamplingWrapperEnv(raw_env):
    """Wrapper that creates point and agent positions from a sampler."""

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
        super(SamplingWrapperEnv, self).seed()
        assert n_agents > 0, "n_agents must be greater than 0"
        assert n_points > 0, "n_points must be greater than 0"
        self.n_agents = n_agents
        self.n_points = n_points
        self.sampler = sampler
        point_positions = self.sampler(self.rng, self.n_points)
        point_mean = np.mean(point_positions, axis=0)
        agent_positions = np.array([point_mean for _ in range(self.n_agents)])
        super(SamplingWrapperEnv, self).__init__(
            point_positions=point_positions,
            agent_positions=agent_positions,
            max_collect=max_collect,
            **kwargs,
        )

    def reset(self, resample=True, seed=None, return_info=False, options=None):
        """Resets the environment to a starting state.

        Resamples point and agent positions if resample is True."""
        if resample:
            self.point_positions = self.sampler(self.rng, self.n_points)
            point_mean = np.mean(self.point_positions, axis=0)
            self.agent_positions = np.array(
                [point_mean for _ in range(self.n_agents)]
            )
        return super(SamplingWrapperEnv, self).reset(
            seed=seed, return_info=return_info, options=options
        )
