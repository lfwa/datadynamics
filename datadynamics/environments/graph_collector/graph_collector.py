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
POINT_SIZE = 7
PATH_SIZE = 2
COLLECTOR_SIZE = 4
COLLECTOR_LEN = 15
FONT_SIZE = 12


def env(**kwargs):
    """Creates a graph collector environment.

    Returns:
        pettingzoo.utils.env.AECEnv: Created environment.
    """
    env = raw_env(**kwargs)
    return env


class raw_env(AECEnv):
    """Raw graph collector environment.

    This environment is based on a weighted (possible directed) graph using
    networkx. The graph represents the environment structure and may define
    obstacles by creating nodes with e.g. no connecting edges as well as
    define collectable points. The weight of each edge defines the cost of
    traversing that edge.
    Agents can move around the graph by traversing edges and collect defined
    points for a reward. Agents can also cheat by collecting points that have
    already been collected. Cheating as well as rewards are
    defined by a user-given function.

    Attributes:
        See AECEnv.
    """

    metadata = {
        "name": "graph_collector",
        "render_modes": ["rgb_array", "human"],
        "is_parrallelizable": False,
        "render_fps": FPS,
    }

    def __init__(
        self,
        graph,
        point_labels,
        init_agent_labels,
        max_collect,
        nodes_per_row=None,
        cheating_cost=lambda point_label: 500 * 0.5,
        collection_reward=lambda point_label: 100,
        reveal_cheating_cost=True,
        reveal_collection_reward=True,
        static_graph=True,
        dynamic_display=False,
        seed=None,
        render_mode=None,
    ):
        """Initializes the graph collector environment.

        Args:
            graph (networkx.Graph): Input directed or undirected graph
                defining the environment. Node labels must be a continuous set
                of integers starting at 0.
            point_labels (list[int]): List of node labels to identify
                collectable points. All duplicate labels will be merged when
                creating points in the environment.
            init_agent_labels (list[int]): List of node labels to identify
                initial agent positions.
            max_collect (list[int]): List of maximum number of points each
                agent can collect.
            nodes_per_row (int, optional): Number of nodes to display per row.
                Defaults to None.
            cheating_cost (function, optional): Function that takes a node
                label and returns the cost of cheating by collecting an
                already collected point. Influences reward for collecting
                points. Defaults to lambda node_label: 500 * 0.5.
            collection_reward (function, optional): Function that takes a point
                label and returns the reward for collecting that point.
                Defaults to lambda point_label: 100.
            reveal_cheating_cost (bool, optional): Whether to reveal the
                cheating costs to the agent in observations. Defaults to True.
            reveal_collection_reward (bool, optional): Whether to reveal the
                collection rewards to the agent in observations. Defaults to
                True.
            static_graph (bool, optional): Whether the underlying graph is
                static and never changes. Significantly impacts performance
                since everything in the environment have to be re-rendered
                for every frame to ensure consistency. May also influence the
                performance of policies as e.g. shortest paths will need to be
                recomputed for every action to determine optimal agent
                movement. Defaults to True.
            dynamic_display (bool, optional): Whether to dynamically adjust
                the display size to the graph size. Defaults to False.
            seed (int, optional): Seed for random number generator. Defaults
                to None.
            render_mode (str, optional): Render mode. Supported modes are
                specified in environment's metadata["render_modes"] dict.
                Defaults to None.
        """
        gymnasium.logger.info("Initializing graph collector environment...")
        assert (
            render_mode in self.metadata["render_modes"] or render_mode is None
        ), (
            f"render_mode: {render_mode} is not supported. "
            f"Supported modes: {self.metadata['render_modes']}"
        )

        gymnasium.logger.info(" - Validating input positions...")
        # Check that all agent and point labels are nodes in the graph.
        assert all(
            label in graph.nodes for label in init_agent_labels
        ), "Agent labels must be nodes in the graph!"
        assert all(
            label in graph.nodes for label in point_labels
        ), "Point labels must be nodes in the graph!"
        # Disallow agents to spawn inside obstacles as they would
        # not be able to move, i.e., agent labels must have neighbors.
        assert all(
            any(True for _ in graph.neighbors(agent_label))
            for agent_label in init_agent_labels
        ), "Agent labels must not encode obstacles!"
        # Allow but issue warning for points inside obstacles as they just
        # cannot be collected.
        if any(
            not any(True for _ in graph.neighbors(point_label))
            for point_label in point_labels
        ):
            gymnasium.logger.warn(
                "Some points are inside obstacles and cannot be collected!"
            )
        # Warn if points are overlapping.
        if len(point_labels) != len(set(point_labels)):
            gymnasium.logger.warn("Some points overlap and will be merged!")

        gymnasium.logger.info(" - Seeding environment...")
        self.seed(seed)

        self.graph = graph
        # Remove duplicate points.
        self._point_labels = list(dict.fromkeys(point_labels))
        self.init_agent_labels = init_agent_labels
        self.render_mode = render_mode
        self.cheating_cost = cheating_cost
        self.collection_reward = collection_reward
        self.reveal_cheating_cost = reveal_cheating_cost
        self.reveal_collection_reward = reveal_collection_reward
        self.static_graph = static_graph

        if nodes_per_row is None:
            nodes_per_row = math.ceil(math.sqrt(len(self.graph.nodes)))
        self.nodes_per_row = nodes_per_row

        # For dynamic displaying, we adjust the screen width s.t.
        # nodes are displayed as squares.
        if dynamic_display:
            global SCREEN_WIDTH
            rows = math.ceil(len(self.graph.nodes) / nodes_per_row)
            aspect_ratio = self.nodes_per_row / rows
            SCREEN_WIDTH = int(SCREEN_HEIGHT * aspect_ratio)

        self.node_width, self.node_height = self._get_node_shape(
            len(self.graph.nodes),
            self.nodes_per_row,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
        )

        self.reward_range = (-np.inf, 0)

        self.agents = [
            f"agent_{i}" for i in range(len(self.init_agent_labels))
        ]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = {
            agent: i for i, agent in enumerate(self.agents)
        }
        self._agent_selector = agent_selector(self.agents)
        self.max_collect = {
            agent: max_collect[i] for i, agent in enumerate(self.agents)
        }

        gymnasium.logger.info(" - Setting up action spaces...")
        self.action_spaces = self._get_action_spaces(
            self.agents, self.graph.nodes
        )
        gymnasium.logger.info(" - Setting up observation spaces...")
        self.observation_spaces = self._get_observation_spaces(
            len(self.graph.nodes),
            len(self._point_labels),
            self.agents,
            self.reveal_cheating_cost,
            self.reveal_collection_reward,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
        )
        gymnasium.logger.info(" - Setting up state space...")
        self.state_space = self._get_state_space(
            len(self.graph.nodes),
            len(self._point_labels),
            len(self.agents),
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
        self.cached_obstacle_surf = None
        self.isopen = False

        gymnasium.logger.info("Environment initialized.")

    def _get_node_shape(
        self, n_nodes, nodes_per_row, screen_width, screen_height
    ):
        """Returns the display width and height of a node.

        Args:
            n_nodes (int): Number of nodes in the graph.
            nodes_per_row (int): Number of nodes to display per row.
            screen_width (int): Width of the display.
            screen_height (int): Height of the display.

        Returns:
            tuple: Tuple containing the display width and height of a node.
        """
        width = screen_width / nodes_per_row
        height = screen_height / math.ceil(n_nodes / nodes_per_row)
        return width, height

    def _get_action_spaces(self, agents, nodes):
        """Retrieves action spaces for all agents.

        Each action is a neighbouring node to move to (by node label) or
        to collect the current point (if the node is defined as one) by
        issuing the `collect` action, -1.

        Args:
            agents (list[str]): List of agent names.
            nodes (list[int]): List of node labels.

        Returns:
            dict: Dictionary of discrete action spaces.
        """
        action_spaces = {
            agent: gymnasium.spaces.Discrete(n=len(nodes) + 1, start=-1)
            for agent in agents
        }

        def sample(mask=None):
            """Generates a sample from the space.

            A sample is a neighbouring node chosen uniformly at random.

            Args:
                mask (np.ndarray, optional): An optimal mask for if an action
                    can be selected where `1` represents valid actions and `0`
                    invalid or infeasible actions. Defaults to None.

            Returns:
                int: Node label of the randomly sampled neighbouring node.
            """
            agent = self.agent_selection
            assert agent is not None, (
                "Agent is required to sample action but none is selected yet. "
                "Did you call reset() before sampling?"
            )
            assert self.collectors is not None, (
                "Collectors are required to sample action but none are "
                "created yet. Did you call reset() before sampling?"
            )

            action_mask, collect_action_validity = self._get_action_mask(agent)
            possible_actions = action_mask.nonzero()[0]
            if collect_action_validity:
                possible_actions = np.append(possible_actions, -1)

            return (
                self.rng.choice(possible_actions)
                if not possible_actions.size == 0
                else None
            )

        # Replace standard sample method s.t. we check for path validity.
        for action_space in action_spaces.values():
            action_space.sample = sample

        return action_spaces

    def _get_observation_spaces(
        self,
        n_nodes,
        n_points,
        agents,
        reveal_cheating_cost,
        reveal_collection_reward,
        screen_width,
        screen_height,
    ):
        """Retrieves observation spaces for all agents.

        Each observation consist of a list of the point and agent positions as
        node labels, collected points, an image representing the environment,
        an action mask representing valid actions for the current agent, and
        whether the agent can issue the `collect` (-1) action.

        Args:
            n_nodes (int): Number of nodes in the graph.
            n_points (int): Number of points in the graph.
            agents (list[str]): List of agent names.
            reveal_cheating_cost (bool): Whether to include cheating cost.
            reveal_collection_reward (bool): Whether to include collection
                rewards.
            screen_width (int): Width of display screen.
            screen_height (int): Height of display screen.

        Returns:
            dict: Dictionary of observation spaces keyed by agent name.
        """
        spaces = {
            # List of node labels, where points/collectors are located.
            "point_labels": gymnasium.spaces.Box(
                low=0, high=n_nodes, shape=(n_points,), dtype=int
            ),
            "collector_labels": gymnasium.spaces.Box(
                low=0, high=n_nodes, shape=(len(agents),), dtype=int
            ),
            # No. of times each point has been collected.
            "collected": gymnasium.spaces.Box(
                low=0, high=np.inf, shape=(n_points,), dtype=int
            ),
            "image": gymnasium.spaces.Box(
                low=0,
                high=255,
                shape=(screen_width, screen_height, 3),
                dtype=np.uint8,
            ),
            # Action mask for the current agent representing valid
            # actions in the current state.
            "action_mask": gymnasium.spaces.Box(
                low=0, high=1, shape=(n_nodes,), dtype=int
            ),
            # Whether it is possible for the current agent to issue
            # the `collect` (-1) action.
            "collect_action_validity": gymnasium.spaces.Discrete(n=2, start=0),
        }

        if reveal_cheating_cost:
            spaces["cheating_cost"] = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(n_points,), dtype=np.float64
            )
        if reveal_collection_reward:
            spaces["collection_reward"] = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(n_points,), dtype=np.float64
            )

        observation_spaces = {
            agent: gymnasium.spaces.Dict(spaces) for agent in agents
        }

        return observation_spaces

    def _get_state_space(
        self, n_nodes, n_points, n_agents, screen_width, screen_height
    ):
        """Retrieves state space.

        The global state consists of a list of the point and agent positions
        as node labels, collected points, and an image representing the
        environment.

        Args:
            n_nodes (int): Number of nodes in the graph.
            n_points (int): Number of points in the graph.
            n_agents (int): Number of agents.
            screen_width (int): Width of display screen.
            screen_height (int): Height of display screen.

        Returns:
            gymnasium.spaces.Dict: State space.
        """
        state_space = gymnasium.spaces.Dict(
            {
                # List of node labels, where points/collectors are located.
                "point_labels": gymnasium.spaces.Box(
                    low=0, high=n_nodes, shape=(n_points,), dtype=int
                ),
                "collector_labels": gymnasium.spaces.Box(
                    low=0, high=n_nodes, shape=(n_agents,), dtype=int
                ),
                # No. of times each point has been collected.
                "collected": gymnasium.spaces.Box(
                    low=0, high=np.inf, shape=(n_points,), dtype=int
                ),
                "image": gymnasium.spaces.Box(
                    low=0,
                    high=255,
                    shape=(screen_width, screen_height, 3),
                    dtype=np.uint8,
                ),
                "cheating_cost": gymnasium.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(n_points,),
                    dtype=np.float64,
                ),
                "collection_reward": gymnasium.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(n_points,),
                    dtype=np.float64,
                ),
            }
        )
        return state_space

    def _get_node_position(
        self, node_label, nodes_per_row, node_width, node_height
    ):
        """Returns the position of a node to be displayed on the screen.

        Args:
            node_label (int): Node label.
            nodes_per_row (int): No. of nodes per row.
            node_width (int): Display width of a node.
            node_height (int): Display height of a node.

        Returns:
            tuple: (x, y) position of the node (with origin at top-left).
        """
        x = (node_label % nodes_per_row) * node_width
        y = (node_label // nodes_per_row) * node_height
        return (x, y)

    def _create_collectors(self, init_agent_labels, agents):
        """Creates collector for each agent as a dict.

        Args:
            init_agent_labels (list[int]): List of node labels representing
                initial agent positions.
            agents (list[str]): List of agent names.

        Returns:
            dict: Dictionary of collectors keyed by agent name.
        """
        collectors = {
            agent: Collector(
                pos=self._get_node_position(
                    node_label=label,
                    nodes_per_row=self.nodes_per_row,
                    node_width=self.node_width,
                    node_height=self.node_height,
                ),
                scaling=0,
                translation=0,
                label=label,
                id=f"collector_{agent}",
            )
            for agent, label in zip(agents, init_agent_labels)
        }
        return collectors

    def _create_points(self, point_labels):
        """Creates points from given node labels.

        Note that this merges all points at the same node into a single
        point.

        Args:
            point_labels (list[int]): Point positions.

        Returns:
            dict: Dictionary of points keyed by node labels.
        """
        points = {
            label: Point(
                pos=self._get_node_position(
                    node_label=label,
                    nodes_per_row=self.nodes_per_row,
                    node_width=self.node_width,
                    node_height=self.node_height,
                ),
                scaling=0,
                translation=0,
                label=label,
                id=f"point_{label}",
            )
            for label in point_labels
        }
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

    def reward(self, cur_node, action):
        """Returns reward for executing action in cur_node.

        If the action is `collect` (-1), we add a reward for collecting the
        point. However, if the point has already been collected, we add a
        penalty for cheating.
        If the action is a label of a new node, the reward is the cost of
        traversing the edge between the current and new node represented by
        the weight of the edge.

        Args:
            cur_node (int): Node label of current node.
            action (int): Action to execute, which is either a node label or
                `collect` (-1).

        Raises:
            ValueError: No edge exists between current and new node of action
                or if the action is `collect` (-1) and cur_node is not a point.

        Returns:
            float: Reward
        """
        if action == -1:
            assert cur_node in self.points, (
                f"Node {cur_node} is not a point. Action cannot be `collect` "
                "(-1)."
            )
            # Add reward for collecting a point, and add penalty if cheating.
            reward = self.collection_reward(cur_node)
            if self.points[cur_node].is_collected():
                reward -= self.cheating_cost(cur_node)
        else:
            try:
                reward = -self.graph.adj[cur_node][action]["weight"]
            except KeyError:
                raise ValueError(
                    f"There is no weighted edge between node {cur_node} and "
                    f"{action}. Reward cannot be calculated."
                )
        return reward

    def _state(
        self,
        graph,
        points,
        collectors,
        reveal_cheating_cost,
        reveal_collection_reward,
    ):
        """Retrieves state of the current global environment.

        Args:
            graph (networkx.Graph): Graph representing the environment.
            points (dict): Dictionary of points keyed by node labels.
            collectors (dict): Dictionary of collectors keyed by agent names.
            reveal_cheating_cost (bool): Whether to reveal cheating cost.
            reveal_collection_reward (bool): Whether to reveal collection
                reward.

        Returns:
            dict: Current global state.
        """
        state = {
            "point_labels": np.array(
                [point.label for point in points.values()], dtype=int
            ),
            "collector_labels": np.array(
                [collector.label for collector in collectors.values()],
                dtype=int,
            ),
            "collected": np.array(
                [point.get_collect_counter() for point in points.values()],
                dtype=int,
            ),
            "image": self._render(render_mode="rgb_array"),
        }
        if reveal_cheating_cost:
            state["cheating_cost"] = np.array(
                [self.cheating_cost(point.label) for point in points.values()],
                dtype=np.float64,
            )
        if reveal_collection_reward:
            state["collection_reward"] = np.array(
                [
                    self.collection_reward(point.label)
                    for point in points.values()
                ],
                dtype=np.float64,
            )
        return state

    def _get_action_mask(self, agent):
        """Retrieves action mask and whether `collect` (-1) can be issued.

        The action mask is an array representing the validity of each action.
        An action is valid if the agent can move to the corresponding node.
        Valid actions are represented by `1`, and invalid actions are
        represented by `0`.
        The `collect` (-1) action is valid if the agent is at a point.

        Args:
            agent (str): Agent name.

        Returns:
            tuple(np.ndarray, bool): Tuple action mask and validity of the
                `collect` (-1) action.
        """
        action_mask = np.zeros(len(self.graph.nodes), dtype=int)
        cur_node = self.collectors[agent].label
        neighbors = self.graph.neighbors(cur_node)
        for neighbor in neighbors:
            action_mask[neighbor] = 1
        collect_action_validity = int(cur_node in self.points)
        return action_mask, collect_action_validity

    def observe(self, agent):
        # FIXME: Warning for api_test /Users/lfwa/Library/Caches/pypoetry/
        # virtualenvs/collector-gjPrMD7k-py3.10/lib/python3.10/site-packages/
        # pettingzoo/test/api_test.py:60: UserWarning: Observation is not
        # NumPy array
        # warnings.warn("Observation is not NumPy array")
        obs = self._state(
            self.graph,
            self.points,
            self.collectors,
            self.reveal_cheating_cost,
            self.reveal_collection_reward,
        )
        (
            obs["action_mask"],
            obs["collect_action_validity"],
        ) = self._get_action_mask(agent)
        return obs

    def state(self):
        return self._state(
            self.graph, self.points, self.collectors, True, True
        )

    def reset(self, seed=None, return_info=False, options=None):
        if seed is not None:
            self.seed(seed)

        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.collectors = self._create_collectors(
            self.init_agent_labels, self.agents
        )
        self.points = self._create_points(self._point_labels)

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
            raise ValueError(
                f"Action {action} is not in the action space for "
                f"agent {agent}."
            )

        action_mask, collect_action_validity = self._get_action_mask(agent)
        collector = self.collectors[agent]
        cur_node = collector.label

        if (
            action is not None
            and action_mask[action]
            or (action == -1 and collect_action_validity)
        ):
            reward = self.reward(cur_node, action)

            if action == -1:
                # Collect point.
                collector.collect(
                    self.points[cur_node], self.total_points_collected
                )
                self.total_points_collected += 1
            else:
                # Move agent to the new node label.
                collector.move(
                    position=self._get_node_position(
                        node_label=action,
                        nodes_per_row=self.nodes_per_row,
                        node_width=self.node_width,
                        node_height=self.node_height,
                    ),
                    label=action,
                )
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
            try:
                self.screen = pygame.display.set_mode(
                    (SCREEN_WIDTH, SCREEN_HEIGHT)
                )
            except Exception as e:
                error_msg = f"Could not initialize pygame display: {e}."
                if self.dynamic_display:
                    error_msg += " Try disabling `dynamic_display`."
                gymnasium.logger.error(error_msg)
                raise e
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        # Add white background.
        self.surf.fill((255, 255, 255))

        self._render_obstacles(
            surf=self.surf,
            nodes=self.graph.nodes,
            nodes_per_row=self.nodes_per_row,
            node_width=self.node_width,
            node_height=self.node_height,
        )
        self._render_points(
            surf=self.surf,
            points=self.points,
            point_size=POINT_SIZE,
        )
        self._render_paths(
            surf=self.surf,
            collectors=self.collectors,
            path_size=PATH_SIZE,
        )
        self._render_collectors(
            surf=self.surf,
            collectors=self.collectors,
            collector_len=COLLECTOR_LEN,
            collector_size=COLLECTOR_SIZE,
        )
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

    def _render_obstacles(
        self, surf, nodes, nodes_per_row, node_width, node_height
    ):
        """Renders obstacles (nodes w. no neighbors) as black rectangles.

        Args:
            surf (pygame.Surface): Surface to render obstacles on.
            nodes (list): List of nodes.
            nodes_per_row (int): No. of nodes to display per row.
            node_width (int): Display width of a node.
            node_height (int): Display height of a node.
        """
        # For static graphs we only have to render obstacles once.
        if self.cached_obstacle_surf is None or not self.static_graph:
            cached_obstacle_surf = pygame.Surface(
                (SCREEN_WIDTH, SCREEN_HEIGHT)
            )
            # Add white background.
            cached_obstacle_surf.fill((255, 255, 255))
            for node in nodes:
                if any(True for _ in self.graph.neighbors(node)):
                    continue
                x, y = self._get_node_position(
                    node_label=node,
                    nodes_per_row=nodes_per_row,
                    node_width=node_width,
                    node_height=node_height,
                )
                rect = pygame.Rect(x, y, node_width, node_height)
                pygame.draw.rect(
                    cached_obstacle_surf,
                    color=(0, 0, 0),
                    rect=rect,
                )
            self.cached_obstacle_surf = cached_obstacle_surf

        surf.blit(self.cached_obstacle_surf, (0, 0))

    def _render_points(self, surf, points, point_size):
        """Renders all points as circles.

        Points are colored according to their collector as a pie chart.

        Args:
            surf (pygame.Surface): Surface to render points on.
            points (list[Points]): List of points to render.
            point_size (int): Render size of a point.
        """
        for point in points.values():
            x, y = self._center(point.position)
            bounding_box = pygame.Rect(
                x - point_size / 2, y - point_size / 2, point_size, point_size
            )
            total_collections = point.get_collect_counter()
            start_angle = 0

            if total_collections == 0:
                pygame.draw.circle(
                    surf,
                    point.color,
                    (x, y),
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
                    collector.path_positions[i - 1],
                    collector.path_positions[i],
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
            prev_x, prev_y = self._center(path[0])
            x, y = self._center(path[1])
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
        self,
        surf,
        collectors,
        collector_len,
        collector_size,
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
            collectors.values(), lambda col: col.position
        ):
            position = self._center(position)
            colls = list(colls)
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

    def _center(self, position):
        """Returns the position centered on the node.

        Args:
            pos (tuple): Position to center.

        Returns:
            tuple: Centered position.
        """
        return (
            position[0] + self.node_width / 2,
            position[1] + self.node_height / 2,
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
