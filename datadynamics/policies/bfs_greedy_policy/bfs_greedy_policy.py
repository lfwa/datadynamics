from collections import deque

import gymnasium
import numpy as np

from datadynamics.policies.base_policy.base_policy import BasePolicy


def policy(**kwargs):
    """Creates a suitable BFS-based greedy policy for a given environment.

    Returns:
        BasePolicy: BFS-based greedy policy.
    """
    policy = BFSGraphGreedyPolicy(**kwargs)
    return policy


class BFSGraphGreedyPolicy(BasePolicy):
    """Greedy policy using a breadth-first search for every action retrieval.

    This policy runs in O(V + E) time when finding a new goal for an agent and
    as such may slow down stepping through the environment.

    Compatible only with graph_collector environment for graphs with equal
    edge weights between nodes.

    Attributes:
        env (pettingzoo.utils.env.AECEnv): Environment used by policy.
        graph (nx.Graph): Graph used by environment.
        cur_goals (dict): Cached goals for each agent consisting of
            (path, collected, point_idx) tuples keyed by agent name.
    """

    def __init__(self, env, graph):
        """Initialize policy from environment.

        Args:
            env (pettingzoo.utils.env.AECEnv): Environment on which to base
                policy.
            graph (nx.Graph): Graph used by environment.
        """
        assert env.metadata["name"] == "graph_collector", (
            f"{self.__class__.__name__} is only compatible with "
            "graph_collector."
        )

        self.env = env
        self.graph = graph
        self.cur_goals = {}

    def _bfs_shortest_paths(self, source_node, graph):
        """Runs breadth-first search to find the shortest paths from a source.

        Note:
            This runs in O(V + E) time.

        Args:
            source_node (int): Label of source node.
            graph (nx.Graph): Graph to search.

        Returns:
            dict: Dictionary of predecessors and depth keyed by node label.
        """
        discovered = {source_node}
        predecessors_and_depth = {source_node: (None, 0)}
        queue = deque([(source_node, 0)])

        while queue:
            node, depth = queue.popleft()

            for neighbor in graph.neighbors(node):
                if neighbor not in discovered:
                    discovered.add(neighbor)
                    queue.append((neighbor, depth + 1))
                    predecessors_and_depth[neighbor] = (node, depth)

        return predecessors_and_depth

    def _find_goal_full(self, observation, agent, graph):
        """Finds the point with the highest reward for an agent.

        Args:
            observation (dict): Observation for agent.
            agent (str): Name of agent.
            graph (nx.Graph): Graph used by environment.

        Returns:
            tuple: Tuple of (path, collected, point_idx) where path is the
                shortest path to the point, collected is the number of times
                the point has been collected, and point_idx is the index of
                the point in the observation.
        """
        agent_idx = int(agent[-1])
        agent_node = observation["collector_labels"][agent_idx]
        predecessors_and_depth = self._bfs_shortest_paths(agent_node, graph)
        best_reward = -np.inf
        best_point_label = None
        best_point_idx = None
        best_point_collected = None

        for i, point_label in enumerate(observation["point_labels"]):
            if point_label not in predecessors_and_depth:
                # Skip unreachable points.
                continue
            collected = observation["collected"][i]
            reward = -predecessors_and_depth[point_label][1]
            if "collection_reward" in observation:
                reward += observation["collection_reward"][i]
            if "cheating_cost" in observation and collected > 0:
                reward -= observation["cheating_cost"][i]

            if reward > best_reward:
                best_reward = reward
                best_point_label = point_label
                best_point_collected = collected
                best_point_idx = i

        if best_point_label is None:
            return None

        # Backtrack to find the path for the best point.
        path = [best_point_label]
        while path[-1] != agent_node and path[-1] is not None:
            path.append(predecessors_and_depth[path[-1]][0])
        path.reverse()
        path = path[1:] + [-1]

        return path, best_point_collected, best_point_idx

    def action(self, observation, agent):
        if self.env.terminations[agent] or self.env.truncations[agent]:
            # Agent is dead, the only valid action is None.
            return None

        goal_path, goal_collected, goal_point_idx = self.cur_goals.get(
            agent, ([], None, None)
        )

        if (
            not goal_path
            or goal_collected != observation["collected"][goal_point_idx]
        ):
            goal_path, goal_collected, goal_point_idx = self._find_goal_full(
                observation, agent, self.graph
            )
            self.cur_goals[agent] = (goal_path, goal_collected, goal_point_idx)

        action = goal_path.pop(0) if goal_path else None

        if action is None:
            gymnasium.logger.warn(
                f"{agent} cannot reach any points and will issue None "
                "actions."
            )

        return action
