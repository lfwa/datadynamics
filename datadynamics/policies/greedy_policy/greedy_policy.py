import gymnasium
import networkx as nx
import numpy as np

from datadynamics.policies.base_policy.base_policy import BasePolicy


def policy(**kwargs):
    """Creates a suitable greedy policy for a given environment.

    Returns:
        BasePolicy: Greedy policy.
    """
    if kwargs["env"].metadata["name"] == "graph_collector":
        policy = GraphGreedyPolicy(**kwargs)
    else:
        policy = GreedyPolicy(**kwargs)
    return policy


class GreedyPolicy(BasePolicy):
    """Greedy policy for collector environment.

    This policy computes the expected reward for every action in every step
    and chooses the one with the highest expected reward.

    Compatible only with collector environment.

    Note:
        This is only locally optimal and not globally. Best routes to collect
        all points are disregarded and we only search for the next best point
        for every step.
        The policy may degenerate and always sample the same point if the cost
        of cheating is lower than the reward for collecting a point.

    Attributes:
        env (pettingzoo.utils.env.AECEnv): Environment used by policy.
    """

    def __init__(self, env):
        assert env.metadata["name"] == "collector", (
            f"{self.__class__.__name__} is only compatible with " "collector."
        )

        self.env = env

    def action(self, observation, agent):
        if self.env.terminations[agent] or self.env.truncations[agent]:
            # Agent is dead, the only valid action is None.
            return None

        agent_idx = int(agent[-1])
        cur_position = observation["collector_positions"][agent_idx]

        best_reward = -np.inf
        best_action = None

        for i, position in enumerate(observation["point_positions"]):
            cheating = observation["collected"][i] > 0
            reward = -np.linalg.norm(cur_position - position)

            if "collection_reward" in observation:
                reward += observation["collection_reward"][i]
            if "cheating_cost" in observation and cheating:
                reward -= observation["cheating_cost"][i]

            if reward > best_reward:
                best_reward = reward
                best_action = i

        if best_action is None:
            gymnasium.logger.warn(
                f"{agent} cannot reach any points and will issue None "
                "actions."
            )

        return best_action


class GraphGreedyPolicy(BasePolicy):
    """Greedy policy for graph environment using shortest path.

    Compatible only with graph_collector environment.

    Note:
        This is very slow for non-static graphs as shortest paths are
        computed in O(V^3) time for every action search.
        Static graphs use cached shortest paths.

    Attributes:
        env (pettingzoo.utils.env.AECEnv): Environment used by policy.
        shortest_len_paths (dict): Cached shortest paths including path
            lengths for all node pairs.
        cur_goals (dict): Cached goals for each agent consisting of
            (path, collected, point_idx) tuples keyed by agent name.
    """

    def __init__(self, env):
        assert env.metadata["name"] == "graph_collector", (
            f"{self.__class__.__name__} is only compatible with "
            "graph_collector."
        )

        gymnasium.logger.info("Initializing GraphGreedyPolicy...")
        self.env = env
        if self.env.static_graph:
            gymnasium.logger.info(
                " - Computing and caching shortest paths. This runs in O(V^3) "
                "and may take a while..."
            )
            self.shortest_len_paths = dict(
                nx.all_pairs_dijkstra(self.env.graph)
            )
            # Shortest len paths is a dict with node labels as keys and values
            # consisting of a (length dict, path dict) tuple containing
            # shortest paths between all pairs of nodes.
            self.point_labels = set()
            # cur_goals consist of (path, collected, point_idx) keyed by agent.
            self.cur_goals = {}
        gymnasium.logger.info("Completed initialization.")

    def action(self, observation, agent):
        if self.env.terminations[agent] or self.env.truncations[agent]:
            # Agent is dead, the only valid action is None.
            return None

        if not self.env.static_graph:
            gymnasium.logger.info(
                "Recomputing shortest paths in O(V^3) for non-static graph..."
            )
            self.shortest_len_paths = dict(
                nx.all_pairs_dijkstra(self.env.graph)
            )
            self.cur_goals = {}
            self.point_labels = set()

        if not self.point_labels:
            # For static graphs, the points should not change or change
            # position as such we only need to compute labels once.
            self.point_labels = set(observation["point_labels"])

        agent_idx = int(agent[-1])
        cur_node = observation["collector_labels"][agent_idx]
        goal_path, goal_collected, goal_point_idx = self.cur_goals.get(
            agent, ([], None, None)
        )

        # Update goal if we completed the goal (goal_path is empty) or if
        # the goal was collected by another agent meanwhile.
        if (
            not goal_path
            or goal_collected != observation["collected"][goal_point_idx]
        ):
            best_reward = -np.inf

            for i, point_label in enumerate(observation["point_labels"]):
                path = self.shortest_len_paths.get(cur_node, ({}, {}))[1].get(
                    point_label, []
                )

                if not path:
                    continue

                collected = observation["collected"][i]
                reward = -self.shortest_len_paths.get(cur_node, ({}, {}))[
                    0
                ].get(point_label, np.inf)
                if "collection_reward" in observation:
                    reward += observation["collection_reward"][i]
                if "cheating_cost" in observation and collected > 0:
                    reward -= observation["cheating_cost"][i]

                if reward > best_reward:
                    best_reward = reward
                    # Trim current node and add a `collect` action.
                    goal_path = path[1:] + [-1]
                    goal_collected = collected
                    goal_point_idx = i

            self.cur_goals[agent] = (goal_path, goal_collected, goal_point_idx)

        action = goal_path.pop(0) if goal_path else None

        if action is None:
            gymnasium.logger.warn(
                f"{agent} cannot reach any points and will issue None "
                "actions."
            )

        return action
