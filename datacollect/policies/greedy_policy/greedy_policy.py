import gymnasium
import networkx as nx
import numpy as np

from datacollect.policies.base_policy.base_policy import BasePolicy


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
    """Locally optimal policy using a greedy approach.

    This policy computes the reward for every action in every step and chooses
    the action with the highest reward.
    Compatible only with collector environment.

    Note:
        This is only locally optimal and not globally. Best routes to collect
        all points are disregarded and we only search for the next best point
        for every step.
        Also if the cost of cheating is small then the policy will degenerate
        and always resample the same point.

    Attributes:
        env (pettingzoo.utils.env.AECEnv): Environment used by policy.
    """

    def __init__(self, env):
        self.env = env

    def action(self, observation, agent):
        if self.env.terminations[agent] or self.env.truncations[agent]:
            # Agent is dead, the only valid action is None.
            return None
        best_reward = -np.inf
        best_action = None
        for action, point in enumerate(self.env.points):
            reward = self.env.reward(
                self.env.collectors[agent],
                point,
            )
            if reward > best_reward:
                best_reward = reward
                best_action = action
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
        shortest_paths (dict): Cached shortest paths for all node pairs.
        cur_goals (dict): Cached goals for each agent.
    """

    def __init__(self, env):
        assert (
            env.metadata["name"] == "graph_collector"
        ), "GraphGreedyPolicy is only compatible with graph_collector."

        gymnasium.logger.info("Initializing GraphOptimalPolicy...")
        self.env = env
        if self.env.static_graph:
            gymnasium.logger.info(
                " - Computing and caching shortest paths. This runs in O(V^3) "
                "and may take a while..."
            )
            self.shortest_paths = dict(
                nx.all_pairs_dijkstra_path(self.env.graph)
            )
        # cur_goals consist of (path, point_in_path_collected) keyed by agent.
        self.cur_goals = {}
        gymnasium.logger.info("Completed initialization.")

    def action(self, observation, agent):
        if self.env.terminations[agent] or self.env.truncations[agent]:
            # Agent is dead, the only valid action is None.
            return None

        cur_node = self.env.collectors[agent].label
        goal_path, points_in_goal_path_collected = self.cur_goals.get(
            agent, ([], {})
        )

        # Update shortest paths and reset goals if graph is not static.
        if not self.env.static_graph:
            self.shortest_paths = dict(
                nx.all_pairs_dijkstra_path(self.env.graph)
            )
            goal_path = []
            points_in_goal_path_collected = {}

        # Use cached goal if it exists and the point has not been collected
        #  since last time. Otherwise, find new goal.
        if not goal_path or any(
            [
                collected != self.env.points[p].is_collected()
                for p, collected in points_in_goal_path_collected.items()
            ]
        ):
            best_reward = -np.inf
            goal_path = []
            points_in_goal_path_collected = {}

            for node_label, point in self.env.points.items():
                path = self.shortest_paths.get(cur_node, {}).get(
                    node_label, []
                )
                points_in_path_collected = {}
                # Ensure path exists.
                if len(path) == 0:
                    continue
                elif len(path) == 1:
                    # Ensure there is a self-loop.
                    if not nx.is_path(self.env.graph, [cur_node, path[0]]):
                        continue
                    reward = self.env.reward(cur_node, path[0])
                    # Add to points in path
                    points_in_path_collected[path[0]] = point.is_collected()
                else:
                    reward = 0
                    for i in range(len(path) - 1):
                        reward += self.env.reward(path[i], path[i + 1])
                        if path[i + 1] in self.env.points:
                            points_in_path_collected[
                                path[i + 1]
                            ] = self.env.points[path[i + 1]].is_collected()
                    # Trim first node as it is the current node.
                    path = path[1:]
                if reward > best_reward:
                    best_reward = reward
                    goal_path = path[:]
                    points_in_goal_path_collected = points_in_path_collected
            if not goal_path:
                gymnasium.logger.warn(
                    f"{agent} cannot reach any points and will issue None "
                    "actions."
                )
            self.cur_goals[agent] = (goal_path, points_in_goal_path_collected)

        return goal_path.pop(0) if goal_path else None
