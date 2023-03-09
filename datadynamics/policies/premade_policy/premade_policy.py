from collections import deque

import gymnasium

from datadynamics.policies.base_policy.base_policy import BasePolicy


def policy(**kwargs):
    """Creates a premade policy for a given environment

    Returns:
        BasePolicy: Premade policy.
    """
    policy = PremadePolicy(**kwargs)
    return policy


class PremadePolicy(BasePolicy):
    """Policy using a premade list of goals for each agent.

    This policy runs in O(V + E) time when reaching a goal due to having to
    search for the shortest path to the given goal.

    Compatible only with graph_collector environment for graphs with equal
    edge weights between nodes.

    Attributes:
        env (pettingzoo.utils.env.AECEnv): Environment used by policy.
        graph (nx.Graph): Graph used by environment.
        cur_goals (dict): Cached goals for each agent consisting of
            (path, collected, point_idx) tuples keyed by agent name.
    """

    def __init__(self, env, graph, goal_dict):
        """Initialize policy from environment.

        Args:
            env (pettingzoo.utils.env.AECEnv): Environment on which to base
                policy.
            graph (nx.Graph): Graph used by environment.
            goal_dict (dict): Dictionary of goals for each agent (keys can be
                arbitrary).
        """
        assert env.metadata["name"] == "graph_collector", (
            f"{self.__class__.__name__} is only compatible with "
            "graph_collector."
        )

        self.env = env
        self.graph = graph
        self.cur_goals = {}
        self.goal_dict = {}

        assert len(goal_dict) == len(env.possible_agents), (
            f"You must provide only one list of goals for each agent. "
            f"Provided {len(goal_dict)} goals for {len(env.possible_agents)} "
            "agents."
        )
        for agent, key in zip(env.possible_agents, goal_dict):
            # Copy goals since we pop from them.
            self.goal_dict[agent] = goal_dict[key][:]

    def _bfs_shortest_path(self, source_node, target, graph):
        """Runs BFS to find the shortest path from source to target.

        Note:
            This runs in O(V + E) time in the worst case.

        Args:
            source_node (int): Label of source node.
            target (int): Label of target node.
            graph (nx.Graph): Graph to search.

        Returns:
            list: Shortest path from source to target

        Raises:
            ValueError: If no path exists from source to target.
        """
        discovered = {source_node}
        predecessors_and_depth = {source_node: (None, 0)}
        queue = deque([(source_node, 0)])

        while queue:
            node, depth = queue.popleft()

            if node == target:
                # Reconstruct path
                path = [target]
                while path[-1] != source_node and path[-1] is not None:
                    path.append(predecessors_and_depth[path[-1]][0])
                path.reverse()
                path = path[1:] + [-1]
                return path

            for neighbor in graph.neighbors(node):
                if neighbor not in discovered:
                    discovered.add(neighbor)
                    queue.append((neighbor, depth + 1))
                    predecessors_and_depth[neighbor] = (node, depth)

        raise ValueError(f"No path exists from {source_node} to {target}.")

    def action(self, observation, agent):
        if self.env.terminations[agent] or self.env.truncations[agent]:
            # Agent is dead, the only valid action is None.
            if self.goal_dict[agent]:
                gymnasium.logger.warn(
                    f"Agent {agent} is dead but has goals remaining."
                )
            return None

        goal_path = self.cur_goals.get(agent, ([]))

        if not goal_path:
            agent_idx = int(agent[-1])
            agent_node = observation["collector_labels"][agent_idx]
            goal_path = self._bfs_shortest_path(
                agent_node, self.goal_dict[agent].pop(0), self.graph
            )
            self.cur_goals[agent] = goal_path

        action = goal_path.pop(0)

        return action
