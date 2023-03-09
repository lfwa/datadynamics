import networkx as nx
import numpy as np
import pytest

from datadynamics.environments import graph_collector_v0
from datadynamics.policies import greedy_policy_v0


@pytest.mark.parametrize(
    ("graph, point_labels, init_agent_labels, max_collect"),
    [
        (
            nx.from_numpy_array(
                np.array(
                    [
                        [1, 1, 0, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 1, 0, 0, 0, 0],
                        [0, 1, 1, 0, 0, 1, 0, 0, 0],
                        [1, 0, 0, 1, 1, 0, 1, 0, 0],
                        [0, 1, 0, 1, 1, 1, 0, 0, 0],
                        [0, 0, 1, 0, 1, 1, 0, 0, 1],
                        [0, 0, 0, 1, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 1],
                    ]
                )
            ),
            [2, 3, 8],
            [0, 1],
            [3, 3],
        )
    ],
)
def test_graph_collector(
    graph,
    point_labels,
    init_agent_labels,
    max_collect,
):
    env = graph_collector_v0.env(
        graph=graph,
        point_labels=point_labels,
        init_agent_labels=init_agent_labels,
        max_collect=max_collect,
        render_mode="rgb_array",
    )
    policy = greedy_policy_v0.policy(env=env)
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        action = policy.action(observation, agent)
        env.step(action)
