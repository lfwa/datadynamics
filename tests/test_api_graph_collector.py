import networkx as nx
import numpy as np
import pytest
from pettingzoo.test import api_test

from datacollect.environments import graph_collector_v0


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
def test_api_graph_collector(
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
    api_test(env, num_cycles=10, verbose_progress=True)
