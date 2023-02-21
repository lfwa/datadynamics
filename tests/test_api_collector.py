import numpy as np
import pytest
from pettingzoo.test import api_test

from datacollect.environments import collector_v0


@pytest.mark.parametrize(
    "point_positions, init_agent_positions, max_collect",
    [
        (
            np.array([[i, i] for i in range(100)]),
            np.array([[0, 0], [1, 1]]),
            [50, 80],
        )
    ],
)
def test_api_collector(point_positions, init_agent_positions, max_collect):
    env = collector_v0.env(
        point_positions=point_positions,
        init_agent_positions=init_agent_positions,
        max_collect=max_collect,
    )
    api_test(env, num_cycles=10, verbose_progress=True)
