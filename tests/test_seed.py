import numpy as np
from pettingzoo.test import seed_test

from collector import collector_v0


def test_seed():
    env = collector_v0.env(
        point_positions=np.random.multivariate_normal(
            np.array([0, 0]), np.array([[1, 0], [0, 1]]), 100
        ),
        agent_positions=np.array([[0.5, 0.5], [0.75, 0.75]]),
        max_collect=[50, 80],
    )
    seed_test(env, num_cycles=10, test_kept_state=True)
