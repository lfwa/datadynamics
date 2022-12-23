from collector import collector_v0

from pettingzoo.test import api_test
import numpy as np


def test_api():
    env = collector_v0.env(
        point_positions=np.random.multivariate_normal(
            np.array([0, 0]), np.array([[1, 0], [0, 1]]), 100
        ),
        agent_positions=np.array([[0.5, 0.5], [0.75, 0.75]]),
        max_collect=[50, 80],
    )
    api_test(env, num_cycles=10, verbose_progress=True)


if __name__ == "__main__":
    test_api()
