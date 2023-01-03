import numpy as np
from pettingzoo.test import seed_test

from datacollect import collector_v0


def test_seed():
    def env_constructor():
        return collector_v0.env(
            point_positions=np.array([[0, 0], [1, 1]]),
            agent_positions=np.array([[0.5, 0.5], [0.75, 0.75]]),
            max_collect=[10, 15],
        )

    seed_test(env_constructor, num_cycles=10, test_kept_state=False)
