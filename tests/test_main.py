from collector import collector_v0

import numpy as np


def test_main():
    env = collector_v0.env(
        point_positions=np.random.multivariate_normal(
            np.array([0, 0]), np.array([[1, 0], [0, 1]]), 100
        ),
        agent_positions=np.array([[0.5, 0.5], [0.75, 0.75]]),
        max_collect=[100, 100],
    )
    actions = []
    for i in range(100):
        actions.append(i)
        actions.append(i)
    actions.append(None)
    actions.append(None)
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        action = actions.pop(0)
        print(action)
        env.step(action)


if __name__ == "__main__":
    test_main()
