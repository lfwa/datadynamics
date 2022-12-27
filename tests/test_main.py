import numpy as np

from collector import collector_v0


def test_main():
    env = collector_v0.env(
        point_positions=np.random.uniform(0, 10, (100, 2)),
        agent_positions=np.array([[0.5, 0.5], [0.75, 0.75]]),
        max_collect=[100, 100],
        render_mode="rgb_array",
    )
    actions = []
    for i in range(100):
        actions.append(i)
        actions.append(99 - i)
    actions.append(None)
    actions.append(None)
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        action = actions.pop(0)
        env.step(action)
