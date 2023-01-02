import numpy as np

from collector import collector_v0
from collector.policies.dummy_policy import DummyPolicy


def test_main():
    env = collector_v0.env(
        point_positions=np.random.uniform(0, 10, (100, 2)),
        agent_positions=np.array([[0.5, 0.5], [0.75, 0.75]]),
        max_collect=[2, 3],
        render_mode="human",
    )
    policy = DummyPolicy(env)
    actions = []
    for i in range(100):
        actions.append(i)
        actions.append(99 - i)
    actions.append(None)
    actions.append(None)
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        action = policy.action(observation, agent)
        env.step(action)
