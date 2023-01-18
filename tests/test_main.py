import numpy as np

from datacollect import collector_v0
from datacollect.policies import dummy_policy_v0


def test_main():
    env = collector_v0.env(
        point_positions=np.random.uniform(0, 10, (100, 2)),
        agent_positions=np.array([[0.5, 0.5], [0.75, 0.75]]),
        max_collect=[110, 102],
        render_mode="rgb_array",
    )
    policy = dummy_policy_v0.policy(env=env)
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        action = policy.action(observation, agent)
        env.step(action)
