import numpy as np

from datadynamics.environments import collector_v0
from datadynamics.policies import dummy_policy_v0

env = collector_v0.env(
    point_positions=np.random.uniform(0, 10, (300, 2)),
    init_agent_positions=np.array([[0, 0], [1, 1]]),
    max_collect=[120, 180],
    render_mode="human",
)
policy = dummy_policy_v0.policy(env=env)
env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    action = policy.action(observation, agent)
    env.step(action)
