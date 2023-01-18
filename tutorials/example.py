from datacollect import collector_v0
from datacollect.policies import greedy_policy

env = collector_v0.env(
    n_points=300, n_agents=2, max_collect=[120, 180], render_mode="human"
)
policy = greedy_policy.GreedyPolicy(env=env)
env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    action = policy.action(observation, agent)
    env.step(action)
