import networkx as nx
import numpy as np

from datacollect.environments import graph_collector_v0
from datacollect.policies import greedy_policy_v0

"""
Define following graph:

A A N
P N N
N - P

where
A = agents,
N = nodes,
P = collectable points,
- = obstacles.
"""
adjacency_matrix = np.array(
    [
        [1, 1, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1],
    ]
)
graph = nx.from_numpy_array(adjacency_matrix)
points_labels = [2, 3, 8]
init_agent_labels = [0, 1]
max_colllect = [3, 3]

env = graph_collector_v0.env(
    graph=graph,
    point_labels=points_labels,
    init_agent_labels=init_agent_labels,
    max_collect=max_colllect,
    render_mode="human",
)
policy = greedy_policy_v0.policy(env=env)
env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    action = policy.action(observation, agent)
    env.step(action)
