import pytest

from datacollect import collector_v0


@pytest.mark.parametrize(
    "retries, n_points, n_agents, max_collect",
    [
        (2, 100, 1, [10]),
        (2, 100, 2, [10 for _ in range(2)]),
        (2, 100, 3, [10 for _ in range(3)]),
        (2, 100, 5, [10 for _ in range(5)]),
    ],
)
def test_wrapper(retries, n_points, n_agents, max_collect):
    env = collector_v0.env(
        n_points=n_points,
        n_agents=n_agents,
        max_collect=max_collect,
        render_mode="rgb_array",
    )
    for _ in range(retries):
        env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            action = (
                env.action_space(agent).sample() if not termination else None
            )
            env.step(action)
