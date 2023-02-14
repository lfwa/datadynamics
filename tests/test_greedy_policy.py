import numpy as np
import pytest

from datacollect.environments import collector_v0
from datacollect.policies import greedy_policy_v0


@pytest.mark.parametrize(
    (
        "point_positions, init_agent_positions, max_collect, cheat_cost, "
        "caught_probability"
    ),
    [
        (
            np.array([[i, i] for i in range(100)]),
            np.array([[0, 0], [0, 0]]),
            [100, 100],
            1000,
            0.5,
        )
    ],
)
def test_api(
    point_positions,
    init_agent_positions,
    max_collect,
    cheat_cost,
    caught_probability,
):
    env = collector_v0.env(
        point_positions=point_positions,
        init_agent_positions=init_agent_positions,
        max_collect=max_collect,
        cheat_cost=cheat_cost,
        caught_probability=caught_probability,
        render_mode="rgb_array",
    )
    policy = greedy_policy_v0.policy(env=env)
    expected_action_first = 0
    expected_action_last = 1
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        action = policy.action(observation, agent)
        if env.terminations[agent] or env.truncations[agent]:
            assert action is None
        elif env._agent_selector.is_last():
            assert action == expected_action_last
            expected_action_last = min(expected_action_last + 2, 99)

        else:
            assert action == expected_action_first
            expected_action_first = min(expected_action_first + 2, 98)
        env.step(action)
