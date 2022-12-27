from collector import collector_v0


def test_wrapper():
    env = collector_v0.env(
        n_points=100,
        n_agents=2,
        max_collect=[100, 100],
        render_mode="rgb_array",
    )
    retries = 2
    for _ in range(retries):
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


if __name__ == "__main__":
    test_wrapper()
