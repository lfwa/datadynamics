from collector import collector_v0


def test_wrapper(retries, n_points_list, n_agents_list, max_collect_list):
    for n_points, n_agents, max_collect in zip(
        n_points_list, n_agents_list, max_collect_list
    ):
        env = collector_v0.env(
            n_points=n_points,
            n_agents=n_agents,
            max_collect=max_collect,
            render_mode="human",
        )
        for _ in range(retries):
            env.reset()
            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()
                action = (
                    env.action_space(agent).sample()
                    if not termination
                    else None
                )
                env.step(action)


if __name__ == "__main__":
    n_agents_list = [1, 2, 3, 5]
    n_points_list = [100 for i in range(len(n_agents_list))]
    max_collect_list = [
        [10 for i in range(n_agents)] for n_agents in n_agents_list
    ]
    test_wrapper(
        retries=2,
        n_points_list=n_points_list,
        n_agents_list=n_agents_list,
        max_collect_list=max_collect_list,
    )
