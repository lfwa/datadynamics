def extract_dataset(env):
    """Returns a dictionary of point labels collected by each agent.

    Args:
        env (AECEnv): Env to extract point labels from.

    Returns:
        dict: Dictionary of point labels keyed by agent name.
    """
    point_labels_collected = {}
    for agent in env.possible_agents:
        collector = env.collectors[agent]
        point_labels = [point.label for point in collector.points]
        point_labels_collected[agent] = point_labels
    return point_labels_collected
