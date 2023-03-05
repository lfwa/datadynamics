import pickle

import numpy as np


def collections(env):
    """Returns a dict of point labels collected by each agent with timestamps.

    Args:
        env (AECEnv): Env to extract point labels from.

    Returns:
        dict: Dict of point labels together with timestamps keyed by agent
            name.
    """
    colls = {}
    for agent in env.possible_agents:
        collector = env.collectors[agent]
        points_and_timestamps = [
            [point.label, timestamp] for point, timestamp in collector.points
        ]
        colls[agent] = points_and_timestamps
    return colls


def save_collections(env, filename):
    """Extracts and saves point labels collected by each agent with timestamps.

    Args:
        env (AECEnv): Env to extract point labels from.
        filename (str): Filename to save point labels to.
    """
    colls = collections(env)
    with open(filename, "wb") as f:
        pickle.dump(colls, f)
        print(f"Saved collections to {filename}.")


def feats_targets_timestamps(collections, include_timestamps=True):
    """Extracts features, targets and timestamps from collections.

    Args:
        collections (dict): Dict of point labels together with timestamps keyed
            by agent name.
        include_timestamps (bool, optional): Whether to include timestamps in
            features. Defaults to True.

    Returns:
        tuple: Tuple of timestamps, features and targets.
    """
    timestamps = []
    feats = []
    targets = []
    for i, feat in enumerate(collections.values()):
        timestamps += [f[-1] for f in feat]
        if not include_timestamps:
            feat = [f[:-1] for f in feat]
        feats += feat
        targets += [i] * len(feat)
    sort_indices = np.argsort(timestamps)
    feats = np.array(feats)[sort_indices]
    targets = np.array(targets)[sort_indices]
    timestamps = np.array(timestamps)[sort_indices]
    return timestamps, feats, targets
