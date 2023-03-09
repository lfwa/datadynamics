import pickle

import tqdm

try:
    from otdd.pytorch.datasets import dataset_from_numpy
    from otdd.pytorch.distance import DatasetDistance
except ImportError:
    raise ImportError(
        "OTDD or one of its dependecies (likely geomloss) is not installed. "
        "These packages are included in the poetry dev dependencies."
        "Alternatively, you can install geomloss through the fork hosted at "
        "https://github.com/lfwa/geomloss or the original repository at "
        "https://github.com/jeanfeydy/geomloss. Similarly, you can install OTDD through the fork hosted at https://github.com/lfwa/otdd.git or "
        "the original repository at https://github.com/microsoft/otdd."
    )

from datadynamics.utils.post_processing import extract


def otdd(
    d1_collections_filename, d2_collections_filename, include_timestamps=True
):
    """Optimal transport dataset distance between two collections over time.

    The collections must be of equal length and created using
    datadynamics.utils.post_processing.save_collections. We use Microsoft's
    OTDD library to compute the distance between the collections for each
    timestamp to see how the distance changes over time during the simulation.

    Warning:
        This function requires the OTDD library to be installed which is not
        included by default in datadynamics.

    Note:
        We skip any timestamps for which the distance cannot be computed.
        Also, the OTDD values will likely not be affected by whether or not
        timestamps are included in the input features.

    Args:
        d1_collections_filename (str): The filename of the first collection.
        d2_collections_filename (str): The filename of the second collection.
        include_timestamps (bool, optional): Whether to include timestamps in
            the input features. Defaults to True.

    Returns:
        tuple: A tuple of two lists. The first list contains the timestamps
            for which the distance was computed. The second list contains the
            distances for each timestamp.
    """
    with open(d1_collections_filename, "rb") as f:
        d1_collections = pickle.load(f)
    with open(d2_collections_filename, "rb") as f:
        d2_collections = pickle.load(f)

    d1_timestamps, d1_feats, d1_targets = extract.feats_targets_timestamps(
        d1_collections, include_timestamps
    )
    d2_timestamps, d2_feats, d2_targets = extract.feats_targets_timestamps(
        d2_collections, include_timestamps
    )
    n1, n2 = len(d1_timestamps), len(d2_timestamps)
    assert n1 == n2, "The collections must be of equal length."

    completed_timestamps = []
    distances = []

    for i in tqdm.tqdm(range(1, n1 + 1), desc="Computing OTDD"):
        try:
            d1 = dataset_from_numpy(d1_feats[:i], d1_targets[:i])
            d2 = dataset_from_numpy(d2_feats[:i], d2_targets[:i])
            dist = DatasetDistance(d1, d2, inner_ot_method="exact")
            d = dist.distance(maxsamples=1000)
            distances.append(d.item())
            completed_timestamps.append(d1_timestamps[i - 1])
        except Exception as e:
            print(f"Skipping {i} due to {e}...")
            continue

    return completed_timestamps, distances
