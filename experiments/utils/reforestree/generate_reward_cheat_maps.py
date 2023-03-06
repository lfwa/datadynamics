"""Script to generate reward and cheating cost maps for graph_collector_v0.

Note:
    The reward and cheating costs should be dependent on the size of the
    environment graph! This is because the cost of traversing the graph
    is set by the weight of the edges.
"""

import argparse
import os
import pickle

import numpy as np

from experiments.utils.knn_shapley import get_shapley_value_np


def main(args):
    """Generate reward and cheating cost maps from point labels."""
    print("Generating reward and cheating cost maps...")
    reward_modes = ["uniform", "knn", "lava"]
    cheating_cost_modes = ["none", "mid", "high"]
    reward_min = args.reward_min
    reward_max = args.reward_max

    with open(args.point_labels_file, "rb") as f:
        point_labels = pickle.load(f)

    with open(args.data_file, "rb") as f:
        X_train, y_train, X_test, y_test = pickle.load(f)

    # Generate collection reward maps.
    collection_reward_maps = {}
    collection_reward_maps["uniform"] = {
        label: reward_max for label in point_labels
    }
    collection_reward_maps["knn"] = knn_reward_map(
        point_labels=point_labels,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        new_min=reward_min,
        new_max=reward_max,
    )
    collection_reward_maps["lava"] = lava_reward_map(
        point_labels=point_labels,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        new_min=reward_min,
        new_max=reward_max,
        save_file=f"{args.save_dir}/reward_maps/lava.pkl",
    )

    for reward_mode in reward_modes:
        reward_map = collection_reward_maps[reward_mode]
        with open(f"{args.save_dir}/reward_maps/{reward_mode}.pkl", "wb") as f:
            pickle.dump(reward_map, f)

    # Generate cheating cost maps.
    # Cheating costs should be scaled to the reward and account for cost of
    # traversing, so for a 300x300 graph depending on sparsity it should be
    # around 100.
    cheating_cost_maps = {"none": {}, "mid": {}, "high": {}}
    cheating_cost_maps["none"]["uniform"] = {
        label: 0 for label in point_labels
    }
    cheating_cost_maps["none"]["knn"] = {label: 0 for label in point_labels}
    cheating_cost_maps["none"]["lava"] = {label: 0 for label in point_labels}

    cheating_cost_maps["mid"]["uniform"] = {
        label: reward_min * 0.8 for label in point_labels
    }
    cheating_cost_maps["mid"]["knn"] = {
        label: reward_min * 0.8 for label in point_labels
    }
    cheating_cost_maps["mid"]["lava"] = {
        label: reward_min * 0.8 for label in point_labels
    }

    cheating_cost_maps["high"]["uniform"] = {
        label: reward_max * 10 * 0.5 for label in point_labels
    }
    cheating_cost_maps["high"]["knn"] = {
        label: reward_max * 10 * 0.5 for label in point_labels
    }
    cheating_cost_maps["high"]["lava"] = {
        label: reward_max * 10 * 0.5 for label in point_labels
    }

    for cheating_cost_mode in cheating_cost_modes:
        for reward_mode in reward_modes:
            cheating_map = cheating_cost_maps[cheating_cost_mode][reward_mode]
            with open(
                (
                    f"{args.save_dir}/cheat_maps/{reward_mode}/"
                    f"{cheating_cost_mode}.pkl"
                ),
                "wb",
            ) as f:
                pickle.dump(cheating_map, f)


def knn_reward_map(
    point_labels, X_train, y_train, X_test, y_test, new_min, new_max
):
    """Generate KNN reward map."""
    shapley_values = get_shapley_value_np(
        X_train=X_train.to_numpy(),
        y_train=y_train.to_numpy(),
        X_test=X_test.to_numpy(),
        y_test=y_test.to_numpy(),
    )
    assert len(point_labels) == len(
        shapley_values
    ), "Length mismatch! Number of points must match number of shapley values."
    shapley_values = np.interp(
        shapley_values,
        (shapley_values.min(), shapley_values.max()),
        (new_min, new_max),
    )
    reward_map = {
        point_label: shapley_values[i]
        for i, point_label in enumerate(point_labels)
    }
    return reward_map


def lava_reward_map(
    point_labels, X_train, y_train, X_test, y_test, new_min, new_max, save_file
):
    assert os.path.isfile(save_file), (
        "LAVA reward map must be precomputed outside the repository due to "
        "dependencies failing. See "
        "https://github.com/lfwa/LAVA/blob/main/create_lava_maps.py"
    )
    with open(save_file, "rb") as f:
        reward_map = pickle.load(f)
    return reward_map
    # train_dataset=dataset_from_numpy(X_train.to_numpy(), y_train.to_numpy())
    # test_dataset = dataset_from_numpy(X_test.to_numpy(), y_test.to_numpy())
    # training_size = 1000
    # dual_sol = get_OT_dual_sol(
    #     feature_extractor="euclidean",
    #     trainloader=train_dataset,
    #     testloader=test_dataset,
    #     training_size=training_size,
    # )
    # vals =
    # np.array(lava.values(dual_sol=dual_sol, training_size=training_size))
    # vals = np.interp(vals, (vals.min(), vals.max()), (new_min, new_max))
    # reward_map = {label: vals[i] for i, label in enumerate(point_labels)}
    # return reward_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--point_labels_file",
        type=str,
        help="File storing point labels",
        required=True,
    )
    parser.add_argument(
        "--data_file",
        type=str,
        help="File storing X_train, y_train, X_test, y_test",
        required=True,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory to save maps to",
        required=True,
    )
    parser.add_argument(
        "--reward_min",
        type=int,
        help="Minimum reward value",
        default=50,
    )
    parser.add_argument(
        "--reward_max",
        type=int,
        help="Maximum reward value",
        default=100,
    )
    args = parser.parse_args()

    main(args)
