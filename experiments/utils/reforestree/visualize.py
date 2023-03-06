import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from datacollect.utils.metrics.otdd import otdd


def plot_otdd(collections_dir1, collections_dir2, repeats, ax):
    all_timestamps = []
    all_distances = []
    for repeat in range(1, repeats + 1):
        d1_collections_filename = os.path.join(
            collections_dir1, str(repeat), "collections.pkl"
        )
        d2_collections_filename = os.path.join(
            collections_dir2, str(repeat), "collections.pkl"
        )

        # Only compute otdd with timestamps as the results are identical to without
        timestamps, distances_with_timestamps = otdd(
            d1_collections_filename=d1_collections_filename,
            d2_collections_filename=d2_collections_filename,
            include_timestamps=True,
        )
        all_timestamps.append(timestamps)
        all_distances.append(distances_with_timestamps)

    for timestamps in all_timestamps[1:]:
        assert (
            timestamps == all_timestamps[0]
        ), "Timestamps are not the same across runs."

    mean_distances = np.mean(all_distances, axis=0)
    std_distances = np.std(all_distances, axis=0)

    ax.plot(timestamps, mean_distances)
    ax.fill_between(
        timestamps,
        mean_distances - std_distances,
        mean_distances + std_distances,
        alpha=0.2,
    )


def plot_all(input_dir, output_file, repeats):
    fig, ax = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey="row")

    for i, cheating_cost_mode in enumerate(["high", "mid", "none"]):
        for j, reward_mode in enumerate(["uniform", "knn", "lava"]):
            collections_dir1 = os.path.join(
                input_dir,
                "val_dataset",
                "greedy",
                reward_mode,
                cheating_cost_mode,
            )
            collections_dir2 = os.path.join(
                input_dir,
                "col_dataset",
                "greedy",
                reward_mode,
                cheating_cost_mode,
            )
            axis = ax[i][j]
            plot_otdd(
                collections_dir1=collections_dir1,
                collections_dir2=collections_dir2,
                repeats=repeats,
                ax=axis,
            )

            if i == 0:
                axis.set_title(reward_mode.upper())
            if j == 0:
                axis.set_ylabel(
                    f"{cheating_cost_mode.capitalize()} cheating cost"
                )

    fig.supxlabel("Timestep")
    fig.supylabel("OTDD")
    fig.suptitle("Optimal Transport Dataset Distance Over Time")
    fig.tight_layout()

    fig.savefig(output_file)


def main(args):
    plot_all(
        input_dir=args.input_dir,
        output_file=args.output_file,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        help=("Parent directory for simulation results"),
        required=True,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="File to save plot to",
        required=True,
    )
    parser.add_argument(
        "--repeats", type=int, help="Number of repeats", required=True
    )
    args = parser.parse_args()

    main(args)
