import argparse
import os

import matplotlib.pyplot as plt

from datadynamics.utils.metrics.otdd import otdd


def plot_otdd(collections_filename1, collections_filename2, filter_date, ax):
    # Only compute otdd with timestamps as the results are identical to without
    timestamps, distances_with_timestamps = otdd(
        d1_collections_filename=collections_filename1,
        d2_collections_filename=collections_filename2,
        include_timestamps=True,
    )

    ax.plot(timestamps, distances_with_timestamps)
    ax.set_title(filter_date)


def plot_all(
    collections_filenames, output_filename, filter_dates, nrows, ncols, figsize
):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

    for i, (collections_filename1, collections_filename2) in enumerate(
        collections_filenames
    ):
        row = i // ncols
        col = i % ncols
        axis = ax[row, col]
        plot_otdd(
            collections_filename1, collections_filename2, filter_dates[i], axis
        )

    fig.supxlabel("Timestep")
    fig.supylabel("OTDD")
    fig.suptitle("Optimal Transport Dataset Distance Over Time")
    fig.tight_layout()

    fig.savefig(output_filename)


def main(args):
    filter_dates = args.filter_dates.split(" ")

    collections_filenames = [
        (
            os.path.join(
                args.input_dir, date, "results", "premade", "collections.pkl"
            ),
            os.path.join(
                args.input_dir, date, "results", "greedy", "collections.pkl"
            ),
        )
        for date in filter_dates
    ]

    plot_all(
        collections_filenames=collections_filenames,
        output_filename=args.output_file,
        filter_dates=filter_dates,
        nrows=args.nrows,
        ncols=args.ncols,
        figsize=args.figsize,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        help=(
            "Parent directory for simulation results (containing directories "
            "with dates)"
        ),
        required=True,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="File to save plot to",
        required=True,
    )
    parser.add_argument(
        "--filter_dates",
        type=str,
        help="Space-separated list of dates to extract results from",
        required=True,
    )
    parser.add_argument(
        "--nrows",
        type=int,
        help="Number of rows in plot",
        required=True,
    )
    parser.add_argument(
        "--ncols",
        type=int,
        help="Number of columns in plot",
        required=True,
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        help="Figure size",
        required=True,
    )
    args = parser.parse_args()

    main(args)
