"""Script to generate point labels for the graph_collector_v0 environment.

This script generates point labels used to represent points in a grid-like
environment structure from a point mask file. Points are encoding by black as
default. The input point mask file should be a binary image or an image that
can be converted to one.
"""
import argparse
import pickle

from datadynamics.utils.graph_utils import point_extractor


def main(args):
    """Generate point labels from point mask file.

    Example usage:
    python -m datadynamics.scripts.graph.generate_points \
        -i ./data/point_mask.png \
        -o ./data/point_labels.pkl \
        -rs 100 100
    """
    point_labels = point_extractor.from_mask_file(
        args.input_file,
        resize=args.resize,
        inverted=args.inverted,
        flip=args.flip,
    )

    with open(args.output_file, "wb") as f:
        pickle.dump(point_labels, f)
        print(f"Saved point_labels to {args.output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        help="Point mask file to extract labels from",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Output file to save point labels to",
        default="point_labels.pkl",
    )
    parser.add_argument(
        "-inv",
        "--inverted",
        action="store_true",
        help="Enable inverted point mask where white are points",
    )
    parser.add_argument(
        "-rs",
        "--resize",
        type=int,
        nargs=2,
        help="Resize image to specified width and height",
    )
    parser.add_argument(
        "-flip",
        "--flip",
        action="store_true",
        help="Vertically flip the image",
    )
    args = parser.parse_args()

    main(args)
