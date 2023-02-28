"""Script to generate input graphs to the graph_collector_v0 environment.

This script generates a graph  with a grid-like structure from an obstacle
mask file. Obstacles are encoding by black as default. The input obstacle mask
file should be a binary image or an image that can be converted to one. The
file should be a mask that represents the obstacles in the grid-like
environment that the agent cannot traverse through.
"""
import argparse
import json
import pickle

from datacollect.utils.graph_utils import graph_extractor


def main(args):
    """Generate graph from obstacle mask file.

    Example usage:
    python -m datacollect.scripts.graph.generate_graph \
        -i ./data/obstacle_mask.png \
        -o ./data/graph.pkl \
        -m ./data/metadata.json \
        -rs 100 100 \
        -dfw 1.0
    """
    graph, metadata = graph_extractor.from_mask_file(
        args.input_file,
        resize=args.resize,
        default_weight=args.default_weight,
        default_self_loop_weight=args.default_self_loop_weight,
        inverted=args.inverted,
        flip=args.flip,
    )

    with open(args.output_file, "wb") as f:
        pickle.dump(graph, f)
        print(f"Saved graph to {args.output_file}.")
    with open(args.metadata, "w") as f:
        json.dump(metadata, f, indent=4)
        print(f"Saved metadata to {args.metadata}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        help="Obstacle mask file to generate graph from",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Output file to save graph to",
        default="graph.pkl",
    )
    parser.add_argument(
        "-m",
        "--metadata",
        type=str,
        help="Metadata file to save graph to",
        default="metadata.json",
    )
    parser.add_argument(
        "-inv",
        "--inverted",
        action="store_true",
        help=(
            "Enable inverted obstacle mask where white are obstacles and "
            "black is free space"
        ),
    )
    parser.add_argument(
        "-rs",
        "--resize",
        type=int,
        nargs=2,
        help="Resize image to specified width and height",
    )
    parser.add_argument(
        "-dfw",
        "--default_weight",
        type=float,
        help="Default weight of added edges",
        default=1.0,
    )
    parser.add_argument(
        "-dfslw",
        "--default_self_loop_weight",
        type=float,
        help="Default weight of added self loops",
        default=0.0,
    )
    parser.add_argument(
        "-flip",
        "--flip",
        action="store_true",
        help="Vertically flip the image",
    )
    args = parser.parse_args()

    main(args)
