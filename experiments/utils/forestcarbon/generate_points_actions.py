"""Script to generate points and collector actions for graph_collector_v0.

This script generates point labels used to represent points in a grid-like
environment structure as well as actions (node labels) of each point collected
for each agent. The data is generated from csv files and the width and height
of the grid.
"""
import argparse
import pickle

from datadynamics.utils.graph_utils import point_extractor
from experiments.utils.forestcarbon import parse_data


def main(args):
    """Generate points and collector actions from csv."""
    print("Generating points and collector actions from csv...")
    # First generate overall bounding box.
    (_, _, lat_min, long_min, lat_max, long_max) = parse_data.extract_coords(
        survey_details_filename=args.bbox_survey_details,
        tree_records_filename=args.bbox_tree_records,
        locality=args.locality,
    )
    print(
        f"Bounding box for all points in BBOX files: lat_min: {lat_min},"
        f" long_min: {long_min}, lat_max: {lat_max}, long_max: {long_max}"
    )

    # Generate filtered data.
    (
        collector_indices,
        coords,
        lat_min_filter,
        long_min_filter,
        lat_max_filter,
        long_max_filter,
    ) = parse_data.extract_coords(
        survey_details_filename=args.survey_details,
        tree_records_filename=args.tree_records,
        locality=args.locality,
        filter_dates=args.filter_dates.split(" ")
        if args.filter_dates is not None and args.filter_dates != "all all"
        else None,
    )

    print(
        f"Bounding box found for (possibly filtered) points: lat_min: "
        f"{lat_min_filter}, long_min: {long_min_filter}, lat_max: "
        f"{lat_max_filter}, , long_max: {long_max_filter}"
    )

    point_labels = point_extractor.from_coordinates(
        coordinates=coords,
        width=args.grid_width,
        height=args.grid_height,
        bounding_box=(
            lat_min,
            long_min,
            lat_max,
            long_max,
        ),
    )

    print(
        f"Extracted {len(point_labels)} points from {args.survey_details} and "
        f"{args.tree_records}"
    )

    # Merge points in same grid cell by removing duplicates.
    merged_point_labels = []
    unique_labels = set()
    drop_indices = set()
    for i, label in enumerate(point_labels):
        if label not in unique_labels:
            unique_labels.add(label)
            merged_point_labels.append(label)
        else:
            drop_indices.add(i)

    # Remove duplicates to suit merging and map collector indices to actions
    # (node labels) giving us a dict of goals for each collector/agent.
    for collector in collector_indices:
        labels = [
            point_labels[i]
            for i in collector_indices[collector]
            if i not in drop_indices
        ]
        collector_indices[collector] = labels

    print(
        f"Merged {len(drop_indices)} duplicate points to a new total of "
        f"{len(merged_point_labels)} points."
    )

    with open(args.output_point_labels_file, "wb") as f:
        pickle.dump(merged_point_labels, f)
        print(f"Saved point_labels to {args.output_point_labels_file}.")

    with open(args.output_goal_dict_file, "wb") as f:
        pickle.dump(collector_indices, f)
        print(f"Saved goal dict to {args.output_goal_dict_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--survey_details",
        type=str,
        help="Survey details csv file to extract labels from",
        required=True,
    )
    parser.add_argument(
        "--tree_records",
        type=str,
        help="Tree records csv file to extract labels from",
        required=True,
    )
    parser.add_argument(
        "--bbox_survey_details",
        type=str,
        help="Survey details csv file to extract bounding box from",
        required=True,
    )
    parser.add_argument(
        "--bbox_tree_records",
        type=str,
        help="Tree records csv file to extract bounding box from",
        required=True,
    )
    parser.add_argument(
        "--grid_width",
        type=int,
        help="Width of grid to extract labels from",
        required=True,
    )
    parser.add_argument(
        "--grid_height",
        type=int,
        help="Height of grid to extract labels from",
        required=True,
    )
    parser.add_argument(
        "--locality",
        type=str,
        help="Locality of data to extract labels from",
        required=True,
    )
    parser.add_argument(
        "--filter_dates",
        type=str,
        help="Filter dates to extract labels from, e.g. 2022-10-07 2022-10-07",
        default=None,
    )
    parser.add_argument(
        "--output_point_labels_file",
        type=str,
        help="Output file to save point labels to",
        default="point_labels.pkl",
    )
    parser.add_argument(
        "--output_goal_dict_file",
        type=str,
        help="Output file to save goal dict to",
        default="goal_dict.pkl",
    )
    args = parser.parse_args()

    main(args)
