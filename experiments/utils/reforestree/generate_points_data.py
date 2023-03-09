import argparse
import pickle

from datadynamics.utils.graph_utils import point_extractor
from experiments.utils.reforestree import parse_data


def main(args):
    X_train, y_train, X_test, y_test = parse_data.extract_train_test(
        final_dataset_filename=args.final_dataset_file,
        locality=args.locality,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        mode=args.mode,
    )

    with open(args.output_data_file, "wb") as f:
        pickle.dump((X_train, y_train, X_test, y_test), f)
        print(
            "Saved X_train, y_train, X_test, y_test to "
            f"{args.output_data_file}."
        )

    # We will only simulate on X_train since those are the ones we retrieve
    # values for with KNN-shapley and LAVA.
    coords, lat_min, long_min, lat_max, long_max = parse_data.extract_coords(
        X_train, mode=args.mode
    )

    point_labels = point_extractor.from_coordinates(
        coordinates=coords,
        width=args.grid_width,
        height=args.grid_height,
        bounding_box=(lat_min, long_min, lat_max, long_max),
    )

    print(
        f"Extracted {len(point_labels)} points from {args.final_dataset_file} "
        f"with limit {args.train_limit}"
    )

    with open(args.output_point_labels_file, "wb") as f:
        pickle.dump(point_labels, f)
        print(f"Saved point labels to {args.output_point_labels_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final_dataset_file",
        type=str,
        help="File storing final dataset to extract data from",
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
        "--train_limit",
        type=int,
        help="Limit on number of training points to extract",
        default=100,
    )
    parser.add_argument(
        "--test_limit",
        type=int,
        help="Limit on number of testing points to extract",
        default=100,
    )
    parser.add_argument(
        "--output_point_labels_file",
        type=str,
        help="Output file to save point labels to",
        default="point_labels.pkl",
    )
    parser.add_argument(
        "--output_data_file",
        type=str,
        help="Output file to save X_train, y_train, X_test, y_test tuple to",
        default="data.pkl",
    )
    parser.add_argument(
        "--mode",
        choices=["val_dataset", "col_dataset"],
        help="Mode to extract data from",
        required=True,
    )
    args = parser.parse_args()

    main(args)
