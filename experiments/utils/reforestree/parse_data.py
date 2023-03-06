import pandas as pd


def extract_train_test(
    final_dataset_filename, locality, train_limit, test_limit
):
    final_dataset = pd.read_csv(final_dataset_filename)

    class_map = {
        "other": 0,
        "banana": 1,
        "cacao": 2,
        "citrus": 3,
        "fruit": 4,
        "timber": 5,
    }

    # Filter by locality, shuffle, and limit samples.
    final_dataset = final_dataset[final_dataset["site"] == locality]
    final_dataset = final_dataset.sample(frac=1, random_state=42)
    final_dataset = final_dataset[: train_limit + test_limit]

    # Split into train and test.
    train = final_dataset[:train_limit]
    test = final_dataset[train_limit : train_limit + test_limit]

    # We keep the following features:
    # xmin, ymin, xmax, ymax, score, tile_index, lon_d, lat_d
    X_train = pd.DataFrame(
        {
            "xmin": train["xmin"],
            "ymin": train["ymin"],
            "xmax": train["xmax"],
            "ymax": train["ymax"],
            "score": train["score"],
            "tile_index": train["tile_index"],
            "lon_d": train["lon_d"],
            "lat_d": train["lat_d"],
        }
    )
    X_test = pd.DataFrame(
        {
            "xmin": test["xmin"],
            "ymin": test["ymin"],
            "xmax": test["xmax"],
            "ymax": test["ymax"],
            "score": test["score"],
            "tile_index": test["tile_index"],
            "lon_d": test["lon_d"],
            "lat_d": test["lat_d"],
        }
    )
    # Target is the group, we map to integers using class_map.
    y_train = train["group"].map(class_map)
    y_test = test["group"].map(class_map)

    return X_train, y_train, X_test, y_test


def extract_coords(X):
    # Return lat,lon as tuples from X and compute bounding box.
    lat = X["lat_d"].to_numpy()
    lon = X["lon_d"].to_numpy()

    lat_min = lat.min()
    lat_max = lat.max()
    long_min = lon.min()
    long_max = lon.max()

    # Create list of lat, lon tuples.
    coords = list(zip(lat, lon))
    return coords, lat_min, long_min, lat_max, long_max


# def outdated_extract_train_test(
#     field_data_filename,
#     all_annotations_filename,
#     locality,
#     train_limit,
#     test_limit,
# ):
#     # !!!WON'T WORK SINCE LAT-LON TARGETS ARE NOT CLASSES! AND WE NEED IT
# SINCE KNN-SHAPLEY ONLY WORKS FOR CLASSIFICATION MODELS, NOT RERESSION!
#     # Field data contains the training data.
#     # All annotations contains the test data.
#     field_data = pd.read_csv(field_data_filename)
#     all_annotations = pd.read_csv(all_annotations_filename)

#     # Filter by locality and limit number of samples.
#     field_data = field_data[field_data["site"] == locality]
#     field_data = field_data[:train_limit]

#     all_annotations = all_annotations[all_annotations["img_name"] ==
# locality]
#     all_annotations = all_annotations[:test_limit]

#     # Create new train and test dataframes using X, Y as features and
# merging lat, lon as targets.
#     train_data = pd.DataFrame(
#         {
#             "X": field_data["X"],
#             "Y": field_data["Y"],
#         }
#     )

#     print(field_data)

#     print(all_annotations)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = extract_train_test(
        "experiments/reforestree/data/final_dataset.csv",
        locality="Carlos Vera Arteaga RGB",
        train_limit=100,
        test_limit=100,
    )
    extract_coords(X_train)
