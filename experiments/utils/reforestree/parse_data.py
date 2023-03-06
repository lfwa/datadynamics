import pandas as pd


def _get_lon_lat_names(mode):
    lon_name = "lon_g" if mode == "val_dataset" else "lon_d"
    lat_name = "lat_g" if mode == "val_dataset" else "lat_d"
    return lon_name, lat_name


def extract_train_test(
    final_dataset_filename, locality, train_limit, test_limit, mode
):
    final_dataset = pd.read_csv(final_dataset_filename)

    # Filter by locality, shuffle, and limit samples.
    final_dataset = final_dataset[final_dataset["site"] == locality]
    final_dataset = final_dataset.sample(frac=1, random_state=42)
    final_dataset = final_dataset[: train_limit + test_limit]

    # Split into train and test.
    train = final_dataset[:train_limit]
    test = final_dataset[train_limit : train_limit + test_limit]

    lon_name, lat_name = _get_lon_lat_names(mode)

    X_train = pd.DataFrame(
        {
            "lon": train[lon_name],
            "lat": train[lat_name],
        }
    )
    X_test = pd.DataFrame(
        {
            "lon": train[lon_name],
            "lat": train[lat_name],
        }
    )

    if mode == "validation_dataset":
        class_map = {
            True: 1,
            False: 0,
        }
        # Target is the group, we map to integers using class_map.
        y_train = train["is_banana"].map(class_map)
        y_test = test["is_banana"].map(class_map)
    else:
        class_map = {
            "other": 0,
            "banana": 1,
            "cacao": 0,
            "citrus": 0,
            "fruit": 0,
            "timber": 0,
        }
        # Target is the group, we map to integers using class_map.
        y_train = train["group"].map(class_map)
        y_test = test["group"].map(class_map)

    return X_train, y_train, X_test, y_test


def extract_coords(X, mode):
    # Return lat,lon as tuples from X and compute bounding box.
    lat = X["lat"].to_numpy()
    lon = X["lon"].to_numpy()

    lat_min = lat.min()
    lat_max = lat.max()
    long_min = lon.min()
    long_max = lon.max()

    # Create list of lat, lon tuples.
    coords = list(zip(lat, lon))
    return coords, lat_min, long_min, lat_max, long_max
