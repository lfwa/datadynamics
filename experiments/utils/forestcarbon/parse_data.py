import numpy as np
import pandas as pd


def extract_coords(
    survey_details_filename, tree_records_filename, locality, filter_dates=None
):
    """Returns parsed data from survey details and tree records csv files.

    Args:
        survey_details_filename (str): File path to survey details csv file.
        tree_records_filename (str): File path to tree records csv file.
        locality (str): Locality to extract data from (other localities are
            dropped).
        filter_dates (list, optional): Range of dates to select (start and end
            are inclusive). Defaults to None.

    Returns:
        tuple: Tuple containing: (1) Dictionary of collectors and their
            respective collected points, (2) List of coordinates for points,
            (3) Bounding box of coordinates.
    """
    survey_details = pd.read_csv(survey_details_filename)
    tree_records = pd.read_csv(
        tree_records_filename,
        parse_dates=["FCD-tree_records-tree_time"],
        dayfirst=True,
        date_parser=lambda x: pd.to_datetime(
            x, errors="coerce", dayfirst=True
        ),
    )

    merged = pd.merge(
        tree_records,
        survey_details,
        how="inner",
        left_on=["PARENT_KEY"],
        right_on="KEY",
    )
    # Only keep those in survey locality.
    merged = merged[merged["FCD-survey_details-locality"] == locality]
    # Drop rows with malformed time stamps.
    merged = merged.dropna(subset=["FCD-tree_records-tree_time"])

    # Map coordinates to numpy arrays.
    # Reverse order of coordinates to match x,y coordinates in environment.
    # We do not reverse but add [::-1] if we want to reverse in lambda!
    merged["FCD-tree_records-tree_coords"] = merged[
        "FCD-tree_records-tree_coords"
    ].map(lambda x: np.array(x.split(","), dtype=np.float32))
    # Drop rows with invalid coordinates (i.e. not 2D).
    merged = merged[
        merged["FCD-tree_records-tree_coords"].map(lambda x: x.shape[0] == 2)
    ]

    # Add collectors, only using team lead since names are spelled differently.
    merged["collectors"] = (
        merged["FCD-survey_details-team_lead"].fillna("")
        # + merged["FCD-survey_details-team_member_1"].fillna("")
        # + merged["FCD-survey_details-team_member_2"].fillna("")
    ).map(lambda x: x.lower().replace(" ", ""))

    # Sort by datetime.
    merged = merged.sort_values(
        by="FCD-tree_records-tree_time", ascending=True
    )

    # Filter dates.
    if filter_dates is not None:
        merged = (
            merged.set_index("FCD-tree_records-tree_time")
            .loc[filter_dates[0] : filter_dates[1]]
            .reset_index()
        )

    # Get minimum and maximum coordinates.
    lat_coords = merged["FCD-tree_records-tree_coords"].map(lambda x: x[0])
    long_coords = merged["FCD-tree_records-tree_coords"].map(lambda x: x[1])
    lat_min = lat_coords.min()
    lat_max = lat_coords.max()
    long_min = long_coords.min()
    long_max = long_coords.max()

    # Dict of collections with their actions (indices into merged).
    collector_actions = merged.reset_index().groupby(by="collectors").indices

    return (
        collector_actions,
        list(merged["FCD-tree_records-tree_coords"]),
        lat_min,
        long_min,
        lat_max,
        long_max,
    )
