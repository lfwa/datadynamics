import math

import numpy as np
from PIL import Image


def _represents_points(pixel, inverted=False):
    # Only intended for binary images.
    return not (pixel ^ inverted)


def from_mask_file(filename, resize=None, inverted=False, flip=False):
    """Returns point labels from an image file representing point mask.

    Args:
        filename (str): Path to image file representing point mask to extract
            point labels from.
        resize (tuple, optional): Resize image to specified width and height.
            Defaults to None.
        inverted (bool, optional): Invert point representations. By default
            (False), 0/False are used to represent points. If enabled (True),
            1/True are used to represent points instead.
        flip (bool, optional): Flip image vertically. Defaults to False.

    Returns:
        list: List of point labels.
    """
    image = Image.open(filename)
    if resize:
        image = image.resize(resize)
    # Convert to binary image.
    image = image.convert("1")
    image_array = np.array(image)
    if flip:
        image_array = np.flip(image_array, axis=0)
    point_labels = from_image_array(image_array, inverted=inverted)
    return point_labels


def from_image_array(image_array, inverted=False):
    """Returns point labels from binary image array.

    The image array is assumed to be binary. We extract indices (point labels)
    in the image array that represent points.

    Args:
        image_array (np.ndarray): Image array to extract point labels from.
        inverted (bool, optional): Invert point representations. By default
            (False), 0/False are used to represent points. If enabled (True),
            1/True are used to represent points instead.

    Returns:
        list: List of point labels.
    """
    image_array = image_array.flatten()
    points = np.array(
        [_represents_points(pixel, inverted=inverted) for pixel in image_array]
    )
    point_labels = np.nonzero(points)[0]
    return list(point_labels)


def from_coordinates(coordinates, width, height, bounding_box, flip_y=True):
    """Returns point labels from coordinates in a width x height grid.

    Args:
        coordinates (list[np.ndarray/tuple]): List of 2D coordinates.
        width (int): Width of grid.
        height (int): Height of grid.
        bounding_box (tuple(float)): Bounding box of grid (lat_min, long_min,
            lat_max, long_max).
        flip_y (bool, optional): Flip y coordinates. Defaults to True.

    Returns:
        list: List of point labels.
    """
    # Latitude, longitude are reversed from x, y coordinates!
    coordinates = np.array(coordinates)
    coordinates = np.flip(coordinates, axis=1)
    # Have to flip bounding box too to fit x, y coordinates.
    y_min, x_min, y_max, x_max = bounding_box
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_step = x_range / width
    y_step = y_range / height

    point_labels = []

    for coord in coordinates:
        x = (
            math.floor((coord[0] - x_min) / x_step)
            if coord[0] != x_max
            else width - 1
        )
        if flip_y:
            y = (
                math.floor((y_max - coord[1]) / y_step)
                if coord[1] != y_min
                else height - 1
            )
        else:
            y = (
                math.floor((coord[1] - y_min) / y_step)
                if coord[1] != y_max
                else height - 1
            )

        label = y * width + x
        point_labels.append(label)

    return point_labels
