import networkx as nx
import numpy as np
import tqdm
from PIL import Image


def _represents_points(pixel, inverted=False):
    # Only intended for binary images.
    return not (pixel ^ inverted)


def from_mask_file(filename, resize=None, inverted=False):
    """Returns point labels from an image file representing point mask.

    Args:
        filename (str): Path to image file representing point mask to extract
            point labels from.
        resize (tuple, optional): Resize image to specified width and height.
            Defaults to None.
        inverted (bool, optional): Invert point representations. By default
            (False), 0/False are used to represent points. If enabled (True),
            1/True are used to represent points instead.

    Returns:
        list: List of point labels.
    """
    image = Image.open(filename)
    if resize:
        image = image.resize(resize)
    # Convert to binary image.
    image = image.convert("1")
    image_array = np.array(image)
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
