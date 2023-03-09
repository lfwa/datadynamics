import networkx as nx
import numpy as np
import tqdm
from PIL import Image


def _represents_obstacle(pixel, inverted=False):
    # Only intended for binary images.
    return not (pixel ^ inverted)


def _get_neighbors(i, j, m, n):
    neighbors = []
    if i > 0:
        neighbors.append((i - 1, j))
    if i < m - 1:
        neighbors.append((i + 1, j))
    if j > 0:
        neighbors.append((i, j - 1))
    if j < n - 1:
        neighbors.append((i, j + 1))
    return neighbors


def from_mask_file(
    filename,
    resize=None,
    default_weight=1.0,
    default_self_loop_weight=0.0,
    inverted=False,
    flip=False,
):
    """Generate graph from an image file representing obstacle mask.

    Args:
        filename (str): Path to image file representing obstaacle mask to
            generate graph from.
        resize (tuple, optional): Resize image to specified width and height.
            Defaults to None (no resizing).
        default_weight (float, optional): Default weight of added edges.
            Defaults to 1.0.
        default_self_loop_weight (float, optional): Default weight of added
            self loops. Defaults to 0.0.
        inverted (bool, optional): Invert obstacle representations. By default
            (False), 0/False are used to represent obstacles, i.e., nodes with
            no incoming or outgoing edges. If enabled (True), 1/True are used
            to represent obstacles instead.
        flip (bool, optional): Flip image vertically. Defaults to False.

    Returns:
        tuple(nx.Graph, dict): Tuple of weighted undirected graph representing
            the image array and a dict containing metadata about the graph.
    """
    image = Image.open(filename)
    if resize:
        image = image.resize(resize)
    # Convert to binary image.
    image = image.convert("1")
    image_array = np.array(image)
    if flip:
        image_array = np.flip(image_array, axis=0)
    graph, metadata = from_image_array(
        image_array,
        default_weight=default_weight,
        default_self_loop_weight=default_self_loop_weight,
        inverted=inverted,
    )
    metadata["obstacle_mask_file"] = filename
    metadata["resize"] = resize
    metadata["flip"] = flip
    return graph, metadata


def from_image_array(
    image_array,
    default_weight=1.0,
    default_self_loop_weight=0.0,
    inverted=False,
):
    """Generate graph from a binary image array.

    Args:
        image_array (np.ndarray): Binary image array to generate graph from.
        default_weight (float, optional): Default weight of added edges.
            Defaults to 1.0.
        default_self_loop_weight (float, optional): Default weight of added
            self loops. Defaults to 0.0.
        inverted (bool, optional): Invert obstacle representations. By default
            (False), 0/False are used to represent obstacles, i.e., nodes with
            no incoming or outgoing edges. If enabled (True), 1/True are used
            to represent obstacles instead.

    Returns:
        tuple(nx.Graph, dict): Tuple of weighted undirected graph representing
            the image array and a dict containing metadata about the graph.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(image_array.size))
    m, n = image_array.shape
    obstacle_node_labels = []
    free_node_labels = []

    # Add weighted edges to graph between adjacent grid cells in image array.
    with tqdm.tqdm(total=m * n, desc="Adding edges to graph") as pbar:
        for i in range(m):
            for j in range(n):
                obstacle = _represents_obstacle(image_array[i, j], inverted)
                node_label = i * n + j
                neighbors = _get_neighbors(i, j, m, n)

                # Add self loop for non-obstacles.
                if not obstacle:
                    free_node_labels.append(node_label)
                    graph.add_edge(
                        node_label, node_label, weight=default_self_loop_weight
                    )

                    # Add edges to neighbors for non-obstacles.
                    for neighbor in neighbors:
                        n_i, n_j = neighbor
                        neighbor_node_label = n_i * n + n_j
                        n_obstacle = _represents_obstacle(
                            image_array[n_i, n_j], inverted
                        )

                        if not n_obstacle:
                            graph.add_edge(
                                node_label,
                                neighbor_node_label,
                                weight=default_weight,
                            )
                else:
                    obstacle_node_labels.append(node_label)
                pbar.update(1)
    metadata = {
        "default_weight": default_weight,
        "default_self_loop_weight": default_self_loop_weight,
        "inverted": inverted,
        "nodes_per_row": image_array.shape[1],
        "nodes_per_col": image_array.shape[0],
        "nodes": len(graph.nodes),
        "obstacle_node_labels": obstacle_node_labels,
        "free_node_labels": free_node_labels,
    }
    return graph, metadata
