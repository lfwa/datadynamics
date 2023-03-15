# Simulating Data Collection Dynamics with a Graph

In this tutorial, we will explore how to use Datadynamics to simulate data collection tasks in a graph-based environment.


## Graph-based Environments

The graph-based environment encodes its structure as a weighted and possibly directed graph, supporting both an underlying dynamic and changing graph as well as static graphs.
Agents can traverse the graph to collect data points for rewards, where the cost of traversing is determined by the edge weights.
The graph encoding enables the creation of almost all environment structures, including the approximation of the plane-based environment through a grid or navigation mesh-like structure.
However, this comes at a performance cost due to the increased number of objects to render and the often high time complexity of graph algorithms used for pathfinding in some policies.

## Defining a Graph

In Datadynamics, graphs should be encoded as a weighted (directed or undirected) graph using the [NetworkX](https://networkx.org/) library.
Each node in the graph represents a location that the agents can occupy, and each edge represents a connection between two locations.
The weight of an edge represents the cost (negated reward) of traversing it.

The graphs should also contain node that represent points that agents can collect.

Graphs can be created in a variety of different ways as specified in the [NetworkX documentation](https://networkx.org/).
A simple way to create a graph is from an [adjacency matrix](https://en.wikipedia.org/wiki/Adjacency_matrix) indicated whether pairs of nodes are adjacent or not in the graph.
You can e.g. use the following code:

```python
import networkx as nx
import numpy as np

adjacency_matrix = np.array(
    [
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
    ]
)

graph = nx.from_numpy_array(adjacency_matrix)
```

In this example, we define an undirected graph with four nodes that are all locally connected.
Each edge has a weight of 1.

## Example of Graph-collector Environment
In the following example, we will define a simple graph with nine nodes that are all locally connected.
Three of the nodes define collectable points that the two agents aim to collect.
The behavior of the agents are set by a greedy policy that always collects the point with the highest expected reward in a round-robin fashion.
Each agent can collect a maximum of 3 points and start out at nodes 0 and 1, respectively.
The environment API follows the standard set out by the [PettingZoo library](https://pettingzoo.farama.org/).

```python
--8<-- "tutorials/graph_collector/graph_collector_example.py"
```
