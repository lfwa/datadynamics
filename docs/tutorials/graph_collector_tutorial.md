# Simulating Data Collection Dynamics with a Graph

In this tutorial, we will explore how to use datadynamics to simulate data collection tasks in a graph-based environment.


## Graph-based Environments

Graph-based environments are environments where the underlying structure or space in which the agents operate is represented as a graph.
Graph-based environments can be useful for encoding obstacles or constraints in the underlying data environment or process.
A graph-based environment is also a simpler alternative to a navigation mesh (navmesh) and similarly allows for creating agent policies for pathfinding through complicated spaces.

## Defining a Graph

In datadynamics, graphs should be encoded as a weighted (directed or undirected) graph using the [NetworkX](https://networkx.org/) library.
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
