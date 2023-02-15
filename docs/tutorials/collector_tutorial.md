# Simulating Data Collection in a 2D Plane

In this tutorial, we will explore how to use Datacollect to simulate data collection tasks in a 2D plane environment.

## 2D Plane Environments

2D plane environments are environments where the underlying space in which the agents operate is represented as a 2D plane with (x, y) coordinates.

## Example of 2D Plane-collector Environment
In the following example, we will define a simple 2D plane consisting of 300 points sampled uniformly at random in the range [0, 10] for each dimension.
We also define two agents that start at coordinates (0, 0) and (1, 1), respectively.
The agents can collect a maximum of 120 and 180 points.
The behavior of the agents are specified by a dummy policy that simply cycles through the available actions (i.e. collecting a point) in a round-robin fashion.

```python
--8<-- "tutorials/collector/collector_example.py"
```

## Example of Wrapper Environment

Datacollect also provides a wrapper environment that can be used to randomly generate points based on a given sampler and a given number of points.
In the following example, we will again define a simple 2D plane consisting of 300 points, but this time sampled randomly from a standard Normal.
Agents can again collect a maximum of 120 and 180 points.
The behavior of the agents are, however, instead specified by a greedy policy that always collects the point with the highest expected reward.

```python
--8<-- "tutorials/collector/wrapper_example.py"
```
