# Simulating Data Collection Dynamics in a Plane

In this tutorial, we will explore how to use Datadynamics to simulate data collection tasks in a plane-based environment.


## Plane-Based Environments

The plane-based environment provides a simple space for quick iterations and testing of new strategies.
The environment is based on a two-dimensional Euclidean space (using (x, y) coordinates), where agents can navigate the environment with a cost proportional to the Euclidean distance travelled.
Although this environment does not allow for complex terrain and obstacle representations, it is optimized for performance.

## Example of Plane-Based Collector Environment
In the following example, we will define a simple a Euclidean plane consisting of 300 points sampled uniformly at random in the range [0, 10] for each dimension.
We also define two agents that start at coordinates (0, 0) and (1, 1), respectively.
The agents can collect a maximum of 120 and 180 points.
The behavior of the agents are specified by a dummy policy that simply cycles through the available actions (i.e. collecting a point) in a round-robin fashion.

```python
--8<-- "tutorials/collector/collector_example.py"
```

## Example of Wrapper Environment

Datadynamics also provides a wrapper environment that can be used to randomly generate points based on a given sampler and a given number of points.
In the following example, we will again define a simple Euclidean plane consisting of 300 points, but this time sampled randomly from a standard Normal.
Agents can again collect a maximum of 120 and 180 points.
The behavior of the agents are, however, instead specified by a greedy policy that always collects the point with the highest expected reward.

```python
--8<-- "tutorials/collector/wrapper_example.py"
```
