# Datacollect
[![pypi](https://img.shields.io/pypi/v/datacollect?label=pypi)](<LINK TO PYPI>)
[![Test Status](https://github.com/lfwa/datacollect/actions/workflows/test.yml/badge.svg)](https://github.com/lfwa/datacollect/actions/workflows/test.yml)
[![Documentation Status](https://github.com/lfwa/datacollect/actions/workflows/documentation.yml/badge.svg)](https://lfwa.github.io/datacollect/)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![License](https://img.shields.io/github/license/lfwa/datacollect)](https://github.com/lfwa/datacollect/blob/main/LICENSE)

Datacollect is a Python library and environment for simulating multi-agent data collection. The library is built on top of [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) and is distributed under the [BSD 3-Clause License](LICENSE).

The documentation is available at [lfwa.github.io/datacollect](https://lfwa.github.io/datacollect/).

## Installation
\<PLACEHOLDER>

```bash
pip install
```
## Quick Start

<p align="center">
    <img src="https://raw.githubusercontent.com/lfwa/datacollect/main/datacollect.gif" width="350px"/>
</p>

```python
# See tutorials/example.py
from datacollect.environments import collector_v0
from datacollect.policies import greedy_policy_v0

env = collector_v0.env(
    n_points=300, n_agents=2, max_collect=[120, 180], render_mode="human"
)
policy = greedy_policy_v0.policy(env=env)
env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    action = policy.action(observation, agent)
    env.step(action)
```
