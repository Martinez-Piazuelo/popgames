<p align="center">
  <img src="assets/logo.png" alt="PopGames logo" width="300">
</p>

A Python package to model and simulate population games.

[![PyPI version](https://img.shields.io/pypi/v/popgames.svg)](https://pypi.org/project/popgames/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/popgames.svg)](https://pypi.org/project/popgames/)
[![docs](https://img.shields.io/badge/docs-online-success)](https://martinez-piazuelo.github.io/popgames/)

---

## Documentation

Full API reference and usage examples are available at:

[https://martinez-piazuelo.github.io/popgames/](https://martinez-piazuelo.github.io/popgames/)

---

## Installation

PopGames is available on PyPI and can be installed using `pip`.

### Install from PyPI

```bash
pip install popgames
```

This installs the latest released version together with its dependencies.

### Install the development version

If you want the latest features, you can install the development version directly from GitHub:

```bash
pip install git+https://github.com/Martinez-Piazuelo/popgames.git
```

### Verify the installation

You can verify that PopGames is installed correctly by running:

```python
import popgames
print(popgames.__version__)
```

If no errors occur, the installation was successful.

## Quick Usage Example

PopGames follows an object-oriented programming approach based on four core objects:

* **Population game:** defines the strategic environment of the game
* **Payoff mechanism:** specifies the payoff perceived by the agents
* **Revision process:** establishes how agents update their strategies
* **Simulator:** orchestrates the simulation

This example demonstrates how to simulate a simple **population game** using `popgames`.

We model the classic **Prisoner’s Dilemma**, where agents repeatedly choose between two strategies:

* **Cooperate**
* **Defect**

Even though mutual cooperation is socially optimal, evolutionary dynamics often drive the population toward defection.

---

### Define the fitness function

We begin by defining the fitness function for the Prisoner's Dilemma. The parameters satisfy

```math
T > R > P > S
```

where:

* **T** – temptation to defect
* **R** – reward for mutual cooperation
* **P** – punishment for mutual defection
* **S** – sucker's payoff

The fitness function is then of the form:

```math
f(x) = [[R, S], [T, P]]x
```

which is implemented as:

```python
import numpy as np

T, R, P, S = 3, 2, 1, 0

def fitness_function(x):
    return np.dot(
        np.array([[R, S], [T, P]]),
        x
    )
```

---

### Create the population game and payoff mechanism objects

Next we create the population game object and payoff mechanism objects by specifying the number of strategies
and the fitness function.

```python
from popgames import (
    SinglePopulationGame,
    PayoffMechanism,
)

population_game = SinglePopulationGame(
    num_strategies=2,
    fitness_function=fitness_function,
)

payoff_mechanism = PayoffMechanism(
    h_map=fitness_function,
    n=2,
)
```

---

### Define the revision process object

Agents revise their strategies according to a **Poisson revision process** combined with a **softmax revision protocol**.
Thus, we define the corresponding revision process object as follows.

```python
from popgames import PoissonRevisionProcess
from popgames.revision_protocol import Softmax

revision_process = PoissonRevisionProcess(
    Poisson_clock_rate=1,
    revision_protocol=Softmax(0.1),
)
```

---

### Instantiate the simulator and run the simulation

We simulate a population of 1000 agents starting from an equal split between the two strategies.

```python
from popgames import Simulator

sim = Simulator(
    population_game=population_game,
    payoff_mechanism=payoff_mechanism,
    revision_processes=revision_process,
    num_agents=1000
)

x0 = np.array([0.5, 0.5]).reshape(2, 1) # Initial state

sim.reset(x0=x0)
out = sim.run(T_sim=30)
```

---

### Visualize the results

Finally, we can visualize the results using `matplotlib` as follows.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(3, 3))

plt.plot(out.t, out.x[0, :], label="Cooperators")
plt.plot(out.t, out.x[1, :], label="Defectors")

plt.xlabel("Time")
plt.ylabel("Population fraction")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
```

PopGames also provides ready-to-use visualization options supporting ternary plots, univariate projections, or custom
key performance indicator. For more details please refer to the API reference documentation.

## Contributing

Contributions are welcome. If you would like to improve `popgames`, please follow the workflow below.

### 1. Fork and clone the repository

Fork the repository on GitHub, then clone your fork locally:

```bash
git clone https://github.com/<your-username>/popgames.git
cd popgames
```

Create a new branch for your changes:

```bash
git checkout -b your-feature-branch-name
```

---

### 2. Set up the development environment

This project uses `uv` for dependency and environment management. 

Install `uv` if it is not already installed:

```bash
pip install uv
```

Install the development environment with:

```bash
uv sync
```

This installs the package together with the default dependency groups used for development and documentation.

The repository also uses a **Taskfile** to define common development commands. The Task runner (`go-task`) is 
installed automatically as part of the development dependencies.

---

### 3. Run the test suite

Before submitting a pull request, ensure all tests pass:

```bash
uv run task test
```

---

### 4. Format and check the code

Please ensure your code is properly formatted and passes all checks before opening a pull request:

```bash
uv run task format
uv run task check
```

---

### 5. Submit a pull request

Push your branch to your fork and open a pull request against the main repository.

Please include:

* A clear description of the changes
* Any relevant issue references
* Tests for new functionality when applicable
* Documentation updates if needed

---

### Development notes

* Keep contributions focused and minimal.
* Follow the existing project structure and coding style.
* Run formatting, linting, and tests locally before submitting your pull request.
