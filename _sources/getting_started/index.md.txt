# Getting Started

**PopGames** is a Python library for modeling and simulating population games.

## Source Code

🔗 **GitHub repository:**  
[https://github.com/Martinez-Piazuelo/popgames](https://github.com/Martinez-Piazuelo/popgames)

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

This example demonstrates how to simulate a simple **population game** using `popgames`.

We model the classic **Prisoner’s Dilemma**, where agents repeatedly choose between two strategies:

* **Cooperate**
* **Defect**

Even though mutual cooperation is socially optimal, evolutionary dynamics often drive the population toward defection.

---

### Define the game

We begin by defining the payoff matrix of the Prisoner's Dilemma. The parameters satisfy

[
T > R > P > S
]

where:

* **T** – temptation to defect
* **R** – reward for mutual cooperation
* **P** – punishment for mutual defection
* **S** – sucker's payoff

```python
import numpy as np
import matplotlib.pyplot as plt

from popgames import (
    SinglePopulationGame,
    PayoffMechanism,
    PoissonRevisionProcess,
    Simulator,
)
from popgames.revision_protocol import Softmax

T, R, P, S = 3, 2, 1, 0

def fitness_function(x):
    return np.dot(
        np.array([[R, S], [T, P]]),
        x
    )
```

---

### Create the population game

Next we create the population game object and the associated payoff mechanism.

```python
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

### Define the revision process

Agents revise their strategies according to a **Poisson revision process** combined with a **softmax revision protocol**.

```python
revision_process = PoissonRevisionProcess(
    Poisson_clock_rate=1,
    revision_protocol=Softmax(0.1),
)
```

---

### Run the simulation

We simulate a population of 1000 agents starting from an equal split between the two strategies.

```python
sim = Simulator(
    population_game=population_game,
    payoff_mechanism=payoff_mechanism,
    revision_processes=revision_process,
    num_agents=1000
)

x0 = np.array([0.5, 0.5]).reshape(2, 1)

sim.reset(x0=x0)
out = sim.run(T_sim=30)
```

---

### Visualize the results

Finally, we plot the fraction of agents playing each strategy over time.

```python
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

If everything works as expected, you should see a plot like:

```{image} /_static/prisoners_dilemma.png
:width: 50%
:align: center
:alt: Prisoner's Dilemma trajectory
```

---

### What happens?

Starting from a balanced population, the evolutionary dynamics gradually favor **defection**, 
illustrating the well-known outcome of the Prisoner's Dilemma in evolutionary settings.

---

## Documentation

* **Usage Examples** – hands-on full usage examples
* **Key Concepts** – connections with theoretical background
* **API Reference** – full codebase documentation
* **Contributing to PopGames** - how to contribute to the PopGames package
