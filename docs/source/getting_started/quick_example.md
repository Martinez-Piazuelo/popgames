## Quick Usage Example

PopGames follows an object-oriented design built around four core components:

* **Population game**: defines the strategic environment
* **Payoff mechanism**: specifies the payoff perceived by agents
* **Revision process**: determines how agents update their strategies
* **Simulator**: orchestrates and runs the simulation

This example demonstrates how to simulate a simple **population game** using `popgames`.

We model the classic **Prisoner’s Dilemma**, where agents repeatedly choose between two strategies:

* **Cooperate**
* **Defect**

Although mutual cooperation is socially optimal, evolutionary dynamics often drive the population toward defection.

---

### Define the fitness function

We begin by defining the fitness function for the Prisoner's Dilemma. The payoff parameters satisfy

$$
T > R > P > S,
$$

<!-- README_REPLACE_START -->
```math
T > R > P > S
```
<!-- README_REPLACE_END -->

where:

* **T** – temptation to defect
* **R** – reward for mutual cooperation
* **P** – punishment for mutual defection
* **S** – sucker's payoff

The fitness function is then of the form:

$$
\mathbf{f}(\mathbf{x}) = \begin{bmatrix}R & S\\ T & P\end{bmatrix}\mathbf{x},
$$

<!-- README_REPLACE_START -->
```math
f(x) = [[R, S], [T, P]]x
```
<!-- README_REPLACE_END -->

which can be implemented in Python as follows:

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

Next we instantiate the population game and payoff mechanism objects by specifying the number of strategies
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

Agents revise their strategies according to a Poisson revision process combined with a softmax revision protocol.
We therefore define the corresponding revision process object as follows.

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

We now create the simulator and run the population dynamics.
The simulation considers a population of 1000 agents starting from an equal split between the two strategies.

```python
from popgames import Simulator

sim = Simulator(
    population_game=population_game,
    payoff_mechanism=payoff_mechanism,
    revision_processes=revision_process,
    num_agents=1000
)

x0 = np.array([0.5, 0.5]).reshape(2, 1)  # Initial state

sim.reset(x0=x0)
out = sim.run(T_sim=30)
```

---

### Visualize the results

Finally, we visualize the evolution of the population state using `matplotlib`.

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

PopGames also provides built-in visualization utilities supporting ternary plots, univariate projections, and custom 
key performance indicators. For more details, see the API reference documentation.
