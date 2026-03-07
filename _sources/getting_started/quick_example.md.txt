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

$
T > R > P > S,
$

where:

* **T** – temptation to defect
* **R** – reward for mutual cooperation
* **P** – punishment for mutual defection
* **S** – sucker's payoff

The fitness function is then of the form:

$
\mathbf{f}(\mathbf{x}) = \begin{bmatrix}R & S\\ T & P\end{bmatrix}\mathbf{x},
$

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
