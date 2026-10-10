# Rock-Paper-Scissors

This example models a population of agents playing the classic **Rock-Paper-Scissors** game,
where each strategy beats one strategy and loses to the other.
The unique Nash equilibrium (NE) is the uniform distribution $(1/3, 1/3, 1/3)$.

It compares two revision protocols and shows how they can lead to very different long-run behavior,
even when the game has the same equilibrium:

* **Replicator** (pairwise proportional imitation): agents imitate better-performing opponents.
  The resulting evolutionary dynamics model (EDM) is the replicator dynamics, whose trajectories are
  **closed orbits** around the NE (the product $x_1 x_2 x_3$ is conserved).
  With finitely many agents, random fluctuations make the population drift across orbits.
* **Smith**: agents switch to strategies with higher payoff than their current one,
  regardless of how popular they are. The population **converges** to the NE.

```{literalinclude} ../../examples/rock_paper_scissors.py
:language: python
:linenos:
```

```{image} ../_static/rock_paper_scissors.png
:width: 90%
:align: center
:alt: Rock-Paper-Scissors trajectories under the Replicator and Smith protocols
```

## Running with the Numba backend

Both simulations can run with the optional [Numba backend](../performance/index.md) (requires
``pip install "popgames[numba]"``) by adding ``backend="numba"`` to the simulator:

```python
sim = Simulator(
    population_game=population_game,
    payoff_mechanism=payoff_mechanism,
    revision_processes=revision_process,
    num_agents=1000,
    backend="numba",
)
```

The payoff matrix is declared with ``dtype=float``. The fitness function is compiled automatically with
``numba.njit``, and Numba does not support matrix products between integer and floating-point arrays: with an
integer matrix such as ``np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])``, compilation fails, and the simulator logs a
warning and falls back to the numpy backend. Writing the entries as floats (e.g., ``0.0``) works as well.

If you prefer to compile the fitness function explicitly, decorate it with ``@numba.njit``:

```python
import numba

@numba.njit
def fitness_function(x):
    return np.dot(A, x)
```

Both simulators use the same fitness function, which is therefore compiled only once. Both revision protocols are
built-in, so they are supported by the Numba backend.

With 1000 agents over 100 time units (about 100 000 revision events per protocol), the compilation time is comparable
to the time saved, so both backends take about the same time. For larger populations, e.g., ``num_agents=100_000``,
the Numba backend is about 100 times faster than the numpy backend.
