# Prisoner's Dilemma

This example models a population of agents repeatedly choosing between two strategies:
**cooperate** or **defect**, as in the classic Prisoner's Dilemma game.

It demonstrates how revision dynamics can drive the population toward defection,
even when mutual cooperation would be socially optimal.

```{literalinclude} ../../examples/prisoners_dilemma.py
:language: python
:linenos:
```

```{image} ../_static/prisoners_dilemma.png
:width: 50%
:align: center
:alt: Prisoner's Dilemma trajectory
```


## Running with the Numba backend

The same simulation can run with the optional [Numba backend](../performance/index.md) (requires
``pip install "popgames[numba]"``). Only the simulator changes:

```python
sim = Simulator(
    population_game=population_game,
    payoff_mechanism=payoff_mechanism,
    revision_processes=revision_process,
    num_agents=1000,
    backend="numba",
)
```

The fitness function is compiled automatically with ``numba.njit``. This is why the payoff parameters are declared
as floats (``T, R, P, S = 3.0, 2.0, 1.0, 0.0``): with integers, ``np.array([[R, S], [T, P]])`` would be an integer
matrix, and Numba does not support matrix products between integer and floating-point arrays. With the numpy
backend, both versions work. If the fitness function cannot be compiled, the simulator logs a warning and falls back
to the numpy backend.

With 1000 agents over 30 time units (about 30 000 revision events), this simulation is too small for the Numba
backend to pay off: compiling the fitness function takes longer than the whole simulation with the numpy backend.
The Numba backend is worth it for large populations or long horizons, e.g., ``num_agents=100_000``.
