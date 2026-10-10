# Equilibrium selection in a coordination game

This example uses ensembles of finite-agent simulations
({meth}`~popgames.Simulator.run_ensemble`) to study **equilibrium selection** in a game with several equilibria. It
covers the two typical uses of ensembles:

1. **Same initial state, different seeds**: how much do the outcomes vary because of the randomness of the revisions?
2. **Different initial states, same seed**: where does the population go from each initial state?

## The game

In a coordination game, agents earn more when they play the same strategy as others. We consider three strategies,
with fitness

$$
f_i(\mathbf{x}) = a_i x_i, \qquad \mathbf{a} = (1, 2, 3),
$$

so strategy 3 has the highest payoff when everyone coordinates on it. Each vertex $\mathbf{e}_i$ (everyone plays
strategy $i$) is a strict Nash equilibrium. There is also an unstable mixed equilibrium, where all strategies have the
same payoff ($a_i x_i = 6/11$ for all $i$):

$$
\mathbf{x}^* = \tfrac{1}{11}(6, 3, 2).
$$

Agents revise with the logit protocol ({class}`~popgames.revision_protocol.Softmax`) with a small noise level
$\eta = 0.05$: they almost always pick a strategy with the highest payoff, but every strategy has a positive
probability.

```{literalinclude} ../../examples/coordination_ensemble.py
:language: python
:linenos:
```

## Same initial state, different seeds

The first ensemble starts 100 runs at the mixed equilibrium $\mathbf{x}^*$ (with 110 agents, so that $\mathbf{x}^*$ is
an exact number of agents per strategy), each with its own seed (derived from ``seed=0``).

```{image} ../_static/coordination_ensemble_seeds.png
:width: 90%
:align: center
:alt: Runs from the mixed equilibrium of the coordination game, and their states at t=2
```

At $\mathbf{x}^*$ all strategies earn the same payoff, so logit revisions pick each strategy with the same probability.
The deterministic approximation (EDM, dotted) moves to $\mathbf{e}_3$. The finite population, however, is pushed
around by the randomness of the first revisions, and once a strategy gets ahead, coordination amplifies the lead. The
runs split between the equilibria (left: trajectories of the first 50 runs; right: states of all runs at $t=2$):

```text
Fraction of runs ending at e1, e2, e3: [0.04 0.31 0.65]
```

A third of the runs end at an equilibrium that the deterministic approximation does not predict. A single run could
have suggested either outcome: the ensemble shows the probability of each one.

```{note}
With a protocol whose switching rates vanish when all payoffs are equal (e.g.,
{class}`~popgames.revision_protocol.Smith` or {class}`~popgames.revision_protocol.BNN`), the mixed equilibrium would
be absorbing: no agent ever revises from $\mathbf{x}^*$, and all runs stay there.
```

## Different initial states, same seed

The second ensemble starts one run from each of 10 initial states sampled uniformly at random on the simplex
({func}`~popgames.utilities.sample_initial_states`). All runs use the same seed (``seed=[0] * 10``), so they draw the
same random numbers (common random numbers): the runs differ only through their initial states.

```{image} ../_static/coordination_ensemble_initial_states.png
:width: 50%
:align: center
:alt: Runs from 10 initial states of the coordination game
```

Each color is an initial state, with its deterministic trajectory dotted. The plot shows the basins of attraction of
the equilibria: most initial states lead to $\mathbf{e}_3$, the equilibrium with the highest payoff, and only initial
states with many agents already playing strategy 1 or 2 lead to $\mathbf{e}_1$ or $\mathbf{e}_2$. Away from the
boundaries between the basins, the finite population follows the deterministic approximation closely.

## Combining both

Both designs can be combined, e.g., 20 runs (with independent seeds) from each initial state, to estimate the
probability of reaching each equilibrium from each initial state:

```python
ens = sim.run_ensemble(T_sim=10, x0=np.repeat(x0s, 20, axis=0), seed=0)

for g, runs in enumerate(ens.groups):
    selected = np.bincount(ens.final().x[runs].argmax(axis=1), minlength=3) / len(runs)
    print(f'Initial state {g + 1}: {selected}')
```

See [Ensembles of simulations](rock_paper_scissors_ensemble_example.md) for the other plot types and the details of
``run_ensemble`` and {class}`~popgames.EnsembleResult`. The example runs in a few seconds with the numpy backend; for
larger populations or ensembles, consider the [Numba backend](../performance/index.md) (``backend="numba"``).
