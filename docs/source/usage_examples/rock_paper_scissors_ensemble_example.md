# Ensembles of simulations

Finite-agent simulations are stochastic: two runs from the same initial state follow different trajectories. A
single run shows one possible outcome, while an **ensemble** of runs shows the distribution of outcomes, from the
same initial state or from different ones.
This example revisits the [Rock-Paper-Scissors](rock_paper_scissors_example.md) game and compares the Replicator and
Smith protocols over 50 runs each, with a small population of 200 agents.

```{literalinclude} ../../examples/rock_paper_scissors_ensemble.py
:language: python
:linenos:
```

{meth}`~popgames.Simulator.run_ensemble` resets the simulator to the same initial state before each run (``x0``,
set by the last ``reset``), and samples every run at common times (``t_eval``, 201 evenly spaced times by default).
Between revision events the state of a finite population is constant, so the value of a run at time $t$ is its state
after the last revision event at or before $t$. The result is an {class}`~popgames.EnsembleResult` with arrays of
shape ``(M, n, K)``: one leading axis for the $M$ runs, followed by the layout of ``sim.log``.

## Trajectories on the simplex

``ens.plot(plot_type="ternary")`` draws every run as a semi-transparent line (Replicator on the left, Smith on the
right). The dotted line is the deterministic approximation (EDM).

```{image} ../_static/rps_ensemble_ternary.png
:width: 90%
:align: center
:alt: Ensembles of Rock-Paper-Scissors trajectories under the Replicator and Smith protocols
```

Under the replicator dynamics, every trajectory is a closed orbit, so the EDM has no "restoring force": the random
fluctuations of the finite population accumulate, and the runs drift across orbits. Under the Smith dynamics, the NE
attracts nearby states, and the runs stay close to it.

## Quantile bands

``ens.plot(plot_type="bands")`` shows the median of each strategy over the runs (solid lines), with the bands
containing 90% and 50% of the runs at each time (shaded regions), and the EDM (dotted lines).

```{image} ../_static/rps_ensemble_bands.png
:width: 100%
:align: center
:alt: Quantile bands of the strategic distribution under the Replicator and Smith protocols
```

The bands of the Replicator widen over time: the runs drift across orbits and out of phase with each other. The
bands of the Smith protocol remain narrow.

## Final states

``ens.plot(plot_type="final")`` shows where the runs are at the final time (or at any time ``t``).

```{image} ../_static/rps_ensemble_final.png
:width: 90%
:align: center
:alt: Final states of the runs under the Replicator and Smith protocols
```

The same plot types are available for multi-population games and payoff dynamics models: ``bands`` also plots the
payoffs ``p`` and the PDM state ``q``, and ``kpi`` plots the bands of any KPI (by default, the distance to the GNE).

## Different initial states

The seeds and the initial states of ``run_ensemble`` can be shared by all runs (as above) or given per run:

* ``seed``: an integer (or ``None``) derives independent seeds for the runs, while a sequence gives the seed of each
  run, e.g., ``seed=[7] * M`` to use the same seed in all runs.
* ``x0``: an array of shape ``(n, 1)`` is shared by all runs, while an array of shape ``(M, n, 1)`` gives the initial
  state of each run (and similarly ``q0`` for payoff dynamics models).

The number of runs is the length of the per-run arguments. {func}`~popgames.utilities.sample_initial_states` samples
initial states uniformly at random on the simplex, with its own seed. The following example samples 6 initial states
and simulates 5 runs from each of them:

```{literalinclude} ../../examples/rock_paper_scissors_phase_portrait.py
:language: python
:linenos:
```

Runs with the same initial state form a **group**. The ternary plot becomes a phase portrait, with one color per
group (Replicator on the left, Smith on the right):

```{image} ../_static/rps_ensemble_phase_portrait.png
:width: 90%
:align: center
:alt: Phase portraits of the Replicator and Smith protocols in Rock-Paper-Scissors
```

The other plots also show one group per initial state: ``bands`` plots the bands of each group in separate figures
(or only those of ``initial_state=g``), and ``kpi`` and ``final`` use one color per group.

Some common designs:

```python
# Same initial state, independent seeds
ens = sim.run_ensemble(T_sim, num_runs=50, x0=x0, seed=0)

# Different initial states, same seed (common random numbers): the runs differ only through their initial state
ens = sim.run_ensemble(T_sim, x0=x0s, seed=[7] * len(x0s))

# Every initial state with R independent seeds (runs grouped by initial state)
ens = sim.run_ensemble(T_sim, x0=np.repeat(x0s, R, axis=0), seed=0)

# Every initial state with the same R seeds
ens = sim.run_ensemble(T_sim, x0=np.repeat(x0s, R, axis=0), seed=np.tile(np.arange(R), len(x0s)))
```

## Working with the results

The arrays can also be used directly:

```python
ens.x[r]           # run r, shape (n, K), same layout as sim.log.x
ens.x[:, i]        # strategy i in all runs, shape (M, K)
ens.mean().x       # mean over the runs, shape (n, K)
ens.quantile([0.05, 0.95]).x   # shape (2, n, K)
ens.final().x      # states at the final time, shape (M, n)
ens.deterministic().x          # EDM-PDM from the initial state of each run, shape (M, n, K)
ens.groups[g]                  # indices of the runs of group g (same initial state)
ens.subset(ens.groups[g]).mean().x   # mean over the runs of group g
```

``mean`` and ``quantile`` pool all runs: with several initial states, use ``subset`` to compute them per group.

The ensemble is reproducible: with the same ``seed``, ``run_ensemble`` returns the same runs. Each run has its own
seed (``ens.seeds``) and initial state (``ens.x0`` and ``ens.q0``), so a single run can be reproduced (e.g., to
inspect it in detail with ``sim.log``):

```python
sim.reset(ens.x0[r], ens.q0[r], seed=int(ens.seeds[r]))
sim.run(T_sim=60)
```

``ens.x0`` holds the initial states the runs actually started from: ``reset`` rounds the requested state to a whole
number of agents per strategy.

## Running with the Numba backend

Ensembles multiply the number of revision events by the number of runs: this example simulates about $6 \cdot 10^5$
revision events per protocol, which takes about 15 seconds each with the numpy backend. With the
[Numba backend](../performance/index.md), the whole example runs in about two seconds:

```python
sim = Simulator(
    population_game=population_game,
    payoff_mechanism=payoff_mechanism,
    revision_processes=PoissonRevisionProcess(
        Poisson_clock_rate=1,
        revision_protocol=revision_protocol,
    ),
    num_agents=200,
    backend="numba",
)
```

The fitness function is compiled once per simulator and reused by all the runs of the ensemble.
