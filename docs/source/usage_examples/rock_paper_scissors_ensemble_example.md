# Ensembles of simulations

Finite-agent simulations are stochastic: two runs from the same initial state follow different trajectories. A
single run shows one possible outcome, while an **ensemble** of independent runs shows the distribution of outcomes.
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

## Working with the results

The arrays can also be used directly:

```python
ens.x[r]           # run r, shape (n, K), same layout as sim.log.x
ens.x[:, i]        # strategy i in all runs, shape (M, K)
ens.mean().x       # mean over the runs, shape (n, K)
ens.quantile([0.05, 0.95]).x   # shape (2, n, K)
ens.final().x      # states at the final time, shape (M, n)
ens.deterministic()            # EDM-PDM from x0, sampled at ens.t
```

The ensemble is reproducible: with the same ``seed``, ``run_ensemble`` returns the same runs. Each run has its own
seed (``ens.seeds``), so a single run can be reproduced (e.g., to inspect it in detail with ``sim.log``):

```python
sim.reset(ens.x0, ens.q0, seed=int(ens.seeds[r]))
sim.run(T_sim=60)
```

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
