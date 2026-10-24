# Visualization

``popgames`` has two plotting entry points:

* {meth}`Simulator.plot <popgames.Simulator.plot>` plots the last simulation (``sim.log``).
* {meth}`EnsembleResult.plot <popgames.EnsembleResult.plot>` plots an ensemble of simulations, from
  {meth}`~popgames.Simulator.run_ensemble` (finite agents) or {meth}`~popgames.Simulator.integrate_ensemble`
  (the deterministic approximation).

Both select the plot with ``plot_type``:

| Object | ``plot_type`` | What it shows |
|---|---|---|
| ``Simulator`` | ``"kpi"`` (default) | A key performance indicator over time, by default the distance to the (G)NE |
| | ``"univariate"`` | $\mathbf{x}(t)$ and $\mathbf{p}(t)$ per population, and $\mathbf{q}(t)$ |
| | ``"univariate_split"`` | One plot per entry of $\mathbf{x}$, $\mathbf{p}$, and $\mathbf{q}$ |
| | ``"ternary"`` | The trajectory on the simplex (populations with 3 strategies), optionally over a potential |
| ``EnsembleResult`` | ``"bands"`` (default) | Median and quantile bands (or mean and standard deviation bands) of the runs over time |
| | ``"kpi"`` | The same bands for a KPI |
| | ``"ternary"`` | The runs on the simplex (a phase portrait with several initial states) |
| | ``"final"`` | The distribution of the states at the final (or a given) time |

All plot types share these options:

| Option | Effect |
|---|---|
| ``plot_deterministic_approximation`` | Overlay the deterministic approximation (EDM-PDM) as dotted lines |
| ``filename`` | Save the figure(s); see [Saving figures](#saving-figures) for the file names |
| ``figsize``, ``fontsize`` | Size of each figure and of its text |
| ``show`` | Show the figure(s) (default), or close them after saving |
| ``ax`` | Draw on existing ``matplotlib`` axes, to compose figures; see [Composing figures](#composing-figures) |
| ``legend`` | Draw the legend (default), or leave it out, e.g., for one legend shared by several plots |

The examples below use the Prisoner's Dilemma (PD) and Rock-Paper-Scissors (RPS) games of the
[usage examples](../usage_examples/index.md). They save each figure (``filename=..., show=False``, or ``save(fig, ...)``
for composed figures); in a notebook, simply leave these arguments out. All the code is in
[``docs/examples/visualization_gallery.py``](https://github.com/Martinez-Piazuelo/popgames/blob/main/docs/examples/visualization_gallery.py),
which produces every figure of this page.

```{literalinclude} ../../examples/visualization_gallery.py
:language: python
:start-after: "# [setup]"
:end-before: "# [/setup]"
```

```{literalinclude} ../../examples/visualization_gallery.py
:language: python
:start-after: "# [games]"
:end-before: "# [/games]"
```

## Single simulations

### Trajectories over time

``"univariate"`` plots $\mathbf{x}(t)$ and $\mathbf{p}(t)$ (one figure per population), and $\mathbf{q}(t)$ if the
payoff mechanism has a state. In the PD, defection is dominant, so the population moves to "everyone defects" (with
the logit protocol, a few agents keep cooperating).

```{literalinclude} ../../examples/visualization_gallery.py
:language: python
:start-after: "# [pd_univariate]"
:end-before: "# [/pd_univariate]"
```

::::{grid} 2
:::{grid-item}
```{image} ../_static/visualization/pd_univariate_x_1.png
:alt: Strategic distribution of the Prisoner's Dilemma over time
```
:::
:::{grid-item}
```{image} ../_static/visualization/pd_univariate_p_1.png
:alt: Payoffs of the Prisoner's Dilemma over time
```
:::
::::

The options ``xlim``, ``ylim``, ``xscale``, and ``yscale`` take one value per variable, e.g.,
``ylim={"x": (0, 1)}``.

### Key performance indicators

``"kpi"`` plots a scalar function of the simulation over time. By default, it is the Euclidean distance to the
(G)NE, normalized by its initial value; ``kpi_function`` maps the log (fields ``t``, ``x``, ``q``, ``p``) to any other
KPI, of shape ``(K,)``.

```{literalinclude} ../../examples/visualization_gallery.py
:language: python
:start-after: "# [pd_kpi]"
:end-before: "# [/pd_kpi]"
```

::::{grid} 2
:::{grid-item}
```{image} ../_static/visualization/pd_kpi.png
:alt: Distance to the Nash equilibrium of the Prisoner's Dilemma over time
```
:::
:::{grid-item}
```{image} ../_static/visualization/pd_kpi_average_payoff.png
:alt: Average payoff of the Prisoner's Dilemma population over time
```
:::
::::

### Ternary plots

For populations with 3 strategies, ``"ternary"`` plots the trajectory on the simplex, with the (G)NE (if one is
found) and, for constrained games, the feasible region. Under the Smith protocol, the RPS population spirals to the
NE.

```{literalinclude} ../../examples/visualization_gallery.py
:language: python
:start-after: "# [rps_ternary]"
:end-before: "# [/rps_ternary]"
```

```{image} ../_static/visualization/rps_ternary.png
:width: 50%
:align: center
:alt: Rock-Paper-Scissors trajectory on the simplex under the Smith protocol
```

For potential games, ``potential_function`` (mapping $\mathbf{x}$ of shape ``(n, 1)`` to a ``(1, 1)`` array) is
drawn as a heatmap (``scale`` sets its resolution). The RPS game has no potential, so this example uses a
coordination game, $f_i(\mathbf{x}) = a_i x_i$, with potential $\sum_i a_i x_i^2 / 2$: the population climbs the
potential to the equilibrium $\mathbf{e}_3$.

```{literalinclude} ../../examples/visualization_gallery.py
:language: python
:start-after: "# [potential_heatmap]"
:end-before: "# [/potential_heatmap]"
```

```{image} ../_static/visualization/coordination_potential.png
:width: 50%
:align: center
:alt: Trajectory of a coordination game over its potential
```

### One plot per entry

``"univariate_split"`` gives each entry of $\mathbf{x}$, $\mathbf{p}$, and $\mathbf{q}$ its own plot, here drawn on
a grid of axes with one shared legend (see [Composing figures](#composing-figures)).

```{literalinclude} ../../examples/visualization_gallery.py
:language: python
:start-after: "# [rps_split]"
:end-before: "# [/rps_split]"
```

```{image} ../_static/visualization/rps_split.png
:alt: Each entry of the RPS strategic distribution and payoffs over time
```

### Reference lines

``plot_gne=True`` draws the (G)NE as dashed lines on the plots of $\mathbf{x}$, and ``reference`` draws any values
per variable (one per entry, or a scalar for all entries). Both work with ``"univariate"``, ``"univariate_split"``,
and the ensemble ``"bands"``.

```{literalinclude} ../../examples/visualization_gallery.py
:language: python
:start-after: "# [reference_lines]"
:end-before: "# [/reference_lines]"
```

```{image} ../_static/visualization/rps_reference_lines.png
:alt: RPS strategic distribution and payoffs with the Nash equilibrium as reference lines
```

(composing-figures)=
## Composing figures

By default, each plot creates its own figure(s). With ``ax``, it draws on existing axes instead, so several plots
can share one figure:

* Pass one axes per panel the plot would otherwise create, in the order given in its documentation (e.g., ``x``
  then ``p`` for ``"univariate"``). A single axes is enough for a single panel, and arrays of axes from
  ``plt.subplots`` are taken row by row.
* The figure then belongs to you: ``popgames`` neither saves, shows, nor closes it, and leaves its layout alone.
  Combining ``ax`` with ``filename`` raises an error; save the figure instead.

With ``legend=False``, a plot leaves its legend out, but keeps the entries: {func}`~popgames.plotting.add_shared_legend`
collects them from all axes of the figure (each label once) into one legend.

```{literalinclude} ../../examples/visualization_gallery.py
:language: python
:start-after: "# [compose]"
:end-before: "# [/compose]"
```

```{image} ../_static/visualization/rps_compose.png
:width: 80%
:align: center
:alt: RPS trajectories under the Replicator and Smith protocols, side by side, with one legend
```

## Ensembles

Ensembles show the variability of the finite-agent simulations
(see [the RPS ensemble example](../usage_examples/rock_paper_scissors_ensemble_example.md) for a discussion).

```{literalinclude} ../../examples/visualization_gallery.py
:language: python
:start-after: "# [rps_ensemble]"
:end-before: "# [/rps_ensemble]"
```

### Bands over time

``"bands"`` shows, for each strategy, the median of the runs at each time (solid line) and the bands between the
quantiles in ``bands`` (by default, 90% and 50% of the runs). ``variables`` selects among ``x``, ``p``, and ``q``.

```{literalinclude} ../../examples/visualization_gallery.py
:language: python
:start-after: "# [bands]"
:end-before: "# [/bands]"
```

```{image} ../_static/visualization/rps_bands_x_1.png
:width: 60%
:align: center
:alt: Median and quantile bands of the RPS ensemble over time
```

Its options control the statistics, the layout, the references, and the style:

| Option | Effect |
|---|---|
| ``statistic="mean_std"``, ``num_std`` | Mean and bands of $\pm k$ standard deviations ($k$ in ``num_std``), instead of the median and quantiles |
| ``layout="per_strategy"``, ``ncols`` | One panel per strategy (in a grid of ``ncols`` columns, or on the given axes) |
| ``plot_gne``, ``reference``, ``reference_style`` | Reference lines (dashed, in the color of each strategy, unless ``reference_style`` overrides it) |
| ``linewidth``, ``band_alpha`` | Width of the central lines; opacity of the bands (one value, or one per band, widest first) |
| ``plot_median`` | Draw the central line (median or mean) |
| ``pool_initial_states``, ``initial_state`` | Which runs the bands mix; see [Several initial states](#several-initial-states) |
| ``derived`` | Quantities derived from the state; see [Derived quantities](#derived-quantities) |

```{literalinclude} ../../examples/visualization_gallery.py
:language: python
:start-after: "# [bands_styled]"
:end-before: "# [/bands_styled]"
```

```{image} ../_static/visualization/rps_bands_styled.png
:alt: Mean and standard deviation bands of the RPS ensemble, one panel per strategy, with the Nash equilibrium
```

### KPI

``"kpi"`` shows the same bands for a KPI evaluated on each run (by default, the normalized distance to the (G)NE;
``kpi_function`` maps a run to any other KPI). It accepts the same statistics and style options as ``"bands"``.

```{literalinclude} ../../examples/visualization_gallery.py
:language: python
:start-after: "# [ensemble_kpi]"
:end-before: "# [/ensemble_kpi]"
```

```{image} ../_static/visualization/rps_ensemble_kpi.png
:width: 60%
:align: center
:alt: Distance to the Nash equilibrium of the RPS ensemble over time
```

### Runs on the simplex and final states

``"ternary"`` draws the runs as semi-transparent lines (``max_runs``, ``alpha``), optionally with their mean
(``plot_mean``). ``"final"`` shows where the runs are at the final time, or at time ``t``: a scatter plot on the
simplex for populations with 3 strategies, and histograms (``bins``) otherwise.

```{literalinclude} ../../examples/visualization_gallery.py
:language: python
:start-after: "# [ensemble_ternary_final]"
:end-before: "# [/ensemble_ternary_final]"
```

```{image} ../_static/visualization/rps_ensemble_ternary_final.png
:width: 80%
:align: center
:alt: RPS runs on the simplex and their final states
```

For the PD (2 strategies), ``"final"`` shows the histogram of $x_1$ (the share of cooperators), with the
deterministic approximation (dotted) and the NE (dashed).

```{literalinclude} ../../examples/visualization_gallery.py
:language: python
:start-after: "# [pd_final]"
:end-before: "# [/pd_final]"
```

```{image} ../_static/visualization/pd_final.png
:width: 60%
:align: center
:alt: Histogram of the share of cooperators of the PD runs at time 2
```

(several-initial-states)=
### Several initial states

Runs with the same initial state form a group. With several groups, ``"ternary"`` is a phase portrait (one color
per initial state, which is marked), ``"kpi"`` has one band per group, and ``"bands"`` makes one figure per group (or
only for ``initial_state=g``), unless ``pool_initial_states=True`` mixes all runs in the same bands.

```{literalinclude} ../../examples/visualization_gallery.py
:language: python
:start-after: "# [phase_portrait]"
:end-before: "# [/phase_portrait]"
```

```{image} ../_static/visualization/rps_phase_portrait.png
:alt: Phase portrait of RPS runs from 6 initial states, and their pooled mean and standard deviation
```

## Deterministic ensembles

{meth}`~popgames.Simulator.integrate_ensemble` integrates the deterministic approximation from several initial
states, e.g., to check that every trajectory converges, wherever it starts. The result is an
{class}`~popgames.EnsembleResult` with one run per initial state, so all plots above apply; ``"bands"`` and
``"kpi"`` pool the initial states by default.

```{literalinclude} ../../examples/visualization_gallery.py
:language: python
:start-after: "# [deterministic_ensemble]"
:end-before: "# [/deterministic_ensemble]"
```

```{image} ../_static/visualization/rps_deterministic_ensemble.png
:alt: Mean and standard deviation of the RPS deterministic approximation from 50 initial states
```

(derived-quantities)=
## Derived quantities

``derived`` plots quantities computed from the state, by name: functions mapping a run (fields ``t``, ``x``, ``q``,
``p``) to an array of shape ``(m, K)`` (or ``(K,)``). They are plotted after the variables, with all the options of
``"bands"`` (``variables=[]`` plots only them), and their statistics are computed on their values in the runs.
{meth}`~popgames.EnsembleResult.apply` returns them as an array of shape ``(runs, m, K)``.

Here, the product $x_1 x_2 x_3$ is conserved by the replicator dynamics (dotted line), but the finite population
drifts away from it.

```{literalinclude} ../../examples/visualization_gallery.py
:language: python
:start-after: "# [derived]"
:end-before: "# [/derived]"
```

```{image} ../_static/visualization/rps_derived_product.png
:width: 60%
:align: center
:alt: The product of the RPS strategic distribution, conserved by the replicator dynamics but not by the runs
```

(saving-figures)=
## Saving figures

With ``filename``, each figure is saved in the format of its extension (e.g., ``.pdf`` or ``.png``). Plots that
create several figures append a suffix to the file name:

| Plot | File names (for ``filename="fig.pdf"``) |
|---|---|
| ``"kpi"``, single-population ``"ternary"`` | ``fig.pdf`` |
| ``"univariate"`` | ``fig_x_1.pdf``, ``fig_p_1.pdf`` (per population), ``fig_q.pdf`` |
| ``"univariate_split"`` | ``fig_x_1_1.pdf`` (strategy, population), ..., ``fig_q1.pdf``, ... |
| multi-population ``"ternary"`` | ``fig_pop_1.pdf``, ... |
| ensemble ``"bands"`` | ``fig_x_1.pdf``, ``fig_p_1.pdf``, ``fig_q.pdf``, ``fig_<name>.pdf`` (derived), with ``_x0_<g>`` per initial state unless pooled |
| ensemble ``"ternary"``, ``"final"`` | ``fig.pdf``, or ``fig_pop_<k>.pdf`` with several populations (and ``_x_<i>`` for histograms per strategy) |

For figures composed with ``ax``, save the figure with ``fig.savefig(...)``.
