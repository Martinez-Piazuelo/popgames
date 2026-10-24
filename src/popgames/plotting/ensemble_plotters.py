from __future__ import annotations

import logging
import typing
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from popgames.plotting._figures import Panels, ternary_legend
from popgames.plotting._plot_config import (
    FIGSIZE,
    FIGSIZE_TERNARY,
    FONTSIZE,
)
from popgames.plotting.plotters import (
    REFERENCE_STYLE,
    make_default_kpi_function,
    make_reference_values,
)

if typing.TYPE_CHECKING:
    from typing import Callable, Literal, Sequence

    from matplotlib.axes import Axes

    from popgames.ensemble import EnsembleResult

__all__ = [
    "plot_ensemble_bands",
    "plot_ensemble_kpi",
    "plot_ensemble_ternary",
    "plot_ensemble_final",
]

logger = logging.getLogger(__name__)

DEFAULT_BANDS = ((0.05, 0.95), (0.25, 0.75))
DEFAULT_NUM_STD = (1,)
DETERMINISTIC_STYLE = {"linestyle": "dotted", "linewidth": 1.5}
MAX_GROUP_COLORS = 10  # Groups (initial states) beyond this number are plotted in black


def plot_ensemble_bands(
    ensemble: EnsembleResult,
    variables: Sequence[str] = ("x", "p", "q"),
    bands: Sequence[tuple[float, float]] = DEFAULT_BANDS,
    plot_median: bool = True,
    plot_deterministic_approximation: bool = False,
    initial_state: int = None,
    pool_initial_states: bool = None,
    statistic: Literal["quantile", "mean_std"] = "quantile",
    num_std: Sequence[float] = DEFAULT_NUM_STD,
    linewidth: float = 1.0,
    band_alpha: float | Sequence[float] = None,
    layout: Literal["joint", "per_strategy"] = "joint",
    ncols: int = 3,
    reference: dict[str, np.ndarray] = None,
    plot_gne: bool = False,
    reference_style: dict = None,
    derived: dict[str, Callable[[SimpleNamespace], np.ndarray]] = None,
    xlim: tuple[float, float] = None,
    ylim: dict[str, tuple[float, float]] = None,
    filename: str = None,
    figsize: tuple[float, float] = None,
    fontsize: int = FONTSIZE,
    show: bool = True,
    ax: Axes | Sequence[Axes] = None,
    legend: bool = True,
) -> None:
    """
    Plot the median and quantile bands (or the mean and standard deviation bands) of the runs over time.

    One figure is generated per population for ``x`` and ``p``, and one for ``q`` (if ``d > 0``). Each strategy has
    its own color: the solid line is the median over the runs, and the shaded regions are the bands between the given
    quantiles (e.g., 90% and 50% of the runs at each time with the default bands). With ``statistic='mean_std'``, the
    solid line is the mean instead, and the bands are the mean plus or minus ``num_std`` standard deviations.

    With ``layout='joint'``, all the strategies of a figure share one panel. With ``layout='per_strategy'``, each
    strategy has its own panel (titled with the strategy), laid out in a grid of ``ncols`` columns sharing the time
    axis, e.g., for games with many strategies. Reference values, such as an equilibrium, are plotted as dashed
    horizontal lines in the color of each strategy, and the deterministic approximation as dotted lines.

    Quantities derived from the state (``derived``), e.g., temperatures that depend on ``x``, are plotted like the
    variables, after them. Their statistics are computed on the derived values of the runs (which differs from
    applying the function to the statistics of ``x`` if the function is nonlinear).

    By default, the bands of a finite-agent ensemble only mix runs with the same initial state: if the ensemble has
    several initial states (groups), the figures are generated for each of them (or only for ``initial_state``). With
    ``pool_initial_states=True``, the bands mix all runs instead, e.g., to show how trajectories from many initial
    states converge. This is the default for deterministic ensembles (``Simulator.integrate_ensemble``), which have a
    single run per initial state.

    Args:
        ensemble (EnsembleResult): The ensemble to plot.
        variables (Sequence[str], optional): The variables to plot, among 'x', 'p', and 'q'. Defaults to all of them.
            Use ``[]`` to plot only the derived quantities.
        bands (Sequence[tuple[float, float]], optional): Pairs of lower and upper quantiles delimiting the bands
            (with ``statistic='quantile'``). Defaults to ((0.05, 0.95), (0.25, 0.75)).
        plot_median (bool, optional): Whether to plot the central line: the median, or the mean with
            ``statistic='mean_std'``. Defaults to True.
        plot_deterministic_approximation (bool, optional): Whether to plot the trajectories of the deterministic
            approximation (EDM-PDM) as dotted lines (when pooling several initial states, the same statistic as the
            central line over their trajectories). Ignored for deterministic ensembles. Defaults to False.
        initial_state (int, optional): The group (initial state) to plot, see ``EnsembleResult.groups``. Defaults to
            None, in which case all groups are plotted. Cannot be combined with ``pool_initial_states=True``.
        pool_initial_states (bool, optional): Whether the bands mix the runs of all initial states, in a single set
            of figures. Defaults to None, in which case it is True for deterministic ensembles and False otherwise.
        statistic (str, optional): 'quantile' for the median and quantile bands, or 'mean_std' for the mean and bands
            of plus or minus ``num_std`` (population) standard deviations. Note that the latter may leave the feasible
            region, e.g., become negative. Defaults to 'quantile'.
        num_std (Sequence[float], optional): The widths of the bands in standard deviations (with
            ``statistic='mean_std'``). Defaults to (1,).
        linewidth (float, optional): Line width of the central line. Defaults to 1.
        band_alpha (float | Sequence[float], optional): Opacity of the bands: one value for all of them, or one per
            band (widest first). Defaults to None, in which case 0.15 for the widest band, 0.25 for the next one, etc.
        layout (str, optional): 'joint' for one panel per figure, or 'per_strategy' for one panel per strategy.
            Defaults to 'joint'.
        ncols (int, optional): Number of columns of the grid of panels (with ``layout='per_strategy'``). Defaults
            to 3.
        reference (dict[str, np.ndarray], optional): Reference values plotted as dashed horizontal lines, per
            variable or derived quantity, e.g., ``{'x': x_star}`` with ``x_star`` of shape ``(n, 1)``. A scalar is
            used for all entries, e.g., ``{'T': 30.0}``. Defaults to None.
        plot_gne (bool, optional): Whether to plot the GNE (if one is found) as the reference of ``x``. Cannot be
            combined with ``reference['x']``. Defaults to False.
        reference_style (dict, optional): Keyword arguments of ``matplotlib.axes.Axes.axhline`` overriding the style
            of the reference lines (dashed, in the color of each entry), e.g.,
            ``{'color': 'black', 'linestyle': 'dotted'}``. Defaults to None.
        derived (dict[str, Callable[[SimpleNamespace], np.ndarray]], optional): Quantities derived from the state, by
            name (a short symbol, used in the labels, e.g., ``'T'``): functions mapping a run (see
            ``EnsembleResult.apply``) to an array of shape ``(m, K)``, one row per entry. Defaults to None.
        xlim (tuple[float, float], optional): Limits of the time axis. Defaults to None.
        ylim (dict[str, tuple[float, float]], optional): Limits of the vertical axis per variable or derived quantity,
            e.g., ``{'x': (0, 1)}``. Defaults to None.
        filename (str, optional): If provided, each figure is saved with the variable and population (and the group,
            if there are several and they are not pooled) appended to the file name, e.g., ``bands_x_1.pdf`` or
            ``bands_x_1_x0_2.pdf``. Defaults to None.
        figsize (tuple[float, float], optional): Figure size. Defaults to None, in which case (4, 2) is used per
            panel.
        fontsize (int, optional): Font size. Defaults to 8.
        show (bool, optional): Whether to show the figures. Defaults to True.
        ax (Axes | Sequence[Axes], optional): Existing ``matplotlib`` axes to draw on, one per panel that would
            otherwise be generated, in the order of the initial states (unless pooled) and, for each one, of ``x``
            (per population), ``p`` (per population), ``q``, and the derived quantities (and of the entries, with
            ``layout='per_strategy'``), e.g., the array of axes from ``plt.subplots(2, 3)``. The figures are then
            neither saved, shown, nor closed. Defaults to None.
        legend (bool, optional): Whether to draw the legend(s). If False, the entries remain available through
            ``ax.get_legend_handles_labels()``, e.g., for one legend shared by several plots (see
            ``popgames.plotting.add_shared_legend``). Defaults to True.
    """
    _check_variables(variables)
    if layout not in ("joint", "per_strategy"):
        raise ValueError(f"layout must be 'joint' or 'per_strategy', got {layout!r}.")
    stat = _statistic(statistic, bands, num_std)
    run_sets = _run_sets(ensemble, initial_state, pool_initial_states)
    det = _deterministic(ensemble, plot_deterministic_approximation)
    game = ensemble.simulator.population_game
    P = game.num_populations

    # Values of each variable and derived quantity, of shape (M, rows, K)
    data = {var: getattr(ensemble, var) for var in ("x", "p", "q")}
    data_det = {var: getattr(det, var) for var in data} if det is not None else None
    for name, function in (derived or {}).items():
        if not isinstance(name, str) or name in data:
            raise ValueError(
                f"Derived quantities need a name other than 'x', 'p', and 'q', got {name!r}."
            )
        data[name] = ensemble.apply(function)
        if det is not None:
            data_det[name] = ensemble.apply(function, deterministic=True)
    references = make_reference_values(
        ensemble.simulator,
        reference,
        plot_gne,
        derived_sizes={name: data[name].shape[1] for name in derived or {}},
    )

    panels = []  # (variable, population, rows, labels, file suffix)
    for var in ("x", "p"):
        if var not in variables:
            continue
        pos = 0
        for k in range(P):
            rows = list(range(pos, pos + game.num_strategies[k]))
            labels = [_strategy_label(var, i, k, P) for i in rows]
            panels.append((var, k, rows, labels, f"{var}_{k + 1}"))
            pos += game.num_strategies[k]
    d = ensemble.simulator.payoff_mechanism.d
    if "q" in variables and d > 0:
        rows = list(range(d))
        panels.append(("q", None, rows, [rf"$q_{{{i + 1}}}$" for i in rows], "q"))
    for name in derived or {}:
        rows = list(range(data[name].shape[1]))
        labels = (
            [rf"${name}$"]
            if len(rows) == 1
            else [rf"${name}_{{{i + 1}}}$" for i in rows]
        )
        panels.append((name, None, rows, labels, name))

    per_strategy = layout == "per_strategy"
    num_panels = sum(len(rows) if per_strategy else 1 for _, _, rows, _, _ in panels)
    figures = Panels(ax, len(run_sets) * num_panels, filename)
    for g, runs in run_sets:
        for var, k, rows, labels, suffix in panels:
            values = data[var][runs]
            if per_strategy:
                fig, axes, edges = figures.new_grid(len(rows), ncols, FIGSIZE, figsize)
            else:
                fig, ax = figures.new(figsize or FIGSIZE)
                axes, edges = [ax] * len(rows), [(True, True)] * len(rows)

            for c, (row, label) in enumerate(zip(rows, labels)):
                color = f"C{c % 10}"
                _plot_bands(
                    axes[c],
                    ensemble.t,
                    values[:, row],
                    stat,
                    plot_median,
                    color,
                    label,
                    linewidth,
                    band_alpha,
                )
                if det is not None:
                    center, _ = stat(data_det[var][runs, row])
                    axes[c].plot(ensemble.t, center, color=color, **DETERMINISTIC_STYLE)
                if var in references:
                    axes[c].axhline(
                        references[var][row],
                        **(
                            REFERENCE_STYLE | {"color": color} | (reference_style or {})
                        ),
                    )

            group_title = None
            if g is not None and ensemble.num_groups > 1:
                group_title = f"Initial state {g + 1}"
                suffix = f"{suffix}_x0_{g + 1}"
            if per_strategy:
                titles = [
                    label if group_title is None else f"{label}, {group_title.lower()}"
                    for label in labels
                ]
                to_format = zip(axes, edges, titles)
            else:
                to_format = [(axes[0], (True, True), group_title)]
            ylabel = (
                rf"$\mathbf{{{var}}}(t)$"
                if k is None or P == 1
                else rf"$\mathbf{{{var}}}^{{{k + 1}}}(t)$"
            )
            for ax, (bottom, left), title in to_format:
                if xlim is not None:
                    ax.set_xlim(xlim)
                if isinstance(ylim, dict) and var in ylim:
                    ax.set_ylim(ylim[var])
                if title is not None:
                    ax.set_title(title, fontsize=fontsize)
                ax.tick_params(labelsize=fontsize)
                ax.grid()
                if bottom:
                    ax.set_xlabel(r"$t$", fontsize=fontsize)
                if left:
                    ax.set_ylabel(ylabel, fontsize=fontsize)
            figures.tight_layout(fig)
            if not per_strategy:
                if legend:
                    axes[0].legend(fontsize=fontsize, ncol=len(rows))
            figures.finish(fig, filename, suffix, show)


def plot_ensemble_kpi(
    ensemble: EnsembleResult,
    kpi_function: Callable[[SimpleNamespace], np.ndarray] = None,
    bands: Sequence[tuple[float, float]] = DEFAULT_BANDS,
    plot_median: bool = True,
    plot_deterministic_approximation: bool = False,
    pool_initial_states: bool = None,
    statistic: Literal["quantile", "mean_std"] = "quantile",
    num_std: Sequence[float] = DEFAULT_NUM_STD,
    linewidth: float = 1.0,
    band_alpha: float | Sequence[float] = None,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    yscale: str = "linear",
    filename: str = None,
    figsize: tuple[float, float] = FIGSIZE,
    fontsize: int = FONTSIZE,
    show: bool = True,
    ax: Axes = None,
    legend: bool = True,
) -> None:
    """
    Plot the median and quantile bands (or the mean and standard deviation bands) of a KPI over time.

    The KPI is evaluated on each run separately (see ``EnsembleResult.__getitem__``). If the ensemble has several
    initial states (groups), one band is plotted per group, with one color per group, unless they are pooled into a
    single band (the default for deterministic ensembles).

    Args:
        ensemble (EnsembleResult): The ensemble to plot.
        kpi_function (Callable[[SimpleNamespace], np.ndarray], optional): Function mapping a run (with fields ``t``,
            ``x``, ``q``, and ``p`` in the layout of ``Simulator.log``) to the KPI over time, of shape ``(K,)``.
            Defaults to None, in which case the Euclidean distance to the GNE (normalized by its initial value) is
            used.
        bands (Sequence[tuple[float, float]], optional): Pairs of lower and upper quantiles delimiting the bands
            (with ``statistic='quantile'``). Defaults to ((0.05, 0.95), (0.25, 0.75)).
        plot_median (bool, optional): Whether to plot the central line: the median, or the mean with
            ``statistic='mean_std'``. Defaults to True.
        plot_deterministic_approximation (bool, optional): Whether to plot the KPI of the deterministic approximation
            (EDM-PDM). Ignored for deterministic ensembles. Defaults to False.
        pool_initial_states (bool, optional): Whether to plot a single band mixing the runs of all initial states.
            Defaults to None, in which case it is True for deterministic ensembles and False otherwise.
        statistic (str, optional): 'quantile' or 'mean_std', see ``plot_ensemble_bands``. Defaults to 'quantile'.
        num_std (Sequence[float], optional): The widths of the bands in standard deviations (with
            ``statistic='mean_std'``). Defaults to (1,).
        linewidth (float, optional): Line width of the central line. Defaults to 1.
        band_alpha (float | Sequence[float], optional): Opacity of the bands: one value for all of them, or one per
            band (widest first). Defaults to None, in which case 0.15 for the widest band, 0.25 for the next one, etc.
        xlim (tuple[float, float], optional): Limits of the time axis. Defaults to None.
        ylim (tuple[float, float], optional): Limits of the vertical axis. Defaults to None.
        yscale (str, optional): Scale of the vertical axis. Defaults to 'linear'.
        filename (str, optional): Filename to save the figure. Defaults to None.
        figsize (tuple[float, float], optional): Figure size. Defaults to (4, 2).
        fontsize (int, optional): Font size. Defaults to 8.
        show (bool, optional): Whether to show the figure. Defaults to True.
        ax (Axes, optional): Existing ``matplotlib`` axes to draw on. The figure is then neither saved, shown, nor
            closed. Defaults to None.
        legend (bool, optional): Whether to draw the legend(s). If False, the entries remain available through
            ``ax.get_legend_handles_labels()``, e.g., for one legend shared by several plots (see
            ``popgames.plotting.add_shared_legend``). Defaults to True.
    """
    figures = Panels(ax, 1, filename)
    stat = _statistic(statistic, bands, num_std)
    if pool_initial_states is None:
        pool_initial_states = ensemble.is_deterministic
    run_sets = (
        [(None, np.arange(ensemble.num_runs))]
        if pool_initial_states
        else list(enumerate(ensemble.groups))
    )
    if kpi_function is None:
        kpi_function = make_default_kpi_function(ensemble.simulator)
        if kpi_function is None:
            return None

    G = len(run_sets)
    if G > MAX_GROUP_COLORS:
        logger.warning(
            f"The ensemble has {G} initial states: the KPI plot shows one band per initial state, which may be hard "
            f"to read. Consider pooling them (pool_initial_states=True) or plotting a subset (EnsembleResult.subset)."
        )

    kpi = np.vstack([np.asarray(kpi_function(run)).reshape(-1) for run in ensemble])
    det = _deterministic(ensemble, plot_deterministic_approximation)
    if det is not None:
        kpi_det = np.vstack(
            [
                np.asarray(kpi_function(_run(det, r))).reshape(-1)
                for r in range(ensemble.num_runs)
            ]
        )

    fig, ax = figures.new(figsize)
    for g, (_, runs) in enumerate(run_sets):
        color = "black" if G == 1 else _group_color(g, G)
        label = (
            _runs_label(ensemble)
            if G == 1
            else (rf"$\mathbf{{x}}_0$ #{g + 1}" if G <= MAX_GROUP_COLORS else None)
        )
        _plot_bands(
            ax,
            ensemble.t,
            kpi[runs],
            stat,
            plot_median,
            color,
            label,
            linewidth,
            band_alpha,
        )
        if det is not None:
            ax.plot(
                ensemble.t,
                stat(kpi_det[runs])[0],
                color="magenta" if G == 1 else color,
                label="EDM-PDM" if G == 1 else None,
                **DETERMINISTIC_STYLE,
            )
    if det is not None and G > 1:
        ax.plot([], [], color="gray", label="EDM-PDM", **DETERMINISTIC_STYLE)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_yscale(yscale)
    _format_time_axes(ax, r"$\operatorname{KPI}(t)$", fontsize)
    if legend:
        _legend(figures, ax, fontsize, outside=G > 1)
    else:
        figures.tight_layout(fig)
    figures.finish(fig, filename, None, show)


def plot_ensemble_ternary(
    ensemble: EnsembleResult,
    max_runs: int = 50,
    alpha: float = 0.2,
    plot_mean: bool = False,
    plot_deterministic_approximation: bool = False,
    plot_gne: bool = True,
    filename: str = None,
    figsize: tuple[float, float] = FIGSIZE_TERNARY,
    fontsize: int = FONTSIZE,
    show: bool = True,
    ax: Axes | Sequence[Axes] = None,
    legend: bool = True,
) -> None:
    """
    Plot the trajectories of the runs on the simplex, as semi-transparent lines.

    One ternary plot is generated per population with 3 strategies (other populations are skipped with a warning).
    If the ensemble has several initial states (groups), the plot is a phase portrait: the runs of each group have
    their own color (up to 10 groups; black otherwise), and the initial states are marked.

    Args:
        ensemble (EnsembleResult): The ensemble to plot.
        max_runs (int, optional): Maximum number of runs to plot per initial state (the first ones). Defaults to 50.
        alpha (float, optional): Opacity of each run. Defaults to 0.2.
        plot_mean (bool, optional): Whether to plot the mean over the runs of each initial state. Defaults to False.
        plot_deterministic_approximation (bool, optional): Whether to plot the trajectory of the deterministic
            approximation (EDM-PDM) from each initial state. Defaults to False.
        plot_gne (bool, optional): Whether to plot the GNE (if one is found). Defaults to True.
        filename (str, optional): Filename to save the figures. With several populations, the population is
            appended to the file name, e.g., ``ternary_pop_1.pdf``. Defaults to None.
        figsize (tuple[float, float], optional): Figure size. Defaults to (4, 3).
        fontsize (int, optional): Font size. Defaults to 8.
        show (bool, optional): Whether to show the figures. Defaults to True.
        ax (Axes | Sequence[Axes], optional): Existing ``matplotlib`` axes to draw on, one per population with 3
            strategies. The figures are then neither saved, shown, nor closed. Defaults to None.
        legend (bool, optional): Whether to draw the legend(s). If False, the entries remain available through
            ``ax.get_legend_handles_labels()``, e.g., for one legend shared by several plots (see
            ``popgames.plotting.add_shared_legend``). Defaults to True.
    """
    populations = list(_ternary_populations(ensemble))
    figures = Panels(ax, len(populations), filename)
    G = ensemble.num_groups
    det = _deterministic(ensemble, plot_deterministic_approximation)
    gne = ensemble.simulator.population_game.compute_gne() if plot_gne else None
    run_color = "black" if G == 1 else None
    legend_color = "black" if G == 1 else "gray"

    for k, s, suffix in populations:
        fig, tax = _ternary_figure(figures, figsize)
        for g, runs in enumerate(ensemble.groups):
            color = run_color or _group_color(g, G)
            for r in runs[:max_runs]:
                tax.plot(
                    _to_ternary(ensemble.x[r, s]),
                    linewidth=0.6,
                    color=color,
                    alpha=alpha,
                )
            if plot_mean:
                tax.plot(
                    _to_ternary(ensemble.x[runs, s].mean(axis=0)),
                    linewidth=1.5,
                    color="tab:blue" if G == 1 else color,
                )
            if det is not None:
                tax.plot(
                    _to_ternary(det.x[runs[0], s]),
                    color="magenta" if G == 1 else color,
                    **DETERMINISTIC_STYLE,
                )
            if G > 1:
                tax.plot(
                    _to_ternary(ensemble.x0[runs[0], s]),
                    marker="o",
                    markersize=3,
                    linestyle="",
                    color=color,
                    zorder=3,
                )

        handles = [
            plt.Line2D(
                [], [], linewidth=1, color=legend_color, label=_runs_label(ensemble)
            )
        ]
        if plot_mean:
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    linewidth=1.5,
                    color="tab:blue" if G == 1 else legend_color,
                    label="Mean",
                )
            )
        if det is not None:
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    color="magenta" if G == 1 else legend_color,
                    label="EDM-PDM",
                    **DETERMINISTIC_STYLE,
                )
            )
        if G > 1:
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    markersize=3,
                    linestyle="",
                    color=legend_color,
                    label="Initial states",
                )
            )
        if gne is not None:
            handles.append(_plot_gne(tax, gne[s], ensemble))

        _format_ternary(tax, handles, fontsize, legend)
        figures.finish(fig, filename, suffix, show)


def plot_ensemble_final(
    ensemble: EnsembleResult,
    t: float = None,
    bins: int = 20,
    alpha: float = 0.5,
    plot_deterministic_approximation: bool = False,
    plot_gne: bool = True,
    filename: str = None,
    figsize: tuple[float, float] = None,
    fontsize: int = FONTSIZE,
    show: bool = True,
    ax: Axes | Sequence[Axes] = None,
    legend: bool = True,
) -> None:
    """
    Plot the distribution of the strategic distributions of the runs at a given time (the final time by default).

    One figure is generated per population: a scatter plot on the simplex for populations with 3 strategies, and
    histograms of the strategies otherwise (only of the first strategy for populations with 2 strategies, since the
    second one is determined by the first). If the ensemble has several initial states (groups), the points and
    histograms of each group have their own color (up to 10 groups; black otherwise), and the histograms are plotted
    in one figure per strategy.

    Args:
        ensemble (EnsembleResult): The ensemble to plot.
        t (float, optional): The time (the last sampling time at or before ``t`` is used). Defaults to None, in which
            case the final time is used.
        bins (int, optional): Number of bins of the histograms. Defaults to 20.
        alpha (float, optional): Opacity of the points (scatter plots) or bars (histograms). Defaults to 0.5.
        plot_deterministic_approximation (bool, optional): Whether to mark the state of the deterministic
            approximation (EDM-PDM) from each initial state at the same time. Defaults to False.
        plot_gne (bool, optional): Whether to mark the GNE (if one is found). Defaults to True.
        filename (str, optional): Filename to save the figures. The population (with several populations) and the
            strategy (for histograms with several initial states) are appended to the file name, e.g.,
            ``final_pop_1.pdf`` or ``final_pop_1_x_2.pdf``. Defaults to None.
        figsize (tuple[float, float], optional): Figure size. Defaults to None, in which case (4, 3) is used for
            scatter plots on the simplex and (4, 2) for histograms.
        fontsize (int, optional): Font size. Defaults to 8.
        show (bool, optional): Whether to show the figures. Defaults to True.
        ax (Axes | Sequence[Axes], optional): Existing ``matplotlib`` axes to draw on, one per figure that would
            otherwise be generated, in the order of the populations (and, for histograms with several initial
            states, of the strategies). The figures are then neither saved, shown, nor closed. Defaults to None.
        legend (bool, optional): Whether to draw the legend(s). If False, the entries remain available through
            ``ax.get_legend_handles_labels()``, e.g., for one legend shared by several plots (see
            ``popgames.plotting.add_shared_legend``). Defaults to True.
    """
    G = ensemble.num_groups
    game = ensemble.simulator.population_game
    num_panels = sum(1 if nk <= 3 or G == 1 else nk for nk in game.num_strategies)
    figures = Panels(ax, num_panels, filename)
    snapshot = ensemble.final() if t is None else ensemble.at(t)
    k_t = int(np.flatnonzero(ensemble.t == snapshot.t)[-1])
    det = _deterministic(ensemble, plot_deterministic_approximation)
    x_det = det.x[:, :, k_t] if det is not None else None
    gne = ensemble.simulator.population_game.compute_gne() if plot_gne else None
    if gne is not None:
        gne = gne.reshape(-1)

    P = game.num_populations
    pos = 0
    for k in range(P):
        nk = game.num_strategies[k]
        s = slice(pos, pos + nk)
        pos += nk
        suffix = f"pop_{k + 1}" if P > 1 else None

        if nk == 3:
            fig, tax = _ternary_figure(figures, figsize or FIGSIZE_TERNARY)
            for g, runs in enumerate(ensemble.groups):
                color = "black" if G == 1 else _group_color(g, G)
                tax.plot(
                    _to_ternary(snapshot.x[runs, s].T),
                    marker="o",
                    markersize=2.5,
                    linestyle="",
                    color=color,
                    alpha=alpha,
                )
                if x_det is not None:
                    tax.plot(
                        _to_ternary(x_det[runs[0], s].reshape(3, 1)),
                        marker="D",
                        markersize=4,
                        linestyle="",
                        color="magenta" if G == 1 else color,
                        markeredgecolor="black" if G > 1 else None,
                        zorder=3,
                    )

            legend_color = "black" if G == 1 else "gray"
            handles = [
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    markersize=3,
                    color=legend_color,
                    linestyle="",
                    label=_runs_label(ensemble),
                )
            ]
            if x_det is not None:
                handles.append(
                    plt.Line2D(
                        [],
                        [],
                        marker="D",
                        markersize=4,
                        color="magenta" if G == 1 else legend_color,
                        linestyle="",
                        label="EDM-PDM",
                    )
                )
            if gne is not None:
                handles.append(_plot_gne(tax, gne[s], ensemble))
            _format_ternary(tax, handles, fontsize, legend)
            tax.set_title(rf"$t = {snapshot.t:g}$", fontsize=fontsize, pad=20)
            figures.finish(fig, filename, suffix, show)
            continue

        strategies = range(s.start, s.start + 1) if nk == 2 else range(s.start, s.stop)
        if G == 1:
            # One figure per population, one color per strategy
            fig, ax = figures.new(figsize or FIGSIZE)
            for c, i in enumerate(strategies):
                color = f"C{c % 10}"
                ax.hist(
                    snapshot.x[:, i],
                    bins=bins,
                    color=color,
                    alpha=alpha,
                    label=_strategy_label("x", i, k, P),
                )
                if x_det is not None:
                    ax.axvline(x_det[0, i], color=color, **DETERMINISTIC_STYLE)
                if gne is not None:
                    ax.axvline(gne[i], color=color, linestyle="dashed", linewidth=1)
            _format_histogram(ax, r"$\mathbf{x}(t)$", snapshot.t, fontsize)
            figures.tight_layout(fig)
            if legend:
                ax.legend(fontsize=fontsize, ncol=len(strategies))
            figures.finish(fig, filename, suffix, show)
            continue

        # One figure per strategy, one color per initial state
        for i in strategies:
            fig, ax = figures.new(figsize or FIGSIZE)
            edges = np.histogram_bin_edges(snapshot.x[:, i], bins=bins)
            for g, runs in enumerate(ensemble.groups):
                color = _group_color(g, G)
                ax.hist(
                    snapshot.x[runs, i],
                    bins=edges,
                    color=color,
                    alpha=alpha,
                    label=rf"$\mathbf{{x}}_0$ #{g + 1}"
                    if G <= MAX_GROUP_COLORS
                    else None,
                )
                if x_det is not None:
                    ax.axvline(x_det[runs[0], i], color=color, **DETERMINISTIC_STYLE)
            if gne is not None:
                ax.axvline(gne[i], color="tab:red", linestyle="dashed", linewidth=1)
            _format_histogram(ax, _strategy_label("x", i, k, P), snapshot.t, fontsize)
            figures.tight_layout(fig)
            if legend and G <= MAX_GROUP_COLORS:
                _legend(figures, ax, fontsize, outside=True)
            strategy_suffix = f"x_{i + 1}"
            figures.finish(
                fig,
                filename,
                f"{suffix}_{strategy_suffix}" if suffix else strategy_suffix,
                show,
            )


def _check_variables(variables: Sequence[str]) -> None:
    unknown = set(variables) - {"x", "p", "q"}
    if unknown:
        raise ValueError(
            f"Unknown variables {sorted(unknown)}. Use 'x', 'p', and/or 'q'."
        )


def _selected_groups(ensemble: EnsembleResult, initial_state: int | None) -> list[int]:
    G = ensemble.num_groups
    if initial_state is None:
        if G > MAX_GROUP_COLORS:
            logger.warning(
                f"The ensemble has {G} initial states: plotting the figures for each of them. "
                f"Use initial_state=g to plot a single one."
            )
        return list(range(G))
    if not isinstance(initial_state, (int, np.integer)) or not 0 <= initial_state < G:
        raise ValueError(
            f"initial_state must be an integer in [0, {G - 1}], got {initial_state}."
        )
    return [int(initial_state)]


def _run_sets(
    ensemble: EnsembleResult, initial_state: int | None, pool: bool | None
) -> list[tuple[int | None, np.ndarray]]:
    """
    The sets of runs over which the statistics are computed, as ``(group, runs)`` pairs: one per selected group, or a
    single one with all runs (and group None) if the initial states are pooled (by default, for deterministic
    ensembles).
    """
    if pool is None:
        pool = ensemble.is_deterministic
    if not pool:
        return [
            (g, ensemble.groups[g]) for g in _selected_groups(ensemble, initial_state)
        ]
    if initial_state is not None:
        raise ValueError(
            "initial_state cannot be combined with pooled initial states (pool_initial_states)."
        )
    return [(None, np.arange(ensemble.num_runs))]


def _deterministic(
    ensemble: EnsembleResult, plot_deterministic_approximation: bool
) -> SimpleNamespace | None:
    """The deterministic approximation to overlay, if requested (None for deterministic ensembles: they are it)."""
    if not plot_deterministic_approximation or ensemble.is_deterministic:
        return None
    return ensemble.deterministic()


def _runs_label(ensemble: EnsembleResult) -> str:
    return "EDM-PDM" if ensemble.is_deterministic else "Finite agents"


def _statistic(
    statistic: str,
    bands: Sequence[tuple[float, float]],
    num_std: Sequence[float],
) -> Callable[[np.ndarray], tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]]:
    """
    Make the function mapping ``values`` (shape ``(M, K)``) to the central line (shape ``(K,)``) and the bands (pairs
    of lower and upper bounds, widest first) over the runs.
    """
    if statistic == "quantile":

        def quantile_bands(values):
            intervals = [tuple(np.quantile(values, b, axis=0)) for b in bands]
            return np.median(values, axis=0), intervals

        return quantile_bands

    if statistic == "mean_std":
        widths = sorted(np.asarray(num_std, dtype=float).reshape(-1), reverse=True)
        if not widths or widths[-1] <= 0:
            raise ValueError(
                f"num_std must be a non-empty sequence of positive numbers, got {num_std}."
            )

        def mean_std_bands(values):
            mean, std = values.mean(axis=0), values.std(axis=0)
            return mean, [(mean - w * std, mean + w * std) for w in widths]

        return mean_std_bands

    raise ValueError(f"statistic must be 'quantile' or 'mean_std', got {statistic!r}.")


def _band_alphas(
    band_alpha: float | Sequence[float] | None, num_bands: int
) -> list[float]:
    """
    The opacity of each band, widest first: 0.15, 0.25, ... by default, the same for all bands if a scalar is given,
    or one per band.
    """
    if band_alpha is None:
        return [0.15 + 0.1 * j for j in range(num_bands)]
    alphas = np.asarray(band_alpha, dtype=float).reshape(-1)
    if alphas.size == 1:
        alphas = np.repeat(alphas, num_bands)
    if alphas.size != num_bands or np.any((alphas < 0) | (alphas > 1)):
        raise ValueError(
            f"band_alpha must be a number in [0, 1] or one per band ({num_bands}), got {band_alpha}."
        )
    return alphas.tolist()


def _group_color(g: int, G: int) -> str:
    return f"C{g % 10}" if G <= MAX_GROUP_COLORS else "black"


def _run(ns: SimpleNamespace, r: int) -> SimpleNamespace:
    """Run ``r`` of an ensemble-shaped namespace, in the layout of ``Simulator.log``."""
    return SimpleNamespace(t=ns.t, x=ns.x[r], q=ns.q[r], p=ns.p[r])


def _legend(figures: Panels, ax: Axes, fontsize: int, outside: bool = False) -> None:
    """Add a legend, inside the axes or to their right (e.g., one entry per initial state)."""
    if outside:
        ax.legend(fontsize=fontsize, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    else:
        ax.legend(fontsize=fontsize)
    figures.tight_layout(ax.figure)


def _format_histogram(ax: Axes, xlabel: str, t: float, fontsize: int) -> None:
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel(rf"{xlabel},  $t = {t:g}$", fontsize=fontsize)
    ax.set_ylabel("Number of runs", fontsize=fontsize)
    ax.grid()


def _strategy_label(var: str, i: int, k: int, P: int) -> str:
    return rf"${var}_{{{i + 1}}}$" if P == 1 else rf"${var}_{{{i + 1}}}^{{{k + 1}}}$"


def _plot_bands(
    ax: Axes,
    t: np.ndarray,
    values: np.ndarray,
    stat: Callable,
    plot_center: bool,
    color: str,
    label: str,
    linewidth: float = 1.0,
    band_alpha: float | Sequence[float] | None = None,
) -> None:
    """Plot the central line and bands (see ``_statistic``) of ``values`` (shape ``(M, K)``) over ``t``."""
    center, intervals = stat(values)
    for (lower, upper), alpha in zip(
        intervals, _band_alphas(band_alpha, len(intervals))
    ):
        ax.fill_between(t, lower, upper, color=color, alpha=alpha, linewidth=0)
    if plot_center:
        ax.plot(t, center, color=color, linewidth=linewidth, label=label)
    else:
        ax.fill_between([], [], [], color=color, alpha=0.3, label=label)


def _format_time_axes(ax: Axes, ylabel: str, fontsize: int) -> None:
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel(r"$t$", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.grid()


def _ternary_populations(ensemble: EnsembleResult):
    """Yield ``(k, slice, file suffix)`` for each population with 3 strategies."""
    game = ensemble.simulator.population_game
    P = game.num_populations
    pos = 0
    for k in range(P):
        nk = game.num_strategies[k]
        if nk == 3:
            yield k, slice(pos, pos + nk), (f"pop_{k + 1}" if P > 1 else None)
        else:
            logger.warning(
                f"Ternary plots require 3 strategies per population. Skipping population {k + 1} ({nk} strategies)."
            )
        pos += nk


def _to_ternary(x_k: np.ndarray) -> np.ndarray:
    """Map strategic distributions of shape ``(3, K)`` to ternary coordinates of shape ``(K, 3)``."""
    x_k = np.asarray(x_k, dtype=float)
    return (x_k[[2, 0, 1]] / x_k.sum(axis=0)).T


def _ternary_figure(figures: Panels, figsize: tuple[float, float]):
    fig, tax = figures.new_ternary(figsize)
    tax.boundary(linewidth=1.0)
    return fig, tax


def _plot_gne(tax, gne_k: np.ndarray, ensemble: EnsembleResult) -> plt.Line2D:
    game = ensemble.simulator.population_game
    label = (
        r"$\operatorname{GNE}$"
        if game.d_eq + game.d_ineq > 0
        else r"$\operatorname{NE}$"
    )
    tax.plot(
        _to_ternary(np.reshape(gne_k, (3, 1))),
        marker="*",
        markersize=7,
        linestyle="",
        color="tab:red",
        zorder=4,
    )
    return plt.Line2D(
        [], [], marker="*", markersize=7, color="tab:red", linestyle="", label=label
    )


def _format_ternary(tax, handles: list, fontsize: int, legend: bool = True) -> None:
    tax.top_corner_label(r"$e_1$", fontsize=fontsize)
    tax.left_corner_label(r"$e_2$", fontsize=fontsize)
    tax.right_corner_label(r"$e_3$", fontsize=fontsize)
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")
    ternary_legend(tax, handles, fontsize, legend)
