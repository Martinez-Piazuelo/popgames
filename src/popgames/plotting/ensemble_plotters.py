from __future__ import annotations

import logging
import os
import typing

import matplotlib.pyplot as plt
import numpy as np
import ternary

from popgames.plotting._plot_config import (
    DPI,
    FIGSIZE,
    FIGSIZE_TERNARY,
    FONTSIZE,
)
from popgames.plotting.plotters import make_default_kpi_function

from types import SimpleNamespace

if typing.TYPE_CHECKING:
    from typing import Callable, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from popgames.ensemble import EnsembleResult

__all__ = [
    "plot_ensemble_bands",
    "plot_ensemble_kpi",
    "plot_ensemble_ternary",
    "plot_ensemble_final",
]

logger = logging.getLogger(__name__)

DEFAULT_BANDS = ((0.05, 0.95), (0.25, 0.75))
DETERMINISTIC_STYLE = {"linestyle": "dotted", "linewidth": 1.5}
MAX_GROUP_COLORS = 10  # Groups (initial states) beyond this number are plotted in black


def plot_ensemble_bands(
    ensemble: EnsembleResult,
    variables: Sequence[str] = ("x", "p", "q"),
    bands: Sequence[tuple[float, float]] = DEFAULT_BANDS,
    plot_median: bool = True,
    plot_deterministic_approximation: bool = False,
    initial_state: int = None,
    xlim: tuple[float, float] = None,
    ylim: dict[str, tuple[float, float]] = None,
    filename: str = None,
    figsize: tuple[float, float] = FIGSIZE,
    fontsize: int = FONTSIZE,
    show: bool = True,
) -> None:
    """
    Plot the median and quantile bands of the runs over time.

    One figure is generated per population for ``x`` and ``p``, and one for ``q`` (if ``d > 0``). Each strategy has
    its own color: the solid line is the median over the runs, and the shaded regions are the bands between the given
    quantiles (e.g., 90% and 50% of the runs at each time with the default bands).

    The bands only mix runs with the same initial state: if the ensemble has several initial states (groups), the
    figures are generated for each of them (or only for ``initial_state``).

    Args:
        ensemble (EnsembleResult): The ensemble to plot.
        variables (Sequence[str], optional): The variables to plot, among 'x', 'p', and 'q'. Defaults to all of them.
        bands (Sequence[tuple[float, float]], optional): Pairs of lower and upper quantiles delimiting the bands.
            Defaults to ((0.05, 0.95), (0.25, 0.75)).
        plot_median (bool, optional): Whether to plot the median. Defaults to True.
        plot_deterministic_approximation (bool, optional): Whether to plot the trajectories of the deterministic
            approximation (EDM-PDM) as dotted lines. Defaults to False.
        initial_state (int, optional): The group (initial state) to plot, see ``EnsembleResult.groups``. Defaults to
            None, in which case all groups are plotted.
        xlim (tuple[float, float], optional): Limits of the time axis. Defaults to None.
        ylim (dict[str, tuple[float, float]], optional): Limits of the vertical axis per variable, e.g.,
            ``{'x': (0, 1)}``. Defaults to None.
        filename (str, optional): If provided, each figure is saved with the variable and population (and the group,
            if there are several) appended to the file name, e.g., ``bands_x_1.pdf`` or ``bands_x_1_x0_2.pdf``.
            Defaults to None.
        figsize (tuple[float, float], optional): Figure size. Defaults to (4, 2).
        fontsize (int, optional): Font size. Defaults to 8.
        show (bool, optional): Whether to show the figures. Defaults to True.
    """
    _check_variables(variables)
    group_ids = _selected_groups(ensemble, initial_state)
    det = ensemble.deterministic() if plot_deterministic_approximation else None
    game = ensemble.simulator.population_game
    P = game.num_populations

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

    for g in group_ids:
        runs = ensemble.groups[g]
        for var, k, rows, labels, suffix in panels:
            values = getattr(ensemble, var)[runs]
            fig, ax = plt.subplots(figsize=figsize)
            for c, (row, label) in enumerate(zip(rows, labels)):
                color = f"C{c % 10}"
                _plot_bands(
                    ax, ensemble.t, values[:, row], bands, plot_median, color, label
                )
                if det is not None:
                    ax.plot(
                        ensemble.t,
                        getattr(det, var)[runs[0], row],
                        color=color,
                        **DETERMINISTIC_STYLE,
                    )

            if xlim is not None:
                ax.set_xlim(xlim)
            if isinstance(ylim, dict) and var in ylim:
                ax.set_ylim(ylim[var])
            if ensemble.num_groups > 1:
                ax.set_title(f"Initial state {g + 1}", fontsize=fontsize)
                suffix = f"{suffix}_x0_{g + 1}"
            ylabel = (
                rf"$\mathbf{{{var}}}(t)$"
                if k is None or P == 1
                else rf"$\mathbf{{{var}}}^{{{k + 1}}}(t)$"
            )
            _format_time_axes(ax, ylabel, fontsize)
            ax.legend(fontsize=fontsize, ncol=len(rows))
            _finish(fig, filename, suffix, show)


def plot_ensemble_kpi(
    ensemble: EnsembleResult,
    kpi_function: Callable[[SimpleNamespace], np.ndarray] = None,
    bands: Sequence[tuple[float, float]] = DEFAULT_BANDS,
    plot_median: bool = True,
    plot_deterministic_approximation: bool = False,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    yscale: str = "linear",
    filename: str = None,
    figsize: tuple[float, float] = FIGSIZE,
    fontsize: int = FONTSIZE,
    show: bool = True,
) -> None:
    """
    Plot the median and quantile bands of a KPI over time.

    The KPI is evaluated on each run separately (see ``EnsembleResult.__getitem__``). If the ensemble has several
    initial states (groups), one band is plotted per group, with one color per group.

    Args:
        ensemble (EnsembleResult): The ensemble to plot.
        kpi_function (Callable[[SimpleNamespace], np.ndarray], optional): Function mapping a run (with fields ``t``,
            ``x``, ``q``, and ``p`` in the layout of ``Simulator.log``) to the KPI over time, of shape ``(K,)``.
            Defaults to None, in which case the Euclidean distance to the GNE (normalized by its initial value) is
            used.
        bands (Sequence[tuple[float, float]], optional): Pairs of lower and upper quantiles delimiting the bands.
            Defaults to ((0.05, 0.95), (0.25, 0.75)).
        plot_median (bool, optional): Whether to plot the median. Defaults to True.
        plot_deterministic_approximation (bool, optional): Whether to plot the KPI of the deterministic approximation
            (EDM-PDM). Defaults to False.
        xlim (tuple[float, float], optional): Limits of the time axis. Defaults to None.
        ylim (tuple[float, float], optional): Limits of the vertical axis. Defaults to None.
        yscale (str, optional): Scale of the vertical axis. Defaults to 'linear'.
        filename (str, optional): Filename to save the figure. Defaults to None.
        figsize (tuple[float, float], optional): Figure size. Defaults to (4, 2).
        fontsize (int, optional): Font size. Defaults to 8.
        show (bool, optional): Whether to show the figure. Defaults to True.
    """
    if kpi_function is None:
        kpi_function = make_default_kpi_function(ensemble.simulator)
        if kpi_function is None:
            return None

    G = ensemble.num_groups
    if G > MAX_GROUP_COLORS:
        logger.warning(
            f"The ensemble has {G} initial states: the KPI plot shows one band per initial state, which may be hard "
            f"to read. Consider plotting a subset (EnsembleResult.subset)."
        )

    kpi = np.vstack([np.asarray(kpi_function(run)).reshape(-1) for run in ensemble])
    det = ensemble.deterministic() if plot_deterministic_approximation else None

    fig, ax = plt.subplots(figsize=figsize)
    for g, runs in enumerate(ensemble.groups):
        color = "black" if G == 1 else _group_color(g, G)
        label = (
            "Finite agents"
            if G == 1
            else (rf"$\mathbf{{x}}_0$ #{g + 1}" if G <= MAX_GROUP_COLORS else None)
        )
        _plot_bands(ax, ensemble.t, kpi[runs], bands, plot_median, color, label)
        if det is not None:
            kpi_det = np.asarray(kpi_function(_run(det, runs[0]))).reshape(-1)
            ax.plot(
                ensemble.t,
                kpi_det,
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
    ax.legend(fontsize=fontsize)
    _finish(fig, filename, None, show)


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
    """
    G = ensemble.num_groups
    det = ensemble.deterministic() if plot_deterministic_approximation else None
    gne = ensemble.simulator.population_game.compute_gne() if plot_gne else None
    run_color = "black" if G == 1 else None
    legend_color = "black" if G == 1 else "gray"

    for k, s, suffix in _ternary_populations(ensemble):
        fig, tax = _ternary_figure(figsize)
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
            plt.Line2D([], [], linewidth=1, color=legend_color, label="Finite agents")
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

        _format_ternary(tax, handles, fontsize)
        _finish(fig, filename, suffix, show)


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
    """
    G = ensemble.num_groups
    snapshot = ensemble.final() if t is None else ensemble.at(t)
    k_t = int(np.flatnonzero(ensemble.t == snapshot.t)[-1])
    x_det = (
        ensemble.deterministic().x[:, :, k_t]
        if plot_deterministic_approximation
        else None
    )
    gne = ensemble.simulator.population_game.compute_gne() if plot_gne else None
    if gne is not None:
        gne = gne.reshape(-1)

    game = ensemble.simulator.population_game
    P = game.num_populations
    pos = 0
    for k in range(P):
        nk = game.num_strategies[k]
        s = slice(pos, pos + nk)
        pos += nk
        suffix = f"pop_{k + 1}" if P > 1 else None

        if nk == 3:
            fig, tax = _ternary_figure(figsize or FIGSIZE_TERNARY)
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
                    label="Finite agents",
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
            _format_ternary(tax, handles, fontsize)
            tax.set_title(rf"$t = {snapshot.t:g}$", fontsize=fontsize, pad=20)
            _finish(fig, filename, suffix, show)
            continue

        strategies = range(s.start, s.start + 1) if nk == 2 else range(s.start, s.stop)
        if G == 1:
            # One figure per population, one color per strategy
            fig, ax = plt.subplots(figsize=figsize or FIGSIZE)
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
            ax.legend(fontsize=fontsize, ncol=len(strategies))
            _finish(fig, filename, suffix, show)
            continue

        # One figure per strategy, one color per initial state
        for i in strategies:
            fig, ax = plt.subplots(figsize=figsize or FIGSIZE)
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
            if G <= MAX_GROUP_COLORS:
                ax.legend(fontsize=fontsize)
            strategy_suffix = f"x_{i + 1}"
            _finish(
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


def _group_color(g: int, G: int) -> str:
    return f"C{g % 10}" if G <= MAX_GROUP_COLORS else "black"


def _run(ns: SimpleNamespace, r: int) -> SimpleNamespace:
    """Run ``r`` of an ensemble-shaped namespace, in the layout of ``Simulator.log``."""
    return SimpleNamespace(t=ns.t, x=ns.x[r], q=ns.q[r], p=ns.p[r])


def _format_histogram(ax: Axes, xlabel: str, t: float, fontsize: int) -> None:
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel(rf"{xlabel},  $t = {t:g}$", fontsize=fontsize)
    ax.set_ylabel("Number of runs", fontsize=fontsize)
    ax.grid()
    ax.figure.tight_layout()


def _strategy_label(var: str, i: int, k: int, P: int) -> str:
    return rf"${var}_{{{i + 1}}}$" if P == 1 else rf"${var}_{{{i + 1}}}^{{{k + 1}}}$"


def _plot_bands(
    ax: Axes,
    t: np.ndarray,
    values: np.ndarray,
    bands: Sequence[tuple[float, float]],
    plot_median: bool,
    color: str,
    label: str,
) -> None:
    """Plot the median and quantile bands of ``values`` (shape ``(M, K)``) over ``t``."""
    for j, (lower, upper) in enumerate(bands):
        q_lower, q_upper = np.quantile(values, [lower, upper], axis=0)
        ax.fill_between(
            t, q_lower, q_upper, color=color, alpha=0.15 + 0.1 * j, linewidth=0
        )
    if plot_median:
        ax.plot(t, np.median(values, axis=0), color=color, linewidth=1, label=label)
    else:
        ax.fill_between([], [], [], color=color, alpha=0.3, label=label)


def _format_time_axes(ax: Axes, ylabel: str, fontsize: int) -> None:
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel(r"$t$", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.grid()
    ax.figure.tight_layout()


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


def _ternary_figure(figsize: tuple[float, float]):
    fig, tax = ternary.figure(scale=1)
    fig.set_size_inches(figsize[0], figsize[1])
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


def _format_ternary(tax, handles: list, fontsize: int) -> None:
    tax.top_corner_label(r"$e_1$", fontsize=fontsize)
    tax.left_corner_label(r"$e_2$", fontsize=fontsize)
    tax.right_corner_label(r"$e_3$", fontsize=fontsize)
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")
    tax.legend(handles=handles, loc=1, fontsize=fontsize)


def _finish(fig: Figure, filename: str | None, suffix: str | None, show: bool) -> None:
    """Save the figure (with the suffix appended to the file name), then show or close it."""
    if filename is not None:
        name, ext = os.path.splitext(filename)
        path = f"{name}_{suffix}{ext}" if suffix else filename
        fig.savefig(
            path,
            bbox_inches="tight",
            pad_inches=0.05,
            dpi=DPI if ext == ".png" else None,
        )
    if show:
        plt.show()
    else:
        plt.close(fig)
