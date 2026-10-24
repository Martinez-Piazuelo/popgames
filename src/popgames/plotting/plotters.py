from __future__ import annotations

import logging
import typing

import matplotlib
import numpy as np
from ternary.helpers import simplex_iterator

from popgames.plotting._figures import Panels
from popgames.plotting._plot_config import (
    FIGSIZE,
    FONTSIZE,
)
from popgames.utilities.input_validators import check_function_signature

if typing.TYPE_CHECKING:
    from types import SimpleNamespace
    from typing import Any, Callable, Sequence

    from matplotlib.axes import Axes

    from popgames import Simulator

logger = logging.getLogger(__name__)

REFERENCE_STYLE = {"linestyle": "dashed", "linewidth": 0.9}


def make_default_kpi_function(
    simulator: Simulator,
) -> Callable[[SimpleNamespace], np.ndarray] | None:
    """
    Make the default KPI function: the Euclidean distance to the GNE over time, normalized by its initial value.

    Args:
        simulator (Simulator): The simulator whose population game defines the GNE.

    Returns:
        Callable[[SimpleNamespace], np.ndarray] | None: The KPI function, mapping a log (with ``x`` of shape
        ``(n, K)``) to an array of shape ``(K,)``, or None (with a warning) if no GNE was found.
    """
    gne = simulator.population_game.compute_gne()

    if gne is None:
        logger.warning(
            "No GNE was found. Consider setting the kpi_function argument explicitly."
        )
        return None

    def kpi_function(flattened_sim_log: SimpleNamespace) -> np.ndarray:
        """
        Default kpi_function (normalized Euclidean distance to GNE over time).

        Args:
            flattened_sim_log (SimpleNamespace): Simulation log.

        Returns:
            np.ndarray: KPI evaluation results (normalized Euclidean distance to GNE over time).
        """
        _kpi = np.linalg.norm(gne.reshape(-1, 1) - flattened_sim_log.x, ord=2, axis=0)
        return _kpi / max(_kpi[0], 1e-8)

    return kpi_function


def make_reference_values(
    simulator: Simulator,
    reference: dict[str, np.ndarray] | None = None,
    plot_gne: bool = False,
    derived_sizes: dict[str, int] | None = None,
) -> dict[str, np.ndarray]:
    """
    Make the reference values plotted as horizontal lines (e.g., an equilibrium) per variable.

    Args:
        simulator (Simulator): The simulator defining the dimensions and the GNE.
        reference (dict[str, np.ndarray], optional): Reference values per variable, among 'x', 'p' (``n`` entries,
            e.g., of shape ``(n, 1)``), 'q' (``d`` entries), and the derived quantities (if any). A scalar is used
            for all the entries of the variable. Defaults to None.
        plot_gne (bool, optional): Whether to use the GNE (``PopulationGame.compute_gne``) as the reference of 'x'.
            Cannot be combined with ``reference['x']``. Defaults to False.
        derived_sizes (dict[str, int], optional): The number of entries of each derived quantity, by name. Defaults
            to None.

    Returns:
        dict[str, np.ndarray]: The flattened reference values per variable.
    """
    sizes = {
        "x": simulator.population_game.n,
        "p": simulator.population_game.n,
        "q": simulator.payoff_mechanism.d,
    } | (derived_sizes or {})
    values = {}
    for var, value in (reference or {}).items():
        if var not in sizes:
            raise ValueError(
                f"Unknown reference variable {var!r}. Use one of: {list(sizes)}."
            )
        value = np.asarray(value, dtype=float).reshape(-1)
        if value.size == 1:
            value = np.full(sizes[var], value[0])
        if value.size != sizes[var]:
            raise ValueError(
                f"reference[{var!r}] must have {sizes[var]} entries, got {value.size}."
            )
        values[var] = value

    if plot_gne:
        if "x" in values:
            raise ValueError("plot_gne cannot be combined with reference['x'].")
        gne = simulator.population_game.compute_gne()
        if gne is None:
            logger.warning("No GNE was found: its reference lines are not plotted.")
        else:
            values["x"] = gne.reshape(-1)
    return values


def plot_kpi_over_time(
    simulator: Simulator,
    plot_deterministic_approximation: bool = False,
    **kwargs: dict[str, Any],
) -> None:
    """
    Plot a KPI over time (by default, the normalized Euclidean distance to the GNE).

    Args:
        simulator (Simulator): The simulator object holding the data to plot.
        plot_deterministic_approximation (bool, optional): Whether to plot the KPI under the deterministic
            approximation. Defaults to False.
        **kwargs (dict[str, Any]): Keyword arguments to specify plotting options, e.g., ``kpi_function``, or ``ax``
            to draw on an existing ``matplotlib`` axes (the figure is then neither saved, shown, nor closed).
    """
    filename = kwargs.get("filename", None)
    figsize = kwargs.get("figsize", FIGSIZE)
    fontsize = kwargs.get("fontsize", FONTSIZE)
    show = kwargs.get("show", True)
    panels = Panels(kwargs.get("ax", None), 1, filename)

    xlim = kwargs.get("xlim", None)
    ylim = kwargs.get("ylim", None)
    xscale = kwargs.get("xscale", "linear")
    yscale = kwargs.get("yscale", "linear")

    kpi_function = kwargs.get("kpi_function", None)
    if kpi_function is None:
        kpi_function = make_default_kpi_function(simulator)
        if kpi_function is None:
            return None

    sim_log = simulator.log
    if plot_deterministic_approximation:
        t_sim = (0, simulator.t)
        x0 = sim_log.x[:, [0]]
        q0 = sim_log.q[:, [0]]
        out_det = simulator.integrate_edm_pdm(t_sim, x0, q0, t_eval=sim_log.t)
        kpi_det = kpi_function(out_det)

    out = simulator.log
    kpi = kpi_function(out)

    fig, ax = panels.new(figsize)
    ax.plot(sim_log.t, kpi, label="Finite agents", color="black", linewidth=1)

    if plot_deterministic_approximation:
        ax.plot(
            sim_log.t,
            kpi_det,
            label="EDM-PDM",
            linestyle="dotted",
            color="magenta",
            linewidth=1.5,
        )

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    _format_time_axes(ax, r"$\operatorname{KPI}(t)$", fontsize)
    panels.tight_layout(fig)

    if plot_deterministic_approximation:
        ax.legend(fontsize=fontsize)

    panels.finish(fig, filename, None, show)


def plot_univariate_trajectories_joint(
    simulator: Simulator,
    plot_deterministic_approximation: bool = False,
    **kwargs: dict[str, Any],
) -> None:
    """
    This method generates ``2(num_populations)+1``  plots, one for each vector ``x``, ``q``, and ``p``, against time,
    and for ``x`` and ``p`` the plots are generated per population.

    Optionally: It also plots the trajectory of each element under the deterministic approximation
    of the evolutionary dynamics model.

    Args:
        simulator (Simulator): The simulator object holding the data to plot.
        plot_deterministic_approximation (bool, optional): Whether to plot the related trajectories under
            the deterministic approximation. Defaults to False.
        **kwargs (dict[str, Any]): Keyword arguments to specify plotting options, e.g., ``ax`` to draw on existing
            ``matplotlib`` axes: one per plot, in the order ``x`` (per population), ``p`` (per population), and ``q``
            (if ``d > 0``). The figures are then neither saved, shown, nor closed. ``reference`` (values per
            variable, e.g., ``{'x': x_star}``) and ``plot_gne`` (the GNE as the reference of ``x``) add dashed
            horizontal lines, see ``make_reference_values``.
    """

    filename = kwargs.get("filename", None)
    figsize = kwargs.get("figsize", FIGSIZE)
    fontsize = kwargs.get("fontsize", FONTSIZE)
    show = kwargs.get("show", True)

    xlim = kwargs.get("xlim", None)
    ylim = kwargs.get("ylim", None)
    xscale = kwargs.get("xscale", None)
    yscale = kwargs.get("yscale", None)

    P = simulator.population_game.num_populations
    d = simulator.payoff_mechanism.d
    panels = Panels(kwargs.get("ax", None), 2 * P + (d > 0), filename)
    references = make_reference_values(
        simulator, kwargs.get("reference", None), kwargs.get("plot_gne", False)
    )

    sim_log = simulator.log
    if plot_deterministic_approximation:
        t_sim = (0, simulator.t)
        x0 = sim_log.x[:, [0]]
        q0 = sim_log.q[:, [0]]
        out_det = simulator.integrate_edm_pdm(t_sim, x0, q0, t_eval=sim_log.t)

    out = simulator.log

    for var in ["x", "p"]:
        val = getattr(out, var)

        idx = 0
        idx_det = 0
        for k in range(P):
            fig, ax = panels.new(figsize)
            nk = simulator.population_game.num_strategies[k]
            for _ in range(nk):
                (line,) = ax.plot(
                    sim_log.t,
                    val[idx],
                    label=rf"${{{var}}}_{{{idx + 1}}}$"
                    if P == 1
                    else rf"${{{var}}}_{{{idx + 1}}}^{{{k + 1}}}$",
                    linewidth=1,
                )
                if var in references:
                    ax.axhline(
                        references[var][idx], color=line.get_color(), **REFERENCE_STYLE
                    )
                idx += 1

            if plot_deterministic_approximation:
                ax.set_prop_cycle(None)
                for _ in range(nk):
                    val_det = getattr(out_det, var)
                    ax.plot(
                        sim_log.t,
                        val_det[idx_det, :],
                        linestyle="dotted",
                        linewidth=1.5,
                    )
                    idx_det += 1

            _set_axes_options(ax, var, xlim, ylim, xscale, yscale)
            ylabel = (
                rf"$\mathbf{{{var}}}(t)$"
                if P == 1
                else rf"$\mathbf{{{var}}}^{{{k + 1}}}(t)$"
            )
            _format_time_axes(ax, ylabel, fontsize)
            panels.tight_layout(fig)

            ax.legend(fontsize=fontsize, ncol=nk)
            panels.finish(fig, filename, f"{var}_{k + 1}", show)

    if d > 0:
        fig, ax = panels.new(figsize)
        for i in range(d):
            (line,) = ax.plot(
                sim_log.t, out.q[i, :], label=rf"$q_{{{i + 1}}}$", linewidth=1
            )
            if "q" in references:
                ax.axhline(
                    references["q"][i], color=line.get_color(), **REFERENCE_STYLE
                )

        if plot_deterministic_approximation:
            ax.set_prop_cycle(None)
            for i in range(d):
                ax.plot(sim_log.t, out_det.q[i, :], linestyle="dotted", linewidth=1.5)

        _set_axes_options(ax, "q", xlim, ylim, None, None)
        _format_time_axes(ax, r"$\mathbf{q}(t)$", fontsize)
        panels.tight_layout(fig)

        ax.legend(fontsize=fontsize, ncol=d)
        panels.finish(fig, filename, "q", show)


def plot_univariate_trajectories_split(
    simulator: Simulator,
    plot_deterministic_approximation: bool = False,
    **kwargs: dict[str, Any],
) -> None:
    """
    This method generates several plots, one for each element of the vectors ``x``, ``q``, and ``p``,
    against time.

    Optionally: It also plots the trajectory of each element under the deterministic approximation
    of the evolutionary dynamics model.

    Args:
        simulator (Simulator): The simulator object holding the data to plot.
        plot_deterministic_approximation (bool, optional): Whether to plot the related trajectories under
            the deterministic approximation. Defaults to False.
        **kwargs (dict[str, Any]): Keyword arguments to specify plotting options, e.g., ``ax`` to draw on existing
            ``matplotlib`` axes: one per plot, in the order ``x_1, ..., x_n``, ``p_1, ..., p_n``, and
            ``q_1, ..., q_d``. The figures are then neither saved, shown, nor closed. ``reference`` (values per
            variable, e.g., ``{'x': x_star}``) and ``plot_gne`` (the GNE as the reference of ``x``) add dashed
            horizontal lines, see ``make_reference_values``.
    """

    filename = kwargs.get("filename", None)
    figsize = kwargs.get("figsize", FIGSIZE)
    fontsize = kwargs.get("fontsize", FONTSIZE)
    show = kwargs.get("show", True)

    xlim = kwargs.get("xlim", None)
    ylim = kwargs.get("ylim", None)
    xscale = kwargs.get("xscale", None)
    yscale = kwargs.get("yscale", None)

    n = simulator.population_game.n
    d = simulator.payoff_mechanism.d
    panels = Panels(kwargs.get("ax", None), 2 * n + d, filename)
    references = make_reference_values(
        simulator, kwargs.get("reference", None), kwargs.get("plot_gne", False)
    )

    sim_log = simulator.log
    if plot_deterministic_approximation:
        t_sim = (0, simulator.t)
        x0 = sim_log.x[:, [0]]
        q0 = sim_log.q[:, [0]]
        out_det = simulator.integrate_edm_pdm(t_sim, x0, q0, t_eval=sim_log.t)

    out = simulator.log

    for var in ["x", "p"]:
        val = getattr(out, var)

        idx = 0
        for k in range(simulator.population_game.num_populations):
            for _ in range(simulator.population_game.num_strategies[k]):
                fig, ax = panels.new(figsize)
                ax.plot(
                    sim_log.t,
                    val[idx],
                    label="Finite agents",
                    color="black",
                    linewidth=1,
                )

                if plot_deterministic_approximation:
                    val_det = getattr(out_det, var)
                    ax.plot(
                        sim_log.t,
                        val_det[idx, :],
                        label="EDM-PDM",
                        linestyle="dotted",
                        color="magenta",
                        linewidth=1.5,
                    )

                if var in references:
                    ax.axhline(references[var][idx], color="tab:red", **REFERENCE_STYLE)

                _set_axes_options(ax, var, xlim, ylim, xscale, yscale)
                _format_time_axes(
                    ax, rf"${var}_{{{idx + 1}}}^{{{k + 1}}}(t)$", fontsize
                )
                panels.tight_layout(fig)

                if plot_deterministic_approximation:
                    ax.legend(fontsize=fontsize)

                panels.finish(fig, filename, f"{var}_{idx + 1}_{k + 1}", show)

                idx += 1

    for i in range(d):
        fig, ax = panels.new(figsize)
        ax.plot(
            sim_log.t,
            out.q[i, :],
            label="Finite agents",
            color="black",
            linewidth=1,
        )

        if plot_deterministic_approximation:
            ax.plot(
                sim_log.t,
                out_det.q[i, :],
                label="EDM-PDM",
                linestyle="dotted",
                color="magenta",
                linewidth=1.5,
            )

        if "q" in references:
            ax.axhline(references["q"][i], color="tab:red", **REFERENCE_STYLE)

        _set_axes_options(ax, "q", xlim, ylim, None, None)
        _format_time_axes(ax, rf"$q_{{{i + 1}}}(t)$", fontsize)
        panels.tight_layout(fig)

        if plot_deterministic_approximation:
            ax.legend(fontsize=fontsize)

        panels.finish(fig, filename, f"q{i + 1}", show)


def _set_axes_options(
    ax: Axes,
    var: str,
    xlim: dict | None,
    ylim: dict | None,
    xscale: dict | None,
    yscale: dict | None,
) -> None:
    """Apply the per-variable limits and scales (dictionaries keyed by ``'x'``, ``'p'``, or ``'q'``)."""
    if isinstance(xlim, dict) and var in xlim:
        ax.set_xlim(xlim[var])

    if isinstance(ylim, dict) and var in ylim:
        ax.set_ylim(ylim[var])

    if isinstance(xscale, dict) and var in xscale:
        ax.set_xscale(xscale[var])

    if isinstance(yscale, dict) and var in yscale:
        ax.set_yscale(yscale[var])


def _format_time_axes(ax: Axes, ylabel: str, fontsize: int) -> None:
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel(r"$t$", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.grid()


def plot_ternary_trajectories(
    simulator: Simulator,
    plot_deterministic_approximation: bool = False,
    scale: int = 30,
    potential_function: Callable[[np.ndarray], np.ndarray] = None,
    **kwargs: dict[str, Any],
) -> None:
    """
    Make a ternary plot for the simulated scenario.

    This method is only supported for single-population games with ``n=3``, and for multi-population games where
    each population has ``n^k=3`` strategies.

    Args:
        scale (int): Scaling factor for the ternary plot. Defaults to 30.
        fontsize (int): Fontsize for the plot. Defaults to 8.
        figsize (tuple[int, int]): Figure size. Defaults to (4, 3).
        filename (str, optional): Filename to save the figure. Defaults to None.
        plot_edm_trajectory (bool): Whether to plot edm trajectory or not. Defaults to False.
        potential_function (Callable[[np.ndarray], np.ndarray]): Potential function to plot as a heatmap. Defaults to None.
        show (bool): Whether to show the figure(s). Defaults to True.
        ax (Axes | Sequence[Axes], optional): Existing ``matplotlib`` axes to draw on, one per population (a single
            one for single-population games). The figures are then neither saved, shown, nor closed. Defaults to None.
    """

    filename = kwargs.get("filename", None)
    figsize = kwargs.get("figsize", FIGSIZE)
    fontsize = kwargs.get("fontsize", FONTSIZE)
    show = kwargs.get("show", True)
    ax = kwargs.get("ax", None)

    if simulator.population_game.num_populations > 1:
        if potential_function is not None:
            logger.warning(
                "Plotting potential functions over the ternary plot is only supported for single population games."
            )

        make_ternary_plot_multi_population(
            simulator,
            scale=scale,
            fontsize=fontsize,
            figsize=figsize,
            plot_edm_trajectory=plot_deterministic_approximation,
            filename=filename,
            show=show,
            ax=ax,
        )

    else:
        if potential_function is not None:
            check_function_signature(
                arg=potential_function,
                expected_input_shapes=[(simulator.population_game.n, 1)],
                expected_output_shape=(1, 1),
                name="potential_function",
            )

        make_ternary_plot_single_population(
            simulator,
            potential_function=potential_function,
            plot_edm_trajectory=plot_deterministic_approximation,
            scale=scale,
            fontsize=fontsize,
            figsize=figsize,
            filename=filename,
            show=show,
            ax=ax,
        )


def make_ternary_plot_single_population(
    simulator: Simulator,
    potential_function: Callable[[np.ndarray], np.ndarray] = None,
    scale: int = 30,
    fontsize: int = 8,
    figsize: tuple[int, int] = (4, 3),
    plot_edm_trajectory: bool = False,
    filename: str = None,
    show: bool = True,
    ax: Axes = None,
) -> None:
    """
    Plot the trajectory of a single population in a ternary plot.

    This method requires the population to have n=3 strategies.

    Args:
        simulator (Simulator): Simulator instance holding the data for the plot.
        potential_function (Callable[[np.ndarray], np.ndarray], optional): Potential function to plot heatmap. Defaults to None.
        scale (int, optional): Scaling factor for the heatmap. Defaults to 30.
        fontsize (int, optional): Font size for the heatmap. Defaults to 8.
        figsize (tuple[int, int], optional): Figure size. Defaults to (4,3).
        plot_edm_trajectory (bool, optional): Whether to plot edm trajectory. Defaults to False.
        filename (str, optional): Filename to save the figure. Defaults to None.
        show (bool, optional): Whether to show the figure. Defaults to True.
        ax (Axes, optional): Existing ``matplotlib`` axes to draw on. The figure is then neither saved, shown, nor
            closed. Defaults to None.
    """
    panels = Panels(ax, 1, filename)

    # Check number of strategies
    points = []
    points_edm = []
    n = simulator.population_game.n
    if n != 3:
        logger.error(
            f"ternary_plot() is only supported for games with 3 strategies per population. Population has {n} strategies."
        )
        return None

    # Compute edm trajectory (if enabled)
    sim_log = simulator.log
    if plot_edm_trajectory:
        t_sim = (0, simulator.t)
        x0 = sim_log.x[:, [0]]
        q0 = sim_log.q[:, [0]]
        out = simulator.integrate_edm_pdm(t_sim, x0, q0, t_eval=sim_log.t)
        x_edm = out.x

    # Slice trajectories
    for t, point in enumerate(sim_log.x.T):
        point_k = point.reshape(
            n,
        )
        point_k = scale * point_k / point_k.sum()  # Scale points
        points.append((point_k[2], point_k[0], point_k[1]))  # Permute points

        if plot_edm_trajectory:
            point_k_edm = x_edm[:, t].reshape(
                n,
            )
            point_k_edm = scale * point_k_edm / point_k_edm.sum()
            points_edm.append((point_k_edm[2], point_k_edm[0], point_k_edm[1]))

    # Compute and slice GNE
    gne = simulator.population_game.compute_gne()
    if gne is not None:
        gne = gne.reshape(
            n,
        )
        gne = scale * gne / gne.sum()  # Scale GNE
        gne = [(gne[2], gne[0], gne[1])]  # Permute

    # Initialize plot
    figure, tax = panels.new_ternary(figsize, scale=scale)

    # Plot heatmap (if a potential function is provided)
    if potential_function is not None:
        points_potential: dict[tuple[int, int], float] = {}
        for i, j, k in simplex_iterator(scale):
            x = np.array([j, k, i]).reshape(
                n,
            )  # orientation permutation
            x = simulator.population_game.masses[0] * x / x.sum()

            val = potential_function(x.reshape(n, 1))

            # Force to Python float (handles (1,1), (1,), scalar numpy, etc.)
            val = float(np.asarray(val).squeeze())

            points_potential[(i, j)] = val

        vmin = float(min(points_potential.values()))
        vmax = float(max(points_potential.values()))

        tax.heatmap(
            points_potential,
            style="hexagonal",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            colorbar=False,
        )

    # Plot feasible region (if possible)
    num_constraints = simulator.population_game.d_eq + simulator.population_game.d_ineq
    vertices = simulator.population_game.compute_polyhedron_vertices()
    vertices_scaled = []
    if len(vertices) == 2:
        for vertex in vertices:
            point = scale * vertex / vertex.sum()
            vertices_scaled.append((point[2], point[0], point[1]))
        tax.plot(
            vertices_scaled,
            linewidth=1,
            linestyle="dashed",
            color="tab:red",
            label=r"$\mathcal{X}$" if num_constraints > 0 else None,
        )
    else:
        # TODO: IMPLEMENT THIS
        pass

    # Plot boundary
    tax.boundary(linewidth=1.0)

    # Plot trajectory
    tax.plot(points, linewidth=1, color="black", label=r"$\mathbf{x}(t)$")

    # Plot EDM trajectory (if enabled)
    if plot_edm_trajectory:
        tax.plot(
            points_edm,
            linewidth=1.5,
            linestyle="dotted",
            color="magenta",
            label=r"$\mathbf{x}(t)$ (EDM)",
        )

    # Plot GNE (if available)
    if gne is not None:
        tax.plot(
            gne,
            marker=r"$\star$",
            markersize=7,
            color="tab:red",
            linestyle="",
            linewidth=0,
            label=r"$\operatorname{GNE}$"
            if num_constraints > 0
            else r"$\operatorname{NE}$",
        )

    # Plot formating
    custom_legend = []
    if gne is not None:
        custom_legend.append(
            matplotlib.lines.Line2D(
                [],
                [],
                marker=r"$\star$",
                markersize=7,
                color="tab:red",
                linestyle="",
                linewidth=0,
                label=r"$\operatorname{GNE}$"
                if num_constraints > 0
                else r"$\operatorname{NE}$",
            )
        )

    if len(vertices) >= 2 and num_constraints > 0:
        custom_legend.append(
            matplotlib.lines.Line2D(
                [],
                [],
                linewidth=1,
                linestyle="dashed",
                color="tab:red",
                label=r"$\mathcal{X}$",
            ),
        )

    custom_legend.append(
        matplotlib.lines.Line2D(
            [], [], linewidth=1, color="black", label=r"$\mathbf{x}(t)$"
        )
    )

    if plot_edm_trajectory:
        custom_legend.append(
            matplotlib.lines.Line2D(
                [],
                [],
                linewidth=1.5,
                linestyle="dotted",
                color="magenta",
                label=r"$\mathbf{x}(t)$ (EDM)",
            )
        )

    tax.top_corner_label(r"$e_1$", fontsize=fontsize)
    tax.left_corner_label(r"$e_2$", fontsize=fontsize)
    tax.right_corner_label(r"$e_3$", fontsize=fontsize)
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")
    tax.legend(handles=custom_legend, loc=1, fontsize=fontsize)
    panels.finish(figure, filename, None, show)


def make_ternary_plot_multi_population(
    simulator: Simulator,
    scale=30,
    fontsize=8,
    figsize=(4, 3),
    plot_edm_trajectory=False,
    filename=None,
    show: bool = True,
    ax: Axes | Sequence[Axes] = None,
) -> None:
    """
    Plot the trajectory of a multiple population in multiple ternary plots.

    This method requires every population to have ``n^k=3`` strategies.

    Args:
        simulator (Simulator): Simulator instance holding the data for the plot.
        scale (int, optional): Scaling factor for the heatmap. Defaults to 30.
        fontsize (int, optional): Font size for the heatmap. Defaults to 8.
        figsize (tuple[int, int], optional): Figure size. Defaults to (4,3).
        plot_edm_trajectory (bool, optional): Whether to plot edm trajectory. Defaults to False.
        filename (str, optional): Filename to save the figure. Defaults to None.
        show (bool, optional): Whether to show the figure. Defaults to True.
        ax (Sequence[Axes], optional): Existing ``matplotlib`` axes to draw on, one per population. The figures are
            then neither saved, shown, nor closed. Defaults to None.
    """
    panels = Panels(ax, simulator.population_game.num_populations, filename)

    # Slice trajectories
    points = dict()
    points_edm = dict()
    for k in range(simulator.population_game.num_populations):
        points[k] = []
        points_edm[k] = []
        nk = simulator.population_game.num_strategies[k]
        if nk != 3:
            logger.error(
                f"ternary_plot() is only supported for games with 3 strategies per population. Population {k} has {nk} strategies."
            )
            return None

    # Compute edm trajectory (if enabled)
    sim_log = simulator.log
    if plot_edm_trajectory:
        t_sim = (0, simulator.t)
        x0 = sim_log.x[:, [0]]
        q0 = sim_log.q[:, [0]]
        out = simulator.integrate_edm_pdm(t_sim, x0, q0, t_eval=sim_log.t)
        x_edm = out.x

    # Slice trajectories
    for t, point in enumerate(sim_log.x.T):
        pos = 0
        for k in range(simulator.population_game.num_populations):
            nk = simulator.population_game.num_strategies[k]
            point_k = point[pos : pos + nk].reshape(
                nk,
            )  # Slice trajectory
            point_k = scale * point_k / point_k.sum()  # Scale points
            points[k].append((point_k[2], point_k[0], point_k[1]))  # Permute points

            if plot_edm_trajectory:
                point_k_edm = x_edm[pos : pos + nk, t].reshape(
                    nk,
                )
                point_k_edm = scale * point_k_edm / point_k_edm.sum()
                points_edm[k].append((point_k_edm[2], point_k_edm[0], point_k_edm[1]))

            pos += nk

    # Compute and slice GNE
    gne = simulator.population_game.compute_gne()
    print(f"Computed GNE = {gne.reshape(-1).round(3)}")
    if gne is not None:
        gnes = dict()
        pos = 0
        for k in range(simulator.population_game.num_populations):
            nk = simulator.population_game.num_strategies[k]
            gne_k = gne[pos : pos + nk].reshape(
                nk,
            )  # Slice GNE
            gne_k = scale * gne_k / gne_k.sum()  # Scale GNE
            gnes[k] = [(gne_k[2], gne_k[0], gne_k[1])]  # Permute
            pos += nk

    num_constraints = simulator.population_game.d_eq + simulator.population_game.d_ineq

    # Make ternary plots (one for each population)
    for k in range(simulator.population_game.num_populations):
        # Initialize plot
        figure, tax = panels.new_ternary(figsize, scale=scale)

        # Plot boundary
        tax.boundary(linewidth=1.0)

        # Plot trajectory
        tax.plot(
            points[k],
            linewidth=1,
            color="black",
            label=rf"$\mathbf{{x}}^{{{k + 1}}}(t)$",
        )

        # Plot EDM trajectory (if enabled)
        if plot_edm_trajectory:
            tax.plot(
                points_edm[k],
                linewidth=1.5,
                linestyle="dotted",
                color="magenta",
                label=rf"$\mathbf{{x}}^{{{k + 1}}}(t)$ (EDM)",
            )

        # Plot GNE (if available)
        if gne is not None:
            tax.plot(
                gnes[k],
                marker=r"$\star$",
                markersize=7,
                color="tab:red",
                linestyle="",
                linewidth=0,
                label=r"$\operatorname{GNE}$"
                if num_constraints > 0
                else r"$\operatorname{NE}$",
            )

        # Plot formating
        custom_legend = []
        if gne is not None:
            custom_legend.append(
                matplotlib.lines.Line2D(
                    [],
                    [],
                    marker=r"$\star$",
                    markersize=7,
                    color="tab:red",
                    linestyle="",
                    linewidth=0,
                    label=r"$\operatorname{GNE}$"
                    if num_constraints > 0
                    else r"$\operatorname{NE}$",
                )
            )

        custom_legend.append(
            matplotlib.lines.Line2D(
                [],
                [],
                linewidth=1,
                color="black",
                label=rf"$\mathbf{{x}}^{{{k + 1}}}(t)$",
            )
        )

        if plot_edm_trajectory:
            custom_legend.append(
                matplotlib.lines.Line2D(
                    [],
                    [],
                    linewidth=1.5,
                    linestyle="dotted",
                    color="magenta",
                    label=rf"$\mathbf{{x}}^{{{k + 1}}}(t)$ (EDM)",
                )
            )

        tax.top_corner_label(r"$e_1$", fontsize=fontsize)
        tax.left_corner_label(r"$e_2$", fontsize=fontsize)
        tax.right_corner_label(r"$e_3$", fontsize=fontsize)
        tax.clear_matplotlib_ticks()
        tax.get_axes().axis("off")
        tax.legend(handles=custom_legend, loc=1, fontsize=fontsize)
        panels.finish(figure, filename, f"pop_{k + 1}", show)
