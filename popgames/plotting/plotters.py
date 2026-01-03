from __future__ import annotations
import typing

import logging
logger = logging.getLogger(__name__)

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import ternary
from ternary.helpers import simplex_iterator

from popgames.plotting._plot_config import (
    DPI,
    FIGSIZE,
    FONTSIZE,
)

from popgames.utilities.input_validators import check_function_signature

if typing.TYPE_CHECKING:
    from typing import Any, Callable
    from popgames import Simulator
    from types import SimpleNamespace


def plot_kpi_over_time(
        simulator : Simulator,
        plot_deterministic_approximation : bool = False,
        **kwargs : dict[str, Any]
) -> None:
    filename = kwargs.get('filename', None)
    figsize = kwargs.get('figsize', FIGSIZE)
    fontsize = kwargs.get('fontsize', FONTSIZE)
    show = kwargs.get('show', True)

    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    xscale = kwargs.get('xscale', 'linear')
    yscale = kwargs.get('yscale', 'linear')

    kpi_function = kwargs.get('kpi_function', None)
    if kpi_function is None:

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

    if plot_deterministic_approximation:
        t_sim = (0, simulator.t)
        x0 = simulator.log.x[0]
        q0 = simulator.log.q[0]
        out_det = simulator.integrate_edm_pdm(t_sim, x0, q0, t_eval=simulator.log.t)
        kpi_det = kpi_function(out_det)

    out = simulator._get_flattened_log()  # TODO: enable a non-protected method in Simulator for this
    kpi = kpi_function(out)

    plt.figure(figsize=figsize)
    plt.plot(
        simulator.log.t, kpi,
        label='Finite agents',
        color='black',
        linewidth=1
    )

    if plot_deterministic_approximation:
        plt.plot(
            simulator.log.t, kpi_det,
            label='EDM-PDM',
            linestyle='dotted',
            color='magenta',
            linewidth=1.5
        )

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel(r'$t$', fontsize=fontsize)
    plt.ylabel(r'$\operatorname{KPI}(t)$', fontsize=fontsize)
    plt.grid()
    plt.tight_layout()

    if plot_deterministic_approximation:
        plt.legend(fontsize=fontsize)

    if filename is not None:
        name, ext = filename.split('.')
        dpi = DPI if ext == '.png' else None
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.05, dpi=dpi)

    if show:
        plt.show()

    else:
        plt.close()

def plot_univariate_trajectories_joint(
        simulator : Simulator,
        plot_deterministic_approximation : bool = False,
        **kwargs : dict[str, Any]
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
        **kwargs (dict[str, Any]): Keyword arguments to specify plotting options.
    """

    filename = kwargs.get('filename', None)
    figsize = kwargs.get('figsize', FIGSIZE)
    fontsize = kwargs.get('fontsize', FONTSIZE)
    show = kwargs.get('show', True)

    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    xscale = kwargs.get('xscale', None)
    yscale = kwargs.get('yscale', None)

    if plot_deterministic_approximation:
        t_sim = (0, simulator.t)
        x0 = simulator.log.x[0]
        q0 = simulator.log.q[0]
        out_det = simulator.integrate_edm_pdm(t_sim, x0, q0, t_eval=simulator.log.t)

    out = simulator._get_flattened_log()

    for var in ['x', 'p']:
        val = getattr(out, var)

        P = simulator.population_game.num_populations
        idx = 0
        idx_det = 0
        for k in range(P):
            plt.figure(figsize=figsize)
            nk = simulator.population_game.num_strategies[k]
            for _ in range(nk):
                plt.plot(
                    simulator.log.t, val[idx],
                    label=fr'${{{var}}}_{{{idx+1}}}$' if P==1 else fr'${{{var}}}_{{{idx+1}}}^{{{k+1}}}$',
                    linewidth=1
                )
                idx += 1

            if plot_deterministic_approximation:
                plt.gca().set_prop_cycle(None)
                for _ in range(nk):
                    val_det = getattr(out_det, var)
                    plt.plot(
                        simulator.log.t, val_det[idx_det,:],
                        linestyle='dotted',
                        linewidth=1.5
                    )
                    idx_det += 1

            if isinstance(xlim, dict) and var in xlim:
                plt.xlim(xlim[var])

            if isinstance(ylim, dict) and var in ylim:
                plt.ylim(ylim[var])

            if isinstance(xscale, dict) and var in xscale:
                plt.xscale(xscale[var])

            if isinstance(yscale, dict) and var in yscale:
                plt.yscale(yscale[var])

            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xlabel(r'$t$', fontsize=fontsize)
            ylabel = fr'$\mathbf{{{var}}}(t)$' if P == 1 else fr'$\mathbf{{{var}}}^{{{k+1}}}(t)$'
            plt.ylabel(ylabel, fontsize=fontsize)
            plt.grid()
            plt.tight_layout()

            plt.legend(fontsize=fontsize, ncol=nk)

            if filename is not None:
                name, ext = filename.split('.')
                filename_k = '_'.join([name, f'{var}_{k+1}'])
                filename_k = '.'.join([filename_k, ext])
                dpi = DPI if ext == '.png' else None
                plt.savefig(filename_k, bbox_inches='tight', pad_inches=0.05, dpi=dpi)

            if show:
                plt.show()

            else:
                plt.close()


    d = simulator.payoff_mechanism.d
    if d > 0:
        plt.figure(figsize=figsize)
        for i in range(d):
            plt.plot(
                simulator.log.t, out.q[i, :],
                label=fr'$q_{{{i+1}}}$',
                linewidth=1
            )

        if plot_deterministic_approximation:
            plt.gca().set_prop_cycle(None)
            for i in range(d):
                plt.plot(
                    simulator.log.t, out_det.q[i, :],
                    linestyle='dotted',
                    linewidth=1.5
                )

        if isinstance(xlim, dict) and 'q' in xlim:
            plt.xlim(xlim['q'])

        if isinstance(ylim, dict) and 'q' in ylim:
            plt.ylim(ylim['q'])

        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xlabel(r'$t$', fontsize=fontsize)
        plt.ylabel(fr'$\mathbf{{q}}(t)$', fontsize=fontsize)
        plt.grid()
        plt.tight_layout()

        plt.legend(fontsize=fontsize, ncol=d)

        if filename is not None:
            name, ext = filename.split('.')
            filename_q = '_'.join([name, f'q'])
            filename_q = '.'.join([filename_q, ext])
            dpi = DPI if ext == '.png' else None
            plt.savefig(filename_q, bbox_inches='tight', pad_inches=0.05, dpi=dpi)

        if show:
            plt.show()

        else:
            plt.close()


def plot_univariate_trajectories_split(
        simulator : Simulator,
        plot_deterministic_approximation : bool = False,
        **kwargs : dict[str, Any]
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
        **kwargs (dict[str, Any]): Keyword arguments to specify plotting options.
    """

    filename = kwargs.get('filename', None)
    figsize = kwargs.get('figsize', FIGSIZE)
    fontsize = kwargs.get('fontsize', FONTSIZE)
    show = kwargs.get('show', True)

    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    xscale = kwargs.get('xscale', None)
    yscale = kwargs.get('yscale', None)

    if plot_deterministic_approximation:
        t_sim = (0, simulator.t)
        x0 = simulator.log.x[0]
        q0 = simulator.log.q[0]
        out_det = simulator.integrate_edm_pdm(t_sim, x0, q0, t_eval=simulator.log.t)

    out = simulator._get_flattened_log()

    for var in ['x', 'p']:
        val = getattr(out, var)

        idx = 0
        for k in range(simulator.population_game.num_populations):
            for _ in range(simulator.population_game.num_strategies[k]):
                plt.figure(figsize=figsize)
                plt.plot(
                    simulator.log.t, val[idx],
                    label='Finite agents',
                    color='black',
                    linewidth=1
                )

                if plot_deterministic_approximation:
                    val_det = getattr(out_det, var)
                    plt.plot(
                        simulator.log.t, val_det[idx,:],
                        label='EDM-PDM',
                        linestyle='dotted',
                        color='magenta',
                        linewidth=1.5
                    )

                if isinstance(xlim, dict) and var in xlim:
                    plt.xlim(xlim[var])

                if isinstance(ylim, dict) and var in ylim:
                    plt.ylim(ylim[var])

                if isinstance(xscale, dict) and var in xscale:
                    plt.xscale(xscale[var])

                if isinstance(yscale, dict) and var in yscale:
                    plt.yscale(yscale[var])

                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                plt.xlabel(r'$t$', fontsize=fontsize)
                plt.ylabel(fr'${var}_{{{idx+1}}}^{{{k+1}}}(t)$', fontsize=fontsize)
                plt.grid()
                plt.tight_layout()

                if plot_deterministic_approximation:
                    plt.legend(fontsize=fontsize)

                if filename is not None:
                    name, ext = filename.split('.')
                    filename_ik = '_'.join([name, f'{var}_{idx+1}_{k+1}'])
                    filename_ik = '.'.join([filename_ik, ext])
                    dpi = DPI if ext == '.png' else None
                    plt.savefig(filename_ik, bbox_inches='tight', pad_inches=0.05, dpi=dpi)

                if show:
                    plt.show()

                else:
                    plt.close()

                idx += 1

    if simulator.payoff_mechanism.d > 0:

        for i in range(simulator.payoff_mechanism.d):
            plt.figure(figsize=figsize)
            plt.plot(
                simulator.log.t, out.q[i, :],
                label='Finite agents',
                color='black',
                linewidth=1
            )

            if plot_deterministic_approximation:
                plt.plot(
                    simulator.log.t, out_det.q[i, :],
                    label='EDM-PDM',
                    linestyle='dotted',
                    color='magenta',
                    linewidth=1.5
                )

            if isinstance(xlim, dict) and 'q' in xlim:
                plt.xlim(xlim['q'])

            if isinstance(ylim, dict) and 'q' in ylim:
                plt.ylim(ylim['q'])

            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xlabel(r'$t$', fontsize=fontsize)
            plt.ylabel(fr'$q_{{{i + 1}}}(t)$', fontsize=fontsize)
            plt.grid()
            plt.tight_layout()

            if plot_deterministic_approximation:
                plt.legend(fontsize=fontsize)

            if filename is not None:
                name, ext = filename.split('.')
                filename_qi = '_'.join([name, f'q{i + 1}'])
                filename_qi = '.'.join([filename_qi, ext])
                dpi = DPI if ext == '.png' else None
                plt.savefig(filename_qi, bbox_inches='tight', pad_inches=0.05, dpi=dpi)

            if show:
                plt.show()

            else:
                plt.close()


def plot_ternary_trajectories(
        simulator: Simulator,
        plot_deterministic_approximation: bool = False,
        scale: int = 30,
        potential_function: Callable[[np.ndarray], np.ndarray] = None,
        **kwargs: dict[str, Any]
) -> None:
    """
    Make a ternary plot for the simulated scenario.

    This method is only supported for single-population games with \(n=3\), and for multi-population games where
    each population has \(n^k=3\) strategies.

    Args:
        scale (int): Scaling factor for the ternary plot. Defaults to 30.
        fontsize (int): Fontsize for the plot. Defaults to 8.
        figsize (tuple[int, int]): Figure size. Defaults to (4, 3).
        filename (str, optional): Filename to save the figure. Defaults to None.
        plot_edm_trajectory (bool): Whether to plot edm trajectory or not. Defaults to False.
        potential_function (Callable[[np.ndarray], np.ndarray]): Potential function to plot as a heatmap. Defaults to None.
    """

    filename = kwargs.get('filename', None)
    figsize = kwargs.get('figsize', FIGSIZE)
    fontsize = kwargs.get('fontsize', FONTSIZE)

    if simulator.population_game.num_populations > 1:
        if potential_function is not None:
            logger.warning(
                'Plotting potential functions over the ternary plot is only supported for single population games.'
            )

        make_ternary_plot_multi_population(
            simulator,
            scale=scale,
            fontsize=fontsize,
            figsize=figsize,
            plot_edm_trajectory=plot_deterministic_approximation,
            filename=filename
        )

    else:
        if potential_function is not None:
            check_function_signature(
                arg=potential_function,
                expected_input_shapes=[(simulator.population_game.n, 1)],
                expected_output_shape=(1, 1),
                name='potential_function'
            )

        make_ternary_plot_single_population(
            simulator,
            potential_function=potential_function,
            plot_edm_trajectory=plot_deterministic_approximation,
            scale=scale,
            fontsize=fontsize,
            figsize=figsize,
            filename=filename
        )


def make_ternary_plot_single_population(
        simulator: Simulator,
        potential_function: Callable[[np.ndarray], np.ndarray] = None,
        scale: int = 30,
        fontsize: int = 8,
        figsize: tuple[int, int] = (4, 3),
        plot_edm_trajectory: bool = False,
        filename: str = None
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
    """

    # Check number of strategies
    points = []
    points_edm = []
    n = simulator.population_game.n
    if n != 3:
        logger.error(
            f'ternary_plot() is only supported for games with 3 strategies per population. Population has {n} strategies.')
        return None

    # Compute edm trajectory (if enabled)
    if plot_edm_trajectory:
        t_sim = (0, simulator.t)
        x0 = simulator.log.x[0]
        q0 = simulator.log.q[0]
        out = simulator.integrate_edm_pdm(t_sim, x0, q0, t_eval=simulator.log.t)
        x_edm = out.x

    # Slice trajectories
    for t, point in enumerate(simulator.log.x):
        point_k = point.reshape(n, )
        point_k = scale * point_k / point_k.sum()  # Scale points
        points.append((point_k[2], point_k[0], point_k[1]))  # Permute points

        if plot_edm_trajectory:
            point_k_edm = x_edm[:, t].reshape(n, )
            point_k_edm = scale * point_k_edm / point_k_edm.sum()
            points_edm.append((point_k_edm[2], point_k_edm[0], point_k_edm[1]))

    # Compute and slice GNE
    gne = simulator.population_game.compute_gne()
    if gne is not None:
        gne = gne.reshape(n, )
        gne = scale * gne / gne.sum()  # Scale GNE
        gne = [(gne[2], gne[0], gne[1])]  # Permute

    # Initialize plot
    figure, tax = ternary.figure(scale=scale)
    figure.set_size_inches(figsize[0], figsize[1])

    # Plot heatmap (if a potential function is provided)
    if potential_function is not None:
        points_potential = {}
        for (i, j, k) in simplex_iterator(scale):
            x = np.array([j, k, i]).reshape(n, )  # Permutation for desired orientation: (e1 top, e2 left, e3 right)
            x = simulator.population_game.masses[0] * x / x.sum()
            points_potential[(i, j)] = potential_function(x.reshape(n, 1)).reshape(-1)
        vmin, vmax = min(points_potential.values()), max(points_potential.values())
        tax.heatmap(points_potential, style='hexagonal', vmin=vmin, vmax=vmax, cmap='viridis', colorbar=False)

    # Plot feasible region (if possible)
    num_constraints = simulator.population_game.d_eq + simulator.population_game.d_ineq
    vertices = simulator.population_game.compute_polyhedron_vertices()
    vertices_scaled = []
    if len(vertices) == 2:
        for vertex in vertices:
            point = scale * vertex / vertex.sum()
            vertices_scaled.append((point[2], point[0], point[1]))
        tax.plot(
            vertices_scaled, linewidth=1, linestyle='dashed', color="tab:red",
            label=r'$\mathcal{X}$' if num_constraints > 0 else None)
    else:
        # TODO: IMPLEMENT THIS
        pass

    # Plot boundary
    tax.boundary(linewidth=1.0)

    # Plot trajectory
    tax.plot(points, linewidth=1, color='black', label=r'$\mathbf{x}(t)$')

    # Plot EDM trajectory (if enabled)
    if plot_edm_trajectory:
        tax.plot(points_edm, linewidth=1.5, linestyle='dotted', color='magenta', label=r'$\mathbf{x}(t)$ (EDM)')

    # Plot GNE (if available)
    if gne is not None:
        tax.plot(gne, marker=r'$\star$', markersize=7, color='tab:red', linestyle='', linewidth=0,
                 label=r'$\operatorname{GNE}$' if num_constraints > 0 else r'$\operatorname{NE}$')

    # Plot formating
    custom_legend = []
    if gne is not None:
        custom_legend.append(
            matplotlib.lines.Line2D(
                [], [],
                marker=r'$\star$', markersize=7, color='tab:red', linestyle='', linewidth=0,
                label=r'$\operatorname{GNE}$' if num_constraints > 0 else r'$\operatorname{NE}$')
        )

    if len(vertices) >= 2 and num_constraints > 0:
        custom_legend.append(
            matplotlib.lines.Line2D(
                [], [], linewidth=1, linestyle='dashed', color="tab:red",
                label=r'$\mathcal{X}$'),
        )

    custom_legend.append(
        matplotlib.lines.Line2D([], [], linewidth=1, color='black', label=r'$\mathbf{x}(t)$')
    )

    if plot_edm_trajectory:
        custom_legend.append(
            matplotlib.lines.Line2D([], [], linewidth=1.5, linestyle='dotted', color='magenta',
                                    label=r'$\mathbf{x}(t)$ (EDM)')
        )

    tax.top_corner_label(r'$e_1$', fontsize=fontsize)
    tax.left_corner_label(r'$e_2$', fontsize=fontsize)
    tax.right_corner_label(r'$e_3$', fontsize=fontsize)
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
    tax.legend(handles=custom_legend, loc=1, fontsize=fontsize)
    if filename is not None:
        figure.savefig(filename, format="pdf", bbox_inches="tight")
    tax.show()


def make_ternary_plot_multi_population(
        simulator: Simulator,
        scale=30,
        fontsize=8,
        figsize=(4, 3),
        plot_edm_trajectory=False,
        filename=None
) -> None:
    """
    Plot the trajectory of a multiple population in multiple ternary plots.

    This method requires every population to have \(n^k=3\) strategies.

    Args:
        simulator (Simulator): Simulator instance holding the data for the plot.
        scale (int, optional): Scaling factor for the heatmap. Defaults to 30.
        fontsize (int, optional): Font size for the heatmap. Defaults to 8.
        figsize (tuple[int, int], optional): Figure size. Defaults to (4,3).
        plot_edm_trajectory (bool, optional): Whether to plot edm trajectory. Defaults to False.
        filename (str, optional): Filename to save the figure. Defaults to None.
    """

    # Slice trajectories
    points = dict()
    points_edm = dict()
    for k in range(simulator.population_game.num_populations):
        points[k] = []
        points_edm[k] = []
        nk = simulator.population_game.num_strategies[k]
        if nk != 3:
            logger.error(
                f'ternary_plot() is only supported for games with 3 strategies per population. Population {k} has {nk} strategies.')
            return None

    # Compute edm trajectory (if enabled)
    if plot_edm_trajectory:
        t_sim = (0, simulator.t)
        x0 = simulator.log.x[0]
        q0 = simulator.log.q[0]
        out = simulator.integrate_edm_pdm(t_sim, x0, q0, t_eval=simulator.log.t)
        x_edm = out.x

    # Slice trajectories
    for t, point in enumerate(simulator.log.x):
        pos = 0
        for k in range(simulator.population_game.num_populations):
            nk = simulator.population_game.num_strategies[k]
            point_k = point[pos:pos + nk].reshape(nk, )  # Slice trajectory
            point_k = scale * point_k / point_k.sum()  # Scale points
            points[k].append((point_k[2], point_k[0], point_k[1]))  # Permute points

            if plot_edm_trajectory:
                point_k_edm = x_edm[pos:pos + nk, t].reshape(nk, )
                point_k_edm = scale * point_k_edm / point_k_edm.sum()
                points_edm[k].append((point_k_edm[2], point_k_edm[0], point_k_edm[1]))

            pos += nk

    # Compute and slice GNE
    gne = simulator.population_game.compute_gne()
    print(f'Computed GNE = {gne.reshape(-1).round(3)}')
    if gne is not None:
        gnes = dict()
        pos = 0
        for k in range(simulator.population_game.num_populations):
            nk = simulator.population_game.num_strategies[k]
            gne_k = gne[pos:pos + nk].reshape(nk, )  # Slice GNE
            gne_k = scale * gne_k / gne_k.sum()  # Scale GNE
            gnes[k] = [(gne_k[2], gne_k[0], gne_k[1])]  # Permute
            pos += nk

    num_constraints = simulator.population_game.d_eq + simulator.population_game.d_ineq

    # Make ternary plots (one for each population)
    for k in range(simulator.population_game.num_populations):

        # Initialize plot
        figure, tax = ternary.figure(scale=scale)
        figure.set_size_inches(figsize[0], figsize[1])

        # Plot boundary
        tax.boundary(linewidth=1.0)

        # Plot trajectory
        tax.plot(points[k], linewidth=1, color='black', label=rf'$\mathbf{{x}}^{{{k + 1}}}(t)$')

        # Plot EDM trajectory (if enabled)
        if plot_edm_trajectory:
            tax.plot(points_edm[k], linewidth=1.5, linestyle='dotted', color='magenta',
                     label=rf'$\mathbf{{x}}^{{{k + 1}}}(t)$ (EDM)')

        # Plot GNE (if available)
        if gne is not None:
            tax.plot(
                gnes[k], marker=r'$\star$', markersize=7, color='tab:red', linestyle='', linewidth=0,
                label=r'$\operatorname{GNE}$' if num_constraints > 0 else r'$\operatorname{NE}$'
            )

        # Plot formating
        custom_legend = []
        if gne is not None:
            custom_legend.append(
                matplotlib.lines.Line2D(
                    [], [], marker=r'$\star$', markersize=7, color='tab:red', linestyle='',
                    linewidth=0, label=r'$\operatorname{GNE}$' if num_constraints > 0 else r'$\operatorname{NE}$'
                )
            )

        custom_legend.append(
            matplotlib.lines.Line2D([], [], linewidth=1, color='black', label=rf'$\mathbf{{x}}^{{{k + 1}}}(t)$')
        )

        if plot_edm_trajectory:
            custom_legend.append(
                matplotlib.lines.Line2D([], [], linewidth=1.5, linestyle='dotted', color='magenta',
                                        label=rf'$\mathbf{{x}}^{{{k + 1}}}(t)$ (EDM)')
            )

        tax.top_corner_label(r'$e_1$', fontsize=fontsize)
        tax.left_corner_label(r'$e_2$', fontsize=fontsize)
        tax.right_corner_label(r'$e_3$', fontsize=fontsize)
        tax.clear_matplotlib_ticks()
        tax.get_axes().axis('off')
        tax.legend(handles=custom_legend, loc=1, fontsize=fontsize)
        if filename is not None:
            name, ext = filename.split('.')
            filename_k = '_'.join([name, f'pop_{k+1}'])
            filename_k = '.'.join([filename_k, ext])
            figure.savefig(filename_k, format="pdf", bbox_inches="tight")
        tax.show()