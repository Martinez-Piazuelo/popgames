from __future__ import annotations
import typing

import logging
logger = logging.getLogger(__name__)

import numpy as np
import ternary
from ternary.helpers import simplex_iterator
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 200
matplotlib.rcParams['figure.figsize'] = (5, 5)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.size': 10})

if typing.TYPE_CHECKING:
    from typing import Callable, Optional
    from popgames import Simulator

def make_ternary_plot_single_population(
        simulator : Simulator,
        potential_function : Callable[[np.ndarray], np.ndarray] = None,
        scale : int =30,
        fontsize : int = 8,
        figsize : tuple[int, int] = (4,3),
        plot_edm_trajectory : bool = False,
        filename : str = None
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
        logger.error(f'ternary_plot() is only supported for games with 3 strategies per population. Population has {n} strategies.')
        return None

    # Compute edm trajectory (if enabled)
    if plot_edm_trajectory:
        t_sim = (0, simulator.t)
        x0 = simulator.log.x[0]
        q0 = simulator.log.q[0]
        out = simulator.integrate_edm_pdm(t_sim, x0, q0, t_eval = simulator.log.t)
        x_edm = out.x

    # Slice trajectories
    for t, point in enumerate(simulator.log.x):
        point_k = point.reshape(n,)
        point_k = scale*point_k/point_k.sum()                     # Scale points
        points.append((point_k[2], point_k[0], point_k[1]))       # Permute points

        if plot_edm_trajectory:
            point_k_edm = x_edm[:, t].reshape(n,)
            point_k_edm = scale*point_k_edm/point_k_edm.sum()
            points_edm.append((point_k_edm[2], point_k_edm[0], point_k_edm[1]))

    # Compute and slice GNE
    gne = simulator.population_game.compute_gne()
    if gne is not None:
        gne = gne.reshape(n, )
        gne = scale*gne/gne.sum()                   # Scale GNE
        gne = [(gne[2], gne[0], gne[1])]            # Permute

    # Initialize plot
    figure, tax = ternary.figure(scale=scale)
    figure.set_size_inches(figsize[0], figsize[1])

    # Plot heatmap (if a potential function is provided)
    if potential_function is not None:
        points_potential = {}
        for (i,j,k) in simplex_iterator(scale):
            x = np.array([j, k, i]).reshape(n,)  # Permutation for desired orientation: (e1 top, e2 left, e3 right)
            x = simulator.population_game.masses[0]*x/x.sum()
            points_potential[(i,j)] = potential_function(x)
        vmin, vmax = min(points_potential.values()), max(points_potential.values())
        tax.heatmap(points_potential, style='hexagonal', vmin=vmin, vmax=vmax, cmap='viridis', colorbar=False)

    # Plot feasible region (if possible)
    vertices = simulator.population_game.compute_polyhedron_vertices()        
    vertices_scaled = []
    if len(vertices) == 2:
        for vertex in vertices:
            point = scale*vertex/vertex.sum()
            vertices_scaled.append((point[2], point[0], point[1]))
        tax.plot(vertices_scaled, linewidth=1, linestyle='dashed', color="tab:red", label=r'$\mathcal{X}$')
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
        tax.plot(gne, marker=r'$\star$', markersize=7, color='tab:red', linestyle='', linewidth=0, label=r'$\operatorname{GNE}$')
    
    # Plot formating
    custom_legend = []
    if gne is not None:
        custom_legend.append(
            matplotlib.lines.Line2D([],[], marker=r'$\star$', markersize=7, color='tab:red', linestyle='', linewidth=0, label=r'$\operatorname{GNE}$')
        )
    
    if len(vertices) >= 2:
        custom_legend.append(
            matplotlib.lines.Line2D([],[], linewidth=1, linestyle='dashed', color="tab:red", label=r'$\mathcal{X}$'),
        )

    custom_legend.append(
        matplotlib.lines.Line2D([],[], linewidth=1, color='black', label=r'$\mathbf{x}(t)$')
    )

    if plot_edm_trajectory:
        custom_legend.append(
            matplotlib.lines.Line2D([],[], linewidth=1.5, linestyle='dotted', color='magenta', label=r'$\mathbf{x}(t)$ (EDM)')
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
        simulator : Simulator,
        scale=30,
        fontsize=8,
        figsize=(4,3),
        plot_edm_trajectory = False,
        filename=None
    ) -> None:
    """
    Plot the trajectory of a multiple population in multiple ternary plots.

    This method requires every population to have \(n^k=3\) strategies.

    Args:
        simulator (Simulator): Simulator instance holding the data for the plot.
        potential_function (Callable[[np.ndarray], np.ndarray], optional): Potential function to plot heatmap. Defaults to None.
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
            logger.error(f'ternary_plot() is only supported for games with 3 strategies per population. Population {k} has {nk} strategies.')
            return None

    # Compute edm trajectory (if enabled)
    if plot_edm_trajectory:
        t_sim = (0, simulator.t)
        x0 = simulator.log.x[0]
        q0 = simulator.log.q[0]
        out = simulator.integrate_edm_pdm(t_sim, x0, q0, t_eval = simulator.log.t)
        x_edm = out.x

    # Slice trajectories
    for t, point in enumerate(simulator.log.x):
        pos = 0
        for k in range(simulator.population_game.num_populations):
            nk = simulator.population_game.num_strategies[k]
            point_k = point[pos:pos+nk].reshape(nk,)                      # Slice trajectory
            point_k = scale*point_k/point_k.sum()                         # Scale points
            points[k].append((point_k[2], point_k[0], point_k[1]))        # Permute points

            if plot_edm_trajectory:
                point_k_edm = x_edm[pos:pos+nk, t].reshape(nk,) 
                point_k_edm = scale*point_k_edm/point_k_edm.sum()
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
            gne_k = gne[pos:pos+nk].reshape(nk,)                # Slice GNE
            gne_k = scale*gne_k/gne_k.sum()                     # Scale GNE
            gnes[k] = [(gne_k[2], gne_k[0], gne_k[1])]          # Permute 
            pos += nk

    # Make ternary plots (one for each population)
    for k in range(simulator.population_game.num_populations):

        # Initialize plot
        figure, tax = ternary.figure(scale=scale)
        figure.set_size_inches(figsize[0], figsize[1])

        # Plot boundary
        tax.boundary(linewidth=1.0)

        # Plot trajectory
        tax.plot(points[k], linewidth=1, color='black', label=rf'$\mathbf{{x}}^{{{k+1}}}(t)$')

        # Plot EDM trajectory (if enabled)
        if plot_edm_trajectory:
            tax.plot(points_edm[k], linewidth=1.5, linestyle='dotted', color='magenta', label=rf'$\mathbf{{x}}^{{{k+1}}}(t)$ (EDM)')

        # Plot GNE (if available) 
        if gne is not None:
            tax.plot(gnes[k], marker=r'$\star$', markersize=7, color='tab:red', linestyle='', linewidth=0, label=r'$\operatorname{GNE}$')
        
        # Plot formating
        custom_legend = []
        if gne is not None:
            custom_legend.append(
                matplotlib.lines.Line2D([],[], marker=r'$\star$', markersize=7, color='tab:red', linestyle='', linewidth=0, label=r'$\operatorname{GNE}$')
            )

        custom_legend.append(
            matplotlib.lines.Line2D([],[], linewidth=1, color='black', label=rf'$\mathbf{{x}}^{{{k+1}}}(t)$')
        )

        if plot_edm_trajectory:
            custom_legend.append(
                matplotlib.lines.Line2D([],[], linewidth=1.5, linestyle='dotted', color='magenta', label=rf'$\mathbf{{x}}^{{{k+1}}}(t)$ (EDM)')
            )
        
        tax.top_corner_label(r'$e_1$', fontsize=fontsize)
        tax.left_corner_label(r'$e_2$', fontsize=fontsize)
        tax.right_corner_label(r'$e_3$', fontsize=fontsize)      
        tax.clear_matplotlib_ticks()
        tax.get_axes().axis('off')
        tax.legend(handles=custom_legend, loc=1, fontsize=fontsize)
        if filename is not None:
            name, ext = filename.split('.')
            filename_k = '_'.join([name, f'pop_{k}'])
            filename_k = '.'.join([filename_k, ext])
            figure.savefig(filename_k, format="pdf", bbox_inches="tight")
        tax.show()

def make_pdm_trajectory_plot(
        simulator : Simulator,
        fontsize : int = 8,
        figsize : tuple[int, int] = (4,2),
        plot_edm_related_trajectory : bool = False,
        filename : str =None
    ) -> None:
    """
    Plot the trajectory the PDM.

    Args:
        simulator (Simulator): The simulator object holding the data to plot.
        fontsize (int, optional): The size of the font to use. Defaults to 8.
        figsize (tuple[int, int], optional): The size of the figure to use. Defaults to (4,2).
        plot_edm_related_trajectory (bool, optional): Whether or not to plot EDM related trajectory. Defaults to False.
        filename (str, optional): The filename to save the figure to. Defaults to None.
    """
    if simulator.payoff_mechanism.d > 0:
        q_stacked = np.hstack(simulator.log.q)
        
        if plot_edm_related_trajectory:
            t_sim = (0, simulator.t)
            x0 = simulator.log.x[0]
            q0 = simulator.log.q[0]
            out = simulator.integrate_edm_pdm(t_sim, x0, q0, t_eval = simulator.log.t)
            q_edm = out.q

        for i in range(simulator.payoff_mechanism.d):
            plt.figure(figsize=figsize)
        
            plt.plot(simulator.log.t, q_stacked[i, :], label='Finite agents', color='black', linewidth=1)
            
            if plot_edm_related_trajectory:
                plt.plot(simulator.log.t, q_edm[i, :], label='EDM-PDM', linestyle='dotted', color='magenta', linewidth=1.5)


            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xlabel(r'$t$', fontsize=fontsize)
            plt.ylabel(rf'$q_{{{i+1}}}(t)$', fontsize=fontsize)
            plt.grid()
            if plot_edm_related_trajectory:
                plt.legend(fontsize=fontsize)
            if filename is not None:
                name, ext = filename.split('.')
                filename_qi = '_'.join([name, f'q{i+1}'])
                filename_qi = '.'.join([filename_qi, ext])
                plt.savefig(filename_qi, format="pdf", bbox_inches="tight")
            plt.show()

def make_distance_to_gne_plot(
        simulator : Simulator
    ) -> None:
    """
    TODO: Implement this method
    """
    raise NotImplementedError