from __future__ import annotations
import typing

import logging
logger = logging.getLogger("core")

from types import SimpleNamespace

import numpy as np
import scipy as sp
import copy

from popgames.population_game import PopulationGame
from popgames.payoff_mechanism import PayoffMechanism
from popgames.revision_process import RevisionProcessABC

from popgames.utilities.input_validators import (
    check_type,
    check_valid_list,
    check_scalar_value_bounds,
    check_array_in_simplex,
    check_array_shape,
    check_function_signature
)

from popgames.utilities.plotters import (
    make_ternary_plot_single_population,
    make_ternary_plot_multi_population,
    make_pdm_trajectory_plot,
    make_distance_to_gne_plot
)

if typing.TYPE_CHECKING:
    from typing import Callable


__all__ = [
    'Simulator',
]


class Simulator:
    """
    Simulates the interplay between the three core objects: i) A population game, ii) A payoff mechanism, and iii) a
    list of revision processes (one per population).
    """
    def __init__(
            self,
            population_game : PopulationGame,
            payoff_mechanism : PayoffMechanism,
            revision_processes : list[RevisionProcessABC],
            num_agents : list[int]
    ) -> None:
        """
        Initialize the simulator object.

        Args:
            population_game (PopulationGame): The population game object.
            payoff_mechanism (PayoffMechanism): The payoff mechanism object.
            revision_processes (list[RevisionProcessABC]): The list of revision processes (one for each population).
            num_agents (list[int]): A list specifying the `finite` number of agents in each population.

        """
        # Numerical precision
        self._num_precision = 9 # Number of decimals in rounding operations

        check_type(
            arg=population_game,
            expected_type=PopulationGame,
            arg_name='population_game'
        )
        self.population_game = population_game

        check_type(
            arg=payoff_mechanism,
            expected_type=PayoffMechanism,
            arg_name='payoff_mechanism'
        )
        self.payoff_mechanism = payoff_mechanism

        assert self.population_game.n == self.payoff_mechanism.n, \
                f'Dimension missmatch between Population Game (n={self.population_game.n}) and Payoff Mechanism (n={self.payoff_mechanism.n})S'

        if self.population_game.num_populations > 1:
            check_valid_list(
                arg=revision_processes,
                length=self.population_game.num_populations,
                internal_type=RevisionProcessABC,
                name='revision_processes'
            )
            check_valid_list(
                arg=num_agents,
                length=self.population_game.num_populations,
                internal_type=int,
                name='num_agents',
                strictly_positive=True
            )
            self.revision_processes = revision_processes
            self.num_agents = num_agents
        else:
            check_type(
                arg=revision_processes,
                expected_type=RevisionProcessABC,
                arg_name='revision_processes'
            )
            check_type(
                arg=num_agents,
                expected_type=int,
                arg_name='num_agents'
            )
            check_scalar_value_bounds(
                arg=num_agents,
                arg_name='num_agents',
                strictly_positive=True
            )
            self.revision_processes = [revision_processes]
            self.num_agents = [num_agents]

        # Reset simulation state, revision_times, and logs
        self.reset()

        # Auxiliary constant parameters
        self._slices = []
        pos = 0
        for k in range(self.population_game.num_populations):
            self._slices.append(slice(pos, pos + self.population_game.num_strategies[k]))
            pos += self.population_game.num_strategies[k]

    def reset(
            self,
            x0 : np.ndarray = None,
            q0 : np.ndarray = None
    ) -> None:
        """
        Resets the simulator.

        Args:
            x0 (np.ndarray, Optional): The initial state for the strategic distribution of the society. Defaults to None.
            q0 (np.ndarray, Optional): The initial state for the payoff mechanism's PDM. Defaults to None.
        """

        # Initialize selected strategies based on x0 (if any)
        self._selected_strategies = []

        if x0 is None:
            for k in range(self.population_game.num_populations):
                sel_strategies_pop_k = np.random.randint(0, self.population_game.num_strategies[k], self.num_agents[k])
                self._selected_strategies.append(sel_strategies_pop_k)
        else:
            for k, s in zip(range(self.population_game.num_populations), self._slices):
                check_array_in_simplex(
                    arg=x0[s].reshape(-1,),
                    n=self.population_game.num_strategies[k],
                    m=self.population_game.masses[k],
                    arg_name=f'x0_{k}'
                )
                sel_strategies_pop_k = np.zeros((self.num_agents[k],)).astype(int)
                Ni_k = np.floor(self.num_agents[k]*x0[s]/self.population_game.masses[k]).astype(int)
                pos = 0
                for _ in range(self.num_agents[k] - Ni_k.sum()):
                    Ni_k[pos, 0] += 1
                    pos = (pos + 1) % self.population_game.num_strategies[k]
                pos = 0
                for i in range(self.population_game.num_strategies[k]):
                    sel_strategies_pop_k[pos : pos + Ni_k[i, 0]] = i
                    pos += Ni_k[i, 0]
                
                self._selected_strategies.append(sel_strategies_pop_k)

        # Initialize revision times
        self._revision_times = []
        for k in range(self.population_game.num_populations):
            rev_times_pop_k = self.revision_processes[k].sample_next_revision_time(self.num_agents[k])
            self._revision_times.append(np.round(rev_times_pop_k, self._num_precision))

        # Initialize simulator state
        self.t = 0
        self.x = self._get_strategic_distribution()

        if q0 is not None:
            check_array_shape(q0, (self.payoff_mechanism.d, 1), 'q0')
            self.q = copy.deepcopy(q0)
        else:
            self.q = np.zeros((self.payoff_mechanism.d, 1))

        # Initialize log
        self.log = SimpleNamespace(t = [self.t], 
                                   x = [self.x], 
                                   q = [self.q],
                                   p = [self.payoff_mechanism.h_map(self.q, self.x)])

    def run(
            self,
            T_sim : int,
            verbose : bool = True
    ) -> SimpleNamespace:
        """
        Run the simulation.

        Args:
            T_sim (int): The total time to simulate (in units specified by the agents' alarm clocks).
            verbose (bool): Whether to print simulation information. Defaults to True.

        Returns:
            SimpleNamespace: The simulation results as a SimpleNamespace object.
        """
        check_type(
            arg=T_sim,
            expected_type=int,
            arg_name="T_sim"
        )
        check_scalar_value_bounds(
            arg=T_sim,
            arg_name="T_sim",
            strictly_positive=True
        )

        time_remaining = T_sim

        while(time_remaining > 0):
        
            if verbose: 
                logger.info(f"Simulator's remaining time = {time_remaining:.3F}")
                
            time_step = np.min([np.min(self._revision_times[k]) for k in range(self.population_game.num_populations)])
        
            if time_remaining >= time_step:

                self._microscopic_step(time_step)
                time_remaining = time_remaining - time_step

            else:

                self._microscopic_step(time_remaining)
                time_remaining = 0
        
        return self._get_flattened_log()

    def integrate_edm_pdm(
            self,
            t_span : tuple,
            x0 : np.ndarray,
            q0 : np.ndarray = None,
            t_eval : list = None,
            method : str ='Radau',
            output_trajectory : bool = True
    ) -> SimpleNamespace:
        """
        Numerically integrate the underlying EDM-PDM system.

        This method relies on ``scipy.integrate.solve_ivp``.

        Args:
            t_span (tuple): The time span of the integration.
            x0 (np.ndarray): The initial strategic distribution of the society.
            q0 (np.ndarray): The initial state of the PDM. Defaults to None.
            t_eval (list): The times at which to evaluate the integration. Defaults to None.
            method (str): The integration method. Defaults to 'Radau'.
            output_trajectory (bool): Whether to output the trajectory or just the final state-output pair. Defaults to True.

        Returns:
            SimpleNamespace: The integration results as a SimpleNamespace object.
        """
        check_array_shape(
            arg=x0,
            expected_shape=(self.population_game.n, 1),
            arg_name='x0'
        )

        if q0 is None:
            q0 = np.zeros((self.payoff_mechanism.d, 1))
        else:
            check_array_shape(
                arg=q0,
                expected_shape=(self.payoff_mechanism.d, 1),
                arg_name='q0'
            )

        y0 = np.vstack([q0, x0]).reshape(self.payoff_mechanism.d + self.population_game.n,)
        
        sol = sp.integrate.solve_ivp(
            fun=self._rhs_edm_pdm_wrapped,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method=method
        )
        
        q = sol.y[:self.payoff_mechanism.d, :]
        x = sol.y[self.payoff_mechanism.d:, :]

        if output_trajectory:
            T = sol.y.shape[1]
            p = np.zeros((self.population_game.n, T))
            for t in range(T):
                q_t = q[:, t].reshape(self.payoff_mechanism.d, 1)
                x_t = x[:, t].reshape(self.population_game.n, 1)
                p_t = self.payoff_mechanism.h_map(q_t, x_t) # TODO: Can h_map be evaluated in batches to remove this loop?
                p[:, t] = p_t.reshape(self.population_game.n,) 
            
            out = SimpleNamespace(t=sol.t, x=x, q=q, p=p)
        else:
            p = self.payoff_mechanism.h_map(q[:, -1].reshape(self.payoff_mechanism.d, 1), x0)

            out = SimpleNamespace(t=sol.t[-1], 
                                  x=x[:, -1].reshape(self.population_game.n, 1),
                                  q=q[:, -1].reshape(self.payoff_mechanism.d, 1),
                                  p=p)

        return out

    def _get_strategic_distribution(self) -> np.ndarray:
        """
        Internal method to get the strategic distribution of the soceity.

        Should not be called directly from outside the class.

        Returns:
            np.ndarray: The strategic distribution of the soceity.
        """
        x = []
        for k in range(self.population_game.num_populations):
            n_k = self.population_game.num_strategies[k]
            X_k = np.zeros((n_k, 1))
            for i in range(n_k):
                X_ik = np.where(self._selected_strategies[k] == i)
                X_k[i, 0] = len(X_ik[0])/self.num_agents[k]
            x.append(X_k * self.population_game.masses[k])
        return np.vstack(x)
    
    def _microscopic_step(
            self,
            time_step : float
    ) -> SimpleNamespace:
        """
        Internal method to numerically integrate the PDM over a microscopic step.

        Should not be called directly from outside the class.

        Args:
            time_step (float): The time step for the integration.

        Returns:
            SimpleNamespace: The integration results as a SimpleNamespace object.
        """
        out = self.payoff_mechanism.integrate(
            q0=self.q,
            x0=self.x,
            t_span=(self.t, self.t + time_step),
            method='Radau',
            output_trajectory=False
        )
        self.q = out.q

        for k, s in zip(range(self.population_game.num_populations), self._slices): # loop over populations
            self._revision_times[k] = np.round(np.maximum(self._revision_times[k] - time_step, 0), self._num_precision) # Shift revision times
            revising_agents_k = np.where(self._revision_times[k] == 0)
            selected_strategies_t_k = copy.deepcopy(self._selected_strategies[k])

            for agent in revising_agents_k[0]:
                i = selected_strategies_t_k[agent]
                self._selected_strategies[k][agent] = self.revision_processes[k].sample_next_strategy(out.p[s], self.x[s], i)
                self._revision_times[k][agent] = np.round(self.revision_processes[k].sample_next_revision_time(1)[0], self._num_precision)

            if(not np.array_equal(self._selected_strategies[k], selected_strategies_t_k)):
                self.x = self._get_strategic_distribution()

        self.t += time_step
        self._update_log()

    def _rhs_edm_pdm_wrapped(
            self,
            t : float,
            y : np.ndarray
    ) -> np.ndarray:
        """
        Internal method to wrap the RHS of the overall EDM-PDM system to enable compatibility with ``scipy.integrate.solve_ivp``.

        Should not be called directly from outside the class.

        Args:
            t (float): placeholder.
            y (np.ndarray): Input vector of shape (d+n, 1).

        Returns:
            np.ndarray: Output vector of shape (d+n,).
        """
        q_col = y[:self.payoff_mechanism.d].reshape(self.payoff_mechanism.d, 1)
        x_col = y[self.payoff_mechanism.d:].reshape(self.payoff_mechanism.n, 1)
        p_col = self.payoff_mechanism.h_map(q_col, x_col)
        dy = []

        # PDM (dot q)
        dy.append(self.payoff_mechanism.w_map(q_col, x_col))

        # EDM (dot x)
        for k, s in zip(range(self.population_game.num_populations), self._slices):
            dy.append(self.revision_processes[k].rhs_edm(x=x_col[s], p=p_col[s]))
        return np.vstack(dy).reshape(self.payoff_mechanism.d + self.payoff_mechanism.n,)
    
    def _update_log(self) -> None:
        """
        Internal method to update the simulation log.

        Should not be called directly from outside the class.
        """
        self.log.t.append(self.t)
        self.log.x.append(self.x)
        self.log.q.append(self.q)
        self.log.p.append(self.payoff_mechanism.h_map(self.q, self.x))

    def _get_flattened_log(self) -> SimpleNamespace:
        """
        Internal method to get the flattened log of the simulation.

        Should not be called directly from outside the class.

        Returns:
            SimpleNamespace: The flattened log of the simulation as a SimpleNamespace object.
        """
        flattened_log = SimpleNamespace(
            t = np.hstack(self.log.t),
            x = np.hstack(self.log.x),
            q = np.hstack(self.log.q),
            p = np.hstack(self.log.p)
        )
        return flattened_log

    def ternary_plot(
            self,
            scale : int = 30,
            fontsize : int = 8,
            figsize : tuple[int] = (4, 3),
            filename : str = None,
            plot_edm_trajectory : bool = False,
            potential_function : Callable[[np.ndarray], np.ndarray] = None
    ) -> None:
        """
        Make a ternary plot for the simulated scenario.

        This method is only supported for single-population games with n=3.

        Args:
            scale (int): Scaling factor for the ternary plot. Defaults to 30.
            fontsize (int): Fontsize for the plot. Defaults to 8.
            figsize (tuple[int]): Figure size. Defaults to (4, 3).
            filename (str, optional): Filename to save the figure. Defaults to None.
            plot_edm_trajectory (bool): Whether to plot edm trajectory or not. Defaults to False.
            potential_function (Callable[[np.ndarray], np.ndarray]): Potential function to plot as a heatmap. Defaults to None.
        """
        
        if self.population_game.num_populations > 1:
            if potential_function is not None:
                logger.warning('Plotting potential functions over the ternary plot is only supported for single population games.')
            
            make_ternary_plot_multi_population(
                self,
                scale = scale,
                fontsize = fontsize,
                figsize = figsize,
                plot_edm_trajectory=plot_edm_trajectory,
                filename = filename
            )
        else:
            if potential_function is not None:
                check_function_signature(
                    arg=potential_function,
                    expected_input_shapes=[(self.population_game.n, 1)],
                    expected_output_shape=(1, 1),
                    name='potential_function'
                )

            make_ternary_plot_single_population(
                self,
                potential_function = potential_function,
                plot_edm_trajectory = plot_edm_trajectory,
                scale = scale,
                fontsize = fontsize,
                figsize = figsize,
                filename = filename
            )

    def plot_pdm_trajectory(
            self,
            fontsize : int = 8,
            figsize : tuple[int] = (4, 2),
            filename : str = None,
            plot_edm_related_trajectory : bool = False
    ) -> None:
        """
        Plot the PDM trajectory over time.

        Args:
            fontsize (int): Fontsize. Defaults to 8.
            figsize (tuple[int]): Figure size. Defaults to (4,2)
            filename (str, optional): Filename to save the figure. Defaults to None.
            plot_edm_related_trajectory (bool): Whether to plot edm trajectory or not. Defaults to False.
        """
            
        make_pdm_trajectory_plot(
            self,
            fontsize = fontsize,
            figsize = figsize,
            plot_edm_related_trajectory=plot_edm_related_trajectory,
            filename = filename
        )

    def plot_distance_to_gne(self) -> None:
        """
        Plot the Euclidean distance to a generalized Nash equilibrium (GNE) of the underlying game.
        """
        make_distance_to_gne_plot(self)
