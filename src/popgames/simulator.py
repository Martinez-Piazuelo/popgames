from __future__ import annotations

import copy
import importlib
import logging
import typing
from types import SimpleNamespace

import numpy as np
import scipy as sp

from popgames.payoff_mechanism import PayoffMechanism
from popgames.plotting import VisualizationMixin
from popgames.population_game import PopulationGame
from popgames.revision_process import PoissonRevisionProcess, RevisionProcessABC
from popgames.utilities.input_validators import (
    check_array_in_simplex,
    check_array_shape,
    check_scalar_value_bounds,
    check_type,
    check_valid_list,
)

if typing.TYPE_CHECKING:
    from typing import Union


__all__ = [
    "Simulator",
]


logger = logging.getLogger(__name__)


class Simulator(VisualizationMixin):
    """
    Simulates the interplay between the three core objects: i) A population game, ii) A payoff mechanism, and iii) a
    list of revision processes (one per population).
    """

    _numba_log_capacity = (
        1_000_000  # Maximum log entries per call to the compiled (numba) event loop
    )

    def __init__(
        self,
        population_game: PopulationGame,
        payoff_mechanism: PayoffMechanism,
        revision_processes: Union[RevisionProcessABC, list[RevisionProcessABC]],
        num_agents: Union[int, list[int]],
        fast_path: bool = True,
        pdm_method: str = "RK4",
        pdm_max_step: float = 0.01,
        seed: Union[int, np.random.Generator] = None,
        backend: str = "numpy",
    ) -> None:
        """
        Initialize the simulator object.

        Args:
            population_game (PopulationGame): The population game object.
            payoff_mechanism (PayoffMechanism): The payoff mechanism object.
            revision_processes (Union[RevisionProcessABC, list[RevisionProcessABC]]):
                A single revision process for the single-population case, or a list of revision processes
                for the multi-population case, where each element corresponds to a specific population.
            num_agents (Union[int, list[int]]):
                The number of agents as an integer for the single-population case, or a list of integers
                specifying the number of agents in each population for the multi-population case.
            fast_path (bool): Whether to use the fast simulation path. It is only used if all revision processes
                are ``PoissonRevisionProcess`` instances; otherwise the per-agent simulation path is used.
                Set to False to force the per-agent path (e.g., for comparison). Defaults to True.
            pdm_method (str): Method used to integrate the PDM between revision events. Either 'RK4' or any
                method supported by ``scipy.integrate.solve_ivp`` (e.g., 'Radau' for stiff PDMs).
                Defaults to 'RK4'.
            pdm_max_step (float): Maximum step size used to integrate the PDM between revision events.
                Defaults to 0.01.
            seed (Union[int, np.random.Generator], optional): Seed or random number generator for the simulation.
                Defaults to None, in which case NumPy's global random state is used (see ``np.random.seed``).
                Custom (non-Poisson) revision processes always draw from their own random sources.
            backend (str): Either 'numpy' or 'numba'. The 'numba' backend compiles the fast simulation path with
                Numba (requires ``pip install popgames[numba]``). It requires the fast path, built-in revision
                protocols, ``pdm_method='RK4'``, and payoff mechanism functions that can be compiled with
                ``numba.njit``; otherwise, a warning is logged and the 'numpy' backend is used. Compilation takes a
                few seconds on the first run of each session. Defaults to 'numpy'.
        """
        # Numerical precision
        self._num_precision = 9  # Number of decimals in rounding operations

        check_type(
            arg=population_game,
            expected_type=PopulationGame,
            arg_name="population_game",
        )
        self.population_game = population_game

        check_type(
            arg=payoff_mechanism,
            expected_type=PayoffMechanism,
            arg_name="payoff_mechanism",
        )
        self.payoff_mechanism = payoff_mechanism

        assert self.population_game.n == self.payoff_mechanism.n, (
            f"Dimension missmatch between Population Game (n={self.population_game.n}) and Payoff Mechanism (n={self.payoff_mechanism.n})S"
        )

        if self.population_game.num_populations > 1:
            check_valid_list(
                arg=revision_processes,
                length=self.population_game.num_populations,
                internal_type=RevisionProcessABC,
                name="revision_processes",
            )
            check_valid_list(
                arg=num_agents,
                length=self.population_game.num_populations,
                internal_type=int,
                name="num_agents",
                strictly_positive=True,
            )
            self.revision_processes = revision_processes
            self.num_agents = num_agents
        else:
            check_type(
                arg=revision_processes,
                expected_type=RevisionProcessABC,
                arg_name="revision_processes",
            )
            check_type(arg=num_agents, expected_type=int, arg_name="num_agents")
            check_scalar_value_bounds(
                arg=num_agents, arg_name="num_agents", strictly_positive=True
            )
            self.revision_processes = [revision_processes]
            self.num_agents = [num_agents]

        # Auxiliary constant parameters
        self._slices = []
        pos = 0
        for k in range(self.population_game.num_populations):
            self._slices.append(
                slice(pos, pos + self.population_game.num_strategies[k])
            )
            pos += self.population_game.num_strategies[k]

        # Simulation path
        check_type(arg=fast_path, expected_type=bool, arg_name="fast_path")
        all_poisson = all(
            isinstance(rp, PoissonRevisionProcess) for rp in self.revision_processes
        )
        self.uses_fast_path = fast_path and all_poisson
        if fast_path and not all_poisson:
            logger.info(
                "Fast path requires all revision processes to be PoissonRevisionProcess. "
                "Using the per-agent simulation path instead."
            )

        # PDM integration between revision events
        check_type(arg=pdm_method, expected_type=str, arg_name="pdm_method")
        self.pdm_method = pdm_method
        check_scalar_value_bounds(
            arg=pdm_max_step, arg_name="pdm_max_step", strictly_positive=True
        )
        self.pdm_max_step = pdm_max_step

        # Random number generator (NumPy's global random state if no seed is provided)
        self._rng = np.random if seed is None else np.random.default_rng(seed)

        # Simulation backend
        self._setup_backend(backend)

        # Reset simulation state, revision_times, and logs
        self.reset()

    def reset(self, x0: np.ndarray = None, q0: np.ndarray = None) -> None:
        """
        Resets the simulator.

        Args:
            x0 (np.ndarray, Optional): The initial state for the strategic distribution of the society. Defaults to None.
            q0 (np.ndarray, Optional): The initial state for the payoff mechanism's PDM. Defaults to None.
        """

        # Initialize the number of agents playing each strategy based on x0 (if any)
        self._counts = []

        if x0 is None:
            for k in range(self.population_game.num_populations):
                n_k = self.population_game.num_strategies[k]
                sel_strategies_pop_k = np.floor(
                    n_k * self._rng.random(self.num_agents[k])
                ).astype(int)
                self._counts.append(np.bincount(sel_strategies_pop_k, minlength=n_k))
        else:
            for k, s in zip(range(self.population_game.num_populations), self._slices):
                check_array_in_simplex(
                    arg=x0[s].reshape(
                        -1,
                    ),
                    n=self.population_game.num_strategies[k],
                    m=self.population_game.masses[k],
                    arg_name=f"x0_{k}",
                )
                Ni_k = np.floor(
                    self.num_agents[k] * x0[s] / self.population_game.masses[k]
                ).astype(int)
                pos = 0
                for _ in range(self.num_agents[k] - Ni_k.sum()):
                    Ni_k[pos, 0] += 1
                    pos = (pos + 1) % self.population_game.num_strategies[k]
                self._counts.append(Ni_k.reshape(-1))

        # Initialize per-agent selected strategies and revision times (per-agent path only)
        self._selected_strategies = []
        self._revision_times = []
        if not self.uses_fast_path:
            for k in range(self.population_game.num_populations):
                n_k = self.population_game.num_strategies[k]
                self._selected_strategies.append(
                    np.repeat(np.arange(n_k), self._counts[k])
                )
                rev_times_pop_k = self._sample_next_revision_time(k, self.num_agents[k])
                self._revision_times.append(
                    np.round(rev_times_pop_k, self._num_precision)
                )

        # Initialize simulator state
        self.t = 0
        self.x = np.vstack(
            [
                (self._counts[k] / self.num_agents[k]).reshape(-1, 1)
                * self.population_game.masses[k]
                for k in range(self.population_game.num_populations)
            ]
        )

        if q0 is not None:
            check_array_shape(q0, (self.payoff_mechanism.d, 1), "q0")
            self.q = copy.deepcopy(q0)
        else:
            self.q = np.zeros((self.payoff_mechanism.d, 1))

        self.p = self.payoff_mechanism.h_map(self.q, self.x)

        # Initialize log
        self.log = SimpleNamespace(
            t=[self.t],
            x=[self.x],
            q=[self.q],
            p=[self.p],
        )
        self._log_interval = None
        self._last_log_t = self.t

    def run(
        self, T_sim: int, verbose: bool = False, log_interval: float = None
    ) -> SimpleNamespace:
        """
        Run the simulation.

        Args:
            T_sim (int): The total time to simulate (in units specified by the agents' alarm clocks).
            verbose (bool): Whether to print simulation information. Defaults to False.
            log_interval (float, optional): Minimum time between consecutive log entries. Defaults to None, in
                which case every revision event is logged. The final state is always logged.

        Returns:
            SimpleNamespace: The simulation results as a SimpleNamespace object.
        """
        check_type(arg=T_sim, expected_type=int, arg_name="T_sim")
        check_scalar_value_bounds(arg=T_sim, arg_name="T_sim", strictly_positive=True)
        if log_interval is not None:
            check_scalar_value_bounds(
                arg=log_interval, arg_name="log_interval", strictly_positive=True
            )
        self._log_interval = log_interval

        if self.backend == "numba":
            self._run_numba_path(T_sim)
        elif self.uses_fast_path:
            self._run_fast_path(T_sim, verbose)
        else:
            self._run_per_agent_path(T_sim, verbose)

        if self._last_log_t != self.t:  # Always log the final state
            self._update_log(force=True)

        return self._get_flattened_log()

    def _run_per_agent_path(self, T_sim: int, verbose: bool) -> None:
        """
        Internal method to run the simulation by tracking the strategy and alarm clock of every agent.

        Should not be called directly from outside the class.

        Args:
            T_sim (int): The total time to simulate.
            verbose (bool): Whether to print simulation information.
        """
        time_remaining = T_sim

        while time_remaining > 0:
            if verbose:
                logger.info(
                    f"Simulator's remaining time = {time_remaining:.3F}"
                )  # pragma: no cover

            time_step = np.min(
                [
                    np.min(self._revision_times[k])
                    for k in range(self.population_game.num_populations)
                ]
            )

            if time_remaining >= time_step:
                self._microscopic_step(time_step)
                time_remaining = time_remaining - time_step

            else:
                self._microscopic_step(time_remaining)
                time_remaining = 0

    def _run_fast_path(self, T_sim: int, verbose: bool) -> None:
        """
        Internal method to run the simulation by tracking only the number of agents playing each strategy.

        Valid only when all agents revise at the ticks of independent Poisson alarm clocks. By the superposition
        property, the time to the next revision event is exponentially distributed with rate
        ``sum_k N^k * rate^k``, the revising population is chosen with probability proportional to
        ``N^k * rate^k``, and the revising agent is chosen uniformly at random within that population.

        Should not be called directly from outside the class.

        Args:
            T_sim (int): The total time to simulate.
            verbose (bool): Whether to print simulation information.
        """
        populations_rates = np.array(
            [
                self.num_agents[k] * self.revision_processes[k].Poisson_clock_rate
                for k in range(self.population_game.num_populations)
            ]
        )
        total_rate = populations_rates.sum()
        cumulative_rates = np.cumsum(populations_rates)
        t_end = self.t + T_sim

        while True:
            if verbose:
                logger.info(
                    f"Simulator's remaining time = {t_end - self.t:.3F}"
                )  # pragma: no cover

            time_step = self._rng.exponential(1 / total_rate)
            if self.t + time_step >= t_end:  # No more revisions before t_end
                self._advance_pdm(t_end - self.t)
                self.t = t_end
                break

            self._advance_pdm(time_step)
            self.t += time_step

            # Sample the revising population k and the current strategy i of the revising agent
            k = (
                self._sample_index(cumulative_rates, total_rate)
                if self.population_game.num_populations > 1
                else 0
            )
            counts_k = self._counts[k]
            i = self._sample_index(np.cumsum(counts_k), self.num_agents[k])

            # Sample the next strategy j of the revising agent
            s = self._slices[k]
            j = self.revision_processes[k].sample_next_strategy(
                self.p[s], self.x[s], i, rng=self._rng
            )

            if j != i:
                counts_k[i] -= 1
                counts_k[j] += 1
                scale = self.population_game.masses[k] / self.num_agents[k]
                self.x = self.x.copy()  # Logged states must not be modified in place
                self.x[s.start + i, 0] = counts_k[i] * scale
                self.x[s.start + j, 0] = counts_k[j] * scale
                self.p = self.payoff_mechanism.h_map(self.q, self.x)

            self._update_log()

    def _setup_backend(self, backend: str) -> None:
        """
        Internal method to set up the simulation backend.

        Falls back to the 'numpy' backend (with a warning) if the 'numba' backend cannot be used.
        Should not be called directly from outside the class.

        Args:
            backend (str): Either 'numpy' or 'numba'.
        """
        check_type(arg=backend, expected_type=str, arg_name="backend")
        if backend not in ("numpy", "numba"):
            raise ValueError(
                f"backend must be either 'numpy' or 'numba', got '{backend}'."
            )

        self.backend = "numpy"
        if backend == "numpy":
            return

        try:
            _numba_backend = importlib.import_module("popgames._numba_backend")
        except ImportError as e:
            raise ImportError(
                "The numba backend requires numba. Install it with `pip install popgames[numba]`."
            ) from e

        reason = None
        specs = [
            _numba_backend.protocol_spec(rp.revision_protocol)
            for rp in self.revision_processes
        ]
        if not self.uses_fast_path:
            reason = "it requires the fast path (fast_path=True and PoissonRevisionProcess revision processes)"
        elif any(spec is None for spec in specs):
            k = next(k for k, spec in enumerate(specs) if spec is None)
            reason = (
                f"the revision protocol of population {k} "
                f"({type(self.revision_processes[k].revision_protocol).__name__}) is not a built-in protocol"
            )
        elif self.payoff_mechanism.d > 0 and self.pdm_method != "RK4":
            reason = f"it only supports pdm_method='RK4' (got '{self.pdm_method}')"
        else:
            try:
                h_map, w_map = _numba_backend.compile_payoff_mechanism(
                    self.payoff_mechanism
                )
            except Exception as e:
                logger.debug("Numba compilation error", exc_info=True)
                reason = (
                    "the payoff mechanism functions could not be compiled with numba.njit "
                    f"({type(e).__name__}: {str(e).strip().splitlines()[0]})"
                )

        if reason is not None:
            logger.warning(
                f"The numba backend cannot be used: {reason}. Using the numpy backend instead."
            )
            return

        params = np.zeros((len(specs), max(len(spec[1]) for spec in specs)))
        for k, (_, params_k) in enumerate(specs):
            params[k, : len(params_k)] = params_k

        self.backend = "numba"
        self._numba = SimpleNamespace(
            module=_numba_backend,
            h_map=h_map,
            w_map=w_map,
            kinds=np.array([spec[0] for spec in specs], dtype=np.int64),
            params=params,
        )

    def _run_numba_path(self, T_sim: int) -> None:
        """
        Internal method to run the fast simulation path with the compiled (numba) event loop.

        Should not be called directly from outside the class.

        Args:
            T_sim (int): The total time to simulate.
        """
        nb = self._numba
        n, d = self.payoff_mechanism.n, self.payoff_mechanism.d

        # Numba draws from its own generator: derive it from NumPy's global state if no seed was provided
        if isinstance(self._rng, np.random.Generator):
            rng = self._rng
        else:
            rng = np.random.default_rng(np.random.randint(0, 2**63 - 1, dtype=np.int64))

        counts = np.concatenate(self._counts).astype(np.int64)
        starts = np.array([s.start for s in self._slices] + [n], dtype=np.int64)
        num_agents = np.array(self.num_agents, dtype=np.float64)
        masses = np.array(self.population_game.masses, dtype=np.float64)
        clock_rates = np.array(
            [rp.Poisson_clock_rate for rp in self.revision_processes], dtype=np.float64
        )
        x = np.array(self.x, dtype=np.float64)
        q = np.array(self.q, dtype=np.float64)
        p = np.array(self.p, dtype=np.float64)

        # Log buffers (the compiled loop returns whenever they are full)
        log_interval = 0.0 if self._log_interval is None else self._log_interval
        if log_interval > 0:
            expected_entries = T_sim / log_interval
        else:
            expected_entries = (num_agents * clock_rates).sum() * T_sim
        capacity = int(min(1.2 * expected_entries + 16, self._numba_log_capacity))

        t, t_end = float(self.t), float(self.t + T_sim)
        last_log_t = float(self._last_log_t)
        num_invalid = 0
        done = False
        while not done:
            log_t = np.empty(capacity)
            log_x, log_p = np.empty((n, capacity)), np.empty((n, capacity))
            log_q = np.empty((d, capacity))

            t, num_logged, done, last_log_t, num_invalid_chunk = nb.module.run_events(
                nb.h_map,
                nb.w_map,
                rng,
                t,
                t_end,
                counts,
                x,
                q,
                p,
                starts,
                num_agents,
                masses,
                clock_rates,
                nb.kinds,
                nb.params,
                self.pdm_max_step,
                log_interval,
                last_log_t,
                log_t,
                log_x,
                log_q,
                log_p,
            )
            num_invalid += num_invalid_chunk

            # Store the log entries as blocks (one column per entry)
            if num_logged > 0:
                self.log.t.append(log_t[:num_logged].copy())
                self.log.x.append(log_x[:, :num_logged].copy())
                self.log.q.append(log_q[:, :num_logged].copy())
                self.log.p.append(log_p[:, :num_logged].copy())

        if num_invalid > 0:
            logger.warning(
                f"Invalid switching probabilities in {num_invalid} revision events. Clipping them by default."
            )

        self.t = t
        self._last_log_t = last_log_t
        self.x, self.q, self.p = x, q, p
        self._counts = [counts[s].copy() for s in self._slices]

    def _sample_index(self, cumulative_weights: np.ndarray, total_weight: float) -> int:
        """
        Internal method to sample an index with probability proportional to its (non-cumulative) weight.

        Should not be called directly from outside the class.

        Args:
            cumulative_weights (np.ndarray): Cumulative sum of the non-negative weights.
            total_weight (float): Sum of the weights.

        Returns:
            int: The sampled index.
        """
        index = int(
            np.searchsorted(
                cumulative_weights, self._rng.random() * total_weight, side="right"
            )
        )
        return min(index, len(cumulative_weights) - 1)

    def _advance_pdm(self, time_step: float) -> None:
        """
        Internal method to integrate the PDM over a time step with the strategic distribution held constant.

        Updates the PDM state and the payoff vector. Should not be called directly from outside the class.

        Args:
            time_step (float): The time step for the integration.
        """
        if self.payoff_mechanism.d == 0:  # Memoryless PDM: payoffs depend only on x
            return

        out = self.payoff_mechanism.integrate(
            q0=self.q,
            x0=self.x,
            t_span=(self.t, self.t + time_step),
            method=self.pdm_method,
            output_trajectory=False,
            max_step=self.pdm_max_step,
        )
        self.q = out.q
        self.p = out.p

    def _sample_next_revision_time(self, k: int, size: int) -> np.ndarray:
        """
        Internal method to sample the next revision times of agents in population k.

        Should not be called directly from outside the class.

        Args:
            k (int): Population index.
            size (int): Number of samples.

        Returns:
            np.ndarray: The sampled revision times with shape ``(size,)``.
        """
        rp = self.revision_processes[k]
        if isinstance(rp, PoissonRevisionProcess):
            return rp.sample_next_revision_time(size, rng=self._rng)
        return rp.sample_next_revision_time(size)

    def _sample_next_strategy(
        self, k: int, p: np.ndarray, x: np.ndarray, i: int
    ) -> int:
        """
        Internal method to sample the next strategy of an agent in population k.

        Should not be called directly from outside the class.

        Args:
            k (int): Population index.
            p (np.ndarray): Payoff vector of population k.
            x (np.ndarray): Strategic distribution of population k.
            i (int): Current strategy of the agent.

        Returns:
            int: The newly selected strategy.
        """
        rp = self.revision_processes[k]
        if isinstance(rp, PoissonRevisionProcess):
            return rp.sample_next_strategy(p, x, i, rng=self._rng)
        return rp.sample_next_strategy(p, x, i)

    def integrate_edm_pdm(
        self,
        t_span: tuple,
        x0: np.ndarray,
        q0: np.ndarray = None,
        t_eval: list = None,
        method: str = "Radau",
        output_trajectory: bool = True,
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
            arg=x0, expected_shape=(self.population_game.n, 1), arg_name="x0"
        )

        if q0 is None:
            q0 = np.zeros((self.payoff_mechanism.d, 1))
        else:
            check_array_shape(
                arg=q0, expected_shape=(self.payoff_mechanism.d, 1), arg_name="q0"
            )

        y0 = np.vstack([q0, x0]).reshape(
            self.payoff_mechanism.d + self.population_game.n,
        )

        sol = sp.integrate.solve_ivp(
            fun=self._rhs_edm_pdm_wrapped,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method=method,
        )

        q = sol.y[: self.payoff_mechanism.d, :]  # type: ignore[attr-defined]
        x = sol.y[self.payoff_mechanism.d :, :]  # type: ignore[attr-defined]

        if output_trajectory:
            T = (
                sol.y  # type: ignore[attr-defined]
            ).shape[1]
            p = np.zeros((self.population_game.n, T))
            for t in range(T):
                q_t = q[:, t].reshape(self.payoff_mechanism.d, 1)
                x_t = x[:, t].reshape(self.population_game.n, 1)
                p_t = self.payoff_mechanism.h_map(
                    q_t, x_t
                )  # TODO: Can h_map be evaluated in batches to remove this loop?
                p[:, t] = p_t.reshape(
                    self.population_game.n,
                )

            out = SimpleNamespace(
                t=sol.t,  # type: ignore[attr-defined]
                x=x,
                q=q,
                p=p,
            )
        else:
            p = self.payoff_mechanism.h_map(
                q[:, -1].reshape(self.payoff_mechanism.d, 1),
                x[:, -1].reshape(self.population_game.n, 1),
            )

            out = SimpleNamespace(
                t=(
                    sol.t  # type: ignore[attr-defined]
                )[-1],
                x=x[:, -1].reshape(self.population_game.n, 1),
                q=q[:, -1].reshape(self.payoff_mechanism.d, 1),
                p=p,
            )

        return out

    def _get_strategic_distribution(self) -> np.ndarray:
        """
        Internal method to get the strategic distribution of the society.

        Should not be called directly from outside the class.

        Returns:
            np.ndarray: The strategic distribution of the society.
        """
        x = []
        for k in range(self.population_game.num_populations):
            n_k = self.population_game.num_strategies[k]
            X_k = np.bincount(self._selected_strategies[k], minlength=n_k).reshape(
                n_k, 1
            )
            x.append(X_k / self.num_agents[k] * self.population_game.masses[k])
        return np.vstack(x)

    def _microscopic_step(self, time_step: float) -> None:
        """
        Internal method to numerically integrate the PDM over a microscopic step.

        Should not be called directly from outside the class.

        Args:
            time_step (float): The time step for the integration.
        """
        self._advance_pdm(time_step)
        p = self.p  # Payoffs observed by all agents revising at this step

        for k, s in zip(
            range(self.population_game.num_populations), self._slices
        ):  # loop over populations
            self._revision_times[k] = np.round(
                np.maximum(self._revision_times[k] - time_step, 0), self._num_precision
            )  # Shift revision times
            revising_agents_k = np.where(self._revision_times[k] == 0)
            selected_strategies_t_k = copy.deepcopy(self._selected_strategies[k])

            for agent in revising_agents_k[0]:
                i = selected_strategies_t_k[agent]
                self._selected_strategies[k][agent] = self._sample_next_strategy(
                    k, p[s], self.x[s], i
                )
                self._revision_times[k][agent] = np.round(
                    self._sample_next_revision_time(k, 1)[0],
                    self._num_precision,
                )

            if not np.array_equal(
                self._selected_strategies[k], selected_strategies_t_k
            ):
                self.x = self._get_strategic_distribution()
                self.p = self.payoff_mechanism.h_map(self.q, self.x)

        self.t += time_step
        self._update_log()

    def _rhs_edm_pdm_wrapped(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Internal method to wrap the RHS of the overall EDM-PDM system to enable compatibility with ``scipy.integrate.solve_ivp``.

        Should not be called directly from outside the class.

        Args:
            t (float): placeholder.
            y (np.ndarray): Input vector of shape (d+n, 1).

        Returns:
            np.ndarray: Output vector of shape (d+n,).
        """
        q_col = y[: self.payoff_mechanism.d].reshape(self.payoff_mechanism.d, 1)
        x_col = y[self.payoff_mechanism.d :].reshape(self.payoff_mechanism.n, 1)
        p_col = self.payoff_mechanism.h_map(q_col, x_col)
        dy = []

        # PDM (dot q)
        dy.append(self.payoff_mechanism.w_map(q_col, x_col))

        # EDM (dot x)
        for k, s in zip(range(self.population_game.num_populations), self._slices):
            dy.append(self.revision_processes[k].rhs_edm(x=x_col[s], p=p_col[s]))
        return np.vstack(dy).reshape(
            self.payoff_mechanism.d + self.payoff_mechanism.n,
        )

    def _update_log(self, force: bool = False) -> None:
        """
        Internal method to update the simulation log.

        Should not be called directly from outside the class.

        Args:
            force (bool): Whether to log regardless of the log interval. Defaults to False.
        """
        if (
            not force
            and self._log_interval is not None
            and self.t - self._last_log_t < self._log_interval
        ):
            return

        self._last_log_t = self.t
        self.log.t.append(self.t)
        self.log.x.append(self.x)
        self.log.q.append(self.q)
        self.log.p.append(self.p)

    def _get_flattened_log(self) -> SimpleNamespace:
        """
        Internal method to get the flattened log of the simulation.

        Should not be called directly from outside the class.

        Returns:
            SimpleNamespace: The flattened log of the simulation as a SimpleNamespace object.
        """
        flattened_log = SimpleNamespace(
            t=np.hstack(self.log.t),
            x=np.hstack(self.log.x),
            q=np.hstack(self.log.q),
            p=np.hstack(self.log.p),
        )
        return flattened_log
