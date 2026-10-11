from __future__ import annotations

import typing
from types import SimpleNamespace

import numpy as np

from popgames.plotting import EnsembleVisualizationMixin

if typing.TYPE_CHECKING:
    from typing import Sequence, Union

    from popgames import Simulator


__all__ = [
    "EnsembleResult",
]


class EnsembleResult(EnsembleVisualizationMixin):
    """
    Finite-agent simulations sampled at common times, e.g., from the same initial state with different seeds, or from
    different initial states.

    Ensembles are created by ``Simulator.run_ensemble``. The arrays have one leading axis for the runs, followed by the
    layout of ``Simulator.log``:

    * ``ens.x[r]`` (shape ``(n, K)``) is run ``r``, with ``ens.x[r, i]`` the trajectory of strategy ``i``.
    * ``ens.x[:, i]`` (shape ``(M, K)``) holds the trajectories of strategy ``i`` in all runs.
    * ``ens.x[:, :, k]`` (shape ``(M, n)``) holds the strategic distributions of all runs at time ``ens.t[k]``.

    Runs with the same initial state ``(x0, q0)`` form a **group** (``groups`` and ``group_index``). Statistics such as
    ``mean`` and ``quantile`` pool all runs; use ``subset`` for statistics per group, e.g.,
    ``ens.subset(ens.groups[g]).mean()``. The plots show one group per initial state.

    Attributes:
        t (np.ndarray): The sampling times, of shape ``(K,)``.
        x (np.ndarray): The strategic distributions, of shape ``(M, n, K)``.
        q (np.ndarray): The states of the PDM, of shape ``(M, d, K)``.
        p (np.ndarray): The payoffs, of shape ``(M, n, K)``.
        seeds (np.ndarray): The seeds of the runs, of shape ``(M,)``. Run ``r`` can be reproduced with
            ``sim.reset(ens.x0[r], ens.q0[r], seed=int(ens.seeds[r]))`` followed by ``sim.run(T_sim)``.
        x0 (np.ndarray): The initial strategic distribution of each run, of shape ``(M, n, 1)``. This is the state
            the run actually started from: ``reset`` rounds the requested state to a whole number of agents per
            strategy.
        q0 (np.ndarray): The initial state of the PDM of each run, of shape ``(M, d, 1)``.
        groups (list[np.ndarray]): The indices of the runs of each group (runs with the same initial state), in order
            of first appearance.
        group_index (np.ndarray): The group of each run, of shape ``(M,)``.
        simulator (Simulator): The simulator that produced the runs.
    """

    def __init__(
        self,
        simulator: Simulator,
        t: np.ndarray,
        x: np.ndarray,
        q: np.ndarray,
        p: np.ndarray,
        seeds: np.ndarray,
        x0: np.ndarray,
        q0: np.ndarray,
    ) -> None:
        self.simulator = simulator
        self.t = t
        self.x = x
        self.q = q
        self.p = p
        self.seeds = seeds
        self.x0 = x0
        self.q0 = q0
        self._deterministic = {}

        # Groups of runs with the same initial state, numbered in order of first appearance
        initial_states = np.hstack([x0.reshape(len(x0), -1), q0.reshape(len(q0), -1)])
        _, first, inverse = np.unique(
            initial_states, axis=0, return_index=True, return_inverse=True
        )
        rank = np.empty(first.size, dtype=int)
        rank[np.argsort(first)] = np.arange(first.size)
        self.group_index = rank[inverse.reshape(-1)]
        self.groups = [np.flatnonzero(self.group_index == g) for g in range(first.size)]

    @property
    def num_runs(self) -> int:
        """The number of runs ``M``."""
        return self.x.shape[0]

    def __len__(self) -> int:
        return self.num_runs

    def __getitem__(self, r: int) -> SimpleNamespace:
        """
        Get a single run, in the layout of ``Simulator.log``.

        Args:
            r (int): The index of the run.

        Returns:
            SimpleNamespace: The run with fields ``t`` (shape ``(K,)``), ``x`` (shape ``(n, K)``), ``q`` (shape
            ``(d, K)``), and ``p`` (shape ``(n, K)``).
        """
        return SimpleNamespace(t=self.t, x=self.x[r], q=self.q[r], p=self.p[r])

    def __iter__(self):
        return (self[r] for r in range(self.num_runs))

    def mean(self) -> SimpleNamespace:
        """
        The mean over all runs at each sampling time (see ``subset`` for the mean per initial state).

        Returns:
            SimpleNamespace: Fields ``t`` (shape ``(K,)``), ``x`` (shape ``(n, K)``), ``q`` (shape ``(d, K)``), and
            ``p`` (shape ``(n, K)``).
        """
        return SimpleNamespace(
            t=self.t,
            x=self.x.mean(axis=0),
            q=self.q.mean(axis=0),
            p=self.p.mean(axis=0),
        )

    def quantile(self, quantiles: Union[float, Sequence[float]]) -> SimpleNamespace:
        """
        The quantiles over all runs at each sampling time (see ``subset`` for the quantiles per initial state).

        Args:
            quantiles (Union[float, Sequence[float]]): One or several quantiles in ``[0, 1]``.

        Returns:
            SimpleNamespace: Fields ``t`` (shape ``(K,)``), ``x``, ``q``, and ``p``. For a single quantile, ``x`` has
            shape ``(n, K)`` (similarly for ``q`` and ``p``); for several quantiles, it has shape ``(Q, n, K)``, with
            one entry per quantile.
        """
        return SimpleNamespace(
            t=self.t,
            x=np.quantile(self.x, quantiles, axis=0),
            q=np.quantile(self.q, quantiles, axis=0),
            p=np.quantile(self.p, quantiles, axis=0),
        )

    def at(self, t: float) -> SimpleNamespace:
        """
        The states of all runs at the last sampling time at or before ``t``.

        Args:
            t (float): The time, with ``ens.t[0] <= t``.

        Returns:
            SimpleNamespace: Fields ``t`` (the sampling time used), ``x`` (shape ``(M, n)``), ``q`` (shape
            ``(M, d)``), and ``p`` (shape ``(M, n)``).
        """
        k = int(np.searchsorted(self.t, t, side="right")) - 1
        if k < 0:
            raise ValueError(f"t={t} is before the first sampling time {self.t[0]}.")
        return SimpleNamespace(
            t=self.t[k], x=self.x[:, :, k], q=self.q[:, :, k], p=self.p[:, :, k]
        )

    def final(self) -> SimpleNamespace:
        """
        The states of all runs at the last sampling time (see ``at``).

        Returns:
            SimpleNamespace: Fields ``t``, ``x`` (shape ``(M, n)``), ``q`` (shape ``(M, d)``), and ``p`` (shape
            ``(M, n)``).
        """
        return self.at(self.t[-1])

    def subset(self, runs: Union[Sequence[int], np.ndarray]) -> EnsembleResult:
        """
        The ensemble restricted to some runs, e.g., ``ens.subset(ens.groups[g])`` for the runs of group ``g``.

        Args:
            runs (Union[Sequence[int], np.ndarray]): The indices of the runs (or a boolean mask).

        Returns:
            EnsembleResult: A new ensemble with the selected runs.
        """
        runs = np.arange(self.num_runs)[runs]
        if runs.ndim != 1 or runs.size == 0:
            raise ValueError("subset requires a non-empty selection of runs.")
        return EnsembleResult(
            simulator=self.simulator,
            t=self.t,
            x=self.x[runs],
            q=self.q[runs],
            p=self.p[runs],
            seeds=self.seeds[runs],
            x0=self.x0[runs],
            q0=self.q0[runs],
        )

    @property
    def num_groups(self) -> int:
        """The number of groups (distinct initial states)."""
        return len(self.groups)

    def deterministic(self, method: str = "Radau") -> SimpleNamespace:
        """
        The deterministic approximation (EDM-PDM) from the initial state of each run, sampled at the same times.

        The approximation is computed with ``Simulator.integrate_edm_pdm`` once per group (distinct initial state),
        and cached.

        Args:
            method (str): The integration method. Defaults to 'Radau'.

        Returns:
            SimpleNamespace: Fields ``t`` (shape ``(K,)``), ``x`` (shape ``(M, n, K)``), ``q`` (shape ``(M, d, K)``),
            and ``p`` (shape ``(M, n, K)``), with the same layout as the ensemble.
        """
        if method not in self._deterministic:
            x = np.empty_like(self.x)
            q = np.empty_like(self.q)
            p = np.empty_like(self.p)
            for runs in self.groups:
                r = runs[0]
                out = self.simulator.integrate_edm_pdm(
                    t_span=(0.0, self.t[-1]),
                    x0=self.x0[r],
                    q0=self.q0[r],
                    t_eval=self.t,
                    method=method,
                )
                x[runs], q[runs], p[runs] = out.x, out.q, out.p
            self._deterministic[method] = SimpleNamespace(t=self.t, x=x, q=q, p=p)
        return self._deterministic[method]
