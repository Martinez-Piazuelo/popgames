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
    Independent finite-agent simulations from the same initial state, sampled at common times.

    Ensembles are created by ``Simulator.run_ensemble``. The arrays have one leading axis for the runs, followed by the
    layout of ``Simulator.log``:

    * ``ens.x[r]`` (shape ``(n, K)``) is run ``r``, with ``ens.x[r, i]`` the trajectory of strategy ``i``.
    * ``ens.x[:, i]`` (shape ``(M, K)``) holds the trajectories of strategy ``i`` in all runs.
    * ``ens.x[:, :, k]`` (shape ``(M, n)``) holds the strategic distributions of all runs at time ``ens.t[k]``.

    Attributes:
        t (np.ndarray): The sampling times, of shape ``(K,)``.
        x (np.ndarray): The strategic distributions, of shape ``(M, n, K)``.
        q (np.ndarray): The states of the PDM, of shape ``(M, d, K)``.
        p (np.ndarray): The payoffs, of shape ``(M, n, K)``.
        seeds (np.ndarray): The seeds of the runs, of shape ``(M,)``. Run ``r`` can be reproduced with
            ``sim.reset(ens.x0, ens.q0, seed=int(ens.seeds[r]))`` followed by ``sim.run(T_sim)``.
        x0 (np.ndarray): The initial strategic distribution of the runs, of shape ``(n, 1)``.
        q0 (np.ndarray): The initial state of the PDM of the runs, of shape ``(d, 1)``.
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
        The mean over the runs at each sampling time.

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
        The quantiles over the runs at each sampling time.

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

    def deterministic(self, method: str = "Radau") -> SimpleNamespace:
        """
        The deterministic approximation (EDM-PDM) from the initial state of the runs, sampled at the same times.

        The result is computed with ``Simulator.integrate_edm_pdm`` and cached.

        Args:
            method (str): The integration method. Defaults to 'Radau'.

        Returns:
            SimpleNamespace: Fields ``t`` (shape ``(K,)``), ``x`` (shape ``(n, K)``), ``q`` (shape ``(d, K)``), and
            ``p`` (shape ``(n, K)``).
        """
        if method not in self._deterministic:
            self._deterministic[method] = self.simulator.integrate_edm_pdm(
                t_span=(0.0, self.t[-1]),
                x0=self.x0,
                q0=self.q0,
                t_eval=self.t,
                method=method,
            )
        return self._deterministic[method]
