from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from popgames.payoff_mechanism import PayoffMechanism
from popgames.population_game import PopulationGame, SinglePopulationGame
from popgames.revision_process import PoissonRevisionProcess
from popgames.revision_protocol import Smith
from popgames.simulator import Simulator

_RPS = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)


def _rps_fitness(x: np.ndarray) -> np.ndarray:
    return _RPS @ x


def _make_rps_simulator() -> Simulator:
    sim = Simulator(
        population_game=SinglePopulationGame(
            num_strategies=3, fitness_function=_rps_fitness
        ),
        payoff_mechanism=PayoffMechanism(h_map=_rps_fitness, n=3),
        revision_processes=PoissonRevisionProcess(
            Poisson_clock_rate=1.0, revision_protocol=Smith(scale=0.25)
        ),
        num_agents=50,
        seed=0,
    )
    sim.reset(x0=np.array([[0.5], [0.3], [0.2]]))
    return sim


def _make_two_population_simulator() -> Simulator:
    """Two populations with 3 strategies each."""

    def fitness(x: np.ndarray) -> np.ndarray:
        return -x

    sim = Simulator(
        population_game=PopulationGame(
            num_populations=2,
            num_strategies=[3, 3],
            masses=[1.0, 1.0],
            fitness_function=fitness,
        ),
        payoff_mechanism=PayoffMechanism(h_map=fitness, n=6),
        revision_processes=[
            PoissonRevisionProcess(
                Poisson_clock_rate=1.0, revision_protocol=Smith(scale=0.25)
            )
        ]
        * 2,
        num_agents=[20, 20],
        seed=0,
    )
    sim.reset(x0=np.array([[0.6], [0.2], [0.2], [0.1], [0.1], [0.8]]))
    return sim


class TestAxes(unittest.TestCase):
    """Plotting on axes given by the caller (the ``ax`` option)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = _make_rps_simulator()
        cls.sim.run(T_sim=2)
        cls.ens = cls.sim.run_ensemble(
            T_sim=2, num_runs=3, seed=0, t_eval=np.linspace(0, 2, 11)
        )

    def setUp(self) -> None:
        plt.close("all")
        self.addCleanup(plt.close, "all")

    def _assert_drawn_in_place(self, fig: matplotlib.figure.Figure, axes: list) -> None:
        """No other figure was created, the caller's figure is still open, and every axes was drawn on."""
        self.assertEqual(plt.get_fignums(), [fig.number])
        for ax in axes:
            self.assertTrue(ax.lines or ax.collections, "Nothing drawn on an axes")

    def test_simulator_plots_draw_on_given_axes(self) -> None:
        fig, axes = plt.subplots(2, 4)
        axes = list(axes.flat)
        self.sim.plot(plot_type="kpi", ax=axes[0], show=True)
        self.sim.plot(plot_type="univariate", ax=axes[1:3], show=True)
        self.sim.plot(
            plot_type="ternary",
            ax=axes[3],
            plot_deterministic_approximation=True,
            potential_function=lambda x: np.array([[-np.sum(x**2)]]),
            show=True,
        )
        self._assert_drawn_in_place(fig, axes[:4])

        fig_split, axes_split = plt.subplots(2, 3)
        plt.close(fig)
        self.sim.plot(plot_type="univariate_split", ax=axes_split.flat, show=True)
        self._assert_drawn_in_place(fig_split, list(axes_split.flat))

    def test_ternary_multi_population_draws_one_axes_per_population(self) -> None:
        sim = _make_two_population_simulator()
        sim.run(T_sim=1)
        fig, axes = plt.subplots(1, 2)
        sim.plot(plot_type="ternary", ax=axes, show=False)
        self._assert_drawn_in_place(fig, list(axes))

    def test_ensemble_plots_draw_on_given_axes(self) -> None:
        fig, axes = plt.subplots(1, 5)
        self.ens.plot(plot_type="bands", ax=axes[0:2], show=True)
        self.ens.plot(plot_type="kpi", ax=axes[2], show=True)
        self.ens.plot(plot_type="ternary", ax=axes[3], show=True)
        self.ens.plot(plot_type="final", ax=axes[4], show=True)
        self._assert_drawn_in_place(fig, list(axes))

    def test_wrong_number_of_axes_raises(self) -> None:
        fig, axes = plt.subplots(1, 3)
        with self.assertRaises(ValueError):
            self.sim.plot(plot_type="univariate", ax=axes[0])
        with self.assertRaises(ValueError):
            self.ens.plot(plot_type="bands", ax=list(axes))

    def test_filename_with_axes_raises(self) -> None:
        _, ax = plt.subplots()
        with self.assertRaises(ValueError):
            self.sim.plot(plot_type="kpi", ax=ax, filename="kpi.png")
        with self.assertRaises(ValueError):
            self.ens.plot(plot_type="kpi", ax=ax, filename="kpi.png")


if __name__ == "__main__":
    unittest.main()
