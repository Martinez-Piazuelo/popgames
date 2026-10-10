from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import numpy as np

from popgames.ensemble import EnsembleResult
from popgames.payoff_mechanism import PayoffMechanism
from popgames.population_game import PopulationGame, SinglePopulationGame
from popgames.revision_process import PoissonRevisionProcess
from popgames.revision_protocol import Smith, Softmax
from popgames.simulator import Simulator
from popgames.utilities import sample_initial_states

NUMBA_AVAILABLE = importlib.util.find_spec("numba") is not None

_RPS = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
_RPS_X0 = np.array([[0.5], [0.3], [0.2]])


def _rps_fitness(x: np.ndarray) -> np.ndarray:
    return _RPS @ x


def _make_rps_simulator(num_agents: int = 100, **kwargs) -> Simulator:
    return Simulator(
        population_game=SinglePopulationGame(
            num_strategies=3, fitness_function=_rps_fitness
        ),
        payoff_mechanism=PayoffMechanism(h_map=_rps_fitness, n=3),
        revision_processes=PoissonRevisionProcess(
            Poisson_clock_rate=1.0, revision_protocol=Smith(scale=0.25)
        ),
        num_agents=num_agents,
        **kwargs,
    )


def _make_multi_population_pdm_simulator(**kwargs) -> Simulator:
    """Two populations (2 and 3 strategies) and a PDM with d=1."""
    A, b = np.array([[0.5, 0.5, 0.0, 0.0, 0.0]]), np.array([[0.2]])

    def h_map(q: np.ndarray, x: np.ndarray) -> np.ndarray:
        return -x - A.T @ q

    def w_map(q: np.ndarray, x: np.ndarray) -> np.ndarray:
        return A @ x - b

    return Simulator(
        population_game=PopulationGame(
            num_populations=2,
            num_strategies=[2, 3],
            fitness_function=lambda x: -x,
            masses=[1.0, 2.0],
        ),
        payoff_mechanism=PayoffMechanism(h_map=h_map, w_map=w_map, n=5, d=1),
        revision_processes=[
            PoissonRevisionProcess(1.0, Softmax(eta=0.1)),
            PoissonRevisionProcess(2.0, Smith(scale=0.1)),
        ],
        num_agents=[20, 30],
        **kwargs,
    )


def _hold(log, t_eval: np.ndarray) -> np.ndarray:
    """State of a log after the last entry at or before each time."""
    return log.x[:, np.searchsorted(log.t, t_eval, side="right") - 1]


class TestRunEnsemble(unittest.TestCase):
    def test_shapes_and_default_sampling_times(self) -> None:
        sim = _make_rps_simulator(seed=0)
        sim.reset(x0=_RPS_X0)
        ens = sim.run_ensemble(T_sim=4, num_runs=5, seed=1)

        self.assertIsInstance(ens, EnsembleResult)
        np.testing.assert_allclose(ens.t, np.linspace(0, 4, 201))
        self.assertEqual(ens.x.shape, (5, 3, 201))
        self.assertEqual(ens.q.shape, (5, 0, 201))
        self.assertEqual(ens.p.shape, (5, 3, 201))
        self.assertEqual(ens.seeds.shape, (5,))
        self.assertEqual(len(ens), 5)
        np.testing.assert_allclose(ens.x.sum(axis=1), 1.0)
        np.testing.assert_allclose(ens.p, np.einsum("ij,mjk->mik", _RPS, ens.x))

    def test_runs_start_from_the_state_of_the_last_reset(self) -> None:
        sim = _make_rps_simulator(seed=0)
        sim.reset(x0=_RPS_X0)
        sim.run(T_sim=2)  # The default initial state is still the one of the reset
        ens = sim.run_ensemble(T_sim=1, num_runs=3, seed=1)
        np.testing.assert_allclose(ens.x[:, :, 0], np.tile(_RPS_X0.T, (3, 1)))
        np.testing.assert_allclose(ens.x0, np.tile(_RPS_X0, (3, 1, 1)))

        x0 = np.array([[0.2], [0.2], [0.6]])
        ens = sim.run_ensemble(T_sim=1, num_runs=3, seed=1, x0=x0)
        np.testing.assert_allclose(ens.x[:, :, 0], np.tile(x0.T, (3, 1)))

    def test_runs_are_independent(self) -> None:
        sim = _make_rps_simulator(seed=0)
        sim.reset(x0=_RPS_X0)
        ens = sim.run_ensemble(T_sim=4, num_runs=4, seed=1)
        self.assertEqual(len(set(ens.seeds.tolist())), 4)
        for r in range(1, 4):
            self.assertFalse(np.array_equal(ens.x[0], ens.x[r]))

    def test_same_seed_gives_the_same_ensemble(self) -> None:
        sim = _make_rps_simulator(seed=0)
        sim.reset(x0=_RPS_X0)
        ens_1 = sim.run_ensemble(T_sim=3, num_runs=3, seed=7)
        ens_2 = sim.run_ensemble(T_sim=3, num_runs=3, seed=7)
        ens_3 = sim.run_ensemble(T_sim=3, num_runs=3, seed=8)
        np.testing.assert_array_equal(ens_1.x, ens_2.x)
        np.testing.assert_array_equal(ens_1.seeds, ens_2.seeds)
        self.assertFalse(np.array_equal(ens_1.x, ens_3.x))

        ens_4 = sim.run_ensemble(T_sim=3, num_runs=3, seed=np.random.default_rng(7))
        ens_5 = sim.run_ensemble(T_sim=3, num_runs=3, seed=np.random.default_rng(7))
        np.testing.assert_array_equal(ens_4.x, ens_5.x)

    def test_without_seed_uses_the_simulator_random_state(self) -> None:
        sim = _make_rps_simulator()  # NumPy's global random state
        sim.reset(x0=_RPS_X0)
        np.random.seed(3)
        ens_1 = sim.run_ensemble(T_sim=2, num_runs=3)
        np.random.seed(3)
        ens_2 = sim.run_ensemble(T_sim=2, num_runs=3)
        np.testing.assert_array_equal(ens_1.x, ens_2.x)

        sim = _make_rps_simulator(seed=5)
        sim.reset(x0=_RPS_X0)
        ens_3 = sim.run_ensemble(T_sim=2, num_runs=3)
        ens_4 = sim.run_ensemble(T_sim=2, num_runs=3)  # The generator has advanced
        self.assertFalse(np.array_equal(ens_3.x, ens_4.x))

    def test_simulator_random_generator_is_restored(self) -> None:
        sim = _make_rps_simulator(seed=0)
        rng = sim._rng
        sim.reset(x0=_RPS_X0)
        sim.run_ensemble(T_sim=1, num_runs=2, seed=1)
        self.assertIs(sim._rng, rng)

        sim = _make_rps_simulator()
        sim.run_ensemble(T_sim=1, num_runs=2, seed=1)
        self.assertIs(sim._rng, np.random)

    def test_each_run_can_be_reproduced_from_its_seed(self) -> None:
        for fast_path in [True, False]:
            with self.subTest(fast_path=fast_path):
                sim = _make_rps_simulator(20, fast_path=fast_path)
                sim.reset(x0=_RPS_X0)
                ens = sim.run_ensemble(T_sim=3, num_runs=3, seed=2)
                # The simulator holds the last run
                np.testing.assert_array_equal(_hold(sim.log, ens.t), ens.x[-1])

                sim.reset(ens.x0[1], ens.q0[1], seed=int(ens.seeds[1]))
                sim.run(T_sim=3)
                np.testing.assert_array_equal(_hold(sim.log, ens.t), ens.x[1])

    def test_sampling_holds_the_state_of_the_last_event(self) -> None:
        sim = _make_rps_simulator(seed=0)
        sim.reset(x0=_RPS_X0)
        ens = sim.run_ensemble(T_sim=2, num_runs=1, seed=4)
        sim.reset(ens.x0[0], ens.q0[0], seed=int(ens.seeds[0]))
        log = sim.run(T_sim=2)

        # Event times, and times halfway between consecutive events
        k = np.array([3, 10, 20])
        t_eval = np.sort(np.concatenate([log.t[k], (log.t[k] + log.t[k + 1]) / 2]))
        ens = sim.run_ensemble(T_sim=2, num_runs=1, seed=4, t_eval=t_eval)
        expected = log.x[:, np.repeat(k, 2)]
        np.testing.assert_array_equal(ens.x[0], expected)

    def test_log_interval_is_used_within_each_run(self) -> None:
        sim = _make_rps_simulator(seed=0)
        sim.reset(x0=_RPS_X0)
        sim.run_ensemble(T_sim=4, num_runs=2, seed=1, log_interval=1.0)
        self.assertLessEqual(sim.log.t.size, 6)

    def test_per_agent_path(self) -> None:
        sim = _make_rps_simulator(10, fast_path=False)
        sim.reset(x0=_RPS_X0)
        ens = sim.run_ensemble(T_sim=2, num_runs=2, seed=0, t_eval=[0.0, 1.0, 2.0])
        self.assertEqual(ens.x.shape, (2, 3, 3))
        np.testing.assert_allclose(ens.x.sum(axis=1), 1.0)

    def test_multi_population_with_pdm(self) -> None:
        sim = _make_multi_population_pdm_simulator()
        sim.reset(x0=np.array([[1.0], [0.0], [2.0], [0.0], [0.0]]), q0=np.ones((1, 1)))
        ens = sim.run_ensemble(
            T_sim=2, num_runs=3, seed=0, t_eval=np.linspace(0, 2, 11)
        )
        self.assertEqual(ens.x.shape, (3, 5, 11))
        self.assertEqual(ens.q.shape, (3, 1, 11))
        np.testing.assert_allclose(ens.x[:, :2].sum(axis=1), 1.0)
        np.testing.assert_allclose(ens.x[:, 2:].sum(axis=1), 2.0)
        np.testing.assert_allclose(ens.q[:, 0, 0], 1.0)
        np.testing.assert_allclose(ens.q0, np.ones((3, 1, 1)))

    def test_mean_tracks_the_edm(self) -> None:
        sim = _make_rps_simulator(1000, seed=0)
        sim.reset(x0=_RPS_X0)
        ens = sim.run_ensemble(T_sim=3, num_runs=20, seed=0)
        np.testing.assert_allclose(
            ens.mean().x, ens.deterministic().x.mean(axis=0), atol=0.01
        )

    def test_invalid_arguments(self) -> None:
        sim = _make_rps_simulator(seed=0)
        with self.assertRaises(TypeError):
            sim.run_ensemble(T_sim=1.5, num_runs=2)
        with self.assertRaises(TypeError):
            sim.run_ensemble(T_sim=1, num_runs=2.0)
        with self.assertRaises(ValueError):
            sim.run_ensemble(T_sim=1, num_runs=0)
        for t_eval in [[0.0, 2.0], [-0.1, 0.5], [0.5, 0.2], [[0.0, 0.5]], []]:
            with self.subTest(t_eval=t_eval), self.assertRaises(ValueError):
                sim.run_ensemble(T_sim=1, num_runs=2, t_eval=t_eval)


class TestResetSeedAndRounding(unittest.TestCase):
    def test_reset_with_seed_replaces_the_generator(self) -> None:
        sim = _make_rps_simulator()
        outs = []
        for _ in range(2):
            sim.reset(x0=_RPS_X0, seed=3)
            outs.append(sim.run(T_sim=2).x)
        np.testing.assert_array_equal(outs[0], outs[1])
        self.assertIsInstance(sim._rng, np.random.Generator)

    def test_reset_counts_are_robust_to_rounding(self) -> None:
        game = SinglePopulationGame(num_strategies=2, fitness_function=lambda x: -x)
        sim = Simulator(
            population_game=game,
            payoff_mechanism=PayoffMechanism(h_map=lambda x: -x, n=2),
            revision_processes=PoissonRevisionProcess(1.0, Smith(scale=0.5)),
            num_agents=100,
        )
        sim.reset(x0=np.array([[0.71], [0.29]]))  # 100 * 0.29 = 28.999999999999996
        np.testing.assert_array_equal(sim._counts[0], [71, 29])


class TestEnsembleResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sim = _make_rps_simulator(seed=0)
        sim.reset(x0=_RPS_X0)
        cls.ens = sim.run_ensemble(
            T_sim=4, num_runs=6, seed=0, t_eval=np.linspace(0, 4, 9)
        )

    def test_single_run_has_the_log_layout(self) -> None:
        run = self.ens[2]
        self.assertIs(run.t, self.ens.t)
        np.testing.assert_array_equal(run.x, self.ens.x[2])
        self.assertEqual(run.q.shape, (0, 9))
        self.assertEqual(len(list(self.ens)), 6)

    def test_mean_and_quantiles(self) -> None:
        np.testing.assert_allclose(self.ens.mean().x, self.ens.x.mean(axis=0))
        median = self.ens.quantile(0.5)
        self.assertEqual(median.x.shape, (3, 9))
        np.testing.assert_allclose(median.x, np.median(self.ens.x, axis=0))
        bands = self.ens.quantile([0.1, 0.9])
        self.assertEqual(bands.x.shape, (2, 3, 9))
        self.assertEqual(bands.q.shape, (2, 0, 9))
        self.assertTrue(np.all(bands.x[0] <= bands.x[1]))

    def test_states_at_a_given_time(self) -> None:
        snapshot = self.ens.at(1.7)  # Last sampling time at or before 1.7 is 1.5
        self.assertEqual(snapshot.t, 1.5)
        np.testing.assert_array_equal(snapshot.x, self.ens.x[:, :, 3])
        self.assertEqual(snapshot.q.shape, (6, 0))
        final = self.ens.final()
        self.assertEqual(final.t, 4.0)
        np.testing.assert_array_equal(final.x, self.ens.x[:, :, -1])
        with self.assertRaises(ValueError):
            self.ens.at(-1.0)

    def test_deterministic_approximation_is_cached(self) -> None:
        det = self.ens.deterministic()
        self.assertIs(self.ens.deterministic(), det)
        np.testing.assert_allclose(det.t, self.ens.t)
        self.assertEqual(det.x.shape, (6, 3, 9))
        self.assertEqual(det.q.shape, (6, 0, 9))
        np.testing.assert_allclose(det.x[:, :, [0]], self.ens.x0)

    def test_single_initial_state_forms_one_group(self) -> None:
        self.assertEqual(self.ens.num_groups, 1)
        np.testing.assert_array_equal(self.ens.groups[0], np.arange(6))
        np.testing.assert_array_equal(self.ens.group_index, np.zeros(6))

    def test_subset(self) -> None:
        sub = self.ens.subset([4, 1])
        self.assertEqual(len(sub), 2)
        np.testing.assert_array_equal(sub.x, self.ens.x[[4, 1]])
        np.testing.assert_array_equal(sub.seeds, self.ens.seeds[[4, 1]])
        np.testing.assert_array_equal(sub.x0, self.ens.x0[[4, 1]])
        mask = np.zeros(6, dtype=bool)
        mask[2] = True
        np.testing.assert_array_equal(self.ens.subset(mask).x, self.ens.x[[2]])
        with self.assertRaises(ValueError):
            self.ens.subset([])


class TestEnsemblePlots(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sim = _make_rps_simulator(seed=0)
        sim.reset(x0=_RPS_X0)
        cls.ens = sim.run_ensemble(
            T_sim=2, num_runs=4, seed=0, t_eval=np.linspace(0, 2, 11)
        )
        sim = _make_multi_population_pdm_simulator()
        sim.reset(x0=np.array([[1.0], [0.0], [2.0], [0.0], [0.0]]))
        cls.ens_multi = sim.run_ensemble(
            T_sim=2, num_runs=4, seed=0, t_eval=np.linspace(0, 2, 11)
        )

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _files(self) -> list[str]:
        return sorted(os.listdir(self._tmp.name))

    def _path(self, name: str) -> str:
        return os.path.join(self._tmp.name, name)

    def test_bands(self) -> None:
        self.ens.plot(
            plot_type="bands",
            plot_deterministic_approximation=True,
            filename=self._path("bands.png"),
            show=False,
        )
        self.assertEqual(self._files(), ["bands_p_1.png", "bands_x_1.png"])

    def test_bands_multi_population_with_pdm(self) -> None:
        self.ens_multi.plot(
            plot_type="bands",
            variables=("x", "q"),
            bands=[(0.1, 0.9)],
            plot_median=False,
            filename=self._path("bands.pdf"),
            show=False,
        )
        self.assertEqual(
            self._files(), ["bands_q.pdf", "bands_x_1.pdf", "bands_x_2.pdf"]
        )

    def test_kpi(self) -> None:
        self.ens.plot(
            plot_type="kpi",
            plot_deterministic_approximation=True,
            filename=self._path("kpi.png"),
            show=False,
        )
        self.ens_multi.plot(
            plot_type="kpi",
            kpi_function=lambda run: run.q[0],
            filename=self._path("kpi_q.png"),
            show=False,
        )
        self.assertEqual(self._files(), ["kpi.png", "kpi_q.png"])

    def test_ternary(self) -> None:
        self.ens.plot(
            plot_type="ternary",
            plot_mean=True,
            plot_deterministic_approximation=True,
            filename=self._path("ternary.png"),
            show=False,
        )
        self.assertEqual(self._files(), ["ternary.png"])

    def test_ternary_skips_populations_without_3_strategies(self) -> None:
        with self.assertLogs("popgames.plotting.ensemble_plotters", level="WARNING"):
            self.ens_multi.plot(
                plot_type="ternary",
                plot_gne=False,
                filename=self._path("ternary.png"),
                show=False,
            )
        self.assertEqual(self._files(), ["ternary_pop_2.png"])

    def test_final(self) -> None:
        self.ens.plot(
            plot_type="final",
            plot_deterministic_approximation=True,
            filename=self._path("final.png"),
            show=False,
        )
        self.ens_multi.plot(
            plot_type="final",
            t=1.0,
            plot_deterministic_approximation=True,
            plot_gne=False,
            filename=self._path("final_multi.png"),
            show=False,
        )
        self.assertEqual(
            self._files(),
            ["final.png", "final_multi_pop_1.png", "final_multi_pop_2.png"],
        )

    def test_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            self.ens.plot(plot_type="does_not_exist")
        with self.assertRaises(ValueError):
            self.ens.plot(plot_type="bands", variables=("y",), show=False)
        with self.assertRaises(TypeError):
            self.ens.plot(plot_type="bands", unknown_option=1, show=False)


_X0S = np.array([[[0.5], [0.3], [0.2]], [[0.2], [0.2], [0.6]], [[0.1], [0.8], [0.1]]])


class TestRunEnsemblePerRunArguments(unittest.TestCase):
    def test_per_run_seeds_are_used_as_given(self) -> None:
        sim = _make_rps_simulator(seed=0)
        sim.reset(x0=_RPS_X0)
        ens = sim.run_ensemble(T_sim=2, seed=[5, 9, 5])
        self.assertEqual(len(ens), 3)
        np.testing.assert_array_equal(ens.seeds, [5, 9, 5])
        np.testing.assert_array_equal(ens.x[0], ens.x[2])
        self.assertFalse(np.array_equal(ens.x[0], ens.x[1]))

        sim.reset(_RPS_X0, seed=9)
        sim.run(T_sim=2)
        np.testing.assert_array_equal(_hold(sim.log, ens.t), ens.x[1])

    def test_per_run_initial_states(self) -> None:
        sim = _make_rps_simulator(seed=0)
        ens = sim.run_ensemble(T_sim=2, x0=_X0S, seed=0)
        self.assertEqual(len(ens), 3)  # Inferred from x0
        np.testing.assert_allclose(ens.x[:, :, [0]], _X0S)
        np.testing.assert_allclose(ens.x0, _X0S)
        self.assertEqual(ens.num_groups, 3)
        np.testing.assert_array_equal(ens.group_index, [0, 1, 2])

    def test_initial_states_are_stored_as_rounded_by_reset(self) -> None:
        sim = _make_rps_simulator(100, seed=0)
        x0 = np.array([[1 / 3], [1 / 3], [1 / 3]])
        ens = sim.run_ensemble(T_sim=1, num_runs=2, x0=x0, seed=0)
        np.testing.assert_allclose(ens.x0[0], [[0.34], [0.33], [0.33]])
        np.testing.assert_allclose(ens.x[:, :, [0]], ens.x0)

    def test_same_seed_from_different_initial_states_uses_common_random_numbers(
        self,
    ) -> None:
        sim = _make_rps_simulator(seed=0)
        ens = sim.run_ensemble(T_sim=2, x0=_X0S, seed=[7] * 3)
        logs = []
        for r in range(2):
            sim.reset(ens.x0[r], ens.q0[r], seed=int(ens.seeds[r]))
            logs.append(sim.run(T_sim=2))
        # Same random numbers: the revision events happen at the same times
        np.testing.assert_array_equal(logs[0].t, logs[1].t)
        self.assertFalse(np.array_equal(logs[0].x, logs[1].x))

    def test_groups_follow_the_order_of_first_appearance(self) -> None:
        sim = _make_rps_simulator(seed=0)
        ens = sim.run_ensemble(T_sim=1, x0=np.tile(_X0S[[2, 0]], (3, 1, 1)), seed=0)
        self.assertEqual(ens.num_groups, 2)
        np.testing.assert_array_equal(ens.group_index, [0, 1, 0, 1, 0, 1])
        np.testing.assert_array_equal(ens.groups[0], [0, 2, 4])
        np.testing.assert_allclose(ens.x0[ens.groups[0][0]], _X0S[2])

        ens = sim.run_ensemble(T_sim=1, x0=np.repeat(_X0S, 2, axis=0), seed=0)
        np.testing.assert_array_equal(ens.group_index, [0, 0, 1, 1, 2, 2])
        sub = ens.subset(ens.groups[1])
        self.assertEqual(sub.num_groups, 1)
        np.testing.assert_allclose(sub.x0, np.tile(_X0S[1], (2, 1, 1)))

    def test_per_run_q0_with_pdm(self) -> None:
        sim = _make_multi_population_pdm_simulator()
        x0 = np.array([[1.0], [0.0], [2.0], [0.0], [0.0]])
        q0s = np.array([[[0.0]], [[1.0]], [[0.0]]])
        ens = sim.run_ensemble(T_sim=1, x0=x0, q0=q0s, seed=0)
        self.assertEqual(len(ens), 3)
        np.testing.assert_allclose(ens.q[:, :, 0], q0s[:, :, 0])
        np.testing.assert_array_equal(ens.group_index, [0, 1, 0])

    def test_deterministic_approximation_is_integrated_once_per_group(self) -> None:
        sim = _make_rps_simulator(seed=0)
        ens = sim.run_ensemble(
            T_sim=2, x0=np.repeat(_X0S, 2, axis=0), seed=0, t_eval=[0.0, 1.0, 2.0]
        )
        with patch.object(
            sim, "integrate_edm_pdm", wraps=sim.integrate_edm_pdm
        ) as integrate:
            det = ens.deterministic()
        self.assertEqual(integrate.call_count, 3)
        self.assertEqual(det.x.shape, (6, 3, 3))
        np.testing.assert_allclose(det.x[:, :, [0]], ens.x0)
        np.testing.assert_array_equal(det.x[0], det.x[1])
        expected = sim.integrate_edm_pdm(
            t_span=(0, 2), x0=_X0S[2], t_eval=[0.0, 1.0, 2.0]
        )
        np.testing.assert_allclose(det.x[5], expected.x)

    def test_number_of_runs_must_be_consistent(self) -> None:
        sim = _make_rps_simulator(seed=0)
        with self.assertRaises(ValueError):
            sim.run_ensemble(T_sim=1)  # No way to know the number of runs
        with self.assertRaises(ValueError):
            sim.run_ensemble(T_sim=1, num_runs=2, x0=_X0S)
        with self.assertRaises(ValueError):
            sim.run_ensemble(T_sim=1, x0=_X0S, seed=[1, 2])
        sim.run_ensemble(T_sim=1, num_runs=3, x0=_X0S, seed=[1, 2, 3])

    def test_invalid_per_run_arguments(self) -> None:
        sim = _make_rps_simulator(seed=0)
        for x0 in [np.ones((3, 3)) / 3, np.ones((2, 2, 1)), np.empty((0, 3, 1))]:
            with self.subTest(x0=x0.shape), self.assertRaises(ValueError):
                sim.run_ensemble(T_sim=1, num_runs=2, x0=x0)
        for seed in [[], [-1, 2], [0.5, 1.5], [[1, 2]]]:
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                sim.run_ensemble(T_sim=1, seed=seed)
        with self.assertRaises(ValueError):
            sim.run_ensemble(T_sim=1, num_runs=2, q0=np.zeros((2, 1, 1)))


class TestSampleInitialStates(unittest.TestCase):
    def test_shape_masses_and_support(self) -> None:
        game = PopulationGame(
            num_populations=2,
            num_strategies=[2, 3],
            fitness_function=lambda x: -x,
            masses=[1.0, 2.0],
        )
        x0s = sample_initial_states(game, num=50, seed=0)
        self.assertEqual(x0s.shape, (50, 5, 1))
        self.assertTrue(np.all(x0s >= 0))
        np.testing.assert_allclose(x0s[:, :2].sum(axis=1), 1.0)
        np.testing.assert_allclose(x0s[:, 2:].sum(axis=1), 2.0)

    def test_uniform_on_the_simplex(self) -> None:
        game = SinglePopulationGame(num_strategies=3, fitness_function=lambda x: -x)
        x0s = sample_initial_states(game, num=4000, seed=1)
        # Uniform on the simplex: each coordinate is Beta(1, 2), with mean 1/3 and variance 1/18
        np.testing.assert_allclose(x0s.mean(axis=0).ravel(), 1 / 3, atol=0.02)
        np.testing.assert_allclose(x0s.var(axis=0).ravel(), 1 / 18, atol=0.005)

    def test_reproducibility(self) -> None:
        game = SinglePopulationGame(num_strategies=3, fitness_function=lambda x: -x)
        np.testing.assert_array_equal(
            sample_initial_states(game, num=3, seed=2),
            sample_initial_states(game, num=3, seed=2),
        )
        np.random.seed(4)
        a = sample_initial_states(game, num=3)
        np.random.seed(4)
        np.testing.assert_array_equal(a, sample_initial_states(game, num=3))
        with self.assertRaises(ValueError):
            sample_initial_states(game, num=0)

    def test_usable_as_initial_states_of_an_ensemble(self) -> None:
        sim = _make_rps_simulator(seed=0)
        x0s = sample_initial_states(sim.population_game, num=4, seed=0)
        ens = sim.run_ensemble(T_sim=1, x0=x0s, seed=[3] * 4)
        self.assertEqual(ens.num_groups, 4)
        np.testing.assert_allclose(ens.x0, x0s, atol=1 / 100)


class TestEnsemblePlotsWithSeveralInitialStates(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sim = _make_rps_simulator(seed=0)
        cls.ens = sim.run_ensemble(
            T_sim=2,
            x0=np.repeat(_X0S, 3, axis=0),
            seed=0,
            t_eval=np.linspace(0, 2, 11),
        )
        sim = _make_multi_population_pdm_simulator()
        x0s = np.array(
            [[[1.0], [0.0], [2.0], [0.0], [0.0]], [[0.0], [1.0], [0.0], [2.0], [0.0]]]
        )
        cls.ens_multi = sim.run_ensemble(
            T_sim=2, x0=np.repeat(x0s, 2, axis=0), seed=0, t_eval=np.linspace(0, 2, 11)
        )

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _files(self) -> list[str]:
        return sorted(os.listdir(self._tmp.name))

    def _path(self, name: str) -> str:
        return os.path.join(self._tmp.name, name)

    def test_bands_one_set_of_figures_per_initial_state(self) -> None:
        self.ens.plot(
            plot_type="bands",
            variables=["x"],
            plot_deterministic_approximation=True,
            filename=self._path("bands.png"),
            show=False,
        )
        self.assertEqual(
            self._files(),
            ["bands_x_1_x0_1.png", "bands_x_1_x0_2.png", "bands_x_1_x0_3.png"],
        )

    def test_bands_of_a_single_initial_state(self) -> None:
        self.ens.plot(
            plot_type="bands",
            variables=["x"],
            initial_state=1,
            filename=self._path("bands.png"),
            show=False,
        )
        self.assertEqual(self._files(), ["bands_x_1_x0_2.png"])
        for initial_state in [3, -1, 1.0]:
            with self.subTest(initial_state=initial_state):
                with self.assertRaises(ValueError):
                    self.ens.plot(
                        plot_type="bands", initial_state=initial_state, show=False
                    )

    def test_kpi_one_band_per_initial_state(self) -> None:
        self.ens.plot(
            plot_type="kpi",
            plot_deterministic_approximation=True,
            filename=self._path("kpi.png"),
            show=False,
        )
        self.assertEqual(self._files(), ["kpi.png"])

    def test_ternary_phase_portrait(self) -> None:
        self.ens.plot(
            plot_type="ternary",
            plot_mean=True,
            plot_deterministic_approximation=True,
            filename=self._path("ternary.png"),
            show=False,
        )
        self.assertEqual(self._files(), ["ternary.png"])

    def test_final_colored_by_initial_state(self) -> None:
        self.ens.plot(
            plot_type="final",
            plot_deterministic_approximation=True,
            filename=self._path("final.png"),
            show=False,
        )
        self.ens_multi.plot(
            plot_type="final",
            plot_deterministic_approximation=True,
            plot_gne=False,
            filename=self._path("final_multi.png"),
            show=False,
        )
        self.assertEqual(
            self._files(),
            ["final.png", "final_multi_pop_1_x_1.png", "final_multi_pop_2.png"],
        )

    def test_many_initial_states(self) -> None:
        sim = _make_rps_simulator(seed=0)
        x0s = sample_initial_states(sim.population_game, num=12, seed=0)
        ens = sim.run_ensemble(T_sim=1, x0=x0s, seed=0, t_eval=[0.0, 0.5, 1.0])
        self.assertEqual(ens.num_groups, 12)
        with self.assertLogs("popgames.plotting.ensemble_plotters", level="WARNING"):
            ens.plot(plot_type="kpi", plot_deterministic_approximation=True, show=False)
        with self.assertLogs("popgames.plotting.ensemble_plotters", level="WARNING"):
            ens.plot(plot_type="bands", variables=["x"], show=False)
        ens.plot(plot_type="ternary", plot_gne=False, show=False)
        ens.plot(plot_type="final", plot_gne=False, show=False)


@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class TestRunEnsembleNumba(unittest.TestCase):
    def test_numba_backend(self) -> None:
        sim = _make_rps_simulator(100, backend="numba")
        self.assertEqual(sim.backend, "numba")
        sim.reset(x0=_RPS_X0)
        ens = sim.run_ensemble(T_sim=3, num_runs=3, seed=0)
        self.assertEqual(ens.x.shape, (3, 3, 201))
        np.testing.assert_allclose(ens.x.sum(axis=1), 1.0)
        np.testing.assert_array_equal(
            sim.run_ensemble(T_sim=3, num_runs=3, seed=0).x, ens.x
        )

        sim.reset(ens.x0[1], ens.q0[1], seed=int(ens.seeds[1]))
        sim.run(T_sim=3)
        np.testing.assert_array_equal(_hold(sim.log, ens.t), ens.x[1])

    def test_numba_backend_with_pdm(self) -> None:
        sim = _make_multi_population_pdm_simulator(backend="numba")
        self.assertEqual(sim.backend, "numba")
        sim.reset(x0=np.array([[1.0], [0.0], [2.0], [0.0], [0.0]]))
        ens = sim.run_ensemble(T_sim=2, num_runs=2, seed=0)
        self.assertEqual(ens.q.shape, (2, 1, 201))

    def test_numba_backend_with_per_run_arguments(self) -> None:
        sim = _make_rps_simulator(100, backend="numba")
        ens = sim.run_ensemble(T_sim=2, x0=_X0S, seed=[7] * 3)
        np.testing.assert_allclose(ens.x[:, :, [0]], _X0S)
        sim.reset(ens.x0[2], ens.q0[2], seed=7)
        sim.run(T_sim=2)
        np.testing.assert_array_equal(_hold(sim.log, ens.t), ens.x[2])


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
