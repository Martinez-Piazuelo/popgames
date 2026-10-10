from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from popgames.alarm_clock import Poisson
from popgames.payoff_mechanism import PayoffMechanism
from popgames.population_game import PopulationGame, SinglePopulationGame
from popgames.revision_process import PoissonRevisionProcess, RevisionProcessABC
from popgames.revision_protocol import Smith, Softmax
from popgames.simulator import Simulator


def _fitness_ok(x: np.ndarray) -> np.ndarray:
    # identity fitness with the correct shape
    return x


def _h_map_memoryless(x: np.ndarray) -> np.ndarray:
    # payoff equals state (shape-preserving)
    return x


class _DeterministicRevisionProcess(RevisionProcessABC):
    """
    Deterministic revision process for unit tests.

    - sample_next_revision_time(size): returns an array filled with self.fixed_dt
    - sample_next_strategy(p, x, i): returns (i+1) mod n
    - rhs_edm(x, p): returns zeros (n,1)
    """

    def __init__(self, n: int, fixed_dt: float = 0.1):
        super().__init__(
            alarm_clock=Poisson(rate=1.0), revision_protocol=Softmax(eta=1.0)
        )
        self.n = n
        self.fixed_dt = float(fixed_dt)

    def sample_next_revision_time(self, size: int) -> np.ndarray:
        return np.full((size,), self.fixed_dt, dtype=float)

    def sample_next_strategy(self, p: np.ndarray, x: np.ndarray, i: int) -> int:
        return int((i + 1) % self.n)

    def rhs_edm(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)


class TestSimulatorInit(unittest.TestCase):
    def test_init_single_population_wraps_revision_process_and_num_agents(self) -> None:
        game = PopulationGame(
            num_populations=1, num_strategies=[2], fitness_function=_fitness_ok
        )
        pm = PayoffMechanism(h_map=_h_map_memoryless, n=2, d=0)
        rp = _DeterministicRevisionProcess(n=2, fixed_dt=0.1)

        sim = Simulator(
            population_game=game,
            payoff_mechanism=pm,
            revision_processes=rp,
            num_agents=5,
        )
        self.assertEqual(sim.population_game.num_populations, 1)
        self.assertEqual(sim.num_agents, [5])
        self.assertEqual(len(sim.revision_processes), 1)
        self.assertIs(sim.revision_processes[0], rp)

    def test_init_dimension_mismatch_raises(self) -> None:
        game = PopulationGame(
            num_populations=1, num_strategies=[2], fitness_function=_fitness_ok
        )
        pm = PayoffMechanism(h_map=_h_map_memoryless, n=3, d=0)  # mismatch
        rp = _DeterministicRevisionProcess(n=2, fixed_dt=0.1)

        with self.assertRaises(AssertionError):
            Simulator(
                population_game=game,
                payoff_mechanism=pm,
                revision_processes=rp,
                num_agents=5,
            )

    def test_init_multi_population_validates_lists(self) -> None:
        game = PopulationGame(
            num_populations=2, num_strategies=[2, 3], fitness_function=_fitness_ok
        )
        pm = PayoffMechanism(h_map=_h_map_memoryless, n=5, d=0)
        rp0 = _DeterministicRevisionProcess(n=2, fixed_dt=0.1)
        rp1 = _DeterministicRevisionProcess(n=3, fixed_dt=0.2)

        sim = Simulator(
            population_game=game,
            payoff_mechanism=pm,
            revision_processes=[rp0, rp1],
            num_agents=[4, 7],
        )
        self.assertEqual(sim.num_agents, [4, 7])
        self.assertEqual(len(sim.revision_processes), 2)


class TestSimulatorReset(unittest.TestCase):
    def test_reset_with_x0_initializes_selected_strategies_deterministically(
        self,
    ) -> None:
        # single population, 2 strategies, 4 agents
        game = PopulationGame(
            num_populations=1, num_strategies=[2], fitness_function=_fitness_ok
        )
        pm = PayoffMechanism(h_map=_h_map_memoryless, n=2, d=0)
        rp = _DeterministicRevisionProcess(n=2, fixed_dt=0.1)

        sim = Simulator(
            population_game=game,
            payoff_mechanism=pm,
            revision_processes=rp,
            num_agents=4,
        )

        x0 = np.array([[0.25], [0.75]])  # mass=1.0
        sim.reset(x0=x0)

        sel = sim._selected_strategies[0]
        # Expected: floor(4*[0.25,0.75]) = [1,3], no remainder
        self.assertEqual(sel.shape, (4,))
        self.assertEqual(int(np.sum(sel == 0)), 1)
        self.assertEqual(int(np.sum(sel == 1)), 3)

        # x should match strategic distribution (scaled by masses)
        np.testing.assert_allclose(sim.x, x0, atol=1e-12)

    def test_reset_with_q0_validates_and_copies(self) -> None:
        game = PopulationGame(
            num_populations=1, num_strategies=[2], fitness_function=_fitness_ok
        )
        pm = PayoffMechanism(h_map=_h_map_memoryless, n=2, d=0)
        rp = _DeterministicRevisionProcess(n=2, fixed_dt=0.1)
        sim = Simulator(
            population_game=game,
            payoff_mechanism=pm,
            revision_processes=rp,
            num_agents=3,
        )

        q0 = np.zeros((0, 1))
        sim.reset(q0=q0)
        self.assertEqual(sim.q.shape, (0, 1))
        # deep copy check (shape is empty but still ok)
        self.assertIsNot(sim.q, q0)

    def test_reset_initializes_revision_times_per_population(self) -> None:
        game = PopulationGame(
            num_populations=1, num_strategies=[2], fitness_function=_fitness_ok
        )
        pm = PayoffMechanism(h_map=_h_map_memoryless, n=2, d=0)
        rp = _DeterministicRevisionProcess(n=2, fixed_dt=0.123456789123)
        sim = Simulator(
            population_game=game,
            payoff_mechanism=pm,
            revision_processes=rp,
            num_agents=5,
        )

        sim.reset()
        rt = sim._revision_times[0]
        self.assertEqual(rt.shape, (5,))
        # rounded to simulator precision
        self.assertTrue(np.allclose(rt, np.round(rt, sim._num_precision)))


class TestSimulatorCoreDynamics(unittest.TestCase):
    def test_get_strategic_distribution_matches_selected_strategies(self) -> None:
        game = PopulationGame(
            num_populations=1, num_strategies=[3], fitness_function=_fitness_ok
        )
        pm = PayoffMechanism(h_map=_h_map_memoryless, n=3, d=0)
        rp = _DeterministicRevisionProcess(n=3, fixed_dt=0.1)
        sim = Simulator(
            population_game=game,
            payoff_mechanism=pm,
            revision_processes=rp,
            num_agents=10,
        )

        # force a known selection: 2 of 0, 3 of 1, 5 of 2
        sim._selected_strategies = [np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])]
        x = sim._get_strategic_distribution()
        expected = np.array([[0.2], [0.3], [0.5]])
        np.testing.assert_allclose(x, expected, atol=1e-12)

    def test_microscopic_step_updates_time_log_and_state_when_strategies_change(
        self,
    ) -> None:
        # Choose x0 such that switching changes x
        game = PopulationGame(
            num_populations=1, num_strategies=[2], fitness_function=_fitness_ok
        )
        pm = PayoffMechanism(h_map=_h_map_memoryless, n=2, d=0)
        rp = _DeterministicRevisionProcess(n=2, fixed_dt=0.1)
        sim = Simulator(
            population_game=game,
            payoff_mechanism=pm,
            revision_processes=rp,
            num_agents=4,
        )

        x0 = np.array([[0.25], [0.75]])  # selected = [0,1,1,1]
        sim.reset(x0=x0)

        t0 = sim.t
        log_len0 = len(sim.log.t)

        sim._microscopic_step(time_step=0.1)

        # time advances
        self.assertTrue(np.isclose(sim.t, t0 + 0.1, atol=1e-12))
        # log updated
        self.assertEqual(len(sim.log.t), log_len0 + 1)

        # all agents revise at dt=0.1 and switch to opposite => x becomes [0.75, 0.25]
        expected_x = np.array([[0.75], [0.25]])
        np.testing.assert_allclose(sim.x, expected_x, atol=1e-12)

    def test_run_validates_T_sim_and_calls_microscopic_steps(self) -> None:
        game = PopulationGame(
            num_populations=1, num_strategies=[2], fitness_function=_fitness_ok
        )
        pm = PayoffMechanism(h_map=_h_map_memoryless, n=2, d=0)
        rp = _DeterministicRevisionProcess(n=2, fixed_dt=0.1)
        sim = Simulator(
            population_game=game,
            payoff_mechanism=pm,
            revision_processes=rp,
            num_agents=4,
        )

        with self.assertRaises(TypeError):
            sim.run(T_sim=1.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            sim.run(T_sim=0)

        with patch.object(
            sim, "_microscopic_step", wraps=sim._microscopic_step
        ) as step_mock:
            sim.run(T_sim=1)

            # At least ceil(1 / 0.1) = 10 steps, possibly +1 due to a 0-step edge case
            self.assertGreaterEqual(step_mock.call_count, 10)
            self.assertLessEqual(step_mock.call_count, 11)

            self.assertTrue(np.isclose(sim.t, 1.0, atol=1e-12))


class TestSimulatorIntegrateEDMPDM(unittest.TestCase):
    def test_integrate_edm_pdm_validates_shapes(self) -> None:
        game = PopulationGame(
            num_populations=1, num_strategies=[2], fitness_function=_fitness_ok
        )
        pm = PayoffMechanism(h_map=_h_map_memoryless, n=2, d=0)
        rp = _DeterministicRevisionProcess(n=2, fixed_dt=0.1)
        sim = Simulator(
            population_game=game,
            payoff_mechanism=pm,
            revision_processes=rp,
            num_agents=4,
        )

        with self.assertRaises(ValueError):
            sim.integrate_edm_pdm(t_span=(0.0, 1.0), x0=np.zeros((3, 1)))  # wrong shape

    @patch("popgames.simulator.sp.integrate.solve_ivp")
    def test_integrate_edm_pdm_output_trajectory_true(
        self, solve_ivp_mock: MagicMock
    ) -> None:
        # d=0 simplifies: y consists only of x
        game = PopulationGame(
            num_populations=1, num_strategies=[2], fitness_function=_fitness_ok
        )
        pm = PayoffMechanism(h_map=_h_map_memoryless, n=2, d=0)
        rp = _DeterministicRevisionProcess(n=2, fixed_dt=0.1)
        sim = Simulator(
            population_game=game,
            payoff_mechanism=pm,
            revision_processes=rp,
            num_agents=4,
        )

        # Fake solution with two time points
        t = np.array([0.0, 1.0])
        x0 = np.array([[0.2], [0.8]])
        x1 = np.array([[0.5], [0.5]])
        y = np.hstack([x0, x1])  # shape (n, T) since d=0

        solve_ivp_mock.return_value = SimpleNamespace(t=t, y=y)

        out = sim.integrate_edm_pdm(t_span=(0.0, 1.0), x0=x0, output_trajectory=True)

        self.assertTrue(hasattr(out, "t"))
        self.assertTrue(hasattr(out, "x"))
        self.assertTrue(hasattr(out, "q"))
        self.assertTrue(hasattr(out, "p"))

        np.testing.assert_allclose(out.t, t)
        self.assertEqual(out.q.shape, (0, 2))
        self.assertEqual(out.x.shape, (2, 2))
        self.assertEqual(out.p.shape, (2, 2))

        # With h_map(x)=x, p should equal x at each time
        np.testing.assert_allclose(out.p, out.x, atol=1e-12)

    @patch("popgames.simulator.sp.integrate.solve_ivp")
    def test_integrate_edm_pdm_output_trajectory_false(
        self, solve_ivp_mock: MagicMock
    ) -> None:
        game = PopulationGame(
            num_populations=1, num_strategies=[2], fitness_function=_fitness_ok
        )
        pm = PayoffMechanism(h_map=_h_map_memoryless, n=2, d=0)
        rp = _DeterministicRevisionProcess(n=2, fixed_dt=0.1)
        sim = Simulator(
            population_game=game,
            payoff_mechanism=pm,
            revision_processes=rp,
            num_agents=4,
        )

        t = np.array([0.0, 1.0])
        x0 = np.array([[0.2], [0.8]])
        x1 = np.array([[0.6], [0.4]])
        y = np.hstack([x0, x1])

        solve_ivp_mock.return_value = SimpleNamespace(t=t, y=y)

        out = sim.integrate_edm_pdm(t_span=(0.0, 1.0), x0=x0, output_trajectory=False)

        # In this mode, t is final time, and x/q/p are final column vectors
        self.assertIsInstance(out.t, (float, np.floating))
        self.assertEqual(out.x.shape, (2, 1))
        self.assertEqual(out.q.shape, (0, 1))
        self.assertEqual(out.p.shape, (2, 1))

        np.testing.assert_allclose(out.x, x1, atol=1e-12)
        np.testing.assert_allclose(out.p, x1, atol=1e-12)


_RPS = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])


def _rps_fitness(x: np.ndarray) -> np.ndarray:
    return _RPS @ x


def _make_rps_simulator(num_agents: int, **kwargs) -> Simulator:
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


_RPS_X0 = np.array([[0.5], [0.3], [0.2]])


class TestSimulatorPathSelection(unittest.TestCase):
    def test_fast_path_is_used_by_default_with_poisson_processes(self) -> None:
        sim = _make_rps_simulator(10)
        self.assertTrue(sim.uses_fast_path)

    def test_fast_path_can_be_disabled(self) -> None:
        sim = _make_rps_simulator(10, fast_path=False)
        self.assertFalse(sim.uses_fast_path)
        self.assertEqual(sim._revision_times[0].shape, (10,))

    def test_non_poisson_processes_fall_back_to_per_agent_path(self) -> None:
        game = PopulationGame(
            num_populations=1, num_strategies=[2], fitness_function=_fitness_ok
        )
        pm = PayoffMechanism(h_map=_h_map_memoryless, n=2, d=0)
        rp = _DeterministicRevisionProcess(n=2, fixed_dt=0.1)

        with self.assertLogs("popgames.simulator", level="INFO"):
            sim = Simulator(
                population_game=game,
                payoff_mechanism=pm,
                revision_processes=rp,
                num_agents=4,
            )
        self.assertFalse(sim.uses_fast_path)

    def test_init_rejects_invalid_options(self) -> None:
        with self.assertRaises(TypeError):
            _make_rps_simulator(10, fast_path=1)
        with self.assertRaises(ValueError):
            _make_rps_simulator(10, pdm_max_step=0.0)


class TestSimulatorFastPath(unittest.TestCase):
    def test_run_reaches_final_time_and_preserves_mass(self) -> None:
        sim = _make_rps_simulator(100, seed=0)
        sim.reset(x0=_RPS_X0)
        out = sim.run(T_sim=3)

        self.assertEqual(sim.t, 3)
        self.assertEqual(out.t[-1], 3)
        self.assertTrue(np.all(np.diff(out.t) > 0))
        np.testing.assert_allclose(out.x.sum(axis=0), 1.0)
        np.testing.assert_allclose(sim._counts[0].sum(), 100)
        np.testing.assert_allclose(out.p, _RPS @ out.x)

    def test_consecutive_runs_continue_from_last_state(self) -> None:
        sim = _make_rps_simulator(100, seed=0)
        sim.reset(x0=_RPS_X0)
        sim.run(T_sim=1)
        out = sim.run(T_sim=2)
        self.assertEqual(sim.t, 3)
        self.assertEqual(out.t[0], 0)
        self.assertEqual(out.t[-1], 3)

    def test_logged_states_are_not_modified_by_later_events(self) -> None:
        sim = _make_rps_simulator(100, seed=0)
        sim.reset(x0=_RPS_X0)
        sim.run(T_sim=2)
        np.testing.assert_allclose(sim.log.x[:, [0]], _RPS_X0)
        # Consecutive log entries differ by at most one agent switching strategy
        jumps = np.abs(np.diff(sim.log.x, axis=1)).sum(axis=0)
        self.assertTrue(np.all(np.isclose(jumps, 0.0) | np.isclose(jumps, 2 / 100)))

    def test_multi_population_run_preserves_masses(self) -> None:
        def fitness(x: np.ndarray) -> np.ndarray:
            return -x

        game = PopulationGame(
            num_populations=2,
            num_strategies=[2, 3],
            fitness_function=fitness,
            masses=[1.0, 2.0],
        )
        sim = Simulator(
            population_game=game,
            payoff_mechanism=PayoffMechanism(h_map=fitness, n=5),
            revision_processes=[
                PoissonRevisionProcess(1.0, Smith(scale=0.5)),
                PoissonRevisionProcess(3.0, Smith(scale=0.5)),
            ],
            num_agents=[50, 80],
            seed=0,
        )
        self.assertTrue(sim.uses_fast_path)
        sim.reset(x0=np.array([[1.0], [0.0], [2.0], [0.0], [0.0]]))
        out = sim.run(T_sim=5)

        np.testing.assert_allclose(out.x[:2].sum(axis=0), 1.0)
        np.testing.assert_allclose(out.x[2:].sum(axis=0), 2.0)
        # Both populations converge towards the uniform NE
        np.testing.assert_allclose(
            out.x[:, -1], [0.5, 0.5, 2 / 3, 2 / 3, 2 / 3], atol=0.1
        )


class TestSimulatorLogProperty(unittest.TestCase):
    def test_log_has_one_column_per_entry_and_matches_run_output(self) -> None:
        for fast_path in [True, False]:
            sim = _make_rps_simulator(50, seed=0, fast_path=fast_path)
            sim.reset(x0=_RPS_X0)
            out = sim.run(T_sim=2)
            log = sim.log

            K = log.t.shape[0]
            self.assertEqual(log.x.shape, (3, K))
            self.assertEqual(log.q.shape, (0, K))
            self.assertEqual(log.p.shape, (3, K))
            np.testing.assert_array_equal(log.t, out.t)
            np.testing.assert_array_equal(log.x, out.x)
            np.testing.assert_allclose(log.x[:, [0]], _RPS_X0)

    def test_log_is_refreshed_after_new_entries_and_reset(self) -> None:
        sim = _make_rps_simulator(50, seed=0)
        sim.reset(x0=_RPS_X0)
        self.assertEqual(sim.log.t.shape, (1,))

        sim.run(T_sim=1)
        K1 = sim.log.t.shape[0]
        self.assertIs(sim.log, sim.log)  # cached while nothing new is logged
        sim.run(T_sim=1)
        self.assertGreater(sim.log.t.shape[0], K1)
        self.assertEqual(sim.log.t[-1], 2)

        sim.reset(x0=_RPS_X0)
        self.assertEqual(sim.log.t.shape, (1,))


class TestSimulatorSeed(unittest.TestCase):
    def _run(self, **kwargs) -> SimpleNamespace:
        sim = _make_rps_simulator(50, **kwargs)
        sim.reset(x0=_RPS_X0)
        return sim.run(T_sim=2)

    def test_same_seed_gives_identical_runs_on_both_paths(self) -> None:
        for fast_path in [True, False]:
            out1 = self._run(seed=7, fast_path=fast_path)
            out2 = self._run(seed=7, fast_path=fast_path)
            np.testing.assert_array_equal(out1.t, out2.t)
            np.testing.assert_array_equal(out1.x, out2.x)

    def test_different_seeds_give_different_runs(self) -> None:
        out1 = self._run(seed=1)
        out2 = self._run(seed=2)
        self.assertFalse(np.array_equal(out1.t, out2.t))

    def test_accepts_generator(self) -> None:
        out1 = self._run(seed=np.random.default_rng(5))
        out2 = self._run(seed=5)
        np.testing.assert_array_equal(out1.x, out2.x)

    def test_without_seed_uses_global_numpy_random_state(self) -> None:
        np.random.seed(11)
        out1 = self._run()
        np.random.seed(11)
        out2 = self._run()
        np.testing.assert_array_equal(out1.x, out2.x)

    def test_random_initial_state_is_seeded(self) -> None:
        sim1 = _make_rps_simulator(50, seed=3)
        sim2 = _make_rps_simulator(50, seed=3)
        np.testing.assert_array_equal(sim1.x, sim2.x)
        self.assertAlmostEqual(float(sim1.x.sum()), 1.0)


class TestSimulatorLogInterval(unittest.TestCase):
    def test_log_interval_limits_log_entries_on_both_paths(self) -> None:
        for fast_path in [True, False]:
            sim = _make_rps_simulator(200, seed=0, fast_path=fast_path)
            sim.reset(x0=_RPS_X0)
            out = sim.run(T_sim=5, log_interval=0.5)

            self.assertAlmostEqual(out.t[-1], 5)
            self.assertTrue(np.all(np.diff(out.t[:-1]) >= 0.5))
            self.assertLessEqual(len(out.t), 12)

    def test_log_interval_must_be_positive(self) -> None:
        sim = _make_rps_simulator(10)
        with self.assertRaises(ValueError):
            sim.run(T_sim=1, log_interval=0.0)


class TestSimulatorPDMIntegration(unittest.TestCase):
    def _make_sim(self, pdm_method: str) -> Simulator:
        A, b = np.array([[0.5, 0.5, 0.0]]), np.array([[0.2]])

        def h_map(q: np.ndarray, x: np.ndarray) -> np.ndarray:
            return -x - A.T @ q

        def w_map(q: np.ndarray, x: np.ndarray) -> np.ndarray:
            return A @ x - b

        return Simulator(
            population_game=SinglePopulationGame(
                num_strategies=3, fitness_function=lambda x: -x
            ),
            payoff_mechanism=PayoffMechanism(h_map=h_map, w_map=w_map, n=3, d=1),
            revision_processes=PoissonRevisionProcess(1.0, Softmax(eta=0.1)),
            num_agents=20,
            pdm_method=pdm_method,
            seed=0,
        )

    def test_rk4_and_radau_give_the_same_trajectory(self) -> None:
        outs = []
        for method in ["RK4", "Radau"]:
            sim = self._make_sim(method)
            sim.reset(x0=np.array([[1.0], [0.0], [0.0]]), q0=np.zeros((1, 1)))
            outs.append(sim.run(T_sim=2))

        # Same seed and (numerically) same payoffs => same revision events
        np.testing.assert_array_equal(outs[0].x, outs[1].x)
        np.testing.assert_allclose(outs[0].q, outs[1].q, atol=1e-6)

    def test_pdm_state_matches_closed_form_while_x_is_constant(self) -> None:
        sim = self._make_sim("RK4")
        sim.reset(x0=np.array([[1.0], [0.0], [0.0]]), q0=np.zeros((1, 1)))
        out = sim.run(T_sim=1)
        # Before the first switch, x = e_1 so q' = 0.5 - 0.2 = 0.3
        first_switch = np.argmax(np.any(out.x != out.x[:, [0]], axis=0))
        t = out.t[:first_switch]
        np.testing.assert_allclose(out.q[0, :first_switch], 0.3 * t, atol=1e-9)


class TestSimulatorStatisticalEquivalence(unittest.TestCase):
    """The fast path and the per-agent path simulate the same stochastic process."""

    def test_mean_final_state_agrees_between_paths(self) -> None:
        num_replicates, num_agents, T = 40, 100, 4
        finals = {}
        for fast_path in [True, False]:
            sim = _make_rps_simulator(num_agents, fast_path=fast_path, seed=123)
            xs = []
            for _ in range(num_replicates):
                sim.reset(x0=_RPS_X0)
                xs.append(sim.run(T_sim=T, log_interval=T).x[:, -1])
            finals[fast_path] = np.array(xs)

        diff = finals[True].mean(axis=0) - finals[False].mean(axis=0)
        std_err = np.sqrt(
            (finals[True].var(axis=0) + finals[False].var(axis=0)) / num_replicates
        )
        self.assertTrue(np.all(np.abs(diff) < 4 * std_err + 1e-12), (diff, std_err))

    def test_large_population_tracks_the_edm(self) -> None:
        sim = _make_rps_simulator(5000, seed=0)
        sim.reset(x0=_RPS_X0)
        out = sim.run(T_sim=3, log_interval=0.1)
        edm = sim.integrate_edm_pdm(t_span=(0, 3), x0=_RPS_X0, t_eval=out.t)
        np.testing.assert_allclose(out.x, edm.x, atol=0.03)


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
