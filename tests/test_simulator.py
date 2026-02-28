from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from popgames.alarm_clock import Poisson
from popgames.payoff_mechanism import PayoffMechanism
from popgames.population_game import PopulationGame
from popgames.revision_process import RevisionProcessABC
from popgames.revision_protocol import Softmax
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


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
