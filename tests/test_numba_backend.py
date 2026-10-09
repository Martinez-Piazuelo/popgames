from __future__ import annotations

import importlib.util
import sys
import unittest
from unittest.mock import patch

import numpy as np

from popgames.payoff_mechanism import PayoffMechanism
from popgames.population_game import PopulationGame, SinglePopulationGame
from popgames.revision_process import PoissonRevisionProcess
from popgames.revision_protocol import (
    BNN,
    CCSmith,
    Replicator,
    RevisionProtocolABC,
    Smith,
    Softmax,
)
from popgames.simulator import Simulator
from tests.helpers import run_plot_smoketest

NUMBA_AVAILABLE = importlib.util.find_spec("numba") is not None

_RPS = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
_RPS_X0 = np.array([[0.5], [0.3], [0.2]])


def _rps_fitness(x: np.ndarray) -> np.ndarray:
    return _RPS @ x


def _make_rps_simulator(
    num_agents: int, protocol: RevisionProtocolABC = None, **kwargs
) -> Simulator:
    sim = Simulator(
        population_game=SinglePopulationGame(
            num_strategies=3, fitness_function=_rps_fitness
        ),
        payoff_mechanism=PayoffMechanism(h_map=_rps_fitness, n=3),
        revision_processes=PoissonRevisionProcess(
            Poisson_clock_rate=1.0,
            revision_protocol=Smith(scale=0.25) if protocol is None else protocol,
        ),
        num_agents=num_agents,
        **kwargs,
    )
    sim.reset(x0=_RPS_X0)
    return sim


_A_EQ, _B_EQ = np.array([[0.5, 0.5, 0.0]]), np.array([[0.2]])


def _pdm_h_map(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    return -x - _A_EQ.T @ q


def _pdm_w_map(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    return _A_EQ @ x - _B_EQ


def _make_pdm_simulator(num_agents: int, **kwargs) -> Simulator:
    sim = Simulator(
        population_game=SinglePopulationGame(
            num_strategies=3, fitness_function=lambda x: -x
        ),
        payoff_mechanism=PayoffMechanism(h_map=_pdm_h_map, w_map=_pdm_w_map, n=3, d=1),
        revision_processes=PoissonRevisionProcess(1.0, Softmax(eta=0.1)),
        num_agents=num_agents,
        **kwargs,
    )
    sim.reset(x0=np.array([[1.0], [0.0], [0.0]]), q0=np.zeros((1, 1)))
    return sim


class _CustomProtocol(RevisionProtocolABC):
    def __call__(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.full((p.shape[0], p.shape[0]), 0.1)


class _SmithSubclass(Smith):
    pass


class TestNumbaNotInstalled(unittest.TestCase):
    def test_numba_backend_without_numba_raises_import_error(self) -> None:
        with patch.dict(sys.modules, {"popgames._numba_backend": None}):
            with self.assertRaises(ImportError) as cm:
                _make_rps_simulator(10, backend="numba")
        self.assertIn("pip install popgames[numba]", str(cm.exception))

    def test_invalid_backend_raises(self) -> None:
        with self.assertRaises(ValueError):
            _make_rps_simulator(10, backend="cuda")
        with self.assertRaises(TypeError):
            _make_rps_simulator(10, backend=1)

    def test_numpy_backend_is_the_default(self) -> None:
        self.assertEqual(_make_rps_simulator(10).backend, "numpy")


@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class TestNumbaKernels(unittest.TestCase):
    def test_protocol_columns_match_python_protocols(self) -> None:
        from popgames import _numba_backend

        rng = np.random.default_rng(0)
        protocols = [
            Softmax(eta=0.3),
            Smith(scale=0.2),
            BNN(scale=0.2),
            Replicator(scale=0.2),
            CCSmith(scale=0.2, x_bar=np.array([[0.4], [0.6], [0.5], [0.3]])),
        ]
        for protocol in protocols:
            kind, params = _numba_backend.protocol_spec(protocol)
            for _ in range(5):
                p = rng.normal(size=(4, 1))
                x = rng.random((4, 1))
                expected = protocol(p, x)
                for i in range(4):
                    col = _numba_backend._protocol_column(
                        kind, params, p[:, 0], x[:, 0], i
                    )
                    np.testing.assert_allclose(
                        col, expected[:, i], err_msg=repr(protocol)
                    )

    def test_protocol_spec_rejects_custom_protocols_and_subclasses(self) -> None:
        from popgames import _numba_backend

        self.assertIsNone(_numba_backend.protocol_spec(_CustomProtocol()))
        self.assertIsNone(_numba_backend.protocol_spec(_SmithSubclass(scale=0.1)))

    def test_compile_user_function_reuses_compiled_functions(self) -> None:
        from popgames import _numba_backend

        def f(x):
            return 2.0 * x

        compiled = _numba_backend.compile_user_function(f, memoryless=True)
        self.assertIs(
            _numba_backend.compile_user_function(f, memoryless=True), compiled
        )
        x = np.array([[1.0], [2.0]])
        np.testing.assert_allclose(compiled(np.zeros((0, 1)), x), 2.0 * x)

    def test_compile_user_function_accepts_jitted_functions(self) -> None:
        import numba

        from popgames import _numba_backend

        @numba.njit
        def h(q, x):
            return x + q[0, 0]

        compiled = _numba_backend.compile_user_function(h, memoryless=False)
        x = np.array([[1.0], [2.0]])
        np.testing.assert_allclose(compiled(np.ones((1, 1)), x), x + 1.0)

    def test_user_functions_share_the_compiled_event_loop(self) -> None:
        from popgames import _numba_backend

        for c in [1.0, 2.0]:

            def fitness(x, c=c):
                return c * x

            sim = Simulator(
                population_game=SinglePopulationGame(
                    num_strategies=3, fitness_function=fitness
                ),
                payoff_mechanism=PayoffMechanism(h_map=fitness, n=3),
                revision_processes=PoissonRevisionProcess(1.0, Smith(scale=0.1)),
                num_agents=10,
                backend="numba",
            )
            sim.run(T_sim=1)
        self.assertEqual(len(_numba_backend.run_events.signatures), 1)

    def test_sample_index_never_selects_zero_weights(self) -> None:
        from popgames import _numba_backend

        weights = np.array([0.0, 0.3, 0.0, 0.7, 0.0])
        for u in [0.0, 0.29, 0.31, 0.999999, 1.0]:
            j = _numba_backend._sample_index(weights, 1.0, u)
            self.assertIn(j, [1, 3])


@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class TestNumbaBackendSelection(unittest.TestCase):
    def test_numba_backend_is_used_when_requested(self) -> None:
        self.assertEqual(_make_rps_simulator(10, backend="numba").backend, "numba")

    def test_accepts_explicitly_jitted_functions(self) -> None:
        import numba

        fitness = numba.njit(_rps_fitness)
        sim = Simulator(
            population_game=SinglePopulationGame(
                num_strategies=3, fitness_function=fitness
            ),
            payoff_mechanism=PayoffMechanism(h_map=fitness, n=3),
            revision_processes=PoissonRevisionProcess(1.0, Smith(scale=0.25)),
            num_agents=10,
            backend="numba",
        )
        self.assertEqual(sim.backend, "numba")

    def _assert_falls_back(self, make_sim, reason: str) -> Simulator:
        with self.assertLogs("popgames.simulator", level="WARNING") as cm:
            sim = make_sim()
        self.assertEqual(sim.backend, "numpy")
        self.assertTrue(any(reason in msg for msg in cm.output), cm.output)
        return sim

    def test_falls_back_without_fast_path(self) -> None:
        self._assert_falls_back(
            lambda: _make_rps_simulator(10, backend="numba", fast_path=False),
            "requires the fast path",
        )

    def test_falls_back_with_custom_protocol(self) -> None:
        self._assert_falls_back(
            lambda: _make_rps_simulator(10, _CustomProtocol(), backend="numba"),
            "_CustomProtocol",
        )

    def test_falls_back_with_non_rk4_pdm_method(self) -> None:
        self._assert_falls_back(
            lambda: _make_pdm_simulator(10, backend="numba", pdm_method="Radau"),
            "pdm_method='RK4'",
        )

    def test_falls_back_when_functions_cannot_be_compiled(self) -> None:
        A_int = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])  # integer matrix

        def fitness(x):
            return np.dot(A_int, x)

        def make_sim():
            return Simulator(
                population_game=SinglePopulationGame(
                    num_strategies=3, fitness_function=fitness
                ),
                payoff_mechanism=PayoffMechanism(h_map=fitness, n=3),
                revision_processes=PoissonRevisionProcess(1.0, Smith(scale=0.25)),
                num_agents=10,
                backend="numba",
            )

        sim = self._assert_falls_back(make_sim, "could not be compiled")
        sim.reset(x0=_RPS_X0)
        out = sim.run(T_sim=1)  # still runs with the numpy backend
        self.assertEqual(out.t[-1], 1)


@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class TestNumbaBackendRun(unittest.TestCase):
    def test_run_reaches_final_time_and_preserves_mass(self) -> None:
        sim = _make_rps_simulator(100, backend="numba", seed=0)
        out = sim.run(T_sim=3)

        self.assertEqual(sim.t, 3)
        self.assertEqual(out.t[-1], 3)
        self.assertTrue(np.all(np.diff(out.t) > 0))
        np.testing.assert_allclose(out.x.sum(axis=0), 1.0)
        np.testing.assert_allclose(out.p, _RPS @ out.x)
        self.assertEqual(sim._counts[0].sum(), 100)
        np.testing.assert_allclose(sim.x, out.x[:, [-1]])

    def test_consecutive_runs_continue_from_last_state(self) -> None:
        sim = _make_rps_simulator(100, backend="numba", seed=0)
        sim.run(T_sim=1)
        out = sim.run(T_sim=2)
        self.assertEqual(sim.t, 3)
        self.assertEqual(out.t[0], 0)
        self.assertEqual(out.t[-1], 3)
        self.assertTrue(np.all(np.diff(out.t) > 0))

    def test_same_seed_gives_identical_runs(self) -> None:
        out1 = _make_rps_simulator(50, backend="numba", seed=7).run(T_sim=2)
        out2 = _make_rps_simulator(50, backend="numba", seed=7).run(T_sim=2)
        np.testing.assert_array_equal(out1.t, out2.t)
        np.testing.assert_array_equal(out1.x, out2.x)

    def test_without_seed_uses_global_numpy_random_state(self) -> None:
        np.random.seed(11)
        out1 = _make_rps_simulator(50, backend="numba").run(T_sim=2)
        np.random.seed(11)
        out2 = _make_rps_simulator(50, backend="numba").run(T_sim=2)
        np.testing.assert_array_equal(out1.x, out2.x)

    def test_chunked_log_buffers_give_the_same_run(self) -> None:
        out1 = _make_rps_simulator(100, backend="numba", seed=3).run(T_sim=3)

        sim = _make_rps_simulator(100, backend="numba", seed=3)
        sim._numba_log_capacity = 17  # forces many calls to the compiled event loop
        out2 = sim.run(T_sim=3)

        np.testing.assert_array_equal(out1.t, out2.t)
        np.testing.assert_array_equal(out1.x, out2.x)

    def test_log_interval(self) -> None:
        sim = _make_rps_simulator(200, backend="numba", seed=0)
        out = sim.run(T_sim=5, log_interval=0.5)
        self.assertEqual(out.t[-1], 5)
        self.assertTrue(np.all(np.diff(out.t[:-1]) >= 0.5))
        self.assertLessEqual(len(out.t), 12)

    def test_all_builtin_protocols_run(self) -> None:
        protocols = [
            Softmax(eta=0.3),
            Smith(scale=0.2),
            BNN(scale=0.2),
            Replicator(scale=0.4),
            CCSmith(scale=0.2, x_bar=np.array([[0.6], [0.6], [0.6]])),
        ]
        for protocol in protocols:
            sim = _make_rps_simulator(100, protocol, backend="numba", seed=0)
            self.assertEqual(sim.backend, "numba")
            out = sim.run(T_sim=2)
            np.testing.assert_allclose(out.x.sum(axis=0), 1.0)

    def test_invalid_probabilities_are_clipped_with_a_warning(self) -> None:
        sim = _make_rps_simulator(100, Smith(scale=10.0), backend="numba", seed=0)
        with self.assertLogs("popgames.simulator", level="WARNING") as cm:
            out = sim.run(T_sim=1)
        self.assertTrue(any("Invalid switching probabilities" in m for m in cm.output))
        np.testing.assert_allclose(out.x.sum(axis=0), 1.0)

    def test_multi_population_with_different_protocols(self) -> None:
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
                PoissonRevisionProcess(3.0, Softmax(eta=0.05)),
            ],
            num_agents=[50, 80],
            seed=0,
            backend="numba",
        )
        self.assertEqual(sim.backend, "numba")
        sim.reset(x0=np.array([[1.0], [0.0], [2.0], [0.0], [0.0]]))
        out = sim.run(T_sim=5)

        np.testing.assert_allclose(out.x[:2].sum(axis=0), 1.0)
        np.testing.assert_allclose(out.x[2:].sum(axis=0), 2.0)
        np.testing.assert_allclose(
            out.x[:, -1], [0.5, 0.5, 2 / 3, 2 / 3, 2 / 3], atol=0.15
        )

    def test_pdm_state_matches_closed_form_while_x_is_constant(self) -> None:
        sim = _make_pdm_simulator(20, backend="numba", seed=0)
        self.assertEqual(sim.backend, "numba")
        out = sim.run(T_sim=1)
        # Before the first switch, x = e_1 so q' = 0.5 - 0.2 = 0.3
        first_switch = np.argmax(np.any(out.x != out.x[:, [0]], axis=0))
        t = out.t[:first_switch]
        np.testing.assert_allclose(out.q[0, :first_switch], 0.3 * t, atol=1e-9)
        np.testing.assert_allclose(out.p, _pdm_h_map(out.q, out.x))

    def test_plots_work_with_numba_logs(self) -> None:
        sim = _make_rps_simulator(100, backend="numba", seed=0)
        sim.run(T_sim=2)
        for plot_type in ["ternary", "univariate"]:
            run_plot_smoketest(
                sim=sim,
                plot_kwargs=dict(
                    plot_type=plot_type,
                    plot_deterministic_approximation=True,
                    show=False,
                ),
            )


@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class TestNumbaBackendStatisticalEquivalence(unittest.TestCase):
    """The numba and numpy backends simulate the same stochastic process."""

    def test_mean_final_state_agrees_with_numpy_backend(self) -> None:
        num_replicates, num_agents, T = 40, 100, 4
        finals = {}
        for backend in ["numpy", "numba"]:
            sim = _make_rps_simulator(num_agents, backend=backend, seed=123)
            xs = []
            for _ in range(num_replicates):
                sim.reset(x0=_RPS_X0)
                xs.append(sim.run(T_sim=T, log_interval=T).x[:, -1])
            finals[backend] = np.array(xs)

        diff = finals["numba"].mean(axis=0) - finals["numpy"].mean(axis=0)
        std_err = np.sqrt(
            (finals["numba"].var(axis=0) + finals["numpy"].var(axis=0)) / num_replicates
        )
        self.assertTrue(np.all(np.abs(diff) < 4 * std_err + 1e-12), (diff, std_err))

    def test_large_population_tracks_the_edm(self) -> None:
        sim = _make_rps_simulator(20000, backend="numba", seed=0)
        out = sim.run(T_sim=5, log_interval=0.1)
        edm = sim.integrate_edm_pdm(t_span=(0, 5), x0=_RPS_X0, t_eval=out.t)
        np.testing.assert_allclose(out.x, edm.x, atol=0.02)

    def test_large_population_with_pdm_tracks_the_edm_pdm(self) -> None:
        sim = _make_pdm_simulator(20000, backend="numba", seed=0)
        out = sim.run(T_sim=3, log_interval=0.1)
        det = sim.integrate_edm_pdm(
            t_span=(0, 3), x0=np.array([[1.0], [0.0], [0.0]]), t_eval=out.t
        )
        np.testing.assert_allclose(out.x, det.x, atol=0.02)
        np.testing.assert_allclose(out.q, det.q, atol=0.02)


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
