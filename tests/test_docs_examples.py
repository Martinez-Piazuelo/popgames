import importlib.util
import unittest

import numpy as np

import popgames as pg
from tests.helpers import (
    run_sim_and_capture,
)

NUMBA_AVAILABLE = importlib.util.find_spec("numba") is not None


class TestDocsExamplesSmoke(unittest.TestCase):
    def test_example_prisoner_dilemma(self):
        T, R, P, S = 3.0, 2.0, 1.0, 0.0

        def fitness_function(x):
            return np.dot(np.array([[R, S], [T, P]]), x)

        population_game = pg.SinglePopulationGame(
            num_strategies=2,
            fitness_function=fitness_function,
        )

        payoff_mechanism = pg.PayoffMechanism(
            h_map=fitness_function,
            n=2,
        )

        revision_process = pg.PoissonRevisionProcess(
            Poisson_clock_rate=1,
            revision_protocol=pg.revision_protocol.Softmax(0.1),
        )

        sim = pg.Simulator(
            population_game=population_game,
            payoff_mechanism=payoff_mechanism,
            revision_processes=revision_process,
            num_agents=1000,
        )

        x0 = np.array([0.5, 0.5]).reshape(2, 1)
        sim.reset(x0=x0)

        # Run reduced simulation and capture the flattened log (sim._get_flattened_log)
        result = run_sim_and_capture(sim=sim, T_sim=2)

        # The simulator's output is the flattened log.
        log = result.snapshots["log"]

        # Shape checks (robust to number of logged steps K)
        self.assertEqual(log.t.ndim, 1)  # (K,)
        self.assertEqual(log.x.shape[0], 2)  # (n, K) with n=2
        self.assertEqual(log.q.shape[0], 0)  # (d, K) with d=0
        self.assertEqual(log.p.shape[0], 2)  # (n, K) with n=2

        # Sanity: all logs use same K
        K = log.t.shape[0]
        self.assertEqual(log.x.shape[1], K)
        self.assertEqual(log.q.shape[1], K)
        self.assertEqual(log.p.shape[1], K)

        # Final-state shapes (last column)
        xT = log.x[:, [-1]]  # keep as (2,1)
        qT = log.q[:, [-1]]  # keep as (0,1)
        pT = log.p[:, [-1]]  # keep as (0, 1)
        self.assertEqual(xT.shape, (2, 1))
        self.assertEqual(qT.shape, (0, 1))
        self.assertEqual(pT.shape, (2, 1))

    def test_example_rock_paper_scissors(self):
        A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)

        def fitness_function(x):
            return np.dot(A, x)

        def make_sim(revision_protocol):
            return pg.Simulator(
                population_game=pg.SinglePopulationGame(
                    num_strategies=3,
                    fitness_function=fitness_function,
                ),
                payoff_mechanism=pg.PayoffMechanism(h_map=fitness_function, n=3),
                revision_processes=pg.PoissonRevisionProcess(
                    Poisson_clock_rate=1,
                    revision_protocol=revision_protocol,
                ),
                num_agents=1000,
            )

        x0 = np.array([0.5, 0.3, 0.2]).reshape(3, 1)
        x_ne = np.ones((3, 1)) / 3
        t_eval = np.linspace(0, 100, 201)

        # Replicator: closed orbits, i.e., x1 * x2 * x3 is conserved along the EDM
        replicator_sim = make_sim(pg.revision_protocol.Replicator(scale=0.5))
        edm = replicator_sim.integrate_edm_pdm(t_span=(0, 100), x0=x0, t_eval=t_eval)
        np.testing.assert_allclose(np.prod(edm.x, axis=0), np.prod(x0), rtol=1e-2)

        # Smith: converges to the NE
        smith_sim = make_sim(pg.revision_protocol.Smith(scale=0.25))
        edm = smith_sim.integrate_edm_pdm(t_span=(0, 100), x0=x0, t_eval=t_eval)
        np.testing.assert_allclose(edm.x[:, [-1]], x_ne, atol=2e-2)

        # Smoke test: finite-agent simulation with the replicator protocol
        replicator_sim.reset(x0=x0)
        result = run_sim_and_capture(sim=replicator_sim, T_sim=2)
        log = result.snapshots["log"]
        self.assertEqual(log.x.shape[0], 3)
        np.testing.assert_allclose(log.x.sum(axis=0), 1.0)


@unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
class TestDocsExamplesNumbaBackend(unittest.TestCase):
    def test_example_prisoner_dilemma_numba(self):
        T, R, P, S = 3.0, 2.0, 1.0, 0.0

        def fitness_function(x):
            return np.dot(np.array([[R, S], [T, P]]), x)

        sim = pg.Simulator(
            population_game=pg.SinglePopulationGame(
                num_strategies=2, fitness_function=fitness_function
            ),
            payoff_mechanism=pg.PayoffMechanism(h_map=fitness_function, n=2),
            revision_processes=pg.PoissonRevisionProcess(
                Poisson_clock_rate=1,
                revision_protocol=pg.revision_protocol.Softmax(0.1),
            ),
            num_agents=1000,
            backend="numba",
            seed=0,
        )
        self.assertEqual(sim.backend, "numba")
        sim.reset(x0=np.array([0.5, 0.5]).reshape(2, 1))
        out = sim.run(T_sim=30)
        # Defection takes over
        self.assertGreater(out.x[1, -1], 0.9)

    def test_example_rock_paper_scissors_numba(self):
        import numba

        A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)

        def fitness_function(x):
            return np.dot(A, x)

        @numba.njit
        def fitness_function_jitted(x):
            return np.dot(A, x)

        x0 = np.array([0.5, 0.3, 0.2]).reshape(3, 1)
        for fitness in [fitness_function, fitness_function_jitted]:
            for protocol in [
                pg.revision_protocol.Replicator(scale=0.5),
                pg.revision_protocol.Smith(scale=0.25),
            ]:
                sim = pg.Simulator(
                    population_game=pg.SinglePopulationGame(
                        num_strategies=3, fitness_function=fitness
                    ),
                    payoff_mechanism=pg.PayoffMechanism(h_map=fitness, n=3),
                    revision_processes=pg.PoissonRevisionProcess(
                        Poisson_clock_rate=1, revision_protocol=protocol
                    ),
                    num_agents=1000,
                    backend="numba",
                    seed=0,
                )
                self.assertEqual(sim.backend, "numba")
                sim.reset(x0=x0)
                out = sim.run(T_sim=10)
                np.testing.assert_allclose(out.x.sum(axis=0), 1.0)

    def test_example_rock_paper_scissors_integer_matrix_falls_back(self):
        A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])

        def fitness_function(x):
            return np.dot(A, x)

        with self.assertLogs("popgames.simulator", level="WARNING"):
            sim = pg.Simulator(
                population_game=pg.SinglePopulationGame(
                    num_strategies=3, fitness_function=fitness_function
                ),
                payoff_mechanism=pg.PayoffMechanism(h_map=fitness_function, n=3),
                revision_processes=pg.PoissonRevisionProcess(
                    1, pg.revision_protocol.Smith(scale=0.25)
                ),
                num_agents=1000,
                backend="numba",
            )
        self.assertEqual(sim.backend, "numpy")


if __name__ == "__main__":
    unittest.main()
