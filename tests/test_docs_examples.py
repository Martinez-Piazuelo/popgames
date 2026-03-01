import unittest

import numpy as np

import popgames as pg
from tests.helpers import (
    run_sim_and_capture,
)


class TestDocsExamplesSmoke(unittest.TestCase):
    def test_example_prisoner_dilemma(self):
        T, R, P, S = 3, 2, 1, 0

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


if __name__ == "__main__":
    unittest.main()
