import unittest

import numpy as np
from scipy.linalg import block_diag

import popgames as pg
from tests.helpers import (
    patch_compute_gne_max_iter,
    run_plot_smoketest,
    run_sim_and_capture,
)


class TestBookExamplesSmoke(unittest.TestCase):
    def test_example_intro(self):
        np.random.seed(698)  # fix the random seed

        Q = np.diag([-0.6, -0.7, -0.8])
        r = np.array([0.0, 0.1, 0.1]).reshape(-1, 1)
        A, b = np.array([[0.5, 0.5, 0]]), np.array([0.2])

        def f(x):  # Fitness function
            return np.dot(Q, x) + r

        def w(q, x):  # RHS of payoff dynamics model
            return np.dot(A, x) - b

        def h(q, x):  # Output of payoff mechanism
            return f(x) - np.dot(A.T, q)

        population_game = pg.SinglePopulationGame(
            num_strategies=3, fitness_function=f, mass=1, A_eq=A, b_eq=b
        )

        payoff_mechanism = pg.PayoffMechanism(h_map=h, w_map=w, n=3, d=1)

        revision_process = pg.PoissonRevisionProcess(
            Poisson_clock_rate=1.0, revision_protocol=pg.protocol.Softmax(eta=0.01)
        )

        sim = pg.Simulator(
            population_game=population_game,
            payoff_mechanism=payoff_mechanism,
            revision_processes=revision_process,
            num_agents=1000,
        )

        x0, q0 = np.array([1, 0, 0]).reshape(-1, 1), np.array([[0]])
        sim.reset(x0=x0, q0=q0)

        # Run reduced simulation and capture the flattened log (sim._get_flattened_log)
        result = run_sim_and_capture(sim=sim, T_sim=2)

        # The simulator's output is the flattened log.
        log = result.snapshots["log"]

        # Shape checks (robust to number of logged steps K)
        self.assertEqual(log.t.ndim, 1)  # (K,)
        self.assertEqual(log.x.shape[0], 3)  # (n, K) with n=3
        self.assertEqual(log.q.shape[0], 1)  # (d, K) with d=1
        self.assertEqual(log.p.shape[0], 3)  # (n, K) with n=3

        # Sanity: all logs use same K
        K = log.t.shape[0]
        self.assertEqual(log.x.shape[1], K)
        self.assertEqual(log.q.shape[1], K)
        self.assertEqual(log.p.shape[1], K)

        # Final-state shapes (last column)
        xT = log.x[:, [-1]]  # keep as (3,1)
        qT = log.q[:, [-1]]  # keep as (1,1)
        pT = log.p[:, [-1]]  # keep as (3, 1)
        self.assertEqual(xT.shape, (3, 1))
        self.assertEqual(qT.shape, (1, 1))
        self.assertEqual(pT.shape, (3, 1))

        def varphi(x):  # Potential function to plot
            return 0.5 * np.dot(x.T, np.dot(Q, x)) + np.dot(r.T, x)

        # Plot smoke test (no GUI) and patched compute_gne
        with patch_compute_gne_max_iter(sim.population_game, max_iter=5):
            run_plot_smoketest(
                sim=sim,
                plot_kwargs=dict(
                    plot_type="ternary",
                    potential_function=varphi,
                    plot_deterministic_approximation=True,
                    show=False,
                ),
            )

    def test_example_research_topic(self):
        a, b, c = 1, 0.4, 0.5  # Parameter values
        modify_game = False  # Set to True to simulate \hat{G}

        def f(x):
            return a * np.exp(-np.power((x - b) / c, 2)) - 1.73 * x * modify_game

        population_game = pg.SinglePopulationGame(fitness_function=f, num_strategies=3)

        payoff_mechanism = pg.PayoffMechanism(
            h_map=f,
            n=3,
        )

        revision_process = pg.PoissonRevisionProcess(
            Poisson_clock_rate=1.0, revision_protocol=pg.protocol.Smith(scale=0.1)
        )

        sim = pg.Simulator(
            population_game=population_game,
            payoff_mechanism=payoff_mechanism,
            revision_processes=revision_process,
            num_agents=700,
        )

        x0 = np.array([[1.1 / 3], [0.9 / 3], [1 / 3]])
        sim.reset(x0=x0)

        # Run reduced simulation and capture the flattened log (sim._get_flattened_log)
        result = run_sim_and_capture(sim=sim, T_sim=2)

        # The simulator's output is the flattened log.
        log = result.snapshots["log"]

        # Shape checks (robust to number of logged steps K)
        self.assertEqual(log.t.ndim, 1)  # (K,)
        self.assertEqual(log.x.shape[0], 3)  # (n, K) with n=3
        self.assertEqual(log.q.shape[0], 0)  # (d, K) with d=0
        self.assertEqual(log.p.shape[0], 3)  # (n, K) with n=3

        # Sanity: all logs use same K
        K = log.t.shape[0]
        self.assertEqual(log.x.shape[1], K)
        self.assertEqual(log.q.shape[1], K)
        self.assertEqual(log.p.shape[1], K)

        # Final-state shapes (last column)
        xT = log.x[:, [-1]]  # keep as (3,1)
        qT = log.q[:, [-1]]  # keep as (0,1)
        pT = log.p[:, [-1]]  # keep as (3, 1)
        self.assertEqual(xT.shape, (3, 1))
        self.assertEqual(qT.shape, (0, 1))
        self.assertEqual(pT.shape, (3, 1))

        # Plot smoke test (no GUI) and patched compute_gne
        with patch_compute_gne_max_iter(sim.population_game, max_iter=5):
            run_plot_smoketest(
                sim=sim,
                plot_kwargs=dict(
                    plot_type="univariate",
                    plot_deterministic_approximation=True,
                    show=False,
                ),
            )

            run_plot_smoketest(
                sim=sim,
                plot_kwargs=dict(
                    plot_type="kpi",
                    plot_deterministic_approximation=True,
                    show=False,
                ),
            )

    def test_example_load_balancing(self):
        # Parameters of the game
        a, s = np.array([[1], [5], [2]]), 10

        # Selected revision protocols to choose from
        rev_protocols = [
            pg.protocol.BNN(scale=0.1),
            pg.protocol.Smith(scale=0.1),
            pg.protocol.Softmax(eta=0.01),
        ]

        def f(x):
            return s - a * x

        population_game = pg.SinglePopulationGame(num_strategies=3, fitness_function=f)

        payoff_mechanism = pg.PayoffMechanism(h_map=f, n=3)

        revision_process = pg.PoissonRevisionProcess(
            Poisson_clock_rate=1,
            revision_protocol=rev_protocols[0],  # Set to 0, 1, or 2
            # depending on the
            # desired protocol to use
        )

        sim = pg.Simulator(
            population_game=population_game,
            payoff_mechanism=payoff_mechanism,
            revision_processes=revision_process,
            num_agents=400,
        )

        sim.reset(x0=np.array([1 / 3, 1 / 3, 1 / 3]).reshape(3, 1))

        # Run reduced simulation and capture the flattened log (sim._get_flattened_log)
        result = run_sim_and_capture(sim=sim, T_sim=2)

        # The simulator's output is the flattened log.
        log = result.snapshots["log"]

        # Shape checks (robust to number of logged steps K)
        self.assertEqual(log.t.ndim, 1)  # (K,)
        self.assertEqual(log.x.shape[0], 3)  # (n, K) with n=3
        self.assertEqual(log.q.shape[0], 0)  # (d, K) with d=0
        self.assertEqual(log.p.shape[0], 3)  # (n, K) with n=3

        # Sanity: all logs use same K
        K = log.t.shape[0]
        self.assertEqual(log.x.shape[1], K)
        self.assertEqual(log.q.shape[1], K)
        self.assertEqual(log.p.shape[1], K)

        # Final-state shapes (last column)
        xT = log.x[:, [-1]]  # keep as (3,1)
        qT = log.q[:, [-1]]  # keep as (0,1)
        pT = log.p[:, [-1]]  # keep as (3, 1)
        self.assertEqual(xT.shape, (3, 1))
        self.assertEqual(qT.shape, (0, 1))
        self.assertEqual(pT.shape, (3, 1))

        def varphi(x):
            return (s * x).sum() - 0.5 * np.dot(x.T, a * x)

        # Plot smoke test (no GUI) and patched compute_gne
        with patch_compute_gne_max_iter(sim.population_game, max_iter=5):
            run_plot_smoketest(
                sim=sim,
                plot_kwargs=dict(
                    plot_type="ternary",
                    potential_function=varphi,
                    plot_deterministic_approximation=True,
                    show=False,
                ),
            )

    def test_example_robots(self):
        np.random.seed(698)  # For reproducibility

        n = 4  # Number of zones
        x_bar = np.array([1, 0.2, 0.2, 0.2]).reshape(n, 1)
        q_bar = np.random.uniform(0, 2, (n, 1))
        gamma = np.random.uniform(0, 1, (n, 1))

        max0 = lambda x: np.maximum(x, 0)  # max(., 0)

        def f(x):
            return -x

        def h_map(q, x):
            return f(x) + q

        def w_map(q, x):
            return max0(q_bar - q) * max0(gamma - x) - q * max0(x - gamma)

        population_game = pg.SinglePopulationGame(
            num_strategies=n,
            fitness_function=f,
            A_ineq=np.eye(n),
            b_ineq=x_bar,
        )

        payoff_mechanism = pg.PayoffMechanism(
            h_map=h_map,
            n=n,
            w_map=w_map,
            d=n,
        )

        revision_process = pg.PoissonRevisionProcess(
            Poisson_clock_rate=1,
            revision_protocol=pg.protocol.CCSmith(scale=0.1, x_bar=x_bar),
        )

        sim = pg.Simulator(
            population_game=population_game,
            payoff_mechanism=payoff_mechanism,
            revision_processes=revision_process,
            num_agents=50,
        )

        sim.reset(x0=np.array([1, 0, 0, 0]).reshape(n, 1))

        # Run reduced simulation and capture the flattened log (sim._get_flattened_log)
        result = run_sim_and_capture(sim=sim, T_sim=2)

        # The simulator's output is the flattened log.
        log = result.snapshots["log"]

        # Shape checks (robust to number of logged steps K)
        self.assertEqual(log.t.ndim, 1)  # (K,)
        self.assertEqual(log.x.shape[0], 4)  # (n, K) with n=4
        self.assertEqual(log.q.shape[0], 4)  # (d, K) with d=4
        self.assertEqual(log.p.shape[0], 4)  # (n, K) with n=4

        # Sanity: all logs use same K
        K = log.t.shape[0]
        self.assertEqual(log.x.shape[1], K)
        self.assertEqual(log.q.shape[1], K)
        self.assertEqual(log.p.shape[1], K)

        # Final-state shapes (last column)
        xT = log.x[:, [-1]]  # keep as (4,1)
        qT = log.q[:, [-1]]  # keep as (4,1)
        pT = log.p[:, [-1]]  # keep as (4, 1)
        self.assertEqual(xT.shape, (4, 1))
        self.assertEqual(qT.shape, (4, 1))
        self.assertEqual(pT.shape, (4, 1))

        # Plot smoke test (no GUI) and patched compute_gne
        with patch_compute_gne_max_iter(sim.population_game, max_iter=5):
            run_plot_smoketest(
                sim=sim,
                plot_kwargs=dict(
                    plot_type="univariate",
                    plot_deterministic_approximation=True,
                    show=False,
                ),
            )

            run_plot_smoketest(
                sim=sim,
                plot_kwargs=dict(
                    plot_type="kpi",
                    plot_deterministic_approximation=True,
                    show=False,
                ),
            )

    def test_example_ride_hailing_1(self):
        Q = (
            -np.diag([0.7, 0.8, 0.7, 0.9, 0.7, 0.9])
            + np.diag([0.3, 0.2, 0.3], 3)
            + np.diag([0.1, 0.3, 0.1], -3)
        )
        r = np.array([-0.6, -0.7, -0.8, 0.1, 0.2, 0.3]).reshape(6, 1)

        def fitness_function(x):
            return np.dot(Q, x) + r

        population_game = pg.PopulationGame(
            num_populations=2,
            num_strategies=[3, 3],
            fitness_function=fitness_function,
            masses=[2, 1],
        )

        payoff_mechanism = pg.PayoffMechanism(h_map=fitness_function, n=6)

        revision_processes = [
            pg.PoissonRevisionProcess(
                Poisson_clock_rate=0.1, revision_protocol=pg.protocol.Smith(scale=0.5)
            ),
            pg.PoissonRevisionProcess(
                Poisson_clock_rate=1.0, revision_protocol=pg.protocol.BNN(scale=0.5)
            ),
        ]

        sim = pg.Simulator(
            population_game=population_game,
            payoff_mechanism=payoff_mechanism,
            revision_processes=revision_processes,
            num_agents=[2000, 1000],
        )

        x0 = np.array([1, 1, 0, 0, 1, 0]).reshape(6, 1)
        sim.reset(x0, q0=None)

        # Run reduced simulation and capture the flattened log (sim._get_flattened_log)
        result = run_sim_and_capture(sim=sim, T_sim=2)

        # The simulator's output is the flattened log.
        log = result.snapshots["log"]

        # Shape checks (robust to number of logged steps K)
        self.assertEqual(log.t.ndim, 1)  # (K,)
        self.assertEqual(log.x.shape[0], 6)  # (n, K) with n=6
        self.assertEqual(log.q.shape[0], 0)  # (d, K) with d=0
        self.assertEqual(log.p.shape[0], 6)  # (n, K) with n=6

        # Sanity: all logs use same K
        K = log.t.shape[0]
        self.assertEqual(log.x.shape[1], K)
        self.assertEqual(log.q.shape[1], K)
        self.assertEqual(log.p.shape[1], K)

        # Final-state shapes (last column)
        xT = log.x[:, [-1]]  # keep as (6,1)
        qT = log.q[:, [-1]]  # keep as (0,1)
        pT = log.p[:, [-1]]  # keep as (6, 1)
        self.assertEqual(xT.shape, (6, 1))
        self.assertEqual(qT.shape, (0, 1))
        self.assertEqual(pT.shape, (6, 1))

        # Plot smoke test (no GUI) and patched compute_gne
        with patch_compute_gne_max_iter(sim.population_game, max_iter=5):
            run_plot_smoketest(
                sim=sim,
                plot_kwargs=dict(
                    plot_type="ternary",
                    plot_deterministic_approximation=True,
                    show=False,
                ),
            )

    def test_example_ride_hailing_2(self):
        Q = (
            -np.diag([0.7, 0.8, 0.7, 0.9, 0.7, 0.9])
            + np.diag([0.3, 0.2, 0.3], 3)
            + np.diag([0.1, 0.3, 0.1], -3)
        )
        r = np.array([-0.6, -0.7, -0.8, 0.1, 0.2, 0.3]).reshape(6, 1)
        A = np.array([[1, 0, 0, -1, 0, 0]])
        b = np.array([[0.4]])

        def fitness_function(x):
            return np.dot(Q, x) + r

        def w_map(q, x):
            c = np.dot(A, x) - b
            ub = np.maximum(0.2 - q, 0)
            return ub * np.maximum(c, 0) - q * np.maximum(-c, 0)

        def h_map(q, x):
            return fitness_function(x) - np.dot(A.T, q)

        population_game = pg.PopulationGame(
            num_populations=2,
            num_strategies=[3, 3],
            fitness_function=fitness_function,
            masses=[2, 1],
            A_ineq=A,
            b_ineq=b,
        )
        payoff_mechanism = pg.PayoffMechanism(h_map=h_map, n=6, w_map=w_map, d=1)
        revision_processes = [
            pg.PoissonRevisionProcess(
                Poisson_clock_rate=0.1, revision_protocol=pg.protocol.Smith(scale=0.5)
            ),
            pg.PoissonRevisionProcess(
                Poisson_clock_rate=1.0, revision_protocol=pg.protocol.BNN(scale=0.5)
            ),
        ]
        sim = pg.Simulator(
            population_game=population_game,
            payoff_mechanism=payoff_mechanism,
            revision_processes=revision_processes,
            num_agents=[2000, 1000],
        )
        x0 = np.array([0.78, 0.63, 0.59, 0.15, 0.50, 0.35]).reshape(6, 1)
        q0 = np.array([0]).reshape(1, 1)
        sim.reset(x0=x0, q0=q0)

        # Run reduced simulation and capture the flattened log (sim._get_flattened_log)
        result = run_sim_and_capture(sim=sim, T_sim=2)

        # The simulator's output is the flattened log.
        log = result.snapshots["log"]

        # Shape checks (robust to number of logged steps K)
        self.assertEqual(log.t.ndim, 1)  # (K,)
        self.assertEqual(log.x.shape[0], 6)  # (n, K) with n=6
        self.assertEqual(log.q.shape[0], 1)  # (d, K) with d=1
        self.assertEqual(log.p.shape[0], 6)  # (n, K) with n=6

        # Sanity: all logs use same K
        K = log.t.shape[0]
        self.assertEqual(log.x.shape[1], K)
        self.assertEqual(log.q.shape[1], K)
        self.assertEqual(log.p.shape[1], K)

        # Final-state shapes (last column)
        xT = log.x[:, [-1]]  # keep as (6,1)
        qT = log.q[:, [-1]]  # keep as (1,1)
        pT = log.p[:, [-1]]  # keep as (6, 1)
        self.assertEqual(xT.shape, (6, 1))
        self.assertEqual(qT.shape, (1, 1))
        self.assertEqual(pT.shape, (6, 1))

        # Plot smoke test (no GUI) and patched compute_gne
        with patch_compute_gne_max_iter(sim.population_game, max_iter=5):
            run_plot_smoketest(
                sim=sim,
                plot_kwargs=dict(
                    plot_type="ternary",
                    plot_deterministic_approximation=True,
                    show=False,
                ),
            )

            run_plot_smoketest(
                sim=sim,
                plot_kwargs=dict(
                    plot_type="univariate_split",
                    plot_deterministic_approximation=True,
                    show=False,
                ),
            )

    def test_example_ride_hailing_3(self):
        Q = (
            -np.diag([0.7, 0.8, 0.7, 0.9, 0.7, 0.9])
            + np.diag([0.3, 0.2, 0.3], 3)
            + np.diag([0.1, 0.3, 0.1], -3)
        )
        r = np.array([-0.6, -0.7, -0.8, 0.1, 0.2, 0.3]).reshape(6, 1)
        A = np.array([[1, 0, 0, -1, 0, 0], [0, -1, 0, 0, -1, 0]])
        b = np.array([[0.4], [-1.6]])

        def fitness_function(x):
            return np.dot(Q, x) + r

        def w_map(q, x):
            c = np.dot(A, x) - b
            ub = np.maximum(0.2 - q, 0)
            return ub * np.maximum(c, 0) - q * np.maximum(-c, 0)

        def h_map(q, x):
            return fitness_function(x) - np.dot(A.T, q)

        population_game = pg.PopulationGame(
            num_populations=2,
            num_strategies=[3, 3],
            fitness_function=fitness_function,
            masses=[2, 1],
            A_ineq=A,
            b_ineq=b,
        )
        payoff_mechanism = pg.PayoffMechanism(h_map=h_map, n=6, w_map=w_map, d=2)
        revision_processes = [
            pg.PoissonRevisionProcess(
                Poisson_clock_rate=0.1, revision_protocol=pg.protocol.Smith(scale=0.5)
            ),
            pg.PoissonRevisionProcess(
                Poisson_clock_rate=1.0, revision_protocol=pg.protocol.BNN(scale=0.5)
            ),
        ]
        sim = pg.Simulator(
            population_game=population_game,
            payoff_mechanism=payoff_mechanism,
            revision_processes=revision_processes,
            num_agents=[2000, 1000],
        )
        x0 = np.array([0.67, 0.69, 0.64, 0.27, 0.44, 0.29]).reshape(6, 1)
        q0 = np.array([0.175, 0]).reshape(2, 1)
        sim.reset(x0=x0, q0=q0)

        # Run reduced simulation and capture the flattened log (sim._get_flattened_log)
        result = run_sim_and_capture(sim=sim, T_sim=2)

        # The simulator's output is the flattened log.
        log = result.snapshots["log"]

        # Shape checks (robust to number of logged steps K)
        self.assertEqual(log.t.ndim, 1)  # (K,)
        self.assertEqual(log.x.shape[0], 6)  # (n, K) with n=6
        self.assertEqual(log.q.shape[0], 2)  # (d, K) with d=2
        self.assertEqual(log.p.shape[0], 6)  # (n, K) with n=6

        # Sanity: all logs use same K
        K = log.t.shape[0]
        self.assertEqual(log.x.shape[1], K)
        self.assertEqual(log.q.shape[1], K)
        self.assertEqual(log.p.shape[1], K)

        # Final-state shapes (last column)
        xT = log.x[:, [-1]]  # keep as (6,1)
        qT = log.q[:, [-1]]  # keep as (2,1)
        pT = log.p[:, [-1]]  # keep as (6, 1)
        self.assertEqual(xT.shape, (6, 1))
        self.assertEqual(qT.shape, (2, 1))
        self.assertEqual(pT.shape, (6, 1))

        # Plot smoke test (no GUI) and patched compute_gne
        with patch_compute_gne_max_iter(sim.population_game, max_iter=5):
            run_plot_smoketest(
                sim=sim,
                plot_kwargs=dict(
                    plot_type="ternary",
                    plot_deterministic_approximation=True,
                    show=False,
                ),
            )

            run_plot_smoketest(
                sim=sim,
                plot_kwargs=dict(
                    plot_type="univariate_split",
                    plot_deterministic_approximation=True,
                    show=False,
                ),
            )

    def test_example_urban_traffic(self):
        def fitness_function(x):
            return np.dot(
                -np.array(
                    [
                        [2, 1, 0, 0, 0],
                        [1, 2, 0, 0, 0],
                        [0, 0, 2, 0, 0],
                        [0, 0, 0, 2, 1],
                        [0, 0, 0, 1, 2],
                    ]
                ),
                x,
            )

        # Constraints: Ax - b <= 0
        A = np.array([[1, 1, 0, 0, 0], [0, 0, 0, 1, 1]])
        b = np.array([[0.3], [0.2]])

        def h_map(q, x):
            return fitness_function(x) - np.dot(A.T, q)

        def w_map(q, x):
            c = np.dot(A, x) - b
            return np.maximum(c, 0) - q * np.maximum(-c, 0)

        population_game = pg.SinglePopulationGame(
            num_strategies=5, fitness_function=fitness_function, A_ineq=A, b_ineq=b
        )

        payoff_mechanism = pg.PayoffMechanism(h_map=h_map, n=5, w_map=w_map, d=2)

        revision_process = pg.PoissonRevisionProcess(
            Poisson_clock_rate=1, revision_protocol=pg.protocol.Smith(scale=0.1)
        )

        sim = pg.Simulator(
            population_game=population_game,
            payoff_mechanism=payoff_mechanism,
            revision_processes=revision_process,
            num_agents=500,
        )

        x0 = np.array([0.5, 0, 0, 0, 0.5]).reshape(5, 1)
        sim.reset(x0=x0)

        # Run reduced simulation and capture the flattened log (sim._get_flattened_log)
        result = run_sim_and_capture(sim=sim, T_sim=2)

        # The simulator's output is the flattened log.
        log = result.snapshots["log"]

        # Shape checks (robust to number of logged steps K)
        self.assertEqual(log.t.ndim, 1)  # (K,)
        self.assertEqual(log.x.shape[0], 5)  # (n, K) with n=5
        self.assertEqual(log.q.shape[0], 2)  # (d, K) with d=2
        self.assertEqual(log.p.shape[0], 5)  # (n, K) with n=5

        # Sanity: all logs use same K
        K = log.t.shape[0]
        self.assertEqual(log.x.shape[1], K)
        self.assertEqual(log.q.shape[1], K)
        self.assertEqual(log.p.shape[1], K)

        # Final-state shapes (last column)
        xT = log.x[:, [-1]]  # keep as (5,1)
        qT = log.q[:, [-1]]  # keep as (2,1)
        pT = log.p[:, [-1]]  # keep as (5, 1)
        self.assertEqual(xT.shape, (5, 1))
        self.assertEqual(qT.shape, (2, 1))
        self.assertEqual(pT.shape, (5, 1))

        # Plot smoke test (no GUI) and patched compute_gne
        with patch_compute_gne_max_iter(sim.population_game, max_iter=5):
            run_plot_smoketest(
                sim=sim,
                plot_kwargs=dict(
                    plot_type="univariate",
                    ylim={"x": [0, 1]},
                    show=False,
                ),
            )

            run_plot_smoketest(
                sim=sim,
                plot_kwargs=dict(
                    plot_type="kpi",
                    plot_deterministic_approximation=True,
                    show=False,
                ),
            )

    def test_example_microgrids(self):
        W = np.array(
            [
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 1, 0],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 0, 0],
            ]
        )
        In = np.eye(3)
        L_bar = np.kron(np.diag(W.sum(axis=1)) - W, In)
        A_bar = block_diag(In, In, -In, -In, -In)
        b_bar = np.zeros((15, 1))
        tou1, tou2 = np.array([[1], [1], [3]]), np.array([[3], [1], [1]])
        alpha, taub = np.vstack([tou1, tou1, tou2, tou2, tou2]), 0.1

        def f(x):
            return -alpha

        def w_map(q, x):
            qa, qb1, qb2 = q[:15], q[15:30], q[30:]
            dqa = np.zeros((15, 1))
            dqb1 = -np.dot(L_bar, qb1) - np.dot(L_bar.T, qb2) + np.dot(A_bar, x) - b_bar
            dqb2 = np.dot(L_bar, qb1)
            return np.vstack([dqa, taub * dqb1, taub * dqb2])

        def h_map(q, x):
            return f(x) - np.dot(A_bar.T, q[15:30])

        population_game = pg.PopulationGame(
            num_populations=5,
            num_strategies=[3, 3, 3, 3, 3],
            fitness_function=f,
            masses=[1, 2, 1, 1, 1],
            A_eq=np.hstack([In, In, -In, -In, -In]),
            b_eq=np.zeros((3, 1)),
        )
        payoff_mechanism = pg.PayoffMechanism(h_map=h_map, n=15, w_map=w_map, d=45)
        revision_processes = [
            pg.PoissonRevisionProcess(1.0, pg.protocol.Smith(0.1)),
            pg.PoissonRevisionProcess(1.0, pg.protocol.Smith(0.1)),
            pg.PoissonRevisionProcess(1.0, pg.protocol.BNN(0.1)),
            pg.PoissonRevisionProcess(1.0, pg.protocol.Smith(0.1)),
            pg.PoissonRevisionProcess(1.0, pg.protocol.Softmax(0.01)),
        ]
        sim = pg.Simulator(
            population_game=population_game,
            payoff_mechanism=payoff_mechanism,
            revision_processes=revision_processes,
            num_agents=[100, 200, 100, 100, 100],
        )
        aux = (1 / 3) * np.ones((3, 1))
        sim.reset(x0=np.vstack([aux, 2 * aux, aux, aux, aux]), q0=np.zeros((45, 1)))

        # Run reduced simulation and capture the flattened log (sim._get_flattened_log)
        result = run_sim_and_capture(sim=sim, T_sim=2)

        # The simulator's output is the flattened log.
        log = result.snapshots["log"]

        # Shape checks (robust to number of logged steps K)
        self.assertEqual(log.t.ndim, 1)  # (K,)
        self.assertEqual(log.x.shape[0], 15)  # (n, K) with n=15
        self.assertEqual(log.q.shape[0], 45)  # (d, K) with d=45
        self.assertEqual(log.p.shape[0], 15)  # (n, K) with n=15

        # Sanity: all logs use same K
        K = log.t.shape[0]
        self.assertEqual(log.x.shape[1], K)
        self.assertEqual(log.q.shape[1], K)
        self.assertEqual(log.p.shape[1], K)

        # Final-state shapes (last column)
        xT = log.x[:, [-1]]  # keep as (15,1)
        qT = log.q[:, [-1]]  # keep as (45,1)
        pT = log.p[:, [-1]]  # keep as (15, 1)
        self.assertEqual(xT.shape, (15, 1))
        self.assertEqual(qT.shape, (45, 1))
        self.assertEqual(pT.shape, (15, 1))

        # Plot smoke test (no GUI) and patched compute_gne
        with patch_compute_gne_max_iter(sim.population_game, max_iter=5):
            run_plot_smoketest(
                sim=sim,
                plot_kwargs=dict(
                    plot_type="kpi",
                    plot_deterministic_approximation=True,
                    show=False,
                ),
            )

            run_plot_smoketest(
                sim=sim,
                plot_kwargs=dict(
                    plot_type="ternary",
                    plot_deterministic_approximation=True,
                    show=False,
                ),
            )


if __name__ == "__main__":
    unittest.main()
