"""
Run the introductory example (single population, 3 strategies, PDM with d=1 enforcing an equality constraint).

The model is:

    fitness:   f(x) = Q x + r                     (Q diagonal)
    PDM:       q' = A x - b,   p = f(x) - A^T q   (drives the society towards A x = b)
    protocol:  Softmax(eta)

Examples:
    python scripts/run_concrete_example.py
    python scripts/run_concrete_example.py --backend numba --num-agents 100000 --T-sim 20 --log-interval 0.05
    python scripts/run_concrete_example.py --eta 0.05 --b 0.3 --x0 0.2 0.3 0.5 --plot
    python scripts/run_concrete_example.py --no-fast-path --pdm-method Radau --num-agents 500 --T-sim 5
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

import popgames as pg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--Q-diag",
        type=float,
        nargs=3,
        default=[-0.6, -0.7, -0.8],
        help="diagonal of Q in f(x) = Q x + r",
    )
    model.add_argument(
        "--r", type=float, nargs=3, default=[0.0, 0.1, 0.1], help="r in f(x) = Q x + r"
    )
    model.add_argument(
        "--A",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.0],
        help="constraint row A in A x = b",
    )
    model.add_argument(
        "--b", type=float, default=0.2, help="constraint level b in A x = b"
    )
    model.add_argument("--eta", type=float, default=0.01, help="Softmax noise level")
    model.add_argument(
        "--clock-rate", type=float, default=1.0, help="rate of the Poisson alarm clocks"
    )

    simulation = parser.add_argument_group("simulation")
    simulation.add_argument(
        "--backend",
        choices=["numpy", "numba"],
        default="numpy",
        help="simulation backend",
    )
    simulation.add_argument(
        "--num-agents", type=int, default=1000, help="number of agents"
    )
    simulation.add_argument("--T-sim", type=int, default=10, help="simulated time")
    simulation.add_argument(
        "--x0",
        type=float,
        nargs=3,
        default=[1.0, 0.0, 0.0],
        help="initial strategic distribution",
    )
    simulation.add_argument("--q0", type=float, default=0.0, help="initial PDM state")
    simulation.add_argument(
        "--seed", type=int, default=698, help="negative for no seed"
    )
    simulation.add_argument(
        "--log-interval",
        type=float,
        default=None,
        help="minimum time between log entries; omit to log every event",
    )
    simulation.add_argument(
        "--no-fast-path",
        action="store_true",
        help="force the per-agent simulation path",
    )
    simulation.add_argument(
        "--pdm-method",
        default="RK4",
        help="'RK4' or a scipy.integrate.solve_ivp method (numba: RK4 only)",
    )
    simulation.add_argument(
        "--pdm-max-step", type=float, default=0.01, help="maximum PDM integration step"
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--gne", action="store_true", help="compute the GNE and the distance to it"
    )
    output.add_argument(
        "--plot", action="store_true", help="ternary plot with the EDM-PDM trajectory"
    )
    output.add_argument("--save", default=None, help="save the plot to this PDF file")
    output.add_argument("--log-level", default="WARNING", help="logging level")
    return parser.parse_args()


def build_simulator(args: argparse.Namespace) -> pg.Simulator:
    Q = np.diag(args.Q_diag)
    r = np.array(args.r).reshape(-1, 1)
    A = np.array(args.A).reshape(1, -1)
    b = np.array([[args.b]])

    def f(x):  # Fitness function
        return np.dot(Q, x) + r

    def w(q, x):  # RHS of the payoff dynamics model
        return np.dot(A, x) - b

    def h(
        q, x
    ):  # Output of the payoff mechanism (f is inlined: numba cannot call plain Python functions)
        return np.dot(Q, x) + r - np.dot(A.T, q)

    population_game = pg.SinglePopulationGame(
        num_strategies=3,
        fitness_function=f,
        mass=1,
        A_eq=A,
        b_eq=b.reshape(-1),
        fitness_lipschitz_constant=float(np.max(np.abs(args.Q_diag))),
    )
    payoff_mechanism = pg.PayoffMechanism(h_map=h, w_map=w, n=3, d=1)
    revision_process = pg.PoissonRevisionProcess(
        Poisson_clock_rate=args.clock_rate,
        revision_protocol=pg.protocol.Softmax(eta=args.eta),
    )

    return pg.Simulator(
        population_game=population_game,
        payoff_mechanism=payoff_mechanism,
        revision_processes=revision_process,
        num_agents=args.num_agents,
        fast_path=not args.no_fast_path,
        pdm_method=args.pdm_method,
        pdm_max_step=args.pdm_max_step,
        seed=args.seed if args.seed >= 0 else None,
        backend=args.backend,
    )


def main() -> None:
    args = parse_args()
    pg.configure_logging(args.log_level)
    logging.getLogger("fontTools").setLevel(logging.ERROR)  # noisy when saving PDFs

    start = time.perf_counter()
    sim = build_simulator(args)
    setup_time = time.perf_counter() - start

    x0 = np.array(args.x0).reshape(-1, 1)
    q0 = np.array([[args.q0]])
    sim.reset(x0=x0, q0=q0)

    start = time.perf_counter()
    out = sim.run(T_sim=args.T_sim, log_interval=args.log_interval)
    run_time = time.perf_counter() - start

    path = "fast" if sim.uses_fast_path else "per-agent"
    print(f"backend: {sim.backend} ({path} path)")
    compile_note = " (includes compiling with numba)" if sim.backend == "numba" else ""
    print(f"setup:   {setup_time:8.3f} s{compile_note}")
    print(f"run:     {run_time:8.3f} s  ({len(out.t)} log entries)")
    print(f"x(T) = {np.round(out.x[:, -1], 4)}")
    print(f"q(T) = {np.round(out.q[:, -1], 4)}")
    print(f"p(T) = {np.round(out.p[:, -1], 4)}")
    A = np.array(args.A).reshape(1, -1)
    print(f"A x(T) - b = {(A @ out.x[:, [-1]] - args.b).item():.4f}")

    if args.gne:
        gne = sim.population_game.compute_gne()
        print(f"GNE  = {np.round(gne.ravel(), 4)}")
        print(f"|x(T) - GNE| = {np.linalg.norm(out.x[:, [-1]] - gne):.4f}")

    if args.plot or args.save:
        if not args.plot:  # only save: avoid opening a window
            import warnings

            import matplotlib

            matplotlib.use("Agg")
            warnings.filterwarnings("ignore", message=".*non-interactive.*")

        Q = np.diag(args.Q_diag)
        r = np.array(args.r).reshape(-1, 1)

        def varphi(x):  # Potential function to plot
            return 0.5 * np.dot(x.T, np.dot(Q, x)) + np.dot(r.T, x)

        sim.plot(
            plot_type="ternary",
            potential_function=varphi,
            plot_deterministic_approximation=True,
            filename=args.save,
            show=args.plot,
        )


if __name__ == "__main__":
    main()
