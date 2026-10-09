"""
Benchmark the main components of popgames.

Times the finite-agent simulation (fast and per-agent paths), the deterministic approximation (EDM-PDM integration),
and the GNE solver on two scenarios:

* Rock-Paper-Scissors with a memoryless payoff mechanism (d=0).
* The introductory book example with a dynamic payoff mechanism (d=1) and an equality constraint.

Usage:
    python benchmarks/bench_simulator.py [--num-agents 1000 10000] [--T-sim 10]
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

import popgames as pg


def make_rps(num_agents: int, **kwargs) -> pg.Simulator:
    A = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])

    def f(x):
        return A @ x

    sim = pg.Simulator(
        population_game=pg.SinglePopulationGame(
            num_strategies=3, fitness_function=f, fitness_lipschitz_constant=2.0
        ),
        payoff_mechanism=pg.PayoffMechanism(h_map=f, n=3),
        revision_processes=pg.PoissonRevisionProcess(1.0, pg.protocol.Smith(0.25)),
        num_agents=num_agents,
        seed=0,
        **kwargs,
    )
    sim.reset(x0=np.array([[0.5], [0.3], [0.2]]))
    return sim


def make_intro(num_agents: int, **kwargs) -> pg.Simulator:
    Q = np.diag([-0.6, -0.7, -0.8])
    r = np.array([[0.0], [0.1], [0.1]])
    A, b = np.array([[0.5, 0.5, 0.0]]), np.array([[0.2]])

    def f(x):
        return Q @ x + r

    sim = pg.Simulator(
        population_game=pg.SinglePopulationGame(
            num_strategies=3,
            fitness_function=f,
            A_eq=A,
            b_eq=b.reshape(-1),
            fitness_lipschitz_constant=0.8,
        ),
        payoff_mechanism=pg.PayoffMechanism(
            h_map=lambda q, x: f(x) - A.T @ q,
            w_map=lambda q, x: A @ x - b,
            n=3,
            d=1,
        ),
        revision_processes=pg.PoissonRevisionProcess(1.0, pg.protocol.Softmax(0.01)),
        num_agents=num_agents,
        seed=0,
        **kwargs,
    )
    sim.reset(x0=np.array([[1.0], [0.0], [0.0]]), q0=np.zeros((1, 1)))
    return sim


def report(label: str, seconds: float, num_events: int = None) -> None:
    per_event = f"{1e6 * seconds / num_events:8.1f} us/event" if num_events else ""
    print(f"{label:55s} {seconds:8.3f} s {per_event}")


def time_run(label: str, sim: pg.Simulator, T_sim: int) -> None:
    start = time.perf_counter()
    sim.run(T_sim=T_sim)
    report(label, time.perf_counter() - start, len(sim.log.t) - 1)


def time_call(label: str, fn) -> None:
    start = time.perf_counter()
    fn()
    report(label, time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--num-agents", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument("--T-sim", type=int, default=10)
    args = parser.parse_args()
    logging.disable(logging.WARNING)

    print("== Finite agents ==")
    for N in args.num_agents:
        for fast_path in [True, False]:
            path = "fast" if fast_path else "per-agent"
            time_run(
                f"RPS (d=0)    N={N:<6} T={args.T_sim}  {path}",
                make_rps(N, fast_path=fast_path),
                args.T_sim,
            )
    N, T_intro = args.num_agents[0], max(1, args.T_sim // 5)
    for fast_path, pdm_method in [(True, "RK4"), (False, "RK4"), (False, "Radau")]:
        path = "fast" if fast_path else "per-agent"
        time_run(
            f"Intro (d=1)  N={N:<6} T={T_intro}  {path} + {pdm_method}",
            make_intro(N, fast_path=fast_path, pdm_method=pdm_method),
            T_intro,
        )

    print("== Deterministic approximation (EDM-PDM) ==")
    rps, intro = make_rps(1000), make_intro(1000)
    time_call(
        "RPS (d=0)    T=100 Radau",
        lambda: rps.integrate_edm_pdm((0, 100), np.array([[0.5], [0.3], [0.2]])),
    )
    time_call(
        "Intro (d=1)  T=20  Radau",
        lambda: intro.integrate_edm_pdm((0, 20), np.array([[1.0], [0.0], [0.0]])),
    )

    print("== GNE solver ==")
    time_call("RPS (d=0)", rps.population_game.compute_gne)
    time_call("Intro (d=1)", intro.population_game.compute_gne)


if __name__ == "__main__":
    main()
