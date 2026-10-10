"""
Benchmark the main components of popgames.

Times the finite-agent simulation (per-agent path, and fast path with the numpy and numba backends), the deterministic
approximation (EDM-PDM integration), and the GNE solver on two scenarios:

* Rock-Paper-Scissors with a memoryless payoff mechanism (d=0).
* The introductory book example with a dynamic payoff mechanism (d=1) and an equality constraint.

Usage:
    python benchmarks/bench_simulator.py [--num-agents 1000 10000] [--T-sim 10] [--backends numpy numba]

For the numba backend, the compilation time (first run of a session) is reported separately.
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
            h_map=lambda q, x: Q @ x + r - A.T @ q,  # no calls to f (numba compatible)
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
    out = sim.run(T_sim=T_sim)
    report(label, time.perf_counter() - start, len(out.t) - 1)


def configurations(backends: list[str]) -> list[tuple[str, dict]]:
    configs = [("per-agent", dict(fast_path=False))]
    for backend in backends:
        configs.append((f"fast/{backend}", dict(backend=backend)))
    return configs


def warm_up(make_sim, backends: list[str]) -> None:
    if "numba" in backends:
        start = time.perf_counter()
        make_sim(10, backend="numba").run(T_sim=1)
        report(
            "  numba compilation (first run of a session)", time.perf_counter() - start
        )


def time_call(label: str, fn) -> None:
    start = time.perf_counter()
    fn()
    report(label, time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--num-agents", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument("--T-sim", type=int, default=10)
    parser.add_argument(
        "--backends", nargs="+", default=["numpy"], choices=["numpy", "numba"]
    )
    args = parser.parse_args()
    logging.disable(logging.WARNING)

    print("== Finite agents ==")
    warm_up(make_rps, args.backends)
    for N in args.num_agents:
        for name, kwargs in configurations(args.backends):
            time_run(
                f"RPS (d=0)    N={N:<6} T={args.T_sim}  {name}",
                make_rps(N, **kwargs),
                args.T_sim,
            )
    warm_up(make_intro, args.backends)
    N, T_intro = args.num_agents[0], max(1, args.T_sim // 5)
    for name, kwargs in configurations(args.backends):
        time_run(
            f"Intro (d=1)  N={N:<6} T={T_intro}  {name}",
            make_intro(N, **kwargs),
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
