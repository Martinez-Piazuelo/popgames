import numpy as np
from popgames import (
    SinglePopulationGame,
    PayoffMechanism,
    PoissonRevisionProcess,
    Simulator,
)
from popgames.revision_protocol import Replicator, Smith

A = np.array([[0, -1, 1],    # Rock
              [1, 0, -1],    # Paper
              [-1, 1, 0]],   # Scissors
             dtype=float)

def fitness_function(x):
    return np.dot(A, x)

population_game = SinglePopulationGame(
    num_strategies=3,
    fitness_function=fitness_function,
)

payoff_mechanism = PayoffMechanism(
    h_map=fitness_function,
    n=3,
)

revision_protocols = {
    'Replicator': Replicator(scale=0.5),
    'Smith': Smith(scale=0.25),
}

x0 = np.array([0.5, 0.3, 0.2]).reshape(3, 1)

for name, revision_protocol in revision_protocols.items():
    sim = Simulator(
        population_game=population_game,
        payoff_mechanism=payoff_mechanism,
        revision_processes=PoissonRevisionProcess(
            Poisson_clock_rate=1,
            revision_protocol=revision_protocol,
        ),
        num_agents=200,
    )

    # 50 independent runs from x0, sampled at 201 evenly spaced times in [0, 60]
    sim.reset(x0=x0)
    ens = sim.run_ensemble(T_sim=60, num_runs=50, seed=0)

    # x1 * x2 * x3 is conserved by the replicator dynamics (EDM), but not by the finite population
    print(f'{name}: x1 x2 x3 = {np.prod(x0):.4f} at t=0, '
          f'{np.prod(ens.final().x, axis=1).mean():.4f} at t=60 (mean over the runs)')

    ens.plot(plot_type='ternary', plot_deterministic_approximation=True)
    ens.plot(plot_type='bands', variables=['x'], plot_deterministic_approximation=True)
    ens.plot(plot_type='final', plot_deterministic_approximation=True)
