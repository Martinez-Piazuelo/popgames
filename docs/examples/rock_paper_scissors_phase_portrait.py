import numpy as np
from popgames import (
    SinglePopulationGame,
    PayoffMechanism,
    PoissonRevisionProcess,
    Simulator,
)
from popgames.revision_protocol import Replicator, Smith
from popgames.utilities import sample_initial_states

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

# 6 initial states, sampled uniformly at random on the simplex
x0s = sample_initial_states(population_game, num=6, seed=1)

for revision_protocol in [Replicator(scale=0.5), Smith(scale=0.25)]:
    sim = Simulator(
        population_game=population_game,
        payoff_mechanism=payoff_mechanism,
        revision_processes=PoissonRevisionProcess(
            Poisson_clock_rate=1,
            revision_protocol=revision_protocol,
        ),
        num_agents=200,
    )

    # 5 runs from each initial state (30 runs), with independent seeds
    ens = sim.run_ensemble(T_sim=30, x0=np.repeat(x0s, 5, axis=0), seed=0)

    ens.plot(plot_type='ternary', plot_deterministic_approximation=True, alpha=0.4)
