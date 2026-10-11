import numpy as np
from popgames import (
    SinglePopulationGame,
    PayoffMechanism,
    PoissonRevisionProcess,
    Simulator,
)
from popgames.revision_protocol import Softmax
from popgames.utilities import sample_initial_states

# Coordination game: f_i(x) = a_i * x_i
a = np.array([1.0, 2.0, 3.0]).reshape(3, 1)

def fitness_function(x):
    return a * x

population_game = SinglePopulationGame(
    num_strategies=3,
    fitness_function=fitness_function,
)

payoff_mechanism = PayoffMechanism(
    h_map=fitness_function,
    n=3,
)

revision_process = PoissonRevisionProcess(
    Poisson_clock_rate=1,
    revision_protocol=Softmax(eta=0.05),
)

sim = Simulator(
    population_game=population_game,
    payoff_mechanism=payoff_mechanism,
    revision_processes=revision_process,
    num_agents=110,
)

# Case 1: same initial state, different seeds
x_mixed = np.array([6, 3, 2]).reshape(3, 1) / 11  # Mixed Nash equilibrium: a_i * x_i = 6/11 for all i

# The logit dynamic is not Nash stationary, so x_mixed is not a rest point of its EDM: with equal payoffs, revising
# agents pick each strategy with probability 1/3, so the drift (1/3, 1/3, 1/3) - x_mixed ~ (-0.21, 0.06, 0.15) sends
# the EDM to e3. Finite-agent runs follow the same drift plus O(1/sqrt(N)) noise, and x_mixed is close to the basin
# boundaries, so some runs end at e1 or e2 instead (fewer as N grows).
ens = sim.run_ensemble(T_sim=10, num_runs=100, x0=x_mixed, seed=0)

selected = np.bincount(ens.final().x.argmax(axis=1), minlength=3) / len(ens)
print(f'Fraction of runs ending at e1, e2, e3: {selected}')

ens.plot(plot_type='ternary', max_runs=100, plot_deterministic_approximation=True, plot_gne=False)
ens.plot(plot_type='final', t=2, plot_deterministic_approximation=True, plot_gne=False)

# Case 2: different initial states, same seed
x0s = sample_initial_states(population_game, num=10, seed=3)
ens = sim.run_ensemble(T_sim=10, x0=x0s, seed=[0] * len(x0s))

ens.plot(plot_type='ternary', plot_deterministic_approximation=True, plot_gne=False, alpha=0.8)
