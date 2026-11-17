import numpy as np
import matplotlib.pyplot as plt
from popgames import (
    SinglePopulationGame,
    PayoffMechanism,
    PoissonRevisionProcess,
    Simulator,
)
from popgames.revision_protocol import RevisionProtocolABC, Replicator


# A custom revision protocol: subclass RevisionProtocolABC and implement __call__.
# Given the payoff vector p and the population state x (both with shape (n, 1)),
# it returns the (n, n) matrix of switching probabilities.
# This one re-implements the replicator (pairwise proportional imitation) protocol.
class MyReplicator(RevisionProtocolABC):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, p, x):
        return self.scale * (x / x.sum()) * np.maximum(p - p.T, 0)


# Rock-Paper-Scissors game
A = np.array([[0, -1, 1],
              [1, 0, -1],
              [-1, 1, 0]], dtype=float)

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

x0 = np.array([0.5, 0.3, 0.2]).reshape(3, 1)
t_eval = np.linspace(0, 30, 301)

# Integrate the deterministic EDM-PDM system with the custom and the built-in protocol
results = {}
for name, protocol in [('Custom', MyReplicator(scale=0.5)),
                       ('Built-in', Replicator(scale=0.5))]:
    sim = Simulator(
        population_game=population_game,
        payoff_mechanism=payoff_mechanism,
        revision_processes=PoissonRevisionProcess(Poisson_clock_rate=1, revision_protocol=protocol),
        num_agents=1000,
    )
    results[name] = sim.integrate_edm_pdm(t_span=(0, 30), x0=x0, t_eval=t_eval)

print('Max difference:', np.abs(results['Custom'].x - results['Built-in'].x).max())

# Plot both trajectories: they coincide
fig, ax = plt.subplots(figsize=(6, 3.5))
labels = ['Rock', 'Paper', 'Scissors']
for i in range(3):
    ax.plot(results['Built-in'].t, results['Built-in'].x[i], color=f'C{i}', label=labels[i])
    ax.plot(results['Custom'].t, results['Custom'].x[i], color='black', linestyle='dotted')
ax.plot([], [], color='black', linestyle='dotted', label='Custom')
ax.set_xlabel('Time')
ax.set_ylabel('$x_i(t)$')
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8)
plt.tight_layout()
plt.show()
