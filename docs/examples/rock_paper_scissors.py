import numpy as np
import matplotlib.pyplot as plt
import ternary
from popgames import (
    SinglePopulationGame,
    PayoffMechanism,
    PoissonRevisionProcess,
    Simulator,
)
from popgames.revision_protocol import Replicator, Smith

A = np.array([[0, -1, 1],    # Rock
              [1, 0, -1],    # Paper
              [-1, 1, 0]])   # Scissors

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
T_sim = 100

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for ax, (name, revision_protocol) in zip(axes, revision_protocols.items()):
    revision_process = PoissonRevisionProcess(
        Poisson_clock_rate=1,
        revision_protocol=revision_protocol,
    )

    sim = Simulator(
        population_game=population_game,
        payoff_mechanism=payoff_mechanism,
        revision_processes=revision_process,
        num_agents=1000
    )

    sim.reset(x0=x0)
    out = sim.run(T_sim=T_sim)
    edm = sim.integrate_edm_pdm(t_span=(0, T_sim), x0=x0, t_eval=np.linspace(0, T_sim, 2000))

    _, tax = ternary.figure(ax=ax, scale=1)
    tax.boundary(linewidth=1.0)
    tax.plot(out.x[[2, 0, 1], :].T, linewidth=0.8, color='black', label='Finite agents')
    tax.plot(edm.x[[2, 0, 1], :].T, linewidth=1.5, linestyle='dotted', color='magenta', label='EDM')
    tax.scatter([(1/3, 1/3, 1/3)], marker='*', s=80, color='tab:red', zorder=3, label='NE')
    tax.top_corner_label('Rock')
    tax.left_corner_label('Paper')
    tax.right_corner_label('Scissors')
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
    tax.set_title(name, pad=25)

axes[0].legend(loc='upper left', fontsize=8)
plt.subplots_adjust(wspace=0.35)
plt.show()
