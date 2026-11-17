import numpy as np
import matplotlib.pyplot as plt
import popgames as pg
from popgames.plotting import add_shared_legend

# Microgrid with 3 distributed generators (DGs) sharing a demand P_D.
# x_i is the power supplied by DG_i, and the population mass is P_D.
P_max = np.array([[2.0], [3.0], [5.5]])  # capacities [kW]
c = np.array([[1.2], [0.5], [0.15]])     # efficiency coefficients
P_D = 5.5                                # demand [kW]

# Equilibrium in closed form (capacity bounds are inactive at x*)
mu = (P_max.sum() - P_D) / (P_max / c).sum()
x_star = P_max * (1 - mu / c)
print('x* =', x_star.ravel())

def fitness_function(x):
    # decreasing marginal utility: f_i = c_i (1 - x_i / P_max,i)
    return c * (1 - x / P_max)

def potential_function(x):
    # exact potential: dPhi/dx_i = f_i
    return np.array([[np.sum(c * (x - x**2 / (2 * P_max)))]])

population_game = pg.SinglePopulationGame(
    num_strategies=3,
    fitness_function=fitness_function,
    mass=P_D,
    fitness_lipschitz_constant=float(np.max(c / P_max)),
)

# Memoryless payoff mechanism: p(t) = f(x(t))
payoff_mechanism = pg.PayoffMechanism(h_map=fitness_function, n=3)

protocols = {
    'Smith': pg.protocol.Smith(scale=0.1),
    'BNN': pg.protocol.BNN(scale=0.1),
    'Replicator': pg.protocol.Replicator(scale=0.1),
}

x0 = np.array([[0.1], [0.1], [0.8]]) * P_D  # start near the DG3 vertex

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, (name, protocol) in zip(axes, protocols.items()):
    sim = pg.Simulator(
        population_game=population_game,
        payoff_mechanism=payoff_mechanism,
        revision_processes=pg.PoissonRevisionProcess(1, protocol),
        num_agents=500,
        seed=0,
    )
    sim.reset(x0=x0)
    sim.run(T_sim=300)
    sim.plot(
        plot_type='ternary',
        ax=ax,
        plot_deterministic_approximation=True,
        potential_function=potential_function,
        scale=30,
        legend=False,
    )
    ax.set_title(name, pad=25)

fig.subplots_adjust(left=0.03, right=0.97, wspace=0.25)
add_shared_legend(fig, loc='lower center', bbox_to_anchor=(0.5, 1.03))
plt.show()
