import numpy as np
import matplotlib.pyplot as plt
import popgames as pg

# Data center with 6 server zones sharing a cooling capacity C_total.
# x_i is the cooling power allocated to zone i [kW].
n = 6
T_amb, T_set, sigma = 25.0, 30.0, 2.0                          # [degC]
k = np.array([[2.0], [1.7], [2.5], [1.8], [2.2], [1.6]])       # conductances [kW/degC]
Q = k * (42.0 - T_amb)                                          # heat loads [kW]

# Allocation that keeps every zone at T_set; its total is the cooling capacity
x_star = Q - k * (T_set - T_amb)
C_total = float(x_star.sum())

def zone_temperature(x):
    return T_amb + (Q - x) / k

def fitness_function(x):
    # Gaussian fitness: maximal when the zone is at T_set
    return np.exp(-(zone_temperature(x) - T_set) ** 2 / (2 * sigma**2))

# Dynamic payoff mechanism (d = n): each q_i is a leaky integral of the
# allocation error x_i - x_i*, subtracted from the fitness
alpha, beta = 0.5, 0.1

def w_map(q, x):
    return alpha * (x - x_star) - beta * q

def h_map(q, x):
    return fitness_function(x) - q

population_game = pg.SinglePopulationGame(
    num_strategies=n,
    fitness_function=fitness_function,
    mass=C_total,
)
payoff_mechanism = pg.PayoffMechanism(h_map=h_map, w_map=w_map, n=n, d=n)

sim = pg.Simulator(
    population_game=population_game,
    payoff_mechanism=payoff_mechanism,
    revision_processes=pg.PoissonRevisionProcess(1, pg.protocol.Smith(scale=0.01)),
    num_agents=500,  # not used by the deterministic mode
)

# 20 random initial allocations, integrated deterministically for 60 minutes
rng = np.random.default_rng(0)
M = 20
X0s = rng.dirichlet(np.ones(n), size=M).reshape(M, n, 1) * C_total
ens = sim.integrate_ensemble(
    T_sim=60,
    x0=X0s,
    q0=np.zeros((M, n, 1)),
    t_eval=np.linspace(0, 60, 500),
)

# Zone temperatures: mean +/- 1 std over the initial conditions
fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), sharex=True, sharey=True)
ens.plot(
    plot_type='bands',
    statistic='mean_std',
    num_std=(1,),
    layout='per_strategy',
    variables=[],
    derived={'T': lambda run: zone_temperature(run.x)},
    reference={'T': T_set},
    ax=axes,
)
for i, ax in enumerate(axes.flat):
    ax.set_title(f'Zone {i + 1}')
    ax.set_xlabel('Time [min]' if i >= 3 else '')
    ax.set_ylabel(r'$T_i(t)$ [$^\circ$C]' if i in (0, 3) else '')
fig.tight_layout()
plt.show()
