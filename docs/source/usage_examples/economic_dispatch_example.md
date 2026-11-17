# Economic Dispatch in a Microgrid

This example reproduces Case Study I of the PopGames paper. Three distributed generators (DGs) share a power
demand $P_D$. Each strategy is a generator, and $x_i$ is the power supplied by DG$_i$, so the population mass is
$P_D$ and the simplex constraint $\sum_i x_i = P_D$ is the power balance.

The fitness of each generator decreases with its load,

$$
f_i(x) = c_i \left(1 - \frac{x_i}{P_{\max,i}}\right),
$$

where $c_i$ is an efficiency coefficient and $P_{\max,i}$ the capacity of DG$_i$. Since $f_i$ depends only on $x_i$,
the game is a potential game with potential $\Phi(x) = \sum_i c_i \left(x_i - x_i^2 / (2 P_{\max,i})\right)$,
which is drawn as the background heatmap. Its maximizer $x^*$ is the Nash equilibrium (NE), computed in
closed form in the script.

The payoff mechanism is memoryless ($p = f(x)$), and three built-in protocols (Smith, BNN, and Replicator)
are compared from the same initial condition near the DG$_3$ vertex. In each panel, the black line is a
finite-agent simulation with 500 agents and the dotted line is the deterministic EDM-PDM approximation.

```{literalinclude} ../../examples/economic_dispatch.py
:language: python
:linenos:
```

```{image} ../_static/economic_dispatch.png
:width: 100%
:align: center
:alt: Ternary plots of the economic dispatch game under the Smith, BNN, and Replicator protocols
```

The capacity bounds $x_i \le P_{\max,i}$ are not active at $x^*$ and are not enforced by the memoryless
payoff mechanism. With the [Numba backend](../performance/index.md), add ``backend="numba"`` to the
simulator to speed up the finite-agent runs.
