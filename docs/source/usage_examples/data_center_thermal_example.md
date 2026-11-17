# Thermal Control in a Data Center

This example reproduces, in a reduced form, Case Study II of the PopGames paper. A cooling capacity
$C_{\rm total}$ is shared among six server zones. Each strategy is a zone, and $x_i$ is the cooling power
allocated to zone $i$, whose steady-state temperature is $T_i = T_{\rm amb} + (Q_i - x_i)/k_i$.

The fitness is a Gaussian function of the temperature,

$$
f_i(x) = \exp\left(-\frac{(T_i(x_i) - T_{\rm set})^2}{2\sigma^2}\right),
$$

which is maximal when the zone is at the setpoint $T_{\rm set}$. This example illustrates two features:

* a **nonlinear fitness function**, and
* a **dynamic payoff mechanism** with internal state $q \in \mathbb{R}^6$. Each $q_i$ is a leaky integral of the
  allocation error $x_i - x_i^*$, and the payoff is $p = f(x) - q$. The maps $W(q, x)$ and $H(q, x)$ are passed
  as ``w_map`` and ``h_map``, with ``d=n``, and the initial state $q(0)$ is passed as ``q0``.

Because the Gaussian fitness is symmetric, equal fitness alone does not imply that all zones are at
$T_{\rm set}$. The payoff mechanism uses the reference allocation $x^*$, which keeps every zone at $T_{\rm set}$,
to select this equilibrium.

The deterministic EDM-PDM system with the Smith protocol is integrated from 20 random initial allocations with
``integrate_ensemble``, and the mean $\pm$ one standard deviation of the zone temperatures is plotted. In these
runs, all temperatures approach $T_{\rm set}$ (dashed) within one hour. This is an empirical observation for
these initial conditions, not a convergence guarantee.

```{literalinclude} ../../examples/data_center_thermal.py
:language: python
:linenos:
```

```{image} ../_static/data_center_thermal.png
:width: 100%
:align: center
:alt: Zone temperatures of the data center under the Smith protocol, mean and standard deviation over 20 initial conditions
```
