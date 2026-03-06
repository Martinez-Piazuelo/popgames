Simulator
=========

This module defines the ``Simulator`` class, which provides numerical tools to simulate the
interplay between population games, payoff mechanisms, and revision processes.

The simulator tracks the evolution of strategic distributions and payoffs over time.
It supports both finite-agent simulations and numerical integration of deterministic dynamics models.

The simulator is equipped with a ``VisualizationMixin`` implementing useful visualization tools.

---

**Simulator**

.. autoclass:: popgames.simulator.Simulator
   :members:
   :undoc-members:
   :show-inheritance:

---

**Visualization Mixin**

.. autoclass:: popgames.plotting.visualization_mixin.VisualizationMixin
   :members:
   :undoc-members:
   :show-inheritance:

---

**Supported Plots**

.. automodule:: popgames.plotting.plotters
   :members:
   :undoc-members:
   :show-inheritance: