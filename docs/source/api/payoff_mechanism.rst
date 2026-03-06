Payoff Mechanism
================

This module defines the core class used to specify and integrate payoff mechanisms
in population games.

A payoff mechanism is defined through a **payoff dynamics model (PDM)**, which governs
how payoffs evolve over time based on the current state of the system.

Numerical integration is performed using ``scipy``.

---

**Payoff Mechanism**

.. autoclass:: popgames.payoff_mechanism.PayoffMechanism
   :members:
   :undoc-members:
   :show-inheritance:
