Revision Process
================

The revision process models how agents decide **when** and **how** to revise their strategies.
It consists of two components:

- An **alarm clock**, which determines the timing of revision events.
- A **revision protocol**, which governs the choice of strategy after a revision.

.. toctree::
   :maxdepth: 1
   :caption: Components

   alarm_clock
   revision_protocol

---

**Base Class**

The abstract base class for all revision processes is defined below.

.. autoclass:: popgames.revision_process.RevisionProcessABC
   :members:
   :undoc-members:
   :show-inheritance:

---

**Poisson Revision Process**

The canonical **Poisson** revision process is implemented as follows:

.. autoclass:: popgames.revision_process.PoissonRevisionProcess
   :members:
   :undoc-members:
   :show-inheritance:
