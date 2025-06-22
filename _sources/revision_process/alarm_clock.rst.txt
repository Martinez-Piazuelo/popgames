Alarm Clock
===================

---

**Base Class**

The abstract base class for all alarm clocks is defined below.

.. autoclass:: popgames.alarm_clock.AlarmClockABC
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __call__

---

**Poisson Alarm Clock**

An alarm clock where inter-revision times are independently sampled from an exponential distribution.

.. autoclass:: popgames.alarm_clock.Poisson
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __call__

