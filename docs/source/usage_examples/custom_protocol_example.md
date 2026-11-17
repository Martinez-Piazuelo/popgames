# Custom Revision Protocols

A custom revision protocol is defined by subclassing
{class}`~popgames.revision_protocol.RevisionProtocolABC` and implementing ``__call__(p, x)``, which receives the
payoff vector $p$ and the population state $x$ (both with shape ``(n, 1)``) and returns the ``(n, n)`` matrix of
switching probabilities.

This example re-implements the replicator (pairwise proportional imitation) protocol, which is also available as
the built-in {class}`~popgames.revision_protocol.Replicator`, and checks that both give the same trajectory in the
[Rock-Paper-Scissors](rock_paper_scissors_example.md) game.

```{literalinclude} ../../examples/custom_protocol.py
:language: python
:linenos:
```

```{image} ../_static/custom_protocol.png
:width: 70%
:align: center
:alt: Replicator dynamics with the custom and the built-in protocol
```

Custom protocols work in both simulation modes. The [Numba backend](../performance/index.md) only supports the
built-in protocol classes, so a simulator with a custom protocol runs with the numpy backend.
