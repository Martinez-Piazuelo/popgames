# Rock-Paper-Scissors

This example models a population of agents playing the classic **Rock-Paper-Scissors** game,
where each strategy beats one strategy and loses to the other.
The unique Nash equilibrium (NE) is the uniform distribution $(1/3, 1/3, 1/3)$.

It compares two revision protocols and shows how they can lead to very different long-run behavior,
even when the game has the same equilibrium:

* **Replicator** (pairwise proportional imitation): agents imitate better-performing opponents.
  The resulting evolutionary dynamics model (EDM) is the replicator dynamics, whose trajectories are
  **closed orbits** around the NE (the product $x_1 x_2 x_3$ is conserved).
  With finitely many agents, random fluctuations make the population drift across orbits.
* **Smith**: agents switch to strategies with higher payoff than their current one,
  regardless of how popular they are. The population **converges** to the NE.

```{literalinclude} ../../examples/rock_paper_scissors.py
:language: python
:linenos:
```

```{image} ../_static/rock_paper_scissors.png
:width: 90%
:align: center
:alt: Rock-Paper-Scissors trajectories under the Replicator and Smith protocols
```
