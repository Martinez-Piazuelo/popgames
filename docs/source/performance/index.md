# Performance

Simulating finite populations means simulating every **revision event**: on average, a population of $N$ agents
with Poisson alarm clocks of rate $\lambda$ produces $N \lambda$ revisions per unit of time. A simulation of
$N = 10^5$ agents over $T = 10$ time units thus requires about $10^6$ revision events. This page explains the options
that ``popgames`` offers to simulate them efficiently.

## Simulation paths

The {class}`~popgames.Simulator` supports two ways of simulating revision events:

* **Fast path** (default, ``fast_path=True``). When all revision processes are
  {class}`~popgames.PoissonRevisionProcess` instances, agents are interchangeable and their alarm clocks are
  independent Poisson processes. The simulator then only needs to track *how many* agents play each strategy: the
  time to the next revision is exponentially distributed, and the revising agent is chosen uniformly at random. The
  cost of each revision event does not depend on the number of agents.
* **Per-agent path** (``fast_path=False``, or whenever a revision process is not a
  {class}`~popgames.PoissonRevisionProcess`). The simulator tracks the strategy and the alarm clock of every agent.
  This path supports custom alarm clocks and revision processes, but its cost grows with the number of agents.

Both paths simulate the same stochastic process. Forcing the per-agent path is mostly useful for comparison.

## Backends

The fast path can run with two backends:

* ``backend="numpy"`` (default): the event loop is written in Python and NumPy.
* ``backend="numba"``: the event loop, the built-in revision protocols, and your payoff mechanism functions are
  compiled to machine code with [Numba](https://numba.pydata.org/). It requires the optional dependency:

  ```bash
  pip install "popgames[numba]"
  ```

```python
sim = Simulator(
    population_game=population_game,
    payoff_mechanism=payoff_mechanism,
    revision_processes=revision_process,
    num_agents=100_000,
    backend="numba",
)
```

Indicative timings per revision event (Rock-Paper-Scissors with the Smith protocol, memoryless payoffs; and a
payoff dynamics model with $d=1$), as measured by ``benchmarks/bench_simulator.py``:

| Scenario              | Per-agent path | Fast path (numpy) | Fast path (numba) |
|-----------------------|---------------:|------------------:|------------------:|
| $N=10^3$, $d=0$       |          67 µs |             26 µs |            0.3 µs |
| $N=10^5$, $d=0$       |         362 µs |             24 µs |            0.2 µs |
| $N=10^3$, $d=1$       |         125 µs |             70 µs |            2.2 µs |

### When does the numba backend pay off?

Compilation has a cost. The compiled event loop is cached on disk, so it is compiled only once (the very first time,
which takes a few seconds). Your payoff mechanism functions, however, are compiled in **every session**, which
typically takes around one second. Dividing this cost by the time saved per revision event gives the break-even point:

* memoryless payoffs ($d=0$, about 25 µs saved per event): about $4 \cdot 10^4$ revision events, e.g.,
  $N = 4000$ agents over $T = 10$ time units;
* payoff dynamics models ($d>0$, about 70 µs saved per event in the benchmark): about $1.5 \cdot 10^4$ revision events.

The very first run on a machine also compiles the event loop (a few seconds), which raises these numbers for that run
only. Below the break-even point, e.g., for the [usage examples](../usage_examples/index.md), the numpy backend is
just as fast or faster.

Compiled functions are cached per function object: simulators that share the same functions (e.g., in a parameter
sweep over the revision protocol or the number of agents) only compile them once per session.

### Requirements and fallback

The numba backend requires:

* the fast path (``fast_path=True`` and {class}`~popgames.PoissonRevisionProcess` revision processes),
* built-in revision protocols ({class}`~popgames.revision_protocol.Softmax`,
  {class}`~popgames.revision_protocol.Smith`, {class}`~popgames.revision_protocol.BNN`,
  {class}`~popgames.revision_protocol.Replicator`, {class}`~popgames.revision_protocol.CCSmith`),
* ``pdm_method="RK4"`` (the default) for payoff dynamics models with $d>0$,
* payoff mechanism functions that Numba can compile (see below).

If any of these requirements is not met, the simulator logs a warning explaining why and uses the numpy backend
instead, so your script keeps working. You can check the backend in use with ``sim.backend``.

## Compiling your functions

### Automatic compilation

With ``backend="numba"``, you do not need to write any Numba-specific code: the simulator compiles the functions of
your payoff mechanism (the fitness function ``h_map(x)`` for memoryless payoffs, or ``h_map(q, x)`` and
``w_map(q, x)`` for payoff dynamics models) with
[``numba.njit``](https://numba.readthedocs.io/en/stable/user/jit.html) automatically.

Numba compiles a [subset of Python and NumPy](https://numba.readthedocs.io/en/stable/reference/numpysupported.html),
which covers most fitness functions: matrix products, element-wise operations, and mathematical functions such as
``np.exp``. A few rules to keep in mind:

* **Use floating-point arrays.** The simulator passes ``float64`` arrays of shape ``(n, 1)`` (and ``(d, 1)``), and
  your functions must return ``float64`` arrays of the expected shape. In particular, Numba does not support matrix
  products between integer and floating-point arrays, so declare your matrices as floats:

  ```python
  A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)  # not an integer matrix

  def fitness_function(x):
      return A @ x
  ```

* **Every function you call must be compilable.** Numba cannot call plain Python functions from compiled code. Write
  the computation inline, or decorate the helper functions with ``@numba.njit``:

  ```python
  def f(x):
      return Q @ x + r

  def h_map(q, x):
      return f(x) - A.T @ q          # fails: f is a plain Python function

  def h_map(q, x):
      return Q @ x + r - A.T @ q     # works
  ```

* **Global variables are constants.** Arrays and numbers referenced by your functions (global variables or closure
  variables) are frozen when the function is compiled. If you change them afterwards, create new functions (or new
  simulators with new functions) so that they are compiled again.

If a function cannot be compiled, the simulator logs a warning with the first line of Numba's error message and uses
the numpy backend. Enable debug logging (``popgames.configure_logging("DEBUG")``) to see the full error.

### Explicit compilation

If you prefer to be explicit, decorate your functions with ``@numba.njit`` yourself. The simulator then uses them as
they are (and, as plain callables, they also work with the numpy backend):

```python
import numba
import numpy as np

A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)

@numba.njit
def fitness_function(x):
    return A @ x
```

Explicit compilation lets you choose Numba options (e.g., ``@numba.njit(fastmath=True)``), and it surfaces
compilation errors where the function is defined: call it once with a test input to check it compiles.

## Reproducibility

Pass ``seed`` (an integer or a ``np.random.Generator``) to the simulator for reproducible simulations. Without a
seed, the simulator draws from NumPy's global random state, so ``np.random.seed`` also makes simulations
reproducible. The numpy and numba backends use different random streams: the same seed produces different (but
statistically equivalent) trajectories with each backend.

## Other options

* ``run(T_sim, log_interval=...)`` logs the state at most once every ``log_interval`` time units, instead of after
  every revision event. This limits memory usage in large simulations.
* ``pdm_method`` and ``pdm_max_step`` set how payoff dynamics models are integrated between revision events. The
  default fixed-step ``"RK4"`` method is fast; any method of ``scipy.integrate.solve_ivp`` (e.g., ``"Radau"`` for
  stiff dynamics) can be used with the numpy backend.
