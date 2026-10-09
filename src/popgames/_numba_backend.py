"""
Numba backend for the fast simulation path.

This module is imported only when a simulator is created with ``backend="numba"``, and requires the optional
dependency ``numba`` (``pip install popgames[numba]``).

It compiles the event loop of the fast simulation path (see ``Simulator._run_fast_path``), the built-in revision
protocols, and the user-provided functions of the payoff mechanism. The event loop is compiled when this module is
first imported and cached on disk, so later sessions only compile the user-provided functions.
"""

from __future__ import annotations

import typing
import weakref

import numba
import numpy as np

from popgames.revision_protocol import BNN, CCSmith, Replicator, Smith, Softmax

if typing.TYPE_CHECKING:
    from typing import Callable, Optional

    from popgames.payoff_mechanism import PayoffMechanism
    from popgames.revision_protocol import RevisionProtocolABC


__all__ = [
    "compile_payoff_mechanism",
    "compile_user_function",
    "protocol_spec",
    "run_events",
]


# Identifiers of the built-in revision protocols supported by the numba backend
_SOFTMAX, _SMITH, _BNN, _REPLICATOR, _CCSMITH = range(5)


def protocol_spec(protocol: RevisionProtocolABC) -> Optional[tuple[int, np.ndarray]]:
    """
    Translate a built-in revision protocol into the (kind, parameters) pair used by the compiled kernels.

    Only instances of the built-in classes themselves are supported (not subclasses, which may override them).

    Args:
        protocol (RevisionProtocolABC): The revision protocol.

    Returns:
        Optional[tuple[int, np.ndarray]]: The protocol kind and its parameters, or None if not supported.
    """
    protocol_type = type(protocol)
    if protocol_type is Softmax:
        return _SOFTMAX, np.array([protocol.eta], dtype=np.float64)
    if protocol_type is Smith:
        return _SMITH, np.array([protocol.scale], dtype=np.float64)
    if protocol_type is BNN:
        return _BNN, np.array([protocol.scale], dtype=np.float64)
    if protocol_type is Replicator:
        return _REPLICATOR, np.array([protocol.scale], dtype=np.float64)
    if protocol_type is CCSmith:
        params = np.hstack([[protocol.scale], np.ravel(protocol.x_bar)])
        return _CCSMITH, params.astype(np.float64)
    return None


# Signature of the compiled payoff mechanism functions h_map(q, x) and w_map(q, x). The event loop receives them as
# first-class functions of this type, so it is compiled (and cached on disk) once for all user-provided functions.
_ARRAY_2D = numba.types.float64[:, ::1]
PDM_FUNCTION_SIGNATURE = _ARRAY_2D(_ARRAY_2D, _ARRAY_2D)
_PDM_FUNCTION = numba.types.FunctionType(PDM_FUNCTION_SIGNATURE)

# Compiled functions, cached per user-provided function
_compiled_fitness_functions = weakref.WeakKeyDictionary()
_compiled_pdm_functions = weakref.WeakKeyDictionary()


def _cached(
    cache: weakref.WeakKeyDictionary, key: Callable, build: Callable
) -> Callable:
    try:
        if key not in cache:
            cache[key] = build()
        return cache[key]
    except TypeError:  # key cannot be weakly referenced
        return build()


def _is_compiled(fn: Callable) -> bool:
    return isinstance(fn, numba.core.registry.CPUDispatcher)


def _compile_fitness(fitness: Callable) -> Callable:
    """
    Compile a fitness function ``f(x)`` (memoryless PDM) as an output function ``h_map(q, x)``.
    """
    fitness_compiled = fitness if _is_compiled(fitness) else numba.njit(fitness)

    @numba.njit(PDM_FUNCTION_SIGNATURE)
    def h_map(q, x):
        return fitness_compiled(x)

    return h_map


def _compile_pdm_function(fn: Callable) -> Callable:
    """
    Compile a function ``fn(q, x)`` (dynamic PDM) with the signature expected by the event loop.
    """
    if not _is_compiled(fn):
        return numba.njit(PDM_FUNCTION_SIGNATURE)(fn)

    @numba.njit(PDM_FUNCTION_SIGNATURE)
    def wrapped(q, x):
        return fn(q, x)

    return wrapped


def compile_user_function(fn: Callable, memoryless: bool) -> Callable:
    """
    Compile a user-provided function of the payoff mechanism for the event loop.

    Plain Python functions are compiled with ``numba.njit``. Functions already compiled by the user (e.g., decorated
    with ``@numba.njit``) are used as they are. In both cases, the result is a compiled function ``f(q, x)`` with the
    signature ``float64[:, ::1](float64[:, ::1], float64[:, ::1])``. Compiled functions are cached per input
    function.

    Args:
        fn (Callable): The function to compile: a fitness function ``f(x)`` if ``memoryless``, otherwise
            ``h_map(q, x)`` or ``w_map(q, x)``.
        memoryless (bool): Whether ``fn`` is the fitness function of a memoryless PDM (d=0).

    Returns:
        Callable: The compiled function.

    Raises:
        Exception: Any error raised by numba while compiling the function.
    """
    if memoryless:
        return _cached(_compiled_fitness_functions, fn, lambda: _compile_fitness(fn))
    return _cached(_compiled_pdm_functions, fn, lambda: _compile_pdm_function(fn))


@numba.njit(PDM_FUNCTION_SIGNATURE, cache=True)
def _memoryless_w_map(q, x):
    return np.zeros((0, 1))


def compile_payoff_mechanism(
    payoff_mechanism: PayoffMechanism,
) -> tuple[Callable, Callable]:
    """
    Compile the output (h_map) and dynamics (w_map) functions of a payoff mechanism.

    In the memoryless case (d=0), the fitness function ``h_map(x)`` is wrapped as ``h_map(q, x)`` and a placeholder
    ``w_map`` is provided.

    Args:
        payoff_mechanism (PayoffMechanism): The payoff mechanism.

    Returns:
        tuple[Callable, Callable]: The compiled ``h_map(q, x)`` and ``w_map(q, x)`` functions.

    Raises:
        Exception: Any error raised by numba while compiling the user-provided functions.
    """
    if payoff_mechanism.d == 0:
        h_map = compile_user_function(payoff_mechanism._h_map_user, memoryless=True)
        return h_map, _memoryless_w_map

    h_map = compile_user_function(payoff_mechanism._h_map_user, memoryless=False)
    w_map = compile_user_function(payoff_mechanism._w_map_user, memoryless=False)
    return h_map, w_map


@numba.njit(cache=True)
def _protocol_column(kind, params, p, x, i):
    """
    Switching probabilities from strategy i to every strategy of a population (column i of the protocol matrix).
    """
    n = p.shape[0]
    col = np.empty(n)
    if kind == _SOFTMAX:
        z_max = np.max(p) / params[0]
        total = 0.0
        for j in range(n):
            col[j] = np.exp(p[j] / params[0] - z_max)
            total += col[j]
        col /= total
    elif kind == _SMITH:
        for j in range(n):
            col[j] = max(p[j] - p[i], 0.0) * params[0]
    elif kind == _BNN:
        p_hat = 0.0
        for j in range(n):
            p_hat += x[j] * p[j]
        p_hat /= np.sum(x)
        for j in range(n):
            col[j] = max(p[j] - p_hat, 0.0) * params[0]
    elif kind == _REPLICATOR:
        mass = np.sum(x)
        for j in range(n):
            col[j] = x[j] / mass * max(p[j] - p[i], 0.0) * params[0]
    else:  # _CCSMITH
        for j in range(n):
            col[j] = max(params[1 + j] - x[j], 0.0) * max(p[j] - p[i], 0.0) * params[0]
    return col


@numba.njit(cache=True)
def _sample_index(weights, total, u):
    """
    Sample an index with probability proportional to its weight, given a uniform sample u in [0, 1).
    """
    threshold = u * total
    acc = 0.0
    last_positive = -1
    for j in range(weights.shape[0]):
        if weights[j] > 0:
            acc += weights[j]
            last_positive = j
            if threshold < acc:
                return j
    return last_positive  # threshold beyond the total due to round-off


@numba.njit(cache=True)
def _advance_pdm(h_map, w_map, q, x, p, time_step, max_step):
    """
    Integrate the PDM over a time step with fixed-step RK4 (x held constant). Updates q and p in place.
    """
    num_steps = max(1, int(np.ceil(time_step / max_step)))
    dt = time_step / num_steps
    for _ in range(num_steps):
        k1 = w_map(q, x)
        k2 = w_map(q + 0.5 * dt * k1, x)
        k3 = w_map(q + 0.5 * dt * k2, x)
        k4 = w_map(q + dt * k3, x)
        q[:, 0] += (dt / 6) * (k1[:, 0] + 2 * k2[:, 0] + 2 * k3[:, 0] + k4[:, 0])
    p[:, 0] = h_map(q, x)[:, 0]


_INT_1D, _FLOAT_1D = numba.types.int64[::1], numba.types.float64[::1]
_FLOAT = numba.types.float64


@numba.njit(
    numba.types.Tuple(
        (_FLOAT, numba.types.int64, numba.types.boolean, _FLOAT, numba.types.int64)
    )(
        _PDM_FUNCTION,  # h_map
        _PDM_FUNCTION,  # w_map
        numba.typeof(np.random.default_rng(0)),  # rng
        _FLOAT,  # t
        _FLOAT,  # t_end
        _INT_1D,  # counts
        _ARRAY_2D,  # x
        _ARRAY_2D,  # q
        _ARRAY_2D,  # p
        _INT_1D,  # starts
        _FLOAT_1D,  # num_agents
        _FLOAT_1D,  # masses
        _FLOAT_1D,  # clock_rates
        _INT_1D,  # kinds
        _ARRAY_2D,  # params
        _FLOAT,  # pdm_max_step
        _FLOAT,  # log_interval
        _FLOAT,  # last_log_t
        _FLOAT_1D,  # log_t
        _ARRAY_2D,  # log_x
        _ARRAY_2D,  # log_q
        _ARRAY_2D,  # log_p
    ),
    cache=True,
)
def run_events(
    h_map,
    w_map,
    rng,
    t,
    t_end,
    counts,
    x,
    q,
    p,
    starts,
    num_agents,
    masses,
    clock_rates,
    kinds,
    params,
    pdm_max_step,
    log_interval,
    last_log_t,
    log_t,
    log_x,
    log_q,
    log_p,
):
    """
    Compiled event loop of the fast simulation path.

    Simulates revision events until ``t_end`` or until the log buffers are full, updating ``counts``, ``x``, ``q``,
    and ``p`` in place. Each event is logged unless it occurs less than ``log_interval`` after the last log entry
    (``log_interval <= 0`` logs every event).

    Returns:
        tuple: (current time, number of log entries written, whether t_end was reached, time of the last log
        entry, number of events with invalid switching probabilities).
    """
    num_populations = starts.shape[0] - 1
    d = q.shape[0]
    rates = num_agents * clock_rates
    total_rate = np.sum(rates)
    capacity = log_t.shape[0]
    num_logged = 0
    num_invalid = 0

    while num_logged < capacity:
        time_step = rng.exponential(1 / total_rate)
        if t + time_step >= t_end:  # No more revisions before t_end
            if d > 0:
                _advance_pdm(h_map, w_map, q, x, p, t_end - t, pdm_max_step)
            return t_end, num_logged, True, last_log_t, num_invalid

        if d > 0:
            _advance_pdm(h_map, w_map, q, x, p, time_step, pdm_max_step)
        t += time_step

        # Sample the revising population k and the current strategy i of the revising agent
        k = 0
        if num_populations > 1:
            k = _sample_index(rates, total_rate, rng.random())
        a, b = starts[k], starts[k + 1]
        i = _sample_index(counts[a:b], num_agents[k], rng.random())

        # Sample the next strategy j of the revising agent
        col = _protocol_column(kinds[k], params[k], p[a:b, 0], x[a:b, 0], i)
        col[i] = 0.0
        col[i] = 1.0 - np.sum(col)
        if col[i] < 0:
            num_invalid += 1
            col = np.minimum(np.maximum(col, 0.0), 1.0)
        j = _sample_index(col, np.sum(col), rng.random())

        if j != i:
            counts[a + i] -= 1
            counts[a + j] += 1
            scale = masses[k] / num_agents[k]
            x[a + i, 0] = counts[a + i] * scale
            x[a + j, 0] = counts[a + j] * scale
            p[:, 0] = h_map(q, x)[:, 0]

        if log_interval <= 0 or t - last_log_t >= log_interval:
            log_t[num_logged] = t
            log_x[:, num_logged] = x[:, 0]
            log_q[:, num_logged] = q[:, 0]
            log_p[:, num_logged] = p[:, 0]
            last_log_t = t
            num_logged += 1

    return t, num_logged, False, last_log_t, num_invalid
