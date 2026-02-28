from __future__ import annotations

import contextlib
import typing
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np

if typing.TYPE_CHECKING:
    from typing import Any, Callable, Sequence
    from unittest import TestCase

# -------------------------
# 1) Generic shape checking
# -------------------------


def _as_array(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def assert_array_shape(
    tc: TestCase,
    arr: Any,
    shape: tuple[int, ...],
    *,
    name: str = "array",
) -> None:
    tc.assertIsNotNone(arr, f"{name} is None")
    a = _as_array(arr)
    tc.assertEqual(
        a.shape, shape, f"Unexpected shape for {name}: got {a.shape}, expected {shape}"
    )


def assert_has_attrs(
    tc: TestCase,
    obj: Any,
    attrs: Sequence[str],
    *,
    name: str = "object",
) -> None:
    missing = [a for a in attrs if not hasattr(obj, a)]
    tc.assertFalse(missing, f"{name} is missing attributes: {missing}")


# ----------------------------------------------------
# 2) Simulation runner that returns "things to check"
# ----------------------------------------------------


@dataclass(frozen=True)
class SimRunResult:
    """What we return from running an example sim in tests."""

    sim: Any
    snapshots: dict[str, Any]  # arbitrary captured outputs


def run_sim_and_capture(
    *,
    sim: Any,
    T_sim: float,
    capture: dict[str, Callable[[Any], Any]] | None = None,
    flatten_log_method: str = "_get_flattened_log",
) -> SimRunResult:
    """
    Run sim for T_sim and capture outputs.

    Default behavior captures the simulator's flattened log by calling
    ``sim._get_flattened_log()`` (or another name via flatten_log_method).
    You can also pass extra capture callables via ``capture``.

    Returns:
        SimRunResult with:
          - snapshots["log"]: SimpleNamespace with fields t, x, q, p
            where each field is a stacked numpy array:
              t: (K,)
              x: (n, K)
              q: (d, K)
              p: (..., K)  (depends on your model)
          - plus any additional captures you request.
    """
    sim.run(T_sim=T_sim)

    snaps: dict[str, Any] = {}

    # Always capture flattened log
    if not hasattr(sim, flatten_log_method):
        raise AttributeError(
            f"Simulator missing '{flatten_log_method}'. "
            f"Available: {', '.join(sorted(dir(sim)))}"
        )
    log_fn = getattr(sim, flatten_log_method)
    log = log_fn()  # expected SimpleNamespace(t, x, q, p)
    snaps["log"] = log

    # Expose common fields directly for convenience
    snaps["t"] = getattr(log, "t", None)
    snaps["x"] = getattr(log, "x", None)
    snaps["q"] = getattr(log, "q", None)
    snaps["p"] = getattr(log, "p", None)

    # Any user-specified captures
    if capture:
        for key, fn in capture.items():
            snaps[key] = fn(sim)

    return SimRunResult(sim=sim, snapshots=snaps)


def assert_captures_shapes(
    tc: TestCase,
    result: SimRunResult,
    expected_shapes: dict[str, tuple[int, ...]],
) -> None:
    """
    Validate shapes for captured outputs. If capture stored an Exception, fail nicely.
    """
    for key, shape in expected_shapes.items():
        tc.assertIn(key, result.snapshots, f"Capture '{key}' not found in snapshots.")
        val = result.snapshots[key]
        if isinstance(val, Exception):
            tc.fail(f"Capture '{key}' raised exception: {val!r}")
        assert_array_shape(tc, val, shape, name=key)


# ---------------------------------------------------------
# 3) Plot suppression: run plot() without showing a figure
# ---------------------------------------------------------


@contextlib.contextmanager
def suppress_matplotlib_show_and_close():
    """
    Prevent plots from actually rendering.

    Use:
      with suppress_matplotlib_show_and_close():
          sim.plot(...)
    """
    # Many plotting pipelines use matplotlib.pyplot
    try:
        import matplotlib.pyplot as plt
    except Exception:
        # If matplotlib isn't installed in the test env, just yield (plot may fail anyway)
        yield
        return

    with (
        patch.object(plt, "show", autospec=True) as _p_show,
        patch.object(plt, "close", autospec=True) as _p_close,
    ):
        yield


def run_plot_smoketest(
    *,
    sim: Any,
    plot_kwargs: dict[str, Any],
) -> None:
    """
    Runs ``sim.plot(**plot_kwargs)`` but suppresses GUI rendering.
    This is meant as a *smoke test* only.
    """
    with suppress_matplotlib_show_and_close():
        sim.plot(**plot_kwargs)


# ---------------------------------------------------------
# 4) Patching helpers
# ---------------------------------------------------------


@contextlib.contextmanager
def patch_compute_gne_max_iter(population_game, max_iter: int):
    """
    Force ``population_game.compute_gne()`` to run with a smaller ``max_iter``.
    """
    original = population_game.compute_gne

    def wrapped_compute_gne(*args, **kwargs):
        kwargs["max_iter"] = max_iter
        return original(*args, **kwargs)

    with patch.object(
        population_game, "compute_gne", side_effect=wrapped_compute_gne, autospec=True
    ):
        yield
