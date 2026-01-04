from __future__ import annotations

import unittest

import numpy as np

from popgames.alarm_clock import AlarmClockABC, Poisson
from popgames.revision_process import PoissonRevisionProcess, RevisionProcessABC
from popgames.revision_protocol import RevisionProtocolABC, Softmax


class _DummyRevisionProcess(RevisionProcessABC):
    """Concrete implementation for testing the ABC contract."""

    def sample_next_revision_time(self, size: int) -> np.ndarray:
        return np.zeros((size,))

    def sample_next_strategy(self, p: np.ndarray, x: np.ndarray, i: int) -> int:
        return int(i)

    def rhs_edm(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)


class _FixedAlarmClock(AlarmClockABC):
    """Alarm clock returning fixed nonnegative revision times."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = float(value)

    def __call__(self, size: int):
        return np.full((size,), self.value, dtype=float)


class _FixedRevisionProtocol(RevisionProtocolABC):
    """Revision protocol returning a fixed (n,n) matrix."""

    def __init__(self, mat: np.ndarray) -> None:
        self.mat = mat

    def __call__(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        return self.mat


class _BadRevisionProtocol(RevisionProtocolABC):
    """
    Protocol that can produce columns summing > 1 so that PoissonRevisionProcess must clip
    and logs a warning.
    """

    def __call__(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        n = p.shape[0]
        # Column i will have two off-diagonal entries 0.8 each -> sum(offdiag)=1.6 -> negative self prob.
        mat = np.zeros((n, n), dtype=float)
        if n >= 3:
            # For each column i, set two other rows to 0.8 (if possible)
            for i in range(n):
                rows = [r for r in range(n) if r != i][:2]
                for r in rows:
                    mat[r, i] = 0.8
        else:
            # n=2: one off-diagonal 1.2 -> still invalid
            mat[1, 0] = 1.2
            mat[0, 1] = 1.2
        return mat


class TestRevisionProcessABC(unittest.TestCase):
    def test_concrete_subclass_can_be_instantiated(self) -> None:
        rp = _DummyRevisionProcess(
            alarm_clock=_FixedAlarmClock(),
            revision_protocol=_FixedRevisionProtocol(np.zeros((2, 2))),
        )
        self.assertIsInstance(rp, RevisionProcessABC)

    def test_init_validates_types(self) -> None:
        # alarm_clock wrong type
        with self.assertRaises(TypeError):
            _DummyRevisionProcess(
                alarm_clock=123,  # type: ignore[arg-type]
                revision_protocol=_FixedRevisionProtocol(np.zeros((2, 2))),
            )

        # revision_protocol wrong type
        with self.assertRaises(TypeError):
            _DummyRevisionProcess(alarm_clock=_FixedAlarmClock(), revision_protocol=123)  # type: ignore[arg-type]


class TestPoissonRevisionProcess(unittest.TestCase):
    def test_init_sets_clock_rate_and_components(self) -> None:
        rp = PoissonRevisionProcess(
            Poisson_clock_rate=0.25, revision_protocol=Softmax(eta=0.1)
        )
        self.assertEqual(rp.Poisson_clock_rate, 0.25)
        self.assertIsInstance(rp.alarm_clock, Poisson)
        self.assertIsInstance(rp.revision_protocol, RevisionProtocolABC)

    def test_sample_next_revision_time_shape_and_nonnegative(self) -> None:
        np.random.seed(0)
        rp = PoissonRevisionProcess(
            Poisson_clock_rate=1.0, revision_protocol=Softmax(eta=0.1)
        )
        out = rp.sample_next_revision_time(10)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (10,))
        self.assertTrue(np.all(out >= 0.0))

    def test_sample_next_strategy_returns_valid_index(self) -> None:
        np.random.seed(1)
        rp = PoissonRevisionProcess(
            Poisson_clock_rate=1.0, revision_protocol=Softmax(eta=0.2)
        )

        p = np.array([1.0, -1.0, 2.0]).reshape(3, 1)
        x = np.array([0.2, 0.5, 0.3]).reshape(3, 1)

        i = 1
        j = rp.sample_next_strategy(p, x, i)
        self.assertIsInstance(j, (int, np.integer))
        self.assertTrue(0 <= int(j) < 3)

    def test_sample_next_strategy_probability_fixup_logs_warning_and_still_returns_index(
        self,
    ) -> None:
        np.random.seed(2)
        rp = PoissonRevisionProcess(
            Poisson_clock_rate=1.0, revision_protocol=_BadRevisionProtocol()
        )

        p = np.array([1.0, 0.0, -1.0]).reshape(3, 1)
        x = np.array([0.3, 0.4, 0.3]).reshape(3, 1)

        with self.assertLogs("popgames.revision_process", level="WARNING") as cm:
            j = rp.sample_next_strategy(p, x, i=0)

        self.assertTrue(any("Invalid probabilities" in msg for msg in cm.output))
        self.assertTrue(0 <= int(j) < 3)

    def test_rhs_edm_shape(self) -> None:
        rp = PoissonRevisionProcess(
            Poisson_clock_rate=0.5, revision_protocol=Softmax(eta=0.3)
        )
        p = np.array([1.0, -1.0, 2.0]).reshape(3, 1)
        x = np.array([0.2, 0.6, 0.2]).reshape(3, 1)

        out = rp.rhs_edm(x, p)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (3, 1))

    def test_rhs_edm_is_zero_when_all_payoffs_equal_and_x_uniform(self) -> None:
        rp = PoissonRevisionProcess(
            Poisson_clock_rate=0.5, revision_protocol=Softmax(eta=0.3)
        )
        p = np.array([7.0, 7.0, 7.0]).reshape(3, 1)
        x = np.array([1.0, 1.0, 1.0]).reshape(
            3, 1
        )  # uniform (not necessarily normalized)

        out = rp.rhs_edm(x, p)
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    def test_rhs_edm_closed_form_when_all_payoffs_equal(self) -> None:
        lam = 0.5
        n = 3
        rp = PoissonRevisionProcess(
            Poisson_clock_rate=lam, revision_protocol=Softmax(eta=0.3)
        )

        p = np.array([7.0, 7.0, 7.0]).reshape(n, 1)  # all equal
        x = np.array([0.2, 0.6, 0.2]).reshape(n, 1)

        out = rp.rhs_edm(x, p)

        # With uniform revision matrix (1/n), the RHS simplifies to:
        # lam * (mean(x) - x), where mean(x) = sum(x)/n
        mean_x = float(x.sum()) / n
        expected = lam * (mean_x - x)

        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_rhs_edm_scales_linearly_with_clock_rate(self) -> None:
        p = np.array([1.0, -1.0, 2.0]).reshape(3, 1)
        x = np.array([0.2, 0.6, 0.2]).reshape(3, 1)
        proto = Softmax(eta=0.3)

        rp1 = PoissonRevisionProcess(Poisson_clock_rate=0.5, revision_protocol=proto)
        rp2 = PoissonRevisionProcess(Poisson_clock_rate=1.0, revision_protocol=proto)

        out1 = rp1.rhs_edm(x, p)
        out2 = rp2.rhs_edm(x, p)

        np.testing.assert_allclose(out2, 2.0 * out1, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
