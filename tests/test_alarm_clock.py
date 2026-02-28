from __future__ import annotations

import unittest

import numpy as np

from popgames.alarm_clock import AlarmClockABC, Poisson


class _DummyClock(AlarmClockABC):
    """Concrete implementation for testing the ABC contract."""

    def __call__(self, size: int):
        return 0.0 if size == 1 else np.zeros(size)


class TestAlarmClockABC(unittest.TestCase):
    def test_can_instantiate_concrete_subclass(self) -> None:
        clock = _DummyClock()
        self.assertIsInstance(clock, AlarmClockABC)

    def test_concrete_subclass_call_contract(self) -> None:
        clock = _DummyClock()
        out1 = clock(1)
        out3 = clock(3)
        self.assertIsInstance(out1, float)
        self.assertIsInstance(out3, np.ndarray)
        self.assertEqual(out3.shape, (3,))


class TestPoissonAlarmClock(unittest.TestCase):
    def test_init_sets_rate(self) -> None:
        clock = Poisson(rate=2.5)
        self.assertEqual(clock.rate, 2.5)

    def test_init_rejects_non_positive_rate(self) -> None:
        with self.assertRaises(ValueError):
            Poisson(rate=0.0)
        with self.assertRaises(ValueError):
            Poisson(rate=-1.0)

    def test_call_returns_numpy_array_of_correct_shape(self) -> None:
        np.random.seed(12345)
        clock = Poisson(rate=1.0)

        out = clock(5)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (5,))

    def test_call_samples_are_nonnegative(self) -> None:
        np.random.seed(0)
        clock = Poisson(rate=1.0)

        out = clock(1000)
        # Exponential samples are >= 0 (might allow tiny negative due to float issues, but should not happen)
        self.assertTrue(np.all(out >= 0.0))

    def test_call_is_reproducible_with_seed(self) -> None:
        np.random.seed(42)
        clock1 = Poisson(rate=1.0)
        s1 = clock1(4)

        np.random.seed(42)
        clock2 = Poisson(rate=1.0)
        s2 = clock2(4)

        np.testing.assert_allclose(s1, s2)

    def test_higher_rate_produces_smaller_mean_samples(self) -> None:
        # Deterministic comparison by resetting the seed for each clock
        n = 50_000

        np.random.seed(7)
        slow = Poisson(rate=1.0)
        slow_mean = float(np.mean(slow(n)))

        np.random.seed(7)
        fast = Poisson(rate=4.0)
        fast_mean = float(np.mean(fast(n)))

        # With the same underlying uniforms, increasing rate decreases the exponential scale
        self.assertLess(fast_mean, slow_mean)

    def test_call_accepts_size_one(self) -> None:
        np.random.seed(123)
        clock = Poisson(rate=1.0)
        out = clock(1)
        # numpy.random.exponential returns an array for size=1
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (1,))


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
