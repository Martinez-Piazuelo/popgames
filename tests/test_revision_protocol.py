from __future__ import annotations

import unittest

import numpy as np

from popgames.revision_protocol import (
    BNN,
    CCSmith,
    RevisionProtocolABC,
    Smith,
    Softmax,
)


class _DummyProtocol(RevisionProtocolABC):
    """Concrete implementation for testing the ABC contract."""

    def __call__(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        n = p.shape[0]
        return np.zeros((n, n))


class TestRevisionProtocolABC(unittest.TestCase):
    def test_can_instantiate_concrete_subclass(self) -> None:
        proto = _DummyProtocol()
        self.assertIsInstance(proto, RevisionProtocolABC)

    def test_concrete_subclass_call_returns_square_matrix(self) -> None:
        proto = _DummyProtocol()
        p = np.array([1.0, 2.0, 3.0]).reshape(3, 1)
        x = np.array([0.2, 0.3, 0.5]).reshape(3, 1)
        out = proto(p, x)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (3, 3))


class TestSoftmax(unittest.TestCase):
    def test_init_sets_eta(self) -> None:
        s = Softmax(eta=0.5)
        self.assertEqual(s.eta, 0.5)

    def test_init_rejects_non_positive_eta(self) -> None:
        with self.assertRaises(ValueError):
            Softmax(eta=0.0)
        with self.assertRaises(ValueError):
            Softmax(eta=-1.0)

    def test_call_returns_matrix_of_correct_shape(self) -> None:
        s = Softmax(eta=1.0)
        p = np.array([1.0, -1.0, 2.0]).reshape(3, 1)
        x = np.array([0.1, 0.7, 0.2]).reshape(3, 1)
        out = s(p, x)
        self.assertEqual(out.shape, (3, 3))

    def test_softmax_columns_are_identical(self) -> None:
        s = Softmax(eta=1.0)
        p = np.array([1.0, -1.0, 2.0]).reshape(3, 1)
        x = np.array([0.1, 0.7, 0.2]).reshape(3, 1)
        out = s(p, x)
        # Construction is probabilities @ ones^T, so all columns identical.
        for j in range(1, out.shape[1]):
            np.testing.assert_allclose(out[:, [0]], out[:, [j]])

    def test_softmax_probability_vector_matches_expected(self) -> None:
        s = Softmax(eta=1.0)
        p = np.array([1.0, -1.0, 2.0]).reshape(3, 1)
        x = np.array([0.1, 0.7, 0.2]).reshape(3, 1)

        out = s(p, x)

        # Expected probabilities from definition (stable version)
        logits = np.exp((p / s.eta) - np.max(p / s.eta))
        probs = logits / logits.sum()
        expected = probs @ np.ones_like(probs).T

        np.testing.assert_allclose(out, expected)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=12)
        self.assertTrue(np.all(probs >= 0.0))

    def test_softmax_invariant_to_constant_shift_in_payoffs(self) -> None:
        s = Softmax(eta=0.7)
        p = np.array([1.0, -1.0, 2.0]).reshape(3, 1)
        x = np.array([0.1, 0.7, 0.2]).reshape(3, 1)

        out1 = s(p, x)
        out2 = s(p + 123.456, x)  # adding a constant should not change softmax
        np.testing.assert_allclose(out1, out2)


class TestSmith(unittest.TestCase):
    def test_init_sets_scale(self) -> None:
        sm = Smith(scale=0.2)
        self.assertEqual(sm.scale, 0.2)

    def test_init_rejects_non_positive_scale(self) -> None:
        with self.assertRaises(ValueError):
            Smith(scale=0.0)
        with self.assertRaises(ValueError):
            Smith(scale=-1.0)

    def test_call_matches_reference_formula(self) -> None:
        sm = Smith(scale=0.1)
        p = np.array([1.0, -1.0, 2.0]).reshape(3, 1)
        x = np.array([0.1, 0.7, 0.2]).reshape(3, 1)

        out = sm(p, x)

        expected = np.maximum(p - p.T, 0.0) * sm.scale
        np.testing.assert_allclose(out, expected)
        self.assertEqual(out.shape, (3, 3))
        self.assertTrue(np.all(out >= 0.0))

    def test_diagonal_is_zero(self) -> None:
        sm = Smith(scale=0.5)
        p = np.array([3.0, 2.0, 1.0]).reshape(3, 1)
        out = sm(p, np.ones((3, 1)))
        np.testing.assert_allclose(np.diag(out), 0.0)


class TestBNN(unittest.TestCase):
    def test_init_sets_scale(self) -> None:
        bnn = BNN(scale=0.2)
        self.assertEqual(bnn.scale, 0.2)

    def test_init_rejects_non_positive_scale(self) -> None:
        with self.assertRaises(ValueError):
            BNN(scale=0.0)
        with self.assertRaises(ValueError):
            BNN(scale=-1.0)

    def test_call_matches_reference_formula(self) -> None:
        bnn = BNN(scale=0.1)
        p = np.array([1.0, -1.0, 2.0]).reshape(3, 1)
        x = np.array([0.1, 0.7, 0.2]).reshape(3, 1)

        out = bnn(p, x)

        p_hat = (x.T @ p) / x.sum()
        delta_p = p - p_hat[0]
        expected = np.maximum(delta_p @ np.ones_like(delta_p).T, 0.0) * bnn.scale

        np.testing.assert_allclose(out, expected)
        self.assertEqual(out.shape, (3, 3))
        self.assertTrue(np.all(out >= 0.0))

    def test_uniform_payoffs_yield_zero_matrix(self) -> None:
        bnn = BNN(scale=0.3)
        p = np.array([5.0, 5.0, 5.0]).reshape(3, 1)
        x = np.array([0.2, 0.3, 0.5]).reshape(3, 1)
        out = bnn(p, x)
        np.testing.assert_allclose(out, 0.0)


class TestCCSmith(unittest.TestCase):
    def test_init_sets_scale_and_xbar(self) -> None:
        x_bar = np.array([1.0, 2.0, 3.0]).reshape(3, 1)
        cc = CCSmith(scale=0.5, x_bar=x_bar)
        self.assertEqual(cc.scale, 0.5)
        np.testing.assert_allclose(cc.x_bar, x_bar)

    def test_init_rejects_non_positive_scale(self) -> None:
        x_bar = np.array([1.0, 1.0]).reshape(2, 1)
        with self.assertRaises(ValueError):
            CCSmith(scale=0.0, x_bar=x_bar)
        with self.assertRaises(ValueError):
            CCSmith(scale=-1.0, x_bar=x_bar)

    def test_init_rejects_wrong_xbar_shape(self) -> None:
        # x_bar must have shape (n, 1)
        x_bar_bad = np.array([1.0, 2.0])  # shape (2,)
        with self.assertRaises(ValueError):
            CCSmith(scale=0.1, x_bar=x_bar_bad)  # type: ignore[arg-type]

    def test_call_matches_reference_formula(self) -> None:
        x_bar = np.array([1.0, 2.0, 3.0]).reshape(3, 1)
        cc = CCSmith(scale=0.2, x_bar=x_bar)

        p = np.array([1.0, -1.0, 2.0]).reshape(3, 1)
        x = np.array([0.5, 2.5, 1.0]).reshape(3, 1)

        out = cc(p, x)

        expected = np.maximum(x_bar - x, 0.0) * np.maximum(p - p.T, 0.0) * cc.scale
        np.testing.assert_allclose(out, expected)
        self.assertEqual(out.shape, (3, 3))
        self.assertTrue(np.all(out >= 0.0))

    def test_when_x_exceeds_xbar_rows_become_zero(self) -> None:
        x_bar = np.array([1.0, 1.0, 1.0]).reshape(3, 1)
        cc = CCSmith(scale=1.0, x_bar=x_bar)

        p = np.array([0.0, 1.0, 2.0]).reshape(3, 1)
        x = np.array([2.0, 0.5, 0.5]).reshape(3, 1)  # first exceeds x_bar

        out = cc(p, x)

        # For i=0, max(x_bar - x, 0) = 0 => entire row i should be zeros
        np.testing.assert_allclose(out[0, :], 0.0)


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
