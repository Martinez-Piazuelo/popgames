from __future__ import annotations

import unittest

import numpy as np

from popgames.payoff_mechanism import PayoffMechanism


class TestPayoffMechanismInit(unittest.TestCase):
    def test_rejects_invalid_n_and_d_types_and_bounds(self) -> None:
        def h(x):
            return x

        with self.assertRaises(TypeError):
            PayoffMechanism(h_map=h, n=1.0)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            PayoffMechanism(h_map=h, n=0)

        with self.assertRaises(TypeError):
            PayoffMechanism(h_map=h, n=2, d=1.0)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            PayoffMechanism(h_map=h, n=2, d=-1)

    def test_memoryless_case_unsqueezes_h_map_and_sets_dummy_w_map(self) -> None:
        n = 3

        def h_map(x: np.ndarray) -> np.ndarray:
            return 2.0 * x

        pm = PayoffMechanism(h_map=h_map, n=n, d=0)

        self.assertEqual(pm.n, n)
        self.assertEqual(pm.d, 0)

        # h_map stored should accept (q, x) signature and ignore q
        q0 = np.zeros((0, 1))
        x0 = np.ones((n, 1))
        out = pm.h_map(q0, x0)
        np.testing.assert_allclose(out, 2.0 * x0)

        # dummy w_map returns (0,1)
        w_out = pm.w_map(q0, x0)
        self.assertIsInstance(w_out, np.ndarray)
        self.assertEqual(w_out.shape, (0, 1))

    def test_dynamic_case_accepts_h_and_w_maps_with_correct_signatures(self) -> None:
        n, d = 2, 1

        def h_map(q: np.ndarray, x: np.ndarray) -> np.ndarray:
            # output (n,1)
            return np.vstack([q[0, 0] + x[0, 0], q[0, 0] + x[1, 0]]).reshape(n, 1)

        def w_map(q: np.ndarray, x: np.ndarray) -> np.ndarray:
            # output (d,1)
            return -q

        pm = PayoffMechanism(h_map=h_map, n=n, w_map=w_map, d=d)
        self.assertEqual(pm.n, n)
        self.assertEqual(pm.d, d)

        q0 = np.array([[1.0]])
        x0 = np.array([[2.0], [3.0]])
        np.testing.assert_allclose(pm.h_map(q0, x0), np.array([[3.0], [4.0]]))
        np.testing.assert_allclose(pm.w_map(q0, x0), np.array([[-1.0]]))

    def test_dynamic_case_rejects_missing_w_map(self) -> None:
        n, d = 2, 1

        def h_map(q: np.ndarray, x: np.ndarray) -> np.ndarray:
            return np.zeros((n, 1))

        # w_map is None -> check_function_signature should fail (not callable)
        with self.assertRaises((TypeError, ValueError)):  # type: ignore[misc]
            PayoffMechanism(h_map=h_map, n=n, w_map=None, d=d)  # type: ignore[arg-type]

    def test_memoryless_case_rejects_wrong_h_map_signature(self) -> None:
        n = 2

        def h_bad(q: np.ndarray, x: np.ndarray) -> np.ndarray:
            return x  # wrong signature for d=0 case (expects only x)

        with self.assertRaises(ValueError):
            PayoffMechanism(h_map=h_bad, n=n, d=0)  # type: ignore[arg-type]

    def test_dynamic_case_rejects_wrong_h_map_output_shape(self) -> None:
        n, d = 2, 1

        def h_bad(q: np.ndarray, x: np.ndarray) -> np.ndarray:
            return np.zeros((n,))  # wrong shape

        def w_map(q: np.ndarray, x: np.ndarray) -> np.ndarray:
            return np.zeros((d, 1))

        with self.assertRaises(ValueError):
            PayoffMechanism(h_map=h_bad, n=n, w_map=w_map, d=d)

    def test_dynamic_case_rejects_wrong_w_map_output_shape(self) -> None:
        n, d = 2, 1

        def h_map(q: np.ndarray, x: np.ndarray) -> np.ndarray:
            return np.zeros((n, 1))

        def w_bad(q: np.ndarray, x: np.ndarray) -> np.ndarray:
            return np.zeros((d + 1, 1))  # wrong shape

        with self.assertRaises(ValueError):
            PayoffMechanism(h_map=h_map, n=n, w_map=w_bad, d=d)


class TestPayoffMechanismIntegrateMemoryless(unittest.TestCase):
    def test_integrate_memoryless_returns_expected(self) -> None:
        n = 3

        def h_map(x: np.ndarray) -> np.ndarray:
            return 3.0 * x

        pm = PayoffMechanism(h_map=h_map, n=n, d=0)

        q0 = np.zeros((0, 1))
        x0 = np.array([[1.0], [2.0], [3.0]])

        res = pm.integrate(q0=q0, x0=x0, t_span=(0.0, 1.0), output_trajectory=True)

        # In memoryless case, returned q is empty placeholder
        self.assertTrue(hasattr(res, "q"))
        self.assertTrue(hasattr(res, "p"))
        self.assertEqual(res.q.shape, (0, 1))
        # output_trajectory=True -> p duplicated across two time points
        self.assertEqual(res.p.shape, (n, 2))
        np.testing.assert_allclose(res.p[:, [0]], 3.0 * x0)
        np.testing.assert_allclose(res.p[:, [1]], 3.0 * x0)

    def test_integrate_memoryless_output_trajectory_false(self) -> None:
        n = 2

        def h_map(x: np.ndarray) -> np.ndarray:
            return x + 1.0

        pm = PayoffMechanism(h_map=h_map, n=n, d=0)

        q0 = np.zeros((0, 1))
        x0 = np.array([[5.0], [6.0]])

        res = pm.integrate(q0=q0, x0=x0, t_span=(0.0, 2.0), output_trajectory=False)
        self.assertEqual(res.q.shape, (0, 1))
        self.assertEqual(res.p.shape, (n, 1))
        np.testing.assert_allclose(res.p, x0 + 1.0)


class TestPayoffMechanismDynamicHelpers(unittest.TestCase):
    def test_h_map_wrapped_and_w_map_wrapped_shapes(self) -> None:
        n, d = 2, 1

        def h_map(q: np.ndarray, x: np.ndarray) -> np.ndarray:
            return np.array([[q[0, 0] + x[0, 0]], [q[0, 0] + x[1, 0]]])

        def w_map(q: np.ndarray, x: np.ndarray) -> np.ndarray:
            return -q

        pm = PayoffMechanism(h_map=h_map, n=n, w_map=w_map, d=d)

        y = np.array([10.0, 1.0, 2.0])  # (d+n,) flattened
        h_out = pm._h_map_wrapped(y)
        self.assertIsInstance(h_out, np.ndarray)
        self.assertEqual(h_out.shape, (n,))
        np.testing.assert_allclose(h_out, np.array([11.0, 12.0]))

        w_out = pm._w_map_wrapped(t=0.0, y=y)
        self.assertIsInstance(w_out, np.ndarray)
        self.assertEqual(w_out.shape, (d + n,))
        # w_map -> -q = -10 ; x dynamics are zeros
        np.testing.assert_allclose(w_out, np.array([-10.0, 0.0, 0.0]))


class TestPayoffMechanismIntegrateDynamic(unittest.TestCase):
    def test_integrate_dynamic_output_trajectory_true_shapes(self) -> None:
        # simple stable linear ODE: q' = -q, p = x + q
        n, d = 2, 1

        def h_map(q: np.ndarray, x: np.ndarray) -> np.ndarray:
            return x + np.vstack([q[0, 0], q[0, 0]]).reshape(n, 1)

        def w_map(q: np.ndarray, x: np.ndarray) -> np.ndarray:
            return -q

        pm = PayoffMechanism(h_map=h_map, n=n, w_map=w_map, d=d)

        q0 = np.array([[1.0]])
        x0 = np.array([[2.0], [3.0]])
        res = pm.integrate(
            q0=q0, x0=x0, t_span=(0.0, 0.5), method="RK45", output_trajectory=True
        )

        self.assertTrue(hasattr(res, "t"))
        self.assertTrue(hasattr(res, "q"))
        self.assertTrue(hasattr(res, "p"))

        self.assertEqual(res.q.shape[0], d)
        self.assertEqual(res.p.shape[0], n)
        # trajectory: number of time points should match
        self.assertEqual(res.q.shape[1], res.p.shape[1])
        self.assertEqual(res.q.shape[1], len(res.t))

        # final p should be close to x + q_final (q decays from 1)
        q_final = res.q[:, [-1]]
        p_final = res.p[:, [-1]]
        expected_final = x0 + np.vstack([q_final[0, 0], q_final[0, 0]]).reshape(n, 1)
        np.testing.assert_allclose(p_final, expected_final, atol=1e-6)

    def test_integrate_dynamic_output_trajectory_false_returns_final_state(
        self,
    ) -> None:
        n, d = 2, 1

        def h_map(q: np.ndarray, x: np.ndarray) -> np.ndarray:
            return x + np.vstack([q[0, 0], q[0, 0]]).reshape(n, 1)

        def w_map(q: np.ndarray, x: np.ndarray) -> np.ndarray:
            return -q

        pm = PayoffMechanism(h_map=h_map, n=n, w_map=w_map, d=d)

        q0 = np.array([[1.0]])
        x0 = np.array([[2.0], [3.0]])
        res = pm.integrate(
            q0=q0, x0=x0, t_span=(0.0, 0.25), method="RK45", output_trajectory=False
        )

        # q is the final state column vector; p computed at final state
        self.assertEqual(res.q.shape, (d, 1))
        self.assertEqual(res.p.shape, (n, 1))
        np.testing.assert_allclose(res.p, pm.h_map(res.q, x0), atol=1e-9)


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
