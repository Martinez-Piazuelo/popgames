from __future__ import annotations

import logging
import unittest

import numpy as np

from popgames.population_game import SinglePopulationGame
from popgames.utilities.gnep_solvers import fbos


def _make_constrained_game(lipschitz_constant: float) -> SinglePopulationGame:
    """Single population, n=3, mass 1, with the isotropic-linear fitness f(x) = u - x.

    The tight Lipschitz constant of f is exactly 1. The coupling constraint
    a^T x <= C is active at the true GNE.
    """
    u = np.array([[1.0], [1.2], [1.4]])
    a = np.array([[0.3, 0.6, 1.0]])
    C = 0.65
    return SinglePopulationGame(
        num_strategies=3,
        fitness_function=lambda x: u - x,
        mass=1,
        A_ineq=a,
        b_ineq=C,
        fitness_lipschitz_constant=lipschitz_constant,
    )


class TestFbosTightLipschitz(unittest.TestCase):
    """Regression tests for the stepsize = 1/L boundary bug (see POPGAMES_BUG.md)."""

    A = np.array([[0.3, 0.6, 1.0]])
    EXPECTED_GNE = np.array([0.30, 0.35, 0.35])

    def test_tight_lipschitz_constant_constrained(self) -> None:
        # Passing the tight Lipschitz constant L = 1 used to make the iteration
        # stall on the first step and return the uniform initial point.
        g = _make_constrained_game(lipschitz_constant=1.0)
        gne = g.compute_gne().ravel()

        np.testing.assert_allclose(gne, self.EXPECTED_GNE, atol=1e-3)
        # The coupling constraint must be active at the true GNE.
        self.assertAlmostEqual(
            float((self.A @ gne.reshape(-1, 1)).item()), 0.65, places=3
        )

    def test_does_not_return_uniform_initial_point(self) -> None:
        # The bogus "converged" value was the uniform initial point [1/3, 1/3, 1/3].
        g = _make_constrained_game(lipschitz_constant=1.0)
        gne = g.compute_gne().ravel()

        uniform = np.full(3, 1.0 / 3.0)
        self.assertGreater(np.max(np.abs(gne - uniform)), 1e-2)

    def test_consistent_across_lipschitz_constants(self) -> None:
        # The tight constant must yield the same GNE as comfortably larger ones.
        results = [
            _make_constrained_game(lipschitz_constant=L).compute_gne().ravel()
            for L in (1.0, 1.01, 1.5, 2.0)
        ]
        for gne in results:
            np.testing.assert_allclose(gne, results[-1], atol=1e-3)

    def test_isotropic_linear_unconditional_stall_case(self) -> None:
        # f(x) = u - x with no extra constraints: the solver should still move off
        # the uniform initial point toward proj_delta(u), not stall at iteration 0.
        u = np.array([[1.0], [1.2], [1.4]])
        g = SinglePopulationGame(
            num_strategies=3,
            fitness_function=lambda x: u - x,
            mass=1,
            fitness_lipschitz_constant=1.0,
        )
        gne = g.compute_gne().ravel()

        uniform = np.full(3, 1.0 / 3.0)
        self.assertGreater(np.max(np.abs(gne - uniform)), 1e-2)
        # Simplex feasibility.
        np.testing.assert_allclose(gne.sum(), 1.0, atol=1e-6)
        self.assertTrue(np.all(gne >= -1e-9))


class TestFbosGammaArgument(unittest.TestCase):
    def test_gamma_must_be_in_open_unit_interval(self) -> None:
        g = _make_constrained_game(lipschitz_constant=1.0)
        for bad_gamma in (0.0, 1.0, -0.5, 1.5):
            with self.assertRaises(ValueError):
                fbos(population_game=g, gamma=bad_gamma)

    def test_compute_gne_forwards_gamma(self) -> None:
        # A valid in-range gamma still produces the correct GNE.
        g = _make_constrained_game(lipschitz_constant=1.0)
        gne = g.compute_gne(gamma=0.5).ravel()
        np.testing.assert_allclose(gne, np.array([0.30, 0.35, 0.35]), atol=1e-3)


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    unittest.main()  # pragma: no cover
