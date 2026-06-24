"""Regression tests for the demand-side energy-management example of the paper

    J. Martinez-Piazuelo, C. Ocampo-Martinez, and N. Quijano,
    "Generalized Nash Equilibrium Seeking in Population Games: A Tutorial,"
    Annual Reviews in Control, 2026.

Section 6 of that paper presents a single-population game (n=3 strategies) with a
single coupling inequality constraint, solved with an EDM-PDM design built on
``popgames``. The paper reports the following ground truth:

  * generalized Nash equilibrium    x* = [0.30, 0.35, 0.35]
  * coupling constraint a^T x <= C   active at the GNE (a^T x* = C = 0.65)
  * grid dual multiplier             q_b* = 0.5

These tests pin those numbers so that the published example stays reproducible as
``popgames`` evolves. The fitness f(x) = u - x is isotropic-linear with tight
Lipschitz constant L = 1; passing that tight constant to ``compute_gne`` is what
exercises the FBOS step-size path, so this also guards
against a regression of that solver bug.

Both checks are deterministic (no RNG): ``compute_gne`` starts from the uniform
point and ``integrate_edm_pdm`` integrates the mean closed-loop dynamics with a
fixed ODE solver, so no random seed is required.
"""

from __future__ import annotations

import unittest

import numpy as np

import popgames as pg

# --- Section 6 problem data (mirrors Listing list:example_energy in paper.tex) ---
U = np.array([[1.0], [1.2], [1.4]])
A_INEQ = np.array([[0.3, 0.6, 1.0]])
C_CAP = 0.65
TAU_A, TAU_B, ALPHA_0, ALPHA_1 = 1.0, 1.0, 0.9, 0.1

# Paper ground truth.
X_STAR = np.array([0.30, 0.35, 0.35])
QB_STAR = 0.5


def _fitness(x):
    return U - x


def _w_map(q, x):  # PDM dynamics (4 states: 3 for Sigma_a, 1 for Sigma_b)
    g = (A_INEQ @ x).item() - C_CAP
    return np.vstack(
        [
            TAU_A * (_fitness(x) - q[:3]),
            [[TAU_B * (max(g, 0) - q[3].item() * max(-g, 0))]],
        ]
    )


def _h_map(q, x):  # PDM output (payoff signal)
    return ALPHA_0 * _fitness(x) + ALPHA_1 * q[:3] - A_INEQ.T * q[3].item()


def _build_game():
    # The coupling constraint must be passed to the game so that compute_gne
    # returns the *constrained* GNE. fitness_lipschitz_constant=1 is the tight
    # constant matching theta=1 in the paper's stability analysis.
    return pg.SinglePopulationGame(
        num_strategies=3,
        fitness_function=_fitness,
        mass=1,
        A_ineq=A_INEQ,
        b_ineq=C_CAP,
        fitness_lipschitz_constant=1,
    )


class TestArcEnergyExample(unittest.TestCase):
    def test_compute_gne_matches_paper(self):
        """compute_gne, with the tight Lipschitz constant, returns the paper GNE."""
        gne = _build_game().compute_gne().ravel()

        # Matches the published equilibrium x* = [0.30, 0.35, 0.35].
        np.testing.assert_allclose(gne, X_STAR, atol=2e-3)

        # Coupling constraint a^T x <= C is active at the GNE.
        self.assertAlmostEqual((A_INEQ @ gne.reshape(-1, 1)).item(), C_CAP, places=2)

        # Regression guard for the FBOS step-size bug: the tight constant L=1 must
        # NOT collapse to the uniform point [1/3, 1/3, 1/3].
        self.assertGreater(np.max(np.abs(gne - 1.0 / 3.0)), 1e-2)

    def test_edm_pdm_closed_loop_converges_to_gne(self):
        """The deterministic EDM-PDM mean dynamics converge to the GNE.

        Validates the Section 6 design claim: the Smith EDM in feedback with the
        smoothing-anticipatory / inequality PDM drives the strategic distribution
        to x* and recovers the grid dual multiplier q_b* = 0.5.
        """
        game = _build_game()
        pdm = pg.PayoffMechanism(h_map=_h_map, w_map=_w_map, n=3, d=4)
        edm = pg.PoissonRevisionProcess(1.0, pg.protocol.Smith(0.1))
        sim = pg.Simulator(game, pdm, edm, num_agents=100)

        out = sim.integrate_edm_pdm(
            t_span=(0, 100),
            x0=np.array([[1.0], [0.0], [0.0]]),
            q0=np.zeros((4, 1)),
            output_trajectory=False,
        )

        x_final = out.x.ravel()
        # State converges to the published GNE.
        np.testing.assert_allclose(x_final, X_STAR, atol=1e-2)
        # Constraint active in the closed loop.
        self.assertAlmostEqual((A_INEQ @ out.x).item(), C_CAP, places=2)
        # The fourth PDM state is the grid dual multiplier q_b -> q_b* = 0.5.
        self.assertAlmostEqual(out.q[3].item(), QB_STAR, places=2)


if __name__ == "__main__":
    unittest.main()
