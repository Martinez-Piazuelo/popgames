from __future__ import annotations

import logging
import typing

import cvxpy as cp
import numpy as np

from popgames.utilities.polyhedron import build_auxiliary_matrices, map2delta

if typing.TYPE_CHECKING:
    from popgames import PopulationGame

logger = logging.getLogger(__name__)

# Safety factor for the forward-backward step size. The convergence of the
# modified forward-backward operator splitting method (Tseng, 2000) requires a
# step size strictly smaller than 1 / lipschitz_constant. Using exactly
# 1 / lipschitz_constant sits on the non-convergent boundary and can cause the
# iteration to stall on the first step (returning the initial point as a bogus
# "converged" GNE). A factor in (0, 1) keeps the step strictly inside the
# convergence region.
STEPSIZE_SAFETY_FACTOR = 0.99


def fbos(
    population_game: PopulationGame,
    max_iter: int = 5000,
    tolerance: float = 1e-6,
    gamma: float = STEPSIZE_SAFETY_FACTOR,
) -> np.ndarray:
    """
    Compute a generalized Nash equilibrium (GNE) for the provided population game.

    The GNE is computed using the `Modified Forward-Backward Operator Splitting Method` (Tseng, P., 2000)

    Args:
        population_game (PopulationGame): the population game
        max_iter (int): the maximum number of iterations
        tolerance (float): the tolerance parameter
        gamma (float): safety factor in (0, 1) applied to the step size so that
            ``stepsize = gamma / lipschitz_constant`` stays strictly inside the
            convergence region.

    Returns:
        np.ndarray: the computed GNE (if any).
    """
    if not 0 < gamma < 1:
        raise ValueError(f"gamma must be in the open interval (0, 1), got {gamma}.")

    # Build auxiliary matrices
    aux_matrix, aux_vector, _ = build_auxiliary_matrices(population_game)

    # Compute fixed step-size
    lipschitz_constant = population_game._fitness_lipschitz_constant
    if lipschitz_constant is None:
        logger.warning(
            "No fitness_lipschitz_constant provided, using default L=100. FBOS might not converge."
        )
        lipschitz_constant = 100

    # The step size must be strictly smaller than 1 / lipschitz_constant for the
    # method to converge; the safety factor gamma keeps it off the boundary.
    stepsize = gamma / lipschitz_constant

    x = map2delta(population_game, np.ones((population_game.n, 1)))  # Initial condition
    z = cp.Variable((population_game.n, 1))

    constraints = [z >= 0, aux_matrix @ z == aux_vector]
    if population_game.d_eq > 0:
        constraints.append(population_game.A_eq @ z == population_game.b_eq)
    if population_game.d_ineq > 0:
        constraints.append(population_game.A_ineq @ z <= population_game.b_ineq)

    i = 0
    inf_norm = np.inf
    for i in range(max_iter):
        objective = cp.Minimize(
            0.5
            * cp.square(
                cp.norm(z - (x + stepsize * population_game.fitness_function(x)), 2)
            )
        )
        problem = cp.Problem(objective, constraints)
        problem.solve()

        x_next = z.value + stepsize * (
            population_game.fitness_function(z.value)
            - population_game.fitness_function(x)
        )
        x_next = map2delta(population_game, x_next)

        inf_norm = np.max(np.abs(x_next - x))
        # Require at least one update before honoring the convergence test so that
        # an oversized step that stalls on the very first iteration cannot
        # masquerade as convergence and silently return the initial point.
        if inf_norm < tolerance and i > 0:
            x = x_next
            break

        x = x_next

    if i >= max_iter - 1:
        logger.warning(
            f"Maximum number of iterations ({i}) reached. Computed GNE may not be accurate (error = {inf_norm})."
        )

    return x
