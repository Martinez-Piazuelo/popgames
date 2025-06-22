from __future__ import annotations
import typing

import logging
logger = logging.getLogger(__name__)

import numpy as np
import cvxpy as cp

from popgames.utilities.polyhedron import build_auxiliary_matrices, map2delta

if typing.TYPE_CHECKING:
    from popgames import PopulationGame


def fbos(
    population_game : PopulationGame,
    max_iter : int = 5000,
    tolerance : float = 1e-6
    ) -> np.ndarray:
    """
    Compute a generalized Nash equilibrium (GNE) for the provided population game.

    The GNE is computed using the `Modified Forward-Backward Operator Splitting Method` (Tseng, P., 2000)

    Args:
        population_game (PopulationGame): the population game
        max_iter (int): the maximum number of iterations
        tolerance (float): the tolerance parameter

    Returns:
        np.ndarray: the computed GNE (if any).
    """
    # Build auxiliary matrices
    aux_matrix, aux_vector, _ = build_auxiliary_matrices(population_game)

    # Compute fixed step-size
    if population_game._fitness_lipschitz_constant is None:
        logger.warning('No fitness_lipschitz_constant provided, using default L=100. FBOS might not converge.')
        lipschitz_constant = 100

    stepsize = 1/lipschitz_constant

    x = map2delta(population_game, np.ones((population_game.n, 1)))    # Initial condition
    z = cp.Variable((population_game.n, 1))

    constraints = [z >= 0, aux_matrix@z == aux_vector]
    if population_game.d_eq > 0:
        constraints.append(population_game.A_eq@z == population_game.b_eq)
    if population_game.d_ineq > 0:
        constraints.append(population_game.A_ineq@z <= population_game.b_ineq)

    for iter in range(max_iter):
        objective = cp.Minimize(0.5*cp.square(cp.norm(z - (x + stepsize*population_game.fitness_function(x)), 2)))
        problem = cp.Problem(objective, constraints)
        problem.solve()

        x_next = z.value + stepsize*(population_game.fitness_function(z.value) - population_game.fitness_function(x))
        x_next = map2delta(population_game, x_next)

        inf_norm = np.max(np.abs(x_next - x))
        if (inf_norm < tolerance):
            break

        x = x_next

    if iter >= max_iter-1:
        logger.warning(f'Maximum number of iterations ({iter}) reached. Computed GNE may not be accurate (error = {inf_norm}).')

    return x
