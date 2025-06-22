from __future__ import annotations
import typing

import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy as sp

if typing.TYPE_CHECKING:
    from typing import Optional
    from popgames import PopulationGame

def build_auxiliary_matrices(
        population_game : PopulationGame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute auxiliary matrices to map masses.

    Args:
        population_game (PopulationGame): The population game object to retrieve data from.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The auxiliary matrices.
    """
    aux_vector = np.array(population_game.masses).reshape(population_game.num_populations, 1)
    aux_matrix = np.zeros((population_game.num_populations, population_game.n))
    pos = 0
    for k in range(population_game.num_populations):
        aux_matrix[k, pos:pos+population_game.num_strategies[k]] = 1
        pos += population_game.num_strategies[k]
    aux_vector_ext = np.dot(aux_matrix.T, aux_vector)

    return aux_matrix, aux_vector, aux_vector_ext

def compute_vertices(
        population_game : PopulationGame
) -> Optional[np.ndarray]:
    """
    Compute the vertices of the Polyhedron set for the given population game.

    Args:
        population_game (PopulationGame): The population game object to retrieve data from.

    Returns:
        Optional[np.ndarray]: The vertices of the Polyhedron set for the given population game. If none, returns None.
    """
    # Build auxiliary matrices
    aux_matrix, aux_vector, _ = build_auxiliary_matrices(population_game)
    
    # Concatenate equality constraints
    if population_game.d_eq > 0:
        A_eq = np.vstack((aux_matrix, population_game.A_eq))
        b_eq = np.vstack((aux_vector, population_game.b_eq))
    else:
        A_eq = aux_matrix
        b_eq = aux_vector
    
    # Concatenate inequality constraints
    if population_game.d_ineq > 0:
        A_ineq = np.vstack((-np.eye(population_game.n), population_game.A_ineq))
        b_ineq = np.vstack((np.zeros((population_game.n, 1)), population_game.b_ineq))
    else:
        A_ineq = -np.eye(population_game.n)
        b_ineq = np.zeros((population_game.n, 1))
    
    # Eliminate equality constraints
    x0 = np.linalg.lstsq(A_eq, b_eq, rcond=None)[0]
    Z = sp.linalg.null_space(A_eq)
    A_ineq_red = A_ineq @ Z
    b_ineq_red = b_ineq - A_ineq @ x0

    # Find interior point
    res = sp.optimize.linprog(np.zeros((A_ineq_red.shape[1])), A_ub=A_ineq_red, b_ub=b_ineq_red, method='highs')

    # Case 0
    if not res.success:
        logger.info('No interior point found. The Polyhedron migh be empty!')
        return None
    
    x_int = res.x

    # Case 1:
    if Z.shape[1] == 0:
        logger.info('The Polyhedron is a singleton!')
        return [x0.reshape(-1)]
    
    # Case 2:
    elif Z.shape[1] == 1:
        logger.info('The Polyhedron is a 1D line!')
        vertices = []
        for c in [Z[:, 0], -Z[:, 0]]: # Optimize along both dimensions of the line, i.e., do max and min
            res = sp.optimize.linprog(c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq, method='highs')
            if not res.success:
                logger.warning('Something went wrong!')
                return None
            vertices.append(np.reshape(res.x, -1))
        return vertices 
    
    # Case 3:
    # Compute vertices of the polyhedron
    halfspaces = np.hstack((A_ineq_red, -b_ineq_red))
    hs = sp.spatial.HalfspaceIntersection(halfspaces, x_int)
    vertices_red = hs.intersections
    vertices = []
    for vertex in vertices_red:
        v = np.clip(Z @ vertex.reshape(-1,1) + x0, 0, np.dot(aux_matrix.T, aux_vector)).reshape(-1)
        vertices.append(v)
    return vertices

def map2delta(
        population_game : PopulationGame,
        x : np.ndarray
) -> np.ndarray:
    """
    Auxiliary function to map an input vector to Delta.

    Args:
        population_game (PopulationGame): The population game object.
        x (np.ndarray): The input vector to map.

    Returns:
        np.ndarray: The output vector mapped to Delta.
    """

    # Build auxiliary matrices
    aux_matrix, _, aux_vector_ext = build_auxiliary_matrices(population_game)

    # Map vector to Delta
    masses = np.dot(aux_matrix, x)
    return ((x/np.dot(aux_matrix.T, masses))*aux_vector_ext)  