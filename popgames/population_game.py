from __future__ import annotations
import typing

import numpy as np
import numbers

from popgames.utilities.input_validators import (
    check_type,
    check_array_shape,
    check_scalar_value_bounds,
    check_valid_list,
    check_function_signature,
)
from popgames.utilities.gnep_solvers import fbos
from popgames.utilities.polyhedron import compute_vertices

if typing.TYPE_CHECKING:
    from typing import Callable

__all__ = [
    'PopulationGame',
    'SinglePopulationGame'
]

class PopulationGame:
    """
    Multi-Population game.
    """
    def __init__(
            self,
            num_populations : int,
            num_strategies : list[int],
            fitness_function : Callable[[np.ndarray], np.ndarray],
            masses : list[float] = None,
            A_eq : np.ndarray  = None,
            b_eq : np.ndarray = None,
            A_ineq : np.ndarray = None,
            b_ineq : np.ndarray = None,
            g_ineq : Callable[[np.ndarray], np.ndarray] = None,
            Dg_ineq : Callable[[np.ndarray], np.ndarray] = None,
            fitness_lipschitz_constant : float = None
    ) -> None:
        """
        Initialize the population game object.

        Args:
            num_populations (int): Number of populations.
            num_strategies (list[int]): Number of strategies.
            fitness_function (Callable[[np.ndarray], np.ndarray]): Fitness function.
            masses (list[float]): Population masses.
            A_eq (np.ndarray): Matrix A in equality constraints of the form Ax = b.
            b_eq (np.ndarray): Vector b in equality constraints of the form Ax = b.
            A_ineq (np.ndarray): Matrix A in inequality constraints of the form Ax <= b.
            b_ineq (np.ndarray): Vector b in inequality constraints of the form Ax <= b.
            g_ineq (Callable[[np.ndarray], np.ndarray]): Function g(x) in inequality constraints of the form g(x) <= 0.
            Dg_ineq (Callable[[np.ndarray], np.ndarray]): Jacobian matrix of g(x).
            fitness_lipschitz_constant (float): Lipschitz constant of the fitness function.
        """

        check_type(
            arg=num_populations,
            expected_type=int,
            arg_name='num_populations'
        )
        check_scalar_value_bounds(
            arg=num_populations,
            arg_name='num_populations',
            strictly_positive=True
        )
        self.num_populations = num_populations

        check_valid_list(
            arg=num_strategies,
            length=num_populations,
            internal_type=int,
            name='num_strategies',
            strictly_positive=True
        )
        self.num_strategies = num_strategies
        self.n = sum(num_strategies)
        
        if masses is not None:
            check_valid_list(
                arg=masses,
                length=num_populations,
                internal_type=numbers.Number,
                name='masses',
                strictly_positive=True
            )
        self.masses = masses if masses is not None else self.n*[1.]

        check_function_signature(
            arg=fitness_function,
            expected_input_shapes=[(self.n, 1)],
            expected_output_shape=(self.n, 1),
            name='fitness_function'
        )
        self.fitness_function = fitness_function

        for name, exp_type, ncols in zip(['A_eq', 'b_eq', 'A_ineq', 'b_ineq'], # TODO: add logic for convex inequality constraints
                                         [np.ndarray, (numbers.Number, np.ndarray), np.ndarray, (numbers.Number, np.ndarray)],
                                         [self.n, 1, self.n, 1]):
            value = locals()[name]
            if value is not None:
                check_type(value, exp_type, name)
                value = np.array(value) if type(value) != np.ndarray else value
                assert value.ndim <= 2, f'ERROR: Input {name}={value} of type np.ndarray must have maximum two dimensions.'
                try:
                    value = value.reshape((-1, ncols))
                except:
                    raise ValueError(f'ERROR: Cannot recast input {name} into the expected shape = {(-1, ncols)}.')
            setattr(self, name, value)

        self.d_eq = 0 if self.A_eq is None else self.A_eq.shape[0]
        self.d_ineq = 0 if self.A_ineq is None else self.A_ineq.shape[0]

        if self.d_eq > 0:
            check_array_shape(
                arg=self.b_eq,
                expected_shape=(self.d_eq, 1),
                arg_name='b_eq'
            )
        
        if self.d_ineq > 0:
            check_array_shape(
                arg=self.b_ineq,
                expected_shape=(self.d_ineq, 1),
                arg_name='b_ineq'
            )

        self.g_ineq, self.Dg_ineq = g_ineq, Dg_ineq # TODO: These are place holders for now
        if self.g_ineq is not None:
            pass #TODO

        if fitness_lipschitz_constant is not None:
            check_scalar_value_bounds(
                arg=fitness_lipschitz_constant,
                arg_name='fitness_lipschitz_constant',
                strictly_positive=True
            )
        self._fitness_lipschitz_constant = fitness_lipschitz_constant

    def compute_gne(
            self,
            max_iter : int = 5000,
            tolerance : float = 1e-6
    ) -> np.ndarray:
        """
        Compute a generalized Nash equilibrium (GNE) for the population game, assuming one exists.

        The GNE is computed using the `Modified Forward-Backward Operator Splitting Method` (Tseng, P., 2000).

        Args:
            max_iter (int): Maximum number of iterations.
            tolerance (float): Convergence tolerance.

        Returns:
            np.ndarray: The computed GNE (if any).
        """

        check_type(
            arg=max_iter,
            expected_type=int,
            arg_name='max_iter'
        )
        check_scalar_value_bounds(
            arg=max_iter,
            arg_name='max_iter',
            strictly_positive=True
        )
        check_scalar_value_bounds(
            arg=tolerance,
            arg_name='tolerance',
            strictly_positive=True
        )
        return fbos(
            population_game=self,
            max_iter=max_iter,
            tolerance=tolerance
        )

    def compute_polyhedron_vertices(self) -> list:
        """
        Compute the polyhedron vertices of the feasible set of the population game.

        Returns:
            list: The vertices of the feasible set of the population game.
        """
        return compute_vertices(self)

class SinglePopulationGame(PopulationGame):
    """
    Population game with a single population.
    """
    def __init__(
            self,
            num_strategies: int,
            fitness_function: Callable[[np.ndarray], np.ndarray],
            mass: float = None,
            A_eq : np.ndarray  = None,
            b_eq : np.ndarray = None,
            A_ineq : np.ndarray = None,
            b_ineq : np.ndarray = None,
            g_ineq : Callable[[np.ndarray], np.ndarray] = None,
            Dg_ineq : Callable[[np.ndarray], np.ndarray] = None,
            fitness_lipschitz_constant: float = None,
    ) -> None:
        """
        Initialize the single-population game object.

        Args:
            num_populations (int): Number of populations.
            num_strategies (list[int]): Number of strategies.
            fitness_function (Callable[[np.ndarray], np.ndarray]): Fitness function.
            masses (list[float]): Population masses.
            A_eq (np.ndarray): Matrix A in equality constraints of the form Ax = b.
            b_eq (np.ndarray): Vector b in equality constraints of the form Ax = b.
            A_ineq (np.ndarray): Matrix A in inequality constraints of the form Ax <= b.
            b_ineq (np.ndarray): Vector b in inequality constraints of the form Ax <= b.
            g_ineq (Callable[[np.ndarray], np.ndarray]): Function g(x) in inequality constraints of the form g(x) <= 0.
            Dg_ineq (Callable[[np.ndarray], np.ndarray]): Jacobian matrix of g(x).
            fitness_lipschitz_constant (float): Lipschitz constant of the fitness function.
        """

        super().__init__(
            num_populations=1,
            num_strategies=[num_strategies],
            fitness_function=fitness_function,
            masses=[mass],
            A_eq=A_eq,
            b_eq=b_eq,
            A_ineq=A_ineq,
            b_ineq=b_ineq,
            g_ineq=g_ineq,
            Dg_ineq=Dg_ineq,
            fitness_lipschitz_constant=fitness_lipschitz_constant
        )
