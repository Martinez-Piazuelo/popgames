from __future__ import annotations
import typing

from types import SimpleNamespace
import numpy as np
import scipy as sp

from popgames.utilities.input_validators import (
    check_type,
    check_scalar_value_bounds,
    check_function_signature
)

if typing.TYPE_CHECKING:
    from typing import Union, Callable


__all__ = [
    'PayoffMechanism',
]

class PayoffMechanism:
    """
    Payoff mechanism implemented as a payoff dynamics model (PDM).
    """
    def __init__(
            self,
            h_map : Union[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], np.ndarray]],
            n : int,
            w_map : Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
            d : int = 0
    ) -> None:
        """
        Initialize the payoff mechanism instance.

        Args:
            h_map (Union[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], np.ndarray]]): Output function of the PDM.
            n (int): Dimensionality of the expected input vector x.
            w_map (Callable[[np.ndarray, np.ndarray], np.ndarray], optional): Right-hand side of the PDM dynamics.
            Defaults to None, corresponding to a memory-less PDM.
            d (int, optional): Dimensionality of the PDM state vector x. Defaults to 0 (in the memoryless case).
        """

        # Check input dimensions
        check_type(
            arg=n,
            expected_type=int,
            arg_name='n'
        )
        check_scalar_value_bounds(
            arg=n,
            arg_name='n',
            strictly_positive=True
        )
        self.n = n

        check_type(
            arg=d,
            expected_type=int,
            arg_name='d'
        )
        check_scalar_value_bounds(
            arg=d,
            arg_name='d',
            min_value=0
        )
        self.d = d

        # Check input functions
        if self.d == 0: # -> Case d=0
            check_function_signature(
                arg=h_map,
                expected_input_shapes=[(self.n, 1)],
                expected_output_shape=(self.n, 1),
                name='h_map'
            )
            self.h_map = self._unsqueeze_h_map(h_map)
            self.w_map = self._dummy_w_map()
        else: # -> Case d>0
            check_function_signature(
                arg=h_map,
                expected_input_shapes=[(self.d, 1), (self.n, 1)],
                expected_output_shape=(self.n, 1),
                name='h_map'
            )
            check_function_signature(
                arg=w_map,
                expected_input_shapes=[(self.d, 1), (self.n, 1)],
                expected_output_shape=(self.d, 1),
                name='w_map'
            )
            self.h_map = h_map
            self.w_map = w_map

    def integrate(
            self,
            q0 : np.ndarray,
            x0 : np.ndarray,
            t_span : tuple,
            method : str = 'Radau',
            output_trajectory : bool = True
    ) -> SimpleNamespace:
        """
        Numerically integrate the PDM.

        This method relies on ``scipy.integrate.solve_ivp``.

        Args:
            q0 (np.ndarray): Initial PDM state vector of shape (d, 1).
            x0 (np.ndarray): Initial PDM input vector of shape (n, 1).
            t_span (tuple): Time span of integration.
            method (str, optional): Integration method. Defaults to 'Radau'.
            output_trajectory (bool, optional): Whether to output the trajectory or just the final state-output pair.
                Defaults to True.

        Returns:
            SimpleNamespace: Contains results of the integration as a SimpleNameSpace with keys ``t``, ``q``, and ``p``,
            for the time, state, and output vectors, respectively.
        """
        
        if self.d == 0: # Memoryless case
            
            p = self.h_map(q0, x0)

            if output_trajectory:
                p = np.hstack([p, p]) # There are only two relevant points in t_span

            return SimpleNamespace(q=np.zeros((0, 1)), p=p) # q is an empty placeholder

        else: # Dynamic case
            y_in = np.vstack([q0, x0]).reshape(self.d + self.n,)
            sol = sp.integrate.solve_ivp(self._w_map_wrapped, t_span, y_in, method=method)
            
            if output_trajectory:
                T = sol.y.shape[1]
                q = sol.y[:self.d, :]
                p = np.zeros((self.n, T))
                for t in range(T):
                    p[:, t] = self._h_map_wrapped(sol.y[:, t]) # TODO: Can h_map be evaluated in batches to remove this loop?
            else:
                q = sol.y[:self.d, -1].reshape(self.d, 1)
                p = self.h_map(q, x0)

            return SimpleNamespace(t=sol.t, q=q, p=p)
    
    @staticmethod
    def _unsqueeze_h_map(
            h_map : Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """
        Internal static method to maintain a common signature for the h_map function.

        Should not be called directly from outside the class.

        Args:
            h_map (Union[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], np.ndarray]]):
                Squeezed output function of the PDM.

        Returns:
            Callable[[np.ndarray, np.ndarray], np.ndarray]: Unsqueezed output function of the PDM.
        """
        def unsqueeze(_, x : np.ndarray) -> np.ndarray:
            return h_map(x)
        return unsqueeze
    
    @staticmethod
    def _dummy_w_map() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """
        Internal static method to maintain a common interface for the PDM dynamics.

        Should not be called from outside the class.

        Returns:
            Callable[[np.ndarray, np.ndarray], np.ndarray]: RHS of dummy dynamics.
        """
        def placeholder(*_):
            return np.zeros((0, 1))
        return placeholder
        
    def _h_map_wrapped(
            self,
            y : np.ndarray
    ) -> np.ndarray:
        """
        Internal method to wrap the h_map function and enable compatibility with ``scipy.integrate.solve_ivp``.

        Should not be called from outside the class.

        Args:
            y (np.ndarray): Input vector of shape (d+n, 1).

        Returns:
            np.ndarray: Output vector of shape (n,).
        """
        q = y[:self.d].reshape(self.d, 1)
        x = y[self.d:].reshape(self.n, 1)
        return self.h_map(q, x).reshape(self.n,)

    def _w_map_wrapped(
            self,
            t : float,
            y : np.ndarray
    ) -> np.ndarray:
        """
        Internal method to wrap the w_map function and enable compatibility with ``scipy.integrate.solve_ivp``.

        Should not be called from outside the class.

        Args:
            t (float): Integration time (placeholder).
            y (np.ndarray): Input vector of shape (d+n, 1).

        Returns:
            np.ndarray: Output vector of shape (d+n,).
        """
        q = y[:self.d].reshape(self.d, 1)
        x = y[self.d:].reshape(self.n, 1)
        return np.vstack([self.w_map(q, x), np.zeros_like(x)]).reshape(self.d + self.n,)