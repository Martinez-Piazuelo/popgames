from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from popgames.utilities.input_validators import check_scalar_value_bounds

__all__ = [
    'RevisionProtocolABC',
    'Softmax',
    'Smith',
    'BNN'
]

class RevisionProtocolABC(ABC):
    """
    Abstract base class for revision protocols.
    """

    @abstractmethod
    def __call__(
            self,
            p : np.ndarray,
            x : np.ndarray
    ) -> np.ndarray:
        """
        Subclasses must implement this method to enable the revision protocol to be called as a function.

        Args:
            p (np.ndarray): The payoff vector with shape (n, 1).
            x (np.ndarray): The population state vector with shape (n, 1).

        Returns:
            np.ndarray: The switching probabilities as a matrix with shape (n, n).
        """

class Softmax(RevisionProtocolABC):
    """
    Softmax revision protocol. Also known as Logit-Choice revision protocol.
    """
    def __init__(
            self,
            eta : float
    ) -> None:
        """
        Initialize the Softmax revision protocol object.

        Args:
            eta (float): The `temperature` or `noise` parameter.
        """
        check_scalar_value_bounds(
            arg=eta,
            arg_name='eta',
            strictly_positive=True
        )
        self.eta = eta

    def __call__(
            self,
            p : np.ndarray,
            x : np.ndarray
    ) -> np.ndarray:
        """
        Evaluate the Softmax revision protocol.

        Args:
            p (np.ndarray): The payoff vector with shape (n, 1).
            x (np.ndarray): The population state vector with shape (n, 1).

        Returns:
            np.ndarray: The switching probabilities as a matrix with shape (n, n).

        Examples:
            >>> import numpy as np
            >>> from popgames.revision_protocol import Softmax
            >>> softmax = Softmax(eta=1)
            >>> p = np.array([1, -1, 2]).reshape(3, 1)
            >>> x = np.array([0.1, 0.7, 0.2]).reshape(3, 1)
            >>> softmax(p, x)
            array([[0.25949646, 0.25949646, 0.25949646],
                   [0.03511903, 0.03511903, 0.03511903],
                   [0.70538451, 0.70538451, 0.70538451]])
        """

        logits = np.exp((p/self.eta) - np.max(p/self.eta))
        probabilities = logits/(logits.sum())
        return np.dot(probabilities, np.ones_like(probabilities).T)

class Smith(RevisionProtocolABC):
    """
    Smith revision protocol.
    """
    def __init__(
            self,
            scale : float
    ) -> None:
        """
        Initialize the Smith revision protocol object.

        Args:
            scale (float): The scale parameter to ensure well-posed probabilities.
        """

        check_scalar_value_bounds(
            arg=scale,
            arg_name='scale',
            strictly_positive=True
        )
        self.scale = scale
    
    def __call__(
            self,
            p : np.ndarray,
            x : np.ndarray
    ) -> np.ndarray:
        """
        Evaluate the Smith revision protocol.

        Args:
            p (np.ndarray): The payoff vector with shape (n, 1).
            x (np.ndarray): The population state vector with shape (n, 1).

        Returns:
            np.ndarray: The switching probabilities as a matrix with shape (n, n).

        Examples:
            >>> import numpy as np
            >>> from popgames.revision_protocol import Smith
            >>> smith = Smith(scale=0.1)
            >>> p = np.array([1, -1, 2]).reshape(3, 1)
            >>> x = np.array([0.1, 0.7, 0.2]).reshape(3, 1)
            >>> smith(p, x)
            array([[0. , 0.2, 0. ],
                   [0. , 0. , 0. ],
                   [0.1, 0.3, 0. ]])
        """

        return np.maximum(p - p.T, 0)*self.scale

class BNN(RevisionProtocolABC):
    """
    Brown-von Neumann-Nash (BNN) revision protocol.
    """
    def __init__(
            self,
            scale : float
    ) -> None:
        """
        Initialize the BNN revision protocol object.

        Args:
            scale (float): The scale parameter to ensure well-posed probabilities.
        """

        check_scalar_value_bounds(
            arg=scale,
            arg_name='scale',
            strictly_positive=True
        )
        self.scale = scale
    
    def __call__(
            self,
            p : np.ndarray,
            x : np.ndarray
    ) -> np.ndarray:
        """
        Evaluate the BNN revision protocol.

        Args:
            p (np.ndarray): The payoff vector with shape (n, 1).
            x (np.ndarray): The population state vector with shape (n, 1).

        Returns:
            np.ndarray: The switching probabilities as a matrix with shape (n, n).

        Examples:
            >>> import numpy as np
            >>> from popgames.revision_protocol import BNN
            >>> bnn = BNN(scale=0.1)
            >>> p = np.array([1, -1, 2]).reshape(3, 1)
            >>> x = np.array([0.1, 0.7, 0.2]).reshape(3, 1)
            >>> bnn(p, x)
            array([[0.12, 0.12, 0.12],
                   [0.  , 0.  , 0.  ],
                   [0.22, 0.22, 0.22]])
        """

        p_hat = np.dot(x.T, p)/(x.sum())
        delta_p = p - p_hat[0]
        return np.maximum(np.dot(delta_p, np.ones_like(delta_p).T), 0)*self.scale

