from __future__ import annotations
import typing

from abc import ABC, abstractmethod
from functools import partial
import numpy as np

if typing.TYPE_CHECKING:
    from typing import Union

from popgames.utilities.input_validators import check_scalar_value_bounds

__all__ = ["Poisson"]


class AlarmClock(ABC):
    """
    Abstract base class for alarm clocks.
    """

    @abstractmethod
    def __call__(self, size : int) -> np.ndarray:
        """
        Subclasses must implement this method to enable the alarm clock to be called as a function.
        """

class Poisson(AlarmClock):
    """
    Poisson alarm clock.
    """

    def __init__(
            self,
            rate : float = 1.0
    ) -> None:
        """
        Initialize the Poisson alarm clock.

        Args:
            rate (float): The rate of the alarm clock.
        """

        check_scalar_value_bounds(
            rate,
            'rate',
            strictly_positive=True
        )

        self.rate = rate

        self._pdf = partial(
            np.random.exponential,
            scale=1/self.rate
        )

    def __call__(
            self,
            size : int
    ) -> Union[float, np.ndarray]:
        """
        Call the Poisson alarm clock.

        Args:
            size (int): Number of samples to retrieve from the clock.

        Returns:
             Union[float, np.ndarray]: The revision times.
        """
        return self._pdf(size = size)


