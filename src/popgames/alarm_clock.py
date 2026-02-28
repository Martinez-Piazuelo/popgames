from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from functools import partial

import numpy as np

from popgames.utilities.input_validators import check_scalar_value_bounds

if typing.TYPE_CHECKING:
    from typing import Union

__all__ = ["AlarmClockABC", "Poisson"]


class AlarmClockABC(ABC):
    """
    Abstract base class for alarm clocks.
    """

    @abstractmethod
    def __call__(self, size: int) -> Union[float, np.ndarray]:
        """
        Subclasses must implement this method to enable the alarm clock to be called as a function.

        Args:
            size (int): Number of samples to retrieve from the clock.

        Returns:
             Union[float, np.ndarray]: The revision times.
        """


class Poisson(AlarmClockABC):
    """
    Poisson alarm clock.
    """

    def __init__(self, rate: float = 1.0) -> None:
        """
        Initialize the Poisson alarm clock.

        Args:
            rate (float): The rate of the alarm clock. Defaults to 1.0.
        """

        check_scalar_value_bounds(rate, "rate", strictly_positive=True)

        self.rate = rate

        self._pdf = partial(np.random.exponential, scale=1 / self.rate)

    def __call__(self, size: int) -> Union[float, np.ndarray]:
        """
        Call the Poisson alarm clock.

        Args:
            size (int): Number of samples to retrieve from the clock.

        Returns:
             Union[float, np.ndarray]: The revision times.

        Examples:
            >>> from popgames.alarm_clock import Poisson
            >>> clock = Poisson(rate=1.0)
            >>> clock(3)
            array([2.34438986, 0.53956626, 0.80216914])
        """
        return self._pdf(size=size)
