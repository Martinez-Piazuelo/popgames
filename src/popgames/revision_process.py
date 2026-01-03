from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from popgames.alarm_clock import AlarmClockABC, Poisson
from popgames.revision_protocol import RevisionProtocolABC, Softmax
from popgames.utilities.input_validators import check_type

__all__ = ["RevisionProcessABC", "PoissonRevisionProcess"]

logger = logging.getLogger(__name__)


class RevisionProcessABC(ABC):
    """
    Abstract base class for the revision process.
    """

    def __init__(
        self,
        alarm_clock: AlarmClockABC = None,
        revision_protocol: RevisionProtocolABC = None,
    ) -> None:
        """
        Initialize the revision process object.

        Args:
            alarm_clock (AlarmClockABC): Alarm clock instance.
            revision_protocol (RevisionProtocolABC): Revision protocol instance.
        """

        check_type(arg=alarm_clock, expected_type=AlarmClockABC, arg_name="alarm_clock")
        self.alarm_clock = alarm_clock

        check_type(
            arg=revision_protocol,
            expected_type=RevisionProtocolABC,
            arg_name="revision_protocol",
        )
        self.revision_protocol = revision_protocol

    @abstractmethod
    def sample_next_revision_time(self, size: int) -> np.ndarray:
        """
        Subclasses must implement this method.

        Sample the next revision times for a population of agents.

        Args:
            size (int): Number of agents or samples to generate.

        Returns:
            np.ndarray: A 1D array of shape ``(size,)`` containing the next revision times,
            sampled according to the alarm clock mechanism.
        """

    @abstractmethod
    def sample_next_strategy(self, p: np.ndarray, x: np.ndarray, i: int) -> int:
        """
        Subclasses must implement this method.

        Sample the next strategy for an agent based on current payoffs and strategy.

        Args:
            p (np.ndarray): Payoff vector.
            x (np.ndarray): Population state (strategy distribution).
            i (int): Index of the agent's current strategy.

        Returns:
            int: Index of the newly selected strategy.
        """

    @abstractmethod
    def rhs_edm(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Subclasses must implement this method.

        Evaluate the right-hand side (RHS) of the evolutionary dynamics model (EDM).

        Args:
            x (np.ndarray): Population state (strategic distribution).
            p (np.ndarray): Payoff vector.

        Returns:
            np.ndarray: Time derivative of the strategic distribution, i.e., the RHS of the EDM.
        """


class PoissonRevisionProcess(RevisionProcessABC):
    """
    Poisson Revision Process.
    """

    def __init__(
        self,
        Poisson_clock_rate: int | float = 0.1,
        revision_protocol: RevisionProtocolABC = Softmax(eta=0.1),
    ) -> None:
        """
        Initialize the Poisson revision process object.

        Args:
            Poisson_clock_rate (int): Rate of the Poisson alarm clock. Default is 0.1.
            revision_protocol (RevisionProtocolABC): The class of revision protocol to consider. Default is Softmax(eta=0.1).
        """

        self.Poisson_clock_rate = Poisson_clock_rate

        super().__init__(
            alarm_clock=Poisson(Poisson_clock_rate), revision_protocol=revision_protocol
        )

    def sample_next_revision_time(self, size: int) -> np.ndarray:
        """
        Sample the next revision times for a population of agents equipped with Poisson alarm clocks.

        Args:
            size (int): Number of agents or samples to generate.

        Returns:
            np.ndarray: A 1D array of shape ``(size,)`` containing the next revision times,
            sampled according to the alarm clock mechanism.
        """
        return self.alarm_clock(size)

    def sample_next_strategy(self, p: np.ndarray, x: np.ndarray, i: int) -> int:
        """
        Sample the next strategy for an agent based on current payoffs and strategy.

        Args:
            p (np.ndarray): Payoff vector.
            x (np.ndarray): Population state (strategy distribution).
            i (int): Index of the agent's current strategy.

        Returns:
            int: Index of the newly selected strategy.
        """
        revs = self.revision_protocol(p, x)
        probabilities = revs[:, i].reshape(-1)
        probabilities[i] = 0.0
        probabilities[i] = 1 - sum(probabilities)
        if probabilities[i] < 0:
            logger.warning(
                f"Invalid probabilities = {probabilities}. Cliping them by default."
            )
            probabilities = np.clip(probabilities, 0, 1)
            probabilities = probabilities / probabilities.sum()
        return np.random.choice(np.arange(probabilities.shape[0]), p=probabilities)

    def rhs_edm(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Evaluate the right-hand side (RHS) of the Poisson evolutionary dynamics model (Poisson EDM).

        Args:
            x (np.ndarray): Population state (strategic distribution).
            p (np.ndarray): Payoff vector.

        Returns:
            np.ndarray: Time derivative of the strategic distribution, i.e., the RHS of the Poisson EDM.
        """
        revs = self.revision_protocol(p, x)
        return self.Poisson_clock_rate * np.sum((x.T * revs) - (x * revs.T), 1).reshape(
            -1, 1
        )
