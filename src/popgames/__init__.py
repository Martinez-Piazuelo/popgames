from __future__ import annotations

import importlib.metadata
import logging.config

import popgames.alarm_clock as clock
import popgames.revision_protocol as protocol
from popgames.payoff_mechanism import PayoffMechanism
from popgames.population_game import PopulationGame, SinglePopulationGame
from popgames.revision_process import PoissonRevisionProcess
from popgames.simulator import Simulator

__all__ = [
    "PoissonRevisionProcess",
    "PopulationGame",
    "SinglePopulationGame",
    "PayoffMechanism",
    "Simulator",
    "clock",
    "protocol",
    "configure_logging",
]

try:
    __version__ = importlib.metadata.version("popgames")
except importlib.metadata.PackageNotFoundError:
    __version__ = None


def configure_logging(level: str | int = "INFO") -> None:
    """
    Configures the logging system for the application with a specified logging level.

    Args:
        level (str | int): Logging level. Defaults to "INFO".
    """
    cfg = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "DEBUG",
            },
        },
        "root": {
            "handlers": ["console"],
            "level": level,
        },
    }
    logging.config.dictConfig(cfg)
