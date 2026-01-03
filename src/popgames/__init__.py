import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'colored': {
            '()': 'colorlog.ColoredFormatter',
            'format': '%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'log_colors': {
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'bold_red',
            },
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'colored',
            'level': 'DEBUG',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
}

logging.config.dictConfig(LOGGING_CONFIG)

from popgames.revision_process import PoissonRevisionProcess
from popgames.population_game import PopulationGame, SinglePopulationGame
from popgames.payoff_mechanism import PayoffMechanism
from popgames.simulator import Simulator

# Aliases
import popgames.alarm_clock as clock
import popgames.revision_protocol as protocol


__all__ = [
    'PoissonRevisionProcess',
    'PopulationGame',
    'SinglePopulationGame',
    'PayoffMechanism',
    'Simulator',
]

__version__ = '0.1.0'