from __future__ import annotations

import typing

import numpy as np

from popgames.utilities.input_validators import (
    check_scalar_value_bounds,
    check_type,
)

if typing.TYPE_CHECKING:
    from typing import Union

    from popgames.population_game import PopulationGame

__all__ = [
    "sample_initial_states",
]


def sample_initial_states(
    population_game: PopulationGame,
    num: int,
    seed: Union[int, np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample strategic distributions uniformly at random, e.g., as initial states of ``Simulator.run_ensemble``.

    The strategic distribution of each population is sampled uniformly on its simplex (scaled by the mass of the
    population), independently of the other populations.

    Args:
        population_game (PopulationGame): The population game.
        num (int): The number of strategic distributions.
        seed (Union[int, np.random.Generator], optional): Seed or random number generator. Defaults to None, in which
            case NumPy's global random state is used (see ``np.random.seed``).

    Returns:
        np.ndarray: The strategic distributions, of shape ``(num, n, 1)``.

    Examples:
        >>> import numpy as np
        >>> import popgames as pg
        >>> from popgames.utilities import sample_initial_states
        >>> game = pg.SinglePopulationGame(num_strategies=3, fitness_function=lambda x: -x)
        >>> x0s = sample_initial_states(game, num=5, seed=0)
        >>> x0s.shape
        (5, 3, 1)
        >>> bool(np.allclose(x0s.sum(axis=1), 1.0))
        True
    """
    check_type(arg=num, expected_type=int, arg_name="num")
    check_scalar_value_bounds(arg=num, arg_name="num", strictly_positive=True)
    rng = np.random if seed is None else np.random.default_rng(seed)

    blocks = [
        population_game.masses[k]
        * rng.dirichlet(np.ones(population_game.num_strategies[k]), size=num)
        for k in range(population_game.num_populations)
    ]
    return np.concatenate(blocks, axis=1)[:, :, np.newaxis]
