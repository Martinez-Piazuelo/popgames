from __future__ import annotations
import typing

import logging
logger = logging.getLogger(__name__)


from popgames.plotting._plot_config import (
    FIGSIZE,
    FONTSIZE,
)

from popgames.plotting.plotters import (
    plot_univariate_trajectories,
    plot_ternary_trajectories,
)

if typing.TYPE_CHECKING:
    from typing import Any, Callable
    from popgames import Simulator

__all__ = [
    "VisualizationMixin",
]

SUPPORTED_PLOT_TYPES_LITERAL = typing.Literal[
    'kpi',
    'univariate',
    'ternary',
]

PLOT_DISPATCH : dict[SUPPORTED_PLOT_TYPES_LITERAL, Callable[..., None]] = {
    'univariate' : plot_univariate_trajectories,
    'ternary' : plot_ternary_trajectories,
}

class VisualizationMixin:
    """
    Visualization mixin to add plotting functionality to the Simulator class.

    This class is expected to be mixed into the Simulator class.
    It is defined in a separate module to keep simulator.py clean.
    """

    def plot(
            self : Simulator,
            *,
            plot_type : SUPPORTED_PLOT_TYPES_LITERAL = 'univariate',
            filename : str = None,
            figsize : tuple[int, int] = FIGSIZE,
            fontsize : int = FONTSIZE,
            show : bool = True,
            **kwargs : dict[str, Any],
    ) -> None:
        """
        Unified plotting interface supporting some predefined plot types.

        Plot the results of the last simulation.

        Args:
            plot_type (str): Plot type. Defaults to 'x(t)'.
            filename (str, optional): If not None the figure is saved with that filename. Defaults to None.
            figsize (tuple[int, int], optional): Matplotlib's figsize. Defaults to (3, 3).
            fontsize (int, optional): Matplotlib's fontsize. Defaults to 10.
            show (bool, optional): Whether to show the figure. Defaults to True.
            **kwargs: Additional keyword arguments to bypass to the selected plot_type.
                Warning: unrecognized arguments are silently ignored.
        """
        plot_method = PLOT_DISPATCH.get(plot_type, None)

        if plot_method is None:
            logger.warning(
                f'Provided plot type {plot_type} is not supported.'
                f'Use one of: {list(PLOT_DISPATCH.keys())}.'
            )

        else:
            kwargs = kwargs | {
                'filename': filename,
                'figsize' : figsize,
                'fontsize' : fontsize,
                'show' : show,
            }
            plot_method(self, **kwargs)

