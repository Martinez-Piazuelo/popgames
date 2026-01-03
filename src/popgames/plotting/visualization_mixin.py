from __future__ import annotations

import logging
import typing

from popgames.plotting._plot_config import (
    FIGSIZE,
    FIGSIZE_TERNARY,
    FONTSIZE,
)
from popgames.plotting.plotters import (
    plot_kpi_over_time,
    plot_ternary_trajectories,
    plot_univariate_trajectories_joint,
    plot_univariate_trajectories_split,
)

if typing.TYPE_CHECKING:
    from typing import Any, Callable

    from popgames import Simulator

__all__ = [
    "VisualizationMixin",
]

SUPPORTED_PLOT_TYPES_LITERAL = typing.Literal[
    "kpi",
    "univariate",
    "univariate_split",
    "ternary",
]

PLOT_DISPATCH: dict[SUPPORTED_PLOT_TYPES_LITERAL, Callable[..., None]] = {
    "kpi": plot_kpi_over_time,
    "univariate": plot_univariate_trajectories_joint,
    "univariate_split": plot_univariate_trajectories_split,
    "ternary": plot_ternary_trajectories,
}


logger = logging.getLogger(__name__)


class VisualizationMixin:
    """
    Visualization mixin to add plotting functionality to the Simulator class.

    This class is expected to be mixed into the Simulator class.
    It is defined in a separate module to keep simulator.py clean.
    """

    def plot(
        self: Simulator,
        *,
        plot_type: SUPPORTED_PLOT_TYPES_LITERAL = "kpi",
        filename: str = None,
        figsize: tuple[int, int] = None,
        fontsize: int = None,
        show: bool = True,
        **kwargs: dict[str, Any] | Any,
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
                f"Provided plot type {plot_type} is not supported."
                f"Use one of: {list(PLOT_DISPATCH.keys())}."
            )

        else:
            DEFAULT_FIGSIZE = FIGSIZE_TERNARY if plot_type == "ternary" else FIGSIZE
            kwargs = kwargs | {
                "filename": filename,
                "figsize": figsize if figsize is not None else DEFAULT_FIGSIZE,
                "fontsize": fontsize if fontsize is not None else FONTSIZE,
                "show": show,
            }
            plot_method(self, **kwargs)
