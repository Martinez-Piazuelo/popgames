from __future__ import annotations

import math
import os
import typing

import matplotlib.pyplot as plt
import numpy as np
import ternary
from matplotlib.axes import Axes

from popgames.plotting._plot_config import DPI

if typing.TYPE_CHECKING:
    from typing import Sequence

    from matplotlib.figure import Figure


class Panels:
    """
    Provide the axes of the panels drawn by a plotter, in order.

    By default, each panel is drawn on a new figure, which is saved, shown, or closed by ``finish``. If the caller gives
    its own axes (``ax``), the panels are drawn on them instead, and the caller owns the figures: ``finish`` neither
    saves, shows, nor closes them, and the plotters leave their layout untouched.
    """

    def __init__(
        self,
        ax: Axes | Sequence[Axes] | None,
        num_panels: int,
        filename: str | None = None,
    ) -> None:
        """
        Args:
            ax (Axes | Sequence[Axes] | None): The axes given by the caller: one per panel, or a single one if there is
                exactly one panel. Arrays of axes (e.g., from ``plt.subplots``) are flattened in row-major order. None
                to draw each panel on a new figure.
            num_panels (int): The number of panels the plotter draws.
            filename (str, optional): The filename given by the caller, which cannot be combined with ``ax``.
        """
        if ax is None:
            self.axes = None
        else:
            self.axes = (
                [ax]
                if isinstance(ax, Axes)
                else list(np.ravel(np.array(ax, dtype=object)))
            )
            if len(self.axes) != num_panels:
                raise ValueError(
                    f"This plot draws {num_panels} panel(s), but {len(self.axes)} axes were given. Pass one axes per "
                    f"panel."
                )
            if filename is not None:
                raise ValueError(
                    "filename cannot be combined with ax: save the figure that holds the axes instead."
                )
        self._next = 0

    @property
    def owned(self) -> bool:
        """Whether the plotter creates (and finishes) its own figures."""
        return self.axes is None

    def new(self, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
        """The figure and axes of the next panel."""
        if self.owned:
            return plt.subplots(figsize=figsize)
        ax = self.axes[self._next]
        self._next += 1
        return ax.figure, ax

    def new_grid(
        self,
        num: int,
        ncols: int,
        panel_size: tuple[float, float],
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Figure, list[Axes], list[tuple[bool, bool]]]:
        """
        The figure and axes of the next ``num`` panels, laid out in a grid of ``ncols`` columns (sharing the time
        axis) when the plotter creates the figure.

        Returns:
            tuple[Figure, list[Axes], list[tuple[bool, bool]]]: The figure, the axes, and for each axes whether it is
            on the bottom edge and on the left edge of its grid (where the axis labels go).
        """
        if self.owned:
            ncols = min(ncols, num)
            nrows = math.ceil(num / ncols)
            fig, grid = plt.subplots(
                nrows,
                ncols,
                figsize=figsize or (panel_size[0] * ncols, panel_size[1] * nrows),
                sharex=True,
                squeeze=False,
            )
            axes = list(grid.flat)
            for extra in axes[num:]:
                extra.remove()
            axes = axes[:num]
            edges = [(i + ncols >= num, i % ncols == 0) for i in range(num)]
            for ax, (bottom, _) in zip(axes, edges):
                ax.xaxis.set_tick_params(labelbottom=bottom)
            return fig, axes, edges

        axes = self.axes[self._next : self._next + num]
        self._next += num
        edges = []
        for ax in axes:
            spec = ax.get_subplotspec()
            edges.append(
                (spec.is_last_row(), spec.is_first_col()) if spec else (True, True)
            )
        return axes[0].figure, axes, edges

    def new_ternary(self, figsize: tuple[float, float], scale: float = 1):
        """The figure and ternary axes (``python-ternary``) of the next panel."""
        if self.owned:
            fig, tax = ternary.figure(scale=scale)
            fig.set_size_inches(figsize[0], figsize[1])
        else:
            fig, tax = ternary.figure(ax=self.new(figsize)[1], scale=scale)
        return fig, tax

    def tight_layout(self, fig: Figure) -> None:
        if self.owned:
            fig.tight_layout()

    def finish(
        self, fig: Figure, filename: str | None, suffix: str | None, show: bool
    ) -> None:
        """Save the figure (with the suffix appended to the file name), then show or close it."""
        if not self.owned:
            return
        if filename is not None:
            name, ext = os.path.splitext(filename)
            path = f"{name}_{suffix}{ext}" if suffix else filename
            fig.savefig(
                path,
                bbox_inches="tight",
                pad_inches=0.05,
                dpi=DPI if ext == ".png" else None,
            )
        if show:
            plt.show()
        else:
            plt.close(fig)
