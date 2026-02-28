from __future__ import annotations

import unittest
from unittest.mock import patch

from popgames.plotting.visualization_mixin import (
    FIGSIZE,
    FIGSIZE_TERNARY,
    FONTSIZE,
    PLOT_DISPATCH,
    VisualizationMixin,
)


class _DummySimulator(VisualizationMixin):
    """Minimal object that can use the mixin."""


class TestVisualizationMixin(unittest.TestCase):
    def test_plot_unknown_type_logs_warning_and_does_not_call_plotters(self) -> None:
        sim = _DummySimulator()

        with patch("popgames.plotting.visualization_mixin.logger") as logger_mock:
            # Patch dispatch methods to ensure none are called
            with patch.dict(
                PLOT_DISPATCH,
                {k: unittest.mock.Mock() for k in PLOT_DISPATCH},
                clear=False,
            ):
                sim.plot(plot_type="does_not_exist")  # type: ignore[arg-type]
                logger_mock.warning.assert_called_once()

    def test_plot_kpi_uses_default_figsize_and_fontsize(self) -> None:
        sim = _DummySimulator()

        plotter = unittest.mock.Mock()
        with patch.dict(PLOT_DISPATCH, {"kpi": plotter}, clear=False):
            sim.plot(plot_type="kpi", show=False)

        plotter.assert_called_once()
        _, kwargs = plotter.call_args
        self.assertIs(kwargs["filename"], None)
        self.assertEqual(kwargs["figsize"], FIGSIZE)
        self.assertEqual(kwargs["fontsize"], FONTSIZE)
        self.assertFalse(kwargs["show"])

    def test_plot_ternary_uses_default_ternary_figsize(self) -> None:
        sim = _DummySimulator()

        plotter = unittest.mock.Mock()
        with patch.dict(PLOT_DISPATCH, {"ternary": plotter}, clear=False):
            sim.plot(plot_type="ternary", show=False)

        plotter.assert_called_once()
        _, kwargs = plotter.call_args
        self.assertEqual(kwargs["figsize"], FIGSIZE_TERNARY)
        self.assertEqual(kwargs["fontsize"], FONTSIZE)
        self.assertFalse(kwargs["show"])

    def test_plot_overrides_figsize_and_fontsize_and_filename(self) -> None:
        sim = _DummySimulator()

        plotter = unittest.mock.Mock()
        with patch.dict(PLOT_DISPATCH, {"kpi": plotter}, clear=False):
            sim.plot(
                plot_type="kpi",
                filename="fig.png",
                figsize=(10, 20),
                fontsize=99,
                show=True,
            )

        plotter.assert_called_once()
        _, kwargs = plotter.call_args
        self.assertEqual(kwargs["filename"], "fig.png")
        self.assertEqual(kwargs["figsize"], (10, 20))
        self.assertEqual(kwargs["fontsize"], 99)
        self.assertTrue(kwargs["show"])

    def test_plot_kwargs_are_forwarded_and_merged(self) -> None:
        sim = _DummySimulator()

        plotter = unittest.mock.Mock()
        with patch.dict(PLOT_DISPATCH, {"univariate": plotter}, clear=False):
            sim.plot(
                plot_type="univariate",
                show=False,
                custom_arg=123,
                another="abc",
            )

        plotter.assert_called_once()
        _, kwargs = plotter.call_args

        # Custom kwargs forwarded
        self.assertEqual(kwargs["custom_arg"], 123)
        self.assertEqual(kwargs["another"], "abc")

        # Standard kwargs injected
        self.assertIn("filename", kwargs)
        self.assertIn("figsize", kwargs)
        self.assertIn("fontsize", kwargs)
        self.assertIn("show", kwargs)
        self.assertFalse(kwargs["show"])


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
