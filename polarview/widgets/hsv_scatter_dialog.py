"""Hue vs. Saturation scatter-plot dialog with support for multiple ROIs."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from matplotlib.patheffects import Stroke, Normal
from PyQt6.QtWidgets import QDialog, QVBoxLayout

# Distinct non-black colours for successive ROIs (black reserved for mean/ellipse)
_ROI_COLORS = [
    "red", "blue", "green", "purple",
    "orange", "brown", "magenta", "cyan", "olive", "deeppink",
]


class HSVScatterDialog(QDialog):
    """Display a Hue vs. Saturation scatter plot that accumulates multiple ROIs."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("HSV Scatter Plot — ROI")
        self.setMinimumSize(600, 500)

        self._fig = Figure(figsize=(6, 5), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax = self._fig.add_subplot(111)

        self._ax.set_xlabel("Saturation")
        self._ax.set_ylabel("Hue (degrees)")
        self._ax.set_xlim(0, 1)
        self._ax.set_ylim(0, 360)
        self._ax.set_title("Saturation vs. Hue")
        self._fig.tight_layout()

        self._roi_count = 0

        layout = QVBoxLayout()
        layout.addWidget(self._canvas)
        self.setLayout(layout)

    def add_roi(self, rgb_pixels: np.ndarray, roi_label: str) -> None:
        """Add a new ROI's data to the existing plot.

        Parameters
        ----------
        rgb_pixels : ndarray, shape (N, 3), float64
            RGB pixel values in [0, 1].
        roi_label : str
            Label for this ROI (e.g. "ROI 1").
        """
        rgb_norm = np.clip(rgb_pixels, 0.0, 1.0)
        hsv = rgb_to_hsv(rgb_norm.reshape(-1, 1, 3)).reshape(-1, 3)

        hue = hsv[:, 0]
        saturation = hsv[:, 1]

        roi_color = _ROI_COLORS[self._roi_count % len(_ROI_COLORS)]
        self._roi_count += 1

        ax = self._ax

        # Scatter plot — original size, coloured per ROI
        ax.scatter(
            saturation,
            hue * 360,
            c=roi_color,
            s=30,
            alpha=0.5,
            edgecolors="none",
            zorder=1,
        )

        # Mean and standard deviation
        mean_sat = float(np.mean(saturation))
        mean_hue = float(np.mean(hue)) * 360
        std_sat = float(np.std(saturation))
        std_hue = float(np.std(hue)) * 360

        # White halo so black markers/ellipse pop over coloured dots
        halo = [Stroke(linewidth=4, foreground="white"), Normal()]

        # Plot mean marker as 'x' in black (no label — legend uses colored circle below)
        ax.plot(
            mean_sat, mean_hue, "x",
            color="black", markersize=10, markeredgewidth=2,
            zorder=10, path_effects=halo,
        )

        # Invisible scatter point for legend entry (colored circle matching ROI)
        ax.scatter(
            [], [], c=roi_color, s=40, edgecolors="none",
            label=f"{roi_label}: S={mean_sat:.3f}, H={mean_hue:.1f}°",
        )

        # Plot 2-sigma ellipse in black
        ellipse = Ellipse(
            (mean_sat, mean_hue),
            width=4 * std_sat,
            height=4 * std_hue,
            fill=False,
            edgecolor="black",
            linewidth=2,
            linestyle="--",
            zorder=9, path_effects=halo,
        )
        ax.add_patch(ellipse)

        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(f"Saturation vs. Hue ({self._roi_count} ROI(s))")
        self._fig.tight_layout()
        self._canvas.draw()
