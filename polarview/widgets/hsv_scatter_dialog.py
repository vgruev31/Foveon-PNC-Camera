"""Hue vs. Saturation scatter-plot dialog for an ROI region."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QDialog, QVBoxLayout


class HSVScatterDialog(QDialog):
    """Display a Hue vs. Saturation scatter plot for the given RGB pixels."""

    def __init__(self, rgb_pixels: np.ndarray, parent=None) -> None:
        """
        Parameters
        ----------
        rgb_pixels : ndarray, shape (N, 3), float64
            RGB pixel values in [0, 1].
        """
        super().__init__(parent)
        self.setWindowTitle("HSV Scatter Plot — ROI")
        self.setMinimumSize(600, 500)

        rgb_norm = np.clip(rgb_pixels, 0.0, 1.0)
        # rgb_to_hsv expects (..., 3) with values in [0, 1]
        hsv = rgb_to_hsv(rgb_norm.reshape(-1, 1, 3)).reshape(-1, 3)

        hue = hsv[:, 0]
        saturation = hsv[:, 1]

        # Colour each dot by its actual HSV colour (V=1 for visibility)
        n = len(hue)
        hsv_colours = np.stack([hue, saturation, np.ones(n)], axis=-1)
        dot_colours = hsv_to_rgb(hsv_colours.reshape(-1, 1, 3)).reshape(-1, 3)

        # Build matplotlib figure
        fig = Figure(figsize=(6, 5), dpi=100)
        canvas = FigureCanvasQTAgg(fig)
        ax = fig.add_subplot(111)

        ax.scatter(
            saturation,
            hue * 360,
            c=dot_colours,
            s=30,
            alpha=0.6,
            edgecolors="none",
        )
        ax.set_xlabel("Saturation")
        ax.set_ylabel("Hue (degrees)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 360)
        ax.set_title(f"Saturation vs. Hue ({n:,} pixels)")
        fig.tight_layout()

        layout = QVBoxLayout()
        layout.addWidget(canvas)
        self.setLayout(layout)
