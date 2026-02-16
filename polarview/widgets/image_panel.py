"""QLabel-based image display panel used for TOP, MIDDLE, BOTTOM, and COLOR."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


class ImagePanel(QWidget):
    """A titled image display widget.

    Call :meth:`set_image` with a ``(H, W, 3)`` float64 numpy array
    whose values are in ``[0, 255]``.

    The image is scaled to fit the available space while **preserving its
    original aspect ratio** (no stretching / distortion).
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title_label = QLabel(title)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setStyleSheet("font-weight: bold;")

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(100, 100)
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        # Do NOT use setScaledContents — it ignores aspect ratio.

        self._pixmap: QPixmap | None = None

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(self._title_label)
        layout.addWidget(self._image_label, stretch=1)
        self.setLayout(layout)

    def set_image(self, array: np.ndarray) -> None:
        """Update the displayed image from a numpy array.

        Parameters
        ----------
        array : ndarray, shape (H, W, 3), float64
            RGB pixel values in ``[0, 255]``.
        """
        if array is None:
            return

        img = np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg.copy())
        self._apply_scaled_pixmap()

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Re-scale the pixmap whenever the panel is resized."""
        super().resizeEvent(event)
        self._apply_scaled_pixmap()

    def _apply_scaled_pixmap(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self._image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)
