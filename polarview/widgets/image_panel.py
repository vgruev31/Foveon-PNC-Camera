"""QLabel-based image display panel used for TOP, MIDDLE, BOTTOM, and COLOR."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygon
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


class ImagePanel(QWidget):
    """A titled image display widget.

    Call :meth:`set_image` with a ``(H, W, 3)`` float64 numpy array
    whose values are in ``[0, 255]``.

    The image is scaled to fit the available space while **preserving its
    original aspect ratio** (no stretching / distortion).
    """

    # Emits list of (x, y) tuples in original image coordinates
    roi_selected = pyqtSignal(object)

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title_label = QLabel(title)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setStyleSheet("font-weight: bold;")

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(100, 100)
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )

        self._pixmap: QPixmap | None = None

        # ROI polygon selection state
        self._roi_mode = False
        self._roi_vertices: list[QPoint] = []  # in label coordinates
        self._image_label.installEventFilter(self)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(self._title_label)
        layout.addWidget(self._image_label, stretch=1)
        self.setLayout(layout)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def set_title(self, title: str) -> None:
        """Change the displayed title text."""
        self._title_label.setText(title)

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

    def set_roi_mode(self, enabled: bool) -> None:
        """Enable or disable polygon ROI selection on this panel."""
        self._roi_mode = enabled
        self._roi_vertices.clear()
        if enabled:
            self._image_label.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self._image_label.setCursor(Qt.CursorShape.ArrowCursor)
            self._apply_scaled_pixmap()  # clear any overlay

    # -----------------------------------------------------------------
    # Qt overrides
    # -----------------------------------------------------------------
    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_scaled_pixmap()

    def eventFilter(self, obj, event):  # noqa: N802
        """Handle mouse events on ``_image_label`` for polygon ROI drawing."""
        if obj is not self._image_label or not self._roi_mode:
            return super().eventFilter(obj, event)

        if event.type() == event.Type.MouseButtonDblClick:
            # Double-click: close polygon and emit
            if len(self._roi_vertices) >= 3:
                img_verts = self._label_points_to_image_points(self._roi_vertices)
                if img_verts:
                    self.roi_selected.emit(img_verts)
            self.set_roi_mode(False)
            return True

        if event.type() == event.Type.MouseButtonPress:
            btn = event.button()
            if btn == Qt.MouseButton.LeftButton:
                # Add vertex
                self._roi_vertices.append(event.position().toPoint())
                self._draw_polygon_overlay()
                return True
            if btn == Qt.MouseButton.RightButton:
                # Right-click: close polygon and emit
                if len(self._roi_vertices) >= 3:
                    img_verts = self._label_points_to_image_points(
                        self._roi_vertices,
                    )
                    if img_verts:
                        self.roi_selected.emit(img_verts)
                self.set_roi_mode(False)
                return True

        return super().eventFilter(obj, event)

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------
    def _apply_scaled_pixmap(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self._image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)

    def _draw_polygon_overlay(self) -> None:
        """Redraw the image with the current polygon vertices overlaid."""
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self._image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        overlay = QPixmap(scaled)

        # Account for centering offset
        disp_w = scaled.width()
        disp_h = scaled.height()
        label_w = self._image_label.width()
        label_h = self._image_label.height()
        offset_x = (label_w - disp_w) / 2.0
        offset_y = (label_h - disp_h) / 2.0

        painter = QPainter(overlay)
        pen = QPen(QColor(0, 255, 0), 2)
        painter.setPen(pen)

        # Draw edges between vertices
        pts = self._roi_vertices
        for i in range(len(pts) - 1):
            p1 = QPoint(int(pts[i].x() - offset_x), int(pts[i].y() - offset_y))
            p2 = QPoint(int(pts[i + 1].x() - offset_x), int(pts[i + 1].y() - offset_y))
            painter.drawLine(p1, p2)

        # Draw small circles at each vertex
        painter.setBrush(QColor(0, 255, 0))
        for pt in pts:
            painter.drawEllipse(
                QPoint(int(pt.x() - offset_x), int(pt.y() - offset_y)), 3, 3,
            )

        painter.end()
        self._image_label.setPixmap(overlay)

    def _label_points_to_image_points(
        self, label_pts: list[QPoint],
    ) -> list[tuple[int, int]] | None:
        """Convert label-coordinate points to original image pixel coordinates."""
        if self._pixmap is None:
            return None
        displayed = self._image_label.pixmap()
        if displayed is None:
            return None

        orig_w = self._pixmap.width()
        orig_h = self._pixmap.height()
        disp_w = displayed.width()
        disp_h = displayed.height()
        label_w = self._image_label.width()
        label_h = self._image_label.height()

        offset_x = (label_w - disp_w) / 2.0
        offset_y = (label_h - disp_h) / 2.0

        scale_x = orig_w / disp_w if disp_w else 1.0
        scale_y = orig_h / disp_h if disp_h else 1.0

        result: list[tuple[int, int]] = []
        for pt in label_pts:
            ix = int((pt.x() - offset_x) * scale_x)
            iy = int((pt.y() - offset_y) * scale_y)
            ix = max(0, min(ix, orig_w - 1))
            iy = max(0, min(iy, orig_h - 1))
            result.append((ix, iy))
        return result
