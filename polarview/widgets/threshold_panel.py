"""Grouped Low / High threshold slider + spinner panel for one channel."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class ThresholdPanel(QWidget):
    """A compact panel with Low and High threshold sliders synced with spin-boxes.

    Emits :pyqt:`thresholds_changed(low, high)` whenever either value
    changes.  Values are in the range 0-100 (percentage).
    """

    thresholds_changed = pyqtSignal(float, float)

    def __init__(self, channel_name: str, high_default: float = 100.0, parent=None) -> None:
        super().__init__(parent)

        # --- Low ---
        self._low_slider = QSlider(Qt.Orientation.Horizontal)
        self._low_slider.setRange(0, 100)
        self._low_slider.setValue(0)
        self._low_slider.setFixedHeight(16)

        self._low_spinner = QDoubleSpinBox()
        self._low_spinner.setRange(0.0, 100.0)
        self._low_spinner.setValue(0.0)
        self._low_spinner.setSingleStep(1.0)
        self._low_spinner.setFixedWidth(50)
        self._low_spinner.setFixedHeight(20)

        # --- High ---
        self._high_slider = QSlider(Qt.Orientation.Horizontal)
        self._high_slider.setRange(0, 100)
        self._high_slider.setValue(int(high_default))
        self._high_slider.setFixedHeight(16)

        self._high_spinner = QDoubleSpinBox()
        self._high_spinner.setRange(0.0, 100.0)
        self._high_spinner.setValue(high_default)
        self._high_spinner.setSingleStep(1.0)
        self._high_spinner.setFixedWidth(50)
        self._high_spinner.setFixedHeight(20)

        # --- Layout ---
        # Leave ~20% gap between neighbouring panels
        self.setContentsMargins(0, 0, 0, 0)
        self.setMaximumWidth(260)

        layout = QVBoxLayout()
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(1)

        low_row = QHBoxLayout()
        low_row.setSpacing(2)
        low_row.addWidget(QLabel("Lo:"))
        low_row.addWidget(self._low_slider, stretch=1)
        low_row.addWidget(self._low_spinner)
        layout.addLayout(low_row)

        high_row = QHBoxLayout()
        high_row.setSpacing(2)
        high_row.addWidget(QLabel("Hi:"))
        high_row.addWidget(self._high_slider, stretch=1)
        high_row.addWidget(self._high_spinner)
        layout.addLayout(high_row)

        self.setLayout(layout)

        # --- Signal wiring (block to prevent feedback loops) ---
        self._low_slider.valueChanged.connect(self._on_low_slider)
        self._low_spinner.valueChanged.connect(self._on_low_spinner)
        self._high_slider.valueChanged.connect(self._on_high_slider)
        self._high_spinner.valueChanged.connect(self._on_high_spinner)

    # -- internal sync helpers ------------------------------------------

    def _on_low_slider(self, value: int) -> None:
        self._low_spinner.blockSignals(True)
        self._low_spinner.setValue(float(value))
        self._low_spinner.blockSignals(False)
        self.thresholds_changed.emit(float(value), self._high_spinner.value())

    def _on_low_spinner(self, value: float) -> None:
        self._low_slider.blockSignals(True)
        self._low_slider.setValue(int(value))
        self._low_slider.blockSignals(False)
        self.thresholds_changed.emit(value, self._high_spinner.value())

    def _on_high_slider(self, value: int) -> None:
        self._high_spinner.blockSignals(True)
        self._high_spinner.setValue(float(value))
        self._high_spinner.blockSignals(False)
        self.thresholds_changed.emit(self._low_spinner.value(), float(value))

    def _on_high_spinner(self, value: float) -> None:
        self._high_slider.blockSignals(True)
        self._high_slider.setValue(int(value))
        self._high_slider.blockSignals(False)
        self.thresholds_changed.emit(self._low_spinner.value(), value)

    # -- public properties -----------------------------------------------

    @property
    def low(self) -> float:
        return self._low_spinner.value()

    @property
    def high(self) -> float:
        return self._high_spinner.value()
