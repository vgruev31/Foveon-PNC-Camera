"""Frame selection slider with a display label and play/pause button."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QWidget,
)


class FrameSlider(QWidget):
    """Horizontal slider for selecting which frame to view.

    Emits :pyqt:`frame_changed(int)` with a **0-based** frame index.
    The label shown to the user is **1-based** for consistency with MATLAB.

    Emits :pyqt:`play_next_file()` when the last frame finishes and
    "Play Next File" mode is active.
    """

    frame_changed = pyqtSignal(int)
    play_next_file = pyqtSignal()

    _PLAY_ICON = "\u25B6"   # ▶
    _PAUSE_ICON = "\u23F8"  # ⏸

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._num_frames = 1
        self._playing = False

        # Play / Pause button (large)
        self._play_btn = QPushButton(self._PLAY_ICON)
        self._play_btn.setFixedSize(64, 64)
        btn_font = QFont()
        btn_font.setPointSize(24)
        self._play_btn.setFont(btn_font)
        self._play_btn.setToolTip("Play / Pause")

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setValue(0)

        self._label = QLabel("Frame: 1 / 1")
        self._label.setFixedWidth(120)

        self._slider.setMaximumWidth(300)

        # Mutually exclusive playback mode options
        self._loop_rb = QRadioButton("Loop Video")
        self._next_file_rb = QRadioButton("Play Next File")
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(False)  # allow none selected
        self._loop_rb.setChecked(False)
        self._next_file_rb.setChecked(False)
        # Make them mutually exclusive manually
        self._loop_rb.toggled.connect(self._on_loop_toggled)
        self._next_file_rb.toggled.connect(self._on_next_file_toggled)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._play_btn)
        layout.addWidget(QLabel("Frame:"))
        layout.addWidget(self._slider)
        layout.addWidget(self._label)
        layout.addWidget(self._loop_rb)
        layout.addWidget(self._next_file_rb)
        layout.addStretch()
        self.setLayout(layout)

        # Timer for playback (50 ms = 20 fps max)
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._advance_frame)

        self._slider.valueChanged.connect(self._on_value_changed)
        self._play_btn.clicked.connect(self._toggle_play)

    def _on_loop_toggled(self, checked: bool) -> None:
        if checked and self._next_file_rb.isChecked():
            self._next_file_rb.blockSignals(True)
            self._next_file_rb.setChecked(False)
            self._next_file_rb.blockSignals(False)

    def _on_next_file_toggled(self, checked: bool) -> None:
        if checked and self._loop_rb.isChecked():
            self._loop_rb.blockSignals(True)
            self._loop_rb.setChecked(False)
            self._loop_rb.blockSignals(False)

    def set_range(self, num_frames: int) -> None:
        """Set the total number of frames and reset to frame 0."""
        self._num_frames = max(1, num_frames)
        self._slider.setMaximum(self._num_frames - 1)
        self._slider.setValue(0)
        self._update_label()

    def continue_playing(self) -> None:
        """Resume playback after a new file has been loaded (for Play Next File)."""
        if not self._playing:
            self._playing = True
            self._play_btn.setText(self._PAUSE_ICON)
            self._timer.start()

    @property
    def value(self) -> int:
        """Current frame index (0-based)."""
        return self._slider.value()

    def stop(self) -> None:
        """Stop playback if running."""
        if self._playing:
            self._playing = False
            self._timer.stop()
            self._play_btn.setText(self._PLAY_ICON)

    def _toggle_play(self) -> None:
        if self._playing:
            self.stop()
        else:
            # If at the last frame, wrap back to the beginning
            if self._slider.value() >= self._num_frames - 1:
                self._slider.setValue(0)
            self._playing = True
            self._play_btn.setText(self._PAUSE_ICON)
            self._timer.start()

    def _advance_frame(self) -> None:
        current = self._slider.value()
        if current >= self._num_frames - 1:
            if self._loop_rb.isChecked():
                self._slider.setValue(0)
            elif self._next_file_rb.isChecked():
                # Signal main window to load next file and keep playing
                self.play_next_file.emit()
            else:
                self.stop()
            return
        self._slider.setValue(current + 1)

    def _on_value_changed(self, value: int) -> None:
        self._update_label()
        self.frame_changed.emit(value)

    def _update_label(self) -> None:
        self._label.setText(
            f"{self._slider.value() + 1} / {self._num_frames}"
        )
