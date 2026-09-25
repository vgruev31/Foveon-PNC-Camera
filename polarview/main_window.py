"""Main application window – replaces PolarView.mlapp GUI + callbacks."""

from __future__ import annotations

import datetime
import json
import os
import shutil
from itertools import permutations
from pathlib import Path

import h5py
import numpy as np
import openpyxl
from PIL import Image
from scipy.ndimage import label as ndimage_label, median_filter
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .frame_processor import (
    ChannelThresholds,
    GsenseProcessingParams,
    ProcessingParams,
    compute_hdr_displays,
    process_gsense_frame,
    process_single_frame,
)
from .h5_loader import load_h5
from .image_saver import save_images, save_gsense_images, save_displayed_images
from .video_data import CameraType, H5File, H5Info, VideoData
from .widgets.frame_slider import FrameSlider
from .widgets.image_panel import ImagePanel
from .widgets.roc_dialog import ROCDialog
from .widgets.threshold_panel import ThresholdPanel


# GSense demosaic configuration constants
_GSENSE_OFFSETS = [(0, 0), (0, 1), (1, 0), (1, 1)]
_GSENSE_OFFSET_LABELS = ["(0, 0)", "Skip 1 col", "Skip 1 row", "Skip 1 row + col"]
# Candidate RGBN filter patterns for (0,0)·(0,1)·(1,0)·(1,1).
# Top 7 are the most likely; remaining follow for completeness.
_PREFERRED_PERMS = [
    ('B', 'R', 'G', 'N'),
    ('N', 'R', 'G', 'B'),
    ('B', 'R', 'N', 'G'),
    ('N', 'R', 'B', 'G'),
    ('G', 'R', 'B', 'N'),
    ('G', 'R', 'N', 'B'),
    ('G', 'N', 'R', 'B'),
]
_ALL_PERMS = list(permutations(('R', 'G', 'B', 'N')))
_GSENSE_PERMS = _PREFERRED_PERMS + [p for p in _ALL_PERMS if p not in _PREFERRED_PERMS]
_GSENSE_PERM_DEFAULT_IDX = 0


def _extract_tissue_type(filename: str) -> str:
    """Extract tissue type (LN or TUMOR) from filename.

    Looks for '_LN_' or '_TUMOR_' in the filename. Returns 'LN', 'TUMOR',
    or 'UNKNOWN' if neither is found.
    """
    upper = filename.upper()
    if "_TUMOR_" in upper or "_TUMOR." in upper:
        return "TUMOR"
    if "_LN_" in upper or "_LN." in upper:
        return "LN"
    return "UNKNOWN"


class HDRViewerDialog(QDialog):
    """Floating non-modal window that shows a single HDR image."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.resize(640, 640)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self._panel = ImagePanel(title)
        layout.addWidget(self._panel)

    def set_image(self, img: np.ndarray) -> None:
        self._panel.set_image(img)


class PolarViewMainWindow(QMainWindow):
    """Port of the MATLAB PolarView App Designer GUI."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GSense UV-Color-NIR")
        icon_path = Path(__file__).parent / "assets" / "mantis_shrimp.svg"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.setMinimumSize(1200, 800)

        # -- data model (no globals) --
        self._h5info: H5Info | None = None
        self._h5file: H5File = H5File()
        self._video_data: VideoData = VideoData()
        self._file_loaded: bool = False
        self._camera_type: CameraType = CameraType.UNKNOWN
        self._displayed_images: dict[str, np.ndarray] = {}  # label → [0,255] arrays

        # -- ROI save mode --
        self._roi_save_mode: bool = False

        # -- H5 file navigation --
        self._h5_dir: Path | None = None  # directory of loaded file
        self._h5_files: list[Path] = []
        self._h5_file_index: int = -1

        # -- HDR viewer dialogs (lazily created on first Show HDR click) --
        self._hdr_color_dialog: HDRViewerDialog | None = None
        self._hdr_nir_dialog: HDRViewerDialog | None = None

        self._build_ui()
        self._connect_signals()

    # -----------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # --- Row 1: File I/O ---
        file_row = QHBoxLayout()
        self._open_btn = QPushButton("Open Input File")
        self._filename_edit = QLineEdit()
        self._filename_edit.setReadOnly(True)
        self._filename_edit.setPlaceholderText("No file loaded")
        file_row.addWidget(self._open_btn)
        file_row.addWidget(self._filename_edit, stretch=1)
        main_layout.addLayout(file_row)

        # --- Scrub time info from all H5 files in a directory ---
        scrub_row = QHBoxLayout()
        self._scrub_btn = QPushButton("Scrub Files")
        scrub_row.addWidget(self._scrub_btn)
        self._scrub_dir_edit = QLineEdit()
        self._scrub_dir_edit.setReadOnly(True)
        self._scrub_dir_edit.setPlaceholderText("Select directory...")
        scrub_row.addWidget(self._scrub_dir_edit, stretch=1)
        self._scrub_browse_btn = QPushButton("Browse...")
        scrub_row.addWidget(self._scrub_browse_btn)
        scrub_row.addStretch()
        main_layout.addLayout(scrub_row)

        # --- Row 1b: Nav / Delete (left) + Median Filter / Save (right) ---
        nav_row = QHBoxLayout()
        self._prev_btn = QPushButton("← Previous File")
        self._next_btn = QPushButton("Next File →")
        self._uv_only_cb = QCheckBox("UV Files Only")
        self._uv_roi_cb = QCheckBox("Display UV ROI")
        self._mask_pixels_cb = QCheckBox("Mask UV Pixels")
        self._min_pixels_spin = QSpinBox()
        self._min_pixels_spin.setRange(1, 1000)
        self._min_pixels_spin.setValue(1)
        self._min_pixels_spin.setFixedWidth(60)
        self._min_pixels_spin.setToolTip(
            "Minimum contiguous white pixels to classify as positive"
        )
        self._delete_btn = QPushButton("Delete H5 File")
        self._prev_btn.setEnabled(False)
        self._next_btn.setEnabled(False)
        self._delete_btn.setEnabled(False)
        nav_row.addWidget(self._prev_btn)
        nav_row.addWidget(self._next_btn)
        nav_row.addWidget(self._uv_only_cb)
        nav_row.addWidget(self._uv_roi_cb)
        nav_row.addWidget(self._mask_pixels_cb)
        nav_row.addWidget(self._min_pixels_spin)
        nav_row.addWidget(QLabel("Min Pixels"))
        nav_row.addWidget(self._delete_btn)
        nav_row.addStretch()
        nav_row.addWidget(QLabel("Median Filter:"))
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["None", "3", "5", "7", "9", "11"])
        nav_row.addWidget(self._filter_combo)
        nav_row.addWidget(QLabel("Spike Filter:"))
        self._spike_combo = QComboBox()
        self._spike_combo.addItems(["None", "3", "5", "7", "9", "11"])
        nav_row.addWidget(self._spike_combo)
        main_layout.addLayout(nav_row)

        # --- Row 1c: Rename H5 file ---
        rename_row = QHBoxLayout()
        rename_row.addWidget(QLabel("Rename:  Subject_"))
        self._rename_spin = QSpinBox()
        self._rename_spin.setRange(0, 9999)
        self._rename_spin.setValue(1)
        self._rename_spin.setFixedWidth(70)
        rename_row.addWidget(self._rename_spin)
        rename_row.addWidget(QLabel("_Sample_"))
        self._sample_spin = QSpinBox()
        self._sample_spin.setRange(0, 9999)
        self._sample_spin.setValue(1)
        self._sample_spin.setFixedWidth(70)
        rename_row.addWidget(self._sample_spin)
        rename_row.addWidget(QLabel("_"))
        self._rename_tissue = QComboBox()
        self._rename_tissue.addItems(["LN", "TUMOR"])
        rename_row.addWidget(self._rename_tissue)
        rename_row.addWidget(QLabel("_"))
        self._rename_descriptor = QComboBox()
        self._rename_descriptor.addItems(["COLOR", "COLOR_NIR", "NIR", "UV"])
        rename_row.addWidget(self._rename_descriptor)
        self._rename_btn = QPushButton("Rename H5 File")
        self._rename_btn.setEnabled(False)
        rename_row.addWidget(self._rename_btn)
        rename_row.addStretch()
        main_layout.addLayout(rename_row)

        # --- Row 1d: Save buttons (left) + Show All HSV (right) ---
        save_row = QHBoxLayout()
        self._save_btn = QPushButton("Save Images")
        save_row.addWidget(self._save_btn)
        self._save_all_btn = QPushButton("Save ALL Images")
        self._save_all_btn.setEnabled(False)
        save_row.addWidget(self._save_all_btn)
        self._save_uv_btn = QPushButton("Save UV Images")
        self._save_uv_btn.setEnabled(False)
        save_row.addWidget(self._save_uv_btn)
        self._show_hdr_btn = QPushButton("Show HDR Views")
        self._show_hdr_btn.setEnabled(False)
        self._show_hdr_btn.setToolTip(
            "Open two windows showing HG+LG HDR-fused Color and NIR images."
        )
        save_row.addWidget(self._show_hdr_btn)
        save_row.addStretch()
        # --- Right-aligned video controls ---
        save_row.addWidget(QLabel("FPS:"))
        self._video_fps_combo = QComboBox()
        self._video_fps_combo.addItems(["10", "15", "24", "25", "30", "60"])
        self._video_fps_combo.setCurrentText("25")
        self._video_fps_combo.setToolTip("Frame rate for the saved video.")
        save_row.addWidget(self._video_fps_combo)
        self._save_video_btn = QPushButton("Save Video")
        self._save_video_btn.setEnabled(False)
        save_row.addWidget(self._save_video_btn)
        self._save_video_composite_cb = QCheckBox("Saving Video Composite")
        self._save_video_composite_cb.setChecked(True)
        self._save_video_composite_cb.setToolTip(
            "When checked, the NIR-on-Color composite is included as a "
            "third panel in the saved video; otherwise only HG Color and "
            "HG NIR/UV are saved."
        )
        save_row.addWidget(self._save_video_composite_cb)
        main_layout.addLayout(save_row)

        # --- Row 1e: ROI + ROC buttons ---
        roc_row = QHBoxLayout()
        self._create_roi_btn = QPushButton("Create ROI")
        self._create_roi_btn.setEnabled(False)
        roc_row.addWidget(self._create_roi_btn)
        self._clear_roi_btn = QPushButton("Clear ROI")
        self._clear_roi_btn.setEnabled(False)
        roc_row.addWidget(self._clear_roi_btn)
        self._roi_status_label = QLabel("")
        self._roi_status_label.setStyleSheet("color: green; font-weight: bold;")
        roc_row.addWidget(self._roi_status_label)
        self._roc_btn = QPushButton("Compute ROC")
        self._roc_btn.setEnabled(False)
        roc_row.addWidget(self._roc_btn)
        self._roc_mode_combo = QComboBox()
        self._roc_mode_combo.addItems([
            "Empirical", "Linear", "Smooth (Spline)", "Smooth (KDE)",
            "LOWESS", "Savitzky-Golay", "Bootstrap Average", "Bezier",
            "Leave-One-Out",
        ])
        self._roc_mode_combo.setCurrentIndex(1)  # default to Linear
        roc_row.addWidget(self._roc_mode_combo)
        roc_row.addStretch()
        main_layout.addLayout(roc_row)

        # --- Row 2: Frame slider ---
        self._frame_slider = FrameSlider()
        main_layout.addWidget(self._frame_slider)

        # --- Row 2b: GSense demosaic config (hidden by default) ---
        self._gsense_config_widget = QWidget()
        gsense_cfg = QHBoxLayout(self._gsense_config_widget)
        gsense_cfg.setContentsMargins(0, 2, 0, 2)
        gsense_cfg.addWidget(QLabel("Pixel Offset:"))
        self._offset_combo = QComboBox()
        self._offset_combo.addItems(_GSENSE_OFFSET_LABELS)
        gsense_cfg.addWidget(self._offset_combo)
        gsense_cfg.addSpacing(20)
        gsense_cfg.addWidget(QLabel("Color Pattern (00·01·10·11):"))
        self._perm_combo = QComboBox()
        for perm in _GSENSE_PERMS:
            self._perm_combo.addItem("·".join(perm))
        self._perm_combo.setCurrentIndex(_GSENSE_PERM_DEFAULT_IDX)
        gsense_cfg.addWidget(self._perm_combo)
        self._perm_prev_btn = QPushButton("<")
        self._perm_prev_btn.setFixedWidth(28)
        self._perm_next_btn = QPushButton(">")
        self._perm_next_btn.setFixedWidth(28)
        gsense_cfg.addWidget(self._perm_prev_btn)
        gsense_cfg.addWidget(self._perm_next_btn)
        gsense_cfg.addStretch()
        self._gsense_config_widget.hide()
        main_layout.addWidget(self._gsense_config_widget)

        # --- Row 3: single-row image strip (LEFT → RIGHT: TOP, MIDDLE, BOTTOM, COLOR) ---
        image_row = QHBoxLayout()
        image_row.setSpacing(4)
        image_row.setContentsMargins(0, 0, 0, 0)
        self._top_panel = ImagePanel("TOP (UV)")
        self._middle_panel = ImagePanel("MIDDLE")
        self._bottom_panel = ImagePanel("BOTTOM (NIR)")
        self._color_panel = ImagePanel("COLOR")
        self._overlay_panel = ImagePanel("HG Color+NIR")

        image_row.addWidget(self._top_panel)
        image_row.addWidget(self._middle_panel)
        image_row.addWidget(self._bottom_panel)
        image_row.addWidget(self._color_panel)
        image_row.addWidget(self._overlay_panel)
        main_layout.addLayout(image_row, stretch=1)

        # --- Tissue status label (between images and thresholds) ---
        self._tissue_label = QLabel("")
        self._tissue_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._tissue_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        main_layout.addWidget(self._tissue_label)

        # --- Row 3b: Jet colormap checkboxes for NIR panels (GSense only, hidden by default) ---
        self._jet_row_widget = QWidget()
        jet_row = QHBoxLayout(self._jet_row_widget)
        jet_row.setContentsMargins(0, 2, 0, 2)
        jet_row.addWidget(QLabel(""), stretch=1)  # spacer under HG Color
        self._hg_nir_jet_cb = QCheckBox("Jet Colormap")
        jet_row.addWidget(self._hg_nir_jet_cb, stretch=1)  # under HG NIR
        jet_row.addWidget(QLabel(""), stretch=1)  # spacer under LG Color
        self._lg_nir_jet_cb = QCheckBox("Jet Colormap")
        jet_row.addWidget(self._lg_nir_jet_cb, stretch=1)  # under LG NIR

        # Transparency slider for the 5th panel (HG Color + NIR overlay).
        # Slider value is "NIR transparency %" (0 = solid NIR, 100 = invisible).
        overlay_alpha_widget = QWidget()
        overlay_alpha_row = QHBoxLayout(overlay_alpha_widget)
        overlay_alpha_row.setContentsMargins(0, 0, 0, 0)
        overlay_alpha_row.setSpacing(4)
        overlay_alpha_row.addWidget(QLabel("NIR Transp:"))
        self._overlay_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._overlay_alpha_slider.setRange(0, 100)
        self._overlay_alpha_slider.setValue(35)  # default 35% transp → α=0.65
        self._overlay_alpha_slider.setFixedHeight(16)
        overlay_alpha_row.addWidget(self._overlay_alpha_slider, stretch=1)
        self._overlay_alpha_label = QLabel("35%")
        self._overlay_alpha_label.setFixedWidth(36)
        overlay_alpha_row.addWidget(self._overlay_alpha_label)
        jet_row.addWidget(overlay_alpha_widget, stretch=1)  # under HG Color+NIR

        self._jet_row_widget.hide()
        main_layout.addWidget(self._jet_row_widget)

        # --- Row 4: Threshold sliders (one per image, ~80% width each) ---
        controls_row = QHBoxLayout()
        controls_row.setContentsMargins(0, 0, 0, 0)

        self._top_thresh = ThresholdPanel("TOP", high_default=50.0)
        self._middle_thresh = ThresholdPanel("MIDDLE", high_default=50.0)
        self._bottom_thresh = ThresholdPanel("BOTTOM", high_default=60.0)
        self._color_thresh = ThresholdPanel("COLOR", high_default=100.0)
        for w in (self._top_thresh, self._middle_thresh,
                  self._bottom_thresh, self._color_thresh):
            controls_row.addWidget(w, stretch=1)
        # Spacer column under the 5th image panel (overlay) to keep
        # threshold-row column widths aligned with the image row.
        self._controls_spacer = QWidget()
        controls_row.addWidget(self._controls_spacer, stretch=1)

        main_layout.addLayout(controls_row)

        # Status bar
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

    # -----------------------------------------------------------------
    # Signal wiring
    # -----------------------------------------------------------------
    def _connect_signals(self) -> None:
        self._open_btn.clicked.connect(self._on_open_file)
        self._prev_btn.clicked.connect(self._on_prev_file)
        self._next_btn.clicked.connect(self._on_next_file)
        self._uv_only_cb.toggled.connect(self._on_uv_filter_changed)
        self._uv_roi_cb.toggled.connect(self._on_filter_changed)
        self._mask_pixels_cb.toggled.connect(self._on_filter_changed)
        self._min_pixels_spin.valueChanged.connect(self._on_filter_changed)
        self._delete_btn.clicked.connect(self._on_delete_file)
        self._rename_btn.clicked.connect(self._on_rename_file)
        self._create_roi_btn.clicked.connect(self._on_create_roi)
        self._clear_roi_btn.clicked.connect(self._on_clear_roi)
        self._middle_panel.roi_selected.connect(self._on_roi_selected)
        self._save_all_btn.clicked.connect(self._on_save_all_images)
        self._save_uv_btn.clicked.connect(self._on_save_uv_images)
        self._save_video_btn.clicked.connect(self._on_save_video)
        self._show_hdr_btn.clicked.connect(self._on_show_hdr_views)
        self._roc_btn.clicked.connect(self._on_compute_roc)
        self._scrub_browse_btn.clicked.connect(self._on_scrub_browse)
        self._scrub_btn.clicked.connect(self._on_scrub_files)
        self._frame_slider.frame_changed.connect(self._on_frame_changed)
        self._frame_slider.play_next_file.connect(self._on_play_next_file)
        self._top_thresh.thresholds_changed.connect(self._on_threshold_changed)
        self._middle_thresh.thresholds_changed.connect(self._on_threshold_changed)
        self._bottom_thresh.thresholds_changed.connect(self._on_threshold_changed)
        self._color_thresh.thresholds_changed.connect(self._on_threshold_changed)
        self._filter_combo.currentIndexChanged.connect(self._on_median_changed)
        self._spike_combo.currentIndexChanged.connect(self._on_spike_changed)
        self._save_btn.clicked.connect(self._on_save)
        self._hg_nir_jet_cb.stateChanged.connect(self._on_jet_changed)
        self._lg_nir_jet_cb.stateChanged.connect(self._on_jet_changed)
        self._overlay_alpha_slider.valueChanged.connect(self._on_overlay_alpha_changed)
        self._offset_combo.currentIndexChanged.connect(self._on_gsense_config_changed)
        self._perm_combo.currentIndexChanged.connect(self._on_gsense_config_changed)
        self._perm_prev_btn.clicked.connect(self._on_perm_prev)
        self._perm_next_btn.clicked.connect(self._on_perm_next)

    # -----------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------
    def _on_open_file(self) -> None:
        start_dir = (
            str(Path(self._filename_edit.text()).parent)
            if self._filename_edit.text()
            else ""
        )
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select H5 File", start_dir, "HDF5 Files (*.h5);;All Files (*)"
        )
        if not file_path:
            return

        p = Path(file_path)
        self._h5_dir = p.parent
        # Clear UV-related filters when explicitly opening a file
        self._uv_only_cb.setChecked(False)
        self._uv_roi_cb.setChecked(False)
        self._mask_pixels_cb.setChecked(False)
        self._rebuild_file_list(current=p)

        if self._h5_file_index < 0 or not self._h5_files:
            self.statusBar().showMessage("No H5 files found in directory.")
            return

        self._load_file(self._h5_files[self._h5_file_index])

    def _rebuild_file_list(self, current: Path | None = None) -> None:
        """Rebuild ``_h5_files`` from ``_h5_dir``, respecting UV filter."""
        if self._h5_dir is None:
            return
        all_h5 = sorted(self._h5_dir.glob("*.h5"))
        if self._uv_only_cb.isChecked():
            all_h5 = [f for f in all_h5 if "UV" in f.stem]
        self._h5_files = all_h5
        if current is not None and current in self._h5_files:
            self._h5_file_index = self._h5_files.index(current)
        elif self._h5_files:
            self._h5_file_index = 0
        else:
            self._h5_file_index = -1
        self._update_nav_buttons()

    def _on_uv_filter_changed(self) -> None:
        """Re-filter the file list when the UV-only checkbox is toggled."""
        if self._h5_dir is None:
            return
        # Try to keep the current file selected
        current = (
            self._h5_files[self._h5_file_index]
            if 0 <= self._h5_file_index < len(self._h5_files)
            else None
        )
        self._rebuild_file_list(current=current)
        if not self._h5_files:
            self.statusBar().showMessage("No matching H5 files found.")
        elif current not in self._h5_files and self._h5_files:
            # Current file doesn't match filter — load the first matching file
            self._load_file(self._h5_files[self._h5_file_index])
        else:
            self._update_nav_buttons()

    def _on_prev_file(self) -> None:
        if self._h5_file_index <= 0:
            return
        self._h5_file_index -= 1
        self._load_file(self._h5_files[self._h5_file_index])

    def _on_next_file(self) -> None:
        if self._h5_file_index >= len(self._h5_files) - 1:
            return
        self._h5_file_index += 1
        self._load_file(self._h5_files[self._h5_file_index])

    def _on_play_next_file(self) -> None:
        """Called by the frame slider when 'Play Next File' mode reaches the last frame."""
        if self._h5_file_index >= len(self._h5_files) - 1:
            # No more files — stop playback
            self._frame_slider.stop()
            return
        self._h5_file_index += 1
        self._load_file(self._h5_files[self._h5_file_index])
        # Resume the timer so frames keep advancing
        self._frame_slider.continue_playing()

    def _on_delete_file(self) -> None:
        if not self._file_loaded or self._h5_file_index < 0:
            return

        target = self._h5_files[self._h5_file_index]
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Permanently delete\n{target.name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Delete the file from disk
        try:
            target.unlink()
        except Exception as exc:
            QMessageBox.critical(self, "Delete Failed", str(exc))
            return

        # Delete ROI sidecar if it exists
        roi_sidecar = target.with_suffix(".roi.json")
        if roi_sidecar.exists():
            try:
                roi_sidecar.unlink()
            except Exception:
                pass  # non-critical

        # Remove from the list
        del self._h5_files[self._h5_file_index]

        if len(self._h5_files) == 0:
            # No files left — reset to empty state
            self._h5_file_index = -1
            self._file_loaded = False
            self._h5info = None
            self._filename_edit.clear()
            self._update_nav_buttons()
            self.statusBar().showMessage("File deleted. No more H5 files in directory.")
            return

        # Prefer next file; if we were at the end, go to previous
        if self._h5_file_index >= len(self._h5_files):
            self._h5_file_index = len(self._h5_files) - 1

        self._load_file(self._h5_files[self._h5_file_index])
        self.statusBar().showMessage(
            f"Deleted {target.name}. Now showing: "
            f"{self._h5_files[self._h5_file_index].name}"
        )

    def _on_rename_file(self) -> None:
        if not self._file_loaded or self._h5_file_index < 0:
            return

        old_path = self._h5_files[self._h5_file_index]
        num = self._rename_spin.value()
        sample = self._sample_spin.value()
        descriptor = self._rename_descriptor.currentText()
        tissue = self._rename_tissue.currentText()
        new_name = f"Subject_{num}_Sample_{sample}_{tissue}_{descriptor}.h5"
        # Copy with new name into "renamed files" subfolder in the parent directory
        dest_dir = old_path.parent.parent / "renamed files"
        dest_dir.mkdir(exist_ok=True)
        new_path = dest_dir / new_name

        if new_path.exists():
            QMessageBox.warning(
                self, "Rename Failed",
                f"A file named '{new_name}' already exists in:\n{dest_dir}",
            )
            return

        try:
            shutil.copy2(old_path, new_path)
        except Exception as exc:
            QMessageBox.critical(self, "Copy Failed", str(exc))
            return

        self.statusBar().showMessage(
            f"Copied: {old_path.name} → {dest_dir / new_name}"
        )

    @staticmethod
    def _scrub_time_info(file_path: Path) -> None:
        """Remove time-related and identifying metadata from an H5 file.

        - Deletes ``/camera/timestamp`` dataset (per-frame UTC timestamps)
        - Deletes root attributes: ``time-info``, ``network-info``,
          ``os-info``, ``hardware-info``, ``python-info``
        - Deletes ``/camera`` attribute ``fw-build-time``
        - Sets the file's modification time to January 1, 2000
        """
        with h5py.File(file_path, "a") as f:
            # Remove /camera/timestamp dataset
            if "camera" in f and "timestamp" in f["camera"]:
                del f["camera"]["timestamp"]
            # Remove time/identifying root attributes
            root = f["/"]
            for attr in ("time-info", "network-info", "os-info",
                         "hardware-info", "python-info"):
                if attr in root.attrs:
                    del root.attrs[attr]
            # Remove fw-build-time from /camera group
            if "camera" in f and "fw-build-time" in f["camera"].attrs:
                del f["camera"].attrs["fw-build-time"]

        # Set file system timestamp to Jan 1, 2000 00:00:00
        epoch_2000 = datetime.datetime(2000, 1, 1).timestamp()
        os.utime(file_path, (epoch_2000, epoch_2000))

    def _on_scrub_browse(self) -> None:
        start_dir = self._scrub_dir_edit.text() or ""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory to Scrub", start_dir,
        )
        if directory:
            self._scrub_dir_edit.setText(directory)

    def _on_scrub_files(self) -> None:
        directory = self._scrub_dir_edit.text()
        if not directory:
            QMessageBox.warning(
                self, "No Directory",
                "Please select a directory first using Browse.",
            )
            return

        dir_path = Path(directory)
        if not dir_path.is_dir():
            QMessageBox.warning(
                self, "Invalid Directory",
                f"Directory not found:\n{directory}",
            )
            return

        h5_files = sorted(dir_path.rglob("*.h5"))
        if not h5_files:
            QMessageBox.information(
                self, "No Files",
                f"No .h5 files found in:\n{directory}\n(including subfolders)",
            )
            return

        self.statusBar().showMessage(f"Scrubbing {len(h5_files)} H5 files (including subfolders)...")
        QApplication.processEvents()

        scrubbed = 0
        errors = []
        for fp in h5_files:
            try:
                self._scrub_time_info(fp)
                scrubbed += 1
            except Exception as exc:
                errors.append(f"{fp.name}: {exc}")

        # Also reset timestamps for the root directory and all subdirectories
        try:
            epoch_2000 = datetime.datetime(2000, 1, 1).timestamp()
            os.utime(dir_path, (epoch_2000, epoch_2000))
            for sub in dir_path.rglob("*"):
                if sub.is_dir():
                    os.utime(sub, (epoch_2000, epoch_2000))
        except Exception:
            pass

        msg = f"Scrubbed {scrubbed} of {len(h5_files)} H5 files (including subfolders)."
        if errors:
            msg += f"\n\n{len(errors)} error(s):\n" + "\n".join(errors)
        QMessageBox.information(self, "Scrub Complete", msg)
        self.statusBar().showMessage(f"Scrub complete: {scrubbed} files processed")

    def _update_nav_buttons(self) -> None:
        has_files = len(self._h5_files) > 0
        self._prev_btn.setEnabled(has_files and self._h5_file_index > 0)
        self._next_btn.setEnabled(
            has_files and self._h5_file_index < len(self._h5_files) - 1
        )
        self._delete_btn.setEnabled(self._file_loaded)
        self._rename_btn.setEnabled(self._file_loaded)
        is_foveon = self._camera_type != CameraType.GSENSE
        self._save_all_btn.setEnabled(self._file_loaded)
        self._save_uv_btn.setEnabled(self._file_loaded)
        self._save_video_btn.setEnabled(
            self._file_loaded and self._camera_type == CameraType.GSENSE
        )
        self._show_hdr_btn.setEnabled(
            self._file_loaded and self._camera_type == CameraType.GSENSE
        )
        self._roc_btn.setEnabled(self._file_loaded)
        self._create_roi_btn.setEnabled(self._file_loaded)
        self._clear_roi_btn.setEnabled(self._file_loaded)

    def _load_file(self, p: Path) -> None:
        self.statusBar().showMessage(f"Loading {p}...")
        QApplication.processEvents()

        try:
            self._h5info = load_h5(p)
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", str(exc))
            self.statusBar().showMessage("Load failed")
            return

        try:
            self._h5file = H5File(name=p.stem, path=str(p.parent))
            self._filename_edit.setText(str(p))
            self._camera_type = self._h5info.attr.camera_type

            if self._camera_type == CameraType.GSENSE:
                self._video_data.allocate_gsense(
                    self._h5info.attr.rows, self._h5info.attr.columns
                )
                self._setup_gsense_ui()
            else:
                self._video_data.allocate(
                    self._h5info.attr.rows, self._h5info.attr.columns
                )
                self._setup_foveon_ui()

            self._frame_slider.set_range(self._h5info.attr.num_frames)

            self._file_loaded = True
            self._update_nav_buttons()
            camera_label = self._h5info.attr.camera
            cols_info = self._h5info.attr.columns
            if self._camera_type == CameraType.GSENSE:
                cols_info = self._h5info.attr.columns // 2
                camera_label = f"GSense HG+LG ({cols_info}×{self._h5info.attr.rows} each)"
            self.statusBar().showMessage(
                f"Loaded: {p.name}  "
                f"({self._h5info.attr.rows}×{cols_info}, "
                f"{self._h5info.attr.num_frames} frames, "
                f"{self._h5info.attr.frame_rate:.1f} fps, "
                f"{camera_label})"
            )

            self._process_and_display()
            self._update_roi_status()
            self._update_tissue_status(p)
        except Exception as exc:

            QMessageBox.critical(self, "Load Failed", f"{p.name}: {exc}")
            self.statusBar().showMessage("Load failed")

    def _on_frame_changed(self, _frame_index: int) -> None:
        self._process_and_display()

    def _on_threshold_changed(self, _low: float, _high: float) -> None:
        self._process_and_display()

    def _on_median_changed(self, index: int) -> None:
        """When a median filter is selected, reset spike to None."""
        if index > 0:  # not "None"
            self._spike_combo.blockSignals(True)
            self._spike_combo.setCurrentIndex(0)
            self._spike_combo.blockSignals(False)
        self._process_and_display()

    def _on_spike_changed(self, index: int) -> None:
        """When a spike filter is selected, reset median to None."""
        if index > 0:  # not "None"
            self._filter_combo.blockSignals(True)
            self._filter_combo.setCurrentIndex(0)
            self._filter_combo.blockSignals(False)
        self._process_and_display()

    def _on_filter_changed(self, _index: int) -> None:
        self._process_and_display()

    def _on_save(self) -> None:
        if not self._file_loaded:
            QMessageBox.warning(self, "No File", "No file loaded to save.")
            return

        self.statusBar().showMessage("Saving images...")
        QApplication.processEvents()

        try:
            if self._displayed_images:
                saved = save_displayed_images(self._displayed_images, self._h5file)
            elif self._camera_type == CameraType.GSENSE:
                saved = save_gsense_images(self._video_data, self._h5file)
            else:
                saved = save_images(self._video_data, self._h5file)
            QMessageBox.information(
                self,
                "Saved",
                f"Images saved ({len(saved)} files):\n"
                + "\n".join(str(p) for p in saved),
            )
            self.statusBar().showMessage("Images saved successfully")
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))
            self.statusBar().showMessage("Save failed")

    def _on_save_all_images(self) -> None:
        """Process every H5 file in the current directory and save all images."""
        if not self._file_loaded or not self._h5_files:
            QMessageBox.warning(self, "No File", "No file loaded.")
            return

        h5_dir = self._h5_files[0].parent
        all_h5 = sorted(h5_dir.glob("*.h5"))
        if not all_h5:
            return

        saved_total = 0
        errors = []
        for i, fp in enumerate(all_h5):
            self.statusBar().showMessage(
                f"Saving ALL images: {i + 1}/{len(all_h5)} — {fp.name}"
            )
            QApplication.processEvents()
            try:
                info = load_h5(fp)
                vd = VideoData()
                h5f = H5File(name=fp.stem, path=str(fp.parent))

                if info.attr.camera_type == CameraType.GSENSE:
                    vd.allocate_gsense(info.attr.rows, info.attr.columns)
                    gp = GsenseProcessingParams(
                        frame_number=0,
                        filter_tap=self._get_filter_tap(),
                        spike_tap=self._get_spike_tap(),
                        hg_nir_thresh=ChannelThresholds(
                            self._middle_thresh.low, self._middle_thresh.high),
                        lg_nir_thresh=ChannelThresholds(
                            self._color_thresh.low, self._color_thresh.high),
                        hg_nir_jet=self._hg_nir_jet_cb.isChecked(),
                        lg_nir_jet=self._lg_nir_jet_cb.isChecked(),
                        norm_bits=info.attr.norm_bits,
                        pixel_offset=_GSENSE_OFFSETS[self._offset_combo.currentIndex()],
                        color_perm=_GSENSE_PERMS[self._perm_combo.currentIndex()],
                    )
                    process_gsense_frame(info, vd, gp)
                    display_imgs = {
                        "HG Color": self._apply_color_threshold(
                            vd.hg_color_raw, self._top_thresh.low, self._top_thresh.high),
                        "HG NIR": vd.hg_nir,
                        "LG Color": self._apply_color_threshold(
                            vd.lg_color_raw, self._bottom_thresh.low, self._bottom_thresh.high),
                        "LG NIR": vd.lg_nir,
                    }
                    saved = save_displayed_images(display_imgs, h5f)
                else:
                    vd.allocate(info.attr.rows, info.attr.columns)
                    p = ProcessingParams(
                        frame_number=0,
                        filter_tap=self._get_filter_tap(),
                        spike_tap=self._get_spike_tap(),
                        top_thresh=ChannelThresholds(
                            self._top_thresh.low, self._top_thresh.high),
                        middle_thresh=ChannelThresholds(
                            self._middle_thresh.low, self._middle_thresh.high),
                        bottom_thresh=ChannelThresholds(
                            self._bottom_thresh.low, self._bottom_thresh.high),
                        is_uv="UV" in fp.stem,
                        norm_bits=info.attr.norm_bits,
                    )
                    process_single_frame(info, vd, p)
                    color_img = self._apply_color_threshold(
                        vd.color, self._color_thresh.low, self._color_thresh.high)
                    display_imgs = {
                        "TOP Image": vd.top,
                        "MIDDLE Image": vd.middle,
                        "BOTTOM Image": vd.bottom,
                        "COLOR Image": color_img,
                    }
                    saved = save_displayed_images(display_imgs, h5f)

                saved_total += len(saved)
            except Exception as exc:
                errors.append(f"{fp.name}: {exc}")

        msg = f"Saved {saved_total} images from {len(all_h5)} H5 files."
        if errors:
            msg += f"\n\n{len(errors)} error(s):\n" + "\n".join(errors)
        QMessageBox.information(self, "Save ALL Complete", msg)
        self.statusBar().showMessage(f"Save ALL complete: {saved_total} images")

    def _on_save_uv_images(self) -> None:
        """Process every UV H5 file in the current directory and save images."""
        if not self._file_loaded or not self._h5_files:
            QMessageBox.warning(self, "No File", "No file loaded.")
            return

        h5_dir = self._h5_files[0].parent
        all_h5 = sorted(f for f in h5_dir.glob("*.h5") if "UV" in f.stem)
        if not all_h5:
            QMessageBox.information(self, "No UV Files", "No H5 files with 'UV' in the name found.")
            return

        output_dir = h5_dir / "Processed UV Images"
        output_dir.mkdir(exist_ok=True)

        saved_count = 0
        errors = []
        for i, fp in enumerate(all_h5):
            self.statusBar().showMessage(
                f"Saving UV images: {i + 1}/{len(all_h5)} — {fp.name}"
            )
            QApplication.processEvents()
            try:
                info = load_h5(fp)
                vd = VideoData()
                h5f = H5File(name=fp.stem, path=str(output_dir))

                if info.attr.camera_type == CameraType.GSENSE:
                    vd.allocate_gsense(info.attr.rows, info.attr.columns)
                    gp = GsenseProcessingParams(
                        frame_number=0,
                        filter_tap=self._get_filter_tap(),
                        spike_tap=self._get_spike_tap(),
                        hg_nir_thresh=ChannelThresholds(
                            self._middle_thresh.low, self._middle_thresh.high),
                        lg_nir_thresh=ChannelThresholds(
                            self._color_thresh.low, self._color_thresh.high),
                        hg_nir_jet=self._hg_nir_jet_cb.isChecked(),
                        lg_nir_jet=self._lg_nir_jet_cb.isChecked(),
                        norm_bits=info.attr.norm_bits,
                        pixel_offset=_GSENSE_OFFSETS[self._offset_combo.currentIndex()],
                        color_perm=_GSENSE_PERMS[self._perm_combo.currentIndex()],
                    )
                    process_gsense_frame(info, vd, gp)
                    display_imgs = {
                        "HG Color": self._apply_color_threshold(
                            vd.hg_color_raw, self._top_thresh.low, self._top_thresh.high),
                        "HG NIR": vd.hg_nir,
                        "LG Color": self._apply_color_threshold(
                            vd.lg_color_raw, self._bottom_thresh.low, self._bottom_thresh.high),
                        "LG NIR": vd.lg_nir,
                    }
                    save_displayed_images(display_imgs, h5f)
                else:
                    vd.allocate(info.attr.rows, info.attr.columns)
                    p = ProcessingParams(
                        frame_number=0,
                        filter_tap=self._get_filter_tap(),
                        spike_tap=self._get_spike_tap(),
                        top_thresh=ChannelThresholds(
                            self._top_thresh.low, self._top_thresh.high),
                        middle_thresh=ChannelThresholds(
                            self._middle_thresh.low, self._middle_thresh.high),
                        bottom_thresh=ChannelThresholds(
                            self._bottom_thresh.low, self._bottom_thresh.high),
                        is_uv=True,
                        norm_bits=info.attr.norm_bits,
                    )
                    process_single_frame(info, vd, p)
                    color_img = self._apply_color_threshold(
                        vd.color, self._color_thresh.low, self._color_thresh.high)
                    display_imgs = {
                        "TOP Image": vd.top,
                        "MIDDLE Image": vd.middle,
                        "BOTTOM Image": vd.bottom,
                        "COLOR Image": color_img,
                    }
                    save_displayed_images(display_imgs, h5f)

                saved_count += 1
            except Exception as exc:
                errors.append(f"{fp.name}: {exc}")

        msg = f"Saved {saved_count} UV images to:\n{output_dir}"
        if errors:
            msg += f"\n\n{len(errors)} error(s):\n" + "\n".join(errors)
        QMessageBox.information(self, "Save UV Complete", msg)
        self.statusBar().showMessage(f"Save UV complete: {saved_count} images")

    def _on_save_video(self) -> None:
        """Write an MP4 of the per-frame HG panes for the current file.

        Always includes ``HG Color | HG NIR/UV``.  When the
        "Saving Video Composite" checkbox is checked, a third panel —
        the NIR-on-Color overlay (using the current transparency slider) —
        is appended.  Frame rate comes from the FPS selector.  The video
        is saved one directory above the current H5 file as
        ``<h5_stem>.mp4``.  GSense only.
        """
        if not self._file_loaded or self._h5_file_index < 0:
            QMessageBox.warning(self, "No File", "No file loaded.")
            return
        if self._camera_type != CameraType.GSENSE:
            QMessageBox.warning(
                self, "GSense Only",
                "Save Video supports GSense files only (HG Color + HG NIR/UV).",
            )
            return

        try:
            import imageio.v2 as iio
        except ImportError as exc:
            QMessageBox.critical(
                self, "Missing Dependency",
                f"imageio is required for Save Video.\n{exc}",
            )
            return

        h5_path = self._h5_files[self._h5_file_index]
        out_dir = h5_path.parent.parent
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{h5_path.stem}.mp4"

        n_frames = self._h5info.attr.num_frames
        offset = _GSENSE_OFFSETS[self._offset_combo.currentIndex()]
        perm = _GSENSE_PERMS[self._perm_combo.currentIndex()]
        include_overlay = self._save_video_composite_cb.isChecked()
        overlay_alpha = self._overlay_alpha()
        try:
            fps = int(self._video_fps_combo.currentText())
        except ValueError:
            fps = 25

        # Use a transient VideoData buffer so we don't disturb the on-screen
        # display while iterating frames.
        vd = VideoData()
        vd.allocate_gsense(self._h5info.attr.rows, self._h5info.attr.columns)

        writer = None
        try:
            for f_idx in range(n_frames):
                if f_idx % 5 == 0:
                    self.statusBar().showMessage(
                        f"Rendering video: frame {f_idx + 1}/{n_frames}"
                    )
                    QApplication.processEvents()

                gp = GsenseProcessingParams(
                    frame_number=f_idx,
                    filter_tap=self._get_filter_tap(),
                    spike_tap=self._get_spike_tap(),
                    hg_nir_thresh=ChannelThresholds(
                        self._middle_thresh.low, self._middle_thresh.high),
                    lg_nir_thresh=ChannelThresholds(
                        self._color_thresh.low, self._color_thresh.high),
                    hg_nir_jet=self._hg_nir_jet_cb.isChecked(),
                    lg_nir_jet=self._lg_nir_jet_cb.isChecked(),
                    norm_bits=self._h5info.attr.norm_bits,
                    pixel_offset=offset,
                    color_perm=perm,
                )
                process_gsense_frame(self._h5info, vd, gp)

                hg_color = self._apply_color_threshold(
                    vd.hg_color_raw, self._top_thresh.low, self._top_thresh.high)
                hg_nir = vd.hg_nir

                panes = [hg_color, hg_nir]
                if include_overlay:
                    overlay = self._build_nir_color_overlay(
                        hg_color, hg_nir, vd.hg_nir_raw,
                        self._middle_thresh.low, self._middle_thresh.high,
                        overlay_alpha,
                    )
                    panes.append(overlay)

                combined = np.concatenate(panes, axis=1)
                frame_u8 = np.clip(combined, 0.0, 255.0).astype(np.uint8)

                # H.264 (libx264) requires even width & height; pad if needed.
                h, w = frame_u8.shape[:2]
                if h % 2 or w % 2:
                    pad_h = h % 2
                    pad_w = w % 2
                    frame_u8 = np.pad(
                        frame_u8,
                        ((0, pad_h), (0, pad_w), (0, 0)),
                        mode="edge",
                    )

                if writer is None:
                    writer = iio.get_writer(
                        str(out_path), fps=fps, codec="libx264",
                        quality=8, macro_block_size=1,
                    )
                writer.append_data(frame_u8)
        except Exception as exc:
            if writer is not None:
                writer.close()
            QMessageBox.critical(self, "Save Video Failed", str(exc))
            self.statusBar().showMessage("Save Video failed")
            return
        finally:
            if writer is not None:
                writer.close()

        self.statusBar().showMessage(f"Video saved: {out_path}")
        n_panes = 3 if include_overlay else 2
        QMessageBox.information(
            self, "Video Saved",
            f"Saved {n_frames} frames ({n_panes} panes) at {fps} fps to:\n{out_path}",
        )

    @staticmethod
    def _classify_at_threshold(
        top_filtered: np.ndarray,
        roi_mask: np.ndarray | None,
        thresh_val: float,
        min_pixels: int,
    ) -> bool:
        """Return True if the image is classified positive at the given threshold.

        Parameters
        ----------
        top_filtered : filtered TOP channel, values in [0, 1].
        roi_mask : boolean mask (same shape), or None for whole image.
        thresh_val : threshold in [0, 1] (i.e. Lo / 100).
        min_pixels : minimum contiguous pixel count for positive.
        """
        if roi_mask is not None:
            above = roi_mask & (top_filtered >= thresh_val)
        else:
            above = top_filtered >= thresh_val

        labeled_arr, n_comp = ndimage_label(above)
        for comp_id in range(1, n_comp + 1):
            if int(np.sum(labeled_arr == comp_id)) >= min_pixels:
                return True
        return False

    def _extract_nir_channel(self, info: H5Info, frame: int = 0) -> np.ndarray:
        """Extract the HG NIR channel from an H5Info, applying current demosaic settings.

        Returns a 2-D float64 array normalised to [0, 1].
        """
        from polarview.frame_processor import _spike_filter

        frame = min(frame, info.attr.num_frames - 1)
        raw_frame = info.raw_data[:, :, 0, frame].astype(np.float64)
        norm = 2 ** info.attr.norm_bits
        cols_half = raw_frame.shape[1] // 2
        hg_full = raw_frame[:, :cols_half] / norm

        offset = _GSENSE_OFFSETS[self._offset_combo.currentIndex()]
        perm = _GSENSE_PERMS[self._perm_combo.currentIndex()]
        offset_y, offset_x = offset

        # Step 1: subsample by 2
        subsampled = hg_full[offset_y::2, offset_x::2]
        rows_s = (subsampled.shape[0] // 2) * 2
        cols_s = (subsampled.shape[1] // 2) * 2
        subsampled = subsampled[:rows_s, :cols_s]

        # Step 2: demosaic — extract NIR position
        positions = [
            subsampled[0::2, 0::2],
            subsampled[0::2, 1::2],
            subsampled[1::2, 0::2],
            subsampled[1::2, 1::2],
        ]
        ch = dict(zip(perm, positions))
        nir = ch['N']

        # Apply filters
        roc_spike_tap = self._get_spike_tap()
        roc_median_tap = self._get_filter_tap()
        if roc_spike_tap > 0:
            nir = _spike_filter(nir, roc_spike_tap)
        if roc_median_tap > 0:
            nir = median_filter(nir, size=(roc_median_tap, roc_median_tap),
                                mode="reflect")

        # Normalise to [0, 1]
        lo, hi = nir.min(), nir.max()
        if hi > lo:
            nir = (nir - lo) / (hi - lo)
        else:
            nir = np.zeros_like(nir)
        return nir

    def _on_compute_roc(self) -> None:
        """Compute and display an ROC curve using tissue_log.csv.

        For Foveon: operates on the TOP (UV) channel.
        For GSense: operates on the HG NIR channel.

        For each file, uses binary search to find the maximum Lo threshold
        (0-100) at which the image is classified positive (has a contiguous
        region >= min_pixels pixels above threshold within the ROI).  This
        per-file score is then used for fast score-based ROC computation.
        """
        if not self._file_loaded or not self._h5_files:
            QMessageBox.warning(self, "No File", "No file loaded.")
            return

        h5_dir = self._h5_files[0].parent
        xlsx_path = h5_dir / "tissue_log.xlsx"
        if not xlsx_path.exists():
            QMessageBox.warning(
                self, "No tissue_log.xlsx",
                f"tissue_log.xlsx not found in:\n{h5_dir}\n\n"
                "Place a tissue_log.xlsx (filename, cancerous, frame) in this folder.",
            )
            return

        # Read tissue_log.xlsx — skip TUMOR samples since the ROC focuses on LN
        entries: list[tuple[str, int, int]] = []  # (filename, cancerous, frame)
        n_tumor_skipped = 0
        wb = openpyxl.load_workbook(xlsx_path, read_only=True)
        ws = wb.active
        for row_idx in range(2, ws.max_row + 1):  # skip header
            fname_val = ws.cell(row=row_idx, column=1).value
            if fname_val is None:
                continue
            fname_str = str(fname_val).strip()
            if _extract_tissue_type(fname_str) == "TUMOR":
                n_tumor_skipped += 1
                continue
            cancerous_val = int(ws.cell(row=row_idx, column=2).value)
            frame_val = int(ws.cell(row=row_idx, column=3).value or 0)
            entries.append((fname_str, cancerous_val, frame_val))
        wb.close()

        if not entries:
            QMessageBox.warning(
                self, "Empty Log",
                "tissue_log.xlsx contains no LN entries (TUMOR samples are skipped).",
            )
            return

        # Phase 1: Load all files — extract channel + ROI mask
        from polarview.frame_processor import _spike_filter
        roc_median_tap = self._get_filter_tap()
        roc_spike_tap = self._get_spike_tap()
        filter_desc = "None"
        if roc_median_tap > 0:
            filter_desc = f"Median {roc_median_tap}"
        elif roc_spike_tap > 0:
            filter_desc = f"Spike {roc_spike_tap}"

        is_gsense = self._camera_type == CameraType.GSENSE

        file_data: list[tuple] = []  # (channel, roi_mask, label, fname, fp)
        errors: list[str] = []

        for i, (fname, label, frame) in enumerate(entries):
            self.statusBar().showMessage(
                f"Loading files ({filter_desc}): {i + 1}/{len(entries)} — {fname}"
            )
            QApplication.processEvents()

            fp = h5_dir / fname
            if not fp.exists():
                errors.append(f"{fname}: file not found")
                continue

            try:
                info = load_h5(fp)

                if info.attr.camera_type == CameraType.GSENSE:
                    ch = self._extract_nir_channel(info, frame)
                else:
                    # Foveon: TOP channel
                    frame_idx = min(frame, info.attr.num_frames - 1)
                    raw_frame = info.raw_data[:, :, :, frame_idx].astype(np.float64)
                    norm = 2 ** info.attr.norm_bits
                    ch = raw_frame[::2, ::2, 2] / norm

                    if roc_spike_tap > 0:
                        ch = _spike_filter(ch, roc_spike_tap)
                    if roc_median_tap > 0:
                        ch = median_filter(
                            ch, size=(roc_median_tap, roc_median_tap),
                            mode="reflect",
                        )

                roi_path = fp.with_suffix(".roi.json")
                roi_mask = self._load_roi_mask(roi_path, ch.shape)

                file_data.append((ch, roi_mask, label, fname, fp))
            except Exception as exc:
                errors.append(f"{fname}: {exc}")

        n_files = len(file_data)
        if n_files < 2:
            QMessageBox.warning(
                self, "Insufficient Data",
                f"Only {n_files} file(s) could be processed.\n"
                "Need at least 2 for an ROC curve.",
            )
            return

        if errors:
            QMessageBox.warning(
                self, "Some Files Skipped",
                f"{len(errors)} file(s) could not be processed:\n"
                + "\n".join(errors[:10]),
            )

        # Phase 2: Binary search for each file's critical threshold (score)
        min_pixels = self._min_pixels_spin.value()
        scores = np.zeros(n_files, dtype=float)

        for fi, (ch_filtered, roi_mask, label, fname, fp) in enumerate(file_data):
            self.statusBar().showMessage(
                f"Computing scores (min {min_pixels} px): {fi + 1}/{n_files} — {fname}"
            )
            QApplication.processEvents()

            if not self._classify_at_threshold(ch_filtered, roi_mask, 0.0, min_pixels):
                scores[fi] = -1.0
                continue

            if self._classify_at_threshold(ch_filtered, roi_mask, 1.0, min_pixels):
                scores[fi] = 100.0
                continue

            lo, hi = 0.0, 100.0
            for _ in range(15):
                mid = (lo + hi) / 2.0
                if self._classify_at_threshold(ch_filtered, roi_mask, mid / 100.0, min_pixels):
                    lo = mid
                else:
                    hi = mid
            scores[fi] = lo

        labels_arr = np.array([d[2] for d in file_data])
        filenames_list = [d[3] for d in file_data]
        tissue_list = [_extract_tissue_type(d[3]) for d in file_data]

        n_pos = int(labels_arr.sum())
        skipped_msg = f", {n_tumor_skipped} TUMOR skipped" if n_tumor_skipped else ""
        self.statusBar().showMessage(
            f"ROC computed from {n_files} LN files "
            f"({n_pos} positive, {n_files - n_pos} negative{skipped_msg})"
        )

        # Phase 3: Show ROC dialog and export results to Excel
        dlg = ROCDialog(self)
        roc_mode = self._roc_mode_combo.currentText()
        roc_xlsx_path = h5_dir / "roc_results.xlsx"
        optimal_thresh = dlg.plot_roc(
            labels_arr, scores, filenames_list, tissue_list,
            mode=roc_mode, excel_path=roc_xlsx_path,
        )
        dlg.show()
        if roc_xlsx_path.exists():
            self.statusBar().showMessage(
                f"ROC results saved to {roc_xlsx_path.name}"
            )

        if optimal_thresh is None:
            return

        # Phase 4: Generate thresholded B&W images at the optimal threshold
        output_dir = h5_dir / "ROC Thresholded Images"
        output_dir.mkdir(exist_ok=True)

        for old_png in output_dir.glob("*.png"):
            old_png.unlink()

        thresh_val = optimal_thresh / 100.0
        saved_count = 0

        for fi, (ch_filtered, roi_mask, label, fname, fp) in enumerate(file_data):
            self.statusBar().showMessage(
                f"Generating thresholded images: {fi + 1}/{n_files} — {fname}"
            )
            QApplication.processEvents()

            h_img, w_img = ch_filtered.shape
            rgb = np.zeros((h_img, w_img, 3), dtype=np.uint8)

            if roi_mask is not None:
                above = roi_mask & (ch_filtered >= thresh_val)
            else:
                above = ch_filtered >= thresh_val

            labeled_arr, n_comp = ndimage_label(above)
            keep = np.zeros_like(above)
            for comp_id in range(1, n_comp + 1):
                if int(np.sum(labeled_arr == comp_id)) >= min_pixels:
                    keep[labeled_arr == comp_id] = True
            rgb[keep] = [255, 255, 255]

            if roi_mask is not None:
                roi_path = fp.with_suffix(".roi.json")
                with open(roi_path) as fh:
                    roi_data = json.load(fh)
                verts = roi_data["vertices"]
                for j in range(len(verts)):
                    x0, y0 = int(round(verts[j][0])), int(round(verts[j][1]))
                    x1, y1 = int(round(verts[(j + 1) % len(verts)][0])), int(round(verts[(j + 1) % len(verts)][1]))
                    dx = abs(x1 - x0)
                    dy = abs(y1 - y0)
                    sx = 1 if x0 < x1 else -1
                    sy = 1 if y0 < y1 else -1
                    err = dx - dy
                    while True:
                        if 0 <= y0 < h_img and 0 <= x0 < w_img:
                            rgb[y0, x0] = [0, 255, 0]
                        if x0 == x1 and y0 == y1:
                            break
                        e2 = 2 * err
                        if e2 > -dy:
                            err -= dy
                            x0 += sx
                        if e2 < dx:
                            err += dx
                            y0 += sy

            classified = 1 if scores[fi] >= optimal_thresh else 0
            out_path = output_dir / f"{Path(fname).stem}_GT{label}_CL{classified}.png"
            Image.fromarray(rgb, mode="RGB").save(str(out_path))
            saved_count += 1

        del file_data

        channel_name = "HG NIR" if is_gsense else "TOP (UV)"
        self.statusBar().showMessage(
            f"ROC complete. Saved {saved_count} thresholded images to {output_dir.name}/"
        )
        QMessageBox.information(
            self, "ROC Complete",
            f"Optimal threshold: {optimal_thresh:.1f} / 100\n"
            f"Channel: {channel_name}, Filter: {filter_desc}\n\n"
            f"Set the NIR Lo slider to {optimal_thresh:.1f} and check 'Mask UV Pixels'\n"
            f"to see the same view on the current image.\n\n"
            f"Saved {saved_count} thresholded B&W images to:\n{output_dir}",
        )

    # -----------------------------------------------------------------
    # Camera-type UI switching
    # -----------------------------------------------------------------
    def _setup_foveon_ui(self) -> None:
        """Configure panel visibility for Foveon 3-channel mode."""
        self._top_panel.set_title("TOP (UV)")
        self._middle_panel.set_title("MIDDLE")
        self._bottom_panel.show()
        self._bottom_panel.set_title("BOTTOM (NIR)")
        self._color_panel.show()
        self._color_panel.set_title("COLOR")
        self._overlay_panel.hide()
        self._controls_spacer.hide()
        self._bottom_thresh.show()
        self._color_thresh.show()
        self._jet_row_widget.hide()
        self._gsense_config_widget.hide()
        # Restore threshold panel max width for 4-column layout
        for w in (self._top_thresh, self._middle_thresh,
                  self._bottom_thresh, self._color_thresh):
            w.setMaximumWidth(260)

    def _setup_gsense_ui(self) -> None:
        """Configure panel visibility for GSense 5-panel mode."""
        self._top_panel.set_title("HG Color")
        self._middle_panel.set_title("HG NIR/UV")
        self._bottom_panel.show()
        self._bottom_panel.set_title("LG Color")
        self._color_panel.show()
        self._color_panel.set_title("LG NIR/UV")
        self._overlay_panel.show()
        self._overlay_panel.set_title("HG Color+NIR")
        self._controls_spacer.show()
        self._bottom_thresh.show()
        self._color_thresh.show()
        self._jet_row_widget.show()
        self._gsense_config_widget.show()
        for w in (self._top_thresh, self._middle_thresh,
                  self._bottom_thresh, self._color_thresh):
            w.setMaximumWidth(260)

    def _on_jet_changed(self, _state: int) -> None:
        """Jet colormap checkbox toggled — reprocess GSense display."""
        self._process_and_display()

    def _on_overlay_alpha_changed(self, value: int) -> None:
        """Overlay transparency slider changed — re-render the overlay panel."""
        self._overlay_alpha_label.setText(f"{value}%")
        self._process_and_display()

    def _overlay_alpha(self) -> float:
        """Return the NIR overlay opacity (0..1) from the transparency slider."""
        return 1.0 - self._overlay_alpha_slider.value() / 100.0

    def _on_show_hdr_views(self) -> None:
        """Open (or raise) the HDR Color and HDR NIR viewer windows."""
        if self._camera_type != CameraType.GSENSE:
            return
        if self._hdr_color_dialog is None:
            self._hdr_color_dialog = HDRViewerDialog(
                "HDR Color (HG + LG fused)", self
            )
        if self._hdr_nir_dialog is None:
            self._hdr_nir_dialog = HDRViewerDialog(
                "HDR NIR (HG + LG fused)", self
            )
        self._hdr_color_dialog.show()
        self._hdr_color_dialog.raise_()
        self._hdr_nir_dialog.show()
        self._hdr_nir_dialog.raise_()
        self._update_hdr_views()

    def _update_hdr_views(self) -> None:
        """Recompute HDR images and push them to any visible HDR dialog."""
        color_dlg = self._hdr_color_dialog
        nir_dlg = self._hdr_nir_dialog
        color_visible = color_dlg is not None and color_dlg.isVisible()
        nir_visible = nir_dlg is not None and nir_dlg.isVisible()
        if not (color_visible or nir_visible):
            return
        if self._video_data.hg_rgbn_filtered is None:
            return
        try:
            color_img, nir_img = compute_hdr_displays(
                self._video_data, jet_nir=self._hg_nir_jet_cb.isChecked(),
            )
        except Exception as exc:
            self.statusBar().showMessage(f"HDR error: {exc}")
            return
        if color_visible:
            color_dlg.set_image(color_img)
        if nir_visible:
            nir_dlg.set_image(nir_img)

    def _on_gsense_config_changed(self, _index: int) -> None:
        """Pixel offset or color permutation changed — reprocess GSense display."""
        self._process_and_display()

    def _on_perm_prev(self) -> None:
        idx = self._perm_combo.currentIndex()
        if idx > 0:
            self._perm_combo.setCurrentIndex(idx - 1)

    def _on_perm_next(self) -> None:
        idx = self._perm_combo.currentIndex()
        if idx < self._perm_combo.count() - 1:
            self._perm_combo.setCurrentIndex(idx + 1)

    # -----------------------------------------------------------------
    # ROI (Create / Clear / Save)
    # -----------------------------------------------------------------
    def _on_create_roi(self) -> None:
        """Enter polygon drawing mode on the HG NIR panel to save an ROI."""
        if not self._file_loaded:
            return
        self._roi_save_mode = True
        self._middle_panel.set_roi_mode(True)
        self.statusBar().showMessage(
            "Create ROI: Click points on the HG NIR image to draw polygon. "
            "Right-click or double-click to finish."
        )

    def _on_clear_roi(self) -> None:
        """Delete the .roi.json sidecar file for the current H5 file."""
        if not self._file_loaded or self._h5_file_index < 0:
            return

        h5_path = self._h5_files[self._h5_file_index]
        roi_path = h5_path.with_suffix(".roi.json")

        if not roi_path.exists():
            QMessageBox.information(self, "No ROI", "No ROI file exists for this image.")
            return

        reply = QMessageBox.question(
            self,
            "Clear ROI",
            f"Delete ROI file?\n{roi_path.name}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            roi_path.unlink()
        except Exception as exc:
            QMessageBox.critical(self, "Delete Failed", str(exc))
            return

        self._update_roi_status()
        self.statusBar().showMessage(f"ROI cleared: {roi_path.name}")

    def _on_roi_selected(self, vertices: list[tuple[int, int]]) -> None:
        """Handle completed polygon ROI from the HG NIR panel."""
        if not self._roi_save_mode:
            return
        self._roi_save_mode = False

        if not self._file_loaded or self._h5_file_index < 0:
            return

        h5_path = self._h5_files[self._h5_file_index]
        roi_path = h5_path.with_suffix(".roi.json")

        # Get the displayed image shape
        if self._camera_type == CameraType.GSENSE:
            img_shape = list(self._video_data.hg_nir_raw.shape)
        else:
            img_shape = [self._h5info.attr.rows // 2,
                         self._h5info.attr.columns // 2]

        data = {
            "vertices": [list(v) for v in vertices],
            "image_shape": img_shape,
            "source_file": h5_path.name,
        }

        with open(roi_path, "w") as f:
            json.dump(data, f, indent=2)

        self._update_roi_status()
        self.statusBar().showMessage(
            f"ROI saved: {roi_path.name} ({len(vertices)} vertices)"
        )

    def _update_roi_status(self) -> None:
        """Show or hide the [ROI] status label based on sidecar file existence."""
        if not self._file_loaded or self._h5_file_index < 0:
            self._roi_status_label.setText("")
            return

        h5_path = self._h5_files[self._h5_file_index]
        roi_path = h5_path.with_suffix(".roi.json")
        if roi_path.exists():
            self._roi_status_label.setText("[ROI]")
        else:
            self._roi_status_label.setText("")

    def _update_tissue_status(self, h5_path: Path) -> None:
        """Check tissue_log.xlsx for the current file and update borders + label."""
        panels = [self._top_panel, self._middle_panel,
                  self._bottom_panel, self._color_panel,
                  self._overlay_panel]

        xlsx_path = h5_path.parent / "tissue_log.xlsx"
        if not xlsx_path.exists():
            self._tissue_label.setText("")
            for panel in panels:
                panel.set_border_color(None)
            return

        # Look up the current filename in the Excel log
        fname = h5_path.name
        cancerous = None
        try:
            wb = openpyxl.load_workbook(xlsx_path, read_only=True)
            ws = wb.active
            for row_idx in range(2, ws.max_row + 1):
                cell_val = ws.cell(row=row_idx, column=1).value
                if cell_val and str(cell_val).strip() == fname:
                    cancerous = int(ws.cell(row=row_idx, column=2).value)
                    break
            wb.close()
        except Exception:
            pass

        if cancerous is None:
            self._tissue_label.setText("")
            for panel in panels:
                panel.set_border_color(None)
            return

        from PyQt6.QtGui import QColor
        if cancerous == 1:
            color = QColor(255, 0, 0)
            self._tissue_label.setText("Tissue: Positive")
            self._tissue_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: red;"
            )
        else:
            color = QColor(0, 180, 0)
            self._tissue_label.setText("Tissue: Negative")
            self._tissue_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: green;"
            )

        for panel in panels:
            panel.set_border_color(color)

    @staticmethod
    def _load_roi_mask(roi_path: Path, image_shape: tuple[int, int]) -> np.ndarray | None:
        """Load a .roi.json file and return a boolean mask, or None if not found."""
        from matplotlib.path import Path as MplPath

        if not roi_path.exists():
            return None

        with open(roi_path) as f:
            data = json.load(f)

        vertices = data["vertices"]
        if len(vertices) < 3:
            return None

        h, w = image_shape
        poly_path = MplPath(vertices)
        yy, xx = np.mgrid[:h, :w]
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        mask = poly_path.contains_points(coords).reshape(h, w)
        return mask

    # -----------------------------------------------------------------
    # Processing helpers
    # -----------------------------------------------------------------
    def _get_filter_tap(self) -> int:
        text = self._filter_combo.currentText()
        return 0 if text == "None" else int(text)

    def _get_spike_tap(self) -> int:
        text = self._spike_combo.currentText()
        return 0 if text == "None" else int(text)

    def _get_processing_params(self) -> ProcessingParams:
        is_uv = "UV" in self._h5file.name if self._h5file.name else False
        return ProcessingParams(
            frame_number=self._frame_slider.value,
            filter_tap=self._get_filter_tap(),
            spike_tap=self._get_spike_tap(),
            top_thresh=ChannelThresholds(
                self._top_thresh.low, self._top_thresh.high
            ),
            middle_thresh=ChannelThresholds(
                self._middle_thresh.low, self._middle_thresh.high
            ),
            bottom_thresh=ChannelThresholds(
                self._bottom_thresh.low, self._bottom_thresh.high
            ),
            is_uv=is_uv,
            norm_bits=self._h5info.attr.norm_bits,
        )

    def _process_and_display(self) -> None:
        if not self._file_loaded:
            return

        try:
            if self._camera_type == CameraType.GSENSE:
                self._process_and_display_gsense()
            else:
                self._process_and_display_foveon()
        except Exception as exc:

            self.statusBar().showMessage(f"Display error: {exc}")

    def _process_and_display_foveon(self) -> None:
        params = self._get_processing_params()
        process_single_frame(self._h5info, self._video_data, params)

        # TOP / MIDDLE / BOTTOM are already [0, 255] from the jet colormap.
        top_img = self._video_data.top.copy()

        # Mask Pixels: show only pixels within the ROI that pass the Lo-Hi
        # threshold.  Uses the same filtered channel as the display pipeline
        # (spike filter + median filter) so the masking matches the jet image.
        if self._mask_pixels_cb.isChecked():
            raw = self._video_data.raw_single_frame_double
            norm = 2 ** self._h5info.attr.norm_bits
            top_ch = raw[::2, ::2, 2] / norm

            # Apply the same filters as the display pipeline
            from polarview.frame_processor import _spike_filter
            if params.spike_tap > 0:
                top_ch = _spike_filter(top_ch, params.spike_tap)
            if params.filter_tap > 0:
                top_ch = median_filter(
                    top_ch, size=(params.filter_tap, params.filter_tap),
                    mode="constant", cval=0.0,
                )

            # Build mask of pixels above Lo threshold, within ROI
            top_low = self._top_thresh.low / 100.0
            above_lo = top_ch >= top_low

            # Restrict to ROI if available
            roi_mask = None
            if self._h5_file_index >= 0:
                h5_path = self._h5_files[self._h5_file_index]
                roi_path = h5_path.with_suffix(".roi.json")
                roi_mask = self._load_roi_mask(roi_path, top_img.shape[:2])

            if roi_mask is not None:
                above_lo = above_lo & roi_mask

            # Keep only contiguous regions >= min_pixels
            min_px = self._min_pixels_spin.value()
            labeled_arr, n_comp = ndimage_label(above_lo)
            keep = np.zeros_like(above_lo)
            for comp_id in range(1, n_comp + 1):
                if int(np.sum(labeled_arr == comp_id)) >= min_px:
                    keep[labeled_arr == comp_id] = True

            # Black out everything that isn't in a kept region
            top_img[~keep] = 0.0

        # Display UV ROI: black out pixels outside the ROI
        elif self._uv_roi_cb.isChecked() and self._h5_file_index >= 0:
            h5_path = self._h5_files[self._h5_file_index]
            roi_path = h5_path.with_suffix(".roi.json")
            roi_mask = self._load_roi_mask(roi_path, top_img.shape[:2])
            if roi_mask is not None:
                top_img[~roi_mask] = 0.0

        self._top_panel.set_image(top_img)
        self._middle_panel.set_image(self._video_data.middle)
        self._bottom_panel.set_image(self._video_data.bottom)

        # COLOR: composite RGB in [0, 1] → apply threshold clamping per channel,
        # then scale to [0, 255].
        color_low = self._color_thresh.low / 100.0
        color_high = self._color_thresh.high / 100.0
        color = self._video_data.color.copy()
        color[color < color_low] = 0.0
        color[color > color_high] = 1.0
        denom = color_high - color_low
        if denom > 0:
            color = (color - color_low) / denom
        else:
            color = np.zeros_like(color)
        color = np.clip(color, 0.0, 1.0) * 255.0
        self._color_panel.set_image(color)

        # Cache displayed images so Save Images exports exactly what's on screen
        self._displayed_images = {
            "TOP Image": top_img,
            "MIDDLE Image": self._video_data.middle,
            "BOTTOM Image": self._video_data.bottom,
            "COLOR Image": color,
        }

    @staticmethod
    def _build_nir_color_overlay(
        hg_color: np.ndarray,
        hg_nir: np.ndarray,
        hg_nir_raw: np.ndarray,
        nir_low_pct: float,
        nir_high_pct: float,
        alpha: float,
    ) -> np.ndarray:
        """Blend HG NIR over HG Color where ``hg_nir_raw`` is in the [Lo, Hi] band.

        Outside the band the overlay is fully transparent (color shows through).
        ``alpha`` is the NIR opacity in [0, 1] within the band.
        """
        nir_low = nir_low_pct / 100.0
        nir_high = nir_high_pct / 100.0
        in_band = (hg_nir_raw >= nir_low) & (hg_nir_raw <= nir_high)
        out = hg_color.copy()
        out[in_band] = (
            alpha * hg_nir[in_band] + (1.0 - alpha) * hg_color[in_band]
        )
        return out

    @staticmethod
    def _apply_color_threshold(
        color_raw: np.ndarray, low_pct: float, high_pct: float
    ) -> np.ndarray:
        """Apply per-channel threshold clamping to an RGB [0,1] array → [0,255]."""
        low = low_pct / 100.0
        high = high_pct / 100.0
        color = color_raw.copy()
        color[color < low] = 0.0
        color[color > high] = 1.0
        denom = high - low
        if denom > 0:
            color = (color - low) / denom
        else:
            color = np.zeros_like(color)
        return np.clip(color, 0.0, 1.0) * 255.0

    def _process_and_display_gsense(self) -> None:
        offset = _GSENSE_OFFSETS[self._offset_combo.currentIndex()]
        perm = _GSENSE_PERMS[self._perm_combo.currentIndex()]
        params = GsenseProcessingParams(
            frame_number=self._frame_slider.value,
            filter_tap=self._get_filter_tap(),
            spike_tap=self._get_spike_tap(),
            hg_nir_thresh=ChannelThresholds(
                self._middle_thresh.low, self._middle_thresh.high
            ),
            lg_nir_thresh=ChannelThresholds(
                self._color_thresh.low, self._color_thresh.high
            ),
            hg_nir_jet=self._hg_nir_jet_cb.isChecked(),
            lg_nir_jet=self._lg_nir_jet_cb.isChecked(),
            norm_bits=self._h5info.attr.norm_bits,
            pixel_offset=offset,
            color_perm=perm,
        )
        process_gsense_frame(self._h5info, self._video_data, params)

        # HG Color: per-channel normalised [0,1] → threshold + scale to [0,255]
        hg_color = self._apply_color_threshold(
            self._video_data.hg_color_raw,
            self._top_thresh.low, self._top_thresh.high,
        )
        self._top_panel.set_image(hg_color)

        # HG NIR: already [0, 255] from the processing pipeline
        hg_nir_img = self._video_data.hg_nir.copy()

        if self._mask_pixels_cb.isChecked():
            # Mask pixels: keep only pixels above the NIR Lo threshold,
            # within ROI, in contiguous regions >= min_pixels.
            nir_ch = self._video_data.hg_nir_raw  # [0, 1]
            nir_low = self._middle_thresh.low / 100.0
            above_lo = nir_ch >= nir_low

            # Restrict to ROI if available
            roi_mask = None
            if self._h5_file_index >= 0:
                h5_path = self._h5_files[self._h5_file_index]
                roi_path = h5_path.with_suffix(".roi.json")
                roi_mask = self._load_roi_mask(roi_path, hg_nir_img.shape[:2])

            if roi_mask is not None:
                above_lo = above_lo & roi_mask

            # Keep only contiguous regions >= min_pixels
            min_px = self._min_pixels_spin.value()
            labeled_arr, n_comp = ndimage_label(above_lo)
            keep = np.zeros_like(above_lo)
            for comp_id in range(1, n_comp + 1):
                if int(np.sum(labeled_arr == comp_id)) >= min_px:
                    keep[labeled_arr == comp_id] = True

            hg_nir_img[~keep] = 0.0

        elif self._uv_roi_cb.isChecked() and self._h5_file_index >= 0:
            h5_path = self._h5_files[self._h5_file_index]
            roi_path = h5_path.with_suffix(".roi.json")
            roi_mask = self._load_roi_mask(roi_path, hg_nir_img.shape[:2])
            if roi_mask is not None:
                hg_nir_img[~roi_mask] = 0.0

        self._middle_panel.set_image(hg_nir_img)

        # LG Color
        lg_color = self._apply_color_threshold(
            self._video_data.lg_color_raw,
            self._bottom_thresh.low, self._bottom_thresh.high,
        )
        self._bottom_panel.set_image(lg_color)

        # LG NIR: already [0, 255]
        lg_nir_img = self._video_data.lg_nir.copy()
        self._color_panel.set_image(lg_nir_img)

        # HG Color + NIR overlay (5th panel).  Uses the unmasked threshold-clamped
        # HG NIR so the overlay is independent of "Mask UV Pixels" / "Display UV ROI".
        overlay = self._build_nir_color_overlay(
            hg_color, self._video_data.hg_nir, self._video_data.hg_nir_raw,
            self._middle_thresh.low, self._middle_thresh.high,
            self._overlay_alpha(),
        )
        self._overlay_panel.set_image(overlay)

        # Cache displayed images so Save Images exports exactly what's on screen
        self._displayed_images = {
            "HG Color": hg_color,
            "HG NIR": hg_nir_img,
            "LG Color": lg_color,
            "LG NIR": lg_nir_img,
            "HG Color+NIR": overlay,
        }

        # Push HDR-fused frame to any open HDR viewer windows.
        self._update_hdr_views()
