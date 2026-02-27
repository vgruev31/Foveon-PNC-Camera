"""Main application window – replaces PolarView.mlapp GUI + callbacks."""

from __future__ import annotations

import csv
import datetime
import json
import os
import shutil
import traceback
from itertools import permutations
from pathlib import Path

import h5py
import numpy as np
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
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .frame_processor import (
    ChannelThresholds,
    GsenseProcessingParams,
    ProcessingParams,
    process_gsense_frame,
    process_single_frame,
)
from .h5_loader import load_h5
from .image_saver import save_images
from .video_data import CameraType, H5File, H5Info, VideoData
from .widgets.frame_slider import FrameSlider
from .widgets.image_panel import ImagePanel
from .widgets.hsv_scatter_dialog import HSVScatterDialog
from .widgets.roc_dialog import ROCDialog
from .widgets.threshold_panel import ThresholdPanel


# GSense demosaic configuration constants
_GSENSE_OFFSETS = [(0, 0), (0, 1), (1, 0), (1, 1)]
_GSENSE_OFFSET_LABELS = ["(0, 0)", "Skip 1 col", "Skip 1 row", "Skip 1 row + col"]
# NIR is fixed at position (1,1); cycle through 6 permutations of R/G/B
# for positions (0,0)·(0,1)·(1,0).
_GSENSE_PERMS = [(*rgb, 'N') for rgb in permutations(('R', 'G', 'B'))]
_GSENSE_PERM_DEFAULT_IDX = _GSENSE_PERMS.index(('G', 'R', 'B', 'N'))


class PolarViewMainWindow(QMainWindow):
    """Port of the MATLAB PolarView App Designer GUI."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Foveon Perovskite Camera (PNC)")
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

        # -- H5 file navigation --
        self._h5_dir: Path | None = None  # directory of loaded file
        self._h5_files: list[Path] = []
        self._h5_file_index: int = -1

        # -- ROI save mode --
        self._roi_save_mode: bool = False

        # -- Hold HSV data across Show All HSV calls --
        self._held_hsv_groups: list[tuple[np.ndarray, str, str, int]] = []
        self._held_hsv_dlg: QDialog | None = None

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
        self._cancerous_cb = QCheckBox("Cancerous Tissue")
        rename_row.addWidget(self._cancerous_cb)
        self._rename_btn = QPushButton("Rename H5 File")
        self._rename_btn.setEnabled(False)
        rename_row.addWidget(self._rename_btn)
        rename_row.addStretch()
        rename_row.addWidget(QLabel("ROI:"))
        self._roi_spin = QSpinBox()
        self._roi_spin.setRange(1, 99)
        self._roi_spin.setValue(1)
        self._roi_spin.setFixedWidth(50)
        rename_row.addWidget(self._roi_spin)
        self._show_hsv_btn = QPushButton("Show HSV")
        self._show_hsv_btn.setEnabled(False)
        rename_row.addWidget(self._show_hsv_btn)
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
        save_row.addStretch()
        self._hold_hsv_cb = QCheckBox("Hold Data")
        self._hold_hsv_cb.setChecked(False)
        save_row.addWidget(self._hold_hsv_cb)
        self._hsv_tissue_combo = QComboBox()
        self._hsv_tissue_combo.addItems([
            "ALL", "TUMOR", "LN",
            "ALL LN", "Pos LN", "Neg LN",
        ])
        save_row.addWidget(self._hsv_tissue_combo)
        self._show_all_hsv_btn = QPushButton("Show All HSV")
        self._show_all_hsv_btn.setEnabled(False)
        save_row.addWidget(self._show_all_hsv_btn)
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

        image_row.addWidget(self._top_panel)
        image_row.addWidget(self._middle_panel)
        image_row.addWidget(self._bottom_panel)
        image_row.addWidget(self._color_panel)
        main_layout.addLayout(image_row, stretch=1)

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
        self._save_all_btn.clicked.connect(self._on_save_all_images)
        self._save_uv_btn.clicked.connect(self._on_save_uv_images)
        self._roc_btn.clicked.connect(self._on_compute_roc)
        self._create_roi_btn.clicked.connect(self._on_create_roi)
        self._clear_roi_btn.clicked.connect(self._on_clear_roi)
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
        self._rename_tissue.currentTextChanged.connect(self._on_tissue_changed)
        self._show_hsv_btn.clicked.connect(self._on_show_hsv)
        self._hold_hsv_cb.stateChanged.connect(self._on_hold_hsv_changed)
        self._show_all_hsv_btn.clicked.connect(self._on_show_all_hsv)
        self._top_panel.roi_selected.connect(self._on_top_roi_selected)
        self._hg_nir_jet_cb.stateChanged.connect(self._on_jet_changed)
        self._lg_nir_jet_cb.stateChanged.connect(self._on_jet_changed)
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

        # Copy ROI sidecar if it exists
        old_roi = old_path.with_suffix(".roi.json")
        if old_roi.exists():
            new_roi = new_path.with_suffix(".roi.json")
            try:
                shutil.copy2(old_roi, new_roi)
            except Exception:
                pass  # non-critical

        # Write CSV entry only for UV descriptor
        if descriptor == "UV":
            cancerous = 1 if self._cancerous_cb.isChecked() else 0
            csv_path = dest_dir / "tissue_log.csv"
            write_header = not csv_path.exists()
            try:
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(["filename", "tissue_type", "cancerous"])
                    writer.writerow([new_name, tissue, cancerous])
            except Exception as exc:
                QMessageBox.warning(self, "CSV Write Failed", str(exc))

        self.statusBar().showMessage(
            f"Copied: {old_path.name} → {dest_dir / new_name}"
        )

    def _on_tissue_changed(self, tissue: str) -> None:
        self._cancerous_cb.setChecked(tissue == "TUMOR")

    @staticmethod
    def _scrub_time_info(file_path: Path) -> None:
        """Remove time-related metadata from an H5 file and reset its timestamp.

        Mirrors MATLAB ``RemoveMetaData.m``:
        - Deletes ``/camera/timestamp`` dataset (per-frame timing)
        - Deletes ``time-info`` attribute on the root ``/`` group
        - Sets the file's modification time to January 1, 2000
        """
        with h5py.File(file_path, "a") as f:
            # Remove /camera/timestamp dataset
            if "camera" in f and "timestamp" in f["camera"]:
                del f["camera"]["timestamp"]
            # Remove root 'time-info' attribute
            root = f["/"]
            if "time-info" in root.attrs:
                del root.attrs["time-info"]

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
        self._save_all_btn.setEnabled(self._file_loaded and is_foveon)
        self._save_uv_btn.setEnabled(self._file_loaded and is_foveon)
        self._roc_btn.setEnabled(self._file_loaded and is_foveon)
        self._create_roi_btn.setEnabled(self._file_loaded and is_foveon)
        self._clear_roi_btn.setEnabled(self._file_loaded and is_foveon)
        self._show_hsv_btn.setEnabled(self._file_loaded and is_foveon)
        self._show_all_hsv_btn.setEnabled(self._file_loaded and is_foveon)
        self._roi_spin.setEnabled(self._file_loaded and is_foveon)

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
        except Exception as exc:
            traceback.print_exc()
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
        """Process every H5 file in the current directory and save all 4 images."""
        if not self._file_loaded or not self._h5_files:
            QMessageBox.warning(self, "No File", "No file loaded.")
            return

        h5_dir = self._h5_files[0].parent
        all_h5 = sorted(h5_dir.glob("*.h5"))
        if not all_h5:
            return

        params = self._get_processing_params()
        params.frame_number = 0

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
                vd.allocate(info.attr.rows, info.attr.columns)
                p = ProcessingParams(
                    frame_number=0,
                    filter_tap=params.filter_tap,
                    spike_tap=params.spike_tap,
                    top_thresh=params.top_thresh,
                    middle_thresh=params.middle_thresh,
                    bottom_thresh=params.bottom_thresh,
                    is_uv="UV" in fp.stem,
                    norm_bits=info.attr.norm_bits,
                )
                process_single_frame(info, vd, p)
                h5f = H5File(name=fp.stem, path=str(fp.parent))
                saved = save_images(vd, h5f)
                saved_total += len(saved)
            except Exception as exc:
                errors.append(f"{fp.name}: {exc}")

        msg = f"Saved {saved_total} images from {len(all_h5)} H5 files."
        if errors:
            msg += f"\n\n{len(errors)} error(s):\n" + "\n".join(errors)
        QMessageBox.information(self, "Save ALL Complete", msg)
        self.statusBar().showMessage(f"Save ALL complete: {saved_total} images")

    def _on_save_uv_images(self) -> None:
        """Process every UV H5 file in the current directory and save the TOP and COLOR images."""
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

        params = self._get_processing_params()

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
                vd.allocate(info.attr.rows, info.attr.columns)
                p = ProcessingParams(
                    frame_number=0,
                    filter_tap=params.filter_tap,
                    spike_tap=params.spike_tap,
                    top_thresh=params.top_thresh,
                    middle_thresh=params.middle_thresh,
                    bottom_thresh=params.bottom_thresh,
                    is_uv=True,
                    norm_bits=info.attr.norm_bits,
                )
                process_single_frame(info, vd, p)

                # Save TOP image (already [0, 255] from jet colormap)
                top_img = np.clip(vd.top, 0.0, 255.0).astype(np.uint8)
                top_path = output_dir / f"{fp.stem} TOP Image.png"
                Image.fromarray(top_img, mode="RGB").save(str(top_path))

                # Save COLOR image with threshold clamping (same as display)
                color_low = self._color_thresh.low / 100.0
                color_high = self._color_thresh.high / 100.0
                color_img = vd.color.copy()
                color_img[color_img < color_low] = 0.0
                color_img[color_img > color_high] = 1.0
                denom = color_high - color_low
                if denom > 0:
                    color_img = (color_img - color_low) / denom
                else:
                    color_img = np.zeros_like(color_img)
                color_img = (np.clip(color_img, 0.0, 1.0) * 255.0).astype(np.uint8)
                color_path = output_dir / f"{fp.stem} COLOR Image.png"
                Image.fromarray(color_img, mode="RGB").save(str(color_path))

                saved_count += 1
            except Exception as exc:
                errors.append(f"{fp.name}: {exc}")

        msg = f"Saved {saved_count} UV images to:\n{output_dir}"
        if errors:
            msg += f"\n\n{len(errors)} error(s):\n" + "\n".join(errors)
        QMessageBox.information(self, "Save UV Complete", msg)
        self.statusBar().showMessage(f"Save UV complete: {saved_count} images")

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

    def _on_compute_roc(self) -> None:
        """Compute and display an ROC curve for UV images using tissue_log.csv.

        For each file, uses binary search to find the maximum Lo threshold
        (0-100) at which the image is classified positive (has a contiguous
        region >= min_pixels pixels above threshold within the ROI).  This
        per-file score is then used for fast score-based ROC computation.
        """
        if not self._file_loaded or not self._h5_files:
            QMessageBox.warning(self, "No File", "No file loaded.")
            return

        h5_dir = self._h5_files[0].parent
        csv_path = h5_dir / "tissue_log.csv"
        if not csv_path.exists():
            QMessageBox.warning(
                self, "No tissue_log.csv",
                f"tissue_log.csv not found in:\n{h5_dir}\n\n"
                "Rename UV files first to generate it.",
            )
            return

        # Read tissue_log.csv
        entries: list[tuple[str, str, int]] = []  # (filename, tissue_type, cancerous)
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append((
                    row["filename"].strip(),
                    row["tissue_type"].strip(),
                    int(row["cancerous"]),
                ))

        if not entries:
            QMessageBox.warning(self, "Empty CSV", "tissue_log.csv contains no entries.")
            return

        # Phase 1: Load all files — extract filtered TOP channel + ROI mask
        # Use the same filter currently selected in the GUI
        from polarview.frame_processor import _spike_filter
        roc_median_tap = self._get_filter_tap()
        roc_spike_tap = self._get_spike_tap()
        filter_desc = "None"
        if roc_median_tap > 0:
            filter_desc = f"Median {roc_median_tap}"
        elif roc_spike_tap > 0:
            filter_desc = f"Spike {roc_spike_tap}"

        file_data: list[tuple] = []  # (top_filtered, roi_mask, label, fname, ttype, h5_path)
        errors: list[str] = []

        for i, (fname, ttype, label) in enumerate(entries):
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
                raw_frame = info.raw_data[:, :, :, 0].astype(np.float64)
                norm = 2 ** info.attr.norm_bits
                top_ch = raw_frame[::2, ::2, 2] / norm

                # Apply whichever filter is selected (only one can be active)
                if roc_spike_tap > 0:
                    top_ch = _spike_filter(top_ch, roc_spike_tap)
                if roc_median_tap > 0:
                    top_ch = median_filter(
                        top_ch, size=(roc_median_tap, roc_median_tap),
                        mode="constant", cval=0.0,
                    )

                roi_path = fp.with_suffix(".roi.json")
                roi_mask = self._load_roi_mask(roi_path, top_ch.shape)

                file_data.append((top_ch, roi_mask, label, fname, ttype, fp))
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
        # The score is the max threshold (0-100) at which the file is positive.
        # Classification is monotonic: as threshold rises, pixels drop out,
        # so a positive file can only become negative, never the reverse.
        # ~15 iterations of binary search gives precision < 0.01.
        min_pixels = self._min_pixels_spin.value()
        scores = np.zeros(n_files, dtype=float)

        for fi, (top_filtered, roi_mask, label, fname, ttype, fp) in enumerate(file_data):
            self.statusBar().showMessage(
                f"Computing scores (min {min_pixels} px): {fi + 1}/{n_files} — {fname}"
            )
            QApplication.processEvents()

            # Check if positive at the lowest threshold (0)
            if not self._classify_at_threshold(top_filtered, roi_mask, 0.0, min_pixels):
                scores[fi] = -1.0  # never positive
                continue

            # Check if positive at the highest threshold (100)
            if self._classify_at_threshold(top_filtered, roi_mask, 1.0, min_pixels):
                scores[fi] = 100.0
                continue

            # Binary search in [0, 100] for the crossover point
            lo, hi = 0.0, 100.0
            for _ in range(15):
                mid = (lo + hi) / 2.0
                if self._classify_at_threshold(top_filtered, roi_mask, mid / 100.0, min_pixels):
                    lo = mid
                else:
                    hi = mid
            scores[fi] = lo

        labels_arr = np.array([d[2] for d in file_data])
        filenames_list = [d[3] for d in file_data]
        tissue_list = [d[4] for d in file_data]

        n_pos = int(labels_arr.sum())
        self.statusBar().showMessage(
            f"ROC computed from {n_files} files "
            f"({n_pos} positive, {n_files - n_pos} negative)"
        )

        # Phase 3: Show ROC dialog
        dlg = ROCDialog(self)
        optimal_thresh = dlg.plot_roc(
            labels_arr, scores, filenames_list, tissue_list,
        )
        dlg.show()

        if optimal_thresh is None:
            return

        # Phase 4: Generate thresholded B&W images at the optimal threshold
        output_dir = h5_dir / "ROC Thresholded Images"
        output_dir.mkdir(exist_ok=True)

        # Clear previous PNG files from the output directory
        for old_png in output_dir.glob("*.png"):
            old_png.unlink()

        thresh_val = optimal_thresh / 100.0
        saved_count = 0

        for fi, (top_filtered, roi_mask, label, fname, ttype, fp) in enumerate(file_data):
            self.statusBar().showMessage(
                f"Generating thresholded images: {fi + 1}/{n_files} — {fname}"
            )
            QApplication.processEvents()

            h_img, w_img = top_filtered.shape
            rgb = np.zeros((h_img, w_img, 3), dtype=np.uint8)

            if roi_mask is not None:
                above = roi_mask & (top_filtered >= thresh_val)
            else:
                above = top_filtered >= thresh_val

            # Keep only contiguous regions >= min_pixels
            labeled_arr, n_comp = ndimage_label(above)
            keep = np.zeros_like(above)
            for comp_id in range(1, n_comp + 1):
                if int(np.sum(labeled_arr == comp_id)) >= min_pixels:
                    keep[labeled_arr == comp_id] = True
            rgb[keep] = [255, 255, 255]

            if roi_mask is not None:
                # Draw ROI boundary in green
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

        # Free file data
        del file_data

        self.statusBar().showMessage(
            f"ROC complete. Saved {saved_count} thresholded images to {output_dir.name}/"
        )
        QMessageBox.information(
            self, "ROC Complete",
            f"Optimal threshold: {optimal_thresh:.1f} / 100\n"
            f"Filter: {filter_desc}\n\n"
            f"Set the TOP Lo slider to {optimal_thresh:.1f} and check 'Mask UV Pixels'\n"
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
        self._bottom_thresh.show()
        self._color_thresh.show()
        self._jet_row_widget.hide()
        self._gsense_config_widget.hide()
        # Restore threshold panel max width for 4-column layout
        for w in (self._top_thresh, self._middle_thresh,
                  self._bottom_thresh, self._color_thresh):
            w.setMaximumWidth(260)

    def _setup_gsense_ui(self) -> None:
        """Configure panel visibility for GSense 4-panel demosaiced mode."""
        self._top_panel.set_title("HG Color")
        self._middle_panel.set_title("HG NIR/UV")
        self._bottom_panel.show()
        self._bottom_panel.set_title("LG Color")
        self._color_panel.show()
        self._color_panel.set_title("LG NIR/UV")
        self._bottom_thresh.show()
        self._color_thresh.show()
        self._jet_row_widget.show()
        self._gsense_config_widget.show()
        # Restore 4-column max widths
        for w in (self._top_thresh, self._middle_thresh,
                  self._bottom_thresh, self._color_thresh):
            w.setMaximumWidth(260)

    def _on_jet_changed(self, _state: int) -> None:
        """Jet colormap checkbox toggled — reprocess GSense display."""
        self._process_and_display()

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
    # ROI save mode (Create ROI / Clear ROI)
    # -----------------------------------------------------------------
    def _on_create_roi(self) -> None:
        """Enter polygon drawing mode on the TOP panel to save an ROI."""
        if not self._file_loaded:
            return
        self._roi_save_mode = True
        self._top_panel.set_roi_mode(True)
        self.statusBar().showMessage(
            "Create ROI: Click points on the TOP image to draw polygon. "
            "Right-click or double-click to finish."
        )

    def _save_roi_to_file(self, vertices: list[tuple[int, int]]) -> None:
        """Write polygon vertices to a .roi.json sidecar file."""
        if not self._file_loaded or self._h5_file_index < 0:
            return

        h5_path = self._h5_files[self._h5_file_index]
        roi_path = h5_path.with_suffix(".roi.json")

        # Determine image shape (half-resolution from 2×2 subsampling)
        rows_half = self._h5info.attr.rows // 2
        cols_half = self._h5info.attr.columns // 2

        data = {
            "vertices": [list(v) for v in vertices],
            "image_shape": [rows_half, cols_half],
            "source_file": h5_path.name,
        }

        with open(roi_path, "w") as f:
            json.dump(data, f, indent=2)

        self._update_roi_status()
        self.statusBar().showMessage(
            f"ROI saved: {roi_path.name} ({len(vertices)} vertices)"
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
    # HSV scatter plot
    # -----------------------------------------------------------------
    _hsv_dialog: HSVScatterDialog | None = None

    def _on_show_hsv(self) -> None:
        """Enter polygon ROI selection mode on the TOP panel."""
        if not self._file_loaded:
            return
        roi_num = self._roi_spin.value()
        self.statusBar().showMessage(
            f"ROI {roi_num}: Click points on the TOP image to draw polygon. "
            "Right-click or double-click to finish."
        )
        self._top_panel.set_roi_mode(True)

    def _on_hold_hsv_changed(self, state: int) -> None:
        """Clear accumulated HSV data when Hold Data is unchecked."""
        if state == 0:  # unchecked
            self._held_hsv_groups = []
            self._held_hsv_dlg = None

    def _on_show_all_hsv(self) -> None:
        """Load all UV files, apply ROI + filter + min pixels, plot HSV scatter.

        Negative samples (cancerous=0) and positive samples (cancerous=1)
        are plotted in different colours.  Uses tissue_log.csv for labels.
        """
        from matplotlib.colors import rgb_to_hsv
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Ellipse
        from matplotlib.patheffects import Stroke, Normal
        from polarview.frame_processor import _spike_filter

        if not self._file_loaded or not self._h5_files:
            QMessageBox.warning(self, "No File", "No file loaded.")
            return

        h5_dir = self._h5_files[0].parent
        csv_path = h5_dir / "tissue_log.csv"
        if not csv_path.exists():
            QMessageBox.warning(
                self, "No tissue_log.csv",
                f"tissue_log.csv not found in:\n{h5_dir}\n\n"
                "Rename UV files first to generate it.",
            )
            return

        # Read tissue_log.csv
        entries: list[tuple[str, str, int]] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append((
                    row["filename"].strip(),
                    row["tissue_type"].strip(),
                    int(row["cancerous"]),
                ))

        if not entries:
            QMessageBox.warning(self, "Empty CSV", "tissue_log.csv has no entries.")
            return

        # Filter by selected tissue type / label
        tissue_filter = self._hsv_tissue_combo.currentText()
        if tissue_filter == "TUMOR":
            entries = [(f, t, l) for f, t, l in entries if t == "TUMOR"]
        elif tissue_filter == "LN":
            entries = [(f, t, l) for f, t, l in entries if t == "LN"]
        elif tissue_filter == "ALL LN":
            entries = [(f, t, l) for f, t, l in entries if t == "LN"]
        elif tissue_filter == "Pos LN":
            entries = [(f, t, l) for f, t, l in entries if t == "LN" and l == 1]
        elif tissue_filter == "Neg LN":
            entries = [(f, t, l) for f, t, l in entries if t == "LN" and l == 0]
        # else: "ALL" — keep all entries

        if not entries:
            QMessageBox.warning(
                self, "No Matches",
                f"No {tissue_filter} entries found in tissue_log.csv.",
            )
            return

        # Current GUI settings
        median_tap = self._get_filter_tap()
        spike_tap = self._get_spike_tap()
        min_pixels = self._min_pixels_spin.value()
        top_low = self._top_thresh.low / 100.0

        neg_pixels_all: list[np.ndarray] = []  # list of (N, 3) RGB arrays
        pos_pixels_all: list[np.ndarray] = []

        for i, (fname, ttype, label) in enumerate(entries):
            self.statusBar().showMessage(
                f"Show All HSV: {i + 1}/{len(entries)} — {fname}"
            )
            QApplication.processEvents()

            fp = h5_dir / fname
            if not fp.exists():
                continue

            try:
                info = load_h5(fp)
                raw_frame = info.raw_data[:, :, :, 0].astype(np.float64)
                norm = 2 ** info.attr.norm_bits

                top_ch = raw_frame[::2, ::2, 2] / norm
                mid_ch = raw_frame[::2, ::2, 1] / norm
                bot_ch = raw_frame[::2, ::2, 0] / norm

                # Apply selected filter
                if spike_tap > 0:
                    top_ch = _spike_filter(top_ch, spike_tap)
                    mid_ch = _spike_filter(mid_ch, spike_tap)
                    bot_ch = _spike_filter(bot_ch, spike_tap)
                if median_tap > 0:
                    top_ch = median_filter(top_ch, size=(median_tap, median_tap),
                                           mode="constant", cval=0.0)
                    mid_ch = median_filter(mid_ch, size=(median_tap, median_tap),
                                           mode="constant", cval=0.0)
                    bot_ch = median_filter(bot_ch, size=(median_tap, median_tap),
                                           mode="constant", cval=0.0)

                # Build COLOR composite [0,1]: Red=bottom, Green=middle, Blue=top
                def _norm(ch):
                    lo, hi = ch.min(), ch.max()
                    return (ch - lo) / (hi - lo) if hi > lo else np.zeros_like(ch)

                color = np.stack([_norm(bot_ch), _norm(mid_ch), _norm(top_ch)], axis=-1)

                # Load ROI mask
                roi_path = fp.with_suffix(".roi.json")
                roi_mask = self._load_roi_mask(roi_path, top_ch.shape)

                # Build pixel mask: within ROI, above Lo, in clusters >= min_pixels
                above_lo = top_ch >= top_low
                if roi_mask is not None:
                    above_lo = above_lo & roi_mask

                labeled_arr, n_comp = ndimage_label(above_lo)
                keep = np.zeros_like(above_lo)
                for comp_id in range(1, n_comp + 1):
                    if int(np.sum(labeled_arr == comp_id)) >= min_pixels:
                        keep[labeled_arr == comp_id] = True

                pixels = color[keep]  # (N, 3)
                if pixels.size == 0:
                    continue

                if label == 1:
                    pos_pixels_all.append(pixels)
                else:
                    neg_pixels_all.append(pixels)

            except Exception:
                continue

        if not neg_pixels_all and not pos_pixels_all:
            QMessageBox.warning(self, "No Data", "No valid pixels found across files.")
            return

        # Convert collected pixels to HSV groups
        _HOLD_COLORS = [
            "blue", "red", "green", "purple", "orange",
            "brown", "magenta", "cyan", "olive", "deeppink",
        ]

        new_groups: list[tuple[np.ndarray, str, str, int]] = []
        for group_pixels, default_color, group_label in [
            (neg_pixels_all, "blue", f"{tissue_filter} Neg"),
            (pos_pixels_all, "red", f"{tissue_filter} Pos"),
        ]:
            if not group_pixels:
                continue
            all_rgb = np.concatenate(group_pixels, axis=0)
            all_rgb = np.clip(all_rgb, 0.0, 1.0)
            hsv = rgb_to_hsv(all_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
            new_groups.append((hsv, default_color, group_label, len(all_rgb)))

        hold_mode = self._hold_hsv_cb.isChecked()

        if hold_mode:
            # Assign unique colors from palette based on accumulated count
            for hsv, _, group_label, n_px in new_groups:
                idx = len(self._held_hsv_groups) % len(_HOLD_COLORS)
                color = _HOLD_COLORS[idx]
                self._held_hsv_groups.append((hsv, color, group_label, n_px))
            group_data = self._held_hsv_groups
        else:
            # Fresh plot — use default red/blue colors
            self._held_hsv_groups = []
            group_data = new_groups

        # Build or reuse dialog
        if hold_mode and self._held_hsv_dlg is not None:
            try:
                self._held_hsv_dlg.isVisible()  # test if still alive
                dlg = self._held_hsv_dlg
                # Clear existing figure
                dlg.findChild(FigureCanvasQTAgg).figure.clear()
                fig = dlg.findChild(FigureCanvasQTAgg).figure
                canvas = dlg.findChild(FigureCanvasQTAgg)
            except RuntimeError:
                # Dialog was closed / deleted
                hold_mode = False
                self._held_hsv_dlg = None

        if not hold_mode or self._held_hsv_dlg is None:
            dlg = QDialog(self)
            dlg.setMinimumSize(1200, 550)
            fig = Figure(figsize=(13, 5.5), dpi=100)
            canvas = FigureCanvasQTAgg(fig)
            layout = QVBoxLayout()
            layout.addWidget(canvas)
            dlg.setLayout(layout)
            if hold_mode:
                self._held_hsv_dlg = dlg

        title_parts = sorted({g[2] for g in group_data})
        dlg.setWindowTitle(f"HSV Scatter — {' | '.join(title_parts)}")

        ax2d = fig.add_subplot(121)
        ax3d = fig.add_subplot(122, projection="3d")

        halo = [Stroke(linewidth=3, foreground="white"), Normal()]

        # --- 2D: Hue vs Saturation ---
        for hsv, color_name, group_label, n_px in group_data:
            hue = hsv[:, 0] * 360
            sat = hsv[:, 1]

            ax2d.scatter(sat, hue, c=color_name, s=4, alpha=0.3,
                         edgecolors="none", zorder=1)

            mean_sat = float(np.mean(sat))
            mean_hue = float(np.mean(hue))
            std_sat = float(np.std(sat))
            std_hue = float(np.std(hue))

            ax2d.plot(mean_sat, mean_hue, "x", color="black",
                      markersize=10, markeredgewidth=2, zorder=10,
                      path_effects=halo)

            ax2d.scatter([], [], c=color_name, s=40, edgecolors="none",
                         label=f"{group_label} ({n_px} px): "
                               f"S={mean_sat:.3f}, H={mean_hue:.1f}°")

            ellipse = Ellipse(
                (mean_sat, mean_hue),
                width=4 * std_sat, height=4 * std_hue,
                fill=False, edgecolor=color_name, linewidth=2,
                linestyle="--", zorder=9, path_effects=halo,
            )
            ax2d.add_patch(ellipse)

        ax2d.set_xlabel("Saturation")
        ax2d.set_ylabel("Hue (degrees)")
        ax2d.set_xlim(0, 1)
        ax2d.set_ylim(0, 360)
        ax2d.set_title(f"H vs S — {len(group_data)} group(s)")
        ax2d.legend(fontsize=7, loc="upper right")
        ax2d.grid(True, alpha=0.3)

        # --- 3D: Hue vs Saturation vs Value ---
        for hsv, color_name, group_label, n_px in group_data:
            hue = hsv[:, 0] * 360
            sat = hsv[:, 1]
            val = hsv[:, 2]

            max_pts = 5000
            if len(hue) > max_pts:
                idx = np.random.choice(len(hue), max_pts, replace=False)
                hue_s, sat_s, val_s = hue[idx], sat[idx], val[idx]
            else:
                hue_s, sat_s, val_s = hue, sat, val

            ax3d.scatter(sat_s, hue_s, val_s, c=color_name, s=4, alpha=0.3,
                         edgecolors="none", label=f"{group_label} ({n_px} px)")

        ax3d.set_xlabel("Saturation")
        ax3d.set_ylabel("Hue (°)")
        ax3d.set_zlabel("Value")
        ax3d.set_xlim(0, 1)
        ax3d.set_ylim(0, 360)
        ax3d.set_zlim(0, 1)
        ax3d.set_title("H vs S vs V")
        ax3d.legend(fontsize=7, loc="upper right")

        fig.tight_layout()
        canvas.draw()

        self.statusBar().showMessage(
            f"Show All HSV: {len(neg_pixels_all)} neg files, "
            f"{len(pos_pixels_all)} pos files"
        )
        dlg.show()

    def _on_top_roi_selected(self, vertices: list[tuple[int, int]]) -> None:
        """Handle polygon ROI — save to file or add to HSV scatter."""
        # If in ROI save mode, save to file and return
        if self._roi_save_mode:
            self._roi_save_mode = False
            self._save_roi_to_file(vertices)
            return

        from matplotlib.path import Path as MplPath

        roi_num = self._roi_spin.value()
        self.statusBar().showMessage(
            f"ROI {roi_num}: {len(vertices)} vertices"
        )
        color_data = self._video_data.color  # (H, W, 3), [0, 1]
        if color_data is None:
            return

        h, w = color_data.shape[:2]

        # Build polygon path and mask
        poly_path = MplPath(vertices)
        yy, xx = np.mgrid[:h, :w]
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        mask = poly_path.contains_points(coords).reshape(h, w)

        # Only keep pixels where top channel is within the TOP lo-hi range
        raw = self._video_data.raw_single_frame_double
        top_ch = raw[::2, ::2, 2] / (2 ** self._h5info.attr.norm_bits)
        top_low = self._top_thresh.low / 100.0
        top_high = self._top_thresh.high / 100.0
        in_range = (top_ch >= top_low) & (top_ch <= top_high)
        mask = mask & in_range

        pixels = color_data[mask]  # (N, 3), [0, 1]
        if pixels.size == 0:
            QMessageBox.warning(self, "Empty ROI", "Selected region has no pixels.")
            return

        # Create dialog on first ROI, reuse for subsequent ones
        if self._hsv_dialog is None or not self._hsv_dialog.isVisible():
            self._hsv_dialog = HSVScatterDialog(parent=self)
            self._hsv_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            self._hsv_dialog.destroyed.connect(self._on_hsv_dialog_closed)
            self._hsv_dialog.show()

        self._hsv_dialog.add_roi(pixels, f"ROI {roi_num}")

        # Auto-increment ROI number for convenience
        self._roi_spin.setValue(roi_num + 1)

    def _on_hsv_dialog_closed(self) -> None:
        self._hsv_dialog = None

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
            traceback.print_exc()
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
        self._middle_panel.set_image(self._video_data.hg_nir)

        # LG Color
        lg_color = self._apply_color_threshold(
            self._video_data.lg_color_raw,
            self._bottom_thresh.low, self._bottom_thresh.high,
        )
        self._bottom_panel.set_image(lg_color)

        # LG NIR: already [0, 255]
        self._color_panel.set_image(self._video_data.lg_nir)
