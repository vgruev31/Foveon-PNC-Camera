"""Main application window – replaces PolarView.mlapp GUI + callbacks."""

from __future__ import annotations

import csv
import datetime
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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

from .frame_processor import ChannelThresholds, ProcessingParams, process_single_frame
from .h5_loader import load_h5
from .image_saver import save_images
from .video_data import H5File, H5Info, VideoData
from .widgets.frame_slider import FrameSlider
from .widgets.image_panel import ImagePanel
from .widgets.hsv_scatter_dialog import HSVScatterDialog
from .widgets.threshold_panel import ThresholdPanel


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

        # -- H5 file navigation --
        self._h5_files: list[Path] = []
        self._h5_file_index: int = -1

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
        self._delete_btn = QPushButton("Delete H5 File")
        self._prev_btn.setEnabled(False)
        self._next_btn.setEnabled(False)
        self._delete_btn.setEnabled(False)
        nav_row.addWidget(self._prev_btn)
        nav_row.addWidget(self._next_btn)
        nav_row.addWidget(self._delete_btn)
        nav_row.addStretch()
        nav_row.addWidget(QLabel("Median Filter:"))
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["None", "3", "5", "7", "9", "11"])
        nav_row.addWidget(self._filter_combo)
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

        # --- Row 1d: Save buttons ---
        save_row = QHBoxLayout()
        self._save_btn = QPushButton("Save Images")
        save_row.addWidget(self._save_btn)
        self._save_all_btn = QPushButton("Save ALL Images")
        self._save_all_btn.setEnabled(False)
        save_row.addWidget(self._save_all_btn)
        self._save_uv_btn = QPushButton("Save UV Images")
        self._save_uv_btn.setEnabled(False)
        save_row.addWidget(self._save_uv_btn)
        main_layout.addLayout(save_row)

        # --- Row 2: Frame slider ---
        self._frame_slider = FrameSlider()
        main_layout.addWidget(self._frame_slider)

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
        self._delete_btn.clicked.connect(self._on_delete_file)
        self._rename_btn.clicked.connect(self._on_rename_file)
        self._save_all_btn.clicked.connect(self._on_save_all_images)
        self._save_uv_btn.clicked.connect(self._on_save_uv_images)
        self._scrub_browse_btn.clicked.connect(self._on_scrub_browse)
        self._scrub_btn.clicked.connect(self._on_scrub_files)
        self._frame_slider.frame_changed.connect(self._on_frame_changed)
        self._frame_slider.play_next_file.connect(self._on_play_next_file)
        self._top_thresh.thresholds_changed.connect(self._on_threshold_changed)
        self._middle_thresh.thresholds_changed.connect(self._on_threshold_changed)
        self._bottom_thresh.thresholds_changed.connect(self._on_threshold_changed)
        self._color_thresh.thresholds_changed.connect(self._on_threshold_changed)
        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        self._save_btn.clicked.connect(self._on_save)
        self._rename_tissue.currentTextChanged.connect(self._on_tissue_changed)
        self._show_hsv_btn.clicked.connect(self._on_show_hsv)
        self._top_panel.roi_selected.connect(self._on_top_roi_selected)

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
        # Scan directory for all .h5 files and find current index
        self._h5_files = sorted(p.parent.glob("*.h5"))
        try:
            self._h5_file_index = self._h5_files.index(p)
        except ValueError:
            self._h5_files = [p]
            self._h5_file_index = 0

        self._load_file(p)

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

        h5_files = sorted(dir_path.glob("*.h5"))
        if not h5_files:
            QMessageBox.information(
                self, "No Files",
                f"No .h5 files found in:\n{directory}",
            )
            return

        self.statusBar().showMessage(f"Scrubbing {len(h5_files)} H5 files...")
        QApplication.processEvents()

        scrubbed = 0
        errors = []
        for fp in h5_files:
            try:
                self._scrub_time_info(fp)
                scrubbed += 1
            except Exception as exc:
                errors.append(f"{fp.name}: {exc}")

        # Also reset directory timestamp
        try:
            epoch_2000 = datetime.datetime(2000, 1, 1).timestamp()
            os.utime(dir_path, (epoch_2000, epoch_2000))
        except Exception:
            pass

        msg = f"Scrubbed {scrubbed} of {len(h5_files)} H5 files."
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
        self._save_all_btn.setEnabled(self._file_loaded)
        self._save_uv_btn.setEnabled(self._file_loaded)
        self._show_hsv_btn.setEnabled(self._file_loaded)

    def _load_file(self, p: Path) -> None:
        self.statusBar().showMessage(f"Loading {p}...")
        QApplication.processEvents()

        try:
            self._h5info = load_h5(p)
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", str(exc))
            self.statusBar().showMessage("Load failed")
            return

        self._h5file = H5File(name=p.stem, path=str(p.parent))
        self._filename_edit.setText(str(p))

        self._video_data.allocate(
            self._h5info.attr.rows, self._h5info.attr.columns
        )
        self._frame_slider.set_range(self._h5info.attr.num_frames)

        self._file_loaded = True
        self._update_nav_buttons()
        self.statusBar().showMessage(
            f"Loaded: {p.name}  "
            f"({self._h5info.attr.rows}×{self._h5info.attr.columns}, "
            f"{self._h5info.attr.num_frames} frames, "
            f"{self._h5info.attr.frame_rate:.1f} fps)"
        )

        self._process_and_display()

    def _on_frame_changed(self, _frame_index: int) -> None:
        self._process_and_display()

    def _on_threshold_changed(self, _low: float, _high: float) -> None:
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

    def _on_top_roi_selected(self, vertices: list[tuple[int, int]]) -> None:
        """Handle polygon ROI — extract thresholded RGB pixels and add to HSV scatter."""
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

    def _get_processing_params(self) -> ProcessingParams:
        is_uv = "UV" in self._h5file.name if self._h5file.name else False
        return ProcessingParams(
            frame_number=self._frame_slider.value,
            filter_tap=self._get_filter_tap(),
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

        params = self._get_processing_params()
        process_single_frame(self._h5info, self._video_data, params)

        # TOP / MIDDLE / BOTTOM are already [0, 255] from the jet colormap.
        self._top_panel.set_image(self._video_data.top)
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
