# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PolarView (PNC) is a PyQt6 desktop application for viewing and processing multi-spectral polarization camera data. It supports two camera types: **Foveon** (stacked RGB, 3 channels) and **GSense** (FSI dual-gain monochrome with Bayer-like RGBN demosaic). It loads HDF5 (`.h5`) video files, processes frames with jet colormaps or grayscale rendering, supports ROI selection with HSV scatter analysis and ROC curve computation, and exports PNGs. This is a Python port of the original MATLAB PolarView application.

## Running the Application

```bash
python run.py
# or
python -m polarview
```

Dependencies: `pip install -r requirements.txt` (PyQt6, numpy, scipy, h5py, Pillow, matplotlib, statsmodels).

There are no tests or linters configured for this project.

## Architecture

### Data Flow Pipeline

1. **File Open** — user picks H5 file via `QFileDialog`, or navigates with Previous/Next buttons
2. **Load** (`h5_loader.py`) — reads `/camera/frames`, auto-detects camera type from HDF5 attributes (`sensor-desc`, `imager`, `model-name`, `manufacture`), reorders axes, applies bit-shift normalization
3. **Frame Processing** (`frame_processor.py`) — two separate pipelines:
   - **Foveon**: extracts 3 channels via 2×2 subsampling, builds COLOR composite, applies optional median/spike filters, jet colormaps per channel
   - **GSense**: splits frame into HG/LG halves, subsamples, demosaics 2×2 RGBN pattern, produces COLOR + NIR images per gain
4. **Display** (`main_window.py`) — renders image panels; Foveon shows TOP/MIDDLE/BOTTOM/COLOR, GSense shows HG-Color/HG-NIR/LG-Color/LG-NIR
5. **Save** (`image_saver.py`) — exports processed images as PNGs to a `Processed Images/` subfolder

### Two Camera Type Paths

The app branches on `CameraType` (enum in `video_data.py`) throughout processing and display:

- **Foveon** (`CameraType.FOVEON`): 3-channel stacked sensor. Channel extraction via `raw[::2, ::2, ch]`. TOP=ch2, MIDDLE=ch1, BOTTOM=ch0. COLOR composite: R=bottom, G=middle, B=top. 14-bit effective, 2 zero LSBs (divide raw by 4).
- **GSense** (`CameraType.GSENSE`): Monochrome dual-gain sensor. Left half = high-gain (HG), right half = low-gain (LG). 2×2 Bayer-like pattern with configurable RGB+NIR assignment. 12-bit effective, 4 zero LSBs (divide raw by 16). Pixel offset and color permutation are user-adjustable via combo boxes.

### Key Widgets

- `ImagePanel` — aspect-ratio-preserving image display with polygon ROI selection (click vertices, right-click to close). Emits `roi_selected` signal with pixel coordinates.
- `HSVScatterDialog` — matplotlib-based Hue vs. Saturation scatter plot supporting multiple ROIs with mean/std ellipses. Can accumulate data across files when "Hold Data" is checked.
- `ROCDialog` — ROC curve computation with multiple smoothing modes (Empirical, Linear, Spline, KDE, LOWESS, Savitzky-Golay, Bootstrap, Bezier). Operates on per-file UV threshold scores.
- `ThresholdPanel` — compact Lo/Hi percentage sliders for per-channel threshold clamping.

### Key State in `PolarViewMainWindow`

- `_camera_type` — determines which processing pipeline and UI layout to use
- `_h5_files` / `_h5_file_index` — file navigation within a directory
- `_held_hsv_groups` — accumulated HSV data across multiple Show All HSV calls (when Hold Data is checked)
- `_roi_save_mode` — when active, ROI selections trigger HSV scatter computation

### Processing Details

- **Median filter**: Applied ONLY to sub-images, NOT to COLOR composite; uses `scipy.ndimage.median_filter` with `mode='constant'`
- **Spike filter**: Outlier removal using `uniform_filter` to detect pixels deviating from local mean
- **Jet colormap** (`colormap.py`): Piecewise-linear implementation matching MATLAB's manual jet computation
- **Scrub**: Removes `/camera/timestamp` dataset and root `time-info` attribute; resets file timestamp to Jan 1, 2000
- **Rename format**: `Subject_<num>_Sample_<num>_<tissue>_<descriptor>.h5` where tissue is LN/TUMOR, descriptor is COLOR/COLOR_NIR/NIR/UV

### HDF5 File Format

Expected structure: `/camera/frames` dataset. Foveon shape `[channels, cols, rows, frames]` (permuted on load). GSense shape `[frames, rows, cols, 1]`. Camera type detected from `/camera` group attributes. Optional: `/camera/integration-time` for frame rate calculation.

## Important Conventions

- Window title: "Foveon Perovskite Camera (PNC)"
- Play/pause: max 20 fps (50ms QTimer), supports loop and auto-advance to next file
- The app icon is a mantis shrimp SVG (multi-spectral vision analogy)
- GSense demosaic: NIR is fixed at position (1,1); the 3 RGB positions cycle through 6 permutations
- All image buffers use float64 numpy arrays; jet-mapped outputs are shape (H, W, 3) with values in [0, 255]
