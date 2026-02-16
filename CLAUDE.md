# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Foveon Perovskite Camera (PNC) is a PyQt6 desktop application for viewing and processing multi-spectral polarization camera data from Foveon/Perovskite sensors. It loads HDF5 (`.h5`) video files containing raw sensor frames with 3 channels (TOP/MIDDLE/BOTTOM), processes them with jet colormaps, and exports PNGs. This is a Python port of the original MATLAB PolarView application.

## Running the Application

```bash
pip install -r requirements.txt
python run.py
```

Or as a module:
```bash
python -m polarview
```

### Dependencies

Listed in `requirements.txt`: PyQt6, numpy, scipy, h5py.

## Architecture

### Directory Structure

```
├── run.py                          # Entry point
├── requirements.txt                # Python dependencies
├── CLAUDE.md                       # This file
├── polarview/
│   ├── __init__.py
│   ├── __main__.py                 # Module entry point
│   ├── main_window.py              # Main GUI window (QMainWindow)
│   ├── frame_processor.py          # Frame processing pipeline
│   ├── h5_loader.py                # HDF5 file loading
│   ├── colormap.py                 # Piecewise-linear jet colormap
│   ├── video_data.py               # Data model classes
│   ├── image_saver.py              # PNG export
│   ├── assets/
│   │   └── mantis_shrimp.svg       # App icon
│   └── widgets/
│       ├── __init__.py
│       ├── frame_slider.py         # Frame slider with play/pause
│       ├── image_panel.py          # Aspect-ratio image display
│       └── threshold_panel.py      # Lo/Hi threshold sliders
```

### Data Flow Pipeline

1. **File Open** — user picks H5 file via `QFileDialog`, or navigates with Previous/Next buttons
2. **Load** (`h5_loader.py`) — reads `/camera/frames` dataset from HDF5, permutes to `[rows, cols, channels, frames]`, extracts attributes (FrameRate, Rows, Columns, NumberOfFrames)
3. **Frame Processing** (`frame_processor.py`) — extracts 3 channels using 2x2 subsampling (`[::2, ::2]`), normalizes by dividing by `2^15`, builds COLOR composite from unfiltered channels, applies optional median filter (only to 3 sub-images, NOT COLOR), applies per-channel low/high threshold clamping, computes jet colormap
4. **Display** (`main_window.py`) — renders TOP, MIDDLE, BOTTOM (jet-mapped), and COLOR images in a single horizontal row via `ImagePanel` widgets
5. **Save** (`image_saver.py`) — exports processed images as PNGs to a `Processed Images/` subfolder

### GUI Layout (top to bottom)

1. **Open Input File** row — file picker + filename display
2. **Scrub Files** row — button to strip time metadata from all H5 files in a directory
3. **Navigation** row — Previous/Next/Delete buttons (left), Median Filter + Save (right)
4. **Rename** row — structured rename: `Subject_<num>_<tissue>_<descriptor>.h5`
5. **Frame Slider** row — slider (maxWidth=300) + play/pause button (64x64) + Loop Video / Play Next File radio buttons
6. **Images** row — 4 images in a horizontal strip: TOP, MIDDLE, BOTTOM, COLOR
7. **Thresholds** row — 4 compact threshold panels (Lo/Hi sliders) for TOP (high=50), MIDDLE (high=50), BOTTOM (high=60), COLOR (high=100)

### Key State

- `VideoData` — holds all image buffers: `raw_single_frame`, `top`/`middle`/`bottom` (jet-mapped), `color` (RGB composite)
- `H5Info` — loaded data with `.data_raw` (raw frames array) and attributes
- `H5File` — stem name and directory path of loaded file
- `_h5_files: list[Path]` / `_h5_file_index: int` — file navigation state

### HDF5 File Format

Expected structure: `/camera/frames` dataset with shape `[channels, cols, rows, frames]` (permuted on load to `[rows, cols, channels, frames]`). Optional: `/camera/integration-time` for frame rate calculation.

### Processing Details

- **Channel extraction**: TOP = channel 2, MIDDLE = channel 1, BOTTOM = channel 0 (via `raw[::2, ::2, ch]`)
- **COLOR composite**: Built from UNFILTERED channels — Red=bottom, Green=middle, Blue=top (each per-channel normalized)
- **Median filter**: Applied ONLY to the 3 sub-images, NOT to COLOR; uses `scipy.ndimage.median_filter` with `mode='constant'`
- **Jet colormap**: Piecewise-linear implementation matching MATLAB's manual jet computation
- **Scrub**: Removes `/camera/timestamp` dataset and root `time-info` attribute; resets file timestamp to Jan 1, 2000

## Important Notes

- The app icon is a cartoon mantis shrimp SVG (mantis shrimp can see many spectral bands, fitting for a multi-spectral camera app)
- Window title: "Foveon Perovskite Camera (PNC)"
- Play/pause: max 20 fps (50ms QTimer), supports loop and auto-advance to next file
- Rename format: `Subject_<num>_<tissue>_<descriptor>.h5` where tissue is LN/TUMOR, descriptor is COLOR/COLOR_NIR/NIR/UV
