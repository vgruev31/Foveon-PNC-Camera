"""PNG image export – replaces the single-file mode of ``LPSP_UV_Save_Images.m``."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image

from .video_data import H5File, VideoData


def _make_unique_filename(
    output_dir: Path, stem: str, label: str, ext: str = "png"
) -> Path:
    """Generate a unique file path, appending ``_N`` if the base name exists."""
    candidate = output_dir / f"{stem} {label}.{ext}"
    if not candidate.exists():
        return candidate
    for n in range(1, 1001):
        candidate = output_dir / f"{stem}_{n} {label}.{ext}"
        if not candidate.exists():
            return candidate
    return output_dir / f"{stem}_{label}_{random.randint(0, 10**9)}.{ext}"


def save_images(video_data: VideoData, h5file: H5File) -> list[Path]:
    """Save COLOR, TOP, MIDDLE, BOTTOM as PNGs.

    Images are written to a ``Processed Images/`` sub-folder next to the
    original H5 file, matching the MATLAB behaviour.

    MATLAB ``imwrite(abs(double_array), path)`` clips doubles to [0, 1]
    then scales to [0, 255].  We replicate that for all four images.

    Returns
    -------
    list[Path]
        Paths of the saved files.
    """
    output_dir = Path(h5file.path) / "Processed Images"
    output_dir.mkdir(exist_ok=True)

    stem = h5file.name

    # TOP/MIDDLE/BOTTOM are already [0, 255] from the jet colormap.
    # COLOR is [0, 1] and needs scaling to [0, 255].
    jet_images = {
        "TOP Image": video_data.top,
        "MIDDLE Image": video_data.middle,
        "BOTTOM Image": video_data.bottom,
    }
    color_images = {
        "COLOR Image": video_data.color,
    }

    saved: list[Path] = []

    for label, arr in jet_images.items():
        out_path = _make_unique_filename(output_dir, stem, label)
        img = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(img, mode="RGB").save(str(out_path))
        saved.append(out_path)

    for label, arr in color_images.items():
        out_path = _make_unique_filename(output_dir, stem, label)
        img = np.clip(arr, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)
        Image.fromarray(img, mode="RGB").save(str(out_path))
        saved.append(out_path)

    return saved
