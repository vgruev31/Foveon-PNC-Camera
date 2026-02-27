"""Data model classes replacing MATLAB's global VIDEO_DATA struct and app properties."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


class CameraType(Enum):
    """Supported camera sensor types."""

    FOVEON = auto()  # Foveon stacked RGB (3 channels)
    GSENSE = auto()  # GSense FSI Dual-Gain (monochrome, HG+LG)
    UNKNOWN = auto()


@dataclass
class H5Attributes:
    """Metadata extracted from the HDF5 file (mirrors app.H5Info.Attr)."""

    rows: int = 0
    columns: int = 0
    num_frames: int = 0
    frame_rate: float = 10.0
    camera: str = ""
    camera_type: CameraType = CameraType.UNKNOWN
    bit_shift: int = 0     # number of zero LSBs to strip (divide by 2^bit_shift)
    norm_bits: int = 14    # effective bit depth for normalization (divide by 2^norm_bits)


@dataclass
class H5Info:
    """Loaded HDF5 data and attributes (mirrors app.H5Info)."""

    raw_data: np.ndarray | None = None  # shape: (rows, cols, channels, frames)
    attr: H5Attributes = field(default_factory=H5Attributes)


@dataclass
class H5File:
    """File identity bookkeeping (mirrors app.H5File)."""

    name: str = ""  # Stem without extension
    path: str = ""  # Parent directory


@dataclass
class VideoData:
    """Working image buffers, replacing the MATLAB ``global VIDEO_DATA`` struct.

    Call :meth:`allocate` after loading an H5 file to size the arrays correctly.
    """

    raw_single_frame_double: np.ndarray | None = None  # (rows, cols, 3) float64
    top_tmp: np.ndarray | None = None  # (rows_half, cols_half) float64
    top: np.ndarray | None = None  # (rows_half, cols_half, 3) float64
    middle_tmp: np.ndarray | None = None
    middle: np.ndarray | None = None
    bottom_tmp: np.ndarray | None = None
    bottom: np.ndarray | None = None
    color: np.ndarray | None = None  # (rows_half, cols_half, 3) float64

    # GSense demosaiced buffers (quarter resolution per channel)
    hg_color_raw: np.ndarray | None = None  # (rows_q, cols_q, 3) float64 [0,1]
    hg_nir_raw: np.ndarray | None = None    # (rows_q, cols_q) float64 [0,1]
    hg_color: np.ndarray | None = None      # (rows_q, cols_q, 3) float64 [0,255]
    hg_nir: np.ndarray | None = None        # (rows_q, cols_q, 3) float64 [0,255]
    lg_color_raw: np.ndarray | None = None
    lg_nir_raw: np.ndarray | None = None
    lg_color: np.ndarray | None = None
    lg_nir: np.ndarray | None = None

    def allocate(self, rows: int, cols: int) -> None:
        """Allocate Foveon working buffers.  Matches ``LPSP_UV_AllocateMemoryInCPU.m``.

        MATLAB ``round(Rows/2)`` on an integer N equals ``(N + 1) // 2``.
        """
        rows_half = (rows + 1) // 2
        cols_half = (cols + 1) // 2

        self.raw_single_frame_double = np.zeros((rows, cols, 3), dtype=np.float64)
        self.top_tmp = np.zeros((rows_half, cols_half), dtype=np.float64)
        self.top = np.zeros((rows_half, cols_half, 3), dtype=np.float64)
        self.middle_tmp = np.zeros((rows_half, cols_half), dtype=np.float64)
        self.middle = np.zeros((rows_half, cols_half, 3), dtype=np.float64)
        self.bottom_tmp = np.zeros((rows_half, cols_half), dtype=np.float64)
        self.bottom = np.zeros((rows_half, cols_half, 3), dtype=np.float64)
        self.color = np.zeros((rows_half, cols_half, 3), dtype=np.float64)

    def allocate_gsense(self, rows: int, cols: int) -> None:
        """Allocate GSense demosaiced buffers.

        *cols* is the full sensor width (e.g. 4096).  Each gain half is
        ``cols // 2`` wide.  Two-step subsampling (×2 for filter-bleed
        removal + ×2 for 2×2 demosaic) divides each dimension by 4,
        giving ``(rows // 4, cols // 8)`` images per channel.

        Example: 2048×4096 → HG/LG 2048×2048 → subsample 1024×1024
        → demosaic 512×512.
        """
        rows_q = rows // 4
        cols_q = cols // 8  # cols / 2 (gain split) / 2 (subsample) / 2 (demosaic)
        self.hg_color_raw = np.zeros((rows_q, cols_q, 3), dtype=np.float64)
        self.hg_nir_raw = np.zeros((rows_q, cols_q), dtype=np.float64)
        self.hg_color = np.zeros((rows_q, cols_q, 3), dtype=np.float64)
        self.hg_nir = np.zeros((rows_q, cols_q, 3), dtype=np.float64)
        self.lg_color_raw = np.zeros((rows_q, cols_q, 3), dtype=np.float64)
        self.lg_nir_raw = np.zeros((rows_q, cols_q), dtype=np.float64)
        self.lg_color = np.zeros((rows_q, cols_q, 3), dtype=np.float64)
        self.lg_nir = np.zeros((rows_q, cols_q, 3), dtype=np.float64)
