"""Data model classes replacing MATLAB's global VIDEO_DATA struct and app properties."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class H5Attributes:
    """Metadata extracted from the HDF5 file (mirrors app.H5Info.Attr)."""

    rows: int = 0
    columns: int = 0
    num_frames: int = 0
    frame_rate: float = 10.0
    camera: str = ""


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

    def allocate(self, rows: int, cols: int) -> None:
        """Allocate all working buffers.  Matches ``LPSP_UV_AllocateMemoryInCPU.m``.

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
