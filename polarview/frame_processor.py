"""Frame processing pipeline – replaces ``LPSP_UV_ProcessSingleFrame.m``."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter

from .colormap import apply_jet_colormap
from .video_data import H5Info, VideoData


@dataclass
class ChannelThresholds:
    """Low/high threshold values for one channel (percentage 0-100)."""

    low: float = 0.0
    high: float = 100.0


@dataclass
class ProcessingParams:
    """All parameters needed to process a single frame."""

    frame_number: int = 0  # 0-based index
    filter_tap: int = 0  # 0 = no filtering; else median kernel size
    top_thresh: ChannelThresholds = field(default_factory=ChannelThresholds)
    middle_thresh: ChannelThresholds = field(default_factory=ChannelThresholds)
    bottom_thresh: ChannelThresholds = field(default_factory=ChannelThresholds)
    is_uv: bool = False  # True when the H5 file is a UV file
    norm_bits: int = 14  # effective bit depth for normalization


def _apply_threshold_and_colormap(
    channel: np.ndarray, low_pct: float, high_pct: float
) -> np.ndarray:
    """Threshold-clamp a single channel and apply the jet colormap.

    Parameters
    ----------
    channel : ndarray, shape (H, W), float64
        Normalised channel values (already divided by 2^15).
    low_pct, high_pct : float
        Threshold percentages (0-100).

    Returns
    -------
    ndarray, shape (H, W, 3), float64
        Jet-colormapped RGB result.
    """
    low = low_pct / 100.0
    high = high_pct / 100.0

    # Clamp below low → 0, above high → 1  (matches MATLAB line-by-line)
    t = channel.copy()
    t[t < low] = 0.0
    t[t > high] = 1.0

    # Normalise into [0, 1] within the threshold window
    denom = high - low
    if denom > 0:
        t = (t - low) / denom
    else:
        t = np.zeros_like(t)

    return apply_jet_colormap(t)


def process_single_frame(
    h5info: H5Info, video_data: VideoData, params: ProcessingParams
) -> None:
    """Process one frame in-place, updating *video_data* buffers.

    Mirrors ``LPSP_UV_ProcessSingleFrame.m`` exactly:

    1. Extract raw frame as float64.
    2. Subsample channels at ``[::2, ::2]`` (MATLAB ``1:2:end``).
    3. Normalise by ``2**norm_bits``.
    4. Optional median filter.
    5. Per-channel threshold clamping + jet colormap.
    """
    raw_frame = h5info.raw_data[:, :, :, params.frame_number].astype(np.float64)
    video_data.raw_single_frame_double[:] = raw_frame

    raw = video_data.raw_single_frame_double
    norm = 2 ** params.norm_bits

    # Channel extraction with 2×2 subsampling
    # MATLAB: TOP = channel 3 (index 2), MIDDLE = channel 2 (index 1),
    #         BOTTOM = channel 1 (index 0)
    top_tmp = raw[::2, ::2, 2] / norm
    middle_tmp = raw[::2, ::2, 1] / norm
    bottom_tmp = raw[::2, ::2, 0] / norm

    def _normalise(ch: np.ndarray) -> np.ndarray:
        lo, hi = ch.min(), ch.max()
        if hi > lo:
            return (ch - lo) / (hi - lo)
        return np.zeros_like(ch)

    # When median filter is active, normalise first (using original range),
    # then filter the normalised data for COLOR. This preserves the full
    # dynamic range and avoids washed-out results.
    if params.filter_tap > 0:
        k = params.filter_tap

        # Apply median filter to sub-images (for TOP/MIDDLE/BOTTOM jet images)
        top_tmp = median_filter(top_tmp, size=(k, k), mode="constant", cval=0.0)
        middle_tmp = median_filter(middle_tmp, size=(k, k), mode="constant", cval=0.0)
        bottom_tmp = median_filter(bottom_tmp, size=(k, k), mode="constant", cval=0.0)

        # COLOR: normalise using original range, then median filter
        video_data.color[:, :, 0] = median_filter(
            _normalise(raw[::2, ::2, 0] / norm),
            size=(k, k), mode="constant", cval=0.0,
        )
        video_data.color[:, :, 1] = median_filter(
            _normalise(raw[::2, ::2, 1] / norm),
            size=(k, k), mode="constant", cval=0.0,
        )
        video_data.color[:, :, 2] = median_filter(
            _normalise(raw[::2, ::2, 2] / norm),
            size=(k, k), mode="constant", cval=0.0,
        )
    else:
        # No filter: COLOR from unfiltered channels
        video_data.color[:, :, 0] = _normalise(bottom_tmp)   # Red
        video_data.color[:, :, 1] = _normalise(middle_tmp)    # Green
        video_data.color[:, :, 2] = _normalise(top_tmp)       # Blue

    # Threshold + jet colormap for each channel
    video_data.top[:] = _apply_threshold_and_colormap(
        top_tmp, params.top_thresh.low, params.top_thresh.high
    )
    video_data.middle[:] = _apply_threshold_and_colormap(
        middle_tmp, params.middle_thresh.low, params.middle_thresh.high
    )
    video_data.bottom[:] = _apply_threshold_and_colormap(
        bottom_tmp, params.bottom_thresh.low, params.bottom_thresh.high
    )
