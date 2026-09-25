"""Frame processing pipeline – replaces ``LPSP_UV_ProcessSingleFrame.m``."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter, uniform_filter

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
    spike_tap: int = 0   # 0 = no spike filter; else kernel size for outlier removal
    top_thresh: ChannelThresholds = field(default_factory=ChannelThresholds)
    middle_thresh: ChannelThresholds = field(default_factory=ChannelThresholds)
    bottom_thresh: ChannelThresholds = field(default_factory=ChannelThresholds)
    is_uv: bool = False  # True when the H5 file is a UV file
    norm_bits: int = 14  # effective bit depth for normalization


@dataclass
class GsenseProcessingParams:
    """Parameters for processing a single GSense dual-gain frame."""

    frame_number: int = 0
    filter_tap: int = 0
    spike_tap: int = 0   # 0 = no spike filter; else kernel size for outlier removal
    hg_nir_thresh: ChannelThresholds = field(default_factory=ChannelThresholds)
    lg_nir_thresh: ChannelThresholds = field(default_factory=ChannelThresholds)
    hg_nir_jet: bool = False  # True = jet colormap, False = grayscale
    lg_nir_jet: bool = False
    norm_bits: int = 12
    pixel_offset: tuple[int, int] = (0, 0)  # (row_offset, col_offset)
    color_perm: tuple[str, str, str, str] = ('B', 'R', 'G', 'N')  # assignment for (0,0)·(0,1)·(1,0)·(1,1)


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


def _apply_threshold_grayscale(
    channel: np.ndarray, low_pct: float, high_pct: float
) -> np.ndarray:
    """Threshold-clamp a single channel and output grayscale RGB.

    Same contract as :func:`_apply_threshold_and_colormap` but produces
    a monochrome image (R = G = B) instead of a jet colormap.

    Returns
    -------
    ndarray, shape (H, W, 3), float64
        Grayscale RGB image with values in [0, 255].
    """
    low = low_pct / 100.0
    high = high_pct / 100.0

    t = channel.copy()
    t[t < low] = 0.0
    t[t > high] = 1.0

    denom = high - low
    if denom > 0:
        t = (t - low) / denom
    else:
        t = np.zeros_like(t)

    gray = np.clip(t, 0.0, 1.0) * 255.0
    return np.stack([gray, gray, gray], axis=-1)


def _spike_filter(channel: np.ndarray, kernel_size: int) -> np.ndarray:
    """Remove salt-and-pepper spikes using local statistics.

    Uses fast uniform (box) filters to compute local mean and standard
    deviation, then replaces only those pixels that deviate by more than
    2 sigma from their neighbourhood mean.  Much faster than a full
    median filter and preserves non-noisy pixels exactly.
    """
    k = kernel_size
    local_mean = uniform_filter(channel, size=k, mode="reflect")
    local_sq_mean = uniform_filter(channel ** 2, size=k, mode="reflect")
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)
    local_std = np.sqrt(local_var)

    deviation = np.abs(channel - local_mean)
    is_spike = deviation > 2.0 * np.maximum(local_std, 1e-8)

    result = channel.copy()
    result[is_spike] = local_mean[is_spike]
    return result


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

    # Capture per-channel normalisation range from the ORIGINAL unfiltered
    # channels.  Filters remove bright spikes → channel max drops → if we
    # auto-level against the post-filter max, the remaining pixels appear
    # brighter.  Using the original range keeps intensities stable.
    top_lo, top_hi = float(top_tmp.min()), float(top_tmp.max())
    middle_lo, middle_hi = float(middle_tmp.min()), float(middle_tmp.max())
    bottom_lo, bottom_hi = float(bottom_tmp.min()), float(bottom_tmp.max())

    # Apply filters per channel (spike first, then median)
    if params.spike_tap > 0:
        sk = params.spike_tap
        top_tmp = _spike_filter(top_tmp, sk)
        middle_tmp = _spike_filter(middle_tmp, sk)
        bottom_tmp = _spike_filter(bottom_tmp, sk)

    if params.filter_tap > 0:
        k = params.filter_tap
        top_tmp = median_filter(top_tmp, size=(k, k), mode="reflect")
        middle_tmp = median_filter(middle_tmp, size=(k, k), mode="reflect")
        bottom_tmp = median_filter(bottom_tmp, size=(k, k), mode="reflect")

    def _normalise_range(ch: np.ndarray, lo: float, hi: float) -> np.ndarray:
        if hi > lo:
            return np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
        return np.zeros_like(ch)

    # Build COLOR composite using ORIGINAL per-channel ranges — applying
    # filters before joining the channels, with intensity preserved.
    video_data.color[:, :, 0] = _normalise_range(bottom_tmp, bottom_lo, bottom_hi)  # R
    video_data.color[:, :, 1] = _normalise_range(middle_tmp, middle_lo, middle_hi)  # G
    video_data.color[:, :, 2] = _normalise_range(top_tmp, top_lo, top_hi)           # B

    # Threshold + jet colormap for each channel (uses filtered data directly)
    video_data.top[:] = _apply_threshold_and_colormap(
        top_tmp, params.top_thresh.low, params.top_thresh.high
    )
    video_data.middle[:] = _apply_threshold_and_colormap(
        middle_tmp, params.middle_thresh.low, params.middle_thresh.high
    )
    video_data.bottom[:] = _apply_threshold_and_colormap(
        bottom_tmp, params.bottom_thresh.low, params.bottom_thresh.high
    )


def process_gsense_frame(
    h5info: H5Info, video_data: VideoData, params: GsenseProcessingParams
) -> None:
    """Process one GSense dual-gain frame with two-step subsampling.

    1. Extract raw frame (rows × cols_full) as float64.
    2. Split into High-Gain (left half) and Low-Gain (right half).
    3. Normalise by ``2**norm_bits``.
    4. Subsample each half by 2 with user-chosen pixel offset
       (removes filter bleed): 2048×2048 → 1024×1024.
    5. Demosaic 2×2 super-pixel pattern using configurable permutation:
       1024×1024 → 512×512 per channel (R, G, B, NIR).
    6. Optional median filter on individual sub-channels.
    7. Build per-channel-normalised color composites [0,1].
    8. NIR threshold clamping + jet/grayscale colormap.
    """
    # raw_data shape: (rows, cols_full, 1, frames)
    raw_frame = h5info.raw_data[:, :, 0, params.frame_number].astype(np.float64)
    norm = 2 ** params.norm_bits
    cols_half = raw_frame.shape[1] // 2

    hg_full = raw_frame[:, :cols_half] / norm
    lg_full = raw_frame[:, cols_half:] / norm

    offset_y, offset_x = params.pixel_offset
    perm = params.color_perm  # e.g. ('N', 'G', 'B', 'R') for (0,0)·(0,1)·(1,0)·(1,1)

    def _demosaic(half: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract R, G, B, NIR sub-images via two-step subsampling.

        The colour filters are physically 2× the pixel pitch.  Each
        filter centre covers one pixel and bleeds into neighbours.

        Step 1 — Subsample by 2 with user-chosen *offset* to select
        the centre pixel of each filter (removes bleed).
        2048×2048 → 1024×1024.

        Step 2 — Demosaic the 2×2 super-pixel pattern using *perm*.
        1024×1024 → 512×512 per channel.
        """
        # Step 1: subsample by 2 starting at (offset_y, offset_x)
        subsampled = half[offset_y::2, offset_x::2]

        # Step 2: demosaic 2×2 super-pixel pattern
        # Ensure even dimensions so all 4 sub-images have equal size
        rows_s = (subsampled.shape[0] // 2) * 2
        cols_s = (subsampled.shape[1] // 2) * 2
        subsampled = subsampled[:rows_s, :cols_s]

        positions = [
            subsampled[0::2, 0::2],  # (0,0)
            subsampled[0::2, 1::2],  # (0,1)
            subsampled[1::2, 0::2],  # (1,0)
            subsampled[1::2, 1::2],  # (1,1)
        ]
        ch = dict(zip(perm, positions))
        return ch['R'], ch['G'], ch['B'], ch['N']

    hg_r, hg_g, hg_b, hg_nir = _demosaic(hg_full)
    lg_r, lg_g, lg_b, lg_nir = _demosaic(lg_full)

    # Capture per-channel normalisation range from ORIGINAL unfiltered
    # sub-channels so filtering doesn't shift apparent intensity.
    def _range(ch: np.ndarray) -> tuple[float, float]:
        return float(ch.min()), float(ch.max())

    hg_r_lo, hg_r_hi = _range(hg_r)
    hg_g_lo, hg_g_hi = _range(hg_g)
    hg_b_lo, hg_b_hi = _range(hg_b)
    hg_nir_lo, hg_nir_hi = _range(hg_nir)
    lg_r_lo, lg_r_hi = _range(lg_r)
    lg_g_lo, lg_g_hi = _range(lg_g)
    lg_b_lo, lg_b_hi = _range(lg_b)
    lg_nir_lo, lg_nir_hi = _range(lg_nir)

    # Optional spike filter on individual sub-channels (runs before median)
    if params.spike_tap > 0:
        sk = params.spike_tap
        hg_r = _spike_filter(hg_r, sk)
        hg_g = _spike_filter(hg_g, sk)
        hg_b = _spike_filter(hg_b, sk)
        hg_nir = _spike_filter(hg_nir, sk)
        lg_r = _spike_filter(lg_r, sk)
        lg_g = _spike_filter(lg_g, sk)
        lg_b = _spike_filter(lg_b, sk)
        lg_nir = _spike_filter(lg_nir, sk)

    # Optional median filter on individual sub-channels
    if params.filter_tap > 0:
        k = params.filter_tap
        hg_r = median_filter(hg_r, size=(k, k), mode="reflect")
        hg_g = median_filter(hg_g, size=(k, k), mode="reflect")
        hg_b = median_filter(hg_b, size=(k, k), mode="reflect")
        hg_nir = median_filter(hg_nir, size=(k, k), mode="reflect")
        lg_r = median_filter(lg_r, size=(k, k), mode="reflect")
        lg_g = median_filter(lg_g, size=(k, k), mode="reflect")
        lg_b = median_filter(lg_b, size=(k, k), mode="reflect")
        lg_nir = median_filter(lg_nir, size=(k, k), mode="reflect")

    # Cache the post-filter sub-channels in their raw / 2^norm_bits scale
    # so HDR fusion can combine HG and LG on a common physical scale.
    video_data.hg_rgbn_filtered = np.stack([hg_r, hg_g, hg_b, hg_nir], axis=-1)
    video_data.lg_rgbn_filtered = np.stack([lg_r, lg_g, lg_b, lg_nir], axis=-1)

    def _normalise_range(ch: np.ndarray, lo: float, hi: float) -> np.ndarray:
        if hi > lo:
            return np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
        return np.zeros_like(ch)

    # Build per-channel-normalised color composites [0, 1] using the
    # ORIGINAL per-channel ranges (filters applied per channel before
    # joining, with intensity preserved).
    video_data.hg_color_raw = np.stack(
        [_normalise_range(hg_r, hg_r_lo, hg_r_hi),
         _normalise_range(hg_g, hg_g_lo, hg_g_hi),
         _normalise_range(hg_b, hg_b_lo, hg_b_hi)],
        axis=-1,
    )
    video_data.lg_color_raw = np.stack(
        [_normalise_range(lg_r, lg_r_lo, lg_r_hi),
         _normalise_range(lg_g, lg_g_lo, lg_g_hi),
         _normalise_range(lg_b, lg_b_lo, lg_b_hi)],
        axis=-1,
    )

    # NIR raw (single channel, normalised to [0, 1] using original range)
    video_data.hg_nir_raw = _normalise_range(hg_nir, hg_nir_lo, hg_nir_hi)
    video_data.lg_nir_raw = _normalise_range(lg_nir, lg_nir_lo, lg_nir_hi)

    # Apply threshold + colormap to NIR channels → [0, 255]
    apply_hg_nir = _apply_threshold_and_colormap if params.hg_nir_jet else _apply_threshold_grayscale
    apply_lg_nir = _apply_threshold_and_colormap if params.lg_nir_jet else _apply_threshold_grayscale

    video_data.hg_nir = apply_hg_nir(
        video_data.hg_nir_raw, params.hg_nir_thresh.low, params.hg_nir_thresh.high
    )
    video_data.lg_nir = apply_lg_nir(
        video_data.lg_nir_raw, params.lg_nir_thresh.low, params.lg_nir_thresh.high
    )


def hdr_fuse(
    hg: np.ndarray, lg: np.ndarray, sat_thresh: float = 0.95
) -> np.ndarray:
    """Fuse a HG and LG channel into one extended-dynamic-range array.

    Where ``hg < sat_thresh`` (HG is unsaturated), the HG value is used.
    Where HG is saturated, ``lg`` scaled by an estimated per-channel gain
    ratio is used instead.  Both inputs must already be in the same
    physical scale (raw / 2^norm_bits).
    """
    mask = (hg < sat_thresh) & (lg > 5e-3)
    if int(mask.sum()) > 100:
        ratio = float(np.median(hg[mask] / lg[mask]))
    else:
        ratio = 1.0
    fused = hg.copy()
    saturated = hg >= sat_thresh
    fused[saturated] = lg[saturated] * ratio
    return fused


def compute_hdr_displays(
    video_data: VideoData, jet_nir: bool = False, sat_thresh: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Build display-ready HDR Color and HDR NIR images from HG+LG fusion.

    Both outputs are float64 RGB ``(H, W, 3)`` arrays in [0, 255].  Color
    uses per-channel auto-levelling so each fused channel fills its own
    range; NIR uses a global auto-level then either grayscale or jet
    colormap depending on ``jet_nir``.
    """
    hg = video_data.hg_rgbn_filtered
    lg = video_data.lg_rgbn_filtered
    if hg is None or lg is None:
        raise ValueError("HDR buffers not populated — call process_gsense_frame first.")

    # Per-channel HDR fusion (R, G, B, NIR)
    fused = np.empty_like(hg)
    for c in range(4):
        fused[..., c] = hdr_fuse(hg[..., c], lg[..., c], sat_thresh)

    # Color: per-channel auto-level → [0, 1] → [0, 255]
    color_disp = np.empty_like(fused[..., :3])
    for c in range(3):
        ch = fused[..., c]
        lo, hi = float(ch.min()), float(ch.max())
        if hi > lo:
            color_disp[..., c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
        else:
            color_disp[..., c] = 0.0
    color_img = color_disp * 255.0

    # NIR: auto-level then grayscale or jet
    nir = fused[..., 3]
    lo, hi = float(nir.min()), float(nir.max())
    if hi > lo:
        nir_norm = np.clip((nir - lo) / (hi - lo), 0.0, 1.0)
    else:
        nir_norm = np.zeros_like(nir)

    if jet_nir:
        nir_img = apply_jet_colormap(nir_norm)
    else:
        gray = nir_norm * 255.0
        nir_img = np.stack([gray, gray, gray], axis=-1)

    return color_img, nir_img
