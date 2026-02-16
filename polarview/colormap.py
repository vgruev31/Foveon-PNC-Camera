"""Proper jet colormap for false-color display of single-channel data.

Control points (standard jet):
    t=0.00 → dark blue  (0.0, 0.0, 0.5)
    t=0.125 → blue      (0.0, 0.0, 1.0)
    t=0.375 → cyan      (0.0, 1.0, 1.0)
    t=0.625 → yellow    (1.0, 1.0, 0.0)
    t=0.875 → red       (1.0, 0.0, 0.0)
    t=1.00 → dark red   (0.5, 0.0, 0.0)
"""

from __future__ import annotations

import numpy as np


def apply_jet_colormap(t: np.ndarray) -> np.ndarray:
    """Apply the jet colormap to a 2-D normalised array.

    Parameters
    ----------
    t : ndarray, shape (H, W)
        Values in [0, 1].

    Returns
    -------
    ndarray, shape (H, W, 3), float64
        RGB image with values in [0, 255].
    """
    t = np.clip(t, 0.0, 1.0)

    # Red channel
    r = np.where(
        t < 0.375, 0.0,
        np.where(
            t < 0.625, 4.0 * (t - 0.375),      # ramp 0→1
            np.where(
                t < 0.875, 1.0,                  # plateau
                -4.0 * (t - 0.875) + 1.0         # ramp 1→0.5
            )
        )
    )

    # Green channel
    g = np.where(
        t < 0.125, 0.0,
        np.where(
            t < 0.375, 4.0 * (t - 0.125),       # ramp 0→1
            np.where(
                t < 0.625, 1.0,                  # plateau
                np.where(
                    t < 0.875, -4.0 * (t - 0.625) + 1.0,  # ramp 1→0
                    0.0
                )
            )
        )
    )

    # Blue channel
    b = np.where(
        t < 0.125, 4.0 * t + 0.5,               # ramp 0.5→1
        np.where(
            t < 0.375, 1.0,                      # plateau
            np.where(
                t < 0.625, -4.0 * (t - 0.375) + 1.0,  # ramp 1→0
                0.0
            )
        )
    )

    result = np.stack([r, g, b], axis=-1) * 255.0
    return np.clip(result, 0.0, 255.0)
