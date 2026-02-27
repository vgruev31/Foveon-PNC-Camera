"""HDF5 file loading – replaces ``h5load.m`` and the load portions of
``LPSP_UV_LoadAndInitializeFromFullPath.m``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .video_data import CameraType, H5Attributes, H5Info

NUM_CHANNELS = 3  # Foveon camera always has 3 photodiodes


def _detect_camera_type(cam_group: h5py.Group) -> tuple[CameraType, str]:
    """Detect the camera type from HDF5 /camera group attributes.

    Returns (CameraType, descriptive_string).
    """
    attrs = dict(cam_group.attrs) if cam_group.attrs else {}

    # Check sensor-desc first (newer firmware)
    sensor_desc = str(attrs.get("sensor-desc", ""))
    if sensor_desc:
        if "GSense" in sensor_desc or "GPIXEL" in sensor_desc:
            return CameraType.GSENSE, sensor_desc
        if "Foveon" in sensor_desc or "F13" in sensor_desc:
            return CameraType.FOVEON, sensor_desc

    # Check imager attribute (older firmware)
    imager = str(attrs.get("imager", ""))
    if imager:
        if "X3" in imager or "Stacked" in imager:
            return CameraType.FOVEON, imager

    # Check model-name
    model = str(attrs.get("model-name", ""))
    if model:
        if "GSense" in model:
            return CameraType.GSENSE, model
        if "Foveon" in model:
            return CameraType.FOVEON, model

    # Check manufacture
    manufacture = str(attrs.get("manufacture", ""))
    if "GPIXEL" in manufacture:
        return CameraType.GSENSE, manufacture
    if "Foveon" in manufacture:
        return CameraType.FOVEON, manufacture

    return CameraType.UNKNOWN, sensor_desc or imager or model or "unknown"


def _reorder_foveon(raw: np.ndarray) -> np.ndarray:
    """Rearrange Foveon h5py data to ``(rows, cols, 3_channels, frames)``.

    The camera has exactly 3 colour channels (photodiodes).  The logic:

    * **3-D data** ``(A, B, C)``: one of the dims is 3 (channels), the
      other two are spatial.  Result is ``(rows, cols, 3)`` — single frame,
      expanded to 4-D by appending a frames axis of size 1.
    * **4-D data** ``(A, B, C, D)``: two dims are large (spatial), one is
      3 (channels), one is the frame count.  Result is
      ``(rows, cols, 3, frames)``.

    If two dims are both 3, the one that matches known channel count is
    treated as channels; the other as frames.
    """
    print(f"[h5_loader] Foveon raw shape: {raw.shape}, ndim={raw.ndim}")

    # ---- 3-D: single frame with 3 channels ----
    if raw.ndim == 3:
        shape = raw.shape
        dims_of_3 = [i for i, s in enumerate(shape) if s == NUM_CHANNELS]
        if not dims_of_3:
            raise ValueError(
                f"3-D data has no dimension of size {NUM_CHANNELS}: {shape}"
            )
        ch_axis = dims_of_3[0]
        spatial = [i for i in range(3) if i != ch_axis]
        # Larger spatial dim first (rows)
        if shape[spatial[0]] < shape[spatial[1]]:
            spatial[0], spatial[1] = spatial[1], spatial[0]
        raw = np.transpose(raw, [spatial[0], spatial[1], ch_axis])
        # Add a frames axis → (rows, cols, 3, 1)
        raw = raw[:, :, :, np.newaxis]
        print(f"[h5_loader] 3-D → expanded to 4-D: {raw.shape}")
        return raw

    # ---- 4-D ----
    if raw.ndim == 4:
        shape = raw.shape
        dims_of_3 = [i for i, s in enumerate(shape) if s == NUM_CHANNELS]

        if len(dims_of_3) == 0:
            raise ValueError(
                f"4-D data has no dimension of size {NUM_CHANNELS}: {shape}"
            )

        if len(dims_of_3) == 1:
            ch_axis = dims_of_3[0]
        else:
            # Multiple dims of size 3 — pick the one that leaves the
            # other as a plausible frame count (also 3 in this case).
            # Prefer the LAST dim-of-3 as channels (common storage order
            # is frames-first or spatial-first with channels last).
            ch_axis = dims_of_3[-1]

        remaining = [i for i in range(4) if i != ch_axis]
        remaining_sizes = [shape[i] for i in remaining]

        # Smallest remaining dim = frames
        order = sorted(range(3), key=lambda k: remaining_sizes[k])
        frame_axis = remaining[order[0]]
        spatial = sorted(
            [remaining[order[1]], remaining[order[2]]],
            key=lambda i: shape[i],
            reverse=True,
        )

        perm = [spatial[0], spatial[1], ch_axis, frame_axis]
        raw = np.transpose(raw, perm)

        # Final safety check: axis 2 must be 3 (channels).
        # If it ended up as axis 3 instead, swap.
        if raw.shape[2] != NUM_CHANNELS and raw.shape[3] == NUM_CHANNELS:
            print("[h5_loader] Swapping axes 2↔3 to fix channel/frame order")
            raw = np.swapaxes(raw, 2, 3)

        print(f"[h5_loader] 4-D reordered: {raw.shape}  "
              f"(rows={raw.shape[0]}, cols={raw.shape[1]}, "
              f"ch={raw.shape[2]}, frames={raw.shape[3]})")
        return raw

    raise ValueError(f"Expected 3-D or 4-D data, got {raw.ndim}-D")


def load_h5(file_path: str | Path) -> H5Info:
    """Load an HDF5 file and return an :class:`H5Info`.

    Reads ``/camera/frames``, detects camera type from metadata,
    and reorders/normalises accordingly.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.suffix.lower() != ".h5":
        raise ValueError(f"Not an .h5 file: {file_path}")

    info = H5Info()

    with h5py.File(file_path, "r") as f:
        if "camera" not in f or "frames" not in f["camera"]:
            raise ValueError(
                f"Unsupported H5 format: expected /camera/frames. "
                f"File: {file_path}"
            )

        cam_group = f["camera"]

        # --- Detect camera type ---
        camera_type, camera_desc = _detect_camera_type(cam_group)
        print(f"[h5_loader] Camera type: {camera_type.name}  ({camera_desc})")

        # --- Load and process frames ---
        raw: np.ndarray = cam_group["frames"][()]

        if camera_type == CameraType.FOVEON:
            raw = _reorder_foveon(raw)
            # Foveon: 14-bit effective, 2 zero LSBs
            info.raw_data = raw // 4
            bit_shift = 2
            norm_bits = 14
        elif camera_type == CameraType.GSENSE:
            # GSense: shape from file is (frames, rows, cols, 1)
            # Transpose to (rows, cols, 1, frames) for consistency
            raw = np.transpose(raw, (1, 2, 3, 0))
            # 12-bit effective, 4 zero LSBs
            info.raw_data = raw // 16
            bit_shift = 4
            norm_bits = 12
            print(f"[h5_loader] GSense reordered: {info.raw_data.shape}  "
                  f"(rows={info.raw_data.shape[0]}, cols={info.raw_data.shape[1]}, "
                  f"ch={info.raw_data.shape[2]}, frames={info.raw_data.shape[3]})")
        else:
            # Unknown camera — try Foveon-style reorder as fallback
            print(f"[h5_loader] Unknown camera, attempting Foveon-style reorder")
            raw = _reorder_foveon(raw)
            info.raw_data = raw // 4
            bit_shift = 2
            norm_bits = 14

        # --- Frame rate from integration-time ---
        frame_rate = 10.0
        try:
            if "integration-time" in cam_group:
                integ_us = cam_group["integration-time"][()].astype(np.float64)
                if integ_us.size > 0:
                    mean_integ = float(np.mean(integ_us))
                    if mean_integ > 0:
                        frame_rate = 1.0 / (mean_integ / 1e6)
        except Exception:
            pass

        if not np.isfinite(frame_rate) or frame_rate <= 0:
            frame_rate = 30.0

        # --- Build attributes ---
        info.attr = H5Attributes(
            rows=info.raw_data.shape[0],
            columns=info.raw_data.shape[1],
            num_frames=info.raw_data.shape[-1] if info.raw_data.ndim == 4 else 1,
            frame_rate=frame_rate,
            camera=camera_desc,
            camera_type=camera_type,
            bit_shift=bit_shift,
            norm_bits=norm_bits,
        )

    return info
