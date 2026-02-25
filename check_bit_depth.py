"""Quick diagnostic: check effective bit depth of raw sensor data in an H5 file.

Usage:  python check_bit_depth.py <path_to_h5_file>
"""

import sys
from pathlib import Path

import h5py
import numpy as np


def check_bit_depth(file_path: str) -> None:
    p = Path(file_path)
    if not p.exists():
        print(f"File not found: {p}")
        return

    with h5py.File(p, "r") as f:
        raw = f["camera"]["frames"][()]

    print(f"File:   {p.name}")
    print(f"Shape:  {raw.shape}")
    print(f"Dtype:  {raw.dtype}")
    print(f"Min:    {raw.min()}")
    print(f"Max:    {raw.max()}")
    print()

    # Cast to integer for bitwise checks
    data = raw.astype(np.int64).ravel()

    # Check which bits are ever set across all values
    all_or = np.bitwise_or.reduce(data)
    print(f"Bitwise OR of all values: {all_or}  (binary: {all_or:016b})")
    print()

    # Check each bit position
    print("Bit usage (0 = LSB):")
    for bit in range(16):
        count = np.count_nonzero(data & (1 << bit))
        pct = 100.0 * count / len(data)
        marker = "*" if count > 0 else " "
        print(f"  bit {bit:2d}: {count:>12,} values set ({pct:6.2f}%) {marker}")

    print()

    # Summary: find lowest bit that is ever set
    lowest_set = None
    for bit in range(16):
        if np.any(data & (1 << bit)):
            lowest_set = bit
            break

    if lowest_set is not None and lowest_set > 0:
        print(f"Lowest bit ever set: bit {lowest_set}")
        print(f"  -> Data appears to be {16 - lowest_set}-bit effective,")
        print(f"     padded with {lowest_set} zero LSBs.")
        print(f"  -> Effective max = 2^{16 - lowest_set} - 1 = {(1 << (16 - lowest_set)) - 1}")
        print(f"     (shifted: {((1 << (16 - lowest_set)) - 1) << lowest_set})")
    elif lowest_set == 0:
        print("All 16 bits are used — data appears to be full 16-bit.")
    else:
        print("All values are zero.")

    # Show value distribution
    print()
    print("Value distribution (percentiles):")
    for pct in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
        val = np.percentile(data, pct)
        print(f"  {pct:3d}th percentile: {val:>8.0f}  (binary: {int(val):016b})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_bit_depth.py <path_to_h5_file>")
        sys.exit(1)
    check_bit_depth(sys.argv[1])
