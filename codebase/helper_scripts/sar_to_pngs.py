#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAR ➜ PNG exporter (NO clipping, NO CLI)

- Assumes a dual‑pol SAR GeoTIFF with values in dB.
- Reads VV from band 1 and VH from band 2.
- Produces three PNGs:
  1) VV (16‑bit grayscale)
  2) VH (16‑bit grayscale)
  3) VV_over_VH (linear ratio, 16‑bit grayscale)
- No percentile stretch and no fixed display range. We scale linearly from the
  *actual* min..max of valid pixels (ignoring NaN/Inf/nodata) into **uint16** (0..65535)
  to preserve contrast without clipping.
- Invalid pixels (nodata/NaN/Inf) are set to 0 in the PNGs.

How to use:
1) Set INPUT_TIF to your SAR file path.
2) Run the script (e.g., `python sar_to_pngs_noclip.py`).
3) PNGs will be written next to the input file.

"""

import os
import numpy as np
import rasterio
from PIL import Image

# --------- USER: set your input SAR TIF here ---------
INPUT_TIF = "/home/dll0505/thesis/code/code/ROIs2017_winter_s1_102_p100.tif"  # <- edit me
# -----------------------------------------------------


def _scale_minmax_to_uint16(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Scale linearly from actual min..max (valid pixels only) to 0..65535 uint16.
    No clipping beyond the inherent 0..65535 output range. Masked pixels -> 0.
    """
    valid = arr[~mask]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.uint16)

    vmin = float(np.nanmin(valid))
    vmax = float(np.nanmax(valid))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return np.zeros_like(arr, dtype=np.uint16)

    x = (arr.astype(np.float32) - vmin) / (vmax - vmin)
    x = np.clip(x, 0.0, 1.0)
    x = np.round(x * 65535.0).astype(np.uint16)
    x = np.where(mask, 0, x)
    return x


def main():
    in_tif = os.path.abspath(INPUT_TIF)
    out_dir = os.path.dirname(in_tif)
    base = os.path.splitext(os.path.basename(in_tif))[0]

    with rasterio.open(in_tif) as src:
        if src.count < 2:
            raise ValueError("Expected at least 2 bands (VV=1, VH=2)")
        vv_db = src.read(1).astype(np.float32)
        vh_db = src.read(2).astype(np.float32)
        nodata_val = src.nodata

    # Build invalid mask from VV or VH invalids
    mask = ~np.isfinite(vv_db) | ~np.isfinite(vh_db)
    if nodata_val is not None:
        mask |= (vv_db == nodata_val) | (vh_db == nodata_val)

    # Save VV (dB) and VH (dB) as uint16 PNGs using true min..max (no clipping)
    vv_u16 = _scale_minmax_to_uint16(vv_db, mask)
    vh_u16 = _scale_minmax_to_uint16(vh_db, mask)

    # Compute vv/vh **linear ratio** (inputs are in dB):
    # ratio_lin = 10^((VVdB - VHdB)/10)
    ratio_lin = np.power(10.0, (vv_db - vh_db) / 10.0, dtype=np.float32)
    ratio_u16 = _scale_minmax_to_uint16(ratio_lin, mask)

    # Save PNGs
    Image.fromarray(vv_u16, mode="I;16").save(os.path.join(out_dir, f"{base}_VV_u16.png"))
    Image.fromarray(vh_u16, mode="I;16").save(os.path.join(out_dir, f"{base}_VH_u16.png"))
    Image.fromarray(ratio_u16, mode="I;16").save(os.path.join(out_dir, f"{base}_VV_over_VH_linear_u16.png"))


if __name__ == "__main__":
    main()
