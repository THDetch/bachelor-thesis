"""Freezes the per-band p2-p98 display windows KID maps to 8 bit.

Training split only. Run once; the output is an input to eval_kid_indices.py and must
not be regenerated from validation or test data.

    python make_kid_stretch.py
"""

import json
import os

import numpy as np

DATA_DIR = "dataset_all_seasons_scenesplit"
OUT = os.path.join(DATA_DIR, "kid_stretch.json")

BAND_NAMES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",
              "B8A", "B9", "B10", "B11", "B12"]
BANDS = {"B2": 1, "B3": 2, "B4": 3, "B8": 7}
PX_PER_PATCH = 1000
SEED = 42
LO_PCT, HI_PCT = 2.0, 98.0


def main():
    if os.path.isfile(OUT):
        raise SystemExit(f"{OUT} already exists; refusing to overwrite a frozen window")

    y = np.load(os.path.join(DATA_DIR, "y_train.npy"), mmap_mode="r")
    with open(os.path.join(DATA_DIR, "S2_clip.json")) as f:
        clip = {int(k): tuple(v) for k, v in json.load(f).items()}

    n, h, w, c = y.shape
    rng = np.random.default_rng(SEED)
    rows = rng.integers(0, h, size=(n, PX_PER_PATCH))
    cols = rng.integers(0, w, size=(n, PX_PER_PATCH))

    out = {}
    for name, idx in BANDS.items():
        vals = np.empty(n * PX_PER_PATCH, dtype=np.float32)
        for i in range(n):
            vals[i * PX_PER_PATCH:(i + 1) * PX_PER_PATCH] = \
                y[i, rows[i], cols[i], idx]
        lo_n, hi_n = np.percentile(vals, [LO_PCT, HI_PCT])

        # clip JSON is 1-based over the 13 bands; stored arrays are [-1, 1]
        c_lo, c_hi = clip[idx + 1]
        to_dn = lambda v: c_lo + (float(v) + 1.0) * 0.5 * (c_hi - c_lo)
        out[name] = {"band_index": idx,
                     "p2_dn": round(to_dn(lo_n), 1),
                     "p98_dn": round(to_dn(hi_n), 1)}
        print(f"{name}  p2 {out[name]['p2_dn']:8.1f} DN   "
              f"p98 {out[name]['p98_dn']:8.1f} DN")

    with open(OUT, "w") as f:
        json.dump({"split": "train", "patches": int(n),
                   "pixels_per_patch": PX_PER_PATCH, "seed": SEED,
                   "percentiles": [LO_PCT, HI_PCT], "bands": out}, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
