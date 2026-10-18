# Reports how many normalized target pixels sit exactly at the clip bounds (-1 / +1).
#
# Why this matters: the generator ends in tanh, which can never reach +-1 for ordinary
# activations. Any target pixel at exactly +-1 is therefore (a) an irreducible L1 residual
# and (b) a trivial, permanent cue the discriminator can use to separate real from fake.
# Saturation comes from the global [0, 10000] S2 clip, and from no-data pixels that the
# preparation script maps to the low clip bound.
#
# Usage:
#   python check_target_saturation.py
# Adjust the CONFIG block below to point at the dataset you just prepared.

import json
import os

import numpy as np


# -------------------- CONFIG --------------------

DATA_DIR = "dataset_all_seasons_scenesplit"
SPLIT = "train"              # train | val | test
SAMPLES = 500                # patches to sample; set to 0 to scan the whole split
SEED = 42

# Flag a band above this fraction of saturated pixels.
WARN_FRACTION = 0.01


# -------------------- HELPERS --------------------

BAND_NAMES_13 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
BAND_NAMES_10 = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


def band_names(n_bands):
    if n_bands == 13:
        return BAND_NAMES_13
    if n_bands == 10:
        return BAND_NAMES_10
    return [f"C{i}" for i in range(n_bands)]


def main():
    y_path = os.path.join(DATA_DIR, f"y_{SPLIT}.npy")
    x_path = os.path.join(DATA_DIR, f"X_{SPLIT}.npy")

    Y = np.load(y_path, mmap_mode="r")
    X = np.load(x_path, mmap_mode="r")

    n_total = Y.shape[0]
    n_bands = Y.shape[-1]
    names = band_names(n_bands)

    rng = np.random.default_rng(SEED)
    if SAMPLES and SAMPLES < n_total:
        idx = np.sort(rng.choice(n_total, size=SAMPLES, replace=False))
    else:
        idx = np.arange(n_total)

    print(f"{y_path}: {n_total} patches, {n_bands} bands")
    print(f"Sampling {len(idx)} patches (seed {SEED})\n")

    px_per_band = len(idx) * Y.shape[1] * Y.shape[2]

    at_lo = np.zeros(n_bands, dtype=np.int64)
    at_hi = np.zeros(n_bands, dtype=np.int64)
    vmin = np.full(n_bands, np.inf)
    vmax = np.full(n_bands, -np.inf)
    vsum = np.zeros(n_bands, dtype=np.float64)
    vsq = np.zeros(n_bands, dtype=np.float64)

    for i in idx:
        a = np.asarray(Y[i], dtype=np.float32).reshape(-1, n_bands)
        at_lo += (a <= -1.0).sum(axis=0)
        at_hi += (a >= 1.0).sum(axis=0)
        vmin = np.minimum(vmin, a.min(axis=0))
        vmax = np.maximum(vmax, a.max(axis=0))
        vsum += a.sum(axis=0, dtype=np.float64)
        vsq += (a.astype(np.float64) ** 2).sum(axis=0)

    mean = vsum / px_per_band
    std = np.sqrt(np.maximum(0.0, vsq / px_per_band - mean**2))

    frac_lo = at_lo / px_per_band
    frac_hi = at_hi / px_per_band

    print(f"{'band':>5} {'at -1':>9} {'at +1':>9} {'min':>8} {'max':>8} {'mean':>8} {'std':>8}")
    print("-" * 60)

    flagged = []
    for c, name in enumerate(names):
        flag = ""
        if frac_lo[c] > WARN_FRACTION or frac_hi[c] > WARN_FRACTION:
            flag = "  <-- saturated"
            flagged.append(name)
        print(
            f"{name:>5} {frac_lo[c]:>8.3%} {frac_hi[c]:>8.3%} "
            f"{vmin[c]:>8.4f} {vmax[c]:>8.4f} {mean[c]:>8.4f} {std[c]:>8.4f}{flag}"
        )

    # SAR inputs are not tanh-constrained, but a dead channel is still worth knowing about.
    x0 = np.asarray(X[idx[0]], dtype=np.float32)
    print(f"\nSAR input (first sampled patch): min {x0.min():.4f}  max {x0.max():.4f}")

    print()
    if flagged:
        print(f"FLAGGED (> {WARN_FRACTION:.0%} of pixels at a clip bound): {', '.join(flagged)}")
        print("Options: per-band percentile clipping, or clip targets to +-0.999 so tanh can reach them.")
    else:
        print(f"No band exceeds {WARN_FRACTION:.0%} saturation. Clipping is not feeding the discriminator a free cue.")

    out = {
        "data_dir": DATA_DIR,
        "split": SPLIT,
        "sampled_patches": int(len(idx)),
        "pixels_per_band": int(px_per_band),
        "warn_fraction": WARN_FRACTION,
        "bands": {
            name: {
                "frac_at_minus_one": float(frac_lo[c]),
                "frac_at_plus_one": float(frac_hi[c]),
                "min": float(vmin[c]),
                "max": float(vmax[c]),
                "mean": float(mean[c]),
                "std": float(std[c]),
            }
            for c, name in enumerate(names)
        },
        "flagged": flagged,
    }

    out_path = os.path.join(DATA_DIR, f"saturation_{SPLIT}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
