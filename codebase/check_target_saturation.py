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
SAMPLES = 2000               # patches to sample; set to 0 to scan the whole split
SEED = 42

# Saturation is concentrated in a few snowy/cloudy scenes, so the tail needs more
# patches than the mean does. Percentiles use a random subsample of pixels per patch.
PCT_PIXELS_PER_PATCH = 1000

# Bands used for the RGB display window, in (R, G, B) order.
RGB_BANDS = ("B4", "B3", "B2")
DISPLAY_PCT = (2.0, 98.0)

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


def load_clip_ranges(data_dir, n_bands):
    """Per-band (lo, hi) in DN, in channel order, from S2_clip.json."""
    path = os.path.join(data_dir, "S2_clip.json")
    if not os.path.exists(path):
        return [(0.0, 10000.0)] * n_bands
    with open(path) as f:
        clip = json.load(f)
    keys = sorted(clip, key=lambda k: int(k))
    return [tuple(clip[k]) for k in keys]


def to_dn(x_norm, lo, hi):
    return lo + (x_norm + 1.0) * 0.5 * (hi - lo)


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

    clip = load_clip_ranges(DATA_DIR, n_bands)

    at_lo = np.zeros(n_bands, dtype=np.int64)
    at_hi = np.zeros(n_bands, dtype=np.int64)
    vmin = np.full(n_bands, np.inf)
    vmax = np.full(n_bands, -np.inf)
    vsum = np.zeros(n_bands, dtype=np.float64)
    vsq = np.zeros(n_bands, dtype=np.float64)

    k = min(PCT_PIXELS_PER_PATCH, Y.shape[1] * Y.shape[2])
    pct_pool = np.empty((len(idx) * k, n_bands), dtype=np.float32)
    pct_rng = np.random.default_rng(SEED + 1)

    for j, i in enumerate(idx):
        a = np.asarray(Y[i], dtype=np.float32).reshape(-1, n_bands)
        at_lo += (a <= -1.0).sum(axis=0)
        at_hi += (a >= 1.0).sum(axis=0)
        vmin = np.minimum(vmin, a.min(axis=0))
        vmax = np.maximum(vmax, a.max(axis=0))
        vsum += a.sum(axis=0, dtype=np.float64)
        vsq += (a.astype(np.float64) ** 2).sum(axis=0)

        pick = pct_rng.choice(a.shape[0], size=k, replace=False)
        pct_pool[j * k:(j + 1) * k] = a[pick]

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

    # -------------------- percentiles, in DN --------------------
    qs = [1.0, 2.0, 50.0, 98.0, 99.0]
    pct_norm = np.percentile(pct_pool, qs, axis=0)          # (len(qs), n_bands)
    pct_dn = np.empty_like(pct_norm)
    for c in range(n_bands):
        lo, hi = clip[c] if c < len(clip) else (0.0, 10000.0)
        pct_dn[:, c] = to_dn(pct_norm[:, c], lo, hi)

    print(f"\nPer-band percentiles in DN ({pct_pool.shape[0]:,} sampled pixels per band)")
    print(f"{'band':>5} {'p1':>8} {'p2':>8} {'p50':>8} {'p98':>8} {'p99':>8}")
    print("-" * 50)
    for c, name in enumerate(names):
        print(f"{name:>5} " + " ".join(f"{pct_dn[q, c]:>8.0f}" for q in range(len(qs))))

    # Shared display window for the RGB panels: one window for R, G and B, so the
    # rendering cannot white-balance away a colour error.
    lo_q = qs.index(DISPLAY_PCT[0])
    hi_q = qs.index(DISPLAY_PCT[1])
    rgb_cols = [names.index(b) for b in RGB_BANDS if b in names]
    win_lo = float(min(pct_dn[lo_q, c] for c in rgb_cols))
    win_hi = float(max(pct_dn[hi_q, c] for c in rgb_cols))
    win = (round(win_lo / 50) * 50, round(win_hi / 50) * 50)

    print(f"\nSuggested fixed RGB display window ({'/'.join(RGB_BANDS)}, "
          f"p{DISPLAY_PCT[0]:g}-p{DISPLAY_PCT[1]:g} over the {SPLIT} split):")
    print(f"  FIXED_STRETCH_DN = {win}")

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
        "percentile_pixels_per_band": int(pct_pool.shape[0]),
        "warn_fraction": WARN_FRACTION,
        "display_window_dn": list(win),
        "display_window_bands": list(RGB_BANDS),
        "display_window_percentiles": list(DISPLAY_PCT),
        "percentiles_dn": {
            name: {f"p{q:g}": float(pct_dn[qi, c]) for qi, q in enumerate(qs)}
            for c, name in enumerate(names)
        },
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
