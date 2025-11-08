import os
import csv
import json
import math
import numpy as np
import rasterio

root = "/home/dll0505/thesis/code/_s2_one_winter"   # change if needed

out_dir = "./s2_stats"
os.makedirs(out_dir, exist_ok=True)

# —— Config ——
file_glob_suffix = ".tif"
sample_pixels_per_band_per_file = 100_000   # reduce if memory tight; increase for better percentile estimates
outlier_low_pct, outlier_high_pct = 1, 99   # define "robust range"
outlier_factor = 3.0                        # mark values beyond [p1 - k*(p99-p1), p99 + k*(p99-p1)] as extreme
csv_per_file = os.path.join(out_dir, "per_file_band_stats.csv")
csv_global = os.path.join(out_dir, "global_band_stats.csv")
json_global = os.path.join(out_dir, "global_band_stats.json")

# Prepare CSV headers
per_file_headers = [
    "file","band","count","min","max","p1","p5","p50","p95","p99",
    "mean","std","fraction_below_p1","fraction_above_p99"
]

# Collect samples to estimate GLOBAL percentiles robustly
global_samples = {}  # band_idx -> list of samples (will concatenate progressively)

def sample_band(arr, k):
    flat = arr.ravel()
    if flat.size <= k:
        return flat
    # random subset without replacement
    idx = np.random.choice(flat.size, size=k, replace=False)
    return flat[idx]

# Pass 1: per-file stats + build global sample pool
with open(csv_per_file, "w", newline="") as fcsv:
    w = csv.writer(fcsv)
    w.writerow(per_file_headers)

    for dirpath, dirnames, filenames in os.walk(root):
        # only process S2 subfolders/files
        # if os.path.basename(dirpath).startswith("s2_") or dirpath.endswith("ROIs2017_winter"):
        if True:
            for fname in sorted(filenames):
                if not fname.endswith(file_glob_suffix):
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    with rasterio.open(fpath) as src:
                        # optional: honor internal nodata/mask
                        nodata = src.nodata
                        for b in range(1, src.count+1):
                            band = src.read(b)
                            # Mask nodata if defined
                            if nodata is not None:
                                band = np.where(band == nodata, np.nan, band)
                            # Also respect the alpha/mask if present
                            try:
                                m = src.read_masks(b)  # 0 where invalid
                                band = np.where(m == 0, np.nan, band)
                            except Exception:
                                pass

                            valid = band[np.isfinite(band)]
                            if valid.size == 0:
                                # write empty row
                                w.writerow([fpath, b, 0, "", "", "", "", "", "", "", "", "", ""])
                                continue

                            # Stats
                            mn, mx = valid.min(), valid.max()
                            p1, p5, p50, p95, p99 = np.percentile(valid, [1,5,50,95,99])
                            mean, std = float(valid.mean()), float(valid.std())

                            # Outlier fractions
                            below = (valid < p1).sum() / valid.size
                            above = (valid > p99).sum() / valid.size

                            w.writerow([fpath, b, valid.size, mn, mx, p1, p5, p50, p95, p99, mean, std, below, above])

                            # Sample for global percentiles
                            s = sample_band(valid, sample_pixels_per_band_per_file)
                            if s.size:
                                if b not in global_samples:
                                    global_samples[b] = s
                                else:
                                    # concatenate in chunks to avoid giant arrays
                                    global_samples[b] = np.concatenate([global_samples[b], s])

                except Exception as e:
                    print(f"ERROR reading {fpath}: {e}")

# Pass 2: compute GLOBAL percentiles and suggested clipping per band
global_results = []
for b in sorted(global_samples.keys()):
    s = global_samples[b]
    # drop NaNs if any
    s = s[np.isfinite(s)]
    if s.size == 0:
        continue
    gmin, gmax = float(np.min(s)), float(np.max(s))
    gp = np.percentile(s, [0.1, 1, 2, 5, 50, 95, 98, 99, 99.9])
    p01, p1, p2, p5, p50, p95, p98, p99, p999 = map(float, gp)

    # define robust range and extreme banding
    robust_lo, robust_hi = p1, p99
    robust_span = max(1e-9, robust_hi - robust_lo)
    extreme_lo = robust_lo - outlier_factor * robust_span
    extreme_hi = robust_hi + outlier_factor * robust_span

    global_results.append({
        "band": b,
        "global_min": gmin,
        "global_max": gmax,
        "p0_1": p01, "p1": p1, "p2": p2, "p5": p5, "p50": p50,
        "p95": p95, "p98": p98, "p99": p99, "p99_9": p999,
        "suggested_clip_low": p2,
        "suggested_clip_high": p98,
        "extreme_low": extreme_lo,
        "extreme_high": extreme_hi,
        "samples_used": int(s.size)
    })

# Write GLOBAL CSV + JSON
with open(csv_global, "w", newline="") as fcsv:
    w = csv.writer(fcsv)
    w.writerow([
        "band","global_min","global_max",
        "p0.1","p1","p2","p5","p50","p95","p98","p99","p99.9",
        "suggested_clip_low(p2)","suggested_clip_high(p98)",
        "extreme_low","extreme_high","samples_used"
    ])
    for r in global_results:
        w.writerow([
            r["band"], r["global_min"], r["global_max"],
            r["p0_1"], r["p1"], r["p2"], r["p5"], r["p50"],
            r["p95"], r["p98"], r["p99"], r["p99_9"],
            r["suggested_clip_low"], r["suggested_clip_high"],
            r["extreme_low"], r["extreme_high"], r["samples_used"]
        ])

with open(json_global, "w") as fjson:
    json.dump(global_results, fjson, indent=2)

print(f"\n✅ Done. Per-file stats → {csv_per_file}")
print(f"✅ Global band stats → {csv_global}")
print(f"✅ Global band stats (JSON) → {json_global}")
