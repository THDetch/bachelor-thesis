import os
import csv
import json
import re
import numpy as np
import rasterio

# ====== CONFIG ======
root = "/home/dll0505/thesis/code/_s1_one_winter"   # <— your S1 root
out_dir = "./s1_stats"
os.makedirs(out_dir, exist_ok=True)

file_glob_suffix = ".tif"
sample_pixels_per_band_per_file = 100_000
outlier_factor = 3.0
csv_per_file = os.path.join(out_dir, "per_file_band_stats.csv")
csv_global = os.path.join(out_dir, "global_band_stats.csv")
json_global = os.path.join(out_dir, "global_band_stats.json")
mapping_report = os.path.join(out_dir, "band_mapping_report.csv")

# DB conversion: "AUTO" | "YES" | "NO"
DB_MODE = "AUTO"
DB_EPS = 1e-6

# ====== HELPERS ======
def extract_s1_band_name(src, b: int):
    """Try to get 'VV'/'VH' from metadata; fallback to sensible names."""
    # descriptions
    try:
        if src.descriptions and len(src.descriptions) >= b:
            d = (src.descriptions[b-1] or "").strip().upper()
            if "VV" in d: return "VV"
            if "VH" in d: return "VH"
    except Exception:
        pass
    # tags
    try:
        tags = src.tags(b)
        cand = []
        for k in ["POLARIZATION","POL","BANDNAME","BAND","NAME","DESCRIPTION","DESC"]:
            if k in tags and isinstance(tags[k], str):
                cand.append(tags[k].upper())
        for v in tags.values():
            if isinstance(v, str):
                cand.append(v.upper())
        for s in cand:
            if "VV" in s: return "VV"
            if "VH" in s: return "VH"
    except Exception:
        pass
    # fallback by index (common order)
    return "VV" if b == 1 else ("VH" if b == 2 else f"band{b}")

def to_db_if_needed(arr: np.ndarray, mode: str):
    """Return (arr_db, used_db: bool)."""
    if mode == "NO":
        return arr, False
    # If already looks like dB (typical S1 ~[-35, 5]), leave as is
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr, False
    mn, mx, med = np.min(finite), np.max(finite), np.median(finite)
    looks_db = (mn < 0 and mx <= 10) or (mx <= 5) or (med < 0)
    if mode == "YES":
        # convert linear → dB
        arr = 10.0 * np.log10(np.clip(arr, DB_EPS, None))
        return arr, True
    if mode == "AUTO":
        if looks_db:
            return arr, False
        # If values are mostly > 1 and positive, likely linear sigma0
        mostly_linear = (mx > 10) or (med > 1.0)
        if mostly_linear and np.nanmin(arr) >= 0:
            arr = 10.0 * np.log10(np.clip(arr, DB_EPS, None))
            return arr, True
        else:
            # fallback: do nothing
            return arr, False
    return arr, False

def sample_band(arr, k):
    flat = arr.ravel()
    if flat.size <= k:
        return flat
    idx = np.random.choice(flat.size, size=k, replace=False)
    return flat[idx]

# ====== PASS 1: per-file stats, collect samples by band_name ======
per_file_headers = [
    "file","band_index","band_name","units","count","min","max",
    "p1","p5","p50","p95","p99","mean","std",
    "fraction_below_p1","fraction_above_p99"
]
global_samples = {}   # band_name -> np.array
name_to_indices = {}  # band_name -> {indices}

with open(csv_per_file, "w", newline="") as fcsv:
    w = csv.writer(fcsv)
    w.writerow(per_file_headers)

    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            if not fname.endswith(file_glob_suffix):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with rasterio.open(fpath) as src:
                    nd = src.nodata
                    for b in range(1, src.count+1):
                        bname = extract_s1_band_name(src, b)
                        name_to_indices.setdefault(bname, set()).add(b)

                        band = src.read(b).astype(np.float64)
                        # apply nodata/mask
                        if nd is not None:
                            band = np.where(band == nd, np.nan, band)
                        try:
                            m = src.read_masks(b)  # 0 invalid
                            band = np.where(m == 0, np.nan, band)
                        except Exception:
                            pass

                        # Convert to dB if needed
                        band_db, used_db = to_db_if_needed(band, DB_MODE)
                        units = "dB" if used_db else "native"

                        valid = band_db[np.isfinite(band_db)]
                        if valid.size == 0:
                            w.writerow([fpath, b, bname, units, 0, "", "", "", "", "", "", "", "", "", ""])
                            continue

                        mn, mx = float(np.min(valid)), float(np.max(valid))
                        p1, p5, p50, p95, p99 = np.percentile(valid, [1,5,50,95,99])
                        mean, std = float(valid.mean()), float(valid.std())
                        below = float((valid < p1).sum() / valid.size)
                        above = float((valid > p99).sum() / valid.size)

                        w.writerow([fpath, b, bname, units, valid.size, mn, mx,
                                    p1, p5, p50, p95, p99, mean, std, below, above])

                        flat = valid.ravel()
                        if flat.size > sample_pixels_per_band_per_file:
                            idx = np.random.choice(flat.size, sample_pixels_per_band_per_file, replace=False)
                            flat = flat[idx]
                        if bname not in global_samples:
                            global_samples[bname] = flat
                        else:
                            global_samples[bname] = np.concatenate([global_samples[bname], flat])

            except Exception as e:
                print(f"ERROR reading {fpath}: {e}")

# Mapping report (check if VV/VH ever swap indices)
with open(mapping_report, "w", newline="") as fcsv:
    w = csv.writer(fcsv)
    w.writerow(["band_name","seen_indices"])
    for bname, idxs in sorted(name_to_indices.items(), key=lambda x: x[0]):
        w.writerow([bname, ";".join(map(str, sorted(list(idxs))))])

# ====== PASS 2: global stats by band_name ======
global_results = []
for bname in sorted(global_samples.keys()):
    s = global_samples[bname]
    s = s[np.isfinite(s)]
    if s.size == 0:
        continue

    gmin, gmax = float(np.min(s)), float(np.max(s))
    p01, p1, p2, p5, p50, p95, p98, p99, p999 = map(float, np.percentile(s, [0.1,1,2,5,50,95,98,99,99.9]))
    robust_lo, robust_hi = p1, p99
    span = max(1e-9, robust_hi - robust_lo)
    extreme_lo = robust_lo - outlier_factor * span
    extreme_hi = robust_hi + outlier_factor * span

    global_results.append({
        "band_name": bname,
        "global_min": gmin, "global_max": gmax,
        "p0_1": p01, "p1": p1, "p2": p2, "p5": p5, "p50": p50,
        "p95": p95, "p98": p98, "p99": p99, "p99_9": p999,
        "suggested_clip_low": p2,
        "suggested_clip_high": p98,
        "extreme_low": extreme_lo,
        "extreme_high": extreme_hi,
        "samples_used": int(s.size),
        "seen_indices": sorted(list(name_to_indices.get(bname, [])))
    })

with open(csv_global, "w", newline="") as fcsv:
    w = csv.writer(fcsv)
    w.writerow([
        "band_name","seen_indices","global_min","global_max",
        "p0.1","p1","p2","p5","p50","p95","p98","p99","p99.9",
        "suggested_clip_low(p2)","suggested_clip_high(p98)",
        "extreme_low","extreme_high","samples_used"
    ])
    for r in global_results:
        w.writerow([
            r["band_name"], ";".join(map(str, r["seen_indices"])),
            r["global_min"], r["global_max"],
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
print(f"✅ Mapping report → {mapping_report}")
print(f"ℹ️ dB mode: {DB_MODE}")
