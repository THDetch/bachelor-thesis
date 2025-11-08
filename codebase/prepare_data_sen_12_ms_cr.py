# IMPORTANT: This script gets samples from the SEN12-MS-CR dataset, crops and mormalize them depending on the season. Make sure to adjust pathes/variables/regex accordingly, if you are using other seasons/subsets from the SEN12-MS-CR dataset!!
# Make sure to download the dataset or a subset of it beforehand :) Here is the link: https://mediatum.ub.tum.de/1554803

import os
import sys
import re
import json
import shutil
import gc
import random
import numpy as np
import rasterio
from tqdm import tqdm
from collections import defaultdict, Counter
from numpy.lib.format import open_memmap

# -------------------- CONFIG --------------------
S1_ROOT        = "/home/dll0505/AdditionalStorage/thesis/sen12-ms-cr/sub_dataset/ROIs2017_winter_s1"
S2_CLEAN_ROOT  = "/home/dll0505/AdditionalStorage/thesis/sen12-ms-cr/sub_dataset/ROIs2017_winter_s2"
S2_CLOUDY_ROOT = "/home/dll0505/AdditionalStorage/thesis/sen12-ms-cr/sub_dataset/ROIs2017_winter_s2_cloudy"

OUTPUT_DIR = "/home/dll0505/AdditionalStorage/thesis/sen12-ms-cr/dataset_cr_winter_test_2000"

MAX_PATCHES = 2000   # total patches to include (set None to take all)
DTYPE = np.float32 

# Sampler controls
RNG_SEED = 123
BALANCE_ACROSS_SCENES = True  # True -> stratified round-robin; False -> pure random
PER_SCENE_CAP = None          # e.g., 80 to limit how many per scene (use with BALANCE_ACROSS_SCENES)

# Keep GDAL cache small
os.environ.setdefault("GDAL_CACHEMAX", "64")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

# Normalization clips (as requested: 0..10000)
S1_CLIP = {"VV": (-25.0, 0.0), "VH": (-32.5, 0.0)}
S2_CLIP = {
    1:(0.0,10000.0),  2:(0.0,10000.0),  3:(0.0,10000.0),
    4:(0.0,10000.0),  5:(0.0,10000.0),  6:(0.0,10000.0),
    7:(0.0,10000.0),  8:(0.0,10000.0),  9:(0.0,10000.0),
    10:(0.0,10000.0), 11:(0.0,10000.0), 12:(0.0,10000.0),
    13:(0.0,10000.0),
}

# Regex patterns for names like:
#   ROIs2017_winter_s1_102_p100.tif
#   ROIs2017_winter_s2_102_p100.tif
#   ROIs2017_winter_s2_cloudy_102_p100.tif
PAT_S1  = re.compile(r"ROIs2017_winter_s1_(\d+)_p(\d+)\.tif$", re.IGNORECASE)
PAT_S2  = re.compile(r"ROIs2017_winter_s2_(\d+)_p(\d+)\.tif$", re.IGNORECASE)
PAT_S2C = re.compile(r"ROIs2017_winter_s2_cloudy_(\d+)_p(\d+)\.tif$", re.IGNORECASE)

# -------------------- HELPERS --------------------
def list_all_tifs(root):
    out = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".tif"):
                out.append(os.path.join(r, f))
    return out

def parse_key(path, pat):
    """Return (scene, patch) as strings, or None if no match."""
    m = pat.search(os.path.basename(path))
    if not m:
        return None
    return m.group(1), m.group(2)

def load_tif_as_hwc(path):
    with rasterio.Env(GDAL_CACHEMAX=int(os.environ["GDAL_CACHEMAX"])):
        with rasterio.open(path, sharing=False) as src:
            arr = src.read()  # (C,H,W)
    return np.transpose(arr, (1, 2, 0)).astype(DTYPE)

def to_tanh_range(x, lo, hi):
    x = np.clip(x, lo, hi)
    return 2.0 * (x - lo) / (hi - lo + 1e-9) - 1.0

def s1_transform_db_tanh(x_hwc):
    # assume bands are [VV, VH] in channel order
    out = np.empty_like(x_hwc, dtype=DTYPE)
    lo, hi = S1_CLIP["VV"]; out[..., 0] = to_tanh_range(x_hwc[..., 0], lo, hi)
    lo, hi = S1_CLIP["VH"]; out[..., 1] = to_tanh_range(x_hwc[..., 1], lo, hi)
    return out

def s2_transform_13bands_tanh(y_hwc):
    assert y_hwc.shape[-1] == 13, f"Expected 13 bands, got {y_hwc.shape[-1]}"
    out = np.empty_like(y_hwc, dtype=DTYPE)
    for b in range(13):
        lo, hi = S2_CLIP[b + 1]
        out[..., b] = to_tanh_range(y_hwc[..., b], lo, hi)
    return out

def round_robin_take(groups, scene_order, total, cap=None):
    """groups: dict scene -> list of keys; scene_order: list of scenes"""
    taken = []
    idx = {s: 0 for s in scene_order}
    per_scene_taken = Counter()
    while len(taken) < total:
        progressed = False
        for s in scene_order:
            # respect per-scene cap if set
            if cap is not None and per_scene_taken[s] >= cap:
                continue
            i = idx[s]
            if i < len(groups[s]):
                taken.append(groups[s][i])
                idx[s] += 1
                per_scene_taken[s] += 1
                progressed = True
                if len(taken) == total:
                    break
        if not progressed:
            break
    return taken

# -------------------- DISCOVERY & MATCHING --------------------
print("Scanning SEN12MS-CR winter roots:")
print(f"  S1={S1_ROOT}")
print(f"  S2(clean)={S2_CLEAN_ROOT}")
print(f"  S2(cloudy)={S2_CLOUDY_ROOT}")

s1_files  = list_all_tifs(S1_ROOT)
s2_files  = list_all_tifs(S2_CLEAN_ROOT)
s2c_files = list_all_tifs(S2_CLOUDY_ROOT)

# Build maps: key=(scene,patch) -> path
S1_MAP, S2_MAP, S2C_MAP = {}, {}, {}
skipped_s1 = skipped_s2 = skipped_s2c = 0

for p in s1_files:
    k = parse_key(p, PAT_S1)
    if k: S1_MAP[k] = p
    else: skipped_s1 += 1

for p in s2_files:
    k = parse_key(p, PAT_S2)
    if k: S2_MAP[k] = p
    else: skipped_s2 += 1

for p in s2c_files:
    k = parse_key(p, PAT_S2C)
    if k: S2C_MAP[k] = p
    else: skipped_s2c += 1

common_keys = sorted(set(S1_MAP.keys()) & set(S2_MAP.keys()) & set(S2C_MAP.keys()),
                     key=lambda x: (int(x[0]), int(x[1])))

if not common_keys:
    print("ERROR: No matching triplets found. Check naming/roots.")
    print(f"Parsed S1: {len(S1_MAP)} (skipped {skipped_s1}) | "
          f"S2: {len(S2_MAP)} (skipped {skipped_s2}) | "
          f"S2_cloudy: {len(S2C_MAP)} (skipped {skipped_s2c})")
    sys.exit(1)

scene_counts_all = Counter([k[0] for k in common_keys])
print(f"Found {len(common_keys)} matched triplets across {len(scene_counts_all)} scenes.")
print("Top scenes (all):", scene_counts_all.most_common(10))

# -------------------- RANDOM / BALANCED SELECTION --------------------
random.seed(RNG_SEED)
target_total = len(common_keys) if (MAX_PATCHES is None or MAX_PATCHES <= 0) else min(MAX_PATCHES, len(common_keys))

if BALANCE_ACROSS_SCENES:
    # Group keys by scene; shuffle within each scene & randomize scene order
    by_scene = defaultdict(list)
    for k in common_keys:
        by_scene[k[0]].append(k)
    for s in by_scene:
        random.shuffle(by_scene[s])
    scenes = list(by_scene.keys())
    random.shuffle(scenes)
    selected_keys = round_robin_take(by_scene, scenes, target_total, cap=PER_SCENE_CAP)
else:
    # Pure random sample over all keys
    selected_keys = random.sample(common_keys, k=target_total)

scene_counts_sel = Counter([k[0] for k in selected_keys])
print(f"Using {len(selected_keys)} patches for TEST.")
print(f"Selected unique scenes: {len(scene_counts_sel)}")
print("Top scenes (selected):", scene_counts_sel.most_common(10))

# -------------------- SHAPE PROBE --------------------
first_s1  = load_tif_as_hwc(S1_MAP[selected_keys[0]])
first_s2  = load_tif_as_hwc(S2_MAP[selected_keys[0]])
first_s2c = load_tif_as_hwc(S2C_MAP[selected_keys[0]])

H, W, Cin  = first_s1.shape
_, _, Cout = first_s2.shape

if Cin != 2:
    print(f"WARNING: S1 channels = {Cin} (expected 2). Adjust s1_transform_db_tanh if needed.")
if Cout != 13:
    print(f"ERROR: S2 channels = {Cout} (expected 13).")
    sys.exit(1)

print(f"Patch size: {H}x{W} | S1 C={Cin} | S2 C={Cout}")

# -------------------- DISK SPACE CHECK --------------------
bytes_per_sample = H * W * (Cin + Cout + Cout) * np.dtype(DTYPE).itemsize  # X + y + y_cloudy
need_bytes = len(selected_keys) * bytes_per_sample
free_bytes = shutil.disk_usage(os.path.dirname(OUTPUT_DIR) or ".").free
print(f"Estimated need: {need_bytes/1024**3:.2f} GB | Free: {free_bytes/1024**3:.2f} GB")
if free_bytes < need_bytes * 1.05:
    print("ERROR: Not enough free space (need ~5% headroom). "
          "Reduce MAX_PATCHES, change OUTPUT_DIR, or switch to float16.")
    sys.exit(1)

# -------------------- ALLOCATE MEMMAPS --------------------
N = len(selected_keys)
os.makedirs(OUTPUT_DIR, exist_ok=True)

X_path  = os.path.join(OUTPUT_DIR, "X_test.npy")
y_path  = os.path.join(OUTPUT_DIR, "y_test.npy")          # clean GT
yC_path = os.path.join(OUTPUT_DIR, "y_cloudy_test.npy")   # cloudy (optional)

X_mm  = open_memmap(X_path,  mode="w+", dtype=DTYPE, shape=(N, H, W, Cin))
y_mm  = open_memmap(y_path,  mode="w+", dtype=DTYPE, shape=(N, H, W, Cout))
yC_mm = open_memmap(yC_path, mode="w+", dtype=DTYPE, shape=(N, H, W, Cout))

# Save clip configs
with open(os.path.join(OUTPUT_DIR, "S1_clip.json"), "w") as f:
    json.dump(S1_CLIP, f, indent=2)
with open(os.path.join(OUTPUT_DIR, "S2_clip.json"), "w") as f:
    json.dump({str(k): v for k, v in S2_CLIP.items()}, f, indent=2)

# -------------------- PROCESS & NORMALIZE --------------------
print("Processing patches into test memmaps...")
cursor = 0
for key in tqdm(selected_keys, total=N, desc="test patches"):
    s1f  = S1_MAP[key]
    s2f  = S2_MAP[key]
    s2cf = S2C_MAP[key]

    X_full  = load_tif_as_hwc(s1f)    # (H,W,2)
    y_full  = load_tif_as_hwc(s2f)    # (H,W,13) clean
    yC_full = load_tif_as_hwc(s2cf)   # (H,W,13) cloudy

    if X_full.shape[-1] != Cin or y_full.shape[-1] != Cout or yC_full.shape[-1] != Cout:
        print(f"Skipping malformed triplet: {s1f}")
        del X_full, y_full, yC_full
        gc.collect()
        continue

    # EXACT normalization as training
    Xn  = s1_transform_db_tanh(X_full)
    yn  = s2_transform_13bands_tanh(y_full)
    yCn = s2_transform_13bands_tanh(yC_full)

    X_mm[cursor]  = Xn
    y_mm[cursor]  = yn
    yC_mm[cursor] = yCn
    cursor += 1

    del X_full, y_full, yC_full, Xn, yn, yCn
    gc.collect()

if cursor != N:
    print(f"WARNING: wrote {cursor} of {N}; truncating arrays.")
    X_mm.flush(); y_mm.flush(); yC_mm.flush()
    # Truncate memmaps by re-saving compact arrays:
    X = np.array(X_mm[:cursor], copy=True); del X_mm
    y = np.array(y_mm[:cursor], copy=True); del y_mm
    yC= np.array(yC_mm[:cursor], copy=True); del yC_mm
    np.save(X_path, X); np.save(y_path, y); np.save(yC_path, yC)
else:
    X_mm.flush(); y_mm.flush(); yC_mm.flush()

# -------------------- SANITY CHECKS --------------------
print("\n=== Sanity checks (sampled) ===")
rng = np.random.default_rng(0)

def sampled_check(name, path, samples=8):
    arr = np.load(path, mmap_mode="r")
    n = arr.shape[0]
    if n == 0:
        print(f"{name}: empty"); return
    idx = rng.choice(n, size=min(samples, n), replace=False)
    mn, mx, nans = np.inf, -np.inf, 0
    for i in idx:
        a = arr[i]
        mn = min(mn, float(a.min()))
        mx = max(mx, float(a.max()))
        nans += int(np.isnan(a).sum())
    print(f"{name}: shape={arr.shape} sampled_min={mn:.6f}, sampled_max={mx:.6f}, NaNs={nans}")
    assert mn >= -1.0001 and mx <= 1.0001, f"{name} out of [-1,1] bounds!"
    assert nans == 0, f"{name} contains NaNs!"

sampled_check("X_test", X_path)
sampled_check("y_test", y_path)
sampled_check("y_cloudy_test", yC_path)

print("\nAll tests (sampled) passed ✅")
print("Saved:")
print(f"  {OUTPUT_DIR}/X_test.npy")
print(f"  {OUTPUT_DIR}/y_test.npy          (clean GT)")
print(f"  {OUTPUT_DIR}/y_cloudy_test.npy   (cloudy; optional)")
print("Clip configs: S1_clip.json, S2_clip.json")
