# This script prepares, crops and mormalizes (makes them ready for training) the data/images from the SEN12-MS dataset, which is the main script for the thesis.
# Make sure to download the dataset or a subset of it beforehand :) Here is the link: https://mediatum.ub.tum.de/1474000
import os
import sys
import json
import hashlib
import shutil
import gc
import numpy as np
from tqdm import tqdm
import rasterio
from numpy.lib.format import open_memmap

# Try to keep GDAL internal caching small (helps memory spikes)
os.environ.setdefault("GDAL_CACHEMAX", "64")  # MB
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

# -------------------- CONFIG --------------------
S1_CLIP = {"VV": (-25.0, 0.0), "VH": (-32.5, 0.0)}
# Old clips depending on the 2nd and 98th percentiles
# S2_CLIP = {
#     1: (975.0, 2010.9),  2: (740.0, 1952.0),  3: (529.0, 2292.0),
#     4: (341.0, 3243.0),  5: (299.0, 3444.0),  6: (266.0, 3743.0),
#     7: (252.0, 4140.0),  8: (216.0, 3993.0),  9: (214.0, 4293.0),
#     10:(88.0, 2014.0),   11:(7.0, 167.0),     12:(95.0, 5402.0),
#     13:(69.0, 4358.0),
# }

S2_CLIP = {
    1: (0.0, 10000.0),  2: (0.0, 10000.0),  3: (0.0, 10000.0),
    4: (0.0, 10000.0),  5: (0.0, 10000.0),  6: (0.0, 10000.0),
    7: (0.0, 10000.0),  8: (0.0, 10000.0),  9: (0.0, 10000.0),
    10:(0.0, 10000.0), 11:(0.0, 10000.0), 12:(0.0, 10000.0),
    13:(0.0, 10000.0),
}


output_dir = "dataset_winter_all"
data_dir = "/home/dll0505/AdditionalStorage/thesis/sen12-ms/sub_dataset/ROIs2017_winter"

CROP_SIZE = 256
DTYPE = np.float32

# Desired image-level fractions
P_TRAIN = 0.0
P_VAL   = 0.0
P_TEST  = 0.023  # <-- now test matches val

SAMPLE_CHECKS_PER_SPLIT = 10  # read only a few samples for sanity checks

# If you want exact image counts, set numbers here (else leave as None to use percentages)
FORCE_COUNTS_IMAGES = {
    "train": None,  # e.g., 3215
    "val":   None,  # e.g.,  981
    "test":  None,  # e.g.,  981
}

# -------------------- HELPERS --------------------
def load_tif_as_hwc(path):
    with rasterio.Env(GDAL_CACHEMAX=int(os.environ["GDAL_CACHEMAX"])):
        with rasterio.open(path, sharing=False) as src:
            img = src.read()                # (C, H, W)
    return np.transpose(img, (1, 2, 0)).astype(DTYPE)  # (H, W, C)

def to_tanh_range(x, lo, hi):
    x = np.clip(x, lo, hi)
    return 2.0 * (x - lo) / (hi - lo + 1e-9) - 1.0

def s1_transform_db_tanh(x_hwc):
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

def crop_grid(H, W, crop=CROP_SIZE):
    assert H % crop == 0 and W % crop == 0, f"Image dims ({H}x{W}) not divisible by {crop}."
    for hh in range(0, H, crop):
        for ww in range(0, W, crop):
            yield hh, ww

def hash_score(p):
    h = hashlib.md5(os.path.basename(p).encode("utf-8")).hexdigest()
    return (int(h[:8], 16) / 0xFFFFFFFF)  # [0,1)

# -------------------- DISCOVERY --------------------
s1_files, s2_files = [], []
print(f"Searching for .tif files in '{data_dir}'...")
for root, _, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".tif"):
            full_path = os.path.join(root, file)
            if "_s1_" in file:
                s1_files.append(full_path)
            elif "_s2_" in file:
                s2_files.append(full_path)

s1_files.sort(key=lambda f: os.path.basename(f))
s2_files.sort(key=lambda f: os.path.basename(f))

if not s1_files or not s2_files:
    print(f"ERROR: No .tif files found in '{data_dir}'."); sys.exit(1)

if len(s1_files) != len(s2_files):
    print(f"WARNING: Mismatch in S1 ({len(s1_files)}) and S2 ({len(s2_files)}) counts. Will zip to shortest.")

pair_count = min(len(s1_files), len(s2_files))
s1_files = s1_files[:pair_count]
s2_files = s2_files[:pair_count]
print(f"Found {pair_count} paired files")

# Validate dims from first file (no big arrays)
with rasterio.open(s1_files[0]) as src:
    H, W = src.height, src.width
if (H % CROP_SIZE) or (W % CROP_SIZE):
    print(f"ERROR: Image dims ({H}x{W}) not divisible by {CROP_SIZE}."); sys.exit(1)

crops_per_image = (H // CROP_SIZE) * (W // CROP_SIZE)
print(f"Each image -> {crops_per_image} crops of {CROP_SIZE}x{CROP_SIZE}")

# -------------------- QUOTA-BASED DETERMINISTIC SPLIT (by images) --------------------
pairs = [(hash_score(s1), s1, s2) for s1, s2 in zip(s1_files, s2_files)]
pairs.sort(key=lambda t: t[0])  # sort by deterministic hash score

M = len(pairs)
# compute image quotas (optionally force exact counts)
if any(v is not None for v in FORCE_COUNTS_IMAGES.values()):
    n_train = FORCE_COUNTS_IMAGES["train"] if FORCE_COUNTS_IMAGES["train"] is not None else int(round(P_TRAIN * M))
    n_val   = FORCE_COUNTS_IMAGES["val"]   if FORCE_COUNTS_IMAGES["val"]   is not None else int(round(P_VAL   * M))
    n_test  = FORCE_COUNTS_IMAGES["test"]  if FORCE_COUNTS_IMAGES["test"]  is not None else int(round(P_TEST  * M))
else:
    n_train = int(round(P_TRAIN * M))
    n_val   = int(round(P_VAL   * M))
    n_test  = int(round(P_TEST  * M))

# ensure totals don't exceed M
n_train = min(n_train, M)
n_val   = min(n_val, max(0, M - n_train))
n_test  = min(n_test, max(0, M - n_train - n_val))
n_unused = M - (n_train + n_val + n_test)

print(f"Image quotas -> train: {n_train}, val: {n_val}, test: {n_test}, unused: {n_unused}")

train_pairs = pairs[0:n_train]
val_pairs   = pairs[n_train:n_train+n_val]
test_pairs  = pairs[n_train+n_val:n_train+n_val+n_test]

splits = {
    "train": ([p[1] for p in train_pairs], [p[2] for p in train_pairs]),
    "val":   ([p[1] for p in val_pairs],   [p[2] for p in val_pairs]),
    "test":  ([p[1] for p in test_pairs],  [p[2] for p in test_pairs]),
}

# -------------------- COUNT CROPS PER SPLIT --------------------
counts = {k: len(v[0]) * crops_per_image for k, v in splits.items()}
print("Planned crops per split:", counts)

# Disk space estimate (very rough; no compression)
bytes_per_crop = (CROP_SIZE*CROP_SIZE*(2+13))*4
est_total_gb = sum(counts.values()) * bytes_per_crop / (1024**3)
avail_gb = shutil.disk_usage(".").free / (1024**3)
print(f"Estimated final size (uncompressed): ~{est_total_gb:.2f} GB | Free: {avail_gb:.2f} GB")

# -------------------- ALLOCATE NPY FILES (MEMMAP) --------------------
os.makedirs(output_dir, exist_ok=True)
paths = {
    "train": (os.path.join(output_dir, "X_train.npy"), os.path.join(output_dir, "y_train.npy")),
    "val":   (os.path.join(output_dir, "X_val.npy"),   os.path.join(output_dir, "y_val.npy")),
    "test":  (os.path.join(output_dir, "X_test.npy"),  os.path.join(output_dir, "y_test.npy")),
}

mmaps = {}
for split in ["train", "val", "test"]:
    n = counts[split]
    X_path, y_path = paths[split]
    X_mm = open_memmap(X_path, mode="w+", dtype=DTYPE, shape=(n, CROP_SIZE, CROP_SIZE, 2))
    y_mm = open_memmap(y_path, mode="w+", dtype=DTYPE, shape=(n, CROP_SIZE, CROP_SIZE, 13))
    mmaps[split] = {"X": X_mm, "y": y_mm, "cursor": 0}

# Save clip configs
with open(os.path.join(output_dir, "S1_clip.json"), "w") as f:
    json.dump(S1_CLIP, f, indent=2)
with open(os.path.join(output_dir, "S2_clip.json"), "w") as f:
    json.dump({str(k): v for k, v in S2_CLIP.items()}, f, indent=2)

# -------------------- PASS 2: STREAM → CROP → NORM → WRITE --------------------
print("Processing, cropping and normalizing into .npy files...")

def process_split(split):
    s1_list, s2_list = splits[split]
    Xmm = mmaps[split]["X"]; ymm = mmaps[split]["y"]
    cur = mmaps[split]["cursor"]
    for s1f, s2f in tqdm(zip(s1_list, s2_list), total=len(s1_list), desc=f"{split} pairs"):
        X_full = load_tif_as_hwc(s1f)  # (H,W,2)
        y_full = load_tif_as_hwc(s2f)  # (H,W,13)
        if X_full.shape[-1] != 2 or y_full.shape[-1] != 13:
            print(f"Skipping malformed pair: {s1f} / {s2f}")
            continue
        for hh, ww in crop_grid(H, W, CROP_SIZE):
            X_patch = X_full[hh:hh+CROP_SIZE, ww:ww+CROP_SIZE, :]
            y_patch = y_full[hh:hh+CROP_SIZE, ww:ww+CROP_SIZE, :]

            # EXACT same normalization
            Xn = s1_transform_db_tanh(X_patch)
            yn = s2_transform_13bands_tanh(y_patch)

            Xmm[cur] = Xn
            ymm[cur] = yn
            cur += 1
        del X_full, y_full, X_patch, y_patch, Xn, yn
        gc.collect()
    mmaps[split]["cursor"] = cur

for split in ["train", "val", "test"]:
    process_split(split)

# Validate cursor counts quickly
for split in ["train", "val", "test"]:
    if mmaps[split]["cursor"] != counts[split]:
        print(f"ERROR: Cursor mismatch for {split}: expected {counts[split]}, wrote {mmaps[split]['cursor']}")
        sys.exit(1)

# Flush files
for split in ["train", "val", "test"]:
    mmaps[split]["X"].flush(); mmaps[split]["y"].flush()
mmaps = None
gc.collect()

# -------------------- LIGHT SANITY CHECKS (sample only) --------------------
print("\n=== Sanity checks (sampled) ===")
rng = np.random.default_rng(0)

def sampled_check(name, path, samples=SAMPLE_CHECKS_PER_SPLIT):
    arr = np.load(path, mmap_mode="r")
    n = arr.shape[0]
    if n == 0:
        print(f"{name}: empty")
        return
    idx = rng.choice(n, size=min(samples, n), replace=False)
    mn, mx, nans = np.inf, -np.inf, 0
    for i in idx:
        a = arr[i]
        mn = min(mn, float(a.min()))
        mx = max(mx, float(a.max()))
        nans += int(np.isnan(a).sum())
    print(f"{name}: shape={arr.shape} sampled_min={mn:.6f}, sampled_max={mx:.6f}, sampled_NaNs={nans}")
    assert mn >= -1.0001 and mx <= 1.0001, f"{name} out of [-1,1] bounds!"
    assert nans == 0, f"{name} contains NaNs!"

for split in ["train", "val", "test"]:
    Xp, yp = paths[split]
    sampled_check(f"X_{split}", Xp)
    sampled_check(f"y_{split}", yp)

# Boundary math checks (no data read)
def spot_boundary_S1():
    vv_lo, vv_hi = S1_CLIP["VV"]; vh_lo, vh_hi = S1_CLIP["VH"]
    vv_test = to_tanh_range(np.array([vv_lo, vv_hi], dtype=DTYPE), vv_lo, vv_hi)
    vh_test = to_tanh_range(np.array([vh_lo, vh_hi], dtype=DTYPE), vh_lo, vh_hi)
    assert np.allclose(vv_test, [-1.0, 1.0], atol=1e-5)
    assert np.allclose(vh_test, [-1.0, 1.0], atol=1e-5)
    print("S1 boundary mapping OK.")

def spot_boundary_S2():
    for b in range(13):
        lo, hi = S2_CLIP[b+1]
        test = to_tanh_range(np.array([lo, hi], dtype=DTYPE), lo, hi)
        assert np.allclose(test, [-1.0, 1.0], atol=1e-5)
    print("S2 per-band boundary mappings OK.")

spot_boundary_S1()
spot_boundary_S2()
print("\nAll tests (sampled) passed ✅")
print("Saved exactly like before:")
print(f"  {output_dir}/X_train.npy, y_train.npy")
print(f"  {output_dir}/X_val.npy,   y_val.npy")
print(f"  {output_dir}/X_test.npy,  y_test.npy")
print("Clip configs: S1_clip.json, S2_clip.json")
