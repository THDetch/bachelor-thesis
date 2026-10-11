# This script prepares, crops and normalizes the SEN12-MS data/images,
# making them ready for training.
#
# Important:
# This version uses a SEASON-STRATIFIED split.
# That means train, val, and test each contain samples from every selected season.

import os
import sys
import glob
import json
import hashlib
import shutil
import gc
from collections import defaultdict
import csv
import re

import numpy as np
from tqdm import tqdm
import rasterio
from numpy.lib.format import open_memmap

# Try to keep GDAL internal caching small
os.environ.setdefault("GDAL_CACHEMAX", "64")  # MB
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")


# -------------------- CONFIG --------------------

S1_CLIP = {
    "VV": (-25.0, 0.0),
    "VH": (-32.5, 0.0),
}

S2_CLIP = {
    1: (0.0, 10000.0),
    2: (0.0, 10000.0),
    3: (0.0, 10000.0),
    4: (0.0, 10000.0),
    5: (0.0, 10000.0),
    6: (0.0, 10000.0),
    7: (0.0, 10000.0),
    8: (0.0, 10000.0),
    9: (0.0, 10000.0),
    10: (0.0, 10000.0),
    11: (0.0, 10000.0),
    12: (0.0, 10000.0),
    13: (0.0, 10000.0),
}

data_dir = "/home/dll1801/mdpi_manuscript/datasets/SEN12MS"

# Use all seasons
seasons = ["spring", "summer", "fall", "winter"]

# Output folder
output_dir = "dataset_all_seasons_metadata"

CROP_SIZE = 256
DTYPE = np.float32

# Desired image-level fractions
P_TRAIN = 0.8
P_VAL = 0.1
P_TEST = 0.1

SAMPLE_CHECKS_PER_SPLIT = 10

# Limit total image pairs.
# With 4 seasons and MAX_SAMPLES=32000, this gives up to 8000 pairs per season.
# Set to None to process all available pairs from all selected seasons.
MAX_SAMPLES = 32000

# Debugging
PRINT_MISSING_EXAMPLES = 10


# -------------------- HELPERS --------------------


def extract_roi_info(path):
    """
    Extract useful SEN12MS info from a path like:
    .../ROIs1970_fall_s1/s1_42/ROIs1970_fall_s1_42_p327.tif
    """

    basename = os.path.basename(path)

    # Example match:
    # ROIs1970_fall_s1_42_p327.tif
    pattern = r"(ROIs\d+)_(spring|summer|fall|winter)_s1_(\d+)_p(\d+)\.tif"
    match = re.match(pattern, basename)

    if match:
        roi_group = match.group(1)      # ROIs1970
        season = match.group(2)         # fall
        roi_id = match.group(3)         # 42
        patch_id = match.group(4)       # 327
    else:
        roi_group = "unknown"
        season = "unknown"
        roi_id = "unknown"
        patch_id = "unknown"

    return roi_group, season, roi_id, patch_id


def save_split_csv(split_name, pair_list, output_dir):
    csv_path = os.path.join(output_dir, f"{split_name}_pairs.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "split",
            "roi_group",
            "season",
            "roi_id",
            "patch_id",
            "s1_path",
            "s2_path",
        ])

        for _, s1_path, s2_path in pair_list:
            roi_group, season, roi_id, patch_id = extract_roi_info(s1_path)

            writer.writerow([
                split_name,
                roi_group,
                season,
                roi_id,
                patch_id,
                s1_path,
                s2_path,
            ])

    print(f"Saved split CSV: {csv_path}")

def load_tif_as_hwc(path):
    with rasterio.Env(GDAL_CACHEMAX=int(os.environ["GDAL_CACHEMAX"])):
        with rasterio.open(path, sharing=False) as src:
            img = src.read()  # (C, H, W)

    return np.transpose(img, (1, 2, 0)).astype(DTYPE)  # (H, W, C)


def to_tanh_range(x, lo, hi):
    x = np.clip(x, lo, hi)
    return 2.0 * (x - lo) / (hi - lo + 1e-9) - 1.0


def s1_transform_db_tanh(x_hwc):
    out = np.empty_like(x_hwc, dtype=DTYPE)

    lo, hi = S1_CLIP["VV"]
    out[..., 0] = to_tanh_range(x_hwc[..., 0], lo, hi)

    lo, hi = S1_CLIP["VH"]
    out[..., 1] = to_tanh_range(x_hwc[..., 1], lo, hi)

    return out


def s2_transform_13bands_tanh(y_hwc):
    assert y_hwc.shape[-1] == 13, f"Expected 13 bands, got {y_hwc.shape[-1]}"

    out = np.empty_like(y_hwc, dtype=DTYPE)

    for b in range(13):
        lo, hi = S2_CLIP[b + 1]
        out[..., b] = to_tanh_range(y_hwc[..., b], lo, hi)

    return out


def crop_grid(H, W, crop=CROP_SIZE):
    assert H % crop == 0 and W % crop == 0, (
        f"Image dims ({H}x{W}) not divisible by {crop}."
    )

    for hh in range(0, H, crop):
        for ww in range(0, W, crop):
            yield hh, ww


def hash_score(path):
    """
    Deterministic pseudo-random score based on file basename.
    This gives stable splitting across repeated runs.
    """
    h = hashlib.md5(os.path.basename(path).encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def build_s2_path_from_s1_path(s1_path, s1_dir, s2_dir):
    """
    Build the correct S2 path for a SEN12MS S1 file.

    Example conversion:

    S1:
    ROIs1970_fall_s1/s1_42/ROIs1970_fall_s1_42_p327.tif

    S2:
    ROIs1970_fall_s2/s2_42/ROIs1970_fall_s2_42_p327.tif
    """

    rel = os.path.relpath(s1_path, s1_dir)
    parts = rel.split(os.sep)

    converted_parts = []

    for part in parts:
        # Convert inner folders like s1_42 to s2_42
        if part.startswith("s1_"):
            part = "s2_" + part[len("s1_"):]

        # Convert filenames like ROIs1970_fall_s1_42_p327.tif
        part = part.replace("_s1_", "_s2_")

        converted_parts.append(part)

    rel_s2 = os.path.join(*converted_parts)
    s2_path = os.path.join(s2_dir, rel_s2)

    return s2_path


def detect_season_from_path(path, season_list):
    for season in season_list:
        if f"_{season}_s1" in path:
            return season

    return None


# -------------------- DISCOVERY & PAIRING --------------------

pairs_by_season = defaultdict(list)

samples_per_season = None

if MAX_SAMPLES is not None:
    samples_per_season = max(1, MAX_SAMPLES // len(seasons))

print(f"Searching for .tif files across seasons: {seasons}...")
print(f"Dataset directory: {data_dir}")

for season in seasons:
    s1_dirs = sorted(glob.glob(os.path.join(data_dir, f"ROIs*_{season}_s1")))
    s2_dirs = sorted(glob.glob(os.path.join(data_dir, f"ROIs*_{season}_s2")))

    if not s1_dirs or not s2_dirs:
        print(
            f"WARNING: Could not find S1/S2 directories for season '{season}' "
            f"in '{data_dir}' - skipping."
        )
        print(f"  Found S1 dirs: {s1_dirs}")
        print(f"  Found S2 dirs: {s2_dirs}")
        continue

    s1_dir = s1_dirs[0]
    s2_dir = s2_dirs[0]

    print(f"\nSeason '{season}':")
    print(f"  S1 dir: {s1_dir}")
    print(f"  S2 dir: {s2_dir}")

    s1_season_files = []

    for root, _, files in os.walk(s1_dir):
        for file in files:
            if file.lower().endswith(".tif"):
                s1_season_files.append(os.path.join(root, file))

    s1_season_files.sort(key=lambda f: os.path.basename(f))

    print(f"  Found S1 .tif files: {len(s1_season_files)}")

    if len(s1_season_files) == 0:
        print(f"  WARNING: No .tif files found inside {s1_dir}")
        continue

    print("  Example S1 files:")
    for example in s1_season_files[:3]:
        print(f"    {example}")

    season_pairs = []
    missing_count = 0

    for s1_path in s1_season_files:
        s2_path = build_s2_path_from_s1_path(s1_path, s1_dir, s2_dir)

        if os.path.exists(s2_path):
            score = hash_score(s1_path)
            season_pairs.append((score, s1_path, s2_path))
        else:
            missing_count += 1

            if missing_count <= PRINT_MISSING_EXAMPLES:
                print("  Missing expected S2 pair:")
                print(f"    S1: {s1_path}")
                print(f"    S2: {s2_path}")

    # Deterministic random order for this season
    season_pairs.sort(key=lambda t: t[0])

    available_pairs_before_slicing = len(season_pairs)

    # Limit number of samples per season
    if samples_per_season is not None:
        season_pairs = season_pairs[:samples_per_season]

    print(f"  Paired images before slicing: {available_pairs_before_slicing}")
    print(f"  Paired images after slicing:  {len(season_pairs)}")
    print(f"  Missing pairs: {missing_count}")

    pairs_by_season[season].extend(season_pairs)


# -------------------- CHECK DISCOVERY RESULT --------------------

total_pairs_found = sum(len(v) for v in pairs_by_season.values())

if total_pairs_found == 0:
    print("\nERROR: No paired S1 and S2 images found across any seasons.")
    print("\nPlease run these commands to inspect the real folder structure:")
    print(f"find {data_dir} -path '*fall_s1*' -name '*.tif' | head -10")
    print(f"find {data_dir} -path '*fall_s2*' -name '*.tif' | head -10")
    sys.exit(1)

print("\nPaired images per season after slicing:")
for season in seasons:
    print(f"  {season}: {len(pairs_by_season[season])}")

print(f"\nTotal paired images before split: {total_pairs_found}")


# -------------------- SEASON-STRATIFIED TRAIN / VAL / TEST SPLIT --------------------

train_pairs = []
val_pairs = []
test_pairs = []

print("\nSeason-stratified split:")

for season in seasons:
    season_pairs = pairs_by_season[season]

    if len(season_pairs) == 0:
        print(f"  WARNING: Season '{season}' has 0 pairs and will not contribute.")
        continue

    # Already sorted by deterministic hash, but sort again for safety
    season_pairs.sort(key=lambda t: t[0])

    M_season = len(season_pairs)

    n_train_season = int(round(P_TRAIN * M_season))
    n_val_season = int(round(P_VAL * M_season))
    n_test_season = int(round(P_TEST * M_season))

    # Ensure totals do not exceed this season's count
    n_train_season = min(n_train_season, M_season)
    n_val_season = min(n_val_season, max(0, M_season - n_train_season))
    n_test_season = min(
        n_test_season,
        max(0, M_season - n_train_season - n_val_season)
    )

    unused_season = M_season - (
        n_train_season + n_val_season + n_test_season
    )

    season_train = season_pairs[0:n_train_season]
    season_val = season_pairs[
        n_train_season:n_train_season + n_val_season
    ]
    season_test = season_pairs[
        n_train_season + n_val_season:
        n_train_season + n_val_season + n_test_season
    ]

    train_pairs.extend(season_train)
    val_pairs.extend(season_val)
    test_pairs.extend(season_test)

    print(
        f"  {season}: total={M_season}, "
        f"train={len(season_train)}, "
        f"val={len(season_val)}, "
        f"test={len(season_test)}, "
        f"unused={unused_season}"
    )

# Mix order inside final splits using the deterministic hash score
train_pairs.sort(key=lambda t: t[0])
val_pairs.sort(key=lambda t: t[0])
test_pairs.sort(key=lambda t: t[0])

M = len(train_pairs) + len(val_pairs) + len(test_pairs)

print(f"\nTotal paired images to process after split: {M}")
print(
    f"Final image quotas -> train: {len(train_pairs)}, "
    f"val: {len(val_pairs)}, test: {len(test_pairs)}"
)

print("\nExample train pairs:")
for _, s1_example, s2_example in train_pairs[:3]:
    print("PAIR:")
    print(f"  S1: {s1_example}")
    print(f"  S2: {s2_example}")

print("\nExample val pairs:")
for _, s1_example, s2_example in val_pairs[:3]:
    print("PAIR:")
    print(f"  S1: {s1_example}")
    print(f"  S2: {s2_example}")

print("\nExample test pairs:")
for _, s1_example, s2_example in test_pairs[:3]:
    print("PAIR:")
    print(f"  S1: {s1_example}")
    print(f"  S2: {s2_example}")


# -------------------- SPLIT SEASON SUMMARY CHECK --------------------

def count_seasons_in_pairs(pair_list, season_list):
    counts = {season: 0 for season in season_list}

    for _, s1_path, _ in pair_list:
        detected = detect_season_from_path(s1_path, season_list)

        if detected is not None:
            counts[detected] += 1

    return counts


print("\nSeason distribution inside each final split:")
print(f"  train: {count_seasons_in_pairs(train_pairs, seasons)}")
print(f"  val:   {count_seasons_in_pairs(val_pairs, seasons)}")
print(f"  test:  {count_seasons_in_pairs(test_pairs, seasons)}")

# -------------------- SAVE SPLIT FILE LISTS --------------------

os.makedirs(output_dir, exist_ok=True)

save_split_csv("train", train_pairs, output_dir)
save_split_csv("val", val_pairs, output_dir)
save_split_csv("test", test_pairs, output_dir)

# -------------------- VALIDATE IMAGE DIMENSIONS --------------------

# Use the first available training pair for dimensions.
# If train is empty for any reason, fallback to val/test.
dimension_reference_pair = None

if len(train_pairs) > 0:
    dimension_reference_pair = train_pairs[0]
elif len(val_pairs) > 0:
    dimension_reference_pair = val_pairs[0]
elif len(test_pairs) > 0:
    dimension_reference_pair = test_pairs[0]

if dimension_reference_pair is None:
    print("ERROR: No pairs available after splitting.")
    sys.exit(1)

with rasterio.open(dimension_reference_pair[1]) as src:
    H, W = src.height, src.width

if (H % CROP_SIZE) or (W % CROP_SIZE):
    print(f"ERROR: Image dims ({H}x{W}) not divisible by {CROP_SIZE}.")
    sys.exit(1)

crops_per_image = (H // CROP_SIZE) * (W // CROP_SIZE)

print(f"\nEach image -> {crops_per_image} crops of {CROP_SIZE}x{CROP_SIZE}")


# -------------------- BUILD SPLITS DICTIONARY --------------------

splits = {
    "train": ([p[1] for p in train_pairs], [p[2] for p in train_pairs]),
    "val": ([p[1] for p in val_pairs], [p[2] for p in val_pairs]),
    "test": ([p[1] for p in test_pairs], [p[2] for p in test_pairs]),
}


# -------------------- COUNT CROPS PER SPLIT --------------------

counts = {
    split_name: len(split_data[0]) * crops_per_image
    for split_name, split_data in splits.items()
}

print("\nPlanned crops per split:", counts)

bytes_per_crop = CROP_SIZE * CROP_SIZE * (2 + 13) * 4
est_total_gb = sum(counts.values()) * bytes_per_crop / (1024 ** 3)
avail_gb = shutil.disk_usage(".").free / (1024 ** 3)

print(f"Estimated final size uncompressed: ~{est_total_gb:.2f} GB")
print(f"Free disk space here: {avail_gb:.2f} GB")

if est_total_gb > avail_gb:
    print("WARNING: Estimated output size is larger than available disk space.")


# -------------------- ALLOCATE NPY FILES --------------------

os.makedirs(output_dir, exist_ok=True)

paths = {
    "train": (
        os.path.join(output_dir, "X_train.npy"),
        os.path.join(output_dir, "y_train.npy"),
    ),
    "val": (
        os.path.join(output_dir, "X_val.npy"),
        os.path.join(output_dir, "y_val.npy"),
    ),
    "test": (
        os.path.join(output_dir, "X_test.npy"),
        os.path.join(output_dir, "y_test.npy"),
    ),
}

mmaps = {}

for split in ["train", "val", "test"]:
    n = counts[split]
    X_path, y_path = paths[split]

    X_mm = open_memmap(
        X_path,
        mode="w+",
        dtype=DTYPE,
        shape=(n, CROP_SIZE, CROP_SIZE, 2),
    )

    y_mm = open_memmap(
        y_path,
        mode="w+",
        dtype=DTYPE,
        shape=(n, CROP_SIZE, CROP_SIZE, 13),
    )

    mmaps[split] = {
        "X": X_mm,
        "y": y_mm,
        "cursor": 0,
    }


# -------------------- SAVE CLIP CONFIGS AND SPLIT METADATA --------------------

with open(os.path.join(output_dir, "S1_clip.json"), "w") as f:
    json.dump(S1_CLIP, f, indent=2)

with open(os.path.join(output_dir, "S2_clip.json"), "w") as f:
    json.dump({str(k): v for k, v in S2_CLIP.items()}, f, indent=2)

split_metadata = {
    "data_dir": data_dir,
    "seasons": seasons,
    "output_dir": output_dir,
    "max_samples": MAX_SAMPLES,
    "samples_per_season": samples_per_season,
    "p_train": P_TRAIN,
    "p_val": P_VAL,
    "p_test": P_TEST,
    "crop_size": CROP_SIZE,
    "crops_per_image": crops_per_image,
    "image_counts": {
        "train": len(train_pairs),
        "val": len(val_pairs),
        "test": len(test_pairs),
    },
    "crop_counts": counts,
    "season_distribution": {
        "train": count_seasons_in_pairs(train_pairs, seasons),
        "val": count_seasons_in_pairs(val_pairs, seasons),
        "test": count_seasons_in_pairs(test_pairs, seasons),
    },
}

with open(os.path.join(output_dir, "split_metadata.json"), "w") as f:
    json.dump(split_metadata, f, indent=2)


# -------------------- PROCESS, CROP, NORMALIZE, WRITE --------------------

print("\nProcessing, cropping and normalizing into .npy files...")


def process_split(split):
    s1_list, s2_list = splits[split]

    Xmm = mmaps[split]["X"]
    ymm = mmaps[split]["y"]
    cur = mmaps[split]["cursor"]

    for s1f, s2f in tqdm(
        zip(s1_list, s2_list),
        total=len(s1_list),
        desc=f"{split} pairs",
    ):
        X_full = load_tif_as_hwc(s1f)  # S1: expected shape (H, W, 2)
        y_full = load_tif_as_hwc(s2f)  # S2: expected shape (H, W, 13)

        if X_full.shape[-1] != 2 or y_full.shape[-1] != 13:
            print("Skipping malformed pair:")
            print(f"  S1: {s1f}, shape={X_full.shape}")
            print(f"  S2: {s2f}, shape={y_full.shape}")
            continue

        for hh, ww in crop_grid(H, W, CROP_SIZE):
            X_patch = X_full[hh:hh + CROP_SIZE, ww:ww + CROP_SIZE, :]
            y_patch = y_full[hh:hh + CROP_SIZE, ww:ww + CROP_SIZE, :]

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


# -------------------- VALIDATE CURSOR COUNTS --------------------

for split in ["train", "val", "test"]:
    expected = counts[split]
    written = mmaps[split]["cursor"]

    if written != expected:
        print(
            f"ERROR: Cursor mismatch for {split}: "
            f"expected {expected}, wrote {written}"
        )
        sys.exit(1)


# -------------------- FLUSH FILES --------------------

for split in ["train", "val", "test"]:
    mmaps[split]["X"].flush()
    mmaps[split]["y"].flush()

mmaps = None
gc.collect()


# -------------------- LIGHT SANITY CHECKS --------------------

print("\n=== Sanity checks sampled ===")

rng = np.random.default_rng(0)


def sampled_check(name, path, samples=SAMPLE_CHECKS_PER_SPLIT):
    arr = np.load(path, mmap_mode="r")
    n = arr.shape[0]

    if n == 0:
        print(f"{name}: empty")
        return

    idx = rng.choice(n, size=min(samples, n), replace=False)

    mn = np.inf
    mx = -np.inf
    nans = 0

    for i in idx:
        a = arr[i]
        mn = min(mn, float(a.min()))
        mx = max(mx, float(a.max()))
        nans += int(np.isnan(a).sum())

    print(
        f"{name}: shape={arr.shape}, "
        f"sampled_min={mn:.6f}, sampled_max={mx:.6f}, "
        f"sampled_NaNs={nans}"
    )

    assert mn >= -1.0001 and mx <= 1.0001, (
        f"{name} out of [-1, 1] bounds!"
    )

    assert nans == 0, f"{name} contains NaNs!"


for split in ["train", "val", "test"]:
    Xp, yp = paths[split]
    sampled_check(f"X_{split}", Xp)
    sampled_check(f"y_{split}", yp)


# -------------------- BOUNDARY MATH CHECKS --------------------

def spot_boundary_S1():
    vv_lo, vv_hi = S1_CLIP["VV"]
    vh_lo, vh_hi = S1_CLIP["VH"]

    vv_test = to_tanh_range(
        np.array([vv_lo, vv_hi], dtype=DTYPE),
        vv_lo,
        vv_hi,
    )

    vh_test = to_tanh_range(
        np.array([vh_lo, vh_hi], dtype=DTYPE),
        vh_lo,
        vh_hi,
    )

    assert np.allclose(vv_test, [-1.0, 1.0], atol=1e-5)
    assert np.allclose(vh_test, [-1.0, 1.0], atol=1e-5)

    print("S1 boundary mapping OK.")


def spot_boundary_S2():
    for b in range(13):
        lo, hi = S2_CLIP[b + 1]

        test = to_tanh_range(
            np.array([lo, hi], dtype=DTYPE),
            lo,
            hi,
        )

        assert np.allclose(test, [-1.0, 1.0], atol=1e-5)

    print("S2 per-band boundary mappings OK.")


spot_boundary_S1()
spot_boundary_S2()


# -------------------- DONE --------------------

print("\nAll tests sampled passed ✅")
print("Saved:")
print(f"  {output_dir}/X_train.npy, y_train.npy")
print(f"  {output_dir}/X_val.npy,   y_val.npy")
print(f"  {output_dir}/X_test.npy,  y_test.npy")
print("Clip configs:")
print(f"  {output_dir}/S1_clip.json")
print(f"  {output_dir}/S2_clip.json")
print("Split metadata:")
print(f"  {output_dir}/split_metadata.json")