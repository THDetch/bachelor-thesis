"""KID and spectral-index metrics on the full validation split, per arm.

Frozen spec: see the 2026-11-14 (third) amendment in success_criteria.md. Nothing here
is tunable after the fact - the stretch windows come from the training split via
make_kid_stretch.py, and the band triples, resize and water threshold are fixed below.

    python eval_kid_indices.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torchmetrics.image.kid import KernelInceptionDistance

THIS_DIR = Path(__file__).parent
gan_repo_path = THIS_DIR / "mymodels/pytorch-CycleGAN-and-pix2pix"
if gan_repo_path.is_dir():
    sys.path.insert(0, str(gan_repo_path))

from mymodels.cp2pwrapper import Pix2PixWrapper

DATA_DIR = "dataset_all_seasons_scenesplit"
SPLIT = "val"
MODEL_NAME = "pix2pix"
NORM_G = "batch"
EPOCH = 20
BATCH = 32
GPU_ID = 0

ARMS = ["A_base", "A0", "A1", "A3", "A3-g3", "A3-g10", "L1-only"]

# Inception takes 3 of 13 bands. True colour, then false colour so NIR is measured too.
TRIPLES = {"rgb": ("B4", "B3", "B2"), "false_colour": ("B8", "B4", "B3")}
KID_SUBSET_SIZE = 1000
KID_SUBSETS = 100
KID_SEED = 42
INCEPTION_SIZE = 299

B = {"B2": 1, "B3": 2, "B4": 3, "B8": 7}

# Reflectance floor for the index denominators. Below this the ratio is dominated by
# sensor noise and its sign is arbitrary.
INDEX_DN_FLOOR = 100.0
WATER_NDWI = 0.0


def to_dn(x, clip, idx):
    lo, hi = clip[idx + 1]
    return lo + (x + 1.0) * 0.5 * (hi - lo)


def to_uint8(dn, lo, hi):
    return ((dn - lo) / (hi - lo)).clamp(0, 1).mul(255).round().to(torch.uint8)


def stack_triple(dn, names, stretch):
    chans = [to_uint8(dn[..., B[n]], stretch[n]["p2_dn"], stretch[n]["p98_dn"])
             for n in names]
    x = torch.stack(chans, dim=1).float()
    x = torch.nn.functional.interpolate(
        x, size=(INCEPTION_SIZE, INCEPTION_SIZE), mode="bilinear",
        align_corners=False, antialias=True)
    return x.round().clamp(0, 255).to(torch.uint8)


def indices(dn):
    nir, red, green = dn[..., B["B8"]], dn[..., B["B4"]], dn[..., B["B3"]]
    ndvi = (nir - red) / (nir + red).clamp(min=1e-6)
    ndwi = (green - nir) / (green + nir).clamp(min=1e-6)
    return ndvi, ndwi


def main():
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(DATA_DIR, "kid_stretch.json")) as f:
        stretch = json.load(f)["bands"]
    with open(os.path.join(DATA_DIR, "S2_clip.json")) as f:
        clip = {int(k): tuple(v) for k, v in json.load(f).items()}

    X = np.load(os.path.join(DATA_DIR, f"X_{SPLIT}.npy"), mmap_mode="r")
    Y = np.load(os.path.join(DATA_DIR, f"y_{SPLIT}.npy"), mmap_mode="r")
    n, h, _, c_in = X.shape
    c_out = Y.shape[-1]
    print(f"{SPLIT} split: {n} patches, {c_in} -> {c_out} channels, {h}px\n")

    results = {}
    for arm in ARMS:
        save_dir = f"trained_arm_{arm}/{MODEL_NAME}"
        g_path = os.path.join(save_dir, f"{MODEL_NAME}_epoch_{EPOCH:03d}_netG.pth")
        if not os.path.isfile(g_path):
            ck = os.path.join(save_dir, "checkpoint_last.pt")
            if not os.path.isfile(ck):
                print(f"{arm}: no epoch-{EPOCH} generator, skipping")
                continue
            state = torch.load(ck, map_location=device, weights_only=False)["netG"]
        else:
            state = torch.load(g_path, map_location=device)

        netG = Pix2PixWrapper(in_channels=c_in, out_channels=c_out,
                              img_size=h, norm=NORM_G).to(device)
        netG.load_state_dict(state)
        netG.eval()

        torch.manual_seed(KID_SEED)
        kid = {k: KernelInceptionDistance(subset_size=KID_SUBSET_SIZE,
                                          subsets=KID_SUBSETS,
                                          normalize=False).to(device)
               for k in TRIPLES}

        acc = {k: dict(n=0.0, ae=0.0, err=0.0) for k in
               ("ndvi", "ndwi", "ndvi_water", "ndwi_water",
                "ndvi_land", "ndwi_land")}
        masked_px = 0.0
        total_px = 0.0

        with torch.no_grad():
            for s in range(0, n, BATCH):
                sl = slice(s, min(s + BATCH, n))
                real_A = torch.from_numpy(np.ascontiguousarray(X[sl])).to(device)
                real_B = torch.from_numpy(np.ascontiguousarray(Y[sl])).to(device)
                fake_B = netG(real_A)

                r_dn = torch.stack(
                    [to_dn(real_B[..., i], clip, i) for i in range(c_out)], -1)
                f_dn = torch.stack(
                    [to_dn(fake_B[..., i], clip, i) for i in range(c_out)], -1)

                for key, names in TRIPLES.items():
                    kid[key].update(stack_triple(r_dn, names, stretch), real=True)
                    kid[key].update(stack_triple(f_dn, names, stretch), real=False)

                r_ndvi, r_ndwi = indices(r_dn)
                f_ndvi, f_ndwi = indices(f_dn)

                nir, red, green = (r_dn[..., B["B8"]], r_dn[..., B["B4"]],
                                   r_dn[..., B["B3"]])
                valid = ((nir + red) >= INDEX_DN_FLOOR) & \
                        ((green + nir) >= INDEX_DN_FLOOR)
                water = valid & (r_ndwi > WATER_NDWI)
                land = valid & ~(r_ndwi > WATER_NDWI)
                masked_px += float((~valid).sum())
                total_px += float(valid.numel())

                for name, f, r in (("ndvi", f_ndvi, r_ndvi), ("ndwi", f_ndwi, r_ndwi)):
                    for suffix, m in (("", valid), ("_water", water), ("_land", land)):
                        d = (f - r)[m]
                        a = acc[name + suffix]
                        a["n"] += d.numel()
                        a["ae"] += float(d.abs().sum())
                        a["err"] += float(d.sum())

        row = {"epoch": EPOCH, "split": SPLIT, "n_patches": int(n),
               "kid_subset_size": KID_SUBSET_SIZE, "kid_subsets": KID_SUBSETS,
               "index_dn_floor": INDEX_DN_FLOOR,
               "masked_fraction": masked_px / max(total_px, 1.0)}
        for key in TRIPLES:
            m, sd = kid[key].compute()
            row[f"kid_{key}_mean"] = float(m)
            row[f"kid_{key}_std"] = float(sd)
        for name, a in acc.items():
            row[f"{name}_mae"] = a["ae"] / max(a["n"], 1.0)
            row[f"{name}_bias"] = a["err"] / max(a["n"], 1.0)

        with open(os.path.join(save_dir, "kid_indices.json"), "w") as f:
            json.dump(row, f, indent=2)
        results[arm] = row

        print(f"{arm:9} KID rgb {row['kid_rgb_mean']:.4f}+-{row['kid_rgb_std']:.4f}  "
              f"false {row['kid_false_colour_mean']:.4f}"
              f"+-{row['kid_false_colour_std']:.4f}  "
              f"NDVI mae {row['ndvi_mae']:.4f} bias {row['ndvi_bias']:+.4f}  "
              f"NDWI mae {row['ndwi_mae']:.4f} bias {row['ndwi_bias']:+.4f}  "
              f"masked {row['masked_fraction']:.2%}")

    if not results:
        return
    print(f"\n{'arm':9} {'KID rgb':>9} {'KID false':>10} {'NDVI mae':>9} "
          f"{'NDVI bias':>10} {'NDWI mae':>9} {'NDWI bias':>10} "
          f"{'NDWI mae water':>15}")
    for a in sorted(results, key=lambda k: results[k]["kid_rgb_mean"]):
        r = results[a]
        print(f"{a:9} {r['kid_rgb_mean']:9.4f} {r['kid_false_colour_mean']:10.4f} "
              f"{r['ndvi_mae']:9.4f} {r['ndvi_bias']:+10.4f} "
              f"{r['ndwi_mae']:9.4f} {r['ndwi_bias']:+10.4f} "
              f"{r['ndwi_mae_water']:15.4f}")


if __name__ == "__main__":
    main()
