# Per-epoch probe on a fixed set of validation patches, shared by every experiment arm.
#
# Two rules this module exists to enforce:
#   1. It never touches the global RNG. A DataLoader iterator draws from it even with
#      shuffle=False, so the patches are sliced straight from the memmaps instead and
#      the bootstrap uses its own Generator. Otherwise the training shuffle shifts and
#      A_base stops reproducing the reference run.
#   2. It never runs a module in train mode. A train-mode forward advances spectral
#      norm's power iteration and BatchNorm's running statistics, which would change
#      the weights being measured.

import csv
import os
from collections import defaultdict

import numpy as np
import torch

PROBE_N = 128
HF_SIGMA = 1.0
N_BOOT = 1000
BOOT_SEED = 12345
CI_ALPHA = 0.05

# VIS B2-B4, RE B5-B7, NIR B8/B8A, SWIR B11/B12. Excludes the 60 m bands.
LAND_BANDS = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]


def _gaussian_kernel1d(sigma, device):
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _blur(x, sigma):
    """Separable Gaussian over every channel of an NHWC tensor."""
    k = _gaussian_kernel1d(sigma, x.device)
    r = (k.numel() - 1) // 2

    t = x.permute(0, 3, 1, 2)
    c = t.shape[1]
    t = torch.nn.functional.pad(t, (r, r, r, r), mode="reflect")
    t = torch.nn.functional.conv2d(t, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    t = torch.nn.functional.conv2d(t, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return t.permute(0, 2, 3, 1)


def _load_scene_ids(data_dir, split, n_patches):
    csv_path = os.path.join(data_dir, f"{split}_pairs.csv")
    with open(csv_path, newline="") as f:
        rows = [f"{r['roi_group']}_{r['season']}_{r['roi_id']}" for r in csv.DictReader(f)]

    if n_patches % len(rows) != 0:
        raise ValueError(f"{n_patches} patches is not a multiple of {len(rows)} CSV rows")

    crops_per_image = n_patches // len(rows)
    return [rows[i // crops_per_image] for i in range(n_patches)]


def _scene_derangement(scenes):
    """Permutation where no patch keeps its own SAR and none draws from its own scene.

    Sorting by scene makes each scene contiguous, so rolling by the largest scene size
    moves every element clear of its own block.
    """
    n = len(scenes)
    counts = defaultdict(int)
    for s in scenes:
        counts[s] += 1

    order = np.argsort(np.asarray(scenes), kind="stable")
    perm = np.empty(n, dtype=np.int64)
    perm[order] = np.roll(order, max(counts.values()))
    bad = sum(scenes[perm[i]] == scenes[i] for i in range(n))
    return perm, bad


def load_probe(data_dir, split="val", n=PROBE_N, channel_select=None):
    """Fixed probe patches as CPU tensors, plus a scene-disjoint SAR permutation."""
    X = np.load(os.path.join(data_dir, f"X_{split}.npy"), mmap_mode="r")
    Y = np.load(os.path.join(data_dir, f"y_{split}.npy"), mmap_mode="r")

    idx = np.linspace(0, len(X) - 1, min(n, len(X))).astype(int)
    x = np.asarray(X[idx], dtype=np.float32)
    if channel_select is not None:
        x = x[..., channel_select:channel_select + 1]

    scenes_all = _load_scene_ids(data_dir, split, len(X))
    scenes = [scenes_all[i] for i in idx]
    perm, bad = _scene_derangement(scenes)

    return {
        "indices": idx,
        "X": torch.from_numpy(x),
        "Y": torch.from_numpy(np.asarray(Y[idx], dtype=np.float32)),
        "perm": perm,
        "same_scene_violations": int(bad),
        "n_scenes": len(set(scenes)),
    }


def _bootstrap_deg(d_real, d_fake, d_shuf, n_boot=N_BOOT, seed=BOOT_SEED, alpha=CI_ALPHA):
    rng = np.random.default_rng(seed)
    n = len(d_real)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, n, n)
        lo, hi = d_fake[s].mean(), d_real[s].mean()
        draws[i] = 1.0 - (d_shuf[s].mean() - lo) / (hi - lo)
    return (float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


@torch.no_grad()
def run_probe(netG, netD, probe, device, batch=16, d_noise=None):
    """Discriminator and generator statistics on the fixed probe patches.

    Returns the per-epoch aggregates plus the raw per-patch D scores, which criterion C
    pools across epochs before bootstrapping.

    d_noise, when given, is applied to the optical channels of every D input so the
    probe sees the same input distribution D was trained on.
    """
    modes = (netG.training, netD.training)
    netG.eval()
    netD.eval()

    try:
        d_real, d_fake, d_shuf = [], [], []
        hf_num, hf_den, sd_num, sd_den, hf_corr = [], [], [], [], []

        for s in range(0, len(probe["indices"]), batch):
            sl = slice(s, s + batch)
            real_A = probe["X"][sl].to(device, non_blocking=True)
            real_B = probe["Y"][sl].to(device, non_blocking=True)
            shuf_A = probe["X"][probe["perm"][sl]].to(device, non_blocking=True)

            fake_B = netG(real_A)

            def score(a, b):
                b = b if d_noise is None else d_noise(b)
                return netD(torch.cat((a, b), -1).permute(0, 3, 1, 2)).mean((1, 2, 3))

            d_real.append(score(real_A, real_B).cpu().numpy())
            d_fake.append(score(real_A, fake_B).cpu().numpy())
            d_shuf.append(score(shuf_A, real_B).cpu().numpy())

            f_hf = fake_B - _blur(fake_B, HF_SIGMA)
            r_hf = real_B - _blur(real_B, HF_SIGMA)
            hf_num.append(f_hf.std((1, 2)).cpu().numpy())
            hf_den.append(r_hf.std((1, 2)).cpu().numpy())
            sd_num.append(fake_B.std((1, 2)).cpu().numpy())
            sd_den.append(real_B.std((1, 2)).cpu().numpy())

            # Ratio alone cannot tell real detail from hallucinated detail: noise sprayed
            # over smooth water scores as well as texture that matches the scene.
            cov = (f_hf * r_hf).mean((1, 2)) - f_hf.mean((1, 2)) * r_hf.mean((1, 2))
            hf_corr.append(
                (cov / (f_hf.std((1, 2)) * r_hf.std((1, 2)) + 1e-12)).cpu().numpy())

        d_real = np.concatenate(d_real)
        d_fake = np.concatenate(d_fake)
        d_shuf = np.concatenate(d_shuf)

        hf = np.concatenate(hf_num) / (np.concatenate(hf_den) + 1e-12)
        sd = np.concatenate(sd_num) / (np.concatenate(sd_den) + 1e-12)
        corr = np.concatenate(hf_corr)
        land = [b for b in LAND_BANDS if b < hf.shape[1]]

        deg = 1.0 - (d_shuf.mean() - d_fake.mean()) / (d_real.mean() - d_fake.mean())
        ci_lo, ci_hi = _bootstrap_deg(d_real, d_fake, d_shuf)

        return {
            "probe_D_out_real": float(d_real.mean()),
            "probe_D_out_fake": float(d_fake.mean()),
            "probe_hf_ratio": float(hf[:, land].mean()),
            "probe_hf_corr": float(corr[:, land].mean()),
            "probe_std_ratio": float(sd[:, land].mean()),
            "probe_deg_shuffle": float(deg),
            "probe_deg_shuffle_lo": ci_lo,
            "probe_deg_shuffle_hi": ci_hi,
        }, {"d_real": d_real, "d_fake": d_fake, "d_shuf": d_shuf}
    finally:
        netG.train(modes[0])
        netD.train(modes[1])
