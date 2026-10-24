# Post-hoc discriminator diagnostics. Never updates weights, never touches training code.
#
# Decides what D's cue actually is: texture, contrast/level (radiometry), or both,
# and whether D uses the SAR condition at all.
#
#   (1) blur sweep          ID   all-band Gaussian on the real target, sigma 0.5..8
#                           OOD  per-group blur (VIS/RE/NIR/SWIR/ATM)
#   (2) band statistics     --   G vs target directly: per-patch std, HF, PSD, bias, SSIM
#   (3) moment matching     ~ID  fake corrected to the target's per-patch level/contrast
#   (4) SAR conditioning    ID   SAR deranged across scenes; does D read the condition?
#   (5) splice / LOO        OOD  kept from the previous round, relabelled
#
# ID vs OOD matters: a patch mixing real and generated bands is something D never saw,
# so splice and leave-one-out cannot separate "D reads a joint cue" from "D rejects
# inconsistent mixtures". Blur, moment matching and the SAR test stay on or near the
# real-image manifold and carry the weight of any conclusion.
#
# Usage: adjust CONFIG, then `python diagnose_discriminator.py`

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------- CONFIG --------------------

RUN_NAME = "baseline_scenesplit_sn"
MODEL_DIR = f"trained_{RUN_NAME}/pix2pix"
DATA_DIR = "dataset_all_seasons_scenesplit"
SPLIT = "val"                       # held-out scenes only

# Epoch-100 weights. The 100-epoch run predates `checkpoint_every`, so these files may
# not exist; checkpoint_last.pt holds the same epoch and is used as the fallback.
NETG_FILE = "pix2pix_epoch_100_netG.pth"
NETD_FILE = "pix2pix_epoch_100_netD.pth"
FALLBACK_CHECKPOINT = "checkpoint_last.pt"

NORM_G = "batch"
NORM_D = "instance"
USE_SPECTRAL_NORM = True

N_SAMPLES = 512
N_PANELS = 8
BATCH = 16
GPU_ID = 0
SEED = 42

BLUR_SIGMAS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)   # all-band sweep, ID
GROUP_SIGMAS = (2.0, 4.0, 8.0)                      # per-group sweep, OOD
HF_SIGMA = 1.0                                      # high-frequency residual cut
PSD_BINS = 48
SSIM_WIN_SIGMA = 1.5                                # matches torchmetrics SSIM
SSIM_C2_SQRT = 0.06                                 # sqrt(C2) for K2=0.03, L=2

N_BOOT = 1000
CI_ALPHA = 0.05
HARNESS_TOL = 0.001                                 # abort if the splice harness drifts

OUT_DIR = f"{MODEL_DIR}/diagnostics"


# -------------------- SETUP --------------------

THIS_DIR = Path(__file__).parent
gan_repo_path = THIS_DIR / "mymodels/pytorch-CycleGAN-and-pix2pix"
if gan_repo_path.is_dir():
    sys.path.insert(0, str(gan_repo_path))

from mymodels.cp2pwrapper import Pix2PixWrapper
from models.networks import define_D, init_net, apply_spectral_norm
from myutils import (
    percentile_limits_np,
    apply_stretch_np,
    _sar_rgb_from_norm_percentile,
    S2_DISPLAY_WINDOWS_DN,
    S2_DISPLAY_GAMMA,
)

BAND_NAMES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
RGB_IDX = (3, 2, 1)

BAND_GROUPS = {
    "VIS": [1, 2, 3],        # B2 B3 B4
    "RE": [4, 5, 6],         # B5 B6 B7
    "NIR": [7, 8],           # B8 B8A
    "SWIR": [11, 12],        # B11 B12
    "ATM": [0, 9, 10],       # B1 B9 B10, the 60 m bands
}

PIXEL_METRES = 10.0


def resolve_checkpoint():
    g = os.path.join(MODEL_DIR, NETG_FILE)
    d = os.path.join(MODEL_DIR, NETD_FILE)
    if os.path.isfile(g) and os.path.isfile(d):
        return {"source": "epoch_files", "netG_path": g, "netD_path": d, "epoch": 100}

    ck = os.path.join(MODEL_DIR, FALLBACK_CHECKPOINT)
    if os.path.isfile(ck):
        return {"source": "checkpoint_last", "netG_path": ck, "netD_path": ck, "epoch": None}

    raise FileNotFoundError(
        f"No weights in {MODEL_DIR}: expected {NETG_FILE}/{NETD_FILE} or {FALLBACK_CHECKPOINT}"
    )


def load_models(c_in, c_out, img_size, device, ckpt):
    netG = Pix2PixWrapper(in_channels=c_in, out_channels=c_out, img_size=img_size, norm=NORM_G)

    # Construction order must match training or the spectral-norm parameter names differ.
    netD = define_D(input_nc=c_in + c_out, ndf=64, netD="basic", norm=NORM_D)
    netD = init_net(netD, init_type="normal", init_gain=0.02)
    if USE_SPECTRAL_NORM:
        netD = apply_spectral_norm(netD)

    if ckpt["source"] == "checkpoint_last":
        blob = torch.load(ckpt["netG_path"], map_location="cpu")
        netG.load_state_dict(blob["netG"])
        netD.load_state_dict(blob["netD"])
        ckpt["epoch"] = int(blob.get("epoch", -1)) + 1
    else:
        netG.load_state_dict(torch.load(ckpt["netG_path"], map_location="cpu"))
        netD.load_state_dict(torch.load(ckpt["netD_path"], map_location="cpu"))

    # eval() also freezes the spectral-norm power iteration, so D is identical in every test.
    return netG.to(device).eval(), netD.to(device).eval()


def d_input(real_A, B):
    """Replicate exactly how the training loop feeds D: NHWC concat, then NCHW."""
    return torch.cat((real_A, B), -1).permute(0, 3, 1, 2)


def d_scores(netD, real_A, B):
    """Mean D output per patch, shape (N,). Per-patch so the bootstrap can resample."""
    with torch.no_grad():
        return netD(d_input(real_A, B)).mean(dim=(1, 2, 3)).cpu().numpy()


# -------------------- STATISTICS --------------------

def recovery_ratio(scores, d_fake, d_real, sel):
    """(mean(scores) - mean(D_fake)) / (mean(D_real) - mean(D_fake)) over patches `sel`.

    Ratios are rebuilt from resampled means rather than averaged per patch, so a patch
    whose own real/fake gap is near zero cannot blow up the estimate.
    """
    lo = d_fake[sel].mean()
    hi = d_real[sel].mean()
    return (scores[sel].mean(axis=0) - lo) / (hi - lo)


def degradation_ratio(scores, d_fake, d_real, sel):
    """How far a manipulated REAL patch falls toward fake. 1.0 = all the way."""
    return 1.0 - recovery_ratio(scores, d_fake, d_real, sel)


def bootstrap_ci(stat_fn, n, n_boot=N_BOOT, alpha=CI_ALPHA, seed=SEED):
    rng = np.random.default_rng(seed)
    draws = np.stack([stat_fn(rng.integers(0, n, n)) for _ in range(n_boot)])
    return (np.percentile(draws, 100 * alpha / 2, axis=0),
            np.percentile(draws, 100 * (1 - alpha / 2), axis=0))


def ci_of(scores, d_fake, d_real, n, fn=recovery_ratio):
    """Point estimate plus bootstrap CI for a (N,) or (N,K) score array."""
    arr = scores[:, None] if scores.ndim == 1 else scores
    full = np.arange(n)
    point = fn(arr, d_fake, d_real, full)
    lo, hi = bootstrap_ci(lambda s: fn(arr, d_fake, d_real, s), n)
    return point, lo, hi


# -------------------- BLUR --------------------

def _gaussian_kernel1d(sigma, device):
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_blur_nhwc(x, sigma, bands):
    """Separable Gaussian over `bands` only; remaining bands pass through unchanged."""
    k = _gaussian_kernel1d(sigma, x.device)
    r = (k.numel() - 1) // 2

    sub = x[..., bands].permute(0, 3, 1, 2)
    c = sub.shape[1]
    sub = torch.nn.functional.pad(sub, (r, r, r, r), mode="reflect")
    sub = torch.nn.functional.conv2d(sub, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    sub = torch.nn.functional.conv2d(sub, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)

    out = x.clone()
    out[..., bands] = sub.permute(0, 2, 3, 1)
    return out


def blur_sweep(netD, real_A, real_B, all_bands, sigmas, groups, group_sigmas):
    out = {}
    with torch.no_grad():
        for s in sigmas:
            out[("ALL", s)] = d_scores(netD, real_A, gaussian_blur_nhwc(real_B, s, all_bands))
        for name, bands in groups.items():
            for s in group_sigmas:
                out[(name, s)] = d_scores(netD, real_A, gaussian_blur_nhwc(real_B, s, bands))
    return out


# -------------------- MOMENT MATCHING --------------------

def moment_variants(fake_B, real_B):
    """Fake corrected to the target's per-patch per-band level and/or contrast.

    Isolates radiometry from spatial structure: every variant keeps the generator's
    own texture and only rewrites its first two moments.
    """
    mu_f = fake_B.mean((1, 2), keepdim=True)
    sd_f = fake_B.std((1, 2), keepdim=True) + 1e-6
    mu_r = real_B.mean((1, 2), keepdim=True)
    sd_r = real_B.std((1, 2), keepdim=True)

    variants = {
        "level": fake_B - mu_f + mu_r,
        "contrast": (fake_B - mu_f) / sd_f * sd_r + mu_f,
        "level+contrast": (fake_B - mu_f) / sd_f * sd_r + mu_r,
        # Reverse direction: real pushed onto the fake's moments.
        "reverse_real_to_fake": (real_B - mu_r) / (sd_r + 1e-6) * sd_f + mu_f,
    }

    clamped = {}
    for k, v in variants.items():
        clamped[k] = ((v < -1.0) | (v > 1.0)).float().mean().item()
        variants[k] = v.clamp(-1.0, 1.0)

    return variants, clamped


# -------------------- SAR CONDITIONING --------------------

def scene_derangement(scenes, seed=SEED):
    """Permutation where no patch keeps its own SAR and none draws from its own scene.

    Sorting by scene makes each scene contiguous, so rolling by the largest scene size
    moves every element clear of its own block.
    """
    n = len(scenes)
    counts = defaultdict(int)
    for s in scenes:
        counts[s] += 1
    shift = max(counts.values())

    if shift <= n // 2:
        order = np.argsort(np.asarray(scenes), kind="stable")
        perm = np.empty(n, dtype=np.int64)
        perm[order] = np.roll(order, shift)
        bad = sum(scenes[perm[i]] == scenes[i] for i in range(n))
        if bad == 0:
            return perm, 0

    rng = np.random.default_rng(seed)
    cand = rng.permutation(n)
    for _ in range(1000):
        bad_pos = [i for i in range(n) if scenes[cand[i]] == scenes[i]]
        if not bad_pos:
            return cand, 0
        for i in bad_pos:
            j = int(rng.integers(0, n))
            cand[i], cand[j] = cand[j], cand[i]

    return cand, sum(scenes[cand[i]] == scenes[i] for i in range(n))


def load_scene_ids(data_dir, split, n_patches):
    """Scene id per patch index, from the split CSV written by the preparation script."""
    csv_path = os.path.join(data_dir, f"{split}_pairs.csv")
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(f"{r['roi_group']}_{r['season']}_{r['roi_id']}")

    if n_patches % len(rows) != 0:
        raise ValueError(f"{n_patches} patches is not a multiple of {len(rows)} CSV rows")

    crops_per_image = n_patches // len(rows)
    return [rows[i // crops_per_image] for i in range(n_patches)]


# -------------------- BAND STATISTICS --------------------

def _local_std(x, sigma, bands):
    """Gaussian-weighted local std, matching the SSIM window."""
    mu = gaussian_blur_nhwc(x, sigma, bands)
    mu2 = gaussian_blur_nhwc(x * x, sigma, bands)
    return torch.sqrt(torch.clamp(mu2 - mu * mu, min=0.0))


def _radial_bins(h, w, device, nbins):
    """Radial frequency bins. Corner frequencies above 0.5 cyc/px go to a discard bin:
    they are sampled anisotropically and would contaminate the last real bin."""
    fy = torch.fft.fftfreq(h, device=device).view(-1, 1)
    fx = torch.fft.rfftfreq(w, device=device).view(1, -1)
    r = torch.sqrt(fy ** 2 + fx ** 2)
    idx = torch.clamp((r / 0.5 * nbins).long(), max=nbins).flatten()
    counts = torch.bincount(idx, minlength=nbins + 1).clamp(min=1).float()
    centres = (torch.arange(nbins, device=device).float() + 0.5) / nbins * 0.5
    return idx, counts, centres


def radial_psd(x, bins_idx, bins_count, window):
    """Radially averaged power spectrum per patch and band, shape (N, C, nbins)."""
    xw = (x - x.mean((1, 2), keepdim=True)) * window[None, :, :, None]
    f = torch.fft.rfft2(xw.permute(0, 3, 1, 2))
    p = (f.real ** 2 + f.imag ** 2).flatten(2)
    sums = torch.zeros(p.shape[0], p.shape[1], len(bins_count), device=x.device)
    sums.index_add_(2, bins_idx, p)
    return (sums / bins_count)[:, :, :-1]


def band_statistics(fake_B, real_B, bands, bins_idx, bins_count, window):
    """Per-patch contrast, high-frequency energy, spectra, bias and SSIM window activity.

    Per patch, never pooled across patches: pooling would hide a per-scene level error
    inside what looks like healthy global variance.
    """
    out = {}
    with torch.no_grad():
        f_hf = fake_B - gaussian_blur_nhwc(fake_B, HF_SIGMA, bands)
        r_hf = real_B - gaussian_blur_nhwc(real_B, HF_SIGMA, bands)

        sd_f = fake_B.std((1, 2))
        sd_r = real_B.std((1, 2))
        out["std_pred"] = sd_f.cpu().numpy()
        out["std_target"] = sd_r.cpu().numpy()
        out["std_ratio"] = (sd_f / (sd_r + 1e-12)).cpu().numpy()
        out["hf_pred"] = f_hf.std((1, 2)).cpu().numpy()
        out["hf_target"] = r_hf.std((1, 2)).cpu().numpy()
        out["hf_ratio"] = (f_hf.std((1, 2)) / (r_hf.std((1, 2)) + 1e-12)).cpu().numpy()
        out["mean_err"] = (fake_B - real_B).mean((1, 2)).cpu().numpy()

        psd_f = radial_psd(fake_B, bins_idx, bins_count, window)
        psd_r = radial_psd(real_B, bins_idx, bins_count, window)
        out["psd_ratio"] = (psd_f / (psd_r + 1e-20)).cpu().numpy()

        thr = SSIM_C2_SQRT
        out["ssim_inactive_pred"] = (
            (_local_std(fake_B, SSIM_WIN_SIGMA, bands) < thr).float().mean((1, 2)).cpu().numpy())
        out["ssim_inactive_target"] = (
            (_local_std(real_B, SSIM_WIN_SIGMA, bands) < thr).float().mean((1, 2)).cpu().numpy())

    return out


# -------------------- OOD TESTS (kept) --------------------

def channel_sensitivity(netD, real_A, B):
    """Mean |dD/dx| per input channel, per patch, shape (N, C_in+C_out)."""
    x = d_input(real_A, B).clone().requires_grad_(True)
    grad, = torch.autograd.grad(netD(x).mean(), x)
    return grad.abs().mean(dim=(2, 3)).detach().cpu().numpy()


def splice_in(netD, real_A, fake_B, real_B):
    """Fake patch with ONE real band spliced in, per band, plus the all-band harness."""
    cols = []
    for c in range(fake_B.shape[-1]):
        spliced = fake_B.clone()
        spliced[..., c] = real_B[..., c]
        cols.append(d_scores(netD, real_A, spliced))

    every = fake_B.clone()
    for c in range(fake_B.shape[-1]):
        every[..., c] = real_B[..., c]

    return np.stack(cols, axis=1), d_scores(netD, real_A, every)


def splice_in_cumulative(netD, real_A, fake_B, real_B, order):
    spliced = fake_B.clone()
    cols = []
    for c in order:
        spliced[..., c] = real_B[..., c]
        cols.append(d_scores(netD, real_A, spliced))
    return np.stack(cols, axis=1)


def leave_one_out(netD, real_A, fake_B, real_B):
    """Real patch with ONE band replaced by the generated band."""
    cols = []
    for c in range(real_B.shape[-1]):
        damaged = real_B.clone()
        damaged[..., c] = fake_B[..., c]
        cols.append(d_scores(netD, real_A, damaged))
    return np.stack(cols, axis=1)


# -------------------- IMAGES --------------------

def save_d_maps(netD, real_A, fake_B, real_B, rgb_u8, out_dir, n):
    with torch.no_grad():
        m_fake = netD(d_input(real_A, fake_B)).cpu().numpy()
        m_real = netD(d_input(real_A, real_B)).cpu().numpy()

    # A flat map and a clipped colormap look identical; print the range so they are not.
    print(f"D map range   fake [{m_fake.min():.4f}, {m_fake.max():.4f}]   "
          f"real [{m_real.min():.4f}, {m_real.max():.4f}]")

    for i in range(min(n, m_fake.shape[0])):
        tiles = []
        for m in (m_fake[i, 0], m_real[i, 0]):
            up = np.array(
                Image.fromarray(np.clip(m, 0.0, 1.0).astype(np.float32)).resize(
                    (rgb_u8.shape[2], rgb_u8.shape[1]), Image.NEAREST))
            heat = np.zeros((*up.shape, 3), dtype=np.uint8)
            heat[..., 0] = (255 * (1.0 - up)).astype(np.uint8)
            heat[..., 1] = (255 * up).astype(np.uint8)
            tiles.append(heat)

        canvas = np.concatenate([rgb_u8[i], tiles[0], tiles[1]], axis=1)
        Image.fromarray(canvas).save(os.path.join(out_dir, f"dmap_{i:02d}__rgb_fake_real.png"))

    return [float(m_fake.min()), float(m_fake.max())], [float(m_real.min()), float(m_real.max())]


def save_panels(sar_u8, pred, target, s2_clip, out_dir, n):
    """Prediction and target through the same fixed display window."""
    def band(arr, idx):
        lo, hi = s2_clip[idx + 1]
        return (arr[..., idx] + 1.0) * 0.5 * (hi - lo) + lo

    for i in range(min(n, pred.shape[0])):
        p_rgb = np.zeros((*pred.shape[1:3], 3), dtype=np.uint8)
        t_rgb = np.zeros_like(p_rgb)

        for ch, b in enumerate(RGB_IDX):
            t = band(target[i], b)
            p = band(pred[i], b)
            limits = S2_DISPLAY_WINDOWS_DN.get(b) or percentile_limits_np(t)
            t_rgb[..., ch] = apply_stretch_np(t, limits, gamma=S2_DISPLAY_GAMMA)
            p_rgb[..., ch] = apply_stretch_np(p, limits, gamma=S2_DISPLAY_GAMMA)

        canvas = np.concatenate([sar_u8[i], p_rgb, t_rgb], axis=1)
        Image.fromarray(canvas).save(os.path.join(out_dir, f"panel_{i:02d}__sar_pred_gt.png"))


def plot_blur_sweep(sigmas, deg, lo, hi, group_curves, out_dir):
    plt.figure(figsize=(9, 5))
    plt.plot(sigmas, deg, "o-", color="k", label="ALL bands (ID)")
    plt.fill_between(sigmas, lo, hi, color="k", alpha=0.15)
    for name, (xs, ys) in group_curves.items():
        plt.plot(xs, ys, "s--", alpha=0.8, label=f"{name} only (OOD)")
    plt.axhline(1.0, color="r", lw=0.8, ls=":")
    plt.xlabel("Gaussian sigma (px)")
    plt.ylabel("degradation toward fake")
    plt.title("Blur applied to REAL targets")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "blur_sweep.png"), dpi=200)
    plt.close()


def plot_psd(centres, psd_ratio, groups, out_dir):
    plt.figure(figsize=(9, 5))
    for name, bands in groups.items():
        plt.semilogx(centres[1:], psd_ratio[bands][:, 1:].mean(axis=0), label=name)
    plt.axhline(1.0, color="k", lw=0.8, ls=":")
    plt.xlabel("spatial frequency (cycles/px)")
    plt.ylabel("PSD(pred) / PSD(target)")
    plt.title("Radially averaged power-spectrum ratio")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "psd_ratio.png"), dpi=200)
    plt.close()


def plot_band_bars(names, std_ratio, hf_ratio, bias_dn, out_dir):
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    x = np.arange(len(names))
    ax[0].bar(x, std_ratio, color="tab:blue")
    ax[0].axhline(1.0, color="k", ls=":")
    ax[0].set_ylabel("std(pred)/std(tgt)")
    ax[1].bar(x, hf_ratio, color="tab:orange")
    ax[1].axhline(1.0, color="k", ls=":")
    ax[1].set_ylabel("HF ratio")
    ax[2].bar(x, bias_dn, color="tab:green")
    ax[2].axhline(0.0, color="k", ls=":")
    ax[2].set_ylabel("bias (DN)")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(names)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "band_statistics.png"), dpi=200)
    plt.close()


def plot_moments(labels, vals, los, his, out_dir):
    plt.figure(figsize=(7, 4.5))
    x = np.arange(len(labels))
    err = [np.array(vals) - np.array(los), np.array(his) - np.array(vals)]
    plt.bar(x, vals, color="tab:purple", yerr=err, capsize=4)
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("recovery toward real")
    plt.title("Fake corrected to the target's per-patch moments")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "moment_matching.png"), dpi=200)
    plt.close()


# -------------------- MAIN --------------------

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    X = np.load(os.path.join(DATA_DIR, f"X_{SPLIT}.npy"), mmap_mode="r")
    Y = np.load(os.path.join(DATA_DIR, f"y_{SPLIT}.npy"), mmap_mode="r")
    with open(os.path.join(DATA_DIR, "S2_clip.json")) as f:
        s2_clip = {int(k): tuple(v) for k, v in json.load(f).items()}

    idx = np.sort(np.random.default_rng(SEED).choice(
        len(X), size=min(N_SAMPLES, len(X)), replace=False))
    c_in, c_out, img_size = X.shape[-1], Y.shape[-1], X.shape[1]
    chan_names = ["VV", "VH"][:c_in] + BAND_NAMES[:c_out]
    all_bands = list(range(c_out))
    groups = {k: [b for b in v if b < c_out] for k, v in BAND_GROUPS.items()}
    groups = {k: v for k, v in groups.items() if v}

    # DN half-range per band; std scales by this, so std ratios are identical in DN.
    dn_scale = np.array([(s2_clip[b + 1][1] - s2_clip[b + 1][0]) / 2.0 for b in range(c_out)])

    ckpt = resolve_checkpoint()
    netG, netD = load_models(c_in, c_out, img_size, device, ckpt)
    print(f"{ckpt['source']} epoch {ckpt['epoch']} on {SPLIT}: {len(idx)} patches, {device}")

    scenes_all = load_scene_ids(DATA_DIR, SPLIT, len(X))
    scenes = [scenes_all[i] for i in idx]
    perm, bad = scene_derangement(scenes)
    if bad:
        print(f"[WARN] {bad} patches kept a same-scene SAR in the derangement")

    bins_idx, bins_count, centres = _radial_bins(img_size, img_size, device, PSD_BINS)
    win1d = torch.hann_window(img_size, periodic=False, device=device)
    window = torch.outer(win1d, win1d)

    acc = defaultdict(list)
    blur_acc = defaultdict(list)
    stat_acc = defaultdict(list)
    clamp_keys = ["level", "contrast", "level+contrast", "reverse_real_to_fake"]
    first = {}

    for start in range(0, len(idx), BATCH):
        sel = idx[start:start + BATCH]
        real_A = torch.from_numpy(np.asarray(X[sel], dtype=np.float32)).to(device)
        real_B = torch.from_numpy(np.asarray(Y[sel], dtype=np.float32)).to(device)
        shuf_A = torch.from_numpy(
            np.asarray(X[idx[perm[start:start + BATCH]]], dtype=np.float32)).to(device)

        with torch.no_grad():
            fake_B = netG(real_A)

        acc["fake"].append(d_scores(netD, real_A, fake_B))
        acc["real"].append(d_scores(netD, real_A, real_B))

        sp, ev = splice_in(netD, real_A, fake_B, real_B)
        acc["splice"].append(sp)
        acc["every"].append(ev)
        acc["loo"].append(leave_one_out(netD, real_A, fake_B, real_B))

        for key, v in blur_sweep(netD, real_A, real_B, all_bands,
                                 BLUR_SIGMAS, groups, GROUP_SIGMAS).items():
            blur_acc[key].append(v)

        variants, clamped = moment_variants(fake_B, real_B)
        for k, v in variants.items():
            acc[f"moment::{k}"].append(d_scores(netD, real_A, v))
        acc["clamped"].append(np.array([[clamped[k] for k in clamp_keys]]))

        acc["sar_shuf_real"].append(d_scores(netD, shuf_A, real_B))
        acc["sar_shuf_fake"].append(d_scores(netD, shuf_A, fake_B))
        acc["sar_mean_real"].append(
            d_scores(netD, real_A.mean((1, 2), keepdim=True).expand_as(real_A), real_B))

        acc["sal_fake"].append(channel_sensitivity(netD, real_A, fake_B))

        for k, v in band_statistics(fake_B, real_B, all_bands,
                                    bins_idx, bins_count, window).items():
            stat_acc[k].append(v)

        if not first:
            first = {"real_A": real_A, "fake_B": fake_B, "real_B": real_B}

    cat = {k: np.concatenate(v) for k, v in acc.items()}
    blur = {k: np.concatenate(v) for k, v in blur_acc.items()}
    stats = {k: np.concatenate(v) for k, v in stat_acc.items()}

    d_fake, d_real = cat["fake"], cat["real"]
    n = len(d_fake)
    full = np.arange(n)

    # -------- harness gate --------
    rec_all = recovery_ratio(cat["every"][:, None], d_fake, d_real, full)[0]
    print(f"\nHarness check, every band spliced: {rec_all:.4%}")
    if abs(rec_all - 1.0) > HARNESS_TOL:
        raise SystemExit(f"Harness failed ({rec_all:.4%}); splice plumbing is broken, aborting.")

    print(f"D(fake) {d_fake.mean():.4f}   D(real) {d_real.mean():.4f}   n = {n}\n")

    report = {}

    # -------- 1. blur sweep --------
    deg, deg_lo, deg_hi = [], [], []
    for s in BLUR_SIGMAS:
        p, lo, hi = ci_of(blur[("ALL", s)], d_fake, d_real, n, degradation_ratio)
        deg.append(p[0])
        deg_lo.append(lo[0])
        deg_hi.append(hi[0])

    reached = [s for s, v in zip(BLUR_SIGMAS, deg) if v >= 0.99]
    report["blur_sigma_star"] = float(reached[0]) if reached else None
    report["blur_plateau"] = float(max(deg))
    report["blur_all"] = {
        f"sigma{s}": {"deg": float(d), "ci": [float(a), float(b)], "regime": "ID"}
        for s, d, a, b in zip(BLUR_SIGMAS, deg, deg_lo, deg_hi)}

    print("(1) blur sweep, ALL bands [ID]")
    print(f"{'sigma':>6} {'deg':>9} {'95% CI':>20}")
    print("-" * 38)
    for s, d, a, b in zip(BLUR_SIGMAS, deg, deg_lo, deg_hi):
        print(f"{s:>6.1f} {d:>8.1%} [{a:>7.1%},{b:>7.1%}]")
    print(f"sigma* (deg >= 99%): {report['blur_sigma_star']}   "
          f"plateau: {report['blur_plateau']:.1%}")

    group_curves = {}
    report["blur_groups"] = {}
    print(f"\n   per-group blur [OOD]\n{'group':>6} {'sigma':>6} {'deg':>9} {'95% CI':>20}")
    print("-" * 45)
    for name in groups:
        xs, ys = [], []
        for s in GROUP_SIGMAS:
            p, lo, hi = ci_of(blur[(name, s)], d_fake, d_real, n, degradation_ratio)
            xs.append(s)
            ys.append(p[0])
            report["blur_groups"][f"{name}_sigma{s}"] = {
                "deg": float(p[0]), "ci": [float(lo[0]), float(hi[0])], "regime": "OOD"}
            print(f"{name:>6} {s:>6.1f} {p[0]:>8.1%} [{lo[0]:>7.1%},{hi[0]:>7.1%}]")
        group_curves[name] = (xs, ys)

    plot_blur_sweep(list(BLUR_SIGMAS), deg, deg_lo, deg_hi, group_curves, OUT_DIR)

    # -------- 2. band statistics --------
    std_ratio_med = np.median(stats["std_ratio"], axis=0)
    std_ratio_mean = stats["std_ratio"].mean(axis=0)
    hf_ratio_mean = stats["hf_ratio"].mean(axis=0)
    bias_dn = stats["mean_err"].mean(axis=0) * dn_scale
    bias_scene_sd_dn = stats["mean_err"].std(axis=0) * dn_scale
    psd_ratio = stats["psd_ratio"].mean(axis=0)
    ssim_in_p = stats["ssim_inactive_pred"].mean(axis=0)
    ssim_in_t = stats["ssim_inactive_target"].mean(axis=0)

    print("\n(2) band statistics, per patch then averaged [no D involved]")
    print(f"{'band':>5} {'std ratio':>10} {'(median)':>9} {'HF ratio':>9} "
          f"{'bias DN':>9} {'sd(bias)':>9} {'SSIM off tgt':>13} {'pred':>7}")
    print("-" * 80)
    for c, name in enumerate(BAND_NAMES[:c_out]):
        print(f"{name:>5} {std_ratio_mean[c]:>10.3f} {std_ratio_med[c]:>9.3f} "
              f"{hf_ratio_mean[c]:>9.3f} {bias_dn[c]:>+9.1f} {bias_scene_sd_dn[c]:>9.1f} "
              f"{ssim_in_t[c]:>12.1%} {ssim_in_p[c]:>7.1%}")

    report["band_stats"] = {
        name: {
            "std_ratio_mean": float(std_ratio_mean[c]),
            "std_ratio_median": float(std_ratio_med[c]),
            "std_pred_dn": float(stats["std_pred"].mean(axis=0)[c] * dn_scale[c]),
            "std_target_dn": float(stats["std_target"].mean(axis=0)[c] * dn_scale[c]),
            "hf_ratio_mean": float(hf_ratio_mean[c]),
            "hf_pred_dn": float(stats["hf_pred"].mean(axis=0)[c] * dn_scale[c]),
            "hf_target_dn": float(stats["hf_target"].mean(axis=0)[c] * dn_scale[c]),
            "bias_dn": float(bias_dn[c]),
            "bias_scene_sd_dn": float(bias_scene_sd_dn[c]),
            "ssim_inactive_target": float(ssim_in_t[c]),
            "ssim_inactive_pred": float(ssim_in_p[c]),
        }
        for c, name in enumerate(BAND_NAMES[:c_out])
    }
    centres_np = centres.cpu().numpy()
    report["psd_frequencies_cpp"] = [float(v) for v in centres_np]
    report["psd_wavelength_m"] = [float(PIXEL_METRES / v) if v > 0 else None for v in centres_np]
    report["psd_ratio"] = {
        name: [float(v) for v in psd_ratio[c]] for c, name in enumerate(BAND_NAMES[:c_out])}

    plot_psd(centres_np, psd_ratio, groups, OUT_DIR)
    plot_band_bars(BAND_NAMES[:c_out], std_ratio_mean, hf_ratio_mean, bias_dn, OUT_DIR)

    # -------- 3. moment matching --------
    clamp_frac = dict(zip(clamp_keys, cat["clamped"].mean(axis=0)))

    print("\n(3) moment matching [~ID]")
    print(f"{'variant':>22} {'rec':>9} {'95% CI':>20} {'clamped':>9}")
    print("-" * 64)
    m_labels, m_vals, m_lo, m_hi = [], [], [], []
    report["moment_matching"] = {}
    for k in ("level", "contrast", "level+contrast"):
        p, lo, hi = ci_of(cat[f"moment::{k}"], d_fake, d_real, n, recovery_ratio)
        flag = "  <-- >1%" if clamp_frac[k] > 0.01 else ""
        print(f"{k:>22} {p[0]:>8.1%} [{lo[0]:>7.1%},{hi[0]:>7.1%}] {clamp_frac[k]:>8.2%}{flag}")
        m_labels.append(k)
        m_vals.append(p[0])
        m_lo.append(lo[0])
        m_hi.append(hi[0])
        report["moment_matching"][k] = {
            "rec": float(p[0]), "ci": [float(lo[0]), float(hi[0])],
            "clamped_fraction": float(clamp_frac[k]), "regime": "~ID"}

    p, lo, hi = ci_of(cat["moment::reverse_real_to_fake"], d_fake, d_real, n, degradation_ratio)
    print(f"{'reverse (real->fake)':>22} {p[0]:>8.1%} [{lo[0]:>7.1%},{hi[0]:>7.1%}] "
          f"{clamp_frac['reverse_real_to_fake']:>8.2%}   (deg)")
    report["moment_matching"]["reverse_real_to_fake"] = {
        "deg": float(p[0]), "ci": [float(lo[0]), float(hi[0])],
        "clamped_fraction": float(clamp_frac["reverse_real_to_fake"]), "regime": "~ID"}

    plot_moments(m_labels, m_vals, m_lo, m_hi, OUT_DIR)

    # -------- 4. SAR conditioning --------
    deg_shuf, ds_lo, ds_hi = ci_of(cat["sar_shuf_real"], d_fake, d_real, n, degradation_ratio)
    deg_mean, dm_lo, dm_hi = ci_of(cat["sar_mean_real"], d_fake, d_real, n, degradation_ratio)

    sal = cat["sal_fake"]
    share = sal / (sal.sum(axis=1, keepdims=True) + 1e-20)
    sar_share = share[:, :c_in].sum(axis=1)
    sh_lo, sh_hi = bootstrap_ci(lambda s: np.array([sar_share[s].mean()]), n)

    print("\n(4) SAR conditioning [ID]")
    print(f"  D(real_A, real_B)      {d_real.mean():>8.4f}")
    print(f"  D(shuffled_A, real_B)  {cat['sar_shuf_real'].mean():>8.4f}")
    print(f"  D(real_A, fake_B)      {d_fake.mean():>8.4f}")
    print(f"  D(shuffled_A, fake_B)  {cat['sar_shuf_fake'].mean():>8.4f}")
    print(f"  deg_shuffle            {deg_shuf[0]:>8.1%} [{ds_lo[0]:.1%},{ds_hi[0]:.1%}]")
    print(f"  deg_sar_mean_only      {deg_mean[0]:>8.1%} [{dm_lo[0]:.1%},{dm_hi[0]:.1%}]")
    print(f"  VV+VH saliency share   {sar_share.mean():>8.1%} [{sh_lo[0]:.1%},{sh_hi[0]:.1%}]")

    report["sar_conditioning"] = {
        "d_real_A_real_B": float(d_real.mean()),
        "d_shuffled_A_real_B": float(cat["sar_shuf_real"].mean()),
        "d_real_A_fake_B": float(d_fake.mean()),
        "d_shuffled_A_fake_B": float(cat["sar_shuf_fake"].mean()),
        "deg_shuffle": float(deg_shuf[0]),
        "deg_shuffle_ci": [float(ds_lo[0]), float(ds_hi[0])],
        "deg_sar_mean_only": float(deg_mean[0]),
        "deg_sar_mean_only_ci": [float(dm_lo[0]), float(dm_hi[0])],
        "sar_saliency_share": float(sar_share.mean()),
        "sar_saliency_share_ci": [float(sh_lo[0]), float(sh_hi[0])],
        "same_scene_violations": int(bad),
        "regime": "ID",
    }

    # -------- 5. splice / LOO, relabelled OOD --------
    recovery, rec_lo, rec_hi = ci_of(cat["splice"], d_fake, d_real, n, recovery_ratio)
    loo_drop, loo_lo, loo_hi = ci_of(cat["loo"], d_fake, d_real, n, degradation_ratio)
    order = np.argsort(-recovery)

    cum_acc = []
    for start in range(0, len(idx), BATCH):
        sel = idx[start:start + BATCH]
        real_A = torch.from_numpy(np.asarray(X[sel], dtype=np.float32)).to(device)
        real_B = torch.from_numpy(np.asarray(Y[sel], dtype=np.float32)).to(device)
        with torch.no_grad():
            fake_B = netG(real_A)
        cum_acc.append(splice_in_cumulative(netD, real_A, fake_B, real_B, order))
    cum, cum_lo, cum_hi = ci_of(np.concatenate(cum_acc), d_fake, d_real, n, recovery_ratio)

    print("\n(5) splice-in and leave-one-out [OOD]")
    print(f"{'band':>5} {'splice-in':>22} {'leave-one-out':>22}")
    print("-" * 52)
    for c, name in enumerate(BAND_NAMES[:c_out]):
        print(f"{name:>5} {recovery[c]:>9.1%} [{rec_lo[c]:>6.1%},{rec_hi[c]:>6.1%}] "
              f"{loo_drop[c]:>9.1%} [{loo_lo[c]:>6.1%},{loo_hi[c]:>6.1%}]")

    rng_fake, rng_real = save_d_maps(
        netD, first["real_A"], first["fake_B"], first["real_B"],
        _sar_rgb_from_norm_percentile(first["real_A"]), OUT_DIR, N_PANELS)
    save_panels(_sar_rgb_from_norm_percentile(first["real_A"]),
                first["fake_B"].cpu().numpy(), first["real_B"].cpu().numpy(),
                s2_clip, OUT_DIR, N_PANELS)

    report.update({
        "run_name": RUN_NAME,
        "checkpoint": ckpt,
        "split": SPLIT,
        "seed": SEED,
        "n_patches": int(n),
        "n_boot": N_BOOT,
        "patch_indices": [int(v) for v in idx],
        "d_out_fake": float(d_fake.mean()),
        "d_out_real": float(d_real.mean()),
        "d_map_range": {"fake": rng_fake, "real": rng_real},
        "harness_all_bands": float(rec_all),
        "regime_labels": {
            "blur_all": "ID", "blur_groups": "OOD", "moment_matching": "~ID",
            "sar_conditioning": "ID", "splice_in": "OOD", "leave_one_out": "OOD",
            "cumulative_splice": "OOD", "band_stats": "n/a (no D)",
        },
        "splice_in": {name: {"rec": float(recovery[c]),
                             "ci": [float(rec_lo[c]), float(rec_hi[c])]}
                      for c, name in enumerate(BAND_NAMES[:c_out])},
        "leave_one_out": {name: {"deg": float(loo_drop[c]),
                                 "ci": [float(loo_lo[c]), float(loo_hi[c])]}
                          for c, name in enumerate(BAND_NAMES[:c_out])},
        "cumulative_order": [BAND_NAMES[c] for c in order],
        "cumulative_rec": [float(v) for v in cum],
        "cumulative_ci": [[float(a), float(b)] for a, b in zip(cum_lo, cum_hi)],
        "saliency_share": {nm: float(v) for nm, v in zip(chan_names, share.mean(axis=0))},
    })

    with open(os.path.join(OUT_DIR, "d_diagnostics.json"), "w") as f:
        json.dump(report, f, indent=2)

    write_summary(report, BAND_NAMES[:c_out], OUT_DIR)
    print(f"\nSaved: {OUT_DIR}")


def write_summary(r, band_names, out_dir):
    """Tables only. One line of reading per table; the reviewer draws the conclusions."""
    ck = r["checkpoint"]
    L = ["# Discriminator diagnostics\n",
         f"Run `{r['run_name']}`, {ck['source']} epoch {ck['epoch']}, {r['split']} split, "
         f"N = {r['n_patches']}, seed {r['seed']}, {r['n_boot']} bootstrap resamples.\n",
         f"Harness (all bands spliced): {r['harness_all_bands']:.2%}. "
         f"D(real) {r['d_out_real']:.4f}, D(fake) {r['d_out_fake']:.4f}.\n"]

    L.append("\n## 1. Blur sweep, all bands (ID)\n")
    L.append("| sigma (px) | deg | 95% CI |")
    L.append("|---|---|---|")
    for k, v in r["blur_all"].items():
        L.append(f"| {k.replace('sigma', '')} | {v['deg']:.1%} | "
                 f"[{v['ci'][0]:.1%}, {v['ci'][1]:.1%}] |")
    star = r["blur_sigma_star"]
    L.append(f"\nReaches 99% at sigma = {star}." if star
             else f"\nDoes not reach 99%; plateau {r['blur_plateau']:.1%}.")

    L.append("\n## 2. Moment matching (~ID)\n")
    L.append("| variant | rec / deg | 95% CI | clamped |")
    L.append("|---|---|---|---|")
    for k, v in r["moment_matching"].items():
        val = v.get("rec", v.get("deg"))
        L.append(f"| {k} | {val:.1%} | [{v['ci'][0]:.1%}, {v['ci'][1]:.1%}] | "
                 f"{v['clamped_fraction']:.2%} |")
    L.append("\nLast row is the reverse direction and is a degradation, not a recovery.")

    s = r["sar_conditioning"]
    L.append("\n## 3. SAR conditioning (ID)\n")
    L.append("| quantity | value |")
    L.append("|---|---|")
    for k in ("d_real_A_real_B", "d_shuffled_A_real_B", "d_real_A_fake_B", "d_shuffled_A_fake_B"):
        L.append(f"| {k} | {s[k]:.4f} |")
    L.append(f"| deg_shuffle | {s['deg_shuffle']:.1%} "
             f"[{s['deg_shuffle_ci'][0]:.1%}, {s['deg_shuffle_ci'][1]:.1%}] |")
    L.append(f"| deg_sar_mean_only | {s['deg_sar_mean_only']:.1%} |")
    L.append(f"| VV+VH saliency share | {s['sar_saliency_share']:.1%} "
             f"[{s['sar_saliency_share_ci'][0]:.1%}, {s['sar_saliency_share_ci'][1]:.1%}] |")
    L.append(f"\nSAR was deranged across scenes; "
             f"{s['same_scene_violations']} same-scene violations.")

    L.append("\n## 4. Band statistics (no discriminator)\n")
    L.append("| band | std ratio (mean) | (median) | HF ratio | bias DN | sd(bias) DN |")
    L.append("|---|---|---|---|---|---|")
    for nm in band_names:
        b = r["band_stats"][nm]
        L.append(f"| {nm} | {b['std_ratio_mean']:.3f} | {b['std_ratio_median']:.3f} | "
                 f"{b['hf_ratio_mean']:.3f} | {b['bias_dn']:+.1f} | {b['bias_scene_sd_dn']:.1f} |")
    L.append("\nRatios are per patch then averaged; std ratios are identical in DN and "
             "normalized units because the band scaling is affine.")

    L.append("\n## 5. SSIM window sensitivity\n")
    L.append("| band | windows below sqrt(C2), target | prediction |")
    L.append("|---|---|---|")
    for nm in band_names:
        b = r["band_stats"][nm]
        L.append(f"| {nm} | {b['ssim_inactive_target']:.1%} | {b['ssim_inactive_pred']:.1%} |")
    L.append(f"\nFraction of 11x11 Gaussian windows (sigma {SSIM_WIN_SIGMA}) with local std "
             f"below {SSIM_C2_SQRT} in normalized units.")

    with open(os.path.join(out_dir, "d_diagnostics_summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
