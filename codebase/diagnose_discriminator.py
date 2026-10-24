# Post-hoc discriminator diagnostics.
#
# Answers one question: WHAT is the discriminator using to separate real from fake?
# Run it on a trained checkpoint; it never updates weights.
#
#   (b1) per-channel input sensitivity   -> which of the 15 input channels D reacts to
#   (b2) splice-in  (fake + real band)   -> single, cumulative, and a harness check
#   (b3) leave-one-out (real + fake band)-> does ONE fake band condemn a real patch?
#   (b4) blur test  (blurred real)       -> the only test with no splicing artefacts
#   (c)  D output map overlaid on RGB    -> where in the image D is confident
#   (d)  pred vs GT under ONE stretch    -> what the images really look like
#
# CAVEAT for (b2) and (b3): a patch mixing real and generated bands is out of
# distribution for D -- it never saw one during training. So splice results cannot on
# their own separate "D reads a joint cross-band cue" from "D rejects inconsistent
# mixtures". Only (b4) keeps the input on the real-image manifold.
#
# Usage: adjust CONFIG, then `python diagnose_discriminator.py`

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# -------------------- CONFIG --------------------

RUN_NAME = "baseline_scenesplit_sn"
MODEL_DIR = f"trained_{RUN_NAME}/pix2pix"
DATA_DIR = "dataset_all_seasons_scenesplit"
SPLIT = "val"                       # held-out data, so D is judged on unseen scenes

# Checkpoint pair to analyse. Must match how the run was configured.
NETG_FILE = "pix2pix_best_netG.pth"
NETD_FILE = "pix2pix_best_netD.pth"

NORM_G = "batch"
NORM_D = "instance"
USE_SPECTRAL_NORM = True

N_SAMPLES = 512                     # patches for the sensitivity/ablation statistics
N_PANELS = 8                        # patches to render as images
BATCH = 16
GPU_ID = 0
SEED = 42

BLUR_SIGMAS = (0.5, 1.0, 2.0)       # pixels, applied to the REAL target
N_BOOT = 2000                       # bootstrap resamples over patches
CI_ALPHA = 0.05

OUT_DIR = f"{MODEL_DIR}/diagnostics_CIs"


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
    "ALL": list(range(13)),
    "VIS": [1, 2, 3],        # B2 B3 B4
    "RE": [4, 5, 6],         # B5 B6 B7
    "NIR": [7, 8],           # B8 B8A
    "SWIR": [11, 12],        # B11 B12
}


def load_models(c_in, c_out, img_size, device):
    netG = Pix2PixWrapper(in_channels=c_in, out_channels=c_out, img_size=img_size, norm=NORM_G)
    netG.load_state_dict(torch.load(os.path.join(MODEL_DIR, NETG_FILE), map_location="cpu"))

    # Same construction order as training, or the spectral-norm parameter names will not match.
    netD = define_D(input_nc=c_in + c_out, ndf=64, netD="basic", norm=NORM_D)
    netD = init_net(netD, init_type="normal", init_gain=0.02)
    if USE_SPECTRAL_NORM:
        netD = apply_spectral_norm(netD)
    netD.load_state_dict(torch.load(os.path.join(MODEL_DIR, NETD_FILE), map_location="cpu"))

    return netG.to(device).eval(), netD.to(device).eval()


def d_input(real_A, B):
    """Replicate exactly how the training loop feeds D: NHWC concat, then NCHW."""
    return torch.cat((real_A, B), -1).permute(0, 3, 1, 2)


# -------------------- (b1) INPUT SENSITIVITY --------------------

def channel_sensitivity(netD, real_A, fake_B, real_B, chan_std):
    """Mean |dD/dx| per input channel, raw and scaled by that channel's std.

    Raw magnitudes are not comparable across channels: B11 varies ~35x more than
    B10 in [-1,1], so |grad| conflates "D cares" with "this channel has range".
    """
    out = {}
    for tag, B in (("fake", fake_B), ("real", real_B)):
        x = d_input(real_A, B).clone().requires_grad_(True)
        grad, = torch.autograd.grad(netD(x).mean(), x)
        raw = grad.abs().mean(dim=(0, 2, 3)).cpu().numpy()
        out[tag] = {"raw": raw, "scaled": raw * chan_std}
    return out


def d_scores(netD, real_A, B):
    """Mean D output per patch, shape (N,). Per-patch so the bootstrap can resample."""
    with torch.no_grad():
        return netD(d_input(real_A, B)).mean(dim=(1, 2, 3)).cpu().numpy()


# -------------------- (b2) SPLICE-IN --------------------

def splice_in(netD, real_A, fake_B, real_B):
    """Fake patch with ONE real band spliced in, for each band, plus the all-band check.

    Returns raw per-patch D scores; recovery ratios are formed later so the bootstrap
    can rebuild them from resampled patches. These are C separate experiments, not a
    decomposition -- they have no reason to sum to the all-band value.
    """
    cols = []
    for c in range(fake_B.shape[-1]):
        spliced = fake_B.clone()
        spliced[..., c] = real_B[..., c]
        cols.append(d_scores(netD, real_A, spliced))

    # Splicing every band must reproduce real_B; catches a wrong axis or concat order.
    every = fake_B.clone()
    for c in range(fake_B.shape[-1]):
        every[..., c] = real_B[..., c]

    return np.stack(cols, axis=1), d_scores(netD, real_A, every)


def splice_in_cumulative(netD, real_A, fake_B, real_B, order):
    """Fake patch with the top-k bands of `order` made real simultaneously, k = 1..C."""
    spliced = fake_B.clone()
    cols = []
    for c in order:
        spliced[..., c] = real_B[..., c]
        cols.append(d_scores(netD, real_A, spliced))
    return np.stack(cols, axis=1)


# -------------------- (b3) LEAVE-ONE-OUT --------------------

def leave_one_out(netD, real_A, fake_B, real_B):
    """Real patch with ONE band replaced by the generated band.

    The reverse direction of (b2). If a single fake band is enough to condemn an
    otherwise real patch, D holds a per-band test rather than a joint one.
    """
    cols = []
    for c in range(real_B.shape[-1]):
        damaged = real_B.clone()
        damaged[..., c] = fake_B[..., c]
        cols.append(d_scores(netD, real_A, damaged))
    return np.stack(cols, axis=1)


# -------------------- (b4) BLUR TEST --------------------

def _gaussian_kernel1d(sigma, device):
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_blur_nhwc(x, sigma, bands):
    """Separable Gaussian blur over `bands` only; the remaining bands pass through."""
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


def blur_test(netD, real_A, real_B, groups, sigmas):
    """D's score on REAL targets softened by a Gaussian, per sigma and band group.

    The decisive test for "D detects missing texture": a blurred real image stays on
    the real-image manifold, so a drop cannot be blamed on splicing artefacts.
    """
    out = {}
    with torch.no_grad():
        for name, bands in groups.items():
            for s in sigmas:
                out[(name, s)] = d_scores(netD, real_A, gaussian_blur_nhwc(real_B, s, bands))
    return out


# -------------------- UNCERTAINTY --------------------

def recovery_ratio(scores, d_fake, d_real, sel):
    """(mean(scores) - mean(D_fake)) / (mean(D_real) - mean(D_fake)) over patches `sel`.

    Ratios are rebuilt from resampled means rather than averaged per patch, so a patch
    whose own real/fake gap is near zero cannot blow up the estimate.
    """
    lo = d_fake[sel].mean()
    hi = d_real[sel].mean()
    return (scores[sel].mean(axis=0) - lo) / (hi - lo)


def bootstrap_ci(stat_fn, n, n_boot=N_BOOT, alpha=CI_ALPHA, seed=SEED):
    rng = np.random.default_rng(seed)
    draws = np.stack([stat_fn(rng.integers(0, n, n)) for _ in range(n_boot)])
    return (np.percentile(draws, 100 * alpha / 2, axis=0),
            np.percentile(draws, 100 * (1 - alpha / 2), axis=0))


# -------------------- (c) D OUTPUT MAP --------------------

def save_d_maps(netD, real_A, fake_B, real_B, rgb_u8, out_dir, n):
    with torch.no_grad():
        m_fake = netD(d_input(real_A, fake_B)).cpu().numpy()
        m_real = netD(d_input(real_A, real_B)).cpu().numpy()

    # A flat map and a broken colormap look identical; print the range so they are not.
    print(f"D map range   fake [{m_fake.min():.4f}, {m_fake.max():.4f}]   "
          f"real [{m_real.min():.4f}, {m_real.max():.4f}]")

    for i in range(min(n, m_fake.shape[0])):
        tiles = []
        for m in (m_fake[i, 0], m_real[i, 0]):
            # D outputs a 30x30 patch grid; blow it up to image size for overlay.
            up = np.array(
                Image.fromarray(np.clip(m, 0.0, 1.0).astype(np.float32)).resize(
                    (rgb_u8.shape[2], rgb_u8.shape[1]), Image.NEAREST
                )
            )
            heat = np.zeros((*up.shape, 3), dtype=np.uint8)
            heat[..., 0] = (255 * (1.0 - up)).astype(np.uint8)   # red where D says "fake"
            heat[..., 1] = (255 * up).astype(np.uint8)           # green where D says "real"
            tiles.append(heat)

        canvas = np.concatenate([rgb_u8[i], tiles[0], tiles[1]], axis=1)
        Image.fromarray(canvas).save(os.path.join(out_dir, f"dmap_{i:02d}__rgb_fake_real.png"))


# -------------------- (d) SHARED-STRETCH PANELS --------------------

def save_panels(sar_u8, pred, target, s2_clip, out_dir, n):
    """Prediction and target through the same fixed display window.

    A per-image window taken from the target would be narrow on a low-contrast scene,
    so a moderate prediction bias renders as pure white or pure black and the size of
    the error is unreadable. Fixed windows keep panels comparable across scenes.
    """
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

    idx = np.sort(np.random.default_rng(SEED).choice(len(X), size=min(N_SAMPLES, len(X)), replace=False))
    c_in, c_out, img_size = X.shape[-1], Y.shape[-1], X.shape[1]
    chan_names = ["VV", "VH"][:c_in] + BAND_NAMES[:c_out]

    print(f"{MODEL_DIR}/{NETD_FILE} on {SPLIT}: {len(idx)} patches, device {device}")
    netG, netD = load_models(c_in, c_out, img_size, device)

    groups = {k: [b for b in v if b < c_out] for k, v in BAND_GROUPS.items()}
    groups = {k: v for k, v in groups.items() if v}

    sens_raw = np.zeros(c_in + c_out)
    sens_scaled = np.zeros(c_in + c_out)
    n_batches = 0

    acc = {"fake": [], "real": [], "splice": [], "every": [], "loo": []}
    blur_acc = {(g, s): [] for g in groups for s in BLUR_SIGMAS}
    first = {}

    for start in range(0, len(idx), BATCH):
        sel = idx[start:start + BATCH]
        real_A = torch.from_numpy(np.asarray(X[sel], dtype=np.float32)).to(device)
        real_B = torch.from_numpy(np.asarray(Y[sel], dtype=np.float32)).to(device)

        with torch.no_grad():
            fake_B = netG(real_A)

        chan_std = np.concatenate([
            real_A.std(dim=(0, 1, 2)).cpu().numpy(),
            real_B.std(dim=(0, 1, 2)).cpu().numpy(),
        ])

        s = channel_sensitivity(netD, real_A, fake_B, real_B, chan_std)
        sens_raw += s["fake"]["raw"]
        sens_scaled += s["fake"]["scaled"]

        acc["fake"].append(d_scores(netD, real_A, fake_B))
        acc["real"].append(d_scores(netD, real_A, real_B))

        sp, ev = splice_in(netD, real_A, fake_B, real_B)
        acc["splice"].append(sp)
        acc["every"].append(ev)
        acc["loo"].append(leave_one_out(netD, real_A, fake_B, real_B))

        for key, v in blur_test(netD, real_A, real_B, groups, BLUR_SIGMAS).items():
            blur_acc[key].append(v)

        n_batches += 1

        if not first:
            first = {"real_A": real_A, "fake_B": fake_B, "real_B": real_B}

    sens_raw /= n_batches
    sens_scaled /= n_batches

    d_fake = np.concatenate(acc["fake"])
    d_real = np.concatenate(acc["real"])
    d_splice = np.concatenate(acc["splice"])
    d_every = np.concatenate(acc["every"])
    d_loo = np.concatenate(acc["loo"])
    blur = {k: np.concatenate(v) for k, v in blur_acc.items()}
    n = len(d_fake)
    full = np.arange(n)

    recovery = recovery_ratio(d_splice, d_fake, d_real, full)
    rec_lo, rec_hi = bootstrap_ci(lambda s: recovery_ratio(d_splice, d_fake, d_real, s), n)

    rec_all = recovery_ratio(d_every[:, None], d_fake, d_real, full)[0]

    # 1.0 = the single fake band costs the whole real/fake gap.
    loo_drop = 1.0 - recovery_ratio(d_loo, d_fake, d_real, full)
    loo_lo, loo_hi = bootstrap_ci(
        lambda s: 1.0 - recovery_ratio(d_loo, d_fake, d_real, s), n)

    print(f"\nD(fake) {d_fake.mean():.4f}   D(real) {d_real.mean():.4f}   n = {n}")
    print(f"Harness check, every band spliced: {rec_all:.1%}  (must be 100.0%)\n")

    print("(b1/b2/b3) sensitivity, splice-in recovery, leave-one-out drop  [95% CI]")
    print(f"{'chan':>5} {'x std':>10} {'share':>7} | {'splice-in':>20} | {'leave-one-out':>20}")
    print("-" * 70)
    total = sens_scaled.sum() + 1e-12
    for i, name in enumerate(chan_names):
        if i < c_in:
            sp = loo_s = f"{'-':>20}"
        else:
            c = i - c_in
            sp = f"{recovery[c]:>7.1%} [{rec_lo[c]:>5.1%},{rec_hi[c]:>5.1%}]"
            loo_s = f"{loo_drop[c]:>7.1%} [{loo_lo[c]:>5.1%},{loo_hi[c]:>5.1%}]"
        print(f"{name:>5} {sens_scaled[i]:>10.3e} {sens_scaled[i] / total:>6.1%} | {sp} | {loo_s}")

    order = np.argsort(-recovery)
    print(f"\nSum of the {c_out} single-band recoveries: {recovery.sum():.1%}  "
          f"(C separate experiments, not a decomposition)")

    # Needs the global ordering, so it cannot share the first pass.
    cum_acc = []
    for start in range(0, len(idx), BATCH):
        sel = idx[start:start + BATCH]
        real_A = torch.from_numpy(np.asarray(X[sel], dtype=np.float32)).to(device)
        real_B = torch.from_numpy(np.asarray(Y[sel], dtype=np.float32)).to(device)
        with torch.no_grad():
            fake_B = netG(real_A)
        cum_acc.append(splice_in_cumulative(netD, real_A, fake_B, real_B, order))
    d_cum = np.concatenate(cum_acc)

    cum = recovery_ratio(d_cum, d_fake, d_real, full)
    cum_lo, cum_hi = bootstrap_ci(lambda s: recovery_ratio(d_cum, d_fake, d_real, s), n)

    print("\n(b2c) cumulative splice-in, descending single-band order")
    print(f"{'k':>3} {'band added':>11} {'recovery':>9} {'95% CI':>18}")
    print("-" * 45)
    for k, c in enumerate(order, start=1):
        print(f"{k:>3} {BAND_NAMES[c]:>11} {cum[k - 1]:>8.1%} "
              f"[{cum_lo[k - 1]:>6.1%},{cum_hi[k - 1]:>6.1%}]")

    print("\n(b4) blur test on REAL targets: D output, and how far it falls toward fake")
    print(f"{'group':>6} {'sigma':>6} {'D(blurred real)':>16} {'toward fake':>13} {'95% CI':>18}")
    print("-" * 64)
    blur_report = {}
    for g in groups:
        for s in BLUR_SIGMAS:
            b = blur[(g, s)]
            frac_fn = lambda sel, b=b: 1.0 - recovery_ratio(b[:, None], d_fake, d_real, sel)
            frac = frac_fn(full)[0]
            f_lo, f_hi = bootstrap_ci(lambda sel, f=frac_fn: f(sel), n)
            print(f"{g:>6} {s:>6.1f} {b.mean():>16.4f} {frac:>12.1%} "
                  f"[{f_lo[0]:>6.1%},{f_hi[0]:>6.1%}]")
            blur_report[f"{g}_sigma{s}"] = {
                "d_out": float(b.mean()),
                "fraction_toward_fake": float(frac),
                "ci": [float(f_lo[0]), float(f_hi[0])],
            }

    save_d_maps(netD, first["real_A"], first["fake_B"], first["real_B"],
                _sar_rgb_from_norm_percentile(first["real_A"]), OUT_DIR, N_PANELS)

    save_panels(_sar_rgb_from_norm_percentile(first["real_A"]),
                first["fake_B"].cpu().numpy(), first["real_B"].cpu().numpy(),
                s2_clip, OUT_DIR, N_PANELS)

    summary = {
        "model_dir": MODEL_DIR,
        "netD_file": NETD_FILE,
        "split": SPLIT,
        "n_patches": int(n),
        "n_boot": N_BOOT,
        "caveat": (
            "splice-in and leave-one-out feed D patches that mix real and generated "
            "bands, which are out of distribution for D; on their own they cannot "
            "separate a joint cross-band cue from rejection of inconsistent mixtures. "
            "The blur test does not splice and is the one to trust for 'missing texture'."
        ),
        "d_out_fake": float(d_fake.mean()),
        "d_out_real": float(d_real.mean()),
        "sensitivity_raw": {n_: float(v) for n_, v in zip(chan_names, sens_raw)},
        "sensitivity_std_scaled": {n_: float(v) for n_, v in zip(chan_names, sens_scaled)},
        "ablation_recovery": {n_: float(v) for n_, v in zip(BAND_NAMES[:c_out], recovery)},
        "ablation_recovery_ci": {
            n_: [float(a), float(b)] for n_, a, b in zip(BAND_NAMES[:c_out], rec_lo, rec_hi)},
        "ablation_recovery_sum": float(recovery.sum()),
        "ablation_all_bands_check": float(rec_all),
        "leave_one_out_drop": {n_: float(v) for n_, v in zip(BAND_NAMES[:c_out], loo_drop)},
        "leave_one_out_drop_ci": {
            n_: [float(a), float(b)] for n_, a, b in zip(BAND_NAMES[:c_out], loo_lo, loo_hi)},
        "blur_test": blur_report,
        "ablation_order": [BAND_NAMES[c] for c in order],
        "ablation_cumulative": [float(v) for v in cum],
        "ablation_cumulative_ci": [
            [float(a), float(b)] for a, b in zip(cum_lo, cum_hi)],
    }
    with open(os.path.join(OUT_DIR, "discriminator_diagnostics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    main()
