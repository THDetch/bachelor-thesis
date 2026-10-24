# Post-hoc discriminator diagnostics.
#
# Answers one question: WHAT is the discriminator using to separate real from fake?
# Run it on a trained checkpoint; it never updates weights.
#
#   (b1) per-channel input sensitivity   -> which of the 15 input channels D reacts to
#   (b2) per-channel ablation            -> which channel D actually decides on
#   (c)  D output map overlaid on RGB    -> where in the image D is confident
#   (d)  pred vs GT under ONE stretch    -> what the images really look like
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

N_SAMPLES = 64                      # patches for the sensitivity/ablation statistics
N_PANELS = 8                        # patches to render as images
BATCH = 8
GPU_ID = 0
SEED = 42

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


# -------------------- (b2) ABLATION --------------------

def channel_ablation(netD, real_A, fake_B, real_B):
    """Splice ONE real band into the fake at a time and measure how much D recovers.

        recovery_c = (D(fake, band c real) - D(fake)) / (D(real) - D(fake))

    These are C separate experiments, not a decomposition: they have no reason to sum
    to 1. rec_all splices every band in turn, which must reproduce the real pair and
    return 1.0 -- it catches a wrong channel axis or concat order in d_input.
    """
    with torch.no_grad():
        base_fake = netD(d_input(real_A, fake_B)).mean().item()
        base_real = netD(d_input(real_A, real_B)).mean().item()

        gap = base_real - base_fake
        recovery = []

        for c in range(fake_B.shape[-1]):
            spliced = fake_B.clone()
            spliced[..., c] = real_B[..., c]
            score = netD(d_input(real_A, spliced)).mean().item()
            recovery.append((score - base_fake) / (gap + 1e-12))

        every = fake_B.clone()
        for c in range(fake_B.shape[-1]):
            every[..., c] = real_B[..., c]
        rec_all = (netD(d_input(real_A, every)).mean().item() - base_fake) / (gap + 1e-12)

    return base_fake, base_real, np.array(recovery), rec_all


def cumulative_ablation(netD, real_A, fake_B, real_B, order):
    """Replace the top-k bands of `order` simultaneously, for k = 1..C.

    Answers how many bands have to be real before D changes its verdict. A slow ramp
    means the cue is joint across bands; a jump at small k means a few bands carry it.
    """
    with torch.no_grad():
        base_fake = netD(d_input(real_A, fake_B)).mean().item()
        base_real = netD(d_input(real_A, real_B)).mean().item()
        gap = base_real - base_fake

        spliced = fake_B.clone()
        out = []
        for c in order:
            spliced[..., c] = real_B[..., c]
            score = netD(d_input(real_A, spliced)).mean().item()
            out.append((score - base_fake) / (gap + 1e-12))

    return np.array(out)


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

    sens_raw = np.zeros(c_in + c_out)
    sens_scaled = np.zeros(c_in + c_out)
    recovery = np.zeros(c_out)
    rec_all_sum = 0.0
    d_fake_all, d_real_all = [], []
    n_batches = 0

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

        d_f, d_r, rec, rec_all = channel_ablation(netD, real_A, fake_B, real_B)
        d_fake_all.append(d_f)
        d_real_all.append(d_r)
        recovery += rec
        rec_all_sum += rec_all
        n_batches += 1

        if not first:
            first = {"real_A": real_A, "fake_B": fake_B, "real_B": real_B}

    sens_raw /= n_batches
    sens_scaled /= n_batches
    recovery /= n_batches
    rec_all_sum /= n_batches

    print(f"\nD(fake) {np.mean(d_fake_all):.4f}   D(real) {np.mean(d_real_all):.4f}\n")

    print("(b) input sensitivity and ablation recovery")
    print(f"{'chan':>5} {'|dD/dx|':>11} {'x std':>11} {'share':>7} | {'recovery':>9}")
    print("-" * 52)
    total = sens_scaled.sum() + 1e-12
    for i, name in enumerate(chan_names):
        rec = f"{recovery[i - c_in]:>8.1%}" if i >= c_in else f"{'-':>8}"
        print(f"{name:>5} {sens_raw[i]:>11.3e} {sens_scaled[i]:>11.3e} "
              f"{sens_scaled[i] / total:>6.1%} | {rec}")

    order = np.argsort(-recovery)
    print("\nTop single-band recoveries: " + ", ".join(
        f"{BAND_NAMES[c]} {recovery[c]:.1%}" for c in order[:5]))
    print(f"Sum of the {c_out} single-band recoveries: {recovery.sum():.1%}  "
          f"(not a decomposition; no reason to reach 100%)")
    print(f"Harness check, every band spliced: {rec_all_sum:.1%}  (must be 100.0%)")

    # Needs the global ordering, so it cannot share the first pass.
    cum = np.zeros(c_out)
    for start in range(0, len(idx), BATCH):
        sel = idx[start:start + BATCH]
        real_A = torch.from_numpy(np.asarray(X[sel], dtype=np.float32)).to(device)
        real_B = torch.from_numpy(np.asarray(Y[sel], dtype=np.float32)).to(device)
        with torch.no_grad():
            fake_B = netG(real_A)
        cum += cumulative_ablation(netD, real_A, fake_B, real_B, order)
    cum /= n_batches

    print("\n(b3) cumulative ablation, bands replaced in descending single-band order")
    print(f"{'k':>3} {'band added':>11} {'recovery':>9}")
    print("-" * 26)
    for k, c in enumerate(order, start=1):
        print(f"{k:>3} {BAND_NAMES[c]:>11} {cum[k - 1]:>8.1%}")

    save_d_maps(netD, first["real_A"], first["fake_B"], first["real_B"],
                _sar_rgb_from_norm_percentile(first["real_A"]), OUT_DIR, N_PANELS)

    save_panels(_sar_rgb_from_norm_percentile(first["real_A"]),
                first["fake_B"].cpu().numpy(), first["real_B"].cpu().numpy(),
                s2_clip, OUT_DIR, N_PANELS)

    summary = {
        "model_dir": MODEL_DIR,
        "netD_file": NETD_FILE,
        "split": SPLIT,
        "n_patches": int(len(idx)),
        "d_out_fake": float(np.mean(d_fake_all)),
        "d_out_real": float(np.mean(d_real_all)),
        "sensitivity_raw": {n: float(v) for n, v in zip(chan_names, sens_raw)},
        "sensitivity_std_scaled": {n: float(v) for n, v in zip(chan_names, sens_scaled)},
        "ablation_recovery": {n: float(v) for n, v in zip(BAND_NAMES[:c_out], recovery)},
        "ablation_recovery_sum": float(recovery.sum()),
        "ablation_all_bands_check": float(rec_all_sum),
        "ablation_order": [BAND_NAMES[c] for c in order],
        "ablation_cumulative": [float(v) for v in cum],
    }
    with open(os.path.join(OUT_DIR, "discriminator_diagnostics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    main()
