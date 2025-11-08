# Script to (generate) infer images from the preprocessed SEN12-MS-CR dataset with the trained pix2pix model

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageOps, ImageFilter
import random

# ---------- Repo path for the wrapper ----------
THIS_DIR = Path(__file__).parent
gan_repo_path = THIS_DIR / 'mymodels/pytorch-CycleGAN-and-pix2pix'
if gan_repo_path.is_dir():
    sys.path.insert(0, str(gan_repo_path))
else:
    print(f"[WARN] pix2pix repo not found at {gan_repo_path}; assuming wrapper is importable.")

from mymodels.cp2pwrapper import Pix2PixWrapper  # must exist

# ---------- Defaults ----------
# Point to the dataset you just created
DEFAULT_DATA_DIR = "/home/dll0505/AdditionalStorage/thesis/sen12-ms-cr/dataset_cr_summer_test_2000"
DEFAULT_RESULTS_DIR = "trained-full-copy/generated_cr_summer"
DEFAULT_MODEL_DIR = "trained-full-copy/pix2pix"
DEFAULT_MODEL_FILE = "pix2pix_best_netG.pth"

# ---------- S2 inverse using clip JSON ----------
def load_s2_clip_json(path):
    with open(path, "r") as f:
        clip = json.load(f)
    out = {}
    for k, v in clip.items():
        ki = int(k) if isinstance(k, str) else k
        out[ki] = tuple(v)
    return out  # {1:(L1,H1),...,13:(L13,H13)}

def inverse_from_tanh_using_clip(y_scaled, s2_clip):
    y01 = (y_scaled + 1.0) * 0.5
    y = np.empty_like(y01, dtype=np.float32)
    for c in range(13):
        lo, hi = s2_clip[c+1]
        y[..., c] = lo + y01[..., c] * (hi - lo)
    return y

# ---------- RGB from S2 ----------
def rgb_from_s2(y_orig, mode="global", percentiles=(2,98), gamma=1/2.2, blur_sigma=None):
    """
    y_orig: (H,W,13) original S2 units
    mode: "global" uses per-image min/max; "stretch" uses image percentile stretch
    """
    # Band order: [B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12]
    B4 = y_orig[..., 3]
    B3 = y_orig[..., 2]
    B2 = y_orig[..., 1]
    rgb = np.stack([B4, B3, B2], axis=-1).astype(np.float32)

    if mode == "stretch":
        lo_pct, hi_pct = percentiles
        out = np.zeros_like(rgb, dtype=np.float32)
        for c in range(3):
            ch = rgb[..., c]
            lo = np.percentile(ch, lo_pct)
            hi = np.percentile(ch, hi_pct)
            if hi <= lo:
                lo, hi = float(ch.min()), float(ch.max())
            ch01 = np.clip((ch - lo) / (hi - lo + 1e-9), 0, 1)
            out[..., c] = ch01
    else:
        # conservative global per-image min/max
        out = np.zeros_like(rgb, dtype=np.float32)
        for c in range(3):
            ch = rgb[..., c]
            lo = float(ch.min()); hi = float(ch.max())
            if hi <= lo: hi = lo + 1.0
            out[..., c] = np.clip((ch - lo) / (hi - lo), 0, 1)

    if gamma is not None:
        out = np.power(out, gamma)

    img8 = (out * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(img8)
    if blur_sigma and blur_sigma > 0:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=float(blur_sigma)))
    return pil

# ---------- SAR previews from the actual model input ----------
def sar_to_previews(x_scaled, vv_idx=0, vh_idx=1, blur_sigma=None):
    """
    x_scaled: (H,W,Cin) in [-1,1]
    Returns PIL images for vv, vh, and pseudo (vv,vh,vv)
    """
    vv = x_scaled[..., vv_idx] if x_scaled.shape[-1] > vv_idx else x_scaled[..., 0]
    vh = x_scaled[..., vh_idx] if x_scaled.shape[-1] > vh_idx else x_scaled[..., -1]
    pseudo = np.stack([vv, vh, vv], axis=-1)

    vv8 = (np.clip((vv + 1.0) * 0.5, 0, 1) * 255).astype(np.uint8)
    vh8 = (np.clip((vh + 1.0) * 0.5, 0, 1) * 255).astype(np.uint8)
    pseudo8 = (np.clip((pseudo + 1.0) * 0.5, 0, 1) * 255).astype(np.uint8)

    vvP = Image.fromarray(vv8)
    vhP = Image.fromarray(vh8)
    psP = Image.fromarray(pseudo8)
    if blur_sigma and blur_sigma > 0:
        vvP = vvP.filter(ImageFilter.GaussianBlur(radius=float(blur_sigma)))
        vhP = vhP.filter(ImageFilter.GaussianBlur(radius=float(blur_sigma)))
        psP = psP.filter(ImageFilter.GaussianBlur(radius=float(blur_sigma)))
    return vvP, vhP, psP

# ---------- Panel concat ----------
def concat_h(imgs, pad=4, pad_color=220):
    imgs = [im if isinstance(im, Image.Image) else Image.fromarray(im) for im in imgs]
    h = max(im.height for im in imgs)
    sep = Image.new("RGB", (pad, h), (pad_color, pad_color, pad_color))
    out = Image.new("RGB", (sum(im.width for im in imgs) + pad * (len(imgs)-1), h), (255, 255, 255))
    x = 0
    for i, im in enumerate(imgs):
        if im.height < h:
            delta = h - im.height
            im = ImageOps.expand(im, border=(0, delta//2, 0, delta - delta//2), fill=(255,255,255))
        out.paste(im, (x,0))
        x += im.width
        if i < len(imgs)-1:
            out.paste(sep, (x,0))
            x += pad
    return out

# ---------- Model loader ----------
def load_model(model_dir, model_file, in_ch, out_ch, img_size, device):
    mp = Path(model_dir) / model_file
    if not mp.exists():
        raise FileNotFoundError(f"Model not found: {mp}")
    model = Pix2PixWrapper(in_channels=in_ch, out_channels=out_ch, img_size=img_size)
    # safer load (PyTorch >= 2.4); if your torch is older, drop weights_only
    try:
        state = torch.load(str(mp), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(mp), map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model

# ---------- Main inference (NPY mode) ----------
def infer_from_numpy(
    data_dir, out_dir, model_dir, model_file, device,
    input_channel_select=None,
    viz_mode="global", viz_stretch=(5,95), blur_sigma=None,
    random_k=150, rng_seed=123
):
    os.makedirs(out_dir, exist_ok=True)

    X_path = Path(data_dir) / "X_test.npy"
    y_path = Path(data_dir) / "y_test.npy"              # clean GT (optional)
    y_cloudy_path = Path(data_dir) / "y_cloudy_test.npy"  # cloudy (optional)
    S2clip_path = Path(data_dir) / "S2_clip.json"

    # Load inputs (use mmap to save RAM)
    X = np.load(X_path, mmap_mode="r")
    y_true_scaled = np.load(y_path, mmap_mode="r") if y_path.exists() else None
    y_cloudy_scaled = np.load(y_cloudy_path, mmap_mode="r") if y_cloudy_path.exists() else None

    # Sanity: range check for X
    xmin, xmax = float(np.min(X)), float(np.max(X))
    xmean, xstd = float(np.mean(X)), float(np.std(X))
    print(f"[INFO] X_test range: [{xmin:.4f}, {xmax:.4f}], mean={xmean:.4f}, std={xstd:.4f}")
    if xmin < -1.001 or xmax > 1.001:
        print(f"[WARN] X_test.npy is outside [-1,1]. Check preprocessing.")

    s2_clip = load_s2_clip_json(S2clip_path)

    # Optional input channel selection (must match training!)
    if input_channel_select is not None:
        print(f"[INFO] Using input channel {input_channel_select} only.")
        X = X[..., input_channel_select:input_channel_select+1]

    N, H, W, Cin = X.shape
    Cout = 13
    model = load_model(model_dir, model_file, Cin, Cout, H, device)

    # ----- Choose random subset of indices -----
    rng = random.Random(rng_seed)
    if random_k is None or random_k <= 0:
        # fallback: use the whole set
        indices = list(range(N))
    else:
        k = min(random_k, N)
        indices = rng.sample(range(N), k=k)

    print(f"[NPY inference] random_k={len(indices)} (of {N}); Cin={Cin}, Cout={Cout}")
    print(f"[INFO] y_cloudy present: {y_cloudy_scaled is not None}, y_clean present: {y_true_scaled is not None}")

    # Loop over selected indices
    for idx in tqdm(indices, desc="Generating"):
        x_scaled = X[idx:idx+1]  # (1,H,W,Cin)

        # use torch.tensor to avoid non-writable numpy warning
        x_t = torch.tensor(x_scaled, dtype=torch.float32, device=device)

        with torch.no_grad():
            y_pred_scaled = model(x_t).detach().cpu().numpy()[0]  # (H,W,13) in [-1,1]

        # Saturation diagnostic (tanh space)
        sat_pos = float(np.mean(y_pred_scaled >= 0.999)) * 100.0
        sat_neg = float(np.mean(y_pred_scaled <= -0.999)) * 100.0

        base = f"sample_{idx:06d}"

        # SAR previews from the exact tensor consumed by the model
        vvP, vhP, psP = sar_to_previews(x_scaled[0], blur_sigma=blur_sigma)
        vvP.save(os.path.join(out_dir, base + "_sar_vv.png"))
        vhP.save(os.path.join(out_dir, base + "_sar_vh.png"))
        psP.save(os.path.join(out_dir, base + "_sar_pseudo.png"))

        # Predicted clean (inverse to original units) → RGB
        y_pred_orig = inverse_from_tanh_using_clip(y_pred_scaled, s2_clip)
        pred_rgb = rgb_from_s2(
            y_pred_orig,
            mode=("stretch" if viz_mode=="stretch" else "global"),
            percentiles=viz_stretch,
            blur_sigma=blur_sigma
        )
        pred_rgb.save(os.path.join(out_dir, base + "_pred_rgb.png"))

        # Cloudy optical panel (if available)
        cloudy_rgb = None
        if y_cloudy_scaled is not None:
            y_cloudy_orig = inverse_from_tanh_using_clip(y_cloudy_scaled[idx], s2_clip)
            cloudy_rgb = rgb_from_s2(
                y_cloudy_orig,
                mode=("stretch" if viz_mode=="stretch" else "global"),
                percentiles=viz_stretch,
                blur_sigma=blur_sigma
            )
            cloudy_rgb.save(os.path.join(out_dir, base + "_cloudy_rgb.png"))

        # Ground-truth clean (if available)
        true_rgb = None
        if y_true_scaled is not None:
            y_true_orig_i = inverse_from_tanh_using_clip(y_true_scaled[idx], s2_clip)
            true_rgb = rgb_from_s2(
                y_true_orig_i,
                mode=("stretch" if viz_mode=="stretch" else "global"),
                percentiles=viz_stretch,
                blur_sigma=blur_sigma
            )
            true_rgb.save(os.path.join(out_dir, base + "_true_rgb.png"))

        # ----- Build 4-panel -----
        tiles = [psP]
        if cloudy_rgb is not None:
            tiles.append(cloudy_rgb)
        tiles.append(pred_rgb)
        if true_rgb is not None:
            tiles.append(true_rgb)
        panel = concat_h(tiles)
        panel.save(os.path.join(out_dir, base + "_panel.png"))

        print(f"[SAMPLE {idx}] pred_scaled sat+ {sat_pos:.2f}% | sat- {sat_neg:.2f}%")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Generate optical (S2) from preprocessed arrays; save SAR/Cloudy/Pred/GT panels.")
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="root of preprocessed arrays (X_test.npy, y_test.npy, y_cloudy_test.npy)")
    ap.add_argument("--model_dir", default=DEFAULT_MODEL_DIR, help="dir containing the trained netG")
    ap.add_argument("--model_file", default=DEFAULT_MODEL_FILE, help="generator weights file name")
    ap.add_argument("--out_dir", default=DEFAULT_RESULTS_DIR, help="where to write outputs")
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    ap.add_argument("--input_channel_select", type=int, default=None,
                    help="0 for VV, 1 for VH, None for both (must match training!)")
    ap.add_argument("--viz_mode", choices=["global","stretch"], default="global",
                    help="global=per-image min/max; stretch=per-image percentile stretch")
    ap.add_argument("--viz_stretch", nargs=2, type=float, default=(5,95),
                    help="low high percentiles for stretch mode (ignored if viz_mode=global)")
    ap.add_argument("--blur", type=float, default=0.0,
                    help="Gaussian blur sigma for visualization only (e.g., 1.0); 0 disables")
    ap.add_argument("--random_k", type=int, default=150,
                    help="number of random samples to generate (set 0 to use all samples)")
    ap.add_argument("--rng_seed", type=int, default=123, help="random seed for sampling")
    args = ap.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    infer_from_numpy(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        model_dir=args.model_dir,
        model_file=args.model_file,
        device=device,
        input_channel_select=args.input_channel_select,
        viz_mode=args.viz_mode,
        viz_stretch=tuple(args.viz_stretch),
        blur_sigma=args.blur,
        random_k=args.random_k,
        rng_seed=args.rng_seed,
    )

if __name__ == "__main__":
    main()
