import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageOps, ImageFilter

# ---------- Repo path for the wrapper ----------
THIS_DIR = Path(__file__).parent
gan_repo_path = THIS_DIR / 'mymodels/pytorch-CycleGAN-and-pix2pix'
if gan_repo_path.is_dir():
    sys.path.insert(0, str(gan_repo_path))
else:
    print(f"[WARN] pix2pix repo not found at {gan_repo_path}; assuming wrapper is importable.")

from mymodels.cp2pwrapper import Pix2PixWrapper  # must exist


# ---------- Defaults ----------
DEFAULT_DATA_DIR = "dataset_winter_all"
DEFAULT_RESULTS_DIR = "trained-full-copy/generated_bands"
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
    """
    y_scaled: (..., 13) in [-1,1]
    s2_clip: {1:(lo,hi),...,13:(lo,hi)}
    returns y in original S2-like units using per-band linear mapping
    """
    y01 = (y_scaled + 1.0) * 0.5
    y = np.empty_like(y01, dtype=np.float32)
    for c in range(13):
        lo, hi = s2_clip[c+1]
        y[..., c] = lo + y01[..., c] * (hi - lo)
    return y


# ---------- SAR previews from the actual model input ----------
def sar_to_previews(x_scaled, vv_idx=0, vh_idx=1, blur_sigma=None):
    """
    x_scaled: (H,W,Cin) in [-1,1]
    Returns PIL images for vv, vh, and pseudo (vv,vh,vv)
    """
    def to_uint8(ch):
        ch01 = (ch + 1.0) * 0.5  # [-1,1] -> [0,1]
        return (np.clip(ch01, 0, 1) * 255).astype(np.uint8)

    vv = x_scaled[..., vv_idx] if x_scaled.shape[-1] > vv_idx else x_scaled[..., 0]
    vh = x_scaled[..., vh_idx] if x_scaled.shape[-1] > vh_idx else x_scaled[..., -1]
    vv8 = to_uint8(vv)
    vh8 = to_uint8(vh)
    pseudo = np.stack([vv, vh, vv], axis=-1)
    pseudo8 = (np.clip((pseudo + 1.0) * 0.5, 0, 1) * 255).astype(np.uint8)

    vvP = Image.fromarray(vv8, mode="L")
    vhP = Image.fromarray(vh8, mode="L")
    psP = Image.fromarray(pseudo8, mode="RGB")
    if blur_sigma and blur_sigma > 0:
        vvP = vvP.filter(ImageFilter.GaussianBlur(radius=float(blur_sigma)))
        vhP = vhP.filter(ImageFilter.GaussianBlur(radius=float(blur_sigma)))
        psP = psP.filter(ImageFilter.GaussianBlur(radius=float(blur_sigma)))
    return vvP, vhP, psP


# ---------- Band grayscale helpers ----------
def _scale_to_uint8(arr, lo, hi, gamma=None):
    """Scale arr to [0,255] using lo/hi (floats), optional gamma (apply to [0,1])."""
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    if gamma is not None and gamma > 0:
        x = np.power(x, gamma)
    return (x * 255.0).astype(np.uint8)

def band_pair_to_images(pred_band, true_band=None, mode="stretch", percentiles=(2, 98), gamma=None):
    """
    Convert single-band arrays (pred +/- true) into 8-bit grayscale PIL images with paired scaling.

    mode:
      - "stretch": use percentiles computed *jointly* over pred+true for consistent scaling
      - "global":  use min/max computed *jointly* over pred+true
    """
    if true_band is not None:
        both = np.concatenate([pred_band.reshape(-1), true_band.reshape(-1)])
    else:
        both = pred_band.reshape(-1)

    if mode == "stretch":
        lo = np.percentile(both, percentiles[0])
        hi = np.percentile(both, percentiles[1])
    else:  # "global"
        lo = float(np.min(both))
        hi = float(np.max(both))

    pred8 = _scale_to_uint8(pred_band, lo, hi, gamma=gamma)
    pred_img = Image.fromarray(pred8, mode="L")

    if true_band is not None:
        true8 = _scale_to_uint8(true_band, lo, hi, gamma=gamma)
        true_img = Image.fromarray(true8, mode="L")
    else:
        true_img = None

    return pred_img, true_img


# ---------- Panel concat (robust to different modes) ----------
def concat_h(imgs, pad=4, pad_color=220):
    # Convert all to RGB to avoid mode mismatch in PIL paste
    imgs = [im.convert("RGB") if isinstance(im, Image.Image) else Image.fromarray(im).convert("RGB") for im in imgs]
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
    limit=None, offset=0, input_channel_select=None,
    save_true=True, viz_mode="stretch", viz_stretch=(2,98), viz_gamma=None, blur_sigma=None
):
    """
    Exports per-band grayscale images:
      sample_000123_B01_pred.png, sample_000123_B01_true.png, sample_000123_B01_panel.png, ... B13
    Also saves SAR previews and prints tanh-space diagnostics.
    """
    os.makedirs(out_dir, exist_ok=True)

    X_path = Path(data_dir) / "X_test.npy"
    y_path = Path(data_dir) / "y_test.npy"  # optional
    S2clip_path = Path(data_dir) / "S2_clip.json"

    # Load inputs (already in [-1,1] from your preprocessing)
    X = np.load(X_path)
    y_true_scaled = np.load(y_path) if (save_true and y_path.exists()) else None

    # Sanity: range check for X
    xmin, xmax = float(X.min()), float(X.max())
    xmean, xstd = float(X.mean()), float(X.std())
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

    # Subset selection
    if offset < 0: offset = 0
    if limit is None: limit = N - offset
    end = min(N, offset + limit)
    print(f"[NPY inference] samples {offset}:{end} (of {N}); Cin={Cin}, Cout={Cout}")

    # Quick tanh-space error stats if GT available
    mae_list, rmse_list = [], []

    band_names = [f"B{b:02d}" for b in range(1, 14)]

    for i in tqdm(range(offset, end), desc="Generating"):
        x_scaled = X[i:i+1]                 # (1,H,W,Cin) in [-1,1]
        x_t = torch.from_numpy(x_scaled).to(device)

        with torch.no_grad():
            y_pred_scaled = model(x_t).cpu().numpy()[0]  # (H,W,13) in [-1,1]

        # Per-sample saturation diagnostic in tanh space
        sat_pos = float(np.mean(y_pred_scaled >= 0.999)) * 100.0
        sat_neg = float(np.mean(y_pred_scaled <= -0.999)) * 100.0

        # Save SAR previews from EXACT tensor consumed
        base = f"sample_{i:06d}"
        vvP, vhP, psP = sar_to_previews(x_scaled[0], blur_sigma=blur_sigma)
        vvP.save(os.path.join(out_dir, base + "_sar_vv.png"))
        vhP.save(os.path.join(out_dir, base + "_sar_vh.png"))
        psP.save(os.path.join(out_dir, base + "_sar_pseudo.png"))

        # Inverse to original units for per-band visualization
        y_pred_orig = inverse_from_tanh_using_clip(y_pred_scaled, s2_clip)

        if y_true_scaled is not None:
            y_true_scaled_i = y_true_scaled[i]
            diff = y_pred_scaled - y_true_scaled_i
            mae_list.append(float(np.mean(np.abs(diff))))
            rmse_list.append(float(np.sqrt(np.mean(diff**2))))
            y_true_orig_i = inverse_from_tanh_using_clip(y_true_scaled_i, s2_clip)
        else:
            y_true_orig_i = None

        # Save per-band grayscale images and panels
        for b_idx in range(13):
            bname = band_names[b_idx]
            pred_band = y_pred_orig[..., b_idx]
            true_band = y_true_orig_i[..., b_idx] if y_true_orig_i is not None else None

            pred_img, true_img = band_pair_to_images(
                pred_band,
                true_band=true_band,
                mode=("stretch" if viz_mode == "stretch" else "global"),
                percentiles=viz_stretch,
                gamma=viz_gamma
            )

            pred_img.save(os.path.join(out_dir, f"{base}_{bname}_pred.png"))
            if true_img is not None:
                true_img.save(os.path.join(out_dir, f"{base}_{bname}_true.png"))
                panel = concat_h([pred_img, true_img])
                panel.save(os.path.join(out_dir, f"{base}_{bname}_panel.png"))
            else:
                # no GT: still make a panel with a spacer and label-like placeholder (pred only)
                panel = concat_h([pred_img])
                panel.save(os.path.join(out_dir, f"{base}_{bname}_panel.png"))

        # Log per-sample quickline
        print(f"[SAMPLE {i}] pred_scaled sat+ {sat_pos:.2f}% | sat- {sat_neg:.2f}%")

    if mae_list:
        print(f"[DEBUG] Subset tanh-space MAE:  {np.mean(mae_list):.4f}")
        print(f"[DEBUG] Subset tanh-space RMSE: {np.mean(rmse_list):.4f}")


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(
        description="Generate per-band S2 images from preprocessed arrays; save band-wise visuals + diagnostics."
    )
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="root of preprocessed arrays (X_test.npy, y_test.npy, S2_clip.json)")
    ap.add_argument("--model_dir", default=DEFAULT_MODEL_DIR, help="dir containing the trained netG")
    ap.add_argument("--model_file", default=DEFAULT_MODEL_FILE, help="generator weights file name")
    ap.add_argument("--out_dir", default=DEFAULT_RESULTS_DIR, help="where to write outputs")
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    ap.add_argument("--limit", type=int, default=100, help="max number of samples")
    ap.add_argument("--offset", type=int, default=0, help="start index")
    ap.add_argument("--input_channel_select", type=int, default=None,
                    help="0 for VV, 1 for VH, None for both (must match training!)")
    ap.add_argument("--viz_mode", choices=["global","stretch"], default="stretch",
                    help="global=min/max; stretch=percentiles — both computed jointly over pred+true per band")
    ap.add_argument("--viz_stretch", nargs=2, type=float, default=(2,98),
                    help="low high percentiles for stretch mode (ignored if viz_mode=global)")
    ap.add_argument("--viz_gamma", type=float, default=None,
                    help="optional gamma on [0,1] before 8-bit quantization (e.g., 0.5 for sqrt-like boost)")
    ap.add_argument("--blur", type=float, default=0.0,
                    help="Gaussian blur sigma for SAR previews only (e.g., 1.0); 0 disables")
    ap.add_argument("--no_true", action="store_true", help="don’t load y_test.npy / don’t make GT band panels")
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
        limit=args.limit,
        offset=args.offset,
        input_channel_select=args.input_channel_select,
        save_true=(not args.no_true),
        viz_mode=args.viz_mode,
        viz_stretch=tuple(args.viz_stretch),
        viz_gamma=args.viz_gamma,
        blur_sigma=args.blur,
    )

if __name__ == "__main__":
    main()
