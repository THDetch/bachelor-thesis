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


# ---------- Defaults (adjust if needed) ----------
DEFAULT_DATA_DIR = "/home/dll0505/AdditionalStorage/thesis/sen12-ms/sub_dataset/data-256_0.02_10000_60m_execluded"
DEFAULT_RESULTS_DIR = "generated"
DEFAULT_MODEL_DIR = "trained-both_0.02_10000_60m_execluded_no_scheduler/pix2pix"
DEFAULT_MODEL_FILE = "pix2pix_best_netG.pth"


# ---------- Band name helpers ----------
BAND_NUM_TO_NAME = {
    1: "B1",  2: "B2",  3: "B3",  4: "B4",  5: "B5",  6: "B6",  7: "B7",
    8: "B8",  9: "B8A", 10: "B9", 11: "B10", 12: "B11", 13: "B12"
}

# ---------- S2 inverse using clip JSON ----------
def load_s2_clip_json(path: Path):
    """
    Returns:
      clip_int: dict {band_num: (lo, hi)} for kept bands only (as saved by your preproc)
      kept_band_numbers: sorted list of band numbers present, e.g. [2,3,4,5,6,7,8,9,12,13]
      kept_band_names:   corresponding names, e.g. ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]
    """
    with open(path, "r") as f:
        clip = json.load(f)
    clip_int = {int(k): tuple(v) for k, v in clip.items()}
    kept_band_numbers = sorted(clip_int.keys())
    kept_band_names = [BAND_NUM_TO_NAME[b] for b in kept_band_numbers]
    return clip_int, kept_band_numbers, kept_band_names

def inverse_from_tanh_using_clip(y_scaled: np.ndarray, s2_clip: dict, kept_band_numbers: list[int]) -> np.ndarray:
    """
    y_scaled: (H,W,C_kept) in [-1,1]
    s2_clip: dict {band_num: (lo, hi)} for ONLY the kept bands
    kept_band_numbers: list of band numbers in the same order as channels in y_scaled
    """
    y01 = (y_scaled + 1.0) * 0.5  # [-1,1] -> [0,1]
    y = np.empty_like(y01, dtype=np.float32)
    C = y.shape[-1]
    assert C == len(kept_band_numbers), "Mismatch between y channels and kept bands"
    for c, band_num in enumerate(kept_band_numbers):
        lo, hi = s2_clip[band_num]
        y[..., c] = lo + y01[..., c] * (hi - lo)
    return y


# ---------- RGB from kept S2 stack ----------
def rgb_from_s2(
    y_orig_kept: np.ndarray,
    rgb_indices=(2, 1, 0),      # kept order: [B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12] -> B4,B3,B2 = (2,1,0)
    mode="global",
    percentiles=(2, 98),
    gamma=1/2.2,
    blur_sigma=None
) -> Image.Image:
    """
    y_orig_kept: (H,W,C_kept) in original S2 units (kept-band order)
    """
    r, g, b = rgb_indices
    rgb = np.stack([y_orig_kept[..., r], y_orig_kept[..., g], y_orig_kept[..., b]], axis=-1).astype(np.float32)

    if mode == "stretch":
        lo_pct, hi_pct = percentiles
        out = np.zeros_like(rgb, dtype=np.float32)
        for c in range(3):
            ch = rgb[..., c]
            lo = np.percentile(ch, lo_pct)
            hi = np.percentile(ch, hi_pct)
            if hi <= lo:
                lo, hi = float(ch.min()), float(ch.max())
            out[..., c] = np.clip((ch - lo) / (hi - lo + 1e-9), 0, 1)
    else:
        # "global": scale to [0,1] per channel using its own min/max within the image
        out = np.zeros_like(rgb, dtype=np.float32)
        for c in range(3):
            ch = rgb[..., c]
            lo, hi = float(ch.min()), float(ch.max())
            if hi <= lo:
                hi = lo + 1.0
            out[..., c] = np.clip((ch - lo) / (hi - lo), 0, 1)

    if gamma is not None:
        out = np.power(out, gamma)

    img8 = (out * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(img8)
    if blur_sigma and blur_sigma > 0:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=float(blur_sigma)))
    return pil


# ---------- SAR previews from actual model input ----------
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
        out.paste(im, (x, 0))
        x += im.width
        if i < len(imgs)-1:
            out.paste(sep, (x, 0))
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
    save_true=True, viz_mode="global", viz_stretch=(5,95), blur_sigma=None
):
    os.makedirs(out_dir, exist_ok=True)
    
    X_path = Path(data_dir) / "X_test.npy"
    y_path = Path(data_dir) / "y_test.npy"  # optional
    S2clip_path = Path(data_dir) / "S2_clip.json"

    # Load inputs (already in [-1,1] from preprocessing)
    X = np.load(X_path)
    y_true_scaled = np.load(y_path) if (save_true and y_path.exists()) else None

    # Sanity: range check for X
    xmin, xmax = float(X.min()), float(X.max())
    xmean, xstd = float(X.mean()), float(X.std())
    print(f"[INFO] X_test range: [{xmin:.4f}, {xmax:.4f}], mean={xmean:.4f}, std={xstd:.4f}")
    if xmin < -1.001 or xmax > 1.001:
        print(f"[WARN] X_test.npy is outside [-1,1]. Check preprocessing.")

    # Load S2 clip + kept-band info
    s2_clip, kept_band_numbers, kept_band_names = load_s2_clip_json(S2clip_path)

    # Optional input channel selection (must match training!)
    if input_channel_select is not None:
        if not (0 <= input_channel_select < X.shape[-1]):
            raise ValueError(f"input_channel_select={input_channel_select} out of range for X with {X.shape[-1]} channels")
        print(f"[INFO] Using input channel {input_channel_select} only.")
        X = X[..., input_channel_select:input_channel_select+1]

    N, H, W, Cin = X.shape
    Cout = len(kept_band_numbers)  # dynamic (e.g., 10 after 60 m exclusion)
    model = load_model(model_dir, model_file, Cin, Cout, H, device)

    # Subset selection
    if offset < 0:
        offset = 0
    if limit is None:
        limit = N - offset
    end = min(N, offset + limit)

    print(f"[NPY inference] samples {offset}:{end} (of {N}); Cin={Cin}, Cout={Cout}")
    print(f"[NPY inference] kept bands: {kept_band_names}")

    # Quick tanh-space error stats if GT available
    mae_list, rmse_list = [], []

    for i in tqdm(range(offset, end), desc="Generating"):
        x_scaled = X[i:i+1]  # (1,H,W,Cin) in [-1,1]
        x_t = torch.from_numpy(x_scaled).to(device)

        with torch.no_grad():
            y_pred_scaled = model(x_t).cpu().numpy()[0]  # (H,W,Cout) in [-1,1]

        # Per-sample saturation diagnostic in tanh space
        sat_pos = float(np.mean(y_pred_scaled >= 0.999)) * 100.0
        sat_neg = float(np.mean(y_pred_scaled <= -0.999)) * 100.0

        # Save SAR previews from EXACT tensor consumed
        base = f"sample_{i:06d}"
        vvP, vhP, psP = sar_to_previews(x_scaled[0], blur_sigma=blur_sigma)
        vvP.save(os.path.join(out_dir, base + "_sar_vv.png"))
        vhP.save(os.path.join(out_dir, base + "_sar_vh.png"))
        psP.save(os.path.join(out_dir, base + "_sar_pseudo.png"))

        # Inverse to original units for visualization
        y_pred_orig = inverse_from_tanh_using_clip(y_pred_scaled, s2_clip, kept_band_numbers)
        # RGB using B4,B3,B2 in the kept-band order -> indices (2,1,0)
        pred_rgb = rgb_from_s2(
            y_pred_orig,
            rgb_indices=(2, 1, 0),
            mode=("stretch" if viz_mode == "stretch" else "global"),
            percentiles=viz_stretch,
            blur_sigma=blur_sigma
        )
        pred_rgb.save(os.path.join(out_dir, base + "_pred_rgb.png"))

        # If GT available, make panels and tanh metrics
        if y_true_scaled is not None:
            y_true_scaled_i = y_true_scaled[i]
            # Basic tanh-space metrics
            diff = y_pred_scaled - y_true_scaled_i
            mae_list.append(float(np.mean(np.abs(diff))))
            rmse_list.append(float(np.sqrt(np.mean(diff**2))))

            y_true_orig_i = inverse_from_tanh_using_clip(y_true_scaled_i, s2_clip, kept_band_numbers)
            true_rgb = rgb_from_s2(
                y_true_orig_i,
                rgb_indices=(2, 1, 0),
                mode=("stretch" if viz_mode == "stretch" else "global"),
                percentiles=viz_stretch,
                blur_sigma=blur_sigma
            )
            true_rgb.save(os.path.join(out_dir, base + "_true_rgb.png"))

            panel = concat_h([psP, pred_rgb, true_rgb])
            panel.save(os.path.join(out_dir, base + "_panel.png"))
        else:
            panel = concat_h([psP, pred_rgb])
            panel.save(os.path.join(out_dir, base + "_panel.png"))

        # Log per-sample quickline
        print(f"[SAMPLE {i}] pred_scaled sat+ {sat_pos:.2f}% | sat- {sat_neg:.2f}%")

    if mae_list:
        print(f"[DEBUG] Subset tanh-space MAE:  {np.mean(mae_list):.4f}")
        print(f"[DEBUG] Subset tanh-space RMSE: {np.mean(rmse_list):.4f}")


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Generate optical (S2) from preprocessed arrays; save paired visuals + diagnostics.")
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="root of preprocessed arrays (X_test.npy, y_test.npy, S2_clip.json)")
    ap.add_argument("--model_dir", default=DEFAULT_MODEL_DIR, help="dir containing the trained netG")
    ap.add_argument("--model_file", default=DEFAULT_MODEL_FILE, help="generator weights file name")
    ap.add_argument("--out_dir", default=DEFAULT_RESULTS_DIR, help="where to write outputs")
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    ap.add_argument("--limit", type=int, default=100, help="max number of samples")
    ap.add_argument("--offset", type=int, default=0, help="start index")
    ap.add_argument("--input_channel_select", type=int, default=None,
                    help="0 for VV, 1 for VH, None for both (must match training!)")
    ap.add_argument("--viz_mode", choices=["global", "stretch"], default="global",
                    help="global=per-band within-image min/max; stretch=per-image percentile stretch")
    ap.add_argument("--viz_stretch", nargs=2, type=float, default=(5, 95),
                    help="low high percentiles for stretch mode (ignored if viz_mode=global)")
    ap.add_argument("--blur", type=float, default=0.0,
                    help="Gaussian blur sigma for visualization only (e.g., 1.0); 0 disables")
    ap.add_argument("--no_true", action="store_true", help="don’t load y_test.npy / don’t make GT panels")
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
        blur_sigma=args.blur,
    )

if __name__ == "__main__":
    main()
