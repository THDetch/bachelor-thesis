# This script evaluates the trained pix2pix model across all metrics and individual bands as well as products of them. 
# Make sure you give the same preprocessed dataset directory as the one used for training.
# The script will evaluate the model based on the test sampels of this directory. 
import os
import sys
import json
import csv
import math
import random
import torch
import numpy as np
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from skimage.metrics import structural_similarity
from pathlib import Path

# ----------------------
# Config
# ----------------------
gpu_id = 0
input_channel_select = None  # 0, 1, or None (all)
models_to_run = ["pix2pix"]
# DATA_DIR = "dataset_cr_winter_test"
MODEL_DIR = "trained-full"

DATA_DIR = "/home/dll0505/AdditionalStorage/thesis/sen12-ms-cr/dataset_winter_all"
RESULTS_DIR = f"{MODEL_DIR}/evaluation_results_per_band_all_losses_efficient"


# RESULTS_DIR = f"{MODEL_DIR}/cr_evaluation_summer_band_all_losses_"
os.makedirs(RESULTS_DIR, exist_ok=True)
batch_size = 16

# ----------------------
# Repo paths (optional)
# ----------------------
umamba_repo_path = Path(__file__).parent / 'mymodels/U-Mamba'
umamba_pkg_path = umamba_repo_path / 'umamba'
if umamba_repo_path.is_dir() and umamba_pkg_path.is_dir():
    sys.path.insert(0, str(umamba_repo_path))
    sys.path.insert(0, str(umamba_pkg_path))
else:
    print("U-Mamba repository not found or is incomplete.")

gan_repo_path = Path(__file__).parent / 'mymodels/pytorch-CycleGAN-and-pix2pix'
if gan_repo_path.is_dir():
    sys.path.insert(0, str(gan_repo_path))
else:
    print(f"pytorch-CycleGAN-and-pix2pix repository not found at {gan_repo_path}")

swin_repo_path = Path(__file__).parent / 'mymodels/Swin-Unet'
if swin_repo_path.is_dir():
    sys.path.insert(0, str(swin_repo_path))
else:
    print(f"Swin-Unet repository not found at {swin_repo_path}")

# ----------------------
# Model imports (pix2pix)
# ----------------------
from mymodels.cp2pwrapper import Pix2PixWrapper  # must exist

# ----------------------
# Optional LPIPS
# ----------------------
_lpips_available = True
try:
    import lpips  # pip install lpips
except Exception:
    _lpips_available = False

# ----------------------
# Data: memory-mapped Dataset to avoid loading everything into RAM
# ----------------------
class NPYPairDataset(Dataset):
    def __init__(self, x_path, y_path, channel_select=None, mmap=True):
        self.X = np.load(x_path, mmap_mode="r" if mmap else None)
        self.Y = np.load(y_path, mmap_mode="r" if mmap else None)
        print("############",len(self.X))
        assert len(self.X) == len(self.Y), "X and Y must have same length"
        self.channel_select = channel_select

        x0 = self.X[0]
        y0 = self.Y[0]
        if self.channel_select is not None:
            x0 = x0[..., self.channel_select:self.channel_select+1]
        assert x0.ndim == 3 and y0.ndim == 3, "Samples must be HxWxC (NHWC)"

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        if self.channel_select is not None:
            x = x[..., self.channel_select:self.channel_select+1]
        x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
        return x_t, y_t

# ----------------------
# Helpers
# ----------------------
def psnr_from_mse(mse, data_range):
    if mse <= 0:
        return float('inf')
    if data_range <= 0:
        return float('-inf')
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)

# Inverse transform using S2_clip.json (no sklearn scalers)
def load_s2_clip_json(path):
    with open(path, "r") as f:
        clip = json.load(f)
    clip_int = {}
    for k, v in clip.items():
        ki = int(k) if isinstance(k, str) else k
        clip_int[ki] = tuple(v)
    return clip_int

def inverse_from_tanh_using_clip_batch(y_scaled, s2_clip):
    """
    y_scaled: (N,H,W,C) in [-1,1]
    s2_clip: dict {1:(L1,H1), ..., 13:(L13,H13)}
    returns (N,H,W,C) in original S2 units
    """
    y_scaled = np.asarray(y_scaled, dtype=np.float32)
    y01 = (y_scaled + 1.0) * 0.5  # [-1,1] -> [0,1]
    y_unscaled = np.empty_like(y01, dtype=np.float32)
    C = y01.shape[-1]
    assert C == 13, f"Expected 13 channels for S2, got {C}"
    for c in range(C):
        lo, hi = s2_clip[c + 1]
        y_unscaled[..., c] = lo + y01[..., c] * (hi - lo)
    return y_unscaled

# Sentinel-2 band names
BAND_NAMES = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B10","B11","B12"]

# Spectral groups for summaries
GROUPS = {
    "Coastal": ["B1"],
    "VIS": ["B2","B3","B4"],
    "RedEdge": ["B5","B6","B7"],
    "NIR": ["B8","B8A"],
    "WV": ["B9"],
    "Cirrus": ["B10"],
    "SWIR": ["B11","B12"],
    "10m": ["B2","B3","B4","B8"],
    "20m": ["B5","B6","B7","B8A","B11","B12"],
    "60m": ["B1","B9","B10"],
}

def indices_for(names):
    return [BAND_NAMES.index(n) for n in names]

def group_metrics_from_perband(per_band_vals):
    """
    per_band_vals: dict {metric_name: [C-length list]}
    returns dict {metric_name__group: value}
    """
    out = {}
    for gname, gmembers in GROUPS.items():
        idxs = indices_for(gmembers)
        for metric, arr in per_band_vals.items():
            vals = [arr[i] for i in idxs]
            out[f"{metric}__{gname}"] = float(np.mean(vals))
    return out

# ----------------------
# LPIPS prep & metric (RGB from B4,B3,B2)
# ----------------------
def _extract_rgb_01(truth_orig, pred_orig, s2_clip):
    iR = BAND_NAMES.index("B4")
    iG = BAND_NAMES.index("B3")
    iB = BAND_NAMES.index("B2")

    def scale01(x, lo, hi):
        return np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)

    loR, hiR = s2_clip[iR + 1]
    loG, hiG = s2_clip[iG + 1]
    loB, hiB = s2_clip[iB + 1]

    true_rgb = np.stack([
        scale01(truth_orig[..., iR], loR, hiR),
        scale01(truth_orig[..., iG], loG, hiG),
        scale01(truth_orig[..., iB], loB, hiB),
    ], axis=-1).astype(np.float32)

    pred_rgb = np.stack([
        scale01(pred_orig[..., iR], loR, hiR),
        scale01(pred_orig[..., iG], loG, hiG),
        scale01(pred_orig[..., iB], loB, hiB),
    ], axis=-1).astype(np.float32)

    return true_rgb, pred_rgb

def compute_lpips_rgb_batchwise(truth_orig, pred_orig, s2_clip, device, loss_fn, bs=8):
    true_rgb_01, pred_rgb_01 = _extract_rgb_01(truth_orig, pred_orig, s2_clip)

    def to_lpips_tensor(x01):
        x = (x01 * 2.0 - 1.0).transpose(0, 3, 1, 2)  # N,H,W,C -> N,C,H,W
        return torch.from_numpy(x).float().to(device)

    t = to_lpips_tensor(true_rgb_01)
    p = to_lpips_tensor(pred_rgb_01)

    scores = []
    with torch.no_grad():
        for i in range(0, t.shape[0], bs):
            s = loss_fn(p[i:i+bs], t[i:i+bs])
            scores.append(s.squeeze().detach().cpu().numpy())
    scores = np.concatenate([np.atleast_1d(s) for s in scores]).astype(np.float32)
    return float(np.mean(scores)) if scores.size else 0.0

# ----------------------
# SAM metric (Spectral Angle Mapper) – streaming with bounded memory
# ----------------------
class SAMAccumulator:
    """
    Streaming SAM statistics:
      - exact mean via running sum
      - approximate median/p25/p75 via reservoir sample (bounded size)
    """
    def __init__(self, sample_cap=2_000_000, rng_seed=123):
        self.count = 0
        self.sum_angles = 0.0
        self.sample_cap = int(sample_cap)
        self.sample = []
        self.rng = random.Random(rng_seed)

    def update_batch(self, truth_orig, pred_orig, eps=1e-8, sample_stride=None):
        T = truth_orig.reshape(-1, truth_orig.shape[-1]).astype(np.float64)
        P = pred_orig.reshape(-1, pred_orig.shape[-1]).astype(np.float64)

        if sample_stride is not None and sample_stride > 1:
            T = T[::sample_stride]
            P = P[::sample_stride]

        dot = np.sum(T * P, axis=1)
        nT = np.linalg.norm(T, axis=1)
        nP = np.linalg.norm(P, axis=1)
        denom = np.maximum(nT * nP, eps)
        cosang = np.clip(dot / denom, -1.0, 1.0)
        ang_deg = np.degrees(np.arccos(cosang))

        self.sum_angles += float(np.sum(ang_deg))
        n = ang_deg.shape[0]
        self.count += n

        if len(self.sample) < self.sample_cap:
            space = self.sample_cap - len(self.sample)
            if n <= space:
                self.sample.extend(ang_deg.tolist())
            else:
                self.sample.extend(ang_deg[:space].tolist())
                start = space
                for i in range(start, n):
                    j = self.rng.randrange(0, i + 1)
                    if j < self.sample_cap:
                        self.sample[j] = float(ang_deg[i])
        else:
            for i in range(n):
                j = self.rng.randrange(0, self.count)
                if j < self.sample_cap:
                    self.sample[j] = float(ang_deg[i])

    def finalize(self):
        mean = float(self.sum_angles / max(1, self.count))
        if len(self.sample) == 0:
            return {
                "SAM_mean_deg": mean,
                "SAM_median_deg": float("nan"),
                "SAM_p25_deg": float("nan"),
                "SAM_p75_deg": float("nan"),
            }
        s = np.array(self.sample, dtype=np.float32)
        return {
            "SAM_mean_deg": mean,
            "SAM_median_deg": float(np.percentile(s, 50)),
            "SAM_p25_deg": float(np.percentile(s, 25)),
            "SAM_p75_deg": float(np.percentile(s, 75)),
        }

# ----------------------
# Device
# ----------------------
if torch.cuda.is_available():
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    device_name = torch.cuda.get_device_name(device)
else:
    device = torch.device("cpu")
    device_name = "cpu"
print(f"Using device: {device_name}")
if torch.cuda.is_available():
    print(f"Current CUDA device index: {torch.cuda.current_device()}")


# ----------------------
# Logging
# ----------------------
log_file = os.path.join(RESULTS_DIR, f'evaluation_metrics_channel_{input_channel_select}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler()
    ]
)

# ----------------------
# Evaluation (streaming; no full concatenation)
# ----------------------
def main():
    logging.info("Starting evaluation...")
    logging.info(f"Using device: {device_name}")
    # logging.info(f"S2_clip (first 3): {list(s2_clip.items())[:3]}  (expect 0..10000)")

    # --- Load data paths and S2 clip table --- 
    try:
        logging.info(f"Loading test data from {DATA_DIR}")
        X_test_path = os.path.join(DATA_DIR, 'X_test.npy')
        Y_test_path = os.path.join(DATA_DIR, 'y_test.npy')
        Xv = np.load(X_test_path, mmap_mode="r")
        Yv = np.load(Y_test_path, mmap_mode="r")
        s2_clip_path = os.path.join(DATA_DIR, 'S2_clip.json')
        s2_clip = load_s2_clip_json(s2_clip_path)
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}. Make sure the data exists in '{DATA_DIR}'.")
        return

    C_in = Xv.shape[-1] if input_channel_select is None else 1
    C_out = Yv.shape[-1]
    H, W = Xv.shape[1], Xv.shape[2]
    logging.info(f"Input shape:  ({H}, {W}, {C_in})")
    logging.info(f"Output shape: ({H}, {W}, {C_out})")

    test_dataset = NPYPairDataset(X_test_path, Y_test_path, channel_select=input_channel_select, mmap=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )

    # --- Models (pix2pix) ---
    all_models = [
        {
            'model_class': Pix2PixWrapper,
            'model_name': 'pix2pix',
            'params': {'in_channels': C_in, 'out_channels': C_out, 'img_size': H},
            'model_file': 'pix2pix_best_netG.pth'
        }
    ]
    models_to_evaluate = [m for m in all_models if m['model_name'] in models_to_run]
    logging.info(f"Models to evaluate: {[m['model_name'] for m in models_to_evaluate]}")

    overall_csv = os.path.join(RESULTS_DIR, "metrics_overall.csv")
    perband_csv = os.path.join(RESULTS_DIR, "metrics_per_band.csv")

    # LPIPS setup once
    lpips_fn = None
    if _lpips_available:
        try:
            lpips_fn = lpips.LPIPS(net='alex').to(device)
            lpips_fn.eval()
        except Exception as e:
            logging.warning(f"Could not initialize LPIPS: {e}")
            lpips_fn = None
    else:
        logging.warning("lpips package not found; install with `pip install lpips` to enable LPIPS metric.")

    for model_config in models_to_evaluate:
        model_name = model_config['model_name']
        model_class = model_config['model_class']
        params = model_config['params']
        trained_model_dir = os.path.join(MODEL_DIR, model_name)
        model_path = os.path.join(trained_model_dir, model_config['model_file'])

        if not os.path.exists(model_path):
            logging.warning(f"Model '{model_name}' not found at {model_path}. Skipping.")
            continue

        # Load model
        logging.info(f"Loading model: {model_name} from {model_path}")
        model = model_class(**params)
        # state = torch.load(model_path, map_location=device, weights_only=True)
        # model.load_state_dict(state)

        state = None
        try:
            state = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)

        model = model.to(device)
        model.eval()

        logging.info(f"--- Evaluating {model_name} (streaming) ---")

        # ----- First pass: accumulate sums, mins/maxes, LPIPS, SAM -----
        # Per-channel sums for original-units errors
        abs_sum_c = np.zeros(C_out, dtype=np.float64)
        sq_sum_c  = np.zeros(C_out, dtype=np.float64)
        count_c   = 0  # number of pixels per channel

        # Global min/max (truth) for overall PSNR data_range
        global_min = float('inf')
        global_max = float('-inf')

        # Per-band min/max (truth) for band PSNR & per-band SSIM data_range
        band_min = np.full(C_out, np.inf, dtype=np.float64)
        band_max = np.full(C_out, -np.inf, dtype=np.float64)

        # LPIPS (mean over batches)
        lpips_vals = []

        # SAM accumulator (exact mean + approx quantiles)
        sam_acc = SAMAccumulator(sample_cap=2_000_000, rng_seed=123)

        with torch.no_grad():
            for batch_X, batch_y in tqdm(test_loader, desc="Evaluating", leave=False):
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                preds_scaled = outputs.detach().cpu().numpy()
                true_scaled  = batch_y.numpy()

                # Inverse to original units (per-batch)
                preds = inverse_from_tanh_using_clip_batch(preds_scaled, s2_clip)
                truth = inverse_from_tanh_using_clip_batch(true_scaled,  s2_clip)

                # Global min/max for PSNR data_range
                global_min = min(global_min, float(truth.min()))
                global_max = max(global_max, float(truth.max()))

                # Per-band min/max from truth
                yt = truth.reshape(-1, C_out).astype(np.float64)
                yp = preds.reshape(-1, C_out).astype(np.float64)

                band_min = np.minimum(band_min, yt.min(axis=0))
                band_max = np.maximum(band_max, yt.max(axis=0))

                # Per-channel sums (MAE/RMSE, sklearn-style)
                d = yt - yp
                abs_sum_c += np.abs(d).sum(axis=0)
                sq_sum_c  += (d * d).sum(axis=0)
                count_c   += yt.shape[0]

                # LPIPS (optional)
                if lpips_fn is not None:
                    lp = compute_lpips_rgb_batchwise(truth, preds, s2_clip, device, lpips_fn, bs=8)
                    lpips_vals.append(lp)

                # SAM streaming update
                sam_acc.update_batch(truth, preds, sample_stride=None)

        # --- Aggregated metrics (original units), sklearn-equivalent ---
        mae_c  = abs_sum_c / max(1, count_c)               # per-channel MAE
        mse_c  = sq_sum_c  / max(1, count_c)               # per-channel MSE
        mae_all  = float(np.mean(mae_c))                   # uniform_average
        rmse_all = float(np.sqrt(np.mean(mse_c)))          # sqrt(mean per-channel MSE)
        data_range = max(0.0, global_max - global_min)
        psnr_all = psnr_from_mse(float(np.mean(mse_c)), data_range)  # same MSE as RMSE

        # --- SSIM (overall & per-band) with fixed global data_range(s) ---
        global_dr = max(1e-9, data_range)
        band_dr   = np.maximum(1e-9, (band_max - band_min))  # per-band data_range

        ssim_sum = 0.0
        ssim_count = 0
        ssim_band_sums = np.zeros(C_out, dtype=np.float64)
        ssim_band_count = 0

        with torch.no_grad():
            for batch_X, batch_y in tqdm(test_loader, desc="SSIM pass", leave=False):
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                preds_scaled = outputs.detach().cpu().numpy()
                true_scaled  = batch_y.numpy()

                preds = inverse_from_tanh_using_clip_batch(preds_scaled, s2_clip)
                truth = inverse_from_tanh_using_clip_batch(true_scaled,  s2_clip)

                # Overall SSIM: global data_range for all images
                for i in range(truth.shape[0]):
                    s = structural_similarity(truth[i], preds[i], channel_axis=-1, data_range=global_dr)
                    ssim_sum += float(s)
                    ssim_count += 1

                # Per-band SSIM: fixed per-band data_range
                N = truth.shape[0]
                for c in range(C_out):
                    # use the same dr for all images (global per-band)
                    dr_c = float(band_dr[c])
                    yt = truth[..., c]
                    yp = preds[..., c]
                    # accumulate across images
                    for i in range(N):
                        s = structural_similarity(yt[i], yp[i], data_range=dr_c)
                        ssim_band_sums[c] += float(s)
                ssim_band_count += truth.shape[0]

        ssim_all = ssim_sum / max(1, ssim_count)
        ssim_per = (ssim_band_sums / max(1, ssim_band_count)).astype(float).tolist()

        # --- Per-band MAE/RMSE/PSNR (original units) ---
        rmse_per = np.sqrt(mse_c).astype(float).tolist()
        # PSNR per band uses per-band MSE and per-band data_range
        psnr_per = []
        for c in range(C_out):
            psnr_per.append(psnr_from_mse(float(mse_c[c]), float(band_dr[c])))

        # --- Normalized-space metrics ([-1,1]) – sklearn-equivalent ---
        abs_norm_c = np.zeros(C_out, dtype=np.float64)
        sq_norm_c  = np.zeros(C_out, dtype=np.float64)
        count_norm = 0

        with torch.no_grad():
            for batch_X, batch_y in tqdm(test_loader, desc="Normalized metrics", leave=False):
                out = model(batch_X.to(device))
                ps = out.detach().cpu().numpy()
                ts = batch_y.numpy()

                dt = ts.reshape(-1, C_out).astype(np.float64) - ps.reshape(-1, C_out).astype(np.float64)
                abs_norm_c += np.abs(dt).sum(axis=0)
                sq_norm_c  += (dt * dt).sum(axis=0)
                count_norm += dt.shape[0]

        mae_norm_per  = (abs_norm_c / max(1, count_norm)).astype(float).tolist()
        mse_norm_c    =  sq_norm_c  / max(1, count_norm)
        rmse_norm_per = np.sqrt(mse_norm_c).astype(float).tolist()

        mae_norm  = float(np.mean(mae_norm_per))          # uniform_average across channels
        rmse_norm = float(np.sqrt(np.mean(mse_norm_c)))   # sqrt(mean per-channel MSE)

        # --- LPIPS mean ---
        lpips_rgb_mean = float(np.mean(lpips_vals)) if lpips_vals else None
        if lpips_rgb_mean is not None:
            logging.info(f"  LPIPS_RGB: {lpips_rgb_mean:.6f}")
        else:
            logging.info("  LPIPS_RGB: (skipped or unavailable)")

        # --- SAM finalize ---
        logging.info("Computing SAM (degrees) in streaming mode...")
        sam_stats = sam_acc.finalize()

        # --- Log overall metrics ---
        logging.info("Overall metrics (original units):")
        logging.info(f"  MAE:   {mae_all:.6f}")
        logging.info(f"  RMSE:  {rmse_all:.6f}")
        logging.info(f"  PSNR:  {psnr_all:.6f} dB")
        logging.info(f"  SSIM:  {ssim_all:.6f}")
        logging.info(f"  SAM_mean_deg:   {sam_stats['SAM_mean_deg']:.4f}")
        logging.info(f"  SAM_median_deg: {sam_stats['SAM_median_deg']:.4f}")
        logging.info(f"  SAM_p25_deg:    {sam_stats['SAM_p25_deg']:.4f}")
        logging.info(f"  SAM_p75_deg:    {sam_stats['SAM_p75_deg']:.4f}")

        # --- Group summaries (from per-band ORIGINAL) ---
        perband_dict = {"MAE": mae_c.astype(float).tolist(),
                        "RMSE": rmse_per,
                        "PSNR": psnr_per,
                        "SSIM": ssim_per}
        group_summary = group_metrics_from_perband(perband_dict)

        # ---------------- Save CSV -----------------------------------------
        overall_headers = [
            "model","MAE","RMSE","PSNR","SSIM",
            "MAE_norm","RMSE_norm",
            "LPIPS_RGB",
            "SAM_mean_deg","SAM_median_deg","SAM_p25_deg","SAM_p75_deg"
        ] + sorted(group_summary.keys())

        overall_row = [
            model_name, mae_all, rmse_all, psnr_all, ssim_all,
            mae_norm, rmse_norm,
            (lpips_rgb_mean if lpips_rgb_mean is not None else ""),
            sam_stats["SAM_mean_deg"], sam_stats["SAM_median_deg"], sam_stats["SAM_p25_deg"], sam_stats["SAM_p75_deg"]
        ] + [group_summary[k] for k in sorted(group_summary.keys())]
        # --- Log per-band metrics (original units) ---
        logging.info("Per-band (original units):")
        for c, name in enumerate(BAND_NAMES):
            logging.info(
                f"  {name:>4} | MAE {mae_c[c]:8.3f} | RMSE {rmse_per[c]:8.3f} | "
                f"PSNR {psnr_per[c]:7.3f} | SSIM {ssim_per[c]:.4f}"
            )

        # --- Log group summaries ---
        logging.info("Group summaries (original units):")
        for k in sorted(group_summary.keys()):
            logging.info(f"  {k}: {group_summary[k]:.4f}")

        # --- (Optional) Log normalized-space per-band overview ---
        logging.info("Per-band (normalized [-1,1]):")
        for c, name in enumerate(BAND_NAMES):
            logging.info(
                f"  {name:>4} | MAE_norm {mae_norm_per[c]:8.5f} | RMSE_norm {rmse_norm_per[c]:8.5f}"
            )

        write_header = not os.path.exists(overall_csv)
        with open(overall_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(overall_headers)
            w.writerow(overall_row)

        # Per-band CSV
        perband_headers = ["model","band","MAE","RMSE","PSNR","SSIM","MAE_norm","RMSE_norm"]
        write_header_band = not os.path.exists(perband_csv)
        with open(perband_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header_band:
                w.writerow(perband_headers)
            for c, name in enumerate(BAND_NAMES):
                w.writerow([
                    model_name, name,
                    float(mae_c[c]), float(rmse_per[c]), float(psnr_per[c]), float(ssim_per[c]),
                    float(mae_norm_per[c]), float(rmse_norm_per[c])
                ])
        logging.info(f"S2_clip (first 3): {list(s2_clip.items())[:3]}  (expect 0..10000)")
        logging.info("-" * 50)

if __name__ == "__main__":
    main()
