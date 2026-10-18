import os, numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from PIL import Image

# -------------------- CONFIG --------------------
S1_CLIP = {"VV": (-25.0, 0.0), "VH": (-32.5, 0.0)}

# S2_CLIP = {
#     1: (975.0, 2010.9),  2: (740.0, 1952.0),  3: (529.0, 2292.0),
#     4: (341.0, 3243.0),  5: (299.0, 3444.0),  6: (266.0, 3743.0),
#     7: (252.0, 4140.0),  8: (216.0, 3993.0),  9: (214.0, 4293.0),
#     10:(88.0, 2014.0),   11:(7.0, 167.0),     12:(95.0, 5402.0),
#     13:(69.0, 4358.0),
# }

S2_CLIP = {
    1: (0.0, 10000.0),  2: (0.0, 10000.0),  3: (0.0, 10000.0),
    4: (0.0, 10000.0),  5: (0.0, 10000.0),  6: (0.0, 10000.0),
    7: (0.0, 10000.0),  8: (0.0, 10000.0),  9: (0.0, 10000.0),
    10:(0.0, 10000.0), 11:(0.0, 10000.0), 12:(0.0, 10000.0),
    13:(0.0, 10000.0),
}

# Fixed display windows in DN, keyed by 0-based band index, measured as p2-p98 over the
# 25,697-patch training split (see saturation_train_25697.json). Per channel rather than
# one shared window because blue sits much higher and narrower than red at TOA, so a
# single window renders the panels red-dominant. Set to None for per-image percentiles,
# which makes panels comparable within a row but not between rows.
S2_DISPLAY_WINDOWS_DN = {
    3: (300.0, 3450.0),   # B4 red
    2: (500.0, 2500.0),   # B3 green
    1: (700.0, 2100.0),   # B2 blue
}


class NPYPairDataset(Dataset):
    """Memory-mapped (X, y) pairs.

    Loading the arrays outright costs ~100 GB resident and roughly twice that during the
    numpy-to-tensor copy. Memory mapping leaves the pages to the OS cache instead.
    """

    def __init__(self, x_path, y_path, channel_select=None):
        self.x_path = x_path
        self.y_path = y_path
        self.channel_select = channel_select

        X = np.load(x_path, mmap_mode="r")
        Y = np.load(y_path, mmap_mode="r")

        if len(X) != len(Y):
            raise ValueError(f"Length mismatch: {x_path} has {len(X)}, {y_path} has {len(Y)}")

        self._len = len(X)
        self.x_shape = tuple(X.shape)
        self.y_shape = tuple(Y.shape)

        del X, Y

        # Opened on first access so each DataLoader worker gets its own handle.
        self.X = None
        self.Y = None

    def __len__(self):
        return self._len

    def _ensure_open(self):
        if self.X is None:
            self.X = np.load(self.x_path, mmap_mode="r")
            self.Y = np.load(self.y_path, mmap_mode="r")

    def __getitem__(self, idx):
        self._ensure_open()

        x = self.X[idx]
        y = self.Y[idx]

        if self.channel_select is not None:
            x = x[..., self.channel_select:self.channel_select + 1]

        return (
            torch.from_numpy(np.asarray(x, dtype=np.float32)),
            torch.from_numpy(np.asarray(y, dtype=np.float32)),
        )


def _sampled_range(path, n=32):
    """Range over the first n patches; a full min/max would read the whole file."""
    arr = np.load(path, mmap_mode="r")
    sub = np.asarray(arr[: min(n, arr.shape[0])], dtype=np.float32)
    return float(sub.min()), float(sub.max())


def get_train_val_loaders(
    train_data_folder,
    val_data_folder,
    batch_size,
    channel_select=None,
    drop_last=False,
    num_workers=16,
):
    train_dataset = NPYPairDataset(
        os.path.join(train_data_folder, "X_train.npy"),
        os.path.join(train_data_folder, "y_train.npy"),
        channel_select=channel_select,
    )

    val_dataset = NPYPairDataset(
        os.path.join(val_data_folder, "X_val.npy"),
        os.path.join(val_data_folder, "y_val.npy"),
        channel_select=channel_select,
    )

    y_train_lo, y_train_hi = _sampled_range(os.path.join(train_data_folder, "y_train.npy"))
    y_val_lo, y_val_hi = _sampled_range(os.path.join(val_data_folder, "y_val.npy"))

    print("\nData validation:")
    print(f"train: X {train_dataset.x_shape}  y {train_dataset.y_shape}")
    print(f"val:   X {val_dataset.x_shape}  y {val_dataset.y_shape}")
    print(f"y_train sampled range: [{y_train_lo}, {y_train_hi}]")
    print(f"y_val sampled range: [{y_val_lo}, {y_val_hi}]")

    if channel_select is not None:
        print(f"Selecting channel {channel_select} from input data.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


def save_training_plots_old(history, save_dir, model_name):
    """Save loss curves and learning rate schedule to PNG files"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Loss curves
    ax1.plot(history['loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Training History')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Learning rate schedule
    ax2.plot(history['lr'], color='green')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Learning Rate')
    
    plt.tight_layout()
    
    # Save and optionally display
    plot_path = os.path.join(save_dir, f'{model_name}_training_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def save_training_plots_separate(history, save_dir, model_name):
    """Save separate plots for train loss, val loss, and learning rate to PNG files."""
    import os
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # 1) Training loss (own figure)
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    train_plot_path = os.path.join(save_dir, f'{model_name}_train_loss.png')
    plt.tight_layout()
    plt.savefig(train_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 2) Validation loss (own figure)
    plt.figure(figsize=(10, 5))
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    val_plot_path = os.path.join(save_dir, f'{model_name}_val_loss.png')
    plt.tight_layout()
    plt.savefig(val_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 3) Learning rate schedule (own figure) — unchanged logic, just its own plot
    plt.figure(figsize=(10, 5))
    plt.plot(history['lr'])
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    lr_plot_path = os.path.join(save_dir, f'{model_name}_lr.png')
    plt.tight_layout()
    plt.savefig(lr_plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_training_plots(history, save_dir, model_name):
    """Save training loss, validation loss, and learning rate in separate subplots of one figure."""
    import os
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # Create figure with three subplots (stacked vertically)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    # 1) Training loss
    ax1.plot(history['loss'], label='Train Loss', color='blue')
    ax1.set_title('Training Loss')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # 2) Validation loss
    ax2.plot(history['val_loss'], label='Validation Loss', color='orange')
    ax2.set_title('Validation Loss')
    ax2.set_ylabel('Loss')
    ax2.legend()

    # 3) Learning rate schedule
    ax3.plot(history['lr'], color='green')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Learning Rate')

    plt.tight_layout()

    # Save figure
    plot_path = os.path.join(save_dir, f'{model_name}_training_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()



# ---------------- Visualization & denorm helpers (cleaned) --------------------

# Use your S1 clip config directly
S1_DB_RANGES = {
    "VV": S1_CLIP["VV"],
    "VH": S1_CLIP["VH"],
}

def percentile_limits_np(arr: np.ndarray, p_low: float = 2.0, p_high: float = 98.0):
    """Percentile limits of a single array, NaN-safe. Returns (lo, hi) or None if unusable."""
    a = arr.astype(np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        return None
    lo = np.nanpercentile(a[finite], p_low)
    hi = np.nanpercentile(a[finite], p_high)
    if not np.isfinite(lo):
        lo = np.nanmin(a)
    if not np.isfinite(hi):
        hi = np.nanmax(a)
    if hi <= lo:
        return None
    return float(lo), float(hi)


def apply_stretch_np(arr: np.ndarray, limits) -> np.ndarray:
    """Scale to uint8 using explicit (lo, hi) limits."""
    a = arr.astype(np.float32)
    finite = np.isfinite(a)
    if limits is None:
        return np.zeros_like(a, dtype=np.uint8)
    lo, hi = limits
    scaled = (np.clip(a, lo, hi) - lo) / (hi - lo)
    scaled = (scaled * 255.0).round().astype(np.uint8)
    scaled[~finite] = 0
    return scaled


def percentile_stretch_np(arr: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """2–98% percentile stretch to uint8 (handles NaNs)."""
    return apply_stretch_np(arr, percentile_limits_np(arr, p_low, p_high))

def from_tanh_to_range_torch(x_m11: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """[-1,1] -> [lo,hi] (torch, stays on device)"""
    return (x_m11 + 1.0) * 0.5 * (hi - lo) + lo

def band_denorm_s2_to_np(y_nhwc: torch.Tensor, band_idx_0based: int, clip_dict: dict) -> np.ndarray:
    """Denorm one S2 band from [-1,1] to its physical range, return numpy."""
    lo, hi = clip_dict[band_idx_0based + 1]
    band = from_tanh_to_range_torch(y_nhwc[..., band_idx_0based], lo, hi)
    return band.detach().cpu().numpy()

def _sar_vv_vh_uint8(x_nhwc: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """
    From SAR inputs in [-1,1] (dB-normalized), produce VV and VH grayscale uint8 panels.
    Steps:
      1) Denorm VV,VH back to dB using S1_CLIP
      2) Convert to linear power (10^(dB/10)) for nicer contrast
      3) Percentile-stretch to uint8
    Returns:
      vv_u8, vh_u8 each (N,H,W) uint8
    """
    C = x_nhwc.shape[-1]
    if C < 1:
        N, H, W = x_nhwc.shape[0], x_nhwc.shape[1], x_nhwc.shape[2]
        z = np.zeros((N, H, W), dtype=np.uint8)
        return z, z

    vv_norm = x_nhwc[..., 0]
    vh_norm = x_nhwc[..., 1] if C >= 2 else x_nhwc[..., 0]

    vv_db = from_tanh_to_range_torch(vv_norm, *S1_DB_RANGES["VV"])
    vh_db = from_tanh_to_range_torch(vh_norm, *S1_DB_RANGES["VH"])

    vv_lin = torch.pow(10.0, vv_db / 10.0)
    vh_lin = torch.pow(10.0, vh_db / 10.0)

    vv_np = vv_lin.detach().cpu().numpy()
    vh_np = vh_lin.detach().cpu().numpy()

    N = vv_np.shape[0]
    H, W = vv_np.shape[1], vv_np.shape[2]
    vv_u8 = np.zeros((N, H, W), dtype=np.uint8)
    vh_u8 = np.zeros((N, H, W), dtype=np.uint8)

    for i in range(N):
        vv_u8[i] = percentile_stretch_np(vv_np[i])
        vh_u8[i] = percentile_stretch_np(vh_np[i])

    return vv_u8, vh_u8

def _sar_rgb_from_norm_percentile(x_nhwc: torch.Tensor) -> np.ndarray:
    """
    Build SAR RGB from normalized SAR tensor:
      R = stretch(VV_lin), G = stretch(VH_lin), B = stretch(VV_lin / (VH_lin + eps))
    x_nhwc: (N,H,W,2) in [-1,1] that came from dB clipped ranges S1_CLIP.
    Returns: (N,H,W,3) uint8
    """
    C = x_nhwc.shape[-1]
    if C < 1:
        N, H, W = x_nhwc.shape[0], x_nhwc.shape[1], x_nhwc.shape[2]
        return np.zeros((N, H, W, 3), dtype=np.uint8)

    vv_norm = x_nhwc[..., 0]
    vh_norm = x_nhwc[..., 1] if C >= 2 else x_nhwc[..., 0]

    # [-1,1] -> dB
    vv_db = from_tanh_to_range_torch(vv_norm, *S1_CLIP["VV"])
    vh_db = from_tanh_to_range_torch(vh_norm, *S1_CLIP["VH"])

    # dB -> linear power
    vv_lin = torch.pow(10.0, vv_db / 10.0)
    vh_lin = torch.pow(10.0, vh_db / 10.0)

    # ratio
    eps = 1e-6
    ratio = vv_lin / (vh_lin + eps)

    # to numpy
    vv_np = vv_lin.detach().cpu().numpy()
    vh_np = vh_lin.detach().cpu().numpy()
    r_np  = ratio.detach().cpu().numpy()

    N, H, W = vv_np.shape[0], vv_np.shape[1], vv_np.shape[2]
    out = np.zeros((N, H, W, 3), dtype=np.uint8)
    for i in range(N):
        R = percentile_stretch_np(vv_np[i])
        G = percentile_stretch_np(vh_np[i])
        B = percentile_stretch_np(r_np[i])
        out[i] = np.stack([R, G, B], axis=-1)
    return out

def _s2_to_rgb_percentile(y_nhwc: torch.Tensor, rgb_idx=(3, 2, 1)) -> np.ndarray:
    """
    Build optical RGB from normalized S2 tensor using per-band denorm + percentile stretch.
    rgb_idx: 0-based band indices for (R,G,B). For true S2 RGB (B4,B3,B2) keep (3,2,1).
    Returns (N,H,W,3) uint8.
    """
    r_idx, g_idx, b_idx = rgb_idx
    r_np = band_denorm_s2_to_np(y_nhwc, r_idx, S2_CLIP)
    g_np = band_denorm_s2_to_np(y_nhwc, g_idx, S2_CLIP)
    b_np = band_denorm_s2_to_np(y_nhwc, b_idx, S2_CLIP)

    N, H, W = r_np.shape[0], r_np.shape[1], r_np.shape[2]
    out = np.zeros((N, H, W, 3), dtype=np.uint8)
    for i in range(N):
        R = percentile_stretch_np(r_np[i])
        G = percentile_stretch_np(g_np[i])
        B = percentile_stretch_np(b_np[i])
        out[i] = np.stack([R, G, B], axis=-1)
    return out

def _s2_pair_to_rgb_shared(pred_nhwc: torch.Tensor, target_nhwc: torch.Tensor, rgb_idx=(3, 2, 1)):
    """RGB panels for prediction and target under ONE stretch, taken from the target.

    Stretching each panel by its own statistics erases colour cast and radiometric bias,
    which is exactly what these panels are meant to reveal.
    Returns (pred_u8, target_u8), each (N,H,W,3).
    """
    N, H, W = target_nhwc.shape[0], target_nhwc.shape[1], target_nhwc.shape[2]
    pred_out = np.zeros((N, H, W, 3), dtype=np.uint8)
    tgt_out = np.zeros((N, H, W, 3), dtype=np.uint8)

    for ch, band_idx in enumerate(rgb_idx):
        p_np = band_denorm_s2_to_np(pred_nhwc, band_idx, S2_CLIP)
        t_np = band_denorm_s2_to_np(target_nhwc, band_idx, S2_CLIP)

        fixed = S2_DISPLAY_WINDOWS_DN.get(band_idx) if S2_DISPLAY_WINDOWS_DN else None

        for i in range(N):
            limits = fixed or percentile_limits_np(t_np[i])
            pred_out[i, ..., ch] = apply_stretch_np(p_np[i], limits)
            tgt_out[i, ..., ch] = apply_stretch_np(t_np[i], limits)

    return pred_out, tgt_out


@torch.no_grad()
def save_monitor_samples(epoch_idx: int, netG, batch_X_cpu: torch.Tensor, batch_y_cpu: torch.Tensor,
                         rgb_idx, out_root: str, include_input=True, device=None):
    out_dir = os.path.join(out_root, f"epoch_{epoch_idx+1:03d}")
    os.makedirs(out_dir, exist_ok=True)

    netG.eval()
    X = batch_X_cpu.to(device, non_blocking=True)  # (N,H,W,2) in [-1,1] dB-normed
    Y = batch_y_cpu.to(device, non_blocking=True)  # (N,H,W,13) in [-1,1] S2-normed

    pred = netG(X).detach()  # (N,H,W,13)

    # Build panels. Prediction and target share the target's stretch.
    sar_rgb_u8 = _sar_rgb_from_norm_percentile(X)               # (N,H,W,3) uint8
    pred_rgb_u8, tgt_rgb_u8 = _s2_pair_to_rgb_shared(pred, Y, rgb_idx=rgb_idx)

    # If you ALSO want VV/VH grayscale next to SAR RGB, flip this toggle True:
    ADD_VV_VH = False
    vv_u8 = vh_u8 = None
    if ADD_VV_VH:
        # quick reuse of SAR-to-grayscale (if you kept that helper); otherwise skip
        vv_u8, vh_u8 = _sar_vv_vh_uint8(X)

    N = pred_rgb_u8.shape[0]
    for i in range(N):
        tiles = []
        labels = []

        if include_input:
            tiles.append(sar_rgb_u8[i]); labels.append("sarRGB")
            if ADD_VV_VH:
                vv_rgb = np.stack([vv_u8[i]]*3, axis=-1)
                vh_rgb = np.stack([vh_u8[i]]*3, axis=-1)
                tiles.extend([vv_rgb, vh_rgb]); labels.extend(["VV","VH"])

        tiles.extend([pred_rgb_u8[i], tgt_rgb_u8[i]])
        labels.extend(["pred","tgt"])

        canvas = np.concatenate(tiles, axis=1)
        Image.fromarray(canvas).save(
            os.path.join(out_dir, f"sample_{i:02d}__{'_'.join(labels)}.png")
        )
