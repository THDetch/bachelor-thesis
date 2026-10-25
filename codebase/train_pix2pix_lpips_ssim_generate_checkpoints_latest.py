# The main training script.
# pix2pix -> model trained
# lpips and ssim -> optional losses incorporated during training
# generate -> monitor samples are saved periodically
# checkpoints -> last checkpoint + best checkpoints for multiple validation metrics
#
# Modified version:
# 1) Generator-only warmup for 10 epochs
# 2) G and D start with the same LR and decay together, with option to disable decay
# 3) Detailed logging/plotting of G/D losses and validation metrics
# 4) No early stopping; training always continues to the final epoch
# 5) Dataset split is unchanged; this script only consumes the prepared .npy files
# 6) Saves best models by Val L1, Val SSIM, Val LPIPS, and weighted validation total

import os
import sys
import time
import json
import csv
import random
import logging
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from myutils import get_train_val_loaders, save_monitor_samples
from arm_probe import load_probe, run_probe, _load_scene_ids


# -----------------------------------------------------------------------------
# Device setup
# -----------------------------------------------------------------------------
gpu_id = 0

if torch.cuda.is_available():
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    print(
        f"Using device: {torch.cuda.get_device_name(gpu_id)} "
        f"(id {torch.cuda.current_device()})"
    )
else:
    device = torch.device("cpu")
    print("Using device: CPU")


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
seed = 42


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    # Costs throughput, but config comparisons are meaningless without it.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(seed)


# -----------------------------------------------------------------------------
# Data / Experiment config
# -----------------------------------------------------------------------------
input_channel_select = None  # 0, 1, or None for all S1 channels

# -----------------------------------------------------------------------------
# Experiment arm
# -----------------------------------------------------------------------------
# A_base: current D (InstanceNorm + spectral norm), no extra term -> reference arm.
# Later arms change exactly one thing each; see success_criteria.md.
arm = "A1"

# norm="none" would also switch off conv bias, because NLayerDiscriminator derives
# use_bias from the norm layer's type. Build with InstanceNorm and strip it instead.
strip_norm_from_D = arm in ("A1", "A2", "A3")

# Mismatched pair (wrong SAR, real optical) labelled fake, so the condition becomes the
# only thing separating it from the real pair. Additive: the original terms keep their
# weights and the logged D loss stays the two-term form.
use_mismatch = arm in ("A0", "A3")

epochs = 20
batch_size = 16

# G and D start with the same LR.
# lr = 1e-4

# Separate learning rates
lr_G = 1e-4
lr_D = 5e-5   # D was winning outright; handicap it

# -----------------------------------------------------------------------------
# Warmup and update ratio
# -----------------------------------------------------------------------------
warmup_g_only_epochs = 0  # train only G for first N epochs
g_steps_per_iter = 1       # after warmup: G updates per batch
d_steps_per_iter = 1       # after warmup: D updates per batch


# Scheduler / LR decay configuration
# Off for the stability baseline: a decaying LR would confound the D/G dynamics.
use_lr_decay = False
lr_decay_patience = 10       # Plateau patience for both G and D
lr_decay_factor = 0.5
lr_min = 1e-6
decay_after_warmup_only = True  # Avoid decaying D before it even starts training
decay_starting_epoch = warmup_g_only_epochs
# If you resume from an old checkpoint and want to force both optimizers to a new LR,
# set this to 5e-5, or to {"G": 5e-5, "D": 2.5e-5}. Leave as None for normal checkpoint resume.
set_lrs_on_resume  = None

# A stale checkpoint with epoch + 1 >= epochs makes the training loop a silent no-op.
allow_resume = False

data_dir = "dataset_all_seasons_scenesplit"

model_name = "pix2pix"
run_name = f"arm_{arm}"
save_dir = f"trained_{run_name}/{model_name}"


# -----------------------------------------------------------------------------
# Network configuration
# -----------------------------------------------------------------------------
norm_G = "batch"      # InstanceNorm strips the per-image radiometric level we regress
norm_D = "instance"   # "none" would silently disable conv bias in NLayerDiscriminator
use_spectral_norm = True

# One-sided label smoothing belongs to D only; smoothing G's target caps its objective.
d_target_real = 0.9
g_target_real = 1.0

# How often (in batches) to measure how much of G's gradient comes from D. 0 disables.
grad_probe_every = 200

# Keep a snapshot every N epochs so post-hoc diagnostics can compare early vs late
# training. Best-model files alone only ever show the converged state. 0 disables.
checkpoint_every = 5

# Fixed validation patches for the per-epoch probe, identical across arms.
probe_n = 128

# An arm already below this by the given epoch has lost; stop burning GPU on it.
early_abort_epoch = 15
early_abort_d_out_fake = 0.02



# -----------------------------------------------------------------------------
# Optional perceptual losses
# -----------------------------------------------------------------------------
use_ssim = False
use_lpips = False

# lambda_GAN = 5.0
# lambda_L1 = 50.0
# lambda_ssim = 10.0
# lambda_lpips = 10.0

lambda_GAN = 1.0
lambda_L1 = 100.0
lambda_ssim = 10.0
lambda_lpips = 10.0

# If tensors are in [-1,1], SSIM data_range should be 2.0.
ssim_data_range = 2.0

# LPIPS needs 3 channels. For Sentinel-2: RGB usually B4, B3, B2.
# Your band indices are zero-based into the 13-band S2 tensor.
rgb_indices = (3, 2, 1)


# -----------------------------------------------------------------------------
# Best checkpoint behavior
# -----------------------------------------------------------------------------
save_best_by_val_l1 = True
save_best_by_val_ssim = True
save_best_by_val_lpips = True
save_best_by_val_total = True

# Which validation metric drives the LR scheduler.
# Options: "val_l1" or "val_total"
scheduler_metric = "val_l1"


# -----------------------------------------------------------------------------
# Add pix2pix project to the Python path
# -----------------------------------------------------------------------------
gan_repo_path = Path(__file__).parent / "mymodels/pytorch-CycleGAN-and-pix2pix"

if gan_repo_path.is_dir():
    sys.path.insert(0, str(gan_repo_path))
else:
    print(f"pytorch-CycleGAN-and-pix2pix repository not found at {gan_repo_path}")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
print("Loading data")

train_loader, val_loader = get_train_val_loaders(
    train_data_folder=data_dir,
    val_data_folder=data_dir,
    batch_size=batch_size,
    channel_select=input_channel_select,
    drop_last=True,   # a ragged final batch gives noisy BatchNorm statistics
    return_index=use_mismatch,
)

monitor_batch = next(iter(val_loader))
monitor_batch_X = monitor_batch[0][:8].cpu()
monitor_batch_y = monitor_batch[1][:8].cpu()

samples_dir = os.path.join(save_dir, "samples")
os.makedirs(samples_dir, exist_ok=True)

# Probe shapes using a single batch
first_batch = next(iter(train_loader))
data_shape_sar = first_batch[0].shape  # (N, H, W, C_in)
data_shape_opt = first_batch[1].shape  # (N, H, W, C_out)

print(f"SAR data shape from loader: {data_shape_sar}")

height, width, channels_in = data_shape_sar[1], data_shape_sar[2], data_shape_sar[3]
channels_out = data_shape_opt[3]

print(f"Dimensions: {height}x{width}")
print(f"Input channels: {channels_in}")
print(f"Output channels: {channels_out}")


from mymodels.cp2pwrapper import Pix2PixWrapper
from models.networks import define_D, GANLoss, init_net, apply_spectral_norm


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
netG = Pix2PixWrapper(
    in_channels=channels_in,
    out_channels=channels_out,
    img_size=height,
    norm=norm_G,
).to(device)

# define_D does not initialize; this fork initializes at the call site (see base_model.setup).
netD = define_D(
    input_nc=channels_in + channels_out,
    ndf=64,
    netD="basic",
    norm=norm_D,
)

if strip_norm_from_D:
    for module in netD.modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, torch.nn.InstanceNorm2d):
                setattr(module, child_name, torch.nn.Identity())

netD = init_net(netD, init_type="normal", init_gain=0.02)

if use_spectral_norm:
    netD = apply_spectral_norm(netD)

netD = netD.to(device)


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------
# def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):

criterionGAN_D = GANLoss(
    gan_mode="lsgan", target_real_label=d_target_real, target_fake_label=0.0
).to(device)

criterionGAN_G = GANLoss(
    gan_mode="lsgan", target_real_label=g_target_real, target_fake_label=0.0
).to(device)

criterionL1 = torch.nn.L1Loss()

ssim = None
lpips_fn = None

if use_ssim:
    try:
        from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
        ssim = StructuralSimilarityIndexMeasure(data_range=ssim_data_range).to(device)
    except Exception as e:
        print(f"[WARN] SSIM disabled (import/init failed): {e}")
        use_ssim = False

if use_lpips:
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg").to(device)
    except Exception as e:
        print(f"[WARN] LPIPS disabled (import/init failed): {e}")
        use_lpips = False


# -----------------------------------------------------------------------------
# Optimizers / Schedulers
# -----------------------------------------------------------------------------
optimizer_G = torch.optim.Adam(netG.parameters(), lr=lr_G, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(netD.parameters(), lr=lr_D, betas=(0.5, 0.999))

scheduler_G = None
scheduler_D = None

if use_lr_decay:
    scheduler_G = ReduceLROnPlateau(
        optimizer_G,
        mode="min",
        patience=lr_decay_patience,
        factor=lr_decay_factor,
        min_lr=lr_min,
    )

    scheduler_D = ReduceLROnPlateau(
        optimizer_D,
        mode="min",
        patience=lr_decay_patience,
        factor=lr_decay_factor,
        min_lr=lr_min,
    )


param_count_G = sum(p.numel() for p in netG.parameters() if p.requires_grad)
param_count_D = sum(p.numel() for p in netD.parameters() if p.requires_grad)

print(f"Generator parameter count: {param_count_G:,}")
print(f"Discriminator parameter count: {param_count_D:,}")


# -----------------------------------------------------------------------------
# Startup assertions: fail before burning GPU time on a misbuilt arm
# -----------------------------------------------------------------------------
NORM_TYPES = (torch.nn.modules.batchnorm._BatchNorm, torch.nn.InstanceNorm2d,
              torch.nn.GroupNorm, torch.nn.LayerNorm)


def _has_spectral_norm(m):
    # The fork uses the pre-parametrization API, which renames weight -> weight_orig.
    if hasattr(m, "weight_orig"):
        return True
    try:
        return torch.nn.utils.parametrize.is_parametrized(m, "weight")
    except (AttributeError, ValueError):
        return False


def _sn_weight(m):
    if hasattr(m, "weight_orig"):
        return m.weight_orig
    return m.parametrizations.weight.original


def check_arm_networks():
    print(f"\n[ARM {arm}] discriminator\n{netD}")

    convs = [m for m in netD.modules() if isinstance(m, torch.nn.Conv2d)]
    assert convs, "no Conv2d found in netD"

    for i, m in enumerate(convs):
        if use_spectral_norm:
            assert _has_spectral_norm(m), f"conv {i} is missing spectral norm"
        assert m.bias is not None, f"conv {i} has no bias"

    norms = [m for m in netD.modules() if isinstance(m, NORM_TYPES)]
    expect_norm = arm in ("A_base", "A0")
    assert bool(norms) == expect_norm, (
        f"arm {arm} expects norm layers in D: {expect_norm}, found {len(norms)}")

    assert param_count_D == 2_777_025, (
        f"netD has {param_count_D:,} parameters, expected 2,777,025; "
        "the arms must differ only by their norm modules")

    stds = [float(_sn_weight(m).std()) for m in convs] if use_spectral_norm \
        else [float(m.weight.std()) for m in convs]
    print("[ARM] conv weight std (init N(0, 0.02), applied before SN): "
          + ", ".join(f"{s:.4f}" for s in stds))
    assert all(0.012 < s < 0.030 for s in stds), f"init std out of range: {stds}"

    g_norms = [m for m in netG.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    assert g_norms, "netG is expected to use BatchNorm"

    print(f"[ARM] D convs {len(convs)}, norm modules {len(norms)}, "
          f"params {param_count_D:,}; G BatchNorm layers {len(g_norms)}  OK")


check_arm_networks()


# -----------------------------------------------------------------------------
# Fixed probe patches. Sliced from the memmaps, never through a DataLoader: creating
# a loader iterator draws from the global RNG and would shift the training shuffle.
# -----------------------------------------------------------------------------
probe = load_probe(data_dir, "val", n=probe_n, channel_select=input_channel_select)

probe_scores_dir = os.path.join(save_dir, "probe_scores")
os.makedirs(probe_scores_dir, exist_ok=True)

print(f"[PROBE] {len(probe['indices'])} fixed val patches, "
      f"{probe['n_scenes']} scenes, "
      f"{probe['same_scene_violations']} same-scene SAR violations")

# Scene id per training patch, used only to report how often roll(1) pairs a patch with
# one from its own scene. Such a pair is a weak negative, not a wrong one.
train_scenes = (_load_scene_ids(data_dir, "train", len(train_loader.dataset))
                if use_mismatch else None)


# -----------------------------------------------------------------------------
# Logging / Saving
# -----------------------------------------------------------------------------
os.makedirs(save_dir, exist_ok=True)

log_file = os.path.join(save_dir, f"{model_name}_training.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger()

config = {
    "arm": arm,
    "strip_norm_from_D": strip_norm_from_D,
    "use_mismatch": use_mismatch,
    "epochs": epochs,
    "batch_size": batch_size,
    "seed": seed,
    "grad_probe_every": grad_probe_every,
    "checkpoint_every": checkpoint_every,
    "probe_n": probe_n,
    "probe_indices": [int(v) for v in probe["indices"]],
    "probe_same_scene_violations": probe["same_scene_violations"],
    "early_abort_epoch": early_abort_epoch,
    "early_abort_d_out_fake": early_abort_d_out_fake,
    "lr_initial_G": lr_G,
    "lr_initial_D": lr_D,
    "use_lr_decay": use_lr_decay,
    "lr_decay_patience": lr_decay_patience,
    "lr_decay_factor": lr_decay_factor,
    "lr_min": lr_min,
    "decay_after_warmup_only": decay_after_warmup_only,
    "scheduler_metric": scheduler_metric,
    "allow_resume": allow_resume,
    "data_dir": data_dir,
    "model_name": model_name,
    "run_name": run_name,
    "save_dir": save_dir,
    "norm_G": norm_G,
    "norm_D": norm_D,
    "use_spectral_norm": use_spectral_norm,
    "d_target_real": d_target_real,
    "g_target_real": g_target_real,
    "warmup_g_only_epochs": warmup_g_only_epochs,
    "param_count_G": param_count_G,
    "param_count_D": param_count_D,
    "g_steps_per_iter": g_steps_per_iter,
    "d_steps_per_iter": d_steps_per_iter,
    "use_ssim": use_ssim,
    "use_lpips": use_lpips,
    "lambda_GAN": lambda_GAN,
    "lambda_L1": lambda_L1,
    "lambda_ssim": lambda_ssim,
    "lambda_lpips": lambda_lpips,
    "ssim_data_range": ssim_data_range,
    "rgb_indices": rgb_indices,
    "input_channel_select": input_channel_select,
}

with open(os.path.join(save_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

logger.info("========== CONFIG ==========")
for k, v in config.items():
    logger.info(f"{k}: {v}")


# -----------------------------------------------------------------------------
# History
# -----------------------------------------------------------------------------
history = {
    "epoch": [],

    # Train losses
    "train_G_total": [],
    "train_G_GAN": [],
    "train_G_L1_weighted": [],
    "train_G_L1_raw": [],
    "train_SSIM_loss": [],
    "train_SSIM_metric": [],
    "train_LPIPS_loss": [],
    "train_LPIPS_metric": [],

    # Discriminator
    "train_D_loss": [],
    "train_D_real": [],
    "train_D_fake": [],
    "train_D_mismatch": [],
    "train_D_out_real": [],
    "train_D_out_fake": [],
    "train_D_out_mismatch": [],
    "train_mismatch_same_scene_frac": [],

    # How much of G's gradient comes from the adversarial term
    "train_gan_l1_grad_ratio": [],
    "train_gan_l1_grad_ratio_param": [],

    # Fixed validation patches, identical across arms
    "probe_D_out_real": [],
    "probe_D_out_fake": [],
    "probe_hf_ratio": [],
    "probe_hf_corr": [],
    "probe_std_ratio": [],
    "probe_deg_shuffle": [],
    "probe_deg_shuffle_lo": [],
    "probe_deg_shuffle_hi": [],

    # Validation
    "val_L1": [],
    "val_SSIM": [],
    "val_LPIPS": [],
    "val_total": [],

    # LR
    "lr_G": [],
    "lr_D": [],
}


best_val_l1 = float("inf")
best_val_lpips = float("inf")
best_val_total = float("inf")
best_val_ssim = -float("inf")


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------
checkpoint_path = os.path.join(save_dir, "checkpoint_last.pt")
start_epoch = 0


def save_checkpoint(epoch_idx: int):
    ckpt = {
        "epoch": epoch_idx,
        "netG": netG.state_dict(),
        "netD": netD.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict(),
        "scheduler_G": scheduler_G.state_dict() if scheduler_G is not None else None,
        "scheduler_D": scheduler_D.state_dict() if scheduler_D is not None else None,
        "best_val_l1": best_val_l1,
        "best_val_lpips": best_val_lpips,
        "best_val_ssim": best_val_ssim,
        "best_val_total": best_val_total,
        "history": history,
        "config": config,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }

    torch.save(ckpt, checkpoint_path)


def try_resume():
    global start_epoch
    global best_val_l1, best_val_lpips, best_val_ssim, best_val_total
    global history

    if not os.path.isfile(checkpoint_path):
        print("[RESUME] No checkpoint found. Starting fresh.")
        return

    print(f"[RESUME] Loading checkpoint from {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    netG.load_state_dict(ckpt["netG"])
    netD.load_state_dict(ckpt["netD"])
    optimizer_G.load_state_dict(ckpt["optimizer_G"])
    optimizer_D.load_state_dict(ckpt["optimizer_D"])

    if use_lr_decay:
        if scheduler_G is not None and ckpt.get("scheduler_G") is not None:
            scheduler_G.load_state_dict(ckpt["scheduler_G"])
        if scheduler_D is not None and ckpt.get("scheduler_D") is not None:
            scheduler_D.load_state_dict(ckpt["scheduler_D"])

    best_val_l1 = ckpt.get("best_val_l1", ckpt.get("best_val_loss", best_val_l1))
    best_val_lpips = ckpt.get("best_val_lpips", best_val_lpips)
    best_val_ssim = ckpt.get("best_val_ssim", best_val_ssim)
    best_val_total = ckpt.get("best_val_total", best_val_total)

    loaded_history = ckpt.get("history", None)
    if isinstance(loaded_history, dict):
        for key in history.keys():
            if key in loaded_history:
                history[key] = loaded_history[key]

    start_epoch = ckpt.get("epoch", -1) + 1

    rng = ckpt.get("rng_state")
    if rng is not None:
        # map_location moved these to the GPU; the setters expect CPU tensors.
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])

    print(f"[RESUME] Resuming at epoch {start_epoch + 1} of {epochs}")


if allow_resume:
    try_resume()
else:
    print("[RESUME] Disabled by config (allow_resume=False). Starting fresh.")

# Optional: force both G and D to the same LR after resume.
if set_lrs_on_resume is not None and start_epoch > 0:
    if isinstance(set_lrs_on_resume, dict):
        forced_lr_G = set_lrs_on_resume["G"]
        forced_lr_D = set_lrs_on_resume["D"]
    else:
        forced_lr_G = forced_lr_D = float(set_lrs_on_resume)

    for pg in optimizer_G.param_groups:
        pg["lr"] = forced_lr_G

    for pg in optimizer_D.param_groups:
        pg["lr"] = forced_lr_D

    print(
        f"[LR OVERRIDE] Resumed checkpoint. "
        f"Set G LR to {forced_lr_G:.2e}, D LR to {forced_lr_D:.2e}"
    )
    logger.info(
        f"[LR OVERRIDE] Resumed checkpoint. "
        f"Set G LR to {forced_lr_G:.2e}, D LR to {forced_lr_D:.2e}"
    )

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def set_requires_grad(net, flag: bool):
    for p in net.parameters():
        p.requires_grad = flag


def _total_grad_norm(grads):
    sq = [g.detach().pow(2).sum() for g in grads if g is not None]
    return float(torch.sqrt(torch.stack(sq).sum())) if sq else 0.0


def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2)


def select_rgb_from_nhwc(x: torch.Tensor, rgb_idx) -> torch.Tensor:
    r, g, b = rgb_idx
    return torch.stack([x[..., r], x[..., g], x[..., b]], dim=-1)


def compute_val_total(val_l1, val_ssim, val_lpips):
    """
    Weighted validation objective comparable to the non-adversarial part
    of the generator loss.
    """
    total = lambda_L1 * val_l1

    if use_ssim and val_ssim is not None:
        total += lambda_ssim * (1.0 - val_ssim)

    if use_lpips and val_lpips is not None:
        total += lambda_lpips * val_lpips

    return total


def save_history_csv():
    csv_path = os.path.join(save_dir, "history.csv")

    keys = list(history.keys())
    rows = zip(*[history[k] for k in keys])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)


def save_history_json():
    json_path = os.path.join(save_dir, "history.json")

    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)


def plot_curve(x, ys, labels, title, ylabel, filename):
    plt.figure(figsize=(12, 6))

    for y, label in zip(ys, labels):
        # Skip if the series is all None
        if all(v is None for v in y):
            continue
        plt.plot(x, y, label=label)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=200)
    plt.close()


def save_detailed_training_plots():
    if len(history["epoch"]) == 0:
        return

    x = history["epoch"]

    plot_curve(
        x,
        [
            history["train_G_total"],
            history["train_G_GAN"],
            history["train_G_L1_weighted"],
            history["train_SSIM_loss"],
            history["train_LPIPS_loss"],
        ],
        [
            "G total",
            "G GAN",
            "G L1 weighted",
            "G SSIM loss",
            "G LPIPS loss",
        ],
        "Training Generator Loss Components",
        "Loss",
        "training_generator_loss_components.png",
    )

    plot_curve(
        x,
        [
            history["train_D_loss"],
            history["train_D_real"],
            history["train_D_fake"],
        ],
        [
            "D total",
            "D real",
            "D fake",
        ],
        "Training Discriminator Loss Components",
        "Loss",
        "training_discriminator_loss_components.png",
    )

    plot_curve(
        x,
        [
            history["train_D_out_real"],
            history["train_D_out_fake"],
        ],
        [
            "D(real)",
            "D(fake)",
        ],
        "Discriminator Raw Output (collapse: D(fake) -> 0)",
        "Mean D output",
        "discriminator_outputs.png",
    )

    plot_curve(
        x,
        [
            history["train_gan_l1_grad_ratio"],
            history["train_gan_l1_grad_ratio_param"],
        ],
        [
            "||grad GAN|| / ||grad L1|| at fake_B",
            "same ratio over G parameters",
        ],
        "GAN/L1 Gradient Norm Ratio",
        "Gradient norm ratio",
        "gan_l1_gradient_ratio.png",
    )

    plot_curve(
        x,
        [
            history["probe_hf_ratio"],
            history["probe_hf_corr"],
            history["probe_std_ratio"],
        ],
        [
            "land-band HF std ratio",
            "land-band HF correlation",
            "within-patch std ratio",
        ],
        "Generator Detail vs Target (128 fixed val patches)",
        "pred / target",
        "probe_detail_ratios.png",
    )

    plot_curve(
        x,
        [
            history["probe_deg_shuffle"],
            history["probe_deg_shuffle_lo"],
            history["probe_deg_shuffle_hi"],
        ],
        [
            "deg_shuffle",
            "95% CI low",
            "95% CI high",
        ],
        "Does D Use the SAR Condition? (0 = ignores it)",
        "degradation toward fake",
        "probe_deg_shuffle.png",
    )

    plot_curve(
        x,
        [
            history["train_G_L1_raw"],
            history["val_L1"],
        ],
        [
            "Train L1 raw",
            "Val L1",
        ],
        "Raw L1 Reconstruction Loss",
        "L1",
        "l1_train_vs_val.png",
    )

    plot_curve(
        x,
        [
            history["train_SSIM_metric"],
            history["val_SSIM"],
        ],
        [
            "Train SSIM",
            "Val SSIM",
        ],
        "SSIM Metric",
        "SSIM",
        "ssim_train_vs_val.png",
    )

    plot_curve(
        x,
        [
            history["train_LPIPS_metric"],
            history["val_LPIPS"],
        ],
        [
            "Train LPIPS",
            "Val LPIPS",
        ],
        "LPIPS Metric",
        "LPIPS",
        "lpips_train_vs_val.png",
    )

    plot_curve(
        x,
        [
            history["val_total"],
        ],
        [
            "Val weighted total",
        ],
        "Weighted Validation Objective",
        "Weighted validation objective",
        "validation_weighted_total.png",
    )

    plot_curve(
        x,
        [
            history["lr_G"],
            history["lr_D"],
        ],
        [
            "LR Generator",
            "LR Discriminator",
        ],
        "Learning Rate Schedule",
        "Learning rate",
        "learning_rate_schedule.png",
    )


def save_best_models(metric_name):
    torch.save(
        netG.state_dict(),
        os.path.join(save_dir, f"{model_name}_best_{metric_name}_netG.pth"),
    )

    torch.save(
        netD.state_dict(),
        os.path.join(save_dir, f"{model_name}_best_{metric_name}_netD.pth"),
    )


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
global_step = 0

for epoch in tqdm(range(start_epoch, epochs), desc="Training", unit="epoch", position=0):
    epoch_start = time.time()

    netG.train()
    netD.train()

    # G-only warmup
    g_only = epoch < warmup_g_only_epochs

    train_G_total_sum = 0.0
    train_G_GAN_sum = 0.0
    train_G_L1_weighted_sum = 0.0
    train_G_L1_raw_sum = 0.0
    train_SSIM_loss_sum = 0.0
    train_SSIM_metric_sum = 0.0
    train_LPIPS_loss_sum = 0.0
    train_LPIPS_metric_sum = 0.0

    train_D_loss_sum = 0.0
    train_D_real_sum = 0.0
    train_D_fake_sum = 0.0
    train_D_mismatch_sum = 0.0
    train_D_out_real_sum = 0.0
    train_D_out_fake_sum = 0.0
    train_D_out_mismatch_sum = 0.0
    same_scene_sum = 0

    grad_ratio_sum = 0.0
    grad_ratio_count = 0
    grad_ratio_param_sum = 0.0

    num_samples_seen = 0

    with tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}",
        unit="batch",
        leave=False,
        position=1,
    ) as batch_pbar:

        for batch in batch_pbar:
            if use_mismatch:
                real_A, real_B, batch_idx = batch
                partner = batch_idx.roll(1, 0)
                same_scene_sum += sum(
                    train_scenes[int(a)] == train_scenes[int(b)]
                    for a, b in zip(batch_idx.tolist(), partner.tolist()))
            else:
                real_A, real_B = batch

            real_A = real_A.to(device)
            real_B = real_B.to(device)

            bsz = real_A.size(0)
            num_samples_seen += bsz
            global_step += 1

            # One generator forward per iteration; D consumes a detached copy.
            fake_B = netG(real_A)

            loss_D = torch.tensor(0.0, device=device)
            loss_D_real = torch.tensor(0.0, device=device)
            loss_D_fake = torch.tensor(0.0, device=device)

            # ---------------------------
            # (1) Discriminator updates
            # ---------------------------
            if not g_only and d_steps_per_iter > 0:
                set_requires_grad(netD, True)

                running_D_loss = 0.0
                running_D_real = 0.0
                running_D_fake = 0.0
                running_D_mismatch = 0.0
                running_D_out_real = 0.0
                running_D_out_fake = 0.0
                running_D_out_mismatch = 0.0

                for d_step in range(d_steps_per_iter):
                    optimizer_D.zero_grad()

                    if d_step == 0:
                        fake_B_detached = fake_B.detach()
                    else:
                        with torch.no_grad():
                            fake_B_detached = netG(real_A)

                    pred_real = netD(
                        torch.cat((real_A, real_B), -1).permute(0, 3, 1, 2)
                    )

                    pred_fake = netD(
                        torch.cat((real_A, fake_B_detached), -1).permute(0, 3, 1, 2)
                    )

                    loss_D_real = criterionGAN_D(pred_real, True)
                    loss_D_fake = criterionGAN_D(pred_fake, False)

                    loss_D_mismatch = torch.tensor(0.0, device=device)
                    pred_mismatch_mean = 0.0

                    if use_mismatch:
                        # Real optical, wrong SAR. Roll rather than randperm so no
                        # sample can be paired with itself.
                        pred_mismatch = netD(
                            torch.cat((real_A.roll(1, 0), real_B), -1).permute(0, 3, 1, 2)
                        )
                        loss_D_mismatch = criterionGAN_D(pred_mismatch, False)
                        pred_mismatch_mean = pred_mismatch.mean().item()

                    # Logged D loss keeps the two-term form so it stays comparable
                    # across arms; the mismatch term is added only for the update.
                    loss_D = 0.5 * (loss_D_real + loss_D_fake)

                    (loss_D + 0.5 * loss_D_mismatch).backward()
                    optimizer_D.step()

                    running_D_loss += loss_D.item()
                    running_D_real += loss_D_real.item()
                    running_D_fake += loss_D_fake.item()
                    running_D_mismatch += loss_D_mismatch.item()
                    running_D_out_real += pred_real.mean().item()
                    running_D_out_fake += pred_fake.mean().item()
                    running_D_out_mismatch += pred_mismatch_mean

                avg_D_loss = running_D_loss / d_steps_per_iter
                avg_D_real = running_D_real / d_steps_per_iter
                avg_D_fake = running_D_fake / d_steps_per_iter
                avg_D_mismatch = running_D_mismatch / d_steps_per_iter
                avg_D_out_real = running_D_out_real / d_steps_per_iter
                avg_D_out_fake = running_D_out_fake / d_steps_per_iter
                avg_D_out_mismatch = running_D_out_mismatch / d_steps_per_iter

                train_D_loss_sum += avg_D_loss * bsz
                train_D_real_sum += avg_D_real * bsz
                train_D_fake_sum += avg_D_fake * bsz
                train_D_mismatch_sum += avg_D_mismatch * bsz
                train_D_out_real_sum += avg_D_out_real * bsz
                train_D_out_fake_sum += avg_D_out_fake * bsz
                train_D_out_mismatch_sum += avg_D_out_mismatch * bsz

            else:
                set_requires_grad(netD, False)
                avg_D_loss = 0.0
                avg_D_real = 0.0
                avg_D_fake = 0.0
                avg_D_mismatch = 0.0
                avg_D_out_real = 0.0
                avg_D_out_fake = 0.0
                avg_D_out_mismatch = 0.0

            # ---------------------------
            # (2) Generator updates
            # ---------------------------
            set_requires_grad(netD, False)

            running_G_total = 0.0
            running_G_GAN = 0.0
            running_G_L1_weighted = 0.0
            running_G_L1_raw = 0.0
            running_SSIM_loss = 0.0
            running_SSIM_metric = 0.0
            running_LPIPS_loss = 0.0
            running_LPIPS_metric = 0.0

            for g_step in range(max(1, g_steps_per_iter)):
                optimizer_G.zero_grad()

                # Extra G steps need a fresh graph; the first reuses the forward above.
                if g_step > 0:
                    fake_B = netG(real_A)

                # During G-only warmup, do NOT use GAN loss against a random/frozen D.
                if g_only:
                    loss_G_GAN = torch.tensor(0.0, device=device)
                else:
                    pred_fake = netD(
                        torch.cat((real_A, fake_B), -1).permute(0, 3, 1, 2)
                    )
                    loss_G_GAN = criterionGAN_G(pred_fake, True)

                loss_G_L1_raw = criterionL1(fake_B, real_B)
                loss_G_L1_weighted = loss_G_L1_raw * lambda_L1

                loss_SSIM = torch.tensor(0.0, device=device)
                ssim_val_for_log = None

                if use_ssim and ssim is not None:
                    ssim_val_for_log = ssim(
                        nhwc_to_nchw(fake_B),
                        nhwc_to_nchw(real_B),
                    )
                    loss_SSIM = (1.0 - ssim_val_for_log) * lambda_ssim

                loss_LPIPS = torch.tensor(0.0, device=device)
                lpips_val_for_log = None

                if use_lpips and lpips_fn is not None:
                    fake_rgb = select_rgb_from_nhwc(fake_B, rgb_indices)
                    real_rgb = select_rgb_from_nhwc(real_B, rgb_indices)

                    # Already in [-1, 1]
                    lpips_val_for_log = lpips_fn(
                        nhwc_to_nchw(fake_rgb),
                        nhwc_to_nchw(real_rgb),
                    ).mean()

                    loss_LPIPS = lpips_val_for_log * lambda_lpips

                loss_G = lambda_GAN * loss_G_GAN + loss_G_L1_weighted + loss_SSIM + loss_LPIPS

                # Probe before backward: a pinned GAN loss can still look large while
                # contributing almost no gradient. autograd.grad does not touch .grad.
                if (
                    not g_only
                    and grad_probe_every > 0
                    and global_step % grad_probe_every == 0
                ):
                    grad_gan = torch.autograd.grad(
                        lambda_GAN * loss_G_GAN, fake_B, retain_graph=True
                    )[0].norm().item()

                    grad_l1 = torch.autograd.grad(
                        loss_G_L1_weighted, fake_B, retain_graph=True
                    )[0].norm().item()

                    grad_ratio_sum += grad_gan / (grad_l1 + 1e-12)

                    # The ratio at fake_B says nothing about how much survives backprop
                    # into the weights, which is what the optimizer actually steps on.
                    g_params = [p for p in netG.parameters() if p.requires_grad]
                    p_gan = _total_grad_norm(torch.autograd.grad(
                        lambda_GAN * loss_G_GAN, g_params,
                        retain_graph=True, allow_unused=True))
                    p_l1 = _total_grad_norm(torch.autograd.grad(
                        loss_G_L1_weighted, g_params,
                        retain_graph=True, allow_unused=True))

                    grad_ratio_param_sum += p_gan / (p_l1 + 1e-12)
                    grad_ratio_count += 1

                loss_G.backward()
                optimizer_G.step()

                running_G_total += loss_G.item()
                running_G_GAN += loss_G_GAN.item()
                running_G_L1_weighted += loss_G_L1_weighted.item()
                running_G_L1_raw += loss_G_L1_raw.item()
                running_SSIM_loss += loss_SSIM.item()

                if ssim_val_for_log is not None:
                    running_SSIM_metric += ssim_val_for_log.item()

                running_LPIPS_loss += loss_LPIPS.item()

                if lpips_val_for_log is not None:
                    running_LPIPS_metric += lpips_val_for_log.item()

            denom_g = max(1, g_steps_per_iter)

            avg_G_total = running_G_total / denom_g
            avg_G_GAN = running_G_GAN / denom_g
            avg_G_L1_weighted = running_G_L1_weighted / denom_g
            avg_G_L1_raw = running_G_L1_raw / denom_g
            avg_SSIM_loss = running_SSIM_loss / denom_g
            avg_LPIPS_loss = running_LPIPS_loss / denom_g

            avg_SSIM_metric = (
                running_SSIM_metric / denom_g
                if use_ssim and ssim is not None
                else None
            )

            avg_LPIPS_metric = (
                running_LPIPS_metric / denom_g
                if use_lpips and lpips_fn is not None
                else None
            )

            train_G_total_sum += avg_G_total * bsz
            train_G_GAN_sum += avg_G_GAN * bsz
            train_G_L1_weighted_sum += avg_G_L1_weighted * bsz
            train_G_L1_raw_sum += avg_G_L1_raw * bsz
            train_SSIM_loss_sum += avg_SSIM_loss * bsz
            train_LPIPS_loss_sum += avg_LPIPS_loss * bsz

            if avg_SSIM_metric is not None:
                train_SSIM_metric_sum += avg_SSIM_metric * bsz

            if avg_LPIPS_metric is not None:
                train_LPIPS_metric_sum += avg_LPIPS_metric * bsz

            batch_pbar.set_postfix(
                {
                    "G": f"{avg_G_total:.3f}",
                    "D": f"{avg_D_loss:.3f}",
                    "D(f)": f"{avg_D_out_fake:.3f}",
                    "GAN": f"{avg_G_GAN:.3f}",
                    "L1": f"{avg_G_L1_weighted:.3f}",
                    "SSIM": f"{avg_SSIM_loss:.3f}" if use_ssim else "off",
                    "LPIPS": f"{avg_LPIPS_loss:.3f}" if use_lpips else "off",
                    "lrG": f"{optimizer_G.param_groups[0]['lr']:.2e}",
                    "lrD": f"{optimizer_D.param_groups[0]['lr']:.2e}",
                }
            )

    # ---------------------------
    # Train epoch averages
    # ---------------------------
    n_train = max(1, num_samples_seen)

    train_G_total = train_G_total_sum / n_train
    train_G_GAN = train_G_GAN_sum / n_train
    train_G_L1_weighted = train_G_L1_weighted_sum / n_train
    train_G_L1_raw = train_G_L1_raw_sum / n_train
    train_SSIM_loss = train_SSIM_loss_sum / n_train
    train_LPIPS_loss = train_LPIPS_loss_sum / n_train

    train_SSIM_metric = (
        train_SSIM_metric_sum / n_train
        if use_ssim and ssim is not None
        else None
    )

    train_LPIPS_metric = (
        train_LPIPS_metric_sum / n_train
        if use_lpips and lpips_fn is not None
        else None
    )

    train_D_loss = train_D_loss_sum / n_train
    train_D_real = train_D_real_sum / n_train
    train_D_fake = train_D_fake_sum / n_train
    train_D_mismatch = train_D_mismatch_sum / n_train
    train_D_out_real = train_D_out_real_sum / n_train
    train_D_out_fake = train_D_out_fake_sum / n_train
    train_D_out_mismatch = train_D_out_mismatch_sum / n_train
    mismatch_same_scene_frac = same_scene_sum / n_train if use_mismatch else None

    train_grad_ratio = (
        grad_ratio_sum / grad_ratio_count if grad_ratio_count > 0 else None
    )

    train_grad_ratio_param = (
        grad_ratio_param_sum / grad_ratio_count if grad_ratio_count > 0 else None
    )


    # -----------------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------------
    val_l1_sum = 0.0
    val_ssim_sum = 0.0
    val_lpips_sum = 0.0
    n_val = 0

    netG.eval()

    with tqdm(
        val_loader,
        desc="Validating",
        unit="batch",
        leave=False,
        position=2,
    ) as val_pbar:

        with torch.no_grad():
            for batch in val_pbar:
                batch_X, batch_y = batch[0], batch[1]
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                bsz = batch_X.size(0)
                n_val += bsz

                outputs = netG(batch_X)

                l1 = criterionL1(outputs, batch_y).item()
                val_l1_sum += l1 * bsz

                if use_ssim and ssim is not None:
                    ssim_val = ssim(
                        nhwc_to_nchw(outputs),
                        nhwc_to_nchw(batch_y),
                    ).item()
                    val_ssim_sum += ssim_val * bsz

                if use_lpips and lpips_fn is not None:
                    out_rgb = select_rgb_from_nhwc(outputs, rgb_indices)
                    tgt_rgb = select_rgb_from_nhwc(batch_y, rgb_indices)

                    lp = lpips_fn(
                        nhwc_to_nchw(out_rgb),
                        nhwc_to_nchw(tgt_rgb),
                    ).mean().item()

                    val_lpips_sum += lp * bsz

                val_pbar.set_postfix({"val_L1": f"{l1:.6f}"})

    n_val = max(1, n_val)

    val_l1 = val_l1_sum / n_val

    val_ssim = (
        val_ssim_sum / n_val
        if use_ssim and ssim is not None
        else None
    )

    val_lpips = (
        val_lpips_sum / n_val
        if use_lpips and lpips_fn is not None
        else None
    )

    val_total = compute_val_total(val_l1, val_ssim, val_lpips)

    # Fixed probe patches. Runs eval-mode under no_grad and restores the previous modes:
    # a train-mode forward would advance spectral norm's power iteration and change D.
    probe_metrics, probe_scores = run_probe(netG, netD, probe, device, batch=batch_size)

    np.savez(
        os.path.join(probe_scores_dir, f"epoch_{epoch + 1:03d}.npz"),
        **probe_scores,
    )


    # -----------------------------------------------------------------------------
    # LR scheduling: G and D decay together
    # -----------------------------------------------------------------------------
    metric_for_scheduler = val_l1 if scheduler_metric == "val_l1" else val_total

    scheduler_active_this_epoch = True

    if decay_after_warmup_only and epoch < decay_starting_epoch:
        scheduler_active_this_epoch = False

    if use_lr_decay and scheduler_active_this_epoch:
        scheduler_G.step(metric_for_scheduler)
        scheduler_D.step(metric_for_scheduler)

    current_lr_G = optimizer_G.param_groups[0]["lr"]
    current_lr_D = optimizer_D.param_groups[0]["lr"]


    # -----------------------------------------------------------------------------
    # Save history
    # -----------------------------------------------------------------------------
    history["epoch"].append(epoch + 1)

    history["train_G_total"].append(train_G_total)
    history["train_G_GAN"].append(train_G_GAN)
    history["train_G_L1_weighted"].append(train_G_L1_weighted)
    history["train_G_L1_raw"].append(train_G_L1_raw)
    history["train_SSIM_loss"].append(train_SSIM_loss)
    history["train_SSIM_metric"].append(train_SSIM_metric)
    history["train_LPIPS_loss"].append(train_LPIPS_loss)
    history["train_LPIPS_metric"].append(train_LPIPS_metric)

    history["train_D_loss"].append(train_D_loss)
    history["train_D_real"].append(train_D_real)
    history["train_D_fake"].append(train_D_fake)
    history["train_D_mismatch"].append(train_D_mismatch if use_mismatch else None)
    history["train_D_out_real"].append(train_D_out_real)
    history["train_D_out_fake"].append(train_D_out_fake)
    history["train_D_out_mismatch"].append(train_D_out_mismatch if use_mismatch else None)
    history["train_mismatch_same_scene_frac"].append(mismatch_same_scene_frac)
    history["train_gan_l1_grad_ratio"].append(train_grad_ratio)
    history["train_gan_l1_grad_ratio_param"].append(train_grad_ratio_param)

    for k, v in probe_metrics.items():
        history[k].append(v)

    history["val_L1"].append(val_l1)
    history["val_SSIM"].append(val_ssim)
    history["val_LPIPS"].append(val_lpips)
    history["val_total"].append(val_total)

    history["lr_G"].append(current_lr_G)
    history["lr_D"].append(current_lr_D)


    # -----------------------------------------------------------------------------
    # Console/log output
    # -----------------------------------------------------------------------------
    epoch_time = time.time() - epoch_start

    summary = (
        f"Epoch {epoch + 1}/{epochs} - {epoch_time:.1f}s | "
        f"Train G total: {train_G_total:.6f} | "
        f"G_GAN: {train_G_GAN:.6f} | "
        f"G_L1_raw: {train_G_L1_raw:.6f} | "
        f"G_L1_w: {train_G_L1_weighted:.6f} | "
        f"G_SSIM_loss: {train_SSIM_loss:.6f} | "
        f"G_LPIPS_loss: {train_LPIPS_loss:.6f} | "
        f"D: {train_D_loss:.6f} | "
        f"D_real: {train_D_real:.6f} | "
        f"D_fake: {train_D_fake:.6f} | "
        f"D_out_real: {train_D_out_real:.4f} | "
        f"D_out_fake: {train_D_out_fake:.4f} | "
        f"Val L1: {val_l1:.6f} | "
        f"Val total: {val_total:.6f} | "
        f"LR_G: {current_lr_G:.2e} | "
        f"LR_D: {current_lr_D:.2e}"
    )

    if val_ssim is not None:
        summary += f" | Val SSIM: {val_ssim:.4f}"

    if val_lpips is not None:
        summary += f" | Val LPIPS: {val_lpips:.4f}"

    if train_grad_ratio is not None:
        summary += f" | GAN/L1 grad: {train_grad_ratio:.4f}"
    if train_grad_ratio_param is not None:
        summary += f" | GAN/L1 grad (param): {train_grad_ratio_param:.4f}"

    summary += (
        f" | probe D(f): {probe_metrics['probe_D_out_fake']:.4f}"
        f" | HF: {probe_metrics['probe_hf_ratio']:.4f}"
        f" | HFcorr: {probe_metrics['probe_hf_corr']:.4f}"
        f" | std: {probe_metrics['probe_std_ratio']:.4f}"
        f" | deg_shuffle: {probe_metrics['probe_deg_shuffle']:.4f}"
        f" [{probe_metrics['probe_deg_shuffle_lo']:.4f},"
        f"{probe_metrics['probe_deg_shuffle_hi']:.4f}]"
    )

    if use_mismatch:
        summary += (
            f" | D_mismatch: {train_D_mismatch:.6f}"
            f" | D_out_mismatch: {train_D_out_mismatch:.4f}"
            f" | same-scene: {mismatch_same_scene_frac:.4f}"
        )

    tqdm.write("\n" + summary)
    logger.info(summary)


    # -----------------------------------------------------------------------------
    # Periodic qualitative samples
    # -----------------------------------------------------------------------------
    if (epoch % 10 == 0) or (epoch == epochs - 1):
        try:
            save_monitor_samples(
                epoch_idx=epoch,
                netG=netG,
                batch_X_cpu=monitor_batch_X,
                batch_y_cpu=monitor_batch_y,
                rgb_idx=rgb_indices,
                out_root=samples_dir,
                include_input=True,
                device=device,
            )

            tqdm.write(
                f"Saved monitor samples for epoch {epoch + 1} → "
                f"{os.path.join(samples_dir, f'epoch_{epoch + 1:03d}')}"
            )

        except Exception as e:
            tqdm.write(f"[WARN] Failed to save monitor samples: {e}")
            logger.warning(f"Failed to save monitor samples: {e}")


    # -----------------------------------------------------------------------------
    # Save best models by multiple validation metrics
    # Training continues to the final epoch; there is NO early stopping.
    # -----------------------------------------------------------------------------
    if save_best_by_val_l1 and val_l1 < best_val_l1:
        best_val_l1 = val_l1
        save_best_models("val_l1")

        # Legacy names for compatibility with your old workflow
        torch.save(netG.state_dict(), os.path.join(save_dir, f"{model_name}_best_netG.pth"))
        torch.save(netD.state_dict(), os.path.join(save_dir, f"{model_name}_best_netD.pth"))

        tqdm.write(f"[BEST] New best Val L1: {best_val_l1:.6f}")
        logger.info(f"[BEST] New best Val L1: {best_val_l1:.6f}")

    if save_best_by_val_ssim and val_ssim is not None and val_ssim > best_val_ssim:
        best_val_ssim = val_ssim
        save_best_models("val_ssim")

        tqdm.write(f"[BEST] New best Val SSIM: {best_val_ssim:.6f}")
        logger.info(f"[BEST] New best Val SSIM: {best_val_ssim:.6f}")

    if save_best_by_val_lpips and val_lpips is not None and val_lpips < best_val_lpips:
        best_val_lpips = val_lpips
        save_best_models("val_lpips")

        tqdm.write(f"[BEST] New best Val LPIPS: {best_val_lpips:.6f}")
        logger.info(f"[BEST] New best Val LPIPS: {best_val_lpips:.6f}")

    if save_best_by_val_total and val_total < best_val_total:
        best_val_total = val_total
        save_best_models("val_total")

        tqdm.write(f"[BEST] New best weighted Val total: {best_val_total:.6f}")
        logger.info(f"[BEST] New best weighted Val total: {best_val_total:.6f}")


    # -----------------------------------------------------------------------------
    # Save checkpoint and plots every epoch
    # -----------------------------------------------------------------------------
    save_checkpoint(epoch)

    if checkpoint_every > 0 and ((epoch + 1) % checkpoint_every == 0):
        snap = f"{model_name}_epoch_{epoch + 1:03d}"
        torch.save(netG.state_dict(), os.path.join(save_dir, f"{snap}_netG.pth"))
        torch.save(netD.state_dict(), os.path.join(save_dir, f"{snap}_netD.pth"))

    save_history_csv()
    save_history_json()
    save_detailed_training_plots()

    if (epoch + 1) >= early_abort_epoch and train_D_out_fake < early_abort_d_out_fake:
        msg = (f"[ABORT] arm {arm}: D_out_fake {train_D_out_fake:.5f} < "
               f"{early_abort_d_out_fake} at epoch {epoch + 1}; this arm has failed.")
        tqdm.write(msg)
        logger.info(msg)
        break


# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
save_history_csv()
save_history_json()
save_detailed_training_plots()

print("\nTraining finished.")
print(f"Saved outputs in: {save_dir}")
print("Important files:")
print(f"  Last checkpoint: {checkpoint_path}")
print(f"  History CSV: {os.path.join(save_dir, 'history.csv')}")
print(f"  History JSON: {os.path.join(save_dir, 'history.json')}")
print(f"  Best Val L1 G: {os.path.join(save_dir, f'{model_name}_best_val_l1_netG.pth')}")
print(f"  Best Val Total G: {os.path.join(save_dir, f'{model_name}_best_val_total_netG.pth')}")
