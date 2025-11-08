# The main training script. What the name tells:
# pix2pix -> the model trained
# lpips and ssim -> both losses are incorporated during training. Set theirs lambda = 0 if you want to cancel them out
# generate -> after every 10 epochs, 10 data samples are generated for the sake of visual assessment
# checkpoints -> I guess it's clear what that means :)
import os, time, logging
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import sys
from pathlib import Path
from myutils import get_train_val_loaders, save_training_plots, save_monitor_samples

# -----------------------------------------------------------------------------
# Device setup
# -----------------------------------------------------------------------------
gpu_id = 0
if torch.cuda.is_available():
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    print(
        f"Using device: {torch.cuda.get_device_name(gpu_id)} (id {torch.cuda.current_device()})"
    )
else:
    device = torch.device("cpu")
    print("Using device: CPU")

torch.backends.cudnn.benchmark = True  # faster on fixed-size tensors

# -----------------------------------------------------------------------------
# Data / Experiment config
# -----------------------------------------------------------------------------
input_channel_select = None  # 0, 1, or None (all)
epochs = 300
batch_size = 16
lr = 1e-4
patience = 20
data_dir = "dataset_winter_all"


model_name = "pix2pix"
save_dir = f"trained-full/{model_name}"

# Warmup and update ratio
warmup_g_only_epochs = 20  # train only G for first N epochs
g_steps_per_iter = 1  # after warmup: how many G updates per batch
d_steps_per_iter = 1  # after warmup: how many D updates per batch

# Optional perceptual losses (set True to enable)
use_ssim = True
use_lpips = True

lambda_L1 = 100.0
lambda_ssim = 50.0
lambda_lpips = 50.0

# If your tensors are in [-1,1] (recommended with Tanh), SSIM data_range = 2.0.
# If they are in [0,1], set this to 1.0 AND keep to_m11() active for LPIPS.
ssim_data_range = 2.0

# LPIPS needs 3 channels. Choose your RGB bands (indices into your 13-band S2).
# Common S2 RGB is B4 (red), B3 (green), B2 (blue). Adjust if your band order differs.
rgb_indices = (3, 2, 1)  # example: (R, G, B)

# -----------------------------------------------------------------------------
# Add pix2pix project to the Python path
# -----------------------------------------------------------------------------
gan_repo_path = Path(__file__).parent / "mymodels/pytorch-CycleGAN-and-pix2pix"
if gan_repo_path.is_dir():
    sys.path.insert(0, str(gan_repo_path))
else:
    print(f"pytorch-CycleGAN-and-pix2pix repository not found at {gan_repo_path}")
    sys.exit(1)

print("Loading data")
train_loader, val_loader = get_train_val_loaders(
    train_data_folder=data_dir,
    val_data_folder=data_dir,
    batch_size=batch_size,
    channel_select=input_channel_select,
)


monitor_batch_X, monitor_batch_y = next(iter(val_loader))
monitor_batch_X = monitor_batch_X[:8].cpu()  # pick first 8 samples for speed/clarity
monitor_batch_y = monitor_batch_y[:8].cpu()
samples_dir = os.path.join(save_dir, "samples")
os.makedirs(samples_dir, exist_ok=True)


# Probe shapes using a single batch to avoid consuming multiple batches
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
from models.networks import define_D, GANLoss

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
netG = Pix2PixWrapper(
    in_channels=channels_in, out_channels=channels_out, img_size=height
).to(device)
netD = define_D(input_nc=channels_in + channels_out, ndf=64, netD="basic").to(device)


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------
criterionGAN = GANLoss(gan_mode="lsgan").to(device)
criterionL1 = torch.nn.L1Loss()

# Optional SSIM / LPIPS setup with safe fallbacks
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
# Optimizers / Scheduler
# -----------------------------------------------------------------------------
optimizer_G = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
scheduler_G = ReduceLROnPlateau(optimizer_G, "min", patience=patience // 2, factor=0.5)
# Keep D LR fixed when controlling D frequency manually

param_count_G = sum(p.numel() for p in netG.parameters() if p.requires_grad)
param_count_D = sum(p.numel() for p in netD.parameters() if p.requires_grad)
print(f"Generator parameter count: {param_count_G:,}")
print(f"Discriminator parameter count: {param_count_D:,}")

# -----------------------------------------------------------------------------
# Logging / Saving
# -----------------------------------------------------------------------------
best_val_loss = float("inf")
wait = 0
history = {"loss": [], "val_loss": [], "lr": [], "D_loss": []}

os.makedirs(save_dir, exist_ok=True)

log_file = os.path.join(save_dir, f"{model_name}_training.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

# -----------------------------------------------------------------------------
# Checkpointing (Save + Resume)
# -----------------------------------------------------------------------------
checkpoint_path = os.path.join(save_dir, "checkpoint_last.pt")
start_epoch = 0  # will be overwritten if resuming

def save_checkpoint(epoch_idx: int):
    ckpt = {
        "epoch": epoch_idx,
        "netG": netG.state_dict(),
        "netD": netD.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict(),
        "scheduler_G": scheduler_G.state_dict(),
        "best_val_loss": best_val_loss,
        "wait": wait,
        "history": history,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "patience": patience,
            "data_dir": data_dir,
            "model_name": model_name,
            "warmup_g_only_epochs": warmup_g_only_epochs,
            "g_steps_per_iter": g_steps_per_iter,
            "d_steps_per_iter": d_steps_per_iter,
            "use_ssim": use_ssim,
            "use_lpips": use_lpips,
            "lambda_L1": lambda_L1,
            "lambda_ssim": lambda_ssim,
            "lambda_lpips": lambda_lpips,
            "ssim_data_range": ssim_data_range,
            "rgb_indices": rgb_indices,
        },
    }
    torch.save(ckpt, checkpoint_path)

def try_resume():
    global start_epoch, best_val_loss, wait, history
    if os.path.isfile(checkpoint_path):
        print(f"[RESUME] Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        netG.load_state_dict(ckpt["netG"])
        netD.load_state_dict(ckpt["netD"])
        optimizer_G.load_state_dict(ckpt["optimizer_G"])
        optimizer_D.load_state_dict(ckpt["optimizer_D"])
        scheduler_G.load_state_dict(ckpt["scheduler_G"])
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        wait = ckpt.get("wait", wait)
        history = ckpt.get("history", history)
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"[RESUME] Resuming at epoch {start_epoch+1} of {epochs}")
    else:
        print("[RESUME] No checkpoint found. Starting fresh.")

try_resume()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def set_requires_grad(net, flag: bool):
    for p in net.parameters():
        p.requires_grad = flag

def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2)

def select_rgb_from_nhwc(x: torch.Tensor, rgb_idx) -> torch.Tensor:
    r, g, b = rgb_idx
    return torch.stack([x[..., r], x[..., g], x[..., b]], dim=-1)  # (N,H,W,3)

def to_m11(x: torch.Tensor) -> torch.Tensor:
    # Map [0,1] -> [-1,1]. If your tensors are already in [-1,1], you can bypass this.
    return x * 2.0 - 1.0

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
for epoch in tqdm(range(start_epoch, epochs), desc="Training", unit="epoch", position=0):
    epoch_start = time.time()
    netG.train()
    netD.train()

    # Regime for this epoch
    g_only = epoch < warmup_g_only_epochs

    train_G_loss = 0.0
    train_D_loss = 0.0
    num_samples_seen = 0

    with tqdm(
        train_loader, desc=f"Epoch {epoch+1}", unit="batch", leave=False, position=1
    ) as batch_pbar:
        for real_A, real_B in batch_pbar:
            real_A, real_B = real_A.to(device), real_B.to(device)
            bsz = real_A.size(0)
            num_samples_seen += bsz

            # ---------------------------
            # (1) Discriminator updates
            # ---------------------------
            if not g_only and d_steps_per_iter > 0:
                set_requires_grad(netD, True)
                for _ in range(d_steps_per_iter):
                    optimizer_D.zero_grad()
                    # Fake for D (no G grad to save mem)
                    with torch.no_grad():
                        fake_B_detached = netG(real_A)
                    pred_real = netD(
                        torch.cat((real_A, real_B), -1).permute(0, 3, 1, 2)
                    )
                    pred_fake = netD(
                        torch.cat((real_A, fake_B_detached), -1).permute(0, 3, 1, 2)
                    )
                    loss_D_real = criterionGAN(pred_real, True)
                    loss_D_fake = criterionGAN(pred_fake, False)
                    loss_D = 0.5 * (loss_D_real + loss_D_fake)
                    loss_D.backward()
                    optimizer_D.step()
                train_D_loss += loss_D.item() * bsz
            else:
                # Warmup: keep D frozen
                set_requires_grad(netD, False)

            # ---------------------------
            # (2) Generator updates
            # ---------------------------
            set_requires_grad(netD, False)
            running_g_loss_this_batch = 0.0

            for _ in range(max(1, g_steps_per_iter)):
                optimizer_G.zero_grad()
                fake_B = netG(real_A)

                # Adversarial loss
                pred_fake = netD(torch.cat((real_A, fake_B), -1).permute(0, 3, 1, 2))
                loss_G_GAN = criterionGAN(pred_fake, True)

                # L1 (use criterionL1 consistently)
                loss_G_L1 = criterionL1(fake_B, real_B) * lambda_L1

                # SSIM (expects NCHW); returns higher-is-better similarity
                loss_SSIM = torch.tensor(0.0, device=device)
                if use_ssim and ssim is not None:
                    ssim_val = ssim(nhwc_to_nchw(fake_B), nhwc_to_nchw(real_B))
                    loss_SSIM = (1.0 - ssim_val) * lambda_ssim

                # LPIPS (needs 3ch, NCHW, in [-1,1])
                loss_LPIPS = torch.tensor(0.0, device=device)
                if use_lpips and lpips_fn is not None:
                    fake_rgb = select_rgb_from_nhwc(fake_B, rgb_indices)
                    real_rgb = select_rgb_from_nhwc(real_B, rgb_indices)

                    # If your tensors are already in [-1,1], you can skip to_m11:
                    fake_rgb_m11 = fake_rgb  # or to_m11(fake_rgb) if your data is [0,1]
                    real_rgb_m11 = real_rgb  # or to_m11(real_rgb) if your data is [0,1]

                    lpips_val = lpips_fn(
                        nhwc_to_nchw(fake_rgb_m11), nhwc_to_nchw(real_rgb_m11)
                    ).mean()
                    loss_LPIPS = lpips_val * lambda_lpips

                loss_G = loss_G_GAN + loss_G_L1 + loss_SSIM + loss_LPIPS
                loss_G.backward()
                optimizer_G.step()

                running_g_loss_this_batch += loss_G.item()

            avg_g_loss_this_batch = running_g_loss_this_batch / max(1, g_steps_per_iter)
            train_G_loss += avg_g_loss_this_batch * bsz

            batch_pbar.set_postfix(
                {
                    "G": f"{avg_g_loss_this_batch:.3f}",
                    "D": f"{(loss_D.item() if not g_only else 0.0):.3f}",
                    "L1": f"{loss_G_L1.item():.3f}",
                    "SSIM": f"{loss_SSIM.item():.3f}" if use_ssim else "off",
                    "LPIPS": f"{loss_LPIPS.item():.3f}" if use_lpips else "off",
                    "lr": f"{optimizer_G.param_groups[0]['lr']:.2e}",
                }
            )

    # Epoch averages (per-sample)
    train_G_loss /= max(1, num_samples_seen)
    train_D_loss /= max(1, num_samples_seen)
    history["loss"].append(train_G_loss)
    history["D_loss"].append(train_D_loss)
    current_lr = optimizer_G.param_groups[0]["lr"]

    # -----------------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------------
    val_l1_sum = 0.0
    val_ssim_sum = 0.0
    val_lpips_sum = 0.0
    n_val = 0

    netG.eval()
    with tqdm(
        val_loader, desc="Validating", unit="batch", leave=False, position=2
    ) as val_pbar:
        with torch.no_grad():
            for batch_X, batch_y in val_pbar:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                bsz = batch_X.size(0)
                n_val += bsz

                outputs = netG(batch_X)
                l1 = criterionL1(outputs, batch_y).item()
                val_l1_sum += l1 * bsz

                if use_ssim and ssim is not None:
                    ssim_val = ssim(nhwc_to_nchw(outputs), nhwc_to_nchw(batch_y)).item()
                    val_ssim_sum += ssim_val * bsz

                if use_lpips and lpips_fn is not None:
                    out_rgb = select_rgb_from_nhwc(outputs, rgb_indices)
                    tgt_rgb = select_rgb_from_nhwc(batch_y, rgb_indices)
                    # If already [-1,1], skip to_m11:
                    out_rgb_m11 = out_rgb  # or to_m11(out_rgb) if [0,1]
                    tgt_rgb_m11 = tgt_rgb  # or to_m11(tgt_rgb) if [0,1]
                    lp = (
                        lpips_fn(nhwc_to_nchw(out_rgb_m11), nhwc_to_nchw(tgt_rgb_m11))
                        .mean()
                        .item()
                    )
                    val_lpips_sum += lp * bsz

                val_pbar.set_postfix({"val_L1": f"{l1:.6f}"})

    val_loss = val_l1_sum / max(1, n_val)  # keep scheduler metric as L1
    history["val_loss"].append(val_loss)
    history["lr"].append(current_lr)

    scheduler_G.step(val_loss)

    # Print/log epoch summary
    val_ssim = (val_ssim_sum / n_val) if (use_ssim and n_val > 0) else None
    val_lpips = (val_lpips_sum / n_val) if (use_lpips and n_val > 0) else None

    tqdm.write(f"\nEpoch {epoch+1}/{epochs} - {time.time()-epoch_start:.1f}s")
    tqdm.write(
        f"Train G_Loss: {train_G_loss:.6f} | Train D_Loss: {train_D_loss:.6f} | Val L1: {val_loss:.6f}"
        + (f" | Val SSIM: {val_ssim:.4f}" if val_ssim is not None else "")
        + (f" | Val LPIPS: {val_lpips:.4f}" if val_lpips is not None else "")
    )
    logger.info(
        f"Train G_Loss: {train_G_loss:.6f} | Train D_Loss: {train_D_loss:.6f} | Val L1: {val_loss:.6f}"
        + (f" | Val SSIM: {val_ssim:.4f}" if val_ssim is not None else "")
        + (f" | Val LPIPS: {val_lpips:.4f}" if val_lpips is not None else "")
    )
    tqdm.write(f"Learning Rate (G): {current_lr:.2e}")
    logger.info(f"Learning Rate (G): {current_lr:.2e}")
    # ---------------- NEW: periodic qualitative samples ---------------------------
    if (epoch % 10 == 0) or (epoch == epochs - 1):
        try:
            save_monitor_samples(
                epoch_idx=epoch,
                netG=netG,
                batch_X_cpu=monitor_batch_X,
                batch_y_cpu=monitor_batch_y,
                rgb_idx=rgb_indices,
                out_root=samples_dir,
                include_input=True,  # set False if you don't want SAR pseudo-RGB shown
                device=device
            )
            tqdm.write(
                f"Saved monitor samples for epoch {epoch+1} → {os.path.join(samples_dir, f'epoch_{epoch+1:03d}')}"
            )
        except Exception as e:
            tqdm.write(f"[WARN] Failed to save monitor samples: {e}")
            logger.warning(f"Failed to save monitor samples: {e}")

    # Save best (by Val L1)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
        torch.save(
            netG.state_dict(), os.path.join(save_dir, f"{model_name}_best_netG.pth")
        )
        torch.save(
            netD.state_dict(), os.path.join(save_dir, f"{model_name}_best_netD.pth")
        )
    else:
        wait += 1
        # If you want early stopping, flip this:
        if False and wait >= patience:
            tqdm.write(f"\nEarly stopping triggered at epoch {epoch+1}")
            logger.info(f"\nEarly stopping triggered at epoch {epoch+1}")
            # Save a last checkpoint before exiting early
            save_checkpoint(epoch)
            break

    # -------- Save/overwrite "last" checkpoint every epoch (for resume) ----------
    save_checkpoint(epoch)

save_training_plots(history, save_dir, model_name)
