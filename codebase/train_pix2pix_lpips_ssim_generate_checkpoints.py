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
epochs = 150
batch_size = 16
lr = 1e-4
patience = 20
# data_dir = "dataset_winter_all"
data_dir = "dataset_all_seasons_metadata"


model_name = "pix2pix"
# save_dir = f"trained-full/{model_name}"
save_dir = f"trained_all_seasons_metadata/{model_name}"

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
scheduler_G = ReduceLROnPlateau(optimizer_G, "min", patience=patience, factor=0.5)
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

logger.info(f"Num of Epochs: {epochs}")
logger.info(f"Data Dir: {data_dir}")
logger.info(f"Patience: {patience}")
logger.info(f"Batch Size: {batch_size}")
logger.info(f"Learning Rate: {lr}")
logger.info(f"Lambda L1: {lambda_L1}")
logger.info(f"Lambda SSIM: {lambda_ssim}")
logger.info(f"Lambda LPIPS: {lambda_lpips}")


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
# Manually override ONLY Generator LR after resume
# -----------------------------------------------------------------------------
resume_with_halved_g_lr = True
resume_g_lr = 5e-5  # half of 1e-4

if resume_with_halved_g_lr and start_epoch > 0:
    for param_group in optimizer_G.param_groups:
        param_group["lr"] = resume_g_lr

    print(f"[LR OVERRIDE] Resumed from checkpoint. Set ONLY Generator LR to {resume_g_lr:.2e}")
    logger.info(f"[LR OVERRIDE] Resumed from checkpoint. Set ONLY Generator LR to {resume_g_lr:.2e}")

    print(f"[LR INFO] Discriminator LR kept from checkpoint: {optimizer_D.param_groups[0]['lr']:.2e}")
    logger.info(f"[LR INFO] Discriminator LR kept from checkpoint: {optimizer_D.param_groups[0]['lr']:.2e}")

    # Reset Generator scheduler so it uses the new patience=20 from this resumed run
    scheduler_G = ReduceLROnPlateau(
        optimizer_G,
        mode="min",
        patience=patience,
        factor=0.5
    )

    print(f"[SCHEDULER RESET] Generator LR scheduler reset with patience={patience}, factor=0.5")
    logger.info(f"[SCHEDULER RESET] Generator LR scheduler reset with patience={patience}, factor=0.5")
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

            # Keep this defined even during warmup
            loss_D = torch.tensor(0.0, device=device)

            # ---------------------------
            # (1) Discriminator updates
            # ---------------------------
            if not g_only and d_steps_per_iter > 0:
                set_requires_grad(netD, True)

                for _ in range(d_steps_per_iter):
                    optimizer_D.zero_grad()

                    # Fake for D only; no G gradient here
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
                # Warmup: keep D frozen and do not update it
                set_requires_grad(netD, False)

            # ---------------------------
            # (2) Generator updates
            # ---------------------------
            set_requires_grad(netD, False)
            running_g_loss_this_batch = 0.0

            for _ in range(max(1, g_steps_per_iter)):
                optimizer_G.zero_grad()

                fake_B = netG(real_A)

                # ---------------------------------------------------------
                # Adversarial loss
                # ---------------------------------------------------------
                # During G-only warmup, do NOT use GAN loss.
                # Otherwise, G would optimize against a random frozen D.
                if g_only:
                    loss_G_GAN = torch.tensor(0.0, device=device)
                else:
                    pred_fake = netD(
                        torch.cat((real_A, fake_B), -1).permute(0, 3, 1, 2)
                    )
                    loss_G_GAN = criterionGAN(pred_fake, True)

                # L1 reconstruction loss
                loss_G_L1 = criterionL1(fake_B, real_B) * lambda_L1

                # SSIM loss
                loss_SSIM = torch.tensor(0.0, device=device)
                if use_ssim and ssim is not None:
                    ssim_val = ssim(nhwc_to_nchw(fake_B), nhwc_to_nchw(real_B))
                    loss_SSIM = (1.0 - ssim_val) * lambda_ssim

                # LPIPS perceptual loss
                loss_LPIPS = torch.tensor(0.0, device=device)
                if use_lpips and lpips_fn is not None:
                    fake_rgb = select_rgb_from_nhwc(fake_B, rgb_indices)
                    real_rgb = select_rgb_from_nhwc(real_B, rgb_indices)

                    # Your tensors are already in [-1, 1], so do not apply to_m11()
                    fake_rgb_m11 = fake_rgb
                    real_rgb_m11 = real_rgb

                    lpips_val = lpips_fn(
                        nhwc_to_nchw(fake_rgb_m11),
                        nhwc_to_nchw(real_rgb_m11)
                    ).mean()

                    loss_LPIPS = lpips_val * lambda_lpips

                # Total generator loss
                loss_G = loss_G_GAN + loss_G_L1 + loss_SSIM + loss_LPIPS

                loss_G.backward()
                optimizer_G.step()

                running_g_loss_this_batch += loss_G.item()

            avg_g_loss_this_batch = running_g_loss_this_batch / max(1, g_steps_per_iter)
            train_G_loss += avg_g_loss_this_batch * bsz

            batch_pbar.set_postfix(
                {
                    "G": f"{avg_g_loss_this_batch:.3f}",
                    "D": f"{loss_D.item():.3f}",
                    "GAN": f"{loss_G_GAN.item():.3f}",
                    "L1": f"{loss_G_L1.item():.3f}",
                    "SSIM": f"{loss_SSIM.item():.3f}" if use_ssim else "off",
                    "LPIPS": f"{loss_LPIPS.item():.3f}" if use_lpips else "off",
                    "lr": f"{optimizer_G.param_groups[0]['lr']:.2e}",
                }
            )

    # Epoch averages
    train_G_loss /= max(1, num_samples_seen)
    train_D_loss /= max(1, num_samples_seen)

    history["loss"].append(train_G_loss)
    history["D_loss"].append(train_D_loss)


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
                    ssim_val = ssim(
                        nhwc_to_nchw(outputs),
                        nhwc_to_nchw(batch_y)
                    ).item()
                    val_ssim_sum += ssim_val * bsz

                if use_lpips and lpips_fn is not None:
                    out_rgb = select_rgb_from_nhwc(outputs, rgb_indices)
                    tgt_rgb = select_rgb_from_nhwc(batch_y, rgb_indices)

                    # Already in [-1, 1]
                    out_rgb_m11 = out_rgb
                    tgt_rgb_m11 = tgt_rgb

                    lp = (
                        lpips_fn(
                            nhwc_to_nchw(out_rgb_m11),
                            nhwc_to_nchw(tgt_rgb_m11)
                        )
                        .mean()
                        .item()
                    )

                    val_lpips_sum += lp * bsz

                val_pbar.set_postfix({"val_L1": f"{l1:.6f}"})

    val_loss = val_l1_sum / max(1, n_val)

    scheduler_G.step(val_loss)

    current_lr = optimizer_G.param_groups[0]["lr"]
    history["val_loss"].append(val_loss)
    history["lr"].append(current_lr)

    # Print/log epoch summary
    val_ssim = (val_ssim_sum / n_val) if (use_ssim and n_val > 0) else None
    val_lpips = (val_lpips_sum / n_val) if (use_lpips and n_val > 0) else None

    tqdm.write(f"\nEpoch {epoch+1}/{epochs} - {time.time()-epoch_start:.1f}s")

    tqdm.write(
        f"Train G_Loss: {train_G_loss:.6f} | "
        f"Train D_Loss: {train_D_loss:.6f} | "
        f"Val L1: {val_loss:.6f}"
        + (f" | Val SSIM: {val_ssim:.4f}" if val_ssim is not None else "")
        + (f" | Val LPIPS: {val_lpips:.4f}" if val_lpips is not None else "")
    )

    logger.info(
        f"Train G_Loss: {train_G_loss:.6f} | "
        f"Train D_Loss: {train_D_loss:.6f} | "
        f"Val L1: {val_loss:.6f}"
        + (f" | Val SSIM: {val_ssim:.4f}" if val_ssim is not None else "")
        + (f" | Val LPIPS: {val_lpips:.4f}" if val_lpips is not None else "")
    )
    logger.info(f">>> Current Learning rate: {current_lr}")
    tqdm.write(f"Learning Rate (G): {current_lr:.2e}")
    logger.info(f"Learning Rate (G): {current_lr:.2e}")

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
                f"Saved monitor samples for epoch {epoch+1} → "
                f"{os.path.join(samples_dir, f'epoch_{epoch+1:03d}')}"
            )

        except Exception as e:
            tqdm.write(f"[WARN] Failed to save monitor samples: {e}")
            logger.warning(f"Failed to save monitor samples: {e}")

    # -----------------------------------------------------------------------------
    # Save best model by Val L1
    # -----------------------------------------------------------------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0

        torch.save(
            netG.state_dict(),
            os.path.join(save_dir, f"{model_name}_best_netG.pth")
        )

        torch.save(
            netD.state_dict(),
            os.path.join(save_dir, f"{model_name}_best_netD.pth")
        )

    else:
        wait += 1

        # If you want early stopping, change False to True
        if False and wait >= patience:
            tqdm.write(f"\nEarly stopping triggered at epoch {epoch+1}")
            logger.info(f"\nEarly stopping triggered at epoch {epoch+1}")

            save_checkpoint(epoch)
            break

    # Save/overwrite last checkpoint every epoch for resume
    save_checkpoint(epoch)
    
save_training_plots(history, save_dir, model_name)
