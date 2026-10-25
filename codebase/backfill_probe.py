# Recompute probe metrics for arms that finished before a metric existed.
#
# Rebuilds each arm's networks exactly as its training run did, loads an epoch
# snapshot, and reruns the probe. No training, no weight updates.
#
#   python backfill_probe.py

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).parent
gan_repo_path = THIS_DIR / "mymodels/pytorch-CycleGAN-and-pix2pix"
if gan_repo_path.is_dir():
    sys.path.insert(0, str(gan_repo_path))

from mymodels.cp2pwrapper import Pix2PixWrapper
from models.networks import define_D, init_net, apply_spectral_norm
from arm_probe import load_probe, run_probe

DATA_DIR = "dataset_all_seasons_scenesplit"
MODEL_NAME = "pix2pix"
NORM_G = "batch"
NORM_D = "instance"
PROBE_N = 128
BATCH = 16
GPU_ID = 0

# arm -> whether its D had the norm modules stripped, as in the training config
ARMS = {"A_base": False, "A0": False}
EPOCHS = (5, 10, 15, 20)


def build(c_in, c_out, img_size, strip_norm, device):
    netG = Pix2PixWrapper(in_channels=c_in, out_channels=c_out,
                          img_size=img_size, norm=NORM_G)

    netD = define_D(input_nc=c_in + c_out, ndf=64, netD="basic", norm=NORM_D)
    if strip_norm:
        for module in netD.modules():
            for child_name, child in list(module.named_children()):
                if isinstance(child, torch.nn.InstanceNorm2d):
                    setattr(module, child_name, torch.nn.Identity())

    netD = init_net(netD, init_type="normal", init_gain=0.02)
    netD = apply_spectral_norm(netD)
    return netG.to(device), netD.to(device)


def main():
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    probe = load_probe(DATA_DIR, "val", n=PROBE_N)
    c_in = probe["X"].shape[-1]
    c_out = probe["Y"].shape[-1]
    img_size = probe["X"].shape[1]

    for arm, strip_norm in ARMS.items():
        save_dir = f"trained_arm_{arm}/{MODEL_NAME}"
        if not os.path.isdir(save_dir):
            print(f"{arm}: not found, skipping")
            continue

        netG, netD = build(c_in, c_out, img_size, strip_norm, device)
        scores_dir = os.path.join(save_dir, "probe_scores")
        os.makedirs(scores_dir, exist_ok=True)
        found = {}

        for ep in EPOCHS:
            g = os.path.join(save_dir, f"{MODEL_NAME}_epoch_{ep:03d}_netG.pth")
            d = os.path.join(save_dir, f"{MODEL_NAME}_epoch_{ep:03d}_netD.pth")
            if not (os.path.isfile(g) and os.path.isfile(d)):
                continue

            netG.load_state_dict(torch.load(g, map_location=device))
            netD.load_state_dict(torch.load(d, map_location=device))
            metrics, scores = run_probe(netG, netD, probe, device, batch=BATCH)

            # Also gives criterion C the per-patch scores these runs never saved.
            np.savez(os.path.join(scores_dir, f"epoch_{ep:03d}.npz"), **scores)
            found[ep] = metrics
            print(f"{arm} epoch {ep:3d}:  hf_ratio {metrics['probe_hf_ratio']:.4f}  "
                  f"hf_corr {metrics['probe_hf_corr']:.4f}  "
                  f"deg_shuffle {metrics['probe_deg_shuffle']:+.4f}")

        if not found:
            print(f"{arm}: no epoch snapshots found")
            continue

        last = max(found)
        with open(os.path.join(save_dir, "probe_backfill.json"), "w") as f:
            json.dump({"epochs": sorted(found),
                       "reference_epoch": last,
                       "probe_hf_corr": found[last]["probe_hf_corr"],
                       "probe_hf_ratio": found[last]["probe_hf_ratio"],
                       "note": "recomputed from epoch snapshots; single epoch, "
                               "not the 18-20 window used elsewhere"},
                      f, indent=2)
        print(f"{arm}: wrote probe_backfill.json (reference epoch {last})\n")


if __name__ == "__main__":
    main()
