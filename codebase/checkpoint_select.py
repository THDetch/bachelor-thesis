"""Checkpoint selection shared by the evaluation and inference scripts.

Training writes several generators per run and they are not interchangeable:
`*_best_netG.pth` is whichever epoch minimised validation L1, while the last epoch is
the one success_criteria.md registers as the result of a final run. Selecting on
validation L1 selects the blurriest epoch, so a texture metric read off the "best"
file can disagree with the same metric read off the last one. Both are reported and
labelled rather than one being silently assumed.
"""

import os
import re

import torch

CHECKPOINT_LAST = "checkpoint_last.pt"


def _load(path, device):
    # weights_only=True rejects checkpoint_last.pt, which stores optimizer and RNG state.
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def resolve_generator(model_dir, which, model_name="pix2pix", device="cpu"):
    """Return (label, source_path, state_dict) for which in {"best", "last"}.

    Returns None when that checkpoint does not exist, so a run that predates one of
    them still evaluates on the other instead of aborting.
    """
    if which == "best":
        p = os.path.join(model_dir, f"{model_name}_best_netG.pth")
        return ("best", p, _load(p, device)) if os.path.isfile(p) else None

    if which != "last":
        raise ValueError(f"unknown checkpoint selector {which!r}")

    # Written every epoch, so it is the most reliable record of the final weights.
    p = os.path.join(model_dir, CHECKPOINT_LAST)
    if os.path.isfile(p):
        ckpt = _load(p, device)
        if isinstance(ckpt, dict) and "netG" in ckpt:
            ep = ckpt.get("epoch", -1) + 1  # stored 0-based
            return (f"last_ep{ep:03d}" if ep > 0 else "last", p, ckpt["netG"])

    snaps = []
    for f in os.listdir(model_dir) if os.path.isdir(model_dir) else []:
        m = re.fullmatch(rf"{re.escape(model_name)}_epoch_(\d+)_netG\.pth", f)
        if m:
            snaps.append((int(m.group(1)), f))
    if snaps:
        ep, f = max(snaps)
        p = os.path.join(model_dir, f)
        return f"last_ep{ep:03d}", p, _load(p, device)

    return None
