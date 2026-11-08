# Applies success_criteria.md mechanically to the arm histories. Reads only history.csv.
#
#   python check_arms.py

import csv
import glob
import json
import os

import numpy as np

ARMS = ["A_base", "A0", "A1", "A2", "A3", "A3-g3", "A3-g10", "L1-only"]
REFERENCE_RUN = "trained_baseline_scenesplit_sn/pix2pix/history.csv"

WINDOW = (11, 20)        # S1, S2
LATE = (18, 20)          # S3 numerator, T, F, C
EARLY = (9, 11)          # S3 denominator
FIRST = (1, 3)           # baseline for the D(real) growth trigger
S1_D_OUT_FAKE = 0.10
S1_D_LOSS = 0.02
S2_G_GAN = 0.90
S3_RATIO = 0.50
T_OVER_BASE = 0.05
F_TOLERANCE = 1.10
C_DEG = 0.10


def load(path):
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as f:
        rows = []
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if v not in ("", "None", None) else None)
                         for k, v in r.items()})
    return rows


def at(rows, epoch, key):
    for r in rows:
        if int(r["epoch"]) == epoch:
            return r.get(key)
    return None


def window_mean(rows, key, window=WINDOW):
    vals = [r[key] for r in rows
            if window[0] <= int(r["epoch"]) <= window[1] and r.get(key) is not None]
    # A short history means the arm aborted; that is a failure, not a smaller window.
    if len(vals) < window[1] - window[0] + 1:
        return None
    return sum(vals) / len(vals)


def score_files(armname, window=LATE):
    d = f"trained_arm_{armname}/pix2pix/probe_scores"
    return [p for p in sorted(glob.glob(os.path.join(d, "epoch_*.npz")))
            if window[0] <= int(os.path.basename(p)[6:9]) <= window[1]]


def hf_corr(armname, rows):
    """Windowed mean, or the back-filled single epoch for runs that predate the metric."""
    v = window_mean(rows, "probe_hf_corr", LATE)
    if v is not None:
        return v, ""

    path = f"trained_arm_{armname}/pix2pix/probe_backfill.json"
    if os.path.isfile(path):
        with open(path) as f:
            b = json.load(f)
        return b["probe_hf_corr"], f" (back-filled, epoch {b['reference_epoch']})"
    return None, " (not measured)"


def pooled_deg_shuffle(files, n_boot=1000, seed=12345, alpha=0.05):
    """deg_shuffle from per-patch scores pooled across epochs, then bootstrapped.

    Pooling before the bootstrap, rather than averaging per-epoch point estimates,
    keeps the interval honest about how much data it rests on.
    """
    z = [np.load(p) for p in files]
    d_real = np.concatenate([a["d_real"] for a in z])
    d_fake = np.concatenate([a["d_fake"] for a in z])
    d_shuf = np.concatenate([a["d_shuf"] for a in z])

    rng = np.random.default_rng(seed)
    n = len(d_real)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, n, n)
        lo, hi = d_fake[s].mean(), d_real[s].mean()
        draws[i] = 1.0 - (d_shuf[s].mean() - lo) / (hi - lo)

    deg = 1.0 - (d_shuf.mean() - d_fake.mean()) / (d_real.mean() - d_fake.mean())
    return float(deg), float(np.percentile(draws, 100 * alpha / 2)), \
        float(np.percentile(draws, 100 * (1 - alpha / 2)))


def verdict(ok):
    if ok is None:
        return " REF"
    return "PASS" if ok else "FAIL"


def evaluate(armname, rows, base, is_reference=False):
    out = {}

    d_fake = window_mean(rows, "train_D_out_fake")
    d_loss = window_mean(rows, "train_D_loss")
    sat = "  <- saturation, retry lr_D 2.5e-5" if (d_fake is not None and d_fake < 0.10) else ""
    out["S1"] = (d_fake is not None and d_loss is not None
                 and d_fake >= S1_D_OUT_FAKE and d_loss >= S1_D_LOSS,
                 f"D_out_fake {d_fake if d_fake is None else round(d_fake, 4)}, "
                 f"D_loss {d_loss if d_loss is None else round(d_loss, 4)}{sat}")

    g_gan = window_mean(rows, "train_G_GAN")
    out["S2"] = (g_gan is not None and g_gan <= S2_G_GAN,
                 f"G_GAN {g_gan if g_gan is None else round(g_gan, 4)}")

    late = window_mean(rows, "train_D_out_fake", LATE)
    early = window_mean(rows, "train_D_out_fake", EARLY)
    ratio = late / early if (late and early) else None
    out["S3"] = (ratio is not None and ratio >= S3_RATIO,
                 f"mean(18-20)/mean(9-11) {ratio if ratio is None else round(ratio, 3)}")

    hf = window_mean(rows, "probe_hf_ratio", LATE)
    hf_base = window_mean(base, "probe_hf_ratio", LATE) if base else None
    l1 = window_mean(rows, "val_L1", LATE)
    l1_base = window_mean(base, "val_L1", LATE) if base else None

    # T and F are defined relative to A_base, so the reference arm only supplies them.
    if is_reference:
        out["T"] = (None, f"HF {hf if hf is None else round(hf, 4)} (reference value)")
        out["F"] = (None, f"val_L1 {l1 if l1 is None else round(l1, 5)} (reference value)")
    else:
        out["T"] = (hf is not None and hf_base is not None and hf >= hf_base + T_OVER_BASE,
                    f"HF {hf if hf is None else round(hf, 4)}"
                    + (f", needs >= {round(hf_base + T_OVER_BASE, 4)}" if hf_base else ""))
        out["F"] = (l1 is not None and (l1_base is None or l1 <= F_TOLERANCE * l1_base),
                    f"val_L1 {l1 if l1 is None else round(l1, 5)}"
                    + (f", limit {round(F_TOLERANCE * l1_base, 5)}" if l1_base else ""))

    files = score_files(armname)
    if files:
        deg, lo, _ = pooled_deg_shuffle(files)
        note = f"deg_shuffle {round(deg, 4)}, CI low {round(lo, 4)} (pooled {len(files)} ep)"
        out["_c_estimator"] = "pooled"
    else:
        # Runs from before per-patch scores were persisted fall back to the epoch means.
        deg = window_mean(rows, "probe_deg_shuffle", LATE)
        lo = window_mean(rows, "probe_deg_shuffle_lo", LATE)
        note = (f"deg_shuffle {deg if deg is None else round(deg, 4)}, "
                f"CI low {lo if lo is None else round(lo, 4)} "
                f"(per-epoch mean, NO per-patch scores)")
        out["_c_estimator"] = "per-epoch"
    out["C"] = (deg is not None and lo is not None and deg >= C_DEG and lo > 0, note)

    corr, corr_note = hf_corr(armname, rows)
    grad = window_mean(rows, "train_gan_l1_grad_ratio_param", LATE)

    # U was withdrawn on 2026-11-08: its hf_corr leg does not discriminate. Both numbers
    # are still reported because the lambda_GAN grid manipulates the gradient one.
    out["~"] = (None, f"hf_corr {corr if corr is None else round(corr, 4)}{corr_note}, "
                      f"grad {grad if grad is None else round(grad, 4)}  [reported]")

    # Fallback triggers from success_criteria.md.
    early_r = window_mean(rows, "train_D_out_real", FIRST)
    late_r = window_mean(rows, "train_D_out_real", LATE)
    if early_r and late_r:
        growth = 100 * (late_r - early_r) / early_r
        flag = "  <- too weak, retry lr_D 1e-4" if growth < 10 else ""
        out[" "] = (None, f"D(real) growth {growth:+.1f}%{flag}  [reported]")

    return out


def main():
    base = load(f"trained_arm_A_base/pix2pix/history.csv")
    if base is None:
        print(f"[note] arm_A_base not found; T and F fall back to absolute thresholds\n")

    estimators = {}

    for armname in ARMS:
        path = f"trained_arm_{armname}/pix2pix/history.csv"
        rows = load(path)
        if rows is None:
            print(f"{armname:>7}: not run")
            continue

        n = len(rows)
        tag = "" if n >= WINDOW[1] else f"  (ABORTED after epoch {n})"
        print(f"\n{armname}  [{n} epochs]{tag}")
        res = evaluate(armname, rows, base, armname == "A_base")
        estimators[armname] = res.pop("_c_estimator", None)
        for cid, (ok, detail) in res.items():
            print(f"   {cid:>2}  {verdict(ok):>4}   {detail}")

    used = {v for v in estimators.values() if v}
    if len(used) > 1:
        odd = [k for k, v in estimators.items() if v == "per-epoch"]
        print(f"\n[WARN] C is not estimated the same way in every arm: {', '.join(odd)} "
              f"lack per-patch scores. Rerun them, or read C as indicative there.")

    if base is not None:
        ref = load(REFERENCE_RUN)
        if ref is not None:
            print("\nA_base reproduction vs the reference run (epoch 1):")
            for k in ("train_G_L1_raw", "train_D_out_fake", "train_G_GAN"):
                a, b = at(base, 1, k), at(ref, 1, k)
                if a is not None and b is not None:
                    d = abs(a - b)
                    print(f"   {k:>20}  {a:.6f} vs {b:.6f}   diff {d:.2e}   "
                          f"{verdict(d < 1e-3)}")


if __name__ == "__main__":
    main()
