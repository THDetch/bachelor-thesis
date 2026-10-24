# Applies success_criteria.md mechanically to the arm histories. Reads only history.csv.
#
#   python check_arms.py

import csv
import os

ARMS = ["A_base", "A0", "A1", "A2", "A3"]
REFERENCE_RUN = "trained_baseline_scenesplit_sn/pix2pix/history.csv"

WINDOW = (11, 20)
S1_D_OUT_FAKE = 0.10
S1_D_LOSS = 0.02
S2_G_GAN = 0.90
S3_RATIO = 0.50
T_ABS = 0.25
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


def window_mean(rows, key):
    vals = [r[key] for r in rows
            if WINDOW[0] <= int(r["epoch"]) <= WINDOW[1] and r.get(key) is not None]
    # A short history means the arm aborted; that is a failure, not a smaller window.
    if len(vals) < WINDOW[1] - WINDOW[0] + 1:
        return None
    return sum(vals) / len(vals)


def verdict(ok):
    return "PASS" if ok else "FAIL"


def evaluate(rows, base):
    out = {}

    d_fake = window_mean(rows, "train_D_out_fake")
    d_loss = window_mean(rows, "train_D_loss")
    out["S1"] = (d_fake is not None and d_loss is not None
                 and d_fake >= S1_D_OUT_FAKE and d_loss >= S1_D_LOSS,
                 f"D_out_fake {d_fake if d_fake is None else round(d_fake, 4)}, "
                 f"D_loss {d_loss if d_loss is None else round(d_loss, 4)}")

    g_gan = window_mean(rows, "train_G_GAN")
    out["S2"] = (g_gan is not None and g_gan <= S2_G_GAN,
                 f"G_GAN {g_gan if g_gan is None else round(g_gan, 4)}")

    a, b = at(rows, 10, "train_D_out_fake"), at(rows, 20, "train_D_out_fake")
    ratio = b / a if (a and b) else None
    out["S3"] = (ratio is not None and ratio >= S3_RATIO,
                 f"ep20/ep10 {ratio if ratio is None else round(ratio, 3)}")

    hf = at(rows, 20, "probe_hf_ratio")
    hf_base = at(base, 20, "probe_hf_ratio") if base else None
    ok_t = hf is not None and hf >= T_ABS and (
        hf_base is None or hf >= hf_base + T_OVER_BASE)
    out["T"] = (ok_t, f"HF {hf if hf is None else round(hf, 4)}"
                      + (f", base {round(hf_base, 4)}" if hf_base else ""))

    l1 = at(rows, 20, "val_L1")
    l1_base = at(base, 20, "val_L1") if base else None
    out["F"] = (l1 is not None and (l1_base is None or l1 <= F_TOLERANCE * l1_base),
                f"val_L1 {l1 if l1 is None else round(l1, 5)}"
                + (f", limit {round(F_TOLERANCE * l1_base, 5)}" if l1_base else ""))

    deg = at(rows, 20, "probe_deg_shuffle")
    lo = at(rows, 20, "probe_deg_shuffle_lo")
    out["C"] = (deg is not None and lo is not None and deg >= C_DEG and lo > 0,
                f"deg_shuffle {deg if deg is None else round(deg, 4)}"
                + (f", CI low {round(lo, 4)}" if lo is not None else ""))

    return out


def main():
    base = load(f"trained_arm_A_base/pix2pix/history.csv")
    if base is None:
        print(f"[note] arm_A_base not found; T and F fall back to absolute thresholds\n")

    for armname in ARMS:
        path = f"trained_arm_{armname}/pix2pix/history.csv"
        rows = load(path)
        if rows is None:
            print(f"{armname:>7}: not run")
            continue

        n = len(rows)
        tag = "" if n >= WINDOW[1] else f"  (ABORTED after epoch {n})"
        print(f"\n{armname}  [{n} epochs]{tag}")
        for cid, (ok, detail) in evaluate(rows, base).items():
            print(f"   {cid:>2}  {verdict(ok):>4}   {detail}")

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
