"""Applies the pre-registered A3-g3 vs L1-only comparison and prints the arm table.

Separate from check_arms.py on purpose: that script applies the criteria as currently
registered, and nothing here is registered until the verdict amendment is written.

    python compare_arms.py
"""

import csv
import json
import os

import numpy as np

ARMS = ["A_base", "A0", "A1", "A3", "A3-g3", "A3-g10", "L1-only"]
LAMBDA = {"A_base": 1, "A0": 1, "A1": 1, "A3": 1,
          "A3-g3": 3, "A3-g10": 10, "L1-only": 0}

TREATMENT = "A3-g3"
CONTROL = "L1-only"
LATE = (18, 20)        # registered window for the val_L1 guard
F_TOLERANCE = 1.10
NDVI_TOLERANCE = 0.05  # guard not yet implemented


def history(arm):
    path = f"trained_arm_{arm}/pix2pix/history.csv"
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as f:
        return [{k: (float(v) if v not in ("", "None", None) else None)
                 for k, v in r.items()} for r in csv.DictReader(f)]


def window_mean(rows, key, window=LATE):
    vals = [r[key] for r in rows
            if window[0] <= int(r["epoch"]) <= window[1] and r.get(key) is not None]
    if len(vals) < window[1] - window[0] + 1:
        return None
    return sum(vals) / len(vals)


def backfill(arm):
    path = f"trained_arm_{arm}/pix2pix/probe_backfill.json"
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def probe_at_20(arm, key):
    """Epoch 20 only: checkpoints are every 5 epochs, so 18 and 19 never existed."""
    b = backfill(arm)
    if b and key in b:
        return b[key], "backfill"
    rows = history(arm)
    if rows:
        v = window_mean(rows, key, (20, 20))
        if v is not None:
            return v, "logged"
    return None, "missing"


def main():
    data = {}
    for a in ARMS:
        rows = history(a)
        if rows is None:
            continue
        data[a] = {
            "bp_corr_10m": probe_at_20(a, "probe_bp_corr_10m")[0],
            "bp_ratio_10m": probe_at_20(a, "probe_bp_ratio_10m")[0],
            "hf_corr_10m": probe_at_20(a, "probe_hf_corr_10m")[0],
            "hf_ratio_10m": probe_at_20(a, "probe_hf_ratio_10m")[0],
            "hf_corr": probe_at_20(a, "probe_hf_corr")[0],
            "hf_ratio": probe_at_20(a, "probe_hf_ratio")[0],
            "val_L1": window_mean(rows, "val_L1"),
            "src": probe_at_20(a, "probe_bp_corr_10m")[1],
        }

    missing = [a for a, d in data.items() if d["bp_corr_10m"] is None]
    if missing:
        print(f"[WARN] no band-pass figure for: {', '.join(missing)}. "
              f"Run backfill_probe.py on the training machine first.\n")

    def fmt(v, w=9, p=4):
        return " " * (w - 3) + "n/a" if v is None else f"{v:{w}.{p}f}"

    print("=== epoch 20 probe, val_L1 is the registered mean of epochs 18-20 ===")
    print(f"{'arm':9} {'lam':>4} {'BPcorr10m':>10} {'BPratio10m':>11} "
          f"{'HFcorr10m':>10} {'HFratio10m':>11} {'val_L1':>9}")
    order = sorted((a for a in data if data[a]["bp_corr_10m"] is not None),
                   key=lambda a: -data[a]["bp_corr_10m"])
    order += [a for a in data if data[a]["bp_corr_10m"] is None]
    for a in order:
        d = data[a]
        print(f"{a:9} {LAMBDA[a]:4d} {fmt(d['bp_corr_10m'],10)} "
              f"{fmt(d['bp_ratio_10m'],11)} {fmt(d['hf_corr_10m'],10)} "
              f"{fmt(d['hf_ratio_10m'],11)} {fmt(d['val_L1'],9,5)}")

    for label, rk, ck in (("band-pass, DoG sigma 2-8, native 10 m",
                           "bp_ratio_10m", "bp_corr_10m"),
                          ("high-pass, sigma 1, native 10 m",
                           "hf_ratio_10m", "hf_corr_10m")):
        pts = [(LAMBDA[a], data[a][rk], data[a][ck]) for a in
               (CONTROL, "A3", TREATMENT, "A3-g10")
               if a in data and data[a][ck] is not None]
        if len(pts) > 2:
            r = np.corrcoef([p[1] for p in pts], [p[2] for p in pts])[0, 1]
            print(f"\n  {label}: ratio vs corr across the lambda ladder, r = {r:+.3f}")
            for lam, ratio, corr in pts:
                print(f"    lambda {lam:2d}   ratio {ratio:.4f}   corr {corr:.4f}")

    if TREATMENT not in data or CONTROL not in data:
        return

    t, c = data[TREATMENT], data[CONTROL]
    print(f"\n=== pre-registered comparison: {TREATMENT} vs {CONTROL} ===")

    if t["bp_corr_10m"] is None or c["bp_corr_10m"] is None:
        print("  PRIMARY  band-pass corr (10 m): not computable yet")
        return

    ok_primary = t["bp_corr_10m"] > c["bp_corr_10m"]
    delta = 100 * (t["bp_corr_10m"] - c["bp_corr_10m"]) / abs(c["bp_corr_10m"])
    print(f"  PRIMARY  band-pass corr (10 m)   {TREATMENT} {t['bp_corr_10m']:.4f} vs "
          f"{CONTROL} {c['bp_corr_10m']:.4f}  ({delta:+.1f}%)  "
          f"-> {'PASS' if ok_primary else 'FAIL'}")

    limit = F_TOLERANCE * c["val_L1"]
    ok_l1 = t["val_L1"] <= limit
    print(f"  GUARD    val_L1 (mean 18-20)      {t['val_L1']:.5f} vs limit "
          f"{limit:.5f}  -> {'PASS' if ok_l1 else 'FAIL'}")
    print(f"  GUARD    NDVI/NDWI MAE within {NDVI_TOLERANCE:.0%}   "
          f"not implemented -> PENDING")

    print("\n  Reported, no pass/fail: KID, PSD ratio 30-100 m, HF ratio, hf_corr")

    if not ok_primary:
        print(f"\n  VERDICT  primary fails -> {CONTROL} preferred. "
              f"Guards and reported metrics cannot reverse it.")
    elif ok_l1:
        print(f"\n  VERDICT  primary passes, val_L1 guard passes, "
              f"NDVI/NDWI guard pending.")
    else:
        print(f"\n  VERDICT  primary passes but the val_L1 guard fails "
              f"-> {CONTROL} preferred.")


if __name__ == "__main__":
    main()
