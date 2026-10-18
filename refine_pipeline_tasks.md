# Refine-pipeline task list

Fixes from [code_review_report.md](code_review_report.md), Part I. Branch: `refine-pipeline`.

Legend: `[x]` implemented · `[ ]` not implemented · commit column filled once each fix is committed.

---

## Done before this branch (on `paper`)

| # | Finding | Change | Status | Commit |
|---|---|---|---|---|
| 0 | S1 | Scene-disjoint, season-stratified split in `prepare_data_sen_12_ms.py`; disjointness assertions; `scenes` + `scene_counts` in `split_metadata.json`; `output_dir` → `dataset_all_seasons_scenesplit` | [x] | `25c6521` |
| 0b | — | Prep hardening: fail on unparsable scene id, fail instead of skipping a malformed pair, non-finite pixels mapped to the clip low bound and counted, dead code removed | [x] | `25c6521` |
| 0c | — | `commit-review-fix` skill | [x] | `f3c3e1c` |
| 0d | S9 | `decay_starting_epoch = warmup_g_only_epochs` | [x] | `404d0a5` |

---

## Must land before the training run

| # | Finding | Change | File | Status | Commit |
|---|---|---|---|---|---|
| 1 | S2 | D initialized at the call site — `init_net(netD, "normal", 0.02)`. D was falling back to PyTorch's default init (≈0.037 std vs the intended 0.020), an asymmetry favouring D from step 0 | `train_…_latest.py` | [x] | `0de758e` |
| 2 | S2 | `apply_spectral_norm` helper + `use_spectral_norm = True`, applied **after** init | `networks.py`, `train_…_latest.py` | [x] | `39e550f`, `0de758e` |
| 3 | S4 | Split GAN criterion: `criterionGAN_D` (0.9/0.0) and `criterionGAN_G` (1.0/0.0). One-sided label smoothing no longer caps G's objective | `train_…_latest.py` | [x] | `0de758e` |
| 4 | S2/S5/S6 | Baseline config: `lr_D = 5e-5`, `use_lr_decay = False`, `allow_resume = False`, `norm_G = "batch"`, `norm_D = "instance"`, `data_dir = dataset_all_seasons_scenesplit`, `run_name = "baseline_scenesplit_sn"`; all recorded in `config.json` | `train_…_latest.py` | [x] | `0de758e` |
| 5 | S2 | Instrumentation: raw `D(real)`/`D(fake)` means, `D(f)` in the tqdm bar, GAN/L1 gradient-norm ratio every 200 batches, two new plots | `train_…_latest.py` | [x] | `0de758e` |
| 6 | S10 | `set_seed(42)` before the data loaders, `cudnn.deterministic = True` / `benchmark = False`, RNG state saved and restored in the checkpoint | `train_…_latest.py` | [x] | `0de758e` |
| 7 | S11e | `check_target_saturation.py` — per-band fraction of pixels at exactly ±1, plus mean/std per band. **Run it on the new dataset before training** | new file | [x] | `bf32542` |

**Nothing in this group is outstanding.** All seven are implemented but **not yet committed**.

---

## Can wait until after the run

Remaining: **#9**, **#16**, **#17**.

| # | Finding | Change | File | Status | Commit |
|---|---|---|---|---|---|
| 8 | S11c | Single G forward pass per step (`fake_B.detach()` for D). ~25% faster, and with `norm_G = "batch"` it also stops BatchNorm running stats being updated twice per optimisation step | `train_…_latest.py` | [x] | `0de758e` |
| 9 | S3 | Guard the degenerate selection: with SSIM/LPIPS off, `val_total` collapses to `100·val_L1`, so four "best" criteria silently become one | `train_…_latest.py` | [ ] | |
| 10 | S11d | `drop_last=True` on the train loader (new `drop_last` parameter in `get_train_val_loaders`) | `myutils.py`, `train_…_latest.py` | [x] | `377a356`, `0de758e` |
| 11 | — | Short pilot run (~20% of train patches, 30–40 epochs). **Dropped** — the collapse showed at epoch 1 last time, so the first few epochs of the full run answer the same question | `train_…_latest.py` | [ ] | |
| 12 | S6 | `RUN_NAME`-based paths and `NORM_G = "batch"` in the evaluation and inference scripts | `evaluate_…_efficient.py`, `infer_generate_optical.py` | [x] | `3b019ed` |
| 13 | S11b | `NPYPairDataset` in `myutils.py` — memory-mapped, opened lazily per DataLoader worker; sampled range check instead of a full-file min/max; `persistent_workers` | `myutils.py` | [x] | `377a356` |
| 14 | S7 | Frozen metric spec: `PSNR_MAX`/`SSIM_DATA_RANGE` constants instead of test-derived ranges, per-band NRMSE and R², batch-size-weighted LPIPS, documented spec block | `evaluate_…_efficient.py` | [x] | `3b019ed` |
| 15 | S8 | `set_lrs_on_resume` accepts either a float or `{"G": …, "D": …}` | `train_…_latest.py` | [x] | |
| 16 | S11a | Qualitative figures stretch prediction and ground truth independently, erasing radiometric bias before the figure is drawn | `infer_generate_optical.py` | [ ] | |
| 17 | S11e | Per-band percentile clipping, computed on the training split only. **Decide after reading the saturation report** | `prepare_data_sen_12_ms.py` | [ ] | |

---

## Order of operations for the run

1. Re-run `prepare_data_sen_12_ms.py` on the server → `dataset_all_seasons_scenesplit` (~120 GB).
   Note the non-finite counts it prints.
2. Run `check_target_saturation.py`. If a band exceeds ~1% of pixels at ±1, fix that first (#17)
   — otherwise the run is testing whether spectral norm can overcome a free cue for D.
3. Start training. Watch the **first 2–3 epochs**:

| Signal | Collapsed | Healthy |
|---|---|---|
| `D(f)` in the tqdm bar | ≈ 0.000 | drifts 0.2 – 0.6 |
| `train_D_loss` | → 0 | oscillates 0.05 – 0.20 |
| `train_G_GAN` | pinned ≈ **1.00** (was 0.810 before the target fix) | moves, ~0.25 – 0.6 |
| `GAN/L1 grad` | < 0.01 | ~0.05 – 0.3 |

**The one criterion that matters: `train_G_GAN` must move.**

4. If it still collapses, escalate one step at a time:
   1. update D every 2nd batch
   2. instance noise on D's input, `N(0, 0.05)`, decayed
   3. `lr_D` → 2.5e-5

---

## Known caveats in the current state

- `data_dir` points at `dataset_all_seasons_scenesplit`, which does not exist until step 1. The run
  will fail fast on a missing `.npy` — intended.
- `epochs` is still 100 (item #11 dropped).
- Startup is now instant: the data is memory-mapped rather than read up front. The first epoch may
  be slower while the OS page cache fills.
- `cudnn.benchmark = False` costs roughly 10–30% throughput. That is the price of comparable runs.
- During warm-up all D metrics read 0. `warmup_g_only_epochs = 0`, so this is currently moot.
- A single seeded run is reproducible, not representative. Any claim like "spectral norm helped"
  still needs ≥3 seeds with mean ± std.
