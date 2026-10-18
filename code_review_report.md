# Code Audit Report — response to `instruction_claude.md`

Scope: audit only. **No code was changed.** All citations are to the `paper` branch at `f8757f8`.

**Re-scoped on request:** the paper's implementation is superseded. This report is now organised as:

- **Part I — the current pipeline.** Findings and fixes that apply to the code you are actually
  iterating on (`*_latest.py` + `dataset_all_seasons`). *Read this part.*
- **Part II — manuscript errata.** The paper-era findings, retained because the manuscript still
  has to be corrected regardless of where the code has moved. Historical versions are cited as *(orig)*.
- **Part III — addendum** answering the reviewer's N0–N10, re-mapped onto the current code.

---
---

# PART I — Audit of the current implementation

## I.0 What the current pipeline is

| Stage | File | Target |
|---|---|---|
| Preprocessing | [codebase/prepare_data_sen_12_ms.py](codebase/prepare_data_sen_12_ms.py) | `dataset_all_seasons` — 4 seasons, 32,000 patches, 25,600 / 3,200 / 3,200, season-stratified, split CSVs written |
| Training | [codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py) | `trained_all_seasons_debugged_loss_original_label_smoothing/pix2pix` |
| Evaluation | [codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py) | `MODEL_DIR = trained_all_seasons_debugged_loss_balance_label_smoothing` |
| Figures | [codebase/infer_generate_optical.py](codebase/infer_generate_optical.py) | same `…_balance_label_smoothing` dir |

Resolved config of the committed run (`config.json` + `history.csv`, 100 epochs):
`norm="instance"` (G **and** D) · LSGAN with `target_real_label=0.9` · `lambda_GAN=1`, `lambda_L1=100` ·
**`use_ssim=False`, `use_lpips=False`** · `warmup_g_only_epochs=0` · `lr_G=lr_D=1e-4` ·
`ReduceLROnPlateau` on **both** nets (patience 10, factor 0.5, min 1e-6, active from epoch 20) ·
best-by-val-L1 at **epoch 57**.

## I.1 What has already improved since the paper

These are real fixes and should be kept:

1. **Split lists are now persisted** — `save_split_csv` writes `{train,val,test}_pairs.csv` with
   `roi_group / season / roi_id / patch_id` ([prepare_data_sen_12_ms.py L104-L140](codebase/prepare_data_sen_12_ms.py#L104-L140)).
   This is what made the leakage measurement in I.2 possible at all.
2. **`config.json` per run** — the paper run had nothing.
3. **Full diagnostics** — `history.csv` / `history.json` + 7 plots including `train_D_real`,
   `train_D_fake`, `lr_G`, `lr_D`. This is what exposes the finding in I.3.
4. **Season-stratified sampling** (6,400 patches per season in train) instead of winter-only —
   removes the hemisphere/season confound from the paper (old M8).
5. **Explicit `lambda_GAN`**, multi-criterion checkpoints, `scheduler_D` state in the checkpoint,
   `split_metadata.json`, and a memory-mapped evaluation `Dataset`.
6. **No early-stopping dead code**, no hidden resume-time LR override firing automatically.

## I.2 🔴 S1 — The split is still patch-level: 100 % scene overlap

**Status: CONFIRMED, unchanged from the paper, and now measurable.**

The split key is still `md5(basename)` of the individual patch
([L187-L193](codebase/prepare_data_sen_12_ms.py#L187-L193)), and the "season-stratified" design
stratifies by **season**, not by scene ([L340-L395](codebase/prepare_data_sen_12_ms.py#L340-L395)).
Because SEN12MS patches are 256×256 and `crops_per_image = 1`
([L480](codebase/prepare_data_sen_12_ms.py#L480)), one `.tif` patch = one sample.

Measured on the current `dataset_all_seasons/*.csv`:

```
unique scenes: train 252   val 252   test 252
scene overlap train∩test 252   train∩val 252
fraction of TEST patches whose scene is also in TRAIN: 1.0000
fraction of VAL  patches whose scene is also in TRAIN: 1.0000
```

The SEN12MS paper tiles each scene with a **stride of 128 px, i.e. 50 % overlap between adjacent
patches**, and explicitly recommends splitting "via the meteorological seasons, or via the
individual ROIs". So this is not merely same-scene correlation — a large share of test patches
share **half their pixels** with a training patch.

Every number produced by the current pipeline is therefore optimistically biased, and the bias will
not shrink as you improve the model — it will grow, because a leakier evaluation rewards
memorisation.

**Fix (do this before any further training run).** In `prepare_data_sen_12_ms.py`, replace the
per-patch quota loop with a per-scene assignment:

```python
# group patches by scene, then assign whole scenes to splits
scene_of = lambda p: extract_roi_info(p)[:3]          # (roi_group, season, roi_id)
by_scene = defaultdict(list)
for score, s1, s2 in season_pairs:
    by_scene[scene_of(s1)].append((score, s1, s2))

scenes = sorted(by_scene, key=lambda s: hash_score("|".join(s)))   # deterministic, scene-level
n_tr = int(round(P_TRAIN * len(scenes)))
n_va = int(round(P_VAL   * len(scenes)))
train_scenes = scenes[:n_tr]
val_scenes   = scenes[n_tr:n_tr + n_va]
test_scenes  = scenes[n_tr + n_va:]
```

plus an assertion that the three scene sets are pairwise disjoint, and record the scene lists in
`split_metadata.json`. With 252 scenes an 80/10/10 scene split gives ≈201/25/26 scenes; patch counts
per split will be ragged — that is expected and fine.

Effort: **S** to implement, **L** because every result must be regenerated.

## I.3 🔴 S2 — The discriminator has won; the adversarial term is dead

**Status: NEW — found in the committed `history.csv` of the current run.**

With LSGAN and `target_real_label = 0.9`, `loss_G_GAN = mean((D(fake) − 0.9)²)`, so
`0.9² = 0.81` is exactly the value you get when **D outputs 0 for every generated patch**.

| epoch | G_total | **G_GAN** | L1·100 | D_total | D_real | D_fake | val_L1 |
|---|---|---|---|---|---|---|---|
| 1 | 11.381 | **0.778** | 10.603 | 0.0235 | 0.0218 | 0.0252 | 0.09466 |
| 10 | 7.869 | **0.808** | 7.061 | 0.0047 | 0.0043 | 0.0050 | 0.08250 |
| 30 | 5.563 | **0.811** | 4.753 | 0.0015 | 0.0015 | 0.0016 | 0.07975 |
| 57 | 4.425 | **0.808** | 3.616 | 0.0014 | 0.0015 | 0.0013 | 0.07880 |
| 100 | 3.749 | **0.810** | 2.939 | 0.0000 | 0.0000 | 0.0000 | 0.08019 |

`train_G_GAN` is pinned at **0.810 ± 0.003 for all 100 epochs** while `L1·100` falls 10.60 → 2.94,
and `train_D_loss` decays to `3e-5`. The discriminator separates real from generated perfectly from
the first epochs onward and the generator never moves the adversarial term by any measurable amount.

**The current model is, in effect, a non-adversarial L1 U-Net with an inert GAN branch.** Everything
the network learned came from the L1 term. (Silver lining: this is the no-GAN baseline that the old
M5 asked for — you have it already, you just have it by accident and under the label "cGAN".)

Contributing causes, in order of likely impact:
1. **`lambda_GAN = 1` against `lambda_L1 = 100`.** G has essentially no incentive to fool D, so it
   never leaves the region where D is confident.
2. **D is far too strong**: 1:1 updates, identical LR (`lr_D = lr_G = 1e-4`), a full 70×70 PatchGAN,
   and no regularisation. A pure-L1 generator produces over-smooth output whose local statistics are
   trivially separable from real Sentinel-2 texture, so D's job is easy.
3. **No SSIM/LPIPS** (see S3) — those are exactly the terms that would push G toward realistic
   high-frequency texture and make D's task non-trivial.
4. **Label smoothing on G's target** (see S4) moves the goalpost to 0.9 but does nothing about
   saturation.

**Fixes, in the order I would try them:**

a. **Instrument first.** Log `D(real).mean()` and `D(fake).mean()` directly, not just the losses —
   one line each in the D block. If `D(fake) ≈ 0` and `D(real) ≈ 0.9`, D is saturated.
b. **Handicap D.** The knob is already in the file, just commented out:
   [L67](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py#L67) `# lr_D = 2.5e-5`.
   Enable it, and/or update D every *k*-th batch rather than every batch.
c. **Regularise D.** Spectral norm on the PatchGAN (absent from `networks.py`) or an R1 gradient
   penalty is the standard remedy for an overpowering discriminator; add instance noise to D's
   inputs as a cheap alternative.
d. **Re-enable SSIM + LPIPS** (S3) before raising `lambda_GAN` — raising the weight on a saturated
   term only amplifies a constant.
e. **Then** sweep `lambda_GAN ∈ {1, 5, 10}` with D no longer saturated.

Effort: **S** for (a)+(b), **M** for (c)–(e).

## I.4 🔴 S3 — The composite loss is switched off, and the checkpoint machinery degenerates with it

**Status: CONFIRMED.**

[L97-L108](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py#L97-L108):
```python
use_ssim  = False
use_lpips = False
lambda_ssim   = 10.0     # dead
lambda_lpips  = 10.0     # dead
```
Knock-on effects, all confirmed against `history.csv`:

- `val_SSIM` and `val_LPIPS` are empty for all 100 rows, so
  `if save_best_by_val_ssim and val_ssim is not None …`
  ([L1078](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py#L1078)) and the LPIPS
  equivalent ([L1085](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py#L1085))
  **can never fire**.
- `compute_val_total` ([L460-L473](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py#L460-L473))
  reduces to `100 × val_L1`, a monotone transform of `val_L1`. So "best by val_total" and "best by
  val_L1" select the **same** epoch — confirmed: both minima at epoch 57.
- Net result: the four-criterion selection is really **one** criterion. This matches what is on
  disk: only `pix2pix_best_netD.pth`, `…_best_val_l1_netD.pth`, `…_best_val_total_netD.pth` exist —
  no `val_ssim` / `val_lpips` files.

Also note `lambda_ssim`/`lambda_lpips` are now **10**, not the 50 the paper used; if you re-enable
them, that is a different experiment from the published one and must be labelled as such.

**Fix:** either re-enable both losses (and let the multi-criterion machinery do its job), or delete
the dead branches so the run is honestly described as GAN+L1. Guard against the silent degeneracy:

```python
assert not (save_best_by_val_ssim and not use_ssim), "val_ssim selection requires use_ssim"
if not (use_ssim or use_lpips):
    print("[WARN] val_total == lambda_L1 * val_L1; 'best_val_total' duplicates 'best_val_l1'")
```

Effort: **S**.

## I.5 🔴 S4 — One-sided label smoothing leaks into the generator's objective

**Status: NEW.**

[L204](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py#L204):
```python
criterionGAN = GANLoss(gan_mode="lsgan", target_real_label=0.9, target_fake_label=0.0)
```
The **same object** is used for the generator step
([L765](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py#L765)):
```python
loss_G_GAN = criterionGAN(pred_fake, True)   # target = 0.9, not 1.0
```
One-sided label smoothing (Salimans et al., 2016) is defined on the **discriminator's real target
only**. Here it is also applied to the generator, so G is optimised to make D output 0.9 rather than
1.0 — the adversarial objective is capped by construction, and the `0.81` plateau in I.3 is
precisely `(0 − 0.9)²`.

**Fix:** use two loss objects, or an explicit target for G.
```python
criterionGAN_D = GANLoss("lsgan", target_real_label=0.9, target_fake_label=0.0).to(device)
criterionGAN_G = GANLoss("lsgan", target_real_label=1.0, target_fake_label=0.0).to(device)
```
and call `criterionGAN_G(pred_fake, True)` in the generator step.

Effort: **S**. Changes training — put it on `review-fixes`.

## I.6 🔴 S5 — InstanceNorm is a questionable choice for radiometric regression

**Status: NEW (the switch BatchNorm → InstanceNorm happened silently between runs, with no ablation).**

[L184-L196](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py#L184-L196) sets
`norm="instance"` for **both** G and D, which resolves to
`nn.InstanceNorm2d(affine=False, track_running_stats=False)`
([networks.py L32](codebase/mymodels/pytorch-CycleGAN-and-pix2pix/models/networks.py#L32)).

InstanceNorm subtracts the per-image, per-channel mean and divides by the per-image standard
deviation at **every** layer, and with `affine=False` there are no learnable parameters to restore
the level. The task here is to predict **absolute TOA reflectance**; the per-image radiometric level
*is* part of the target. Removing it at every layer is a structural handicap for exactly the
quantity you report (MAE, RMSE, per-band bias) — and a plausible contributor to the colour-cast
behaviour that the independent-stretch figures (S11) currently hide.

Batch size is 16, so there is no batch-statistics argument for preferring IN here.

**Fix (ablation, not a unilateral change):** `norm ∈ {batch, group, instance}` for G, held otherwise
constant, judged on **per-band mean error (bias)** and NRMSE, not only SSIM. `get_norm_layer` does
not support GroupNorm — add it:
```python
elif norm_type == "group":
    norm_layer = functools.partial(nn.GroupNorm, 8)   # (num_groups, num_channels)
```
Effort: **M**.

## I.7 🟠 S6 — The evaluation scripts point at a model directory that is not in this tree

**Status: CONFIRMED.**

`MODEL_DIR = "trained_all_seasons_debugged_loss_balance_label_smoothing"`
([evaluate…efficient.py L25](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py#L25))
and the same string in
[infer_generate_optical.py L25-L26](codebase/infer_generate_optical.py#L25-L26), while the only run
directory present is `trained_all_seasons_debugged_loss_**original**_label_smoothing`. So there are
at least two label-smoothing variants and the evaluation/figure scripts are wired to the one whose
`config.json` and `history.csv` are **not** tracked.

Separately, `prepare_data_sen_12_ms.py` now writes `dataset_all_seasons_metadata`
([L58](codebase/prepare_data_sen_12_ms.py#L58)) while the training script reads `dataset_all_seasons`
([L80](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py#L80)) — two directories with
byte-identical `split_metadata.json` but no guarantee they were generated from the same run.

**Fix:** a single `RUN_NAME` constant (or a small `paths.py`) imported by prep/train/eval/infer, so
one edit retargets the whole pipeline; commit `config.json` + `history.csv` for every variant you
intend to cite. Effort: **S**.

## I.8 🟠 S7 — Metrics are still normalised by the test set

**Status: CONFIRMED (carried over from Part II C5) — and it now blocks run-to-run comparison.**

`data_range = global_max − global_min` of the **test ground truth**
([evaluate…efficient.py L469](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py#L469)),
used as PSNR's `MAX` and as the aggregate SSIM `data_range`; per-band metrics use a *different*,
per-band range ([L474](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py#L474)).
Aggregate PSNR is **pooled**, not per-image. SSIM is `skimage`'s default **7×7 uniform** window.

While you are iterating on training variants this is the most consequential of the "old" findings:
**two runs scored this way are not strictly comparable**, and aggregate vs per-band numbers are on
different scales within a single run.

**Fix:** freeze the metric spec before the next training run —
`PSNR_MAX = 10000.0`, `SSIM_DATA_RANGE = 10000.0` (constants, not derived), report per-image PSNR as
well as pooled, add per-band `R²` and `NRMSE = RMSE / σ_band`, add a per-band training-mean trivial
baseline, keep LPIPS on a backbone different from any training loss, and weight the LPIPS batch
means by batch size ([L560](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py#L560)
currently uses an unweighted `np.mean`). Effort: **M**.

## I.9 🟠 S8 — `set_lrs_on_resume` will crash if used as documented

**Status: NEW — latent bug.**

[L76-L78](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py#L76-L78):
```python
# ... set this to a float, e.g. 5e-5. Leave as None for normal checkpoint resume.
set_lrs_on_resume  = None
```
but [L425-L426](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py#L425-L426):
```python
forced_lr_G = set_lrs_on_resume["G"]
forced_lr_D = set_lrs_on_resume["D"]
```
Following the comment (`set_lrs_on_resume = 5e-5`) raises
`TypeError: 'float' object is not subscriptable` on the first resume. Harmless today because the
value is `None`, but it is a trap waiting for the next long run that needs resuming.

**Fix:** make the comment match the code — `set_lrs_on_resume = {"G": 5e-5, "D": 5e-5}` — or accept
both forms. Effort: **S**.

## I.10 🟠 S9 — `decay_after_warmup_only` is not tied to the warm-up

**Status: CONFIRMED (naming/semantics).**

`decay_after_warmup_only = True` gates on the independent constant `decay_starting_epoch = 20`
([L74-L75](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py#L74-L75),
[L965-L972](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py#L965-L972)), while
`warmup_g_only_epochs = 0`. In the committed run the schedulers were therefore frozen for the first
20 epochs for a reason unrelated to any warm-up, and `ReduceLROnPlateau`'s `num_bad_epochs` counter
only starts at epoch 20.

**Fix:** `decay_starting_epoch = warmup_g_only_epochs` when the flag is set, or rename the flag to
`decay_starts_at_epoch`. Effort: **S**.

## I.11 🟠 S10 — No seeds; `cudnn.benchmark = True`

**Status: CONFIRMED, and now the highest-leverage cheap fix.**

No `torch.manual_seed`, `np.random.seed`, `random.seed`, `cudnn.deterministic` anywhere in
`codebase/*.py`; `torch.backends.cudnn.benchmark = True`
([L50](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py#L50)). You are now comparing
loss configurations, label-smoothing variants and norm layers — **without seeds you cannot tell a
real effect from run-to-run noise.** The paper's loss ablation (Table 6) already suffered from this.

**Fix:**
```python
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```
called before model construction, with `seed` in `config.json`, plus RNG states in
`save_checkpoint` so a resume is exact. Then run every comparison with ≥3 seeds and report mean ± std.
Effort: **S** to add, **L** to re-run comparisons.

## I.12 🟡 S11 — Remaining smaller items in the current pipeline

| # | Finding | Evidence | Fix |
|---|---|---|---|
| a | **Qualitative figures stretch prediction and ground truth independently** — `rgb_from_s2(mode="global")` rescales each image by its own per-channel min/max, so any colour cast or radiometric bias is erased before the figure is drawn. | [infer_generate_optical.py L50-L82](codebase/infer_generate_optical.py#L50-L82), [L219-L240](codebase/infer_generate_optical.py#L219-L240) | Compute the stretch from the ground truth once and apply it to both panels. |
| b | **Whole dataset loaded into RAM.** `get_train_val_loaders` uses `np.load` without `mmap_mode`: `y_train` alone is 25,600 × 256 × 256 × 13 × 4 B ≈ **87 GB**, plus ≈13 GB for `X_train`. | [myutils.py L28-L33](codebase/myutils.py#L28-L33) | Reuse the `NPYPairDataset` (mmap) the evaluation script already has. |
| c | **Duplicate generator forward per step** — `netG(real_A)` under `no_grad` for D, then again for G. Under InstanceNorm (`track_running_stats=False`) + `use_dropout=False` the two passes are numerically equivalent, so this is now **pure waste (~25 % of step cost)** and is safe to remove without changing results. | [L704](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py#L704), [L756](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py#L756) | `fake_B = netG(real_A)` once; pass `fake_B.detach()` to D. |
| d | **`drop_last` not set.** Currently harmless (25,600 / 16 and 3,200 / 16 are exact) — but a scene-disjoint split (S1) will produce ragged sizes. | [myutils.py L45-L46](codebase/myutils.py#L45-L46) | `drop_last=True` on the train loader when you land S1. |
| e | **Global `[0, 10000]` normalisation for all 13 bands.** With SSIM/LPIPS off the objective is pure L1 in that space, so high-DN bands dominate and B10 sits at ≈ −1. | [prepare_data_sen_12_ms.py L33-L47](codebase/prepare_data_sen_12_ms.py#L33-L47) | Ablate per-band 1–99 % clips computed on the **training split only**; a tighter table is already drafted (commented out) in [myutils.py L9-L15](codebase/myutils.py#L9-L15). |
| f | **No NaN guard.** `np.clip` propagates NaN and the sanity check only counts them. | [prepare_data_sen_12_ms.py L690-L700](codebase/prepare_data_sen_12_ms.py#L690-L700) | Fail, or impute and log, on non-finite input. |
| g | **`lpips_fn` not explicitly frozen** in training (relies on the package's `eval_mode=True` default); gradients accumulate into its linear heads and are never zeroed. | [L217-L222](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py#L217-L222) | `lpips_fn.eval(); lpips_fn.requires_grad_(False)`. |
| h | **LPIPS input range.** `DN/5000 − 1` puts typical land reflectance (300–2000 DN) in ≈`[−0.94, −0.60]`, ~17 % of the nominal `[−1, 1]`. Relevant once LPIPS is re-enabled. | [L115](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py#L115) + clip table | Ablate an LPIPS-specific stretch, e.g. clip `[0, 3000]` DN → `[−1, 1]`, applied identically in loss and metric. |
| i | **`ssim` metric object never `.reset()`** and shared between train and val loops. `forward()` returns the correct per-batch value, so the loss is right; only the unused accumulator is polluted. | [L213-L216](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py#L213-L216) | `ssim.reset()` per epoch. |

## I.13 Recommended plan for the current pipeline

1. **S1 — scene-disjoint split.** Nothing measured before this is trustworthy, and it is a ~20-line
   change to a file that already writes the CSVs. *(S)*
2. **S10 — seeds + determinism**, and **S7 — freeze the metric spec** (constants, per-band R²/NRMSE,
   trivial-mean baseline). These two make every subsequent comparison meaningful. *(S / M)*
3. **S2 + S4 — fix the adversarial collapse**: log `D(real)`/`D(fake)`, drop `lr_D`, add spectral
   norm or R1, and stop smoothing G's target. Re-check that `train_G_GAN` actually moves. *(S → M)*
4. **S3 — decide on the loss**: either re-enable SSIM + LPIPS (and re-tune `lambda_*`), or drop the
   dead branches and describe the model as GAN+L1. *(S)*
5. **S5 — norm-layer ablation** (batch / group / instance) judged on per-band bias. *(M)*
6. **S6, S8, S9, S11 a–i** — housekeeping; do them alongside the above. *(S)*
7. Re-run the 60 m-band and loss ablations on the clean split with ≥3 seeds and the frozen metric
   spec. *(L)*

---
---

# PART II — Manuscript errata (paper-era implementation)

> The code below is superseded. These findings are retained because the manuscript still states
> them and must be corrected before publication, independently of where the code has moved.

## 0. Which code produced which table

This matters, because the repo now contains **three generations** of the pipeline and the
working-tree defaults no longer match the paper.

| Paper artifact | Data | Training script | Model dir |
|---|---|---|---|
| Table 3 (main), Table 4 (per-band), Table 5 (CR) | `dataset_winter_all` (winter, 31,825 patches) | `train_pix2pix_lpips_ssim_generate_checkpoints.py` *(orig)* | `trained-full` |
| Table 6 (loss ablation), Table 7 "All 13 bands" | 20 % of winter | same script, lambdas varied | — |
| Table 7 "Excluding 60m", Table 8 | `data-256_0.02_10000_60m_execluded` (20 % of winter) | [codebase/train_pix2pix_lpips_ssim_generate_60m_excluded.py](codebase/train_pix2pix_lpips_ssim_generate_60m_excluded.py) | `trained-both_0.02_10000_60m_execluded_no_scheduler` |
| *(not in the paper)* | all 4 seasons | [codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py) | `trained_all_seasons_debugged_loss_original_label_smoothing` |

Neither `dataset_winter_all` nor `trained-full` contains split lists or checkpoints in the repo,
so **the exact test set behind Table 3 cannot be reconstructed from this repository**. The
quantitative leakage numbers below were measured on `codebase/dataset_all_seasons/*.csv`, which was
produced by the *same* splitting function.

---

## 1. CRITICAL findings

### C1 Train/val/test split is unspecified, so leakage is likely
**Status: WORSE THAN SUSPECTED**

Evidence:
- The split key is the MD5 of the **patch filename**, not the scene:
  [codebase/prepare_data_sen_12_ms.py](codebase/prepare_data_sen_12_ms.py#L187-L193)
  ```python
  def hash_score(path):
      h = hashlib.md5(os.path.basename(path).encode("utf-8")).hexdigest()
      return int(h[:8], 16) / 0xFFFFFFFF
  ```
  The *(orig)* winter script is identical: `pairs = [(hash_score(s1), s1, s2) ...]`, then
  `train_pairs = pairs[0:n_train]`, `val_pairs = pairs[n_train:...]`, `test_pairs = pairs[...]`.
- SEN12MS patches are 256×256, so [codebase/prepare_data_sen_12_ms.py](codebase/prepare_data_sen_12_ms.py#L480)
  yields `crops_per_image = 1`: one `.npy` sample per `.tif` patch. The split is therefore purely
  patch-level.
- Measured on the saved split lists (`codebase/dataset_all_seasons/{train,val,test}_pairs.csv`,
  25,600 / 3,200 / 3,200):
  ```
  unique scenes: train 252  val 252  test 252
  scene overlap train&test 252   train&val 252
  frac of TEST patches from a scene also present in TRAIN: 1.0000
  frac of VAL  patches from a scene also present in TRAIN: 1.0000
  ```
  **Every scene appears in all three splits. 100 % of test patches come from training scenes.**
  For the winter subset alone: 46 winter scenes, 46/46 shared between train and test.
- The leakage is not merely "same scene", it is **pixel-level**. The SEN12MS paper states:
  > "These remaining scenes were then tiled into patches of 256×256 pixels in size. Again, we have
  > implemented a **stride of 128 pixels, resulting in an overlap between adjacent patches of 50 %**."

  and explicitly recommends:
  > "Deterministic splits into disjunct training and test sets can be achieved via the
  > meteorological seasons, or via the individual ROIs."

  A random patch split therefore guarantees that a large share of test patches share **half of
  their pixels** with a training patch (up to 8 neighbours each).
- The same val set drives `scheduler_G.step(val_loss)`
  ([train_pix2pix_lpips_ssim_generate_checkpoints.py](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L483))
  and checkpoint selection
  ([L543](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L543)); the test set is separate
  from val, so that part of the hypothesis is **ruled out** — but both sets are equally leaked.
- The split is deterministic (MD5 of filename), so it is reproducible *if* the file list is
  reproducible. For the winter run no file list was saved (`save_split_csv` only exists in the
  current version).

Impact on reported results: **Table 3, Table 4, Table 6, Table 7, Table 8 are all optimistically
biased by an unknown but likely large margin.** The direction is unambiguous (too good). All
headline claims that rest on absolute metric values are affected.

Proposed fix:
1. New `make_splits.py`: group by `(roi_group, season, roi_id)`, assign whole scenes to
   train/val/test with a fixed seed (80/10/10 by scene), write `*_pairs.csv` into the dataset dir.
2. Optionally also drop test patches whose patch index is a spatial neighbour of a training patch
   (belt-and-braces; unnecessary once scenes are disjoint).
3. `check_leakage.py` asserting empty scene intersections and printing split sizes.
4. Retrain and re-report every table.

Effort: **L** (retraining dominates).

---

### C2 SEN12MS-CR "cloud removal" evaluation may overlap with training data
**Status: CONFIRMED (structurally); exact overlap % cannot be measured from this repo**

Evidence:
- Training data *(orig)*: `data_dir = ".../sen12-ms/sub_dataset/ROIs2017_winter"`.
- CR evaluation data: [codebase/prepare_data_sen_12_ms_cr.py](codebase/prepare_data_sen_12_ms_cr.py#L16-L18)
  ```python
  S1_ROOT       = ".../sen12-ms-cr/sub_dataset/ROIs2017_winter_s1"
  S2_CLEAN_ROOT = ".../sen12-ms-cr/sub_dataset/ROIs2017_winter_s2"
  ```
  Same ROI seed (`ROIs2017`), same season branch, same file naming
  ([L48-L51](codebase/prepare_data_sen_12_ms_cr.py#L48-L51)):
  `ROIs2017_winter_s1_(\d+)_p(\d+).tif` — i.e. **the same (scene, patch) grid as SEN12MS**.
  SEN12MS-CR is a co-registered extension of SEN12MS over the same ROIs; the S1 patch with key
  `(68, 100)` denotes the same ground footprint in both datasets.
- Sample selection: [L21-L27](codebase/prepare_data_sen_12_ms_cr.py#L21-L27),
  `MAX_PATCHES = 2000`, `RNG_SEED = 123`, `BALANCE_ACROSS_SCENES = True` → stratified round-robin
  over *all* scenes present in the downloaded winter subset.
  **There is no filter anywhere that excludes keys used in training.** Grep for any exclusion
  against the SEN12MS training list returns nothing.
- Consequently, with an 80 % train fraction over the same scene/patch universe, the expected
  overlap of the 2,000 CR samples with training footprints is ≈80 %, and ~100 % at scene level.
- What cannot be determined here: the raw `.tif` trees are not in the repo, so the exact
  intersection count and whether SEN12MS-CR reuses the *identical* S1 acquisition (vs. a different
  date over the same footprint) could not be verified.

Impact: **Table 5 (SSIM 0.899 / PSNR 33.65 / LPIPS 0.152 / SAM 3.86) is very likely measured
largely on training footprints.** Direction: too good. This also explains the red flag of a
SAR-only model beating SAR+cloudy-optical fusion baselines.

Proposed fix: `scripts/cr_overlap_check.py` that builds the `(roi_id, patch_id)` key set of the
winter training list and of the 2,000 CR samples, reports `|intersection|` and `%`, writes a
filtered list; alternatively switch to the official SEN12MS-CR test ROIs. Re-report Table 5.

Effort: **S** for the check, **M** for re-evaluation.

---

### C3 Table 5 compares against baselines under a different protocol
**Status: CONFIRMED**

Evidence: there is no baseline implementation anywhere in `codebase/` — `mymodels/` contains only
`cp2pwrapper.py`, `SiMaVP.py`, `swinunetwrapper.py`, `UMambaWrapper.py` and the vendored pix2pix
repo. No DSen2-CR / FAT / DiffCR code, weights, or inference script exists, so **no baseline was
re-run**; the numbers in [MDPI_manuscript/results.tex](MDPI_manuscript/results.tex) Table 5 are
transcribed from the cited papers.

Our protocol differs in: subset (2,000 winter samples vs. the official all-season CR test split),
metric implementation (see C5), and data range (see C5). Additionally DiffCR's SSIM (0.902) is
higher than ours (0.899), so "outperforms DiffCR" is not supported by the table itself.

Impact: Table 5 and the surrounding claims ("best-in-class PSNR", "outperforms … DiffCR") are not
a like-for-like comparison.

Proposed fix: either re-run DSen2-CR (public weights) on exactly our 2,000 samples with our metric
code, or evaluate on the official split; otherwise reword to "competitive under our protocol" and
add a protocol-difference footnote.

Effort: **M**.

---

### C4 The 60 m band ablation conclusion is an averaging artifact
**Status: CONFIRMED — and there is an additional, unanticipated confound**

Evidence:
1. **Different band sets are averaged.** The 13-band evaluator averages over
   `BAND_NAMES = ["B1","B2",...,"B12"]`
   ([evaluate_pix2pix_clips_per_band_all_losses_efficient.py](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py#L136));
   the 10-band evaluator over `["B2",...,"B12"]`
   ([evaluate_pix2pix_clips_per_band_all_losses_60m_execluded.py](codebase/evaluate_pix2pix_clips_per_band_all_losses_60m_execluded.py#L68)).
   No option exists to restrict the 13-band model to the common 10 bands. The reviewer's
   recomputation (0.827 vs 0.825 SSIM; 27.50 vs 27.34 dB) therefore stands.
2. **SAM is computed over different vector dimensions** — `compute_sam_deg(truth, preds)` over
   `C_out` channels (13 vs 10). 6.71° and 6.33° are not comparable quantities.
3. **The Table 7/8 inconsistency is explained by the `data_range` choice** (see C5):
   aggregate SSIM uses the *global* data range
   ([L473](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py#L473),
   [L493](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py#L493))
   ```python
   global_dr = max(1e-9, data_range)                       # ≈ 10000
   s = structural_similarity(truth[i], preds[i], channel_axis=-1, data_range=global_dr)
   ```
   while per-band SSIM uses the *per-band* range
   ([L474](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py#L474)):
   ```python
   band_dr = np.maximum(1e-9, (band_max - band_min))
   s = structural_similarity(yt[i], yp[i], data_range=dr_c)
   ```
   A larger `data_range` inflates the SSIM stabilising constants `C1, C2`, which pushes SSIM up.
   This is exactly why the aggregate (0.859) exceeds the mean of the per-band values (0.845).
   The same holds for Table 3 vs Table 4 (0.888 vs mean 0.880 over Table 4). PSNR has the same
   asymmetry: aggregate uses `global_dr`, per-band uses `band_dr`.
4. **NEW — the two models were not trained under the same schedule.** The 10-band run has the
   LR scheduler commented out:
   [codebase/train_pix2pix_lpips_ssim_generate_60m_excluded.py](codebase/train_pix2pix_lpips_ssim_generate_60m_excluded.py#L144)
   `# scheduler_G = ReduceLROnPlateau(...)` and
   [L352](codebase/train_pix2pix_lpips_ssim_generate_60m_excluded.py#L352) `# scheduler_G.step(val_loss)`
   (hence `save_dir = "trained-both_0.02_10000_60m_execluded_no_scheduler"`), whereas the 13-band
   run uses `ReduceLROnPlateau`
   ([train_pix2pix_lpips_ssim_generate_checkpoints.py](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L150)).
   The paper claims "identical hyperparameters". It is therefore not established that the observed
   gap is attributable to the 60 m bands at all.

Impact: **Table 7 and the entire §"Influence of 60 m Spectral Bands" conclusion are unsupported.**
Also the Table 7 "All 13 bands" / Table 6 "Full Combination" aggregate SSIM (0.859) is not
comparable to the Table 8 per-band mean.

Proposed fix: add `--bands` to the evaluator so the 13-band model is scored on the common 10 bands
with identical `data_range` per band; compute SAM on the 10 common bands for both; re-train the
10-band model with the scheduler enabled; ≥3 seeds, report mean ± std.

Effort: **M** (evaluator) + **L** (retraining).

---

### C5 Metric implementation inflates some numbers
**Status: CONFIRMED — mechanism identified exactly**

Evidence ([evaluate_pix2pix_clips_per_band_all_losses_efficient.py](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py)):

| Question | Answer | Lines |
|---|---|---|
| Metrics on tensors or PNGs? | **Tensors**, after inverse normalisation to DN. `util.tensor2im`/`save_images` are never used. | L108-L132, L440-L460 |
| PSNR `MAX` | `data_range = global_max - global_min` of the **test ground truth** → ≈10000 because clipping saturates | L469 |
| PSNR pooling | **Pooled**: `psnr_from_mse(np.mean(mse_c), data_range)` — one PSNR for the whole set, not per-image | L105-L110, L470 |
| Per-band PSNR | per-band MSE **and** per-band `data_range` → different normalisation than the aggregate | L512-L515 |
| SSIM library | `skimage.metrics.structural_similarity`, default 7×7 window (**not** 11×11 Gaussian as stated in the paper) | L13, L493 |
| SSIM space | reflectance/DN, `channel_axis=-1` (multichannel → mean over the 13 bands), `data_range=global_dr` | L493 |
| Per-band SSIM | `data_range=band_dr[c]` → systematically lower than the aggregate | L474, L503 |
| SAM | reflectance space ✔, `eps=1e-8` via `np.maximum`, `np.clip(cos, -1, 1)` ✔, **per-pixel** mean over all pixels, degrees ✔ | L203-L225 |
| MAE / RMSE | DN units after inverse transform, pooled over channels (`uniform_average`) | L463-L468 |
| LPIPS | `net='alex'` at eval vs `net='vgg'` in training; RGB = B4/B3/B2 rescaled to [0,1] then to [−1,1] | L371, L166-L200 |

Confirmations:
- `20*log10(10000) - 10*log10(233.69²) = 32.63 dB` reproduces Table 3 exactly → PSNR uses
  MAX = 10000 with pooled RMSE. Since most land reflectance is 0–3000 DN, this inflates PSNR by
  roughly 10 dB relative to a per-band dynamic range, and systematically favours low-variance bands.
- B1's "best SSIM 0.976" is computed with `data_range = band_max − band_min` of B1 over the test
  set — but B1/B9/B10 are smooth, low-variance bands, so a near-constant prediction already scores
  high. There is **no trivial baseline** anywhere in the repo to calibrate this.
- **NEW:** both `global_dr` and `band_dr` are derived from the *test set itself*. Metrics are
  therefore dataset-dependent and **Table 3 and Table 5 are not computed on the same scale** unless
  both test sets happen to saturate the clip bounds.
- **NEW (minor bias):** LPIPS is averaged as an unweighted mean of per-batch means
  (`np.mean(lpips_vals)`, L560), so the final partial batch is over-weighted.

Impact: PSNR in Tables 3, 4, 5, 6, 7, 8 is inflated and not comparable to papers that use a
per-band or [0,1] range. Aggregate vs per-band SSIM/PSNR are on different scales within the same
paper. The paper's claim of an 11×11 SSIM window is wrong.

Proposed fix: new `eval_metrics.py` with explicit, documented, **fixed** ranges
(`data_range=10000` everywhere, plus a second per-band-range column), per-image PSNR option,
`R²`/`NRMSE` per band, a per-band training-mean trivial baseline (and a per-land-cover-class mean
using the SEN12MS MODIS layer), LPIPS with a backbone different from training, optional FID/KID,
CSV + JSON output, and a documented metric-spec block at the top of the file.

Effort: **M**.

---

### C6 Test-time BatchNorm/dropout mode
**Status: RULED OUT (with one caveat)**

Evidence:
- The official `test.py` is never used. Evaluation uses a custom loop:
  `model.eval()` at [L404](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py#L404)
  and `with torch.no_grad():` at L440 / L484 / L524. Same in the *(orig)* script.
- Dropout is hard-disabled in the generator:
  [codebase/mymodels/cp2pwrapper.py](codebase/mymodels/cp2pwrapper.py#L41-L48) → `use_dropout=False`.
- Batch size at eval is 16, not 1, and `netG` runs in `eval()` so BN uses running statistics.
- Norm layer: the wrapper default is `norm="batch"`
  ([cp2pwrapper.py](codebase/mymodels/cp2pwrapper.py#L29)), and the *(orig)* training and *(orig)*
  evaluation both use the default → **consistent**. (The working-tree evaluator now passes
  `norm="instance"` to match the newer all-seasons run; a mismatch would raise in
  `load_state_dict`, so it cannot silently corrupt results.)

Caveat: determinism was not empirically verified by running the evaluation twice (no GPU/data in
this environment). `torch.backends.cudnn.benchmark = True` is set in the training scripts only.

Impact: none.

Proposed fix: state the norm layer (BatchNorm), `use_dropout=False`, `eval()` mode and batch size
in the paper; add an assertion that two consecutive evaluations produce bit-identical CSVs.

Effort: **S**.

---

## 2. MAJOR findings

### M1 Hidden augmentation (official-repo default)
**Status: RULED OUT**

Evidence: the official dataset/transform stack is never used. Data is pre-normalised into `.npy`
and loaded through a plain `TensorDataset`
([codebase/myutils.py](codebase/myutils.py#L43-L46)); only `shuffle=True` is applied. The
pre-processing performs a deterministic non-overlapping crop grid
([prepare_data_sen_12_ms.py](codebase/prepare_data_sen_12_ms.py#L199-L207)) which is a no-op for
256×256 patches. No resize, no random crop, no flip, no `get_transform`, no `--preprocess`.
The paper's "no data augmentation" claim is accurate.

Impact: none. Effort: —.

### M2 Stated hyperparameters vs actual code
**Status: PARTIALLY TRUE**

| Setting | Paper | Code *(orig unless noted)* | Verdict |
|---|---|---|---|
| `gan_mode` | lsgan | `GANLoss(gan_mode="lsgan")` [L121](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L121) | ✔ |
| `lr` | 1e-4 | `lr = 1e-4`, Adam(0.5, 0.999) | ✔ |
| `lr_policy` | ReduceLROnPlateau on val L1, factor 0.5, **patience 10** | `ReduceLROnPlateau(optimizer_G, "min", patience=patience, factor=0.5)` with `patience = 20` [L37](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L37), [L150](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L150), stepped with val L1 [L483](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L483) | ✘ **patience is 20, not 10** |
| D LR constant | yes | only `scheduler_G` exists | ✔ |
| `input_nc`/`output_nc` | 2 / 13 | inferred from data shapes | ✔ |
| D input channels | 15 | `define_D(input_nc=channels_in + channels_out, ...)` [L115](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L115) | ✔ |
| D norm | not stated | `define_D` default `norm="batch"`, 70×70 `basic` PatchGAN, no spectral norm | **undocumented** |
| G norm | not stated | `get_norm_layer("batch")`, `num_downs = log2(256) = 8` ⇒ equivalent to `unet_256` | **undocumented** |
| `lambda_L1` / SSIM / LPIPS | 100 / 50 / 50 | [L55-L57](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L55-L57) | ✔ |
| G warm-up 20 epochs | yes | `g_only = epoch < 20` [L297](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L297); D updates skipped **and** `loss_G_GAN = 0` [L365](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L365) | ✔ correct |
| epochs | 150 | current file `epochs = 150`; **`epochs = 300` at commit `9730959`** | ✘ ambiguous |
| 1:1 updates | yes | `g_steps_per_iter = d_steps_per_iter = 1` | ✔ |
| Best ckpt = lowest val L1 | yes | [L543](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L543) | ✔ |
| Early stopping | (CLAUDE.md: patience 20) | dead code: `if False and wait >= patience:` [L560](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L560) | ✘ **disabled** |
| — | not stated | `resume_with_halved_g_lr = True`, `resume_g_lr = 5e-5`, and the G scheduler is **re-created** on resume [L247-L268](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L247-L268) | ✘ **undocumented manual intervention** |

SSIM/LPIPS handling:
- SSIM loss: `torchmetrics` `StructuralSimilarityIndexMeasure(data_range=2.0)` on all 13 channels
  (NCHW), `loss = (1 - ssim) * 50` — sign and range are **correct**
  ([L378-L379](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L378-L379)).
- LPIPS loss: bands `(3,2,1)` = B4/B3/B2 for the 13-band model and `(2,1,0)` for the 10-band model
  — both correct; tensors are already in [−1,1] and `to_m11()` is correctly **not** applied
  ([L385-L397](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L385-L397)).

Impact: the paper's Methods section misstates the scheduler patience, the SSIM window, and omits
the norm layers, the disabled early stopping and the mid-training LR override. Reported metrics are
not directly affected, but reproducibility is.

Proposed fix: regenerate the hyperparameter table from `config.json`; document the resume override
or remove it; delete the dead early-stopping branch.

Effort: **S**.

### M3 Data normalization is suboptimal for 13 bands
**Status: CONFIRMED**

Evidence: every S2 band is clipped to the identical `[0, 10000]`
([prepare_data_sen_12_ms.py](codebase/prepare_data_sen_12_ms.py#L33-L47)) and mapped with
`to_tanh_range` (clip → scale → ×2−1, order correct, L158-L160). A tighter per-band table exists
but is commented out in [codebase/myutils.py](codebase/myutils.py#L9-L15). Inversion uses the saved
`S2_clip.json` and is exactly the inverse
([evaluate…efficient.py](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py#L120-L132)).
No per-image min–max is used anywhere in the training/eval path. SAR clipping is applied in dB
before scaling (L162-L172) ✔.

**NaN handling is absent**: `np.clip` propagates NaN, and the sanity check only *counts* NaNs
without failing (L690-L700). SEN12MS contains some no-data pixels.

Impact: L1 is dominated by the high-dynamic-range bands; B10 sits near −1 almost everywhere, which
is also why it scores so well. Not an error, but it biases both training and the aggregate metrics.

Proposed fix: ablation with per-band 1st–99th percentile clips computed on the **training split
only**, saved to `S2_clip.json`; add an explicit NaN guard/report in preprocessing.

Effort: **M**.

### M4 Normalization layers in G/D
**Status: CONFIRMED (never varied, never reported)**

Evidence: `norm="batch"` default in [cp2pwrapper.py](codebase/mymodels/cp2pwrapper.py#L29) for the
paper run; the later all-seasons run silently switched to `norm="instance"`
([train…_latest.py](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py#L184-L196)).
No GroupNorm, no spectral norm anywhere. No ablation exists.

Impact: none on current numbers; a gap in the study.
Proposed fix: BN / GroupNorm / InstanceNorm ablation in G and spectral norm in D, compared on SAM
and per-band mean error (bias), not only SSIM. Effort: **M**.

### M5 No non-adversarial baseline
**Status: CONFIRMED**

Evidence: the loss ablation in [MDPI_manuscript/ablations.tex](MDPI_manuscript/ablations.tex)
always includes `L_GAN`; there is no `lambda_GAN` knob in the paper training script at all (the
GAN term is added unweighted, [L411](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L411)).
A `lambda_GAN` variable only appears in the newer script. With weights 1 : 100 : 50 : 50 the
adversarial term may contribute very little.

Impact: the central claim "cGAN-based approach" is not supported against a plain regression U-Net.
Proposed fix: train the identical U-Net with `L1 + SSIM + LPIPS` and no D, same split/seed.
Effort: **M**.

### M6 LPIPS is circular
**Status: PARTIALLY TRUE**

Evidence: training uses `lpips.LPIPS(net="vgg")`
([L140](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L140)); evaluation uses
`lpips.LPIPS(net='alex')`
([L371](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py#L371)). Different
backbones, so the strongest form of circularity is avoided — but it is the same loss family on the
same band triplet, and the paper does not mention the backbones. No FID/KID.

Impact: minor optimistic bias in LPIPS rows (Tables 3, 5, 6).
Proposed fix: state both backbones in the paper; add FID/KID on the RGB stack. Effort: **S**.

### M7 Loss ablation anomaly (+SSIM → LPIPS 0.399 vs baseline 0.287)
**Status: CANNOT DETERMINE**

Evidence: no bug found. The SSIM loss sign and `data_range=2.0` are correct for [−1,1] inputs
([L378-L379](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L378-L379)); the generator
output is `tanh`-bounded so LPIPS never receives out-of-range values. The ablation runs are not in
the repo (no `trained-*` ablation dirs, no logs, no config.json, no seeds), so the specific run
cannot be inspected. A genuine effect (SSIM encouraging smooth, blurry solutions that LPIPS
penalises) is plausible and is consistent with the qualitative figure.

Impact: Table 6 interpretation is unverifiable.
Proposed fix: commit per-ablation `config.json` + `history.csv`; rerun with ≥3 seeds. Effort: **M**.

### M8 Winter-only confounds
**Status: CONFIRMED**

Evidence: SEN12MS seasons are **northern-hemisphere calendar** seasons (winter = 1 Dec 2016 –
28 Feb 2017), and the dataset authors state the seasonal structuring "is only of little semantic
worth"; they ship `seasons.csv` with the climatic season per scene. The repo's season detection is
purely string-based on the path
([prepare_data_sen_12_ms.py](codebase/prepare_data_sen_12_ms.py#L211-L216)) and `seasons.csv` is
never read. There is no latitude, snow, or land-cover stratification anywhere in the evaluator; the
MODIS `lc_*` layer of SEN12MS is never loaded. The S2 targets are cloud-free **seasonal mosaics**
while S1 is a single acquisition — the temporal mismatch is real and unaddressed.

Impact: the explanation in [MDPI_manuscript/results.tex](MDPI_manuscript/results.tex) for the poor
B8/NIR performance ("winter-only training samples, where vegetation-related information … is
comparatively scarce") does not hold globally.

Proposed fix: add `--stratify {hemisphere,snow,landcover}` to the evaluator (latitude from the
GeoTIFF CRS/bounds, NDSI = (B3−B11)/(B3+B11) > 0.4, MODIS IGBP class); use `seasons.csv` for a
climatic-season split; optionally a held-out-season generalisation test.

Effort: **M**.

---

## 3. MODERATE / MINOR

1. **No statistics — CONFIRMED.** No `torch.manual_seed`, `np.random.seed`, or `random.seed`
   anywhere in the training or evaluation scripts (grep across `codebase/*.py` returns only
   `random.seed(123)` in [prepare_data_sen_12_ms_cr.py](codebase/prepare_data_sen_12_ms_cr.py#L162)
   and `rng_seed=123` inside the SAM reservoir sampler). Single runs only; no mean ± std, no CIs.
2. **Table 4: B8 and B8A both 29.62 dB — CANNOT DETERMINE.** No evaluation CSV/log for the
   `trained-full` run is committed. Plausible but suspicious given SSIM differs (0.7738 vs 0.8071).
   Needs `metrics_per_band.csv` from that run.
3. **Checkpointing by lowest val L1 — CONFIRMED.**
   [L541-L556](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L541-L556). Favours
   blurrier outputs. The newer script already adds best-by-SSIM/LPIPS/weighted-total
   ([train…_latest.py](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py#L117-L126)) —
   backport it.
4. **Scheduler asymmetry — CONFIRMED.** Only `scheduler_G` exists in the paper script; D LR is
   constant. `history` records `D_loss` but no D/G balance diagnostics were saved for that run.
5. **SAR preprocessing — CONFIRMED.** No speckle filtering (SEN12MS deliberately ships unfiltered
   data: "we intend to leave any further pre-processing, e.g. speckle filtering, to the end user"),
   no VV/VH ratio channel (the ratio is computed **only for visualisation**,
   [myutils.py](codebase/myutils.py#L243-L252)), no incidence-angle input.
6. **Cloud removal framing — CONFIRMED.** `y_cloudy_test.npy` is produced by the CR preprocessing
   and loaded by [infer_generate_optical_cr.py](codebase/infer_generate_optical_cr.py#L160-L166)
   **for display only**; it is never used in the metric path. No cloud-mask blending, no
   cloudy-vs-clear stratified metrics.
7. **Downstream validation — CONFIRMED absent.** No NDVI/NDWI or classification evaluation.
8. **Uncertainty — CONFIRMED absent.** MC-dropout is not even possible: `use_dropout=False`
   ([cp2pwrapper.py](codebase/mymodels/cp2pwrapper.py#L47)).
9. **Unsupported claims — PARTIALLY TRUE.** `gan_mode` is hard-coded to `"lsgan"`; no BCE run
   exists, so "vanishing gradients with BCE" is not evidenced by anything in the repo. Loss curves
   are saved for the *newer* all-seasons run only
   (`trained_all_seasons_debugged_loss_original_label_smoothing/pix2pix/*.png`, `history.csv`);
   nothing is committed for the `trained-full` run that the paper reports.
10. **Reproducibility — PARTIALLY TRUE.** `codebase/environment.yml` exists ✔. Missing: seeds,
    cuDNN determinism, the winter split lists, the `trained-full` checkpoint/config/logs, and exact
    train/test commands (paths are hard-coded absolute paths on two different machines:
    `/home/dll0505/...` and `/home/dll1801/...`).

---

## 4. Additional issues not anticipated by the brief

**A1 — Qualitative figures use independent contrast stretches for prediction and ground truth.**
[infer_generate_optical.py](codebase/infer_generate_optical.py#L50-L82) and
[infer_generate_optical_cr.py](codebase/infer_generate_optical_cr.py#L47-L88) call `rgb_from_s2`
separately on the prediction and on the target; in the default `mode="global"` each image is
rescaled by **its own per-channel min/max** and gamma-corrected. Any global radiometric bias
(colour cast, over/under-estimation) is therefore removed before the figure is drawn, making the
outputs look better matched than they are. The same applies to the training monitor panels
(`_s2_to_rgb_percentile`, 2–98 % per-image stretch, [myutils.py](codebase/myutils.py#L285-L305)).
*Fix:* use one shared stretch computed from the ground truth and apply it to both. Effort: **S**.

**A2 — Metric scales depend on the test set.** `global_dr` / `band_dr` come from the test ground
truth (C5), so Tables 3, 5, 6, 7, 8 are each normalised by their own data, and cross-table
comparisons (notably Table 3 vs Table 5) are not strictly valid.

**A3 — The 60 m ablation partner model was trained without an LR scheduler** (see C4.4).

**A4 — Mid-training manual LR override on resume** (`resume_g_lr = 5e-5`) plus scheduler
re-creation, [L247-L268](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L247-L268).
Whether the published model went through a resume is not recorded.

**A5 — `epochs = 300` in the committed historical training script** vs "150 epochs" in the paper.

**A6 — Dead early-stopping branch** `if False and wait >= patience:`
([L560](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L560)) while CLAUDE.md documents
"early stopping patience: 20".

**A7 — LPIPS aggregation is an unweighted mean of per-batch means** (C5), biasing the last,
smaller batch.

**A8 — The evaluator runs the generator over the full test set three times** (metrics pass, SSIM
pass, normalised-space pass, L440/L484/L524) instead of once; harmless for correctness, 3× cost.

**A9 — `get_train_val_loaders` loads the whole `X_train.npy`/`y_train.npy` into RAM**
([myutils.py](codebase/myutils.py#L28-L46)) — the evaluator already uses `mmap_mode="r"`; the
trainer does not.

**A10 — Two incompatible generator configurations across scripts** (`norm="batch"` for the paper
model vs `norm="instance"` in the working-tree evaluator and `infer_generate_optical.py`). This
fails loudly rather than silently, but the working tree currently cannot evaluate the paper
checkpoint without editing the file.

**A11 — Winter split lists were never saved.** `save_split_csv` exists only in the current
all-seasons version of the prep script, so the exact Table 3/4/5 test set is not recoverable.

---

## 5. Summary table

| ID | Title | Status |
|---|---|---|
| C1 | Patch-level split / leakage | **WORSE THAN SUSPECTED** |
| C2 | SEN12MS-CR overlaps training footprints | **CONFIRMED** (exact % not measurable here) |
| C3 | Baselines under a different protocol | **CONFIRMED** |
| C4 | 60 m ablation is an averaging artifact | **CONFIRMED** (+ extra scheduler confound) |
| C5 | Metric implementation inflates numbers | **CONFIRMED** |
| C6 | Test-time BN/dropout mode | **RULED OUT** |
| M1 | Hidden augmentation | **RULED OUT** |
| M2 | Stated vs actual hyperparameters | **PARTIALLY TRUE** |
| M3 | Global [0,10000] normalization | **CONFIRMED** |
| M4 | Norm layers never varied/reported | **CONFIRMED** |
| M5 | No non-adversarial baseline | **CONFIRMED** |
| M6 | LPIPS circularity | **PARTIALLY TRUE** (vgg train / alex eval) |
| M7 | Loss ablation anomaly | **CANNOT DETERMINE** |
| M8 | Winter-only confounds | **CONFIRMED** |
| 1 | No statistics / no seeds | **CONFIRMED** |
| 2 | B8 = B8A PSNR | **CANNOT DETERMINE** |
| 3 | Checkpoint by val L1 | **CONFIRMED** |
| 4 | Scheduler asymmetry | **CONFIRMED** |
| 5 | SAR preprocessing gaps | **CONFIRMED** |
| 6 | Cloud-removal framing | **CONFIRMED** |
| 7 | No downstream validation | **CONFIRMED** |
| 8 | No uncertainty estimate | **CONFIRMED** |
| 9 | Unsupported BCE/instability claims | **PARTIALLY TRUE** |
| 10 | Reproducibility | **PARTIALLY TRUE** |

---

## 6. Paper statements that must change

1. **Split (missing entirely).** Add, and correct the design:
   > *Current:* no statement.
   > *Required:* "Patches were split 80/10/10. **In the reported experiments the split was performed
   > at patch level within the 46 winter ROIs, so training, validation and test patches originate
   > from the same scenes and, because SEN12MS tiles scenes with a stride of 128 px, adjacent
   > patches overlap by 50 %. The reported metrics are therefore optimistically biased.**"
   > After the fix: "Splits are disjoint at ROI/scene level (seed 0); file lists are released."

2. **§Materials and Methods, preprocessing.**
   > *Current:* "no data augmentation was applied" — **correct, keep.**
   > *Add:* "the generator is a U-Net-256 with BatchNorm and no dropout; the discriminator is a
   > 70×70 PatchGAN with BatchNorm."

3. **§Training.**
   > *Current:* "ReduceLROnPlateau (factor 0.5, patience 10)".
   > *Corrected:* "ReduceLROnPlateau on validation L1 (factor 0.5, **patience 20**), applied to the
   > generator only; the discriminator learning rate was held constant. Early stopping was not
   > used." Also state the actual epoch count and disclose the resume-time LR override if the
   > published checkpoint was produced by a resumed run.

4. **§Metrics.**
   > *Current:* "SSIM (11×11)".
   > *Corrected:* "SSIM via `skimage.metrics.structural_similarity` with the default 7×7 uniform
   > window, computed in reflectance (DN) space. Aggregate SSIM/PSNR use the global data range of
   > the reference set (≈10 000 DN); per-band values use the per-band range — the two are therefore
   > not directly comparable. PSNR is pooled over the whole test set with MAX = 10 000 DN."

5. **§Results, Table 3/4.**
   > *Add:* "Aggregate SSIM (0.888) is a 13-band multichannel SSIM at `data_range = 10 000`, while
   > the per-band values in Table 4 use per-band ranges; the mean of Table 4 is 0.880."

6. **§Cloud Removal, Table 5.**
   > *Current:* "the Pix2Pix model outperforms several specialized approaches, including DiffCR …
   > best-in-class PSNR (33.65 dB) and LPIPS (0.152)".
   > *Corrected:* "Under our evaluation protocol — 2,000 winter SEN12-MS-CR patches, our metric
   > implementation, MAX = 10 000 DN — the model attains PSNR 33.65 dB and SAM 3.86°. These values
   > are **not directly comparable** to the published baselines, which use the official
   > SEN12-MS-CR test split and their own metric code; no baseline was re-run here. DiffCR reports
   > a higher SSIM (0.902 vs 0.899). **Furthermore, the SEN12-MS-CR winter ROIs are the same
   > ROIs2017 scenes used for training, so these samples are not an independent test set.**"

7. **§Ablation, 60 m bands (Table 7).**
   > *Current:* "excluding the 60 m bands did not improve performance. Instead, radiometric
   > fidelity decreased … All models were trained using identical hyperparameters".
   > *Corrected:* "The 13-band and 10-band aggregates are averages over **different band sets**, and
   > SAM is computed over vectors of different dimensionality, so Table 7 does not support a
   > conclusion about the 60 m bands. Restricted to the ten common bands the two models differ by
   > ≈0.002 SSIM and ≈0.16 dB — within single-seed noise. In addition, the 10-band model was trained
   > **without the learning-rate scheduler**, so hyperparameters were not identical."

8. **§Results, B1 interpretation.**
   > *Current:* "B1 achieves the highest SSIM overall (0.9758) … the model can more easily recover
   > these spatially homogeneous signals".
   > *Corrected:* add that SAR carries no physical information on aerosols/water vapour/cirrus, that
   > SSIM on low-variance bands is largely determined by the stabilising constants, and report the
   > per-band skill relative to a constant/training-mean baseline before interpreting these values.

9. **§Results, NIR explanation.**
   > *Current:* "likely stem from the use of winter-only training samples, where vegetation-related
   > information captured by the NIR band is comparatively scarce".
   > *Corrected:* note that SEN12MS seasons are northern-hemisphere calendar seasons, so the subset
   > contains southern-hemisphere summer scenes; the claim requires hemisphere-stratified evidence.

10. **§Methods, loss motivation.**
    > *Current:* claims about BCE vanishing gradients and early instability.
    > *Corrected:* either remove, or support with the logged loss curves from an actual
    > `gan_mode="vanilla"` run (none exists in the repo).

---

## 7. Status of these findings in the current pipeline

| Paper-era ID | Still present in the current pipeline? |
|---|---|
| C1 split leakage | **Yes, unchanged** → Part I, S1 |
| C2 SEN12MS-CR overlap | Moot for now — the current run does not evaluate on SEN12MS-CR. Re-check before any new CR claim. |
| C3 baseline protocol | Moot until new CR numbers are produced |
| C4 60 m averaging artifact | **Yes** — the evaluator still has no common-band option |
| C5 metric inflation | **Yes, unchanged** → Part I, S7 |
| C6 eval mode | Ruled out in both |
| M1 hidden augmentation | Ruled out in both |
| M2 hyperparameters | Superseded — the current run emits `config.json` |
| M3 global normalization | **Yes** → Part I, S11e |
| M4 norm layers | **Worse** — silently switched to InstanceNorm → Part I, S5 |
| M5 no non-adversarial baseline | **Inadvertently satisfied** — the current run is effectively non-adversarial → Part I, S2 |
| M6 LPIPS circularity | Dormant — LPIPS loss is disabled |
| M7 loss ablation anomaly | Unresolved; blocked on seeds → Part I, S10 |
| M8 winter-only confounds | **Fixed** — the current dataset is season-stratified |
| minor 1 seeds | **Yes, unchanged** → Part I, S10 |
| minor 3 checkpoint by val L1 | **Degenerate** — four criteria collapse to one → Part I, S3 |
| minor 4 scheduler asymmetry | **Reversed** — G and D now decay together |
| minor 5 SAR preprocessing | **Yes, unchanged** |
| minor 6–8 CR blending / downstream / uncertainty | **Yes, unchanged** |
| minor 9 loss curves | **Fixed** — full `history.csv` + plots |
| minor 10 reproducibility | **Partly fixed** — `config.json` and split CSVs now exist; seeds and checkpoints still missing |

The actionable plan lives in **Part I, §I.13**. Nothing in this report has been applied; fixes
should land on a `review-fixes` branch.

---
---

# PART III — Addendum: the reviewer's N0–N10, re-mapped

> ### Scoping note
>
> The training loop the reviewer inspected — the one with `lambda_GAN`, `scheduler_D`,
> `compute_val_total`, `save_best_models("val_l1")`, `set_lrs_on_resume` and `decay_starting_epoch` —
> is [codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py),
> i.e. **the current implementation**, not the one that produced the paper. That is the right file
> to focus on going forward, but it means several N-statuses must be re-mapped, because the
> reviewer assumed it generated Tables 3–8.
>
> The paper's code path was
> [codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py)
> at commit `9730959` (`data_dir = "dataset_winter_all"`, `save_dir = "trained-full"`, SSIM 50 +
> LPIPS 50, 20-epoch warm-up, BatchNorm, no label smoothing).
>
> Below, each item is answered for **both**, labelled *(paper run)* / *(current run)*. Items that
> produced new findings for the current pipeline are cross-referenced into Part I.

## Status updates requested in Addendum §A

| ID | Reviewer's new status | Verified status | Note |
|---|---|---|---|
| C6 | Ruled out for validation, open for eval | **RULED OUT for both** | `model.eval()` at [evaluate…efficient.py L404](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py#L404), `torch.no_grad()` at L440/L484/L524; identical in the historical version. `use_dropout=False`. |
| C1 | Partially confirmed | **WORSE THAN SUSPECTED** | The test set *is* separate from `val_loader` (so that half of the hypothesis is ruled out), but **all three splits share 100 % of their scenes** — measured, see main report C1. |
| M2 warm-up | Ruled out | **RULED OUT — confirmed** | `g_only` skips D updates **and** sets `loss_G_GAN = 0` without querying D ([L297](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L297), [L365](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L365)). |
| Minor #4 | "Paper is wrong" | **Reviewer is wrong for the paper run** | See N1. |
| Minor #18 | Confirmed, possibly worse | **CONFIRMED** | See N2. |

---

## N0 Identify the actual code path used for the paper
**Status: CONFIRMED (resolved)**

Evidence — entry point: [codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py).
Complete dependency list:

| Concern | Source | Line |
|---|---|---|
| Dataset | `myutils.get_train_val_loaders` → `np.load(X_train.npy)` + `TensorDataset` (**no Dataset subclass, no transforms**) | [myutils.py L28-L48](codebase/myutils.py#L28-L48) |
| Generator | `mymodels.cp2pwrapper.Pix2PixWrapper` → stock `UnetGenerator` | [cp2pwrapper.py L41-L50](codebase/mymodels/cp2pwrapper.py#L41-L50) |
| Norm layer | stock `get_norm_layer`, wrapper default `norm="batch"` | [cp2pwrapper.py L29](codebase/mymodels/cp2pwrapper.py#L29), [L37](codebase/mymodels/cp2pwrapper.py#L37) |
| Discriminator | stock `models.networks.define_D(..., netD="basic")` | [train…checkpoints.py L115](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L115) |
| GAN loss | stock `models.networks.GANLoss(gan_mode="lsgan")` → `nn.MSELoss` | [networks.py L231-L232](codebase/mymodels/pytorch-CycleGAN-and-pix2pix/models/networks.py#L231-L232) |
| SSIM loss | `torchmetrics.image.ssim.StructuralSimilarityIndexMeasure(data_range=2.0)` | [L132-L134](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L132-L134) |
| LPIPS loss | `lpips.LPIPS(net="vgg")` | [L140](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L140) |
| Optimiser / scheduler | `torch.optim.Adam`, `torch.optim.lr_scheduler.ReduceLROnPlateau` | [L147-L150](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L147-L150) |

**Stock files actually used:** only `models/networks.py` (`UnetGenerator`, `get_norm_layer`,
`init_net`, `define_D`, `GANLoss`).
**Stock files NOT used:** `train.py`, `test.py`, `options/*.py`, `models/pix2pix_model.py`,
`models/base_model.py`, `data/*`. There is no `train_opt.txt` anywhere in the repo — hyperparameters
exist only as module-level literals (and, for the *latest* run only, in `config.json`).

**Norm layers.**
*(paper run)* G: **BatchNorm2d**, `affine=True, track_running_stats=True`
([networks.py L28](codebase/mymodels/pytorch-CycleGAN-and-pix2pix/models/networks.py#L28)).
D: **BatchNorm2d** (`define_D` default `norm="batch"`, not overridden).
*(latest run)* both **InstanceNorm2d**, `affine=False, track_running_stats=False`
([train…latest.py L184-L196](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py#L184-L196)).
**No spectral norm anywhere** — `grep spectral_norm` over `networks.py` returns nothing.

**Generator.** Stock `UnetGenerator` with `num_downs = int(log2(256)) = 8`, `ngf = 64` ⇒ structurally
identical to `unet_256`, but **`use_dropout=False`** (hard-coded,
[cp2pwrapper.py L47](codebase/mymodels/cp2pwrapper.py#L47)), so the stock inner-block dropout (p=0.5
on the three innermost blocks) is **absent**. 39,163,917 parameters.
**Discriminator.** `NLayerDiscriminator(n_layers=3)` = 70×70 PatchGAN
([networks.py L196](codebase/mymodels/pytorch-CycleGAN-and-pix2pix/models/networks.py#L196)),
`input_nc = 15`, 2,777,025 parameters.

Impact on reported results: none directly, but the paper must state BatchNorm + no dropout, and the
reviewer's assumption "stock dropout on" is wrong.

Proposed fix: emit a `config.json` from the paper script too (the newer script already does);
add the architecture details to §Materials and Methods.

Effort: **S**.

---

## N1 D learning rate is decayed, contradicting the paper
**Status: RULED OUT for the published run / CONFIRMED for the unpublished all-seasons run**

Evidence *(paper run)*: there is exactly one scheduler.
```python
scheduler_G = ReduceLROnPlateau(optimizer_G, "min", patience=patience, factor=0.5)
# Keep D LR fixed when controlling D frequency manually
```
[L150-L151](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L150-L151); the only step call
is `scheduler_G.step(val_loss)` [L483](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L483).
`scheduler_D` does not exist in that file. **The paper's "the discriminator's LR was kept constant"
is correct.**

Scheduler parameters actually used *(paper run)*: `ReduceLROnPlateau`, `mode="min"`, `factor=0.5`,
**`patience=20`** (`patience` is the module-level variable at
[L37](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L37)), `threshold` = PyTorch default
`1e-4`, `threshold_mode="rel"`, `cooldown=0`, `min_lr=0`. Driven by **validation L1**
(`val_loss = val_l1_sum / n_val`, [L481](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L481)).
There is no `use_lr_decay` / `scheduler_metric` / `decay_after_warmup_only` / `decay_starting_epoch`
knob in this script — those exist only in the newer one.

Evidence *(latest run)*: `scheduler_G.step(...)` **and** `scheduler_D.step(...)`
([train…latest.py L~700](codebase/train_pix2pix_lpips_ssim_generate_checkpoints_latest.py)), with
`use_lr_decay=True`, `lr_decay_patience=10`, `lr_decay_factor=0.5`, `lr_min=1e-6`,
`scheduler_metric="val_l1"`, `decay_after_warmup_only=True`, `decay_starting_epoch=20`.
Measured from the committed `history.csv` (100 epochs):
```
lr_G: 1e-4 → 6.25e-6   unique = [6.25e-06, 1.25e-05, 2.5e-05, 5e-05, 1e-04]
lr_D: 1e-4 → 6.25e-6   unique = [6.25e-06, 1.25e-05, 2.5e-05, 5e-05, 1e-04]
```
so yes, **`lr_D` did decay — but only in the run that is not in the paper.**

Impact on reported results: none. **But the paper's scheduler patience (10) is wrong; the code uses
20.** (Already logged as M2.)

**Verdict for the current run:** decaying G and D together is a reasonable design choice, not a
defect — and the logged schedule (four halvings, `1e-4 → 6.25e-6`, from epoch 20) behaved as
intended. The actual problem in that run is not the D schedule but **D saturation** (Part I, S2):
no learning-rate schedule can help a discriminator that already classifies every fake at 0.

Proposed fix: change the paper to "patience 20", or add `patience = 10` and retrain. If the newer
all-seasons run is ever published, its D-schedule must be described separately.

Effort: **S**.

---

## N2 The "best" checkpoint may come from the G-only warm-up
**Status: CONFIRMED (the risk is real and unguarded); the outcome CANNOT BE DETERMINED**

Evidence: the selection branch has no epoch guard —
[L541-L556](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L541-L556):
```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(netG.state_dict(), os.path.join(save_dir, f"{model_name}_best_netG.pth"))
```
`val_loss` is pure validation L1, and epochs `0…19` are G-only (pure L1+SSIM+LPIPS regression).
A pure-L1 regressor almost always reaches a **lower L1** than the same network once an adversarial
term is added, so the warm-up epochs are systematically favoured. If the minimum fell inside the
warm-up, the "cGAN" results in the paper would in fact come from a **non-adversarial** model. The
selected epoch is never logged.

Which checkpoint fed which table — **this part is determinable**: the paper script writes exactly one
generator file, `pix2pix_best_netG.pth` (best val L1), and every evaluation/inference script loads
that same name
([evaluate…efficient.py L~398](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py#L395-L400),
[infer_generate_optical_cr.py L29](codebase/infer_generate_optical_cr.py#L29)).
**So Tables 3–8 all use the lowest-validation-L1 checkpoint.** (The multi-criterion files
`*_best_val_ssim_netG.pth` etc. exist only in the newer script.)

What cannot be determined: `trained-full/` is not in the repo and there is **no `history.csv`, no
training log, and no checkpoint** for it — `.gitignore` excludes `*.pt`/`*.pth` and `*.npy`, and the
only tracked weights in the whole repository are three **discriminator** files from the all-seasons
run:
```
codebase/trained_all_seasons_debugged_loss_original_label_smoothing/pix2pix/pix2pix_best_netD.pth
codebase/trained_all_seasons_debugged_loss_original_label_smoothing/pix2pix/pix2pix_best_val_l1_netD.pth
codebase/trained_all_seasons_debugged_loss_original_label_smoothing/pix2pix/pix2pix_best_val_total_netD.pth
```
**No generator weights exist anywhere in the repository.** For the all-seasons run (`warmup = 0`,
so N2 does not apply there) the history gives `argmin val_L1 = epoch 57 / 100`, `val_total` minimum
also at epoch 57, `val_SSIM`/`val_LPIPS` columns empty (both losses disabled).

Impact: if the winter run's minimum fell at epoch ≤ 20, **the entire adversarial framing of the
paper is void**. This must be resolved from the original training log before publication.

**Verdict for the current run:** `warmup_g_only_epochs = 0`, so the warm-up hazard does not apply.
A *different* degeneracy has replaced it — with SSIM and LPIPS disabled, `val_total` collapses to
`100 × val_L1` and the four selection criteria reduce to one (Part I, S3). Keep the epoch guard
anyway: you will want the warm-up back once the adversarial term is revived (Part I, S2).

Proposed fix:
```python
can_select = epoch >= warmup_g_only_epochs
if can_select and val_loss < best_val_loss:
    ...
    logger.info(f"[BEST] epoch {epoch+1} val_L1={val_loss:.6f}")
```
plus persisting `best_epoch` into the checkpoint and into `config.json`.

Effort: **S** for the guard, **M** to recover/rerun and confirm the published checkpoint.

---

## N3 SSIM `data_range` in the loss / validation
**Status: RULED OUT — but it exposes a different, real inconsistency**

Evidence: `data_range` is supplied at **construction**, not at the call site, which is why it is not
visible in the loop:
[L130-L134](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L130-L134)
```python
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
ssim = StructuralSimilarityIndexMeasure(data_range=ssim_data_range).to(device)   # ssim_data_range = 2.0
```
with `ssim_data_range = 2.0` at [L62](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L62),
correct for [−1, 1]. The `pytorch_msssim` failure mode (`data_range=255`) does not apply — that
package is not used and is not in `environment.yml`.

Effective training-SSIM configuration (torchmetrics defaults, verified against
`torchmetrics/image/ssim.py`):
`gaussian_kernel=True`, `sigma=1.5`, `kernel_size=11`, `k1=0.01`, `k2=0.03`,
`reduction="elementwise_mean"`, `data_range=2.0`, applied to all 13 channels in NCHW.
So the training loss **is** an 11×11 Gaussian SSIM. `loss = (1 - ssim) * 50`
([L378-L379](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L378-L379)) — sign correct.

**The real inconsistency is train vs. eval:**

| | Training / validation loss | Reported metric |
|---|---|---|
| implementation | `torchmetrics` `StructuralSimilarityIndexMeasure` | `skimage.metrics.structural_similarity` |
| window | **11×11 Gaussian, σ = 1.5** | **7×7 uniform** (skimage default when `gaussian_weights=False`) |
| space | normalised [−1, 1] | reflectance / DN |
| data_range | fixed 2.0 | `global_max − global_min` of the test GT (≈10000), or per-band |
| averaging | mean over channels, mean over batch | multichannel mean (aggregate) / per band (Table 4, 8) |

The paper's "SSIM (11×11)" is therefore true of the **loss** and false of the **reported metric**.
This, plus the aggregate-vs-per-band `data_range` asymmetry, fully explains the 0.859 / 0.845
discrepancy (main report C4.3).

Two further notes:
- The `ssim` metric object is **never `.reset()`** and is shared between the training and validation
  loops. `Metric.forward()` returns the batch value (computed against a temporarily reset state) so
  the loss value is correct; only the unused global accumulator is polluted. Benign, but
  `ssim.reset()` per epoch would be cleaner.
- The requested "quick test" (same batch with and without `data_range=2.0`) **could not be
  executed**: `torch`, `torchmetrics`, `lpips` and `skimage` are not installed in this environment
  and the `.npy` datasets are gitignored. It is also now moot, since `data_range` is demonstrably set.

Impact on reported results: no bug in the SSIM loss ⇒ **M7 is not explained by a data_range bug**
(remains CANNOT DETERMINE). The paper's metric description must change.

Proposed fix: use one SSIM implementation for loss, validation and reporting (torchmetrics,
`data_range` fixed and explicit), or document both configurations precisely.

Effort: **S** (documentation) / **M** (unify + re-evaluate).

---

## N4 LPIPS inputs and configuration
**Status: CONFIRMED (range compression is real); band order RULED OUT as an issue**

**Range.** Measurement on real data was not possible (no runtime, `*.npy` gitignored), but the
mapping is closed-form: `to_tanh_range(DN, 0, 10000)` ⇒ `x = DN/5000 − 1`.

| Sentinel-2 DN (TOA ×10⁴) | value fed to LPIPS |
|---|---|
| 0 | −1.00 |
| 500 | −0.90 |
| 1000 | −0.80 |
| 2000 | −0.60 |
| 2500 | −0.50 |
| 10000 | +1.00 |

Typical cloud-free land reflectance in B2/B3/B4 is ≈300–2000 DN, i.e. **≈[−0.94, −0.60]** — about
**17 % of the nominal [−1, 1] range**. VGG/AlexNet features (after LPIPS's `ScalingLayer`, which
assumes [−1, 1]) therefore see a very low-contrast, strongly negative-biased image. The effective
gradient from `L_LPIPS` is correspondingly weak and biased towards the darkest decile of the
activation range. The same mapping is used at evaluation
(`scale01(x, 0, 10000)` then `*2 − 1`,
[evaluate…efficient.py L166-L200](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py#L166-L200)),
so train and eval are **consistent**; the problem is sensitivity, not a mismatch.

**Band order — correct.** The S2 stack is read band-order from the GeoTIFF, i.e.
`[B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12]` (confirmed by `BAND_NAMES` in
[evaluate…efficient.py L136](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py#L136)).
`rgb_indices = (3, 2, 1)` ⇒ `(B4, B3, B2)` = (R, G, B) ✔, and `select_rgb_from_nhwc` stacks them in
that order ([L286-L288](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L286-L288)).
The 10-band model correctly uses `(2, 1, 0)`
([train…60m_excluded.py L59](codebase/train_pix2pix_lpips_ssim_generate_60m_excluded.py#L59)).

**Backbone / freezing.** Training `lpips.LPIPS(net="vgg")`
([L140](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L140)); evaluation
`lpips.LPIPS(net='alex')` + explicit `.eval()`
([evaluate…efficient.py L371-L372](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py#L371-L372)).
In training, `.eval()` is **not** called explicitly — it relies on the package's `eval_mode=True`
default; and `lpips_fn` is never passed to any optimiser (`optimizer_G = Adam(netG.parameters())`,
[L147](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L147)), so it cannot be updated.
Gradients do accumulate into its linear heads and are never zeroed — wasted memory, not a
correctness bug. Add `lpips_fn.eval()` and `requires_grad_(False)` explicitly.

**Circularity (M6):** different backbones (vgg train / alex eval) ⇒ the strong form is avoided.

Impact: LPIPS numbers in Tables 3, 5, 6 are computed on heavily compressed inputs; they are
internally consistent but not comparable to LPIPS values from papers that feed full-range RGB.

Proposed fix: ablate an LPIPS-specific stretch (clip [0, 3000] DN → [−1, 1]) applied identically in
loss and metric; state both backbones and the stretch in the paper; add FID/KID.

Effort: **M**.

---

## N5 Two generator forward passes per iteration
**Status: CONFIRMED — and it matters more for the paper run than the reviewer assumed**

Evidence: [L326-L331](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L326-L331) and
[L360](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L360):
```python
with torch.no_grad():
    fake_B_detached = netG(real_A)     # D step
...
fake_B = netG(real_A)                  # G step
```
`d_steps_per_iter = g_steps_per_iter = 1`
([L48-L49](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L48-L49)), so this is a plain
duplicate pass.

Consequences, re-scoped:
- **Dropout masks:** no effect — `use_dropout=False`.
- **BatchNorm running statistics:** `netG.train()` is active during *both* passes, so BN running
  stats are updated **twice per optimisation step**. With `momentum=0.1` the effective per-step
  momentum becomes `1-0.9² = 0.19`. *(paper run uses BatchNorm ⇒ real effect; the latest run uses
  InstanceNorm with `track_running_stats=False` ⇒ no effect.)* This shifts the eval-time BN
  statistics relative to a single-pass schedule — i.e. it slightly changes what the reported model
  actually computes at test time.
- **Compute:** ≈1 extra G forward per batch (~25 % of step cost).

Impact: a small, uncontrolled perturbation of the eval-time BN statistics for every reported model.
Not an error that invalidates a table, but it is an undocumented deviation from stock pix2pix.

Proposed fix (separate branch — changes training):
```python
fake_B = netG(real_A)
# D step
pred_fake = netD(torch.cat((real_A, fake_B.detach()), -1).permute(0, 3, 1, 2))
...
# G step reuses fake_B
```
Effort: **S** to write, **L** to re-validate (requires retraining).

---

## N6 Undocumented / unverified hyperparameters
**Status: PARTIALLY TRUE — resolved where the source permits**

Resolved values for the **paper run** (all are module-level literals; there is no `config.json`
for `trained-full`, so these are the committed source values at `9730959`):

| Parameter | Value | Line |
|---|---|---|
| `lambda_GAN` | **not a variable** — the GAN term enters unweighted: `loss_G = loss_G_GAN + loss_G_L1 + loss_SSIM + loss_LPIPS` (⇒ effectively 1, matching Eq. 3) | [L411](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L411) |
| `lambda_L1` / `lambda_ssim` / `lambda_lpips` | 100 / 50 / 50 | [L55-L57](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L55-L57) |
| `d_steps_per_iter` / `g_steps_per_iter` | 1 / 1 ✔ | [L48-L49](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L48-L49) |
| `epochs` | `150` in the working tree, **`300` at `9730959`** | [L34](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L34) |
| `warmup_g_only_epochs` | 20 | [L47](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L47) |
| **total schedule** | `for epoch in range(start_epoch, epochs)` — **the warm-up is counted inside `epochs`** ⇒ 150 total = 20 warm-up + 130 adversarial, **not** 20 + 150 | [L295](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L295) |
| `start_epoch` | 0 unless a `checkpoint_last.pt` is found | [L211-L233](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L211-L233) |
| `lr` G and D | both `1e-4`; Adam `betas=(0.5, 0.999)`, **`weight_decay=0`** (default) | [L36](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L36), [L147-L148](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L147-L148) |
| `criterionGAN` | `GANLoss("lsgan")` → `nn.MSELoss`, targets **1.0 / 0.0 — no label smoothing** | [L121](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L121), [networks.py L231](codebase/mymodels/pytorch-CycleGAN-and-pix2pix/models/networks.py#L231) |
| `compute_val_total` | **does not exist** in the paper script — selection is pure val L1 | — |
| `batch_size` | 16 | [L35](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L35) |
| train loader | `shuffle=True`, `num_workers=16`, `pin_memory=True`, **`drop_last` not set ⇒ False** | [myutils.py L45](codebase/myutils.py#L45) |
| val loader | `shuffle=False` (default), `num_workers=16`, `pin_memory=True`, `drop_last=False` | [myutils.py L46](codebase/myutils.py#L46) |

For contrast, the **latest run** (`config.json`, committed): `lambda_GAN = 1.0`, `lambda_L1 = 100`,
`use_ssim = use_lpips = false`, `warmup_g_only_epochs = 0`, `epochs = 100`, `lr_G = lr_D = 1e-4`,
and `GANLoss("lsgan", target_real_label=0.9)` — **label smoothing is present there and only there**.

Impact: "20-epoch warm-up, then 150 epochs" overstates the adversarial budget by 20 epochs
(**130**, not 150), and the committed script says 300. The paper must state one number and it must
match the log.

Proposed fix: dump a `config.json` + resolved-args block at the start of every run and cite it in
the paper.

Effort: **S**.

---

## N7 Warm-up → adversarial transition
**Status: CANNOT DETERMINE**

Evidence: the transition is abrupt by construction — at `epoch == 20`, `g_only` flips to `False`
and in the *same* batch D takes its first update while G already receives
`criterionGAN(netD(fake), True)`. For the first D step of that batch, D is a randomly initialised
`init_net(..., 'normal', 0.02)` network, so G's adversarial gradient at that moment is noise.

What cannot be checked: `trained-full/` has no `history.csv` and no training log in the repo, so the
`val_L1` / `train_G_GAN` behaviour around epoch 20 is unobservable. The only committed history is
the all-seasons run, where `warmup_g_only_epochs = 0` ⇒ not applicable. That run's LR schedule
(`1e-4 → 6.25e-6`, four halvings over 100 epochs) is also not informative for this question.

Impact: the paper's claim that the warm-up "mitigates early instability" is asserted without any
committed evidence (also logged as minor item 9).

Proposed fix: commit `history.csv` for every run; optionally insert 1–2 D-only epochs at the
transition, or ramp `lambda_GAN` from 0 to 1 over ~5 epochs, and show the loss curves.

Effort: **S** (logging) / **M** (transition change + retrain).

---

## N8 Tensor layout (NHWC vs NCHW)
**Status: RULED OUT**

Evidence: the wrapper permutes in **and** out —
[cp2pwrapper.py L53-L69](codebase/mymodels/cp2pwrapper.py#L53-L69):
```python
def forward(self, x):            # x: (N, H, W, C_in)
    x = x.permute(0, 3, 1, 2)    # -> (N, C_in, H, W)
    logits = self.model(x)       # -> (N, C_out, H, W)
    return logits.permute(0, 2, 3, 1)   # -> (N, H, W, C_out)
```
so `netG(real_A)` correctly consumes NHWC and returns NHWC, matching `real_B` for
`criterionL1(fake_B, real_B)`. The D input is permuted at the call site
(`torch.cat((real_A, fake_B), -1).permute(0, 3, 1, 2)` — concatenating on the last axis **is** the
channel axis in NHWC ✔), and SSIM/LPIPS go through `nhwc_to_nchw`. The wrapper's own `__main__`
block already asserts the output shape
([cp2pwrapper.py L100-L104](codebase/mymodels/cp2pwrapper.py#L100-L104)).

Impact: none.

Proposed fix: none required; optionally promote the `__main__` shape assertion into a `tests/`
file so it runs in CI.

Effort: **S**.

---

## N9 Training-loop hygiene
**Status: CONFIRMED (several gaps)**

| Item | Finding | Evidence |
|---|---|---|
| `drop_last` | **Not set** (⇒ `False`) on either loader. With 25,600 train / batch 16 the last batch is full here, but for the winter run (31,825 patches × 0.8 = 25,460 ⇒ last batch of 4) BN statistics are computed on 4 samples in the final step of every epoch. | [myutils.py L45-L46](codebase/myutils.py#L45-L46) |
| `save_checkpoint` contents | `epoch, netG, netD, optimizer_G, optimizer_D, scheduler_G, best_val_loss, wait, history, config`. **No `scheduler_D`** (none exists), **no RNG states** (`torch`/`cuda`/`numpy`/`random`) ⇒ a resume is *not* bit-exact. | [L180-L209](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L180-L209) |
| Was the paper run resumed? | **Unknown and consequential.** `resume_with_halved_g_lr = True` + `resume_g_lr = 5e-5` + a **fresh** `ReduceLROnPlateau` (losing `num_bad_epochs`/`best`) fire automatically whenever `start_epoch > 0`. No log is committed to tell. | [L247-L268](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L247-L268) |
| Seeds / determinism | **None.** No `torch.manual_seed`, `np.random.seed`, `random.seed`, `torch.use_deterministic_algorithms`, or `cudnn.deterministic` anywhere in `codebase/*.py`. `torch.backends.cudnn.benchmark = True` is set, which selects kernels non-deterministically. | [L28](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L28) |
| `save_monitor_samples` | Calls `netG.eval()` + `@torch.no_grad()` but **does not restore the previous mode**. Harmless in practice: it is invoked after the validation block (where the net is already in `eval()`) and `netG.train()` is re-asserted at the top of the next epoch. | [myutils.py L307-L318](codebase/myutils.py#L307-L318), [train…checkpoints.py L516-L528](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L516-L528) |
| `monitor_batch_X/y` | From **`val_loader`** (first 8 samples) ✔ — not training data. | [L85-L88](codebase/train_pix2pix_lpips_ssim_generate_checkpoints.py#L85-L88) |

Impact: no single-run result is invalidated, but **nothing in this repository is bit-reproducible**,
and the possibility of an undocumented resume-with-LR-override sits directly under the published
checkpoint.

Proposed fix: a `set_seed(seed)` helper (torch/cuda/numpy/random + `cudnn.deterministic=True`,
`benchmark=False`), RNG states in the checkpoint, `drop_last=True` on the train loader, an explicit
`--resume` flag instead of implicit auto-resume, and `logger.info` of every resume event.

Effort: **S**.

---

## N10 Evaluation / test script
**Status: PARTIALLY RESOLVED — everything verifiable from source is verified; Table 3 cannot be reproduced**

Script: [codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py)
(10-band twin: `…_60m_execluded.py`).

| Question | Answer | Evidence |
|---|---|---|
| `netG.eval()` | **Yes** | [L404](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py#L404) |
| `torch.no_grad()` | **Yes**, all three passes | L440, L484, L524 |
| Checkpoint loaded | `pix2pix_best_netG.pth` = best-val-L1 (see N2) | L395-L410 |
| Split used | `X_test.npy` / `y_test.npy` — **different from `val_loader`** ✔ (but same scenes, see C1) | [L330-L340](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py#L330-L340) |
| Metric space | **Inverse-normalised DN**, never [−1, 1] (a separate normalised-space pass exists but only reports MAE/RMSE) | [L120-L132](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py#L120-L132) |
| PSNR `MAX` | `global_max − global_min` of the **test ground truth** ≈ 10000; **pooled** over the whole set, not per-image | [L469-L470](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py#L469-L470) |
| SSIM | `skimage`, **7×7 uniform**, `data_range=global_dr` (aggregate) / `band_dr[c]` (per band), per-image then averaged | [L473-L510](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py#L473-L510) |
| SAM | reflectance space ✔, `eps=1e-8` via `np.maximum`, `clip(cos, −1, 1)` ✔, mean over **all pixels**, degrees ✔ | [L203-L225](codebase/evaluate_pix2pix_clips_per_band_all_losses_efficient.py#L203-L225) |
| LPIPS | `net='alex'` (training used `vgg`), RGB = B4/B3/B2, unweighted mean of per-batch means | L371, L190-L200 |
| Tensors or images? | **Tensors.** `util.tensor2im` / `save_images` are never imported; no PNG round-trip in the metric path | whole file |

**Reproducing Table 3: not possible in this repository.** Three blockers: (1) no generator
checkpoint is tracked (`.gitignore` excludes `*.pth`; only three `netD` files from the all-seasons
run exist); (2) `*.npy` is gitignored, so `dataset_winter_all/X_test.npy` is absent; (3) `torch`,
`skimage`, `lpips`, `torchmetrics` are not installed in this environment. The arithmetic check still
holds: `20·log₁₀(10000) − 20·log₁₀(233.69) = 32.63 dB` reproduces the published PSNR exactly, which
independently confirms MAX = 10000 with pooled RMSE.

Impact: closes C6 (ruled out) and the "metrics on PNGs" worry (ruled out); confirms C5 in full.

Proposed fix: as in main report C5 — a single `eval_metrics.py` with fixed, documented ranges, a
`--seed` and a determinism assertion, plus committing the checkpoint (Git LFS or a release asset)
so the tables are reproducible.

Effort: **M**.

---

## Addendum summary table

| ID | Status | One-line verdict |
|---|---|---|
| **N0** | **CONFIRMED (resolved)** | Paper path = `train_pix2pix_lpips_ssim_generate_checkpoints.py`; only `models/networks.py` is reused from the stock repo; G and D are **BatchNorm**, no spectral norm, **no dropout**. |
| **N1** | **RULED OUT** *(paper run)* / **CONFIRMED** *(latest run)* | No `scheduler_D` exists in the published script — the paper is right. But scheduler **patience is 20, not 10**. |
| **N2** | **CONFIRMED (risk) / CANNOT DETERMINE (outcome)** | Best-val-L1 selection is unguarded against warm-up epochs; all tables use `pix2pix_best_netG.pth`; no log or checkpoint survives to say which epoch won. |
| **N3** | **RULED OUT** | `data_range=2.0` is set at construction; training SSIM is 11×11 Gaussian. The real problem is that the **reported** SSIM is skimage 7×7 at `data_range≈10000`. |
| **N4** | **CONFIRMED (range), RULED OUT (band order)** | LPIPS inputs occupy ≈[−0.94, −0.60], ~17 % of the nominal range; `(3,2,1)` = B4,B3,B2 is correct; vgg (train) vs alex (eval). |
| **N5** | **CONFIRMED** | Duplicate G forward ⇒ BN running stats updated twice per step (paper run only); no dropout effect. |
| **N6** | **PARTIALLY TRUE** | All values recovered. Key mismatch: total epochs **include** the warm-up ⇒ 130 adversarial epochs, not 150; committed value is 300; no label smoothing in the paper run. |
| **N7** | **CANNOT DETERMINE** | No `history.csv`/log committed for `trained-full`. |
| **N8** | **RULED OUT** | `Pix2PixWrapper.forward` permutes in and out; NHWC is consistent end-to-end. |
| **N9** | **CONFIRMED** | No seeds, `cudnn.benchmark=True`, no RNG state in checkpoints, `drop_last=False`, implicit auto-resume with a hidden LR override. |
| **N10** | **PARTIALLY RESOLVED** | Eval script verified line-by-line (eval mode ✔, tensors ✔, separate split ✔); Table 3 cannot be re-run — no checkpoint, no data, no runtime in this repo. |

## Additional paper sentences to correct (extending §6 of the main report)

11. **§Training, epoch budget.**
    > *Current:* "warmed up independently for 20 epochs … Subsequently, both networks were jointly
    > trained for 150 epochs."
    > *Corrected:* "trained for a total of 150 epochs, of which the first 20 were a generator-only
    > warm-up and the remaining 130 were adversarial."

12. **§Training, architecture.**
    > *Add:* "The generator is a U-Net with 8 down-sampling stages (`ngf = 64`, 39.2 M parameters),
    > BatchNorm and **no dropout** — unlike the stock pix2pix `unet_256`, whose three innermost
    > blocks use dropout. The discriminator is a 70×70 PatchGAN with BatchNorm (2.8 M parameters);
    > no spectral normalisation is used."

13. **§Training, reproducibility.**
    > *Add:* "No random seed was fixed and `cudnn.benchmark` was enabled; results are from a single
    > run and are not bit-reproducible." (Or fix it and remove the sentence.)

14. **§Metrics, SSIM.** (supersedes item 4) Distinguish explicitly between the **loss**
    (torchmetrics, 11×11 Gaussian, σ = 1.5, `data_range = 2` on [−1, 1] tensors) and the **reported
    metric** (skimage, 7×7 uniform, `data_range` from the test set in DN space).

15. **§Model selection.**
    > *Add:* "The reported model is the checkpoint with the lowest validation L1 over **all** epochs,
    > including the generator-only warm-up; the selected epoch is <N>." — and if that epoch is ≤ 20,
    > the adversarial claims must be withdrawn or the selection re-run with a warm-up guard.

## Revised priority order

Superseded by **Part I, §I.13**, which is scoped to the current pipeline. For the manuscript only:

1. **N2 + N0** — recover the `trained-full` training log and checkpoint; establish which epoch and
   whether it is adversarial. *Nothing in the paper's adversarial framing survives if the published
   model is a warm-up artefact.*
2. **N1 + N6 + N3** — correct the Methods/Metrics sections (patience 20, 130 adversarial epochs,
   two different SSIM configurations, BatchNorm / no dropout).
3. **C1 + C2** — state the leakage and the SEN12MS-CR overlap, or withdraw the affected numbers.
