# SAR -> optical arm study: findings for review

Seven 20-epoch arms on `dataset_all_seasons_scenesplit`, seed 42, scene-disjoint split
(25,697 train / 3,138 val / 3,165 test). Generator identical in every arm
(`Pix2PixWrapper`, UNet-256, `norm="batch"`, no dropout, 39,168,269 params). Arms differ
only in the discriminator configuration and `lambda_GAN`.

All criteria, estimators and the comparison rule were registered in
`success_criteria.md` **before** the runs they decide. Amendments are dated and state
whether they were made before or after seeing the data they affect.

---

## 1. Arms

| arm | D normalization | mismatch term | lambda_GAN |
|---|---|---|---|
| A_base | InstanceNorm + SN | no | 1 |
| A0 | InstanceNorm + SN | yes | 1 |
| A1 | stripped, bias kept, SN | no | 1 |
| A3 | stripped, bias kept, SN | yes | 1 |
| A3-g3 | stripped, bias kept, SN | yes | 3 |
| A3-g10 | stripped, bias kept, SN | yes | 10 |
| L1-only | stripped, bias kept, SN | yes | **0** (D trains, never reaches G) |

Every D has bias on all 5 convs, spectral norm on all 5, and an identical parameter
count of 2,777,025, asserted at startup. `A2` (instance noise) was never run.

Shared: `lambda_L1=100`, LSGAN with D targets 0.9/0.0 and G target 1.0, `lr_G=1e-4`,
`lr_D=5e-5`, no scheduler, no warm-up, 20 epochs, `use_ssim=False`, `use_lpips=False`.

---

## 2. Pre-registered verdict

Rule, registered before L1-only launched: **primary decides; guards must not get clearly
worse; everything else is reported without a verdict.**

```
PRIMARY  band-pass corr, DoG sigma2-sigma8, native 10 m (B2,B3,B4,B8)
         A3-g3 0.1481  vs  L1-only 0.1821   (-18.7%)             FAIL
GUARD    val_L1, mean epochs 18-20, tolerance 1.05x
         A3-g3 0.07879 vs  limit   0.07889                       PASS by 0.0001
GUARD    NDVI MAE, tolerance 1.05x
         A3-g3 0.15294 vs  limit   0.14374                       FAIL
GUARD    NDWI MAE, tolerance 1.05x
         A3-g3 0.12230 vs  limit   0.11828                       FAIL

VERDICT (indicative)  L1-only preferred.
```

The primary alone settles it. Both index guards fail independently, so the conclusion
does not rest on the single-epoch primary or on the 0.0001 val_L1 margin.

---

## 3. Master table, epoch 20

`correct = rho x ratio`, `wrong = sqrt(1-rho^2) x ratio`, both in **standard-deviation
(amplitude) units** as a fraction of the target's band-pass std. Squaring gives variance
share. `val_L1` is the registered mean of epochs 18-20; everything else is epoch 20.

| arm | lam | BPcorr10m | BPratio10m | correct | wrong | HFcorr10m | val_L1 | KID true | KID false | NDVI MAE | NDWI MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **L1-only** | 0 | **0.1821** | 0.4929 | 0.0898 | **0.4847** | **0.0423** | **0.07513** | 0.1318 | 0.1225 | 0.1369 | **0.1127** |
| A_base | 1 | 0.1696 | 0.5131 | 0.0870 | 0.5057 | 0.0153 | 0.07755 | **0.0992** | 0.0927 | **0.1357** | 0.1151 |
| A3 | 1 | 0.1613 | 0.9312 | 0.1502 | 0.9190 | 0.0251 | 0.07903 | 0.1272 | 0.1033 | 0.1447 | 0.1167 |
| A0 | 1 | 0.1542 | 0.9086 | 0.1401 | 0.8977 | 0.0143 | 0.08488 | 0.1066 | 0.1011 | 0.1441 | 0.1216 |
| A3-g3 | 3 | 0.1481 | 1.1946 | **0.1769** | 1.1814 | 0.0206 | 0.07879 | 0.1082 | 0.0883 | 0.1529 | 0.1223 |
| A1 | 1 | 0.1437 | 1.0473 | 0.1505 | 1.0364 | 0.0232 | 0.08512 | 0.1207 | 0.1092 | 0.1429 | 0.1191 |
| A3-g10 | 10 | 0.1276 | 1.2069 | 0.1540 | 1.1970 | 0.0139 | 0.08178 | 0.1086 | **0.0779** | 0.1470 | 0.1228 |

Stability criteria (`check_arms.py`): A3 and A3-g3 pass all of S1, S2, S3, M, C.
A_base fails S1/S2/S3. A0 fails S1/S3. A1 fails C (`D(real)-D(shuf) = -0.0732`,
AUC 0.410 - its D treats SAR-aligned pairs as *more* fake). A3-g10 fails M
(`D(real)-D(fake) = -0.0066`, AUC 0.474, i.e. **at chance on held-out data**, negative in
8 of 20 epochs). L1-only is reported but not scored on S1-S3/M/C because its D is an
observer; for reference its observer D is the strongest in the set, M 0.5913 (AUC 0.987)
and C 0.4421 (AUC 0.960).

---

## 4. Headline: two monotone trends in lambda_GAN, pointing opposite ways

D configuration held fixed; only `lambda_GAN` varies.

| lambda_GAN | band-pass corr (20-80 m) | KID false colour |
|---|---|---|
| 0 (L1-only) | **0.1821** | 0.1225 |
| 1 (A3) | 0.1613 | 0.1033 |
| 3 (A3-g3) | 0.1481 | 0.0883 |
| 10 (A3-g10) | 0.1276 | **0.0779** |
| | monotone **worse** | monotone **better** |

This is the perception-distortion tradeoff traced empirically over a 10x range of one
knob, with a no-discriminator control at one end, measured on both axes in the same
experiment. It is not inferred from theory.

Supporting: ratio vs correlation across the ladder gives **r = -0.925** at 20-80 m and
**r = -0.981** at 10-30 m.

---

## 5. What the adversarial term actually does

Correlation alone understates it. Decomposing the predicted band-pass field:

| lambda_GAN | rho | ratio | correct | wrong |
|---|---|---|---|---|
| 0 | 0.1821 | 0.4929 | 0.0898 | 0.4847 |
| 1 | 0.1613 | 0.9312 | 0.1502 (+67%) | 0.9190 (+90%) |
| 3 | 0.1481 | 1.1946 | **0.1769 (+97%)** | 1.1814 (+144%) |
| 10 | 0.1276 | 1.2069 | 0.1540 (+72%) | 1.1970 (+147%) |

- The adversarial term **does recover real structure**: A3-g3 reproduces roughly twice
  as much correctly oriented 20-80 m amplitude as L1-only.
- The correct component **peaks at lambda 3** and falls at lambda 10.
- Correlation declines only because the orthogonal component grows faster.
- At lambda 3 and 10 the band-pass energy ratio **exceeds 1.0** (1.19, 1.21):
  over-generation, more 20-80 m structure than the targets contain.

Variance share of the correct component: 0.008 (L1-only) vs 0.031 (A3-g3).

---

## 6. Scale dependence

| | 20-80 m (DoG) | 10-30 m (sigma=1 high-pass) |
|---|---|---|
| L1-only correlation | 0.1821 | 0.0423 |
| A3-g3 vs L1-only | -18.7% | -51.3% |
| A3-g10 vs L1-only | -29.9% | -67.1% |

Correlation at 20-80 m is about 4x that at 10-30 m, and the adversarial penalty grows as
the band gets finer. This matches the corrected PSD measured separately: 20-35% of
target power at 640 m falling to about 1% at 25 m. **No arm recovers fine texture**;
the best 10-30 m correlation in the entire study is 0.0423.

---

## 7. KID and spectral indices (full val split, 3,138 patches, epoch 20)

Frozen spec: training-split p2-p98 windows from `saturation_train_25697.json`
(B2 741-2095, B3 528-2455, B4 306-3427, B8 186-4201 DN), 256->299 bilinear with
antialias, KID 100 subsets x 1000. Indices computed in reflectance after the
`S2_clip.json` inverse. Masked fraction 1.0e-06, i.e. the denominator floor excluded
effectively nothing.

**A3-g3 vs L1-only, lower is better on all rows:**

| metric | A3-g3 | L1-only | delta | better |
|---|---|---|---|---|
| KID true colour | 0.1082 | 0.1318 | **-17.9%** | A3-g3 |
| KID false colour | 0.0883 | 0.1225 | **-27.9%** | A3-g3 |
| NDVI MAE | 0.1529 | 0.1369 | +11.7% | L1-only |
| NDWI MAE | 0.1223 | 0.1127 | +8.6% | L1-only |
| NDVI MAE, land | 0.1448 | 0.1330 | +8.8% | L1-only |
| NDWI MAE, water | 0.2303 | 0.2013 | +14.4% | L1-only |

KID gaps are **5.4x** (true colour) and **6.4x** the combined subset standard deviation.
Not marginal.

**L1-only has the worst KID of all seven arms.** Blur is the least Sentinel-2-like
output in the set. That is the cost of preferring it and should be stated as plainly as
the win.

Water is hard for every arm: NDWI MAE over water is 0.20-0.25 against ~0.11 pooled, with
large negative bias throughout. The ground-truth water/land split was worth having; a
pooled mean hid it entirely.

**Why the blurriest model is the least biased on NDVI.** NDVI = (B8-B4)/(B8+B4) is a
ratio, so it is invariant to any error that scales both bands by the same factor:
`(k*B8 - k*B4) / (k*B8 + k*B4)` reduces exactly. Regression to the mean compresses B8
and B4 in the same direction and roughly proportionally, because both are driven by the
same land cover and are strongly spatially correlated, so most of L1-only's per-band
error is **common-mode and cancels in the index**. The adversarial arms instead add
texture that is generated per band and is not common-mode, so their error survives the
ratio. This is why L1-only can be the least accurate per band on high-frequency content
while being the least biased on NDVI (+0.0032 pooled, -0.0019 over land) - the two
facts are consistent, not in tension.

---

## 8. Caveats

1. **One seed per arm.** Finals are 3 seeds; nothing here has an interval across runs.
2. **The primary is a single epoch.** `checkpoint_every=5` means epochs 18 and 19 have
   no snapshots, so the registered 18-20 window cannot be reconstructed for a metric
   added after the fact. The probe is deterministic given weights (backfill reproduced
   every logged epoch-20 figure to 0.0%), so there is no estimator noise, but trajectory
   noise is about 15% epoch to epoch against an 18.7% gap.
3. **20 epochs, not 50.** A GAN plausibly needs longer for the adversarial term to earn
   its place. This is the strongest objection to the current verdict.
4. **SSIM and LPIPS were off in all seven arms.** This compares L1 vs L1+GAN, not the
   previously configured production pipeline. SSIM is separately known to be inert here
   (86-100% of 11x11 windows below sqrt(C2); the normalized and DN `data_range` choices
   are the identical constraint) and has been excluded from the final method rather than
   repaired.
5. **Distortion metrics structurally favour the L1 baseline.** L1 minimisation yields
   the conditional median, which wins correlation-type metrics by construction
   (Blau & Michaeli 2018). A GAN losing on them is expected. What is not merely a
   restatement of that theory: the over-generation above 1.0, the peak of the correct
   component at lambda 3, and the scale dependence of the penalty.
6. **KID measures 3 of 13 bands** per triple; 7 bands are never measured by it.
7. **Augmentation was never tested.** The 100-epoch reference run ended at a +81%
   train/val gap.
8. Criterion **T was retired after seeing the result it decides** (it scored
   high-frequency energy, which is anti-correlated with correctness; the registered
   selection rule would have chosen A3-g10, the arm with a chance-level discriminator).
   Recorded explicitly. The verdict does not depend on it - T was never part of the
   A3-g3 vs L1-only rule.

---

## 9. Observations that may deserve follow-up

- **A_base is quietly strong and nobody planned for it.** Best true-colour KID (0.0992),
  second-best band-pass correlation (0.1696), best pooled NDVI MAE (0.1357) and best
  NDVI MAE over land (0.1268) - while failing every stability criterion. Its D saturated
  early, so it received a brief adversarial push and then trained essentially as L1.
  Counter-evidence: worst NDWI bias over water (-0.1343). **Recorded as an observation
  only, not a candidate configuration** - it reached this position by its discriminator
  failing, which is not a mechanism that can be specified or reproduced on purpose.
- **The correct component peaks at lambda 3**, so a lambda in (1, 3) was never sampled
  and may be where the ratio of correct to wrong structure is best. Any such run would
  be **exploratory and labelled as such**, outside the registered grid.
- **An NDVI bias prediction failed.** Regression to the mean was expected to bias
  L1-only's indices toward zero; it is instead the least biased arm (+0.0032 pooled,
  -0.0019 land) against A0 at -0.0453 and A_base at -0.0312. The mechanism is the
  common-mode cancellation described in section 7.

---

## 10. Next step, as approved

**Framing.** The registered primary decides which configuration is preferred. KID versus
the index and correlation metrics is reported as the perception-distortion tradeoff. No
new primary is designated and neither side is promoted after the fact.

**Finals.** `L1-only`, `A3-g3` and `A3` (lambda 1, a pre-registered grid point), seeds
42/43/44 each, N = 50 epochs: constant LR for epochs 1-35, linear decay to zero over
36-50 applied equally to G and D, last checkpoint as the result.
`probe_bp_corr_10m` and `probe_bp_ratio_10m` are logged every epoch, so the finals get
the registered 48-50 window directly from `history.csv`.

Pre-registered abort carried forward: if any adversarial seed's M falls to chance over
its scoring window, that seed's texture numbers are reported as non-adversarial and
excluded from any claim about what the discriminator contributed.

**Test split is evaluated once**, after the finals are complete, and never used for any
selection.
