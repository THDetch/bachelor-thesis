# Success criteria

Pre-registered before any arm was launched. `check_arms.py` applies these mechanically.

Shared config for every arm: seed 42, full current train split (25,697 patches),
lambda_L1 = 100, lambda_GAN = 1, LSGAN with D targets 0.9/0.0 and G target 1.0,
lr_G 1e-4, lr_D 5e-5, no scheduler, no warm-up, single G forward, 20 epochs,
checkpoint every 5 epochs, new run directory per arm, auto-resume off.

## Arms

| arm | discriminator | extra term | isolates vs |
|---|---|---|---|
| A_base | InstanceNorm + SN | none | reference |
| A0 | InstanceNorm + SN | mismatch | A_base -> conditioning |
| A1 | norm stripped, bias kept, SN | none | A_base -> normalization |
| A2 | D as A1 | instance noise | A1 -> noise |
| A3 | D as A1 | mismatch | A1 -> conditioning; A0 -> normalization |
| A3-g3 | D as A1 | mismatch, lambda_GAN 3 | A3 -> adversarial weight |
| A3-g10 | D as A1 | mismatch, lambda_GAN 10 | A3 -> adversarial weight |
| L1-only | D as A1 | mismatch, lambda_GAN 0 | A3 -> is the GAN doing anything? |

Every arm's D has bias on all 5 convs and spectral norm on all 5, and all five arms have
an identical parameter count of 2,777,025 - `InstanceNorm2d` is built with `affine=False`,
so replacing it with `Identity` removes nothing learnable. The startup assertions check
this, so each contrast above differs by exactly the one thing named.

## Criteria

| id | criterion | threshold | A_base (reference run) |
|---|---|---|---|
| S1 | D not saturated | mean `D_out_fake` >= 0.10 **and** mean `D_loss` >= 0.02, over epochs 11-20 | 0.0329 / 0.0162 |
| S2 | G_GAN not pinned | mean `G_GAN` <= 0.90, over epochs 11-20 | 0.9621 |
| S3 | no drift into collapse | mean `D_out_fake`(18-20) / mean `D_out_fake`(9-11) >= 0.50 | 0.302 |
| T | texture | mean `probe_hf_ratio`(18-20) >= A_base + 0.05 | 0.4347 (bar: 0.4847) |
| F | fidelity | mean full-val `val_L1`(18-20) <= 1.10 x A_base, same window | 0.07755 (limit: 0.08531) |
| ~~C~~ | redefined, see the 2026-11-14 amendment | raw shuffle margin | - |
| M | added 2026-11-14, see that amendment | raw real/fake margin | - |
| ~~U~~ | withdrawn, see the 2026-11-08 amendment | reported, not scored | - |

S1 and S2 use the training-time epoch averages in `history.csv`.
T and C use the 128 fixed validation patches, identical indices in every arm.
F uses the full validation split, unchanged from the reference run.

### Amendment, 2026-10-24

Decided after A_base finished and **before any treatment arm was launched**.

A_base reproduced the reference run's epochs 11-20 mean `D_out_fake` to within 0.07%
and hit `D_out_fake` < 0.05 in the same epoch, but individual epochs differ by up to
24%. The run is not bit-deterministic: `cudnn.deterministic=True` constrains cuDNN
convolutions only, not the other non-deterministic CUDA reductions in the graph.

Four criteria originally read a single epoch, which at that noise level is not a
stable estimator - S3 scored 0.367 on the reference run and 0.249 on A_base for the
same configuration, half the distance to its own threshold. The thresholds and
definitions are unchanged; only the estimator is now a windowed mean:

- S3: mean(18-20) / mean(9-11)
- T: mean(18-20), and the absolute 0.25 clause is dropped - A_base sits at 0.43,
  so every arm would have cleared it and the clause carried no information
- F: mean(18-20)
- C: per-patch scores pooled across epochs 18-20, then bootstrapped

C is evaluated against an absolute threshold, so it needs no A_base reference. A_base
itself ran before per-patch scores were persisted, so its C figure above is the mean of
the three per-epoch point estimates rather than a pooled bootstrap. It is 0.0218
against a 0.10 bar, so the estimator makes no difference to the verdict.

In A0 and A3 the S1 D-loss figure is `0.5 * (L_real + L_fake)` only; `L_mismatch` is
logged separately so the number stays comparable across arms. `D_out_fake` is always
the `(real_A, fake_B)` pair.

### Amendment, 2026-10-25: criterion U

Added **after A1 finished**, and applied retrospectively to every arm.

A1 passed S1, S2 and S3 while being the weakest arm in the set. S1-S3 were written to
detect a *saturated* D; they cannot detect the opposite failure, a D too constrained to
learn. A1's D improved only +4.0% at recognising real images over 20 epochs - a task
that does not depend on G at all - against +30.8% for A_base and +92.5% for A0, and its
outputs stayed inside a 0.04 band for the whole run. Meanwhile its generator's
high-frequency energy grew 132% while the correlation of that energy with the target
stayed flat at 0.02: more texture, none of it in the right place.

U tests that D is supplying a usable signal rather than merely an unsaturated one:

- `probe_hf_corr` - correlation between the prediction's and target's high-frequency
  residuals, land bands, sigma = 1. Distinguishes recovered structure from noise, which
  T's ratio cannot.
- `train_gan_l1_grad_ratio_param` - the GAN/L1 gradient ratio measured over G's
  parameters rather than at `fake_B`. What the optimiser actually steps on.

Both are referenced to A_base, since an absolute threshold for either would be
arbitrary. `probe_hf_corr` did not exist when A_base, A0 and A1 ran; `backfill_probe.py`
recomputes it from the epoch snapshots those runs saved. The back-filled figure is a
single epoch, not the 18-20 window, and `check_arms.py` labels it as such.

### Note for later: TTUR

An SN discriminator without normalization usually wants a higher `lr_D`. The current
5e-5 was chosen to handicap a D that was winning outright, which is the opposite of A1's
problem. If A3's D is also weak - watch `D_out_real` from epoch 1 to 20 - rerun A1 and
A3 with `lr_D = 2e-4` as A1b/A3b before concluding anything about normalization.

### Amendment, 2026-11-08: lambda_GAN grid, selection rule, criterion U withdrawn

A3 took 6 of 7 criteria and failed only U, on its gradient leg: 0.1891 against A_base's
0.3528. `lambda_GAN` is a more direct lever on that quantity than `lr_D`, because it
scales the adversarial gradient G receives rather than changing how fast D learns.

**Grid.** A3 config at `lambda_GAN` in {1, 3, 10}. A3 at 1 is already run.

**Selection.** Among runs passing S1, S2, S3 and F, take the highest mean
`probe_hf_ratio`(18-20). `probe_hf_corr` is reported beside it and is **not** a tie
breaker - across four arms it sat at 0.0158-0.0264, so it cannot separate them. This
matters: HF ratio measures how much high-frequency energy exists, not whether it is in
the right place, so the selected run must not be described as having better texture
unless `hf_corr` also moves.

**Fallback triggers**, defined before the grid runs:

- *saturation*: S1 fails with mean `D_out_fake`(11-20) < 0.10 -> retry at `lr_D` 2.5e-5
- *too weak*: `D_out_real` growth from mean(1-3) to mean(18-20) < 10% -> retry at
  `lr_D` 1e-4. For reference, A1 (frozen) grew 4.0%, A3 grew 31.4%, A0 grew 92.5%.

**Criterion U is withdrawn as pass/fail.** Its `hf_corr` leg does not discriminate. Its
gradient leg does - 0.179 to 1.032 across arms - and it is the quantity this grid
manipulates, so `train_gan_l1_grad_ratio_param` and `probe_hf_corr` are both still
reported by `check_arms.py`, without a verdict attached. Withdrawing a criterion after
seeing results is recorded here deliberately rather than applied silently.

**Control arm.** `L1-only` is the A3 configuration with `lambda_GAN = 0` exactly. D is
still built, still trained, and still probed - it observes without influencing G. This
keeps every metric computable and comparable instead of removing D and losing them, and
it answers whether the adversarial term contributes anything at all.

**Final runs** for whichever configuration wins: N = 50 epochs, constant LR for epochs
1-35, linear decay to zero over 36-50 with the same factor applied to G and D, last
checkpoint taken as the result.

### Amendment, 2026-11-14: C redefined, M added, T extended, L1-only comparison

Added **after A3-g3 and A3-g10 finished**, and applied retrospectively to every arm.
Recorded here rather than applied silently.

**C is redefined and the normalized form is dropped.** The old C used
`deg_shuffle = 1 - (d_shuf - d_fake) / (d_real - d_fake)`, which divides by the
real/fake margin. In A3-g10 that margin reached zero, so the ratio diverged: epoch 9
read -115, epoch 20 read -17.8, and the bootstrap interval spanned several hundred
units. A3-g3 was affected too - its epoch 12 margin of 0.0031 produced a point estimate
of 47. The metric was failing, not the discriminator. Note the algebraic identity
`deg_shuffle x margin = d_real - d_shuf`, so the numerator was always well behaved and
only the normalization was unstable.

- **C**: raw probe `D(real_A, real_B) - D(shuf_A, real_B)`, per-patch, pooled over
  epochs 18-20, bootstrapped. Pass when the lower CI bound > 0.

The absolute 0.10 threshold goes with the normalized form and is not carried over; a
raw margin in D-output units has no arbitrary bar to clear, only a sign.

Re-scored under this definition, A3-g10 passes C at 0.155 [0.142, 0.171]. Its earlier
"FAIL" was an artefact. A1 still fails, at -0.073 [-0.088, -0.059] - the sign is
genuinely negative there, which is the shortcut finding, not an estimator problem.

**M is new and is post-hoc.** S1 and S2 were written to catch a discriminator that
*wins*. Neither detects the opposite failure. A3-g10 posts the healthiest-looking
`D_out_fake` of any arm, 0.347, and passes S1 comfortably - while its held-out real/fake
margin is -0.0066 [-0.017, +0.004], indistinguishable from zero, negative in 8 of 20
epochs. `D_out_fake` being high and D working are not the same statement, and reading
only one of the two outputs cannot tell them apart.

- **M**: raw probe `D(real_A, real_B) - D(real_A, fake_B)`, per-patch, pooled over
  epochs 18-20, bootstrapped. Pass when the lower CI bound > 0.

M and C are the same quantity against two different negatives, which is the point: they
ask whether D separates *unreal* from real, and *unaligned* from real. One number was
being asked to carry both questions, and the division coupled them so a failure in one
corrupted the reading of the other.

*Caveat recorded deliberately.* A raw margin is not invariant to D's output scale, and
that scale is not constant across arms: `sd(d_real)` over epochs 18-20 falls 0.181 ->
0.149 -> 0.105 across A3 -> A3-g3 -> A3-g10. The rank form `AUC(d_real > d_shuf)` gives
0.839 / 0.817 / 0.844, i.e. conditioning unchanged, where the raw margin gives
0.238 / 0.182 / 0.155 and suggests a 35% decline. The two forms therefore order the arms
differently on C. The raw form is adopted for its directness and because it keeps the
units interpretable; the AUC form is reported alongside so the disagreement is visible
rather than buried. **No pass/fail verdict rests on the AUC column.**

**U stays withdrawn.** `train_gan_l1_grad_ratio_param` and `probe_hf_corr` continue to
be reported without a verdict.

**T is extended to report a native-resolution figure.** `probe_hf_ratio` averages the
ten land bands, four of which are 10 m and six of which are upsampled from 20 m. Those
six cannot carry real detail at 10 m, so including them dilutes exactly the quantity T
exists to measure. T's threshold and verdict are unchanged and still use the ten-band
figure; `probe_hf_ratio_10m` over B2, B3, B4, B8 (indices 1, 2, 3, 7) is reported
beside it.

*Back-fill limits, recorded so the gaps in the table are not mistaken for results.*
Checkpoints are written every 5 epochs, so no arm has snapshots at epochs 18 or 19 and
the pooled 18-20 window cannot be reconstructed for any completed run. A0, A1, A3,
A3-g3 and A3-g10 persisted per-patch probe scores while training, so M and C are exact
for them. A_base predates that and has none; its M and C are reported as unavailable
rather than estimated. `probe_hf_ratio_10m` likewise did not exist during any completed
arm and can only be recovered at epoch 20, and only while that arm's generator
checkpoint survives on the training machine.

**L1-only reports S1-S3, M and C but is not scored on them.** With `lambda_GAN = 0` the
discriminator is an observer: it is built, trained and probed, but its gradient never
reaches G. Its D statistics describe a D watching a generator it did not shape, which is
a useful reference point and not a pass/fail question.

The epoch-15 early abort (`D_out_fake < 0.02`) is disabled for this arm for the same
reason. G receives no adversarial gradient, so it never learns to fool D and
`D_out_fake` is expected to collapse; aborting there would truncate the 18-20 window
that every remaining criterion is measured over. The abort is unchanged for all
`lambda_GAN > 0` arms.

**Pre-registered comparison, A3-g3 vs L1-only.** Single seed, so indicative only; the
verdict is not final until confirmed on the 3-seed finals. A3-g3 is preferred **iff all
five hold**:

1. band-pass correlation on the native 10 m bands is higher,
2. KID is lower,
3. `|PSD ratio - 1|` averaged over 30-100 m is smaller,
4. NDVI and NDWI MAE are both within 5% of L1-only,
5. `val_L1`(18-20) <= 1.10 x L1-only.

This is a conjunction by design. The adversarial term has to show a texture gain that is
*in the right place* (1, 3), distributionally real (2), and free of a spectral-index
cost (4), at a fidelity price no worse than the tolerance already in F (5). Any single
one of these failing means the GAN is not earning its place.

**F's reference changes for the final runs.** F was defined against A_base, a
configuration that fails every stability criterion. L1-only is the correct fidelity
reference: it is the same generator and the same schedule with the adversarial term
switched off, so `1.10 x L1-only` prices exactly what the GAN costs. The arm table above
keeps the A_base limit, since those arms were scored against it.

*Not yet implemented.* KID, the band-pass correlation, the PSD ratio and the NDVI/NDWI
MAE are evaluation-time metrics computed after both runs finish; no code for them exists
at the time of writing. They are registered here before L1-only launches so the
comparison cannot be reverse-engineered from the result.

*Dependency.* All four are computed from generator outputs, so the comparison requires
A3-g3's epoch-20 `netG` to still be on the training machine. Checkpoints are not in the
repository. If it has been deleted, A3-g3 must be rerun at seed 42 before the comparison
can be made - the arm is reproducible to the windowed means but not bit-identical, so a
rerun is a new sample, not a recovery of the old one.

### Amendment, 2026-11-14 (second): verdict, T retired, finals changed

Written **after** the L1-only control finished and the registered primary was computed.
The verdict below follows the rule exactly as registered above; nothing was reweighted
after seeing it.

**Verdict (indicative): L1-only is preferred over A3-g3.** The primary fails.

| role | metric | A3-g3 | L1-only | |
|---|---|---|---|---|
| primary | band-pass corr, DoG sigma 2-8, native 10 m | 0.1481 | **0.1821** | FAIL, -18.7% |
| guard | `val_L1`, mean epochs 18-20, limit 1.05x | 0.07879 | 0.07513 | pass, limit 0.07889 |
| guard | NDVI / NDWI MAE and bias | - | - | not implemented |

**Indicative, not final.** One seed per arm, and the primary is a single epoch whose
trajectory scatters about 15% epoch to epoch against an 18.7% gap. The direction is
consistent across two independent scale bands and across the whole lambda ladder, which
is why it is reported at all; the 3-seed finals decide it. The `val_L1` guard at the
tightened 1.05x tolerance passes by **0.0001**, which is inside noise and should be read
as "not clearly worse" rather than as a pass.

The primary decides, so the pending NDVI/NDWI guard and the reported metrics (KID, PSD,
HF ratio, `hf_corr`) cannot reverse it. They are still worth computing to characterise
*how* the two differ, and that work is now descriptive rather than selective.

**The estimator is epoch 20, not the 18-20 window.** `checkpoint_every = 5`, so no arm
has snapshots at epochs 18 or 19 and the window cannot be reconstructed for a metric
added after the fact. It is the same single epoch for every arm, and the probe is
deterministic given the weights - the backfill reproduced every previously logged
epoch-20 figure to 0.0% - so there is no estimator noise. There is still trajectory
noise: `probe_hf_corr` scatters about 15% epoch to epoch. The -18.7% gap sits well
outside that, and the 3-seed finals will settle it properly. The `val_L1` guard does use
the registered 18-20 mean.

**The headline result.** Across lambda_GAN in {0, 1, 3, 10}, with the discriminator
configuration held fixed, band-pass energy rises and band-pass correlation falls,
monotonically, with no exception:

| lambda_GAN | 20-80 m ratio | 20-80 m corr | 10-30 m ratio | 10-30 m corr |
|---|---|---|---|---|
| 0 (L1-only) | 0.4929 | **0.1821** | 0.1416 | **0.0423** |
| 1 (A3) | 0.9312 | 0.1613 | 0.4410 | 0.0251 |
| 3 (A3-g3) | 1.1946 | 0.1481 | 0.6786 | 0.0206 |
| 10 (A3-g10) | 1.2069 | 0.1276 | 0.7911 | 0.0139 |
| | r = -0.925 | | r = -0.981 | |

Two independently defined scale bands, same sign, both near -1. Three further points:

- Correlation at 20-80 m is roughly 4x that at 10-30 m. SAR carries medium-scale
  structure and very little fine texture, which is what the corrected PSD says
  independently (20-35% of target power at 640 m, about 1% at 25 m).
- The adversarial penalty grows as the band gets finer: A3-g3 loses 18.7% at 20-80 m
  and 51.3% at 10-30 m; A3-g10 loses 29.9% and 67.1%.
- At lambda 3 and 10 the band-pass energy ratio **exceeds 1.0** (1.19, 1.21). Those arms
  emit more 20-80 m structure than the targets contain: over-generation.

**Correlation alone understates what the adversarial term contributes.** Decomposing the
predicted band-pass field into the part aligned with the target and the part orthogonal
to it, both as a fraction of the target's band-pass standard deviation:

| lambda_GAN | rho | ratio | correct = rho x ratio | wrong = sqrt(1-rho^2) x ratio |
|---|---|---|---|---|
| 0 (L1-only) | 0.1821 | 0.4929 | 0.0898 | 0.4847 |
| 1 (A3) | 0.1613 | 0.9312 | 0.1502 (+67%) | 0.9190 (+90%) |
| 3 (A3-g3) | 0.1481 | 1.1946 | **0.1769 (+97%)** | 1.1814 (+144%) |
| 10 (A3-g10) | 0.1276 | 1.2069 | 0.1540 (+72%) | 1.1970 (+147%) |

These are **standard-deviation (amplitude) units, not power**; squaring gives the
variance share, which is 0.008 for L1-only against 0.031 for A3-g3.

So the adversarial term does recover real structure - A3-g3 reproduces roughly twice as
much correctly oriented 20-80 m amplitude as L1-only, and the correct component **peaks
at lambda 3** before falling again at lambda 10. Correlation declines because the
orthogonal component grows faster than the aligned one, not because the aligned one is
absent. The honest statement is that the adversarial term adds both true and false
structure, and adds false structure faster.

A3 at lambda 1 reaches a band-pass energy ratio of 0.9312, the closest any arm gets to
the target spectrum, and still correlates worse than L1-only. Matching the energy
spectrum and being correct are separable, which is the direct reason T is retired below.

A_base scores second on the primary (0.1696) with a near-L1 energy ratio (0.5131),
consistent with its discriminator having saturated into irrelevance. It is corroboration
only: A_base, A0 and A1 differ from the lambda ladder in D configuration as well, so they
are not additional points on the same curve.

**Criterion T is retired as pass/fail.** T required `probe_hf_ratio` >= A_base + 0.05,
i.e. it scored high-frequency *energy*. L1-only fails T at 0.1542 while being the best
arm on the registered primary and on `val_L1`. Applied as written, the selection rule
"highest `probe_hf_ratio` among arms passing S1-S3 and F" would have chosen A3-g10 - the
arm with the lowest correlation at both scales and a discriminator at chance on M. T
selects for hallucination. The 2026-11-08 amendment already warned that a selected run
"must not be described as having better texture unless `hf_corr` also moves"; this is
that warning coming due.

T is replaced by:

- **T (new)**: `probe_bp_corr_10m`, DoG sigma 2 minus sigma 8, bands B2/B3/B4/B8.
  Reported for every arm. No absolute threshold - it is the primary of a pairwise
  comparison, not a bar to clear.
- `probe_hf_ratio_10m`, `probe_hf_corr_10m` and `probe_bp_ratio_10m` are reported
  alongside so that energy and correctness stay visible as separate quantities.

Retiring a criterion after seeing the result it decides is recorded here deliberately.
The verdict above does not depend on it: T was never part of the A3-g3 vs L1-only rule.

**Final runs change to two configurations, not one.** Originally the finals were "the
winning configuration". That no longer makes sense: the comparison *is* the result, and
a claim that the adversarial term does not help needs an interval on both sides of it.

- **L1-only and A3-g3, 3 seeds each, N = 50 epochs**, constant LR for epochs 1-35,
  linear decay to zero over 36-50 applied equally to G and D, last checkpoint as the
  result.
- Seeds 42, 43, 44, fixed in advance.
- `probe_bp_corr_10m` and `probe_bp_ratio_10m` are logged by the probe every epoch, so
  the finals get the registered 48-50 window directly from `history.csv`. No snapshot
  backfill is needed and `checkpoint_every` stays at 5.
- Pre-registered abort, carried over: if any A3-g3 seed's M falls to chance over its
  scoring window, that seed's texture numbers are reported as non-adversarial and
  excluded from any claim about what the discriminator contributed.

**SSIM is not a blocker.** It is inert as configured - 86-100% of 11x11 windows sit
below sqrt(C2), and the normalised and DN choices of `data_range` are the identical
constraint, so no faithful value fixes it. It is excluded from the final method rather
than repaired, and every arm ran with `use_ssim=False` and `use_lpips=False`, which
matches what the finals will run. Recorded as a limitation, not an open task.

### Amendment, 2026-11-14 (third): frozen spec for KID and the spectral indices

Written **before** either metric was computed, so the choices below cannot be tuned to a
result. Both are descriptive: the verdict above is already settled by the primary.

**Split.** Both are computed on the **full validation split**, not the 128-patch probe
and not the test split. The arms were selected on validation; test stays untouched until
the finals are reported.

**Checkpoint.** Epoch 20 `netG` for every arm, the same epoch the primary used.

**KID rather than FID.** FID estimates a 2048x2048 covariance; the splits here are a few
thousand patches, so it is strongly biased and N-dependent, and two runs scored on
different N are not comparable. KID is an unbiased MMD estimator with a usable spread
across subsets. FID is not reported.

- `torchmetrics.image.kid.KernelInceptionDistance`, `subset_size` and `subsets` written
  into the output JSON with the result.
- Reported as mean +- std across subsets. A bare point estimate is not reported.
- **Two channel triples**, because Inception takes three and the data has thirteen:
  true colour B4/B3/B2, and false colour B8/B4/B3. The second brings the NIR band into
  the measurement; all four bands are native 10 m. Seven of thirteen bands remain
  unmeasured by KID and that is stated with the number.
- 8-bit mapping from **training-split p2-p98 windows per band**, frozen in
  `kid_stretch.json` and never derived from validation or test. B4/B3/B2 reuse the
  windows already used for every figure in the thesis; B8 is added by the same recipe.
- Resize 256 -> 299 bilinear, `antialias=True`. The resampling choice measurably moves
  KID, so it is recorded rather than left to a default.

**NDVI and NDWI.**

- `NDVI = (B8 - B4) / (B8 + B4)`, `NDWI = (B3 - B8) / (B3 + B8)`. Indices 7, 3, 2 in the
  13-band layout; all three bands are native 10 m.
- Computed in **reflectance**, after the `S2_clip.json` inverse. A ratio is not invariant
  to the affine map to [-1, 1], so computing it in normalised space would give a
  plausible-looking number that means nothing.
- Denominator guard: pixels where the ground-truth sum falls below a fixed DN floor are
  excluded, and **the excluded fraction is reported**. Clamping silently is not an option
  because the clamp would land differently for a blurry and a sharp prediction.
- **MAE and signed bias are both reported.** MAE alone cannot distinguish a model that
  scatters around the truth from one that is consistently high or low, and an index
  built from a regression-to-the-mean predictor is expected to be biased toward zero.
- Reported **split by ground-truth water**, thresholded at NDWI > 0, as well as pooled.
  NDWI is most informative over water and SAR is most distinctive there, so a pooled
  mean would let a model that is right on land and wrong on water hide.

### Amendment, 2026-11-14 (fourth): cuDNN autotuning enabled for the final runs

Decided **before any final run started**, and applied uniformly to all nine.

`cudnn.deterministic=True` / `benchmark=False` were set to make config comparisons
exact. They never achieved that. Those flags constrain cuDNN convolution algorithms
only; the remaining CUDA reductions and atomics still accumulate in an unspecified
order, and float addition is not associative. The reproduction check in `check_arms.py`
measures the result directly - two A_base runs at seed 42, same code and data, differ
after a single epoch:

| metric | run 1 | run 2 | diff |
|---|---|---|---|
| `train_G_L1_raw` | 0.106306 | 0.106631 | 3.26e-04 |
| `train_D_out_fake` | 0.222166 | 0.218416 | 3.75e-03 |
| `train_G_GAN` | 0.690076 | 0.695282 | 5.21e-03 |

So the setting cost throughput for a guarantee that was never in force. It is now
`deterministic=False`, `benchmark=True`. Input shapes are constant - 256x256, batch 16,
`drop_last=True` - which is the case autotuning is designed for, worth roughly 10-25%.

Scope and cost:

- **All nine final runs use the new setting.** Mixing it within the finals would
  confound the comparison they exist to make.
- The seven 20-epoch arms ran with the old setting. Any numeric comparison between the
  arm study and the finals carries that caveat, though the effect is of order 1e-6 per
  operation, far below the seed-to-seed spread the finals are measuring.
- Conclusions rest on windowed means over the last three epochs, which reproduced to
  0.07% even under the old setting.
- AMP and `channels_last` were considered and rejected: both change numerics enough to
  plausibly affect GAN stability, which would make the finals test a different method
  than the arms did.


**Known untested factor.** The 100-epoch reference run ended at a +81% train/val gap
with val L1 flat from epoch 20. Augmentation is the one intervention never tried in this
project and it is not part of this grid. The N = 50 final runs will carry a substantial
gap; that is a limitation to state, not a result to explain away.

**SSIM caveat for reporting.** The evaluation script uses `data_range = 10000`, giving
sqrt(C2) = 300 DN against a typical local std near 100 DN, so 86-100% of SSIM windows
have their contrast and structure terms saturated to 1. Reported SSIM is close to a
luminance-only metric. Per-band-standardized SSIM and per-band-range PSNR columns are
added alongside the fixed-range ones rather than replacing them.

## Early abort

`D_out_fake` < 0.02 at epoch 15 means the arm has failed; the run stops there. An
aborted arm has no epochs 16-20, so S1, S2, S3, T, F and C are recorded as FAIL.

## A_base reproduction check

A_base must reproduce the reference run (`trained_baseline_scenesplit_sn`):

- epoch 1: max abs diff < 1e-3 on `train_G_L1_raw`, `train_D_out_fake`, `train_G_GAN`
- epochs 1-20: `D_out_fake` < 0.05 reached within +/-2 epochs of the reference, and
  epoch-20 values within roughly 10% relative

If it fails, report the library version delta and continue with the new A_base as the
reference, provided it still collapses qualitatively.
