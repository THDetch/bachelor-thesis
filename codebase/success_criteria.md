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
