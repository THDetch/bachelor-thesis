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
| A2 | norm stripped, SN | instance noise | A1 -> noise |
| A3 | norm stripped, SN | mismatch | A1 -> conditioning |

## Criteria

| id | criterion | threshold | A_base (reference run) |
|---|---|---|---|
| S1 | D not saturated | mean `D_out_fake` >= 0.10 **and** mean `D_loss` >= 0.02, over epochs 11-20 | 0.0329 / 0.0162 |
| S2 | G_GAN not pinned | mean `G_GAN` <= 0.90, over epochs 11-20 | 0.9621 |
| S3 | no drift into collapse | mean `D_out_fake`(18-20) / mean `D_out_fake`(9-11) >= 0.50 | 0.302 |
| T | texture | mean `probe_hf_ratio`(18-20) >= A_base + 0.05 | 0.4347 (bar: 0.4847) |
| F | fidelity | mean full-val `val_L1`(18-20) <= 1.10 x A_base, same window | 0.07755 (limit: 0.08531) |
| C | conditional | per-patch `deg_shuffle` pooled over epochs 18-20, then bootstrapped: >= 0.10 and CI lower bound > 0 | 0.0218 |

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
