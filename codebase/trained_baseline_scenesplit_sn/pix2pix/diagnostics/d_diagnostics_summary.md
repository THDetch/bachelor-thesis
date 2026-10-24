# Discriminator diagnostics

Run `baseline_scenesplit_sn`, checkpoint_last epoch 100, val split, N = 512, seed 42, 1000 bootstrap resamples.

Harness (all bands spliced): 100.00%. D(real) 0.8821, D(fake) 0.0054.


## 1. Blur sweep, all bands (ID)

| sigma (px) | deg | 95% CI |
|---|---|---|
| 0.5 | 1.3% | [1.0%, 1.7%] |
| 1.0 | 20.4% | [18.1%, 22.9%] |
| 2.0 | 62.7% | [59.5%, 65.8%] |
| 3.0 | 87.7% | [85.9%, 89.6%] |
| 4.0 | 94.9% | [93.6%, 96.3%] |
| 6.0 | 97.7% | [96.7%, 98.7%] |
| 8.0 | 98.1% | [97.1%, 98.9%] |

Does not reach 99%; plateau 98.1%.

## 2. Moment matching (~ID)

| variant | rec / deg | 95% CI | clamped |
|---|---|---|---|
| level | 1.4% | [0.7%, 2.0%] | 0.26% |
| contrast | 4.9% | [3.8%, 6.0%] | 0.73% |
| level+contrast | 6.6% | [5.2%, 7.8%] | 0.31% |
| reverse_real_to_fake | 24.1% | [21.3%, 27.1%] | 0.00% |

Last row is the reverse direction and is a degradation, not a recovery.

## 3. SAR conditioning (ID)

| quantity | value |
|---|---|
| d_real_A_real_B | 0.8821 |
| d_shuffled_A_real_B | 0.8802 |
| d_real_A_fake_B | 0.0054 |
| d_shuffled_A_fake_B | 0.0209 |
| deg_shuffle | 0.2% [-0.1%, 0.6%] |
| deg_sar_mean_only | 0.1% |
| VV+VH saliency share | 1.3% [1.3%, 1.3%] |

SAR was deranged across scenes; 0 same-scene violations.

## 4. Band statistics (no discriminator)

| band | std ratio (mean) | (median) | HF ratio | bias DN | sd(bias) DN |
|---|---|---|---|---|---|
| B1 | 0.955 | 0.708 | 0.498 | -11.2 | 209.3 |
| B2 | 0.655 | 0.503 | 0.168 | -24.8 | 208.9 |
| B3 | 0.605 | 0.460 | 0.155 | -58.1 | 243.2 |
| B4 | 0.647 | 0.488 | 0.148 | -64.6 | 356.3 |
| B5 | 0.613 | 0.497 | 0.198 | -66.8 | 344.0 |
| B6 | 0.502 | 0.411 | 0.180 | -125.7 | 436.5 |
| B7 | 0.515 | 0.429 | 0.190 | -153.9 | 531.9 |
| B8 | 0.504 | 0.418 | 0.158 | -136.4 | 494.8 |
| B8A | 0.524 | 0.447 | 0.203 | -145.1 | 563.8 |
| B9 | 0.985 | 0.690 | 0.400 | -111.9 | 340.6 |
| B10 | 0.870 | 0.738 | 0.630 | -20.2 | 45.1 |
| B11 | 0.718 | 0.530 | 0.230 | -50.0 | 618.1 |
| B12 | 0.745 | 0.515 | 0.198 | -67.2 | 521.4 |

Ratios are per patch then averaged; std ratios are identical in DN and normalized units because the band scaling is affine.

## 5. SSIM window sensitivity

| band | windows below sqrt(C2), target | prediction |
|---|---|---|
| B1 | 99.9% | 100.0% |
| B2 | 99.1% | 100.0% |
| B3 | 98.6% | 100.0% |
| B4 | 96.5% | 100.0% |
| B5 | 98.0% | 100.0% |
| B6 | 95.0% | 99.9% |
| B7 | 91.3% | 99.7% |
| B8 | 85.7% | 99.7% |
| B8A | 89.5% | 99.5% |
| B9 | 99.9% | 100.0% |
| B10 | 100.0% | 100.0% |
| B11 | 94.3% | 99.7% |
| B12 | 95.9% | 100.0% |

Fraction of 11x11 Gaussian windows (sigma 1.5) with local std below 0.06 in normalized units.
