# Discriminator diagnostics

Run `baseline_scenesplit_sn`, checkpoint_last epoch 100, val split, N = 512, seed 42, 1000 bootstrap resamples.


## 0. Checks

| check | value |
|---|---|
| harness, all bands spliced | 100.00% (must be 100%) |
| D(real) / D(fake) | 0.8821 / 0.0054 |
| Parseval, worst band | B9 at 11.8% (0.2532 vs 0.2871) |
| instance_noise_sigma0 | 0.017119 |

Parseval compares the variance ratio from the binned spectra against the one from pixels; the spectra are Hann-windowed, so a few percent is expected.

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

### 1b. Per-group blur (OOD)

| group | sigma | deg | 95% CI |
|---|---|---|---|
| VIS | 2.0 | 0.2% | [0.1%, 0.4%] |
| VIS | 4.0 | 0.5% | [0.2%, 0.9%] |
| VIS | 8.0 | 0.6% | [0.3%, 1.1%] |
| RE | 2.0 | 1.1% | [0.8%, 1.4%] |
| RE | 4.0 | 2.4% | [1.8%, 3.0%] |
| RE | 8.0 | 5.7% | [4.6%, 6.9%] |
| NIR | 2.0 | 9.4% | [7.7%, 11.4%] |
| NIR | 4.0 | 24.1% | [21.3%, 27.1%] |
| NIR | 8.0 | 35.8% | [32.6%, 39.1%] |
| SWIR | 2.0 | 0.2% | [0.1%, 0.3%] |
| SWIR | 4.0 | 0.2% | [0.1%, 0.4%] |
| SWIR | 8.0 | 0.3% | [0.2%, 0.4%] |
| ATM | 2.0 | 0.0% | [0.0%, 0.0%] |
| ATM | 4.0 | 0.0% | [0.0%, 0.1%] |
| ATM | 8.0 | 0.1% | [0.0%, 0.1%] |

Mixed real and blurred bands are out of distribution for D.

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

| band | std ratio (mean) | (median) | HF ratio | bias DN | sd(bias) DN | sd(target mu) DN | R2(patch mu) |
|---|---|---|---|---|---|---|---|
| B1 | 0.955 | 0.708 | 0.498 | -11.2 | 209.3 | 221.2 | 0.102 |
| B2 | 0.655 | 0.503 | 0.168 | -24.8 | 208.9 | 239.0 | 0.225 |
| B3 | 0.605 | 0.460 | 0.155 | -58.1 | 243.2 | 305.5 | 0.330 |
| B4 | 0.647 | 0.488 | 0.148 | -64.6 | 356.3 | 505.3 | 0.487 |
| B5 | 0.613 | 0.497 | 0.198 | -66.8 | 344.0 | 491.7 | 0.492 |
| B6 | 0.502 | 0.411 | 0.180 | -125.7 | 436.5 | 563.8 | 0.351 |
| B7 | 0.515 | 0.429 | 0.190 | -153.9 | 531.9 | 683.8 | 0.344 |
| B8 | 0.504 | 0.418 | 0.158 | -136.4 | 494.8 | 657.8 | 0.391 |
| B8A | 0.524 | 0.447 | 0.203 | -145.1 | 563.8 | 755.8 | 0.407 |
| B9 | 0.985 | 0.690 | 0.400 | -111.9 | 340.6 | 416.3 | 0.258 |
| B10 | 0.870 | 0.738 | 0.630 | -20.2 | 45.1 | 46.2 | -0.141 |
| B11 | 0.718 | 0.530 | 0.230 | -50.0 | 618.1 | 824.7 | 0.435 |
| B12 | 0.745 | 0.515 | 0.198 | -67.2 | 521.4 | 712.6 | 0.456 |

R2 is of the per-patch band mean; sd(target mu) is the reference it is measured against. Std ratios are identical in DN and normalized units.

## 5. SSIM window sensitivity and local detail

| band | windows below sqrt(C2), target | prediction | median local std, target | prediction |
|---|---|---|---|---|
| B1 | 99.9% | 100.0% | 0.00140 | 0.00112 |
| B2 | 99.1% | 100.0% | 0.00491 | 0.00156 |
| B3 | 98.6% | 100.0% | 0.00704 | 0.00198 |
| B4 | 96.5% | 100.0% | 0.01305 | 0.00328 |
| B5 | 98.0% | 100.0% | 0.01043 | 0.00310 |
| B6 | 95.0% | 99.9% | 0.01810 | 0.00342 |
| B7 | 91.3% | 99.7% | 0.02210 | 0.00429 |
| B8 | 85.7% | 99.7% | 0.02873 | 0.00504 |
| B8A | 89.5% | 99.5% | 0.02361 | 0.00497 |
| B9 | 99.9% | 100.0% | 0.00345 | 0.00169 |
| B10 | 100.0% | 100.0% | 0.00024 | 0.00024 |
| B11 | 94.3% | 99.7% | 0.01811 | 0.00546 |
| B12 | 95.9% | 100.0% | 0.01614 | 0.00477 |

Fraction of 11x11 Gaussian windows (sigma 1.5) with local std below 0.06 in normalized units; sigma0 is the median of the land bands.

## 6. Power-spectrum ratio, PSD(pred)/PSD(target)

| group | 640 m | 113 m | 58 m | 39 m | 30 m | 24 m | 20 m |
|---|---|---|---|---|---|---|---|
| VIS | 0.207 | 0.026 | 0.006 | 0.005 | 0.006 | 0.005 | 0.000 |
| RE | 0.253 | 0.047 | 0.018 | 0.014 | 0.019 | 0.010 | 0.000 |
| NIR | 0.294 | 0.056 | 0.018 | 0.015 | 0.020 | 0.011 | 0.000 |
| SWIR | 0.352 | 0.072 | 0.034 | 0.031 | 0.018 | 0.006 | 0.000 |
| ATM | 0.157 | 0.402 | 0.055 | 0.103 | 0.067 | 0.027 | 0.000 |

ratio of mean spectra, DC bin excluded; per-band curves are in the JSON.
