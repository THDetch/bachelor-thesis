# Training Results Analysis: `trained_arm_A_base` (20 epochs)

## Executive Summary

Your refined cGAN pipeline trained for 20 epochs shows **stable L1 reconstruction improvement but concerning discriminator collapse**. The model is learning to reconstruct plausible outputs but is losing ability to synthesize fine details due to weak adversarial feedback.

---

## Reconstruction Metrics

### L1 Loss (Pixel-Level Accuracy)

| Metric | Epoch 1 | Epoch 20 | Change | Quality |
|--------|---------|----------|--------|---------|
| **Train L1 raw** | 0.1063 | 0.0721 | ↓32% | ✓ Good |
| **Val L1** | 0.0812 | 0.0744 | ↓8.4% | ✓ Acceptable |
| **Best Val L1** | — | 0.0706 (ep 17) | — | ✓ Epoch 17 |
| **Val L1 spike** | — | 0.0969 (ep 15) | — | ⚠ Outlier |

**Interpretation:**
- Training L1 declines smoothly and consistently (no oscillation)
- Validation L1 plateaus ~0.074 by epoch 10, then minor variance
- Epoch 15 spike suggests brief overfitting or validation set drift
- Overall: **reasonable reconstruction, but plateau suggests optimization saturation**

---

## Discriminator Health

### Raw Discriminator Outputs

| Component | Epoch 1 | Epoch 20 | Target | Status |
|-----------|---------|----------|--------|--------|
| **D(real)** | 0.675 | 0.882 | ~0.9 | ✓ Good |
| **D(fake)** | 0.222 | **0.0175** | ~0.3 | ✗ **CRITICAL** |
| **D_loss** | 0.108 | 0.0087 | Low | ✓ Good |

**Interpretation:**
- D(real) is appropriately high (discriminator recognizes real images)
- D(fake) **collapses to near-zero** by epoch 20
  - At epoch 1: D says 22% fake (discriminator is fooled)
  - At epoch 20: D says 1.75% fake (discriminator stops trying)
- This indicates **discriminator has given up or encountered saturation**

### Probe Diagnostics (128 Fixed Validation Patches)

| Metric | Epoch 1 | Epoch 20 | Interpretation |
|--------|---------|----------|-----------------|
| **probe_D_out_real** | 0.546 | 0.822 | Real detection stable |
| **probe_D_out_fake** | 0.088 | **-0.009** | Fake detection **fails** |
| **probe_hf_ratio** | 0.534 | 0.411 | High-freq content lost |
| **probe_std_ratio** | 1.216 | 0.574 | Texture variance lost |

**Interpretation:**
- Discriminator **cannot distinguish generated fakes** from real
- High-frequency (detail) content is **declining** over training
- Within-patch standard deviation **drops 53%** (epochs 1→20)
- **Conclusion: Network is smoothing details to fool weak discriminator, but discriminator stops engaging**

---

## Conditional Input Usage (SAR Conditioning)

### Probe Degradation Shuffle Test

**Question:** Does discriminator actually use the SAR input to make decisions?

| Metric | Epoch 1 | Mid (ep 10) | Epoch 20 | Interpretation |
|--------|---------|------------|----------|-----------------|
| **deg_shuffle** | -0.041 | -0.014 | +0.027 | Slowly increases |
| **95% CI low** | -0.094 | -0.004 | +0.003 | Uncertainty band |
| **95% CI high** | +0.007 | +0.065 | +0.053 | — |

**What this means:**
- **Negative values** = discriminator ignores/penalizes SAR input
- **Zero** = discriminator is agnostic to SAR condition
- **Positive values** = discriminator uses SAR input appropriately

**Observations:**
- Epochs 1-8: Discriminator **does NOT reliably use SAR** (near-zero/negative)
- Epochs 9+: Discriminator **learns to use SAR**, trending positive
- By epoch 20: **+0.027** (significant, within 95% CI)
- **Conclusion: Conditioning is being used, but the effect is weak compared to total discriminator signal**

---

## Gradient Dynamics

### GAN vs L1 Gradient Imbalance

| Component | Epoch 1 | Mid (ep 10) | Peak (ep 15) | Epoch 20 |
|-----------|---------|------------|--------------|----------|
| **Loss-level ratio** | 4.2× | 29.3× | 35.5× | 22.8× |
| **Parameter-level ratio** | 1.09 | 0.48 | 0.67 | 0.28 |

**Interpretation:**
- At the **loss level**, GAN gradients are 4-35× stronger than L1
- But at the **parameter level** (in the model weights), the ratio drops to 0.3-1.1
- **This suggests a gradient bottleneck:** The GAN signal is strong but doesn't propagate effectively into the generator's parameters

**Why this matters:**
- The L1 loss (weighted 100×) is dominating actual parameter updates
- GAN loss is too weak to compete despite large gradient magnitudes
- Result: Generator prioritizes pixel accuracy over adversarial realism

---

## Detail Generation Capacity

### High-Frequency Content Loss

**Probe Detail Ratios (128 fixed validation patches):**

```
land-band HF std ratio:    1.23 → 0.42 (↓66%)  [SEVERE]
within-patch std ratio:    1.23 → 0.57 (↓53%)  [SEVERE]
```

**What this means:**
- **land-band_HF_std_ratio < 1.0** = generator produces less high-frequency detail than target
- By epoch 20: Generator detail is only 42% of target detail
- Texture variation within patches drops by half

**Causes:**
1. Weak discriminator feedback (D(fake) ≈ 0) doesn't push for detail
2. L1 loss prefers smooth, averaged predictions (easier to minimize)
3. GAN loss insufficient to overcome L1's smoothing bias

---

## Training Stability

### Loss Trajectory

| Aspect | Status | Evidence |
|--------|--------|----------|
| **Convergence** | ✓ Stable | L1 decreases monotonically (no collapse) |
| **Discriminator collapse** | ✗ Yes | D(fake) → 0.0175 |
| **Overfitting** | ⚠ Minor | Val spike at ep 15, recovers by ep 17 |
| **Mode coverage** | ✗ Poor | Detail ratios declining |
| **Early abort trigger** | ✗ Not hit | D(fake) > 0.02 threshold (config: 0.02) |

---

## Per-Band Evaluation

**Status:** ❌ **No test set evaluation found in results directory**

No evaluation metrics on held-out test set (MAE, RMSE, SSIM, LPIPS per band).

**Recommendation:** Run inference + evaluation script to assess:
- Per-band MAE/RMSE (visual, red-edge, NIR, SWIR)
- SSIM (structural similarity)
- LPIPS (perceptual quality)
- Relative to baseline

---

## Key Findings Summary

### ✅ What Worked
1. **Stable training:** L1 loss decreased consistently without collapse
2. **Discriminator training:** No divergence or instability
3. **Conditioning:** Discriminator learned to use SAR input over time (+0.027)
4. **Reasonable convergence:** Plateau reached by epoch 10-12

### ❌ What Failed
1. **Discriminator engagement:** D(fake) collapsed to ~0, meaning adversarial signal is ineffective
2. **Detail synthesis:** High-frequency content dropped 53-66%, confirming smooth/blurry outputs
3. **Gradient propagation:** Parameter-level GAN/L1 ratio never exceeds 1.1, suggesting L1 dominates
4. **Test validation:** No per-band evaluation metrics to confirm reconstruction quality

---

## Interpretation: Why Output is Blurry

The visual observation (blurry middle sample) aligns with the metrics:

1. **Discriminator can't push for details** (D(fake) ≈ 0 means D isn't forcing realism)
2. **L1 loss prefers smooth predictions** (lower pixel-level error on averaged outputs)
3. **GAN loss is too weak** (parameter-level ratio ≈ 0.3) to overcome L1's averaging bias
4. **Result:** Network learns to output plausible *average* optical images rather than *sharp* details

---

## Recommendations for Next Training Run

### Immediate Fixes (High Priority)

1. **Reduce λ_L1 from 100 → 50-70**
   - Gives GAN loss more influence
   - Prevents excessive L1-driven smoothing

2. **Increase λ_GAN from 1.0 → 2-5**
   - Strengthens adversarial feedback
   - Target: Keep D(fake) > 0.05 (not < 0.02)

3. **Add spectral normalization to Generator** (currently only on D)
   - Stabilize generator gradient flow
   - May improve parameter-level gradient ratio

4. **Monitor D(fake) as early stopping signal**
   - Stop if D(fake) falls below 0.01 for 5+ epochs
   - Config already has `early_abort_d_out_fake: 0.02` but it didn't trigger

### Advanced Diagnostics

5. **Log layer-wise gradient norms**
   - Identify where gradients vanish between loss and parameters

6. **Use gradient penalty (R1 or WGAN-GP)**
   - Prevent discriminator collapse
   - Stabilize D(fake) trajectory

7. **Experiment with different loss ratios**
   - Try λ_L1=75, λ_GAN=1.0 first
   - If still blurry, increase λ_GAN to 2.0

### Validation

8. **Run full test set evaluation** after training
   - Per-band MAE, RMSE, SSIM
   - Compare to `trained_baseline_scenesplit_sn` if available

9. **Visual inspection across epochs**
   - Save sample outputs every 5 epochs
   - Identify where detail generation stops

---

## Conclusion

Your training pipeline is **functionally correct** but **adversarially weak**. The model learns a valid reconstruction task (L1 loss) but doesn't leverage the adversarial signal (GAN loss) to synthesize realistic details. 

**Next step:** Rebalance loss weights to strengthen GAN feedback. Current config (λ_L1=100, λ_GAN=1) is too conservative; the ratio of 100:1 is too extreme. Try 50:1 or 75:1 to allow adversarial learning to compete.
