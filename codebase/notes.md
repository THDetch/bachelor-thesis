### Saliency on NIR/SWIR, spread across bands
Saliency is not on B1/B9/B10 (4.8% combined) and not on VV/VH (6.5%). It sits on the land bands: B8A 17.8%, B8 13.8%, B7 13.4%, B11 11.6% — NIR/SWIR alone is 56.6%. The ablation confirms it isn't any single band: best recovery 10% (B12), all thirteen together 40%. That's the "spread evenly over the land bands → blur detection" branch of their question.

D barely uses VV/VH (6.5%). It works almost like an unconditional texture detector and largely ignores whether the output matches the SAR input. That is expected while blur is the easiest cue, so don't act on it yet. Re-check saliency once training is stable. If D still ignores SAR then, the "conditional" part isn't contributing.


Harness check, every band spliced: 100.0000%
D(fake) 0.0054   D(real) 0.8821   n = 512

(1) blur sweep, ALL bands [ID]
 sigma       deg               95% CI
--------------------------------------
   0.5     1.3% [   1.0%,   1.7%]
   1.0    20.4% [  18.1%,  22.9%]
   2.0    62.7% [  59.5%,  65.8%]
   3.0    87.7% [  85.9%,  89.6%]
   4.0    94.9% [  93.6%,  96.3%]
   6.0    97.7% [  96.7%,  98.7%]
   8.0    98.1% [  97.1%,  98.9%]
sigma* (deg >= 99%): None   plateau: 98.1%

   per-group blur [OOD]
 group  sigma       deg               95% CI
---------------------------------------------
   VIS    2.0     0.2% [   0.1%,   0.4%]
   VIS    4.0     0.5% [   0.2%,   0.9%]
   VIS    8.0     0.6% [   0.3%,   1.1%]
    RE    2.0     1.1% [   0.8%,   1.4%]
    RE    4.0     2.4% [   1.8%,   3.0%]
    RE    8.0     5.7% [   4.6%,   6.9%]
   NIR    2.0     9.4% [   7.7%,  11.4%]
   NIR    4.0    24.1% [  21.3%,  27.1%]
   NIR    8.0    35.8% [  32.6%,  39.1%]
  SWIR    2.0     0.2% [   0.1%,   0.3%]
  SWIR    4.0     0.2% [   0.1%,   0.4%]
  SWIR    8.0     0.3% [   0.2%,   0.4%]
   ATM    2.0     0.0% [   0.0%,   0.0%]
   ATM    4.0     0.0% [   0.0%,   0.1%]
   ATM    8.0     0.1% [   0.0%,   0.1%]

(2) band statistics, per patch then averaged [no D involved]
 band  std ratio  (median)  HF ratio   bias DN  sd(bias)  SSIM off tgt    pred
--------------------------------------------------------------------------------
   B1      0.955     0.708     0.498     -11.2     209.3        99.9%  100.0%
   B2      0.655     0.503     0.168     -24.8     208.9        99.1%  100.0%
   B3      0.605     0.460     0.155     -58.1     243.2        98.6%  100.0%
   B4      0.647     0.488     0.148     -64.6     356.3        96.5%  100.0%
   B5      0.613     0.497     0.198     -66.8     344.0        98.0%  100.0%
   B6      0.502     0.411     0.180    -125.7     436.5        95.0%   99.9%
   B7      0.515     0.429     0.190    -153.9     531.9        91.3%   99.7%
   B8      0.504     0.418     0.158    -136.4     494.8        85.7%   99.7%
  B8A      0.524     0.447     0.203    -145.1     563.8        89.5%   99.5%
   B9      0.985     0.690     0.400    -111.9     340.6        99.9%  100.0%
  B10      0.870     0.738     0.630     -20.2      45.1       100.0%  100.0%
  B11      0.718     0.530     0.230     -50.0     618.1        94.3%   99.7%
  B12      0.745     0.515     0.198     -67.2     521.4        95.9%  100.0%

(3) moment matching [~ID]
               variant       rec               95% CI   clamped
----------------------------------------------------------------
                 level     1.4% [   0.7%,   2.0%]    0.26%
              contrast     4.9% [   3.8%,   6.0%]    0.73%
        level+contrast     6.6% [   5.2%,   7.8%]    0.31%
  reverse (real->fake)    24.1% [  21.3%,  27.1%]    0.00%   (deg)

(4) SAR conditioning [ID]
  D(real_A, real_B)        0.8821
  D(shuffled_A, real_B)    0.8802
  D(real_A, fake_B)        0.0054
  D(shuffled_A, fake_B)    0.0209
  deg_shuffle                0.2% [-0.1%,0.6%]
  deg_sar_mean_only          0.1% [-0.1%,0.2%]
  VV+VH saliency share       1.3% [1.3%,1.3%]

(5) splice-in and leave-one-out [OOD]
 band              splice-in          leave-one-out
----------------------------------------------------
   B1      1.6% [  0.9%,  2.4%]      0.6% [  0.3%,  1.0%]
   B2      2.0% [  1.1%,  3.0%]      0.9% [  0.6%,  1.2%]
   B3      2.4% [  1.5%,  3.5%]      0.4% [  0.1%,  0.7%]
   B4      5.1% [  3.8%,  6.6%]      1.0% [  0.7%,  1.4%]
   B5      0.1% [ -0.2%,  0.5%]      0.4% [  0.1%,  0.7%]
   B6     -0.2% [ -0.3%, -0.1%]      1.3% [  0.8%,  2.0%]
   B7      0.9% [  0.5%,  1.4%]      6.5% [  5.4%,  7.7%]
   B8     30.1% [ 27.7%, 32.5%]     33.3% [ 30.4%, 36.3%]
  B8A      0.3% [  0.0%,  0.5%]     26.3% [ 23.7%, 29.2%]
   B9     -0.3% [ -0.4%, -0.2%]      1.7% [  1.1%,  2.4%]
  B10     -0.1% [ -0.1%, -0.0%]      0.1% [  0.0%,  0.2%]
  B11      3.6% [  2.6%,  4.7%]      0.6% [  0.3%,  0.9%]
  B12      5.7% [  4.3%,  7.3%]      0.2% [  0.1%,  0.4%]
D map range   fake [-0.0469, 0.0511]   real [0.2940, 1.0853]

[ARM A_base] discriminator
NLayerDiscriminator(
  (model): Sequential(
    (0): Conv2d(15, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (3): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (4): LeakyReLU(negative_slope=0.2, inplace=True)
    (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (7): LeakyReLU(negative_slope=0.2, inplace=True)
    (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))
    (9): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (10): LeakyReLU(negative_slope=0.2, inplace=True)
    (11): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))
  )
)
[ARM] conv weight std (init N(0, 0.02), applied before SN): 0.0203, 0.0199, 0.0200, 0.0200, 0.0200
[ARM] D convs 5, norm modules 3, params 2,777,025; G BatchNorm layers 13  OK
[PROBE] 128 fixed val patches, 29 scenes, 0 same-scene SAR violations
[RESUME] Disabled by config (allow_resume=False). Starting fresh.
Training:   0%|                                                                                                 | 0/20 [00:00<?, ?epoch/s]
Epoch 1:   6%| | 90/1606 [00:15<04:06,  6.14batch/s, G=14.774, D=0.179, D(f)=0.445, GAN=0.531, L1=14.243, SSIM=off, LPIPS=off, lrG=1.00e-0
[training_0:python*                                                                                          "DEGGS111PC16" 19:10 24-Oct-26