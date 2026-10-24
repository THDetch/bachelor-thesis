### Saliency on NIR/SWIR, spread across bands
Saliency is not on B1/B9/B10 (4.8% combined) and not on VV/VH (6.5%). It sits on the land bands: B8A 17.8%, B8 13.8%, B7 13.4%, B11 11.6% — NIR/SWIR alone is 56.6%. The ablation confirms it isn't any single band: best recovery 10% (B12), all thirteen together 40%. That's the "spread evenly over the land bands → blur detection" branch of their question.

D barely uses VV/VH (6.5%). It works almost like an unconditional texture detector and largely ignores whether the output matches the SAR input. That is expected while blur is the easiest cue, so don't act on it yet. Re-check saliency once training is stable. If D still ignores SAR then, the "conditional" part isn't contributing.


D(fake) -0.0093   D(real) 0.8737   n = 512
Harness check, every band spliced: 100.0%  (must be 100.0%)

(b1/b2/b3) sensitivity, splice-in recovery, leave-one-out drop  [95% CI]
 chan      x std   share |            splice-in |        leave-one-out
----------------------------------------------------------------------
   VV  6.552e-07   1.8% |                    - |                    -
   VH  4.494e-07   1.3% |                    - |                    -
   B1  4.443e-07   1.3% |    0.3% [ 0.1%, 0.6%] |    0.8% [ 0.6%, 1.2%]
   B2  9.296e-07   2.6% |    1.0% [ 0.5%, 1.5%] |    0.6% [ 0.4%, 1.0%]
   B3  1.299e-06   3.7% |    1.7% [ 1.0%, 2.5%] |    1.1% [ 0.7%, 1.5%]
   B4  1.860e-06   5.2% |    3.5% [ 2.5%, 4.6%] |    1.1% [ 0.7%, 1.6%]
   B5  2.219e-06   6.3% |    0.3% [ 0.1%, 0.6%] |    0.6% [ 0.3%, 0.8%]
   B6  3.324e-06   9.4% |    0.4% [ 0.3%, 0.6%] |    1.3% [ 0.8%, 1.8%]
   B7  4.738e-06  13.4% |    4.3% [ 3.6%, 5.0%] |    8.2% [ 7.0%, 9.5%]
   B8  4.343e-06  12.3% |   21.8% [19.6%,23.9%] |   34.1% [31.5%,36.8%]
  B8A  6.460e-06  18.2% |    1.1% [ 1.0%, 1.3%] |   29.2% [26.5%,32.1%]
   B9  8.912e-07   2.5% |    0.1% [-0.0%, 0.2%] |    1.2% [ 0.8%, 1.6%]
  B10  6.891e-08   0.2% |   -0.0% [-0.0%,-0.0%] |    0.1% [ 0.1%, 0.2%]
  B11  4.680e-06  13.2% |    5.7% [ 4.6%, 6.8%] |    0.5% [ 0.3%, 0.7%]
  B12  3.091e-06   8.7% |    7.2% [ 5.8%, 8.8%] |    0.2% [ 0.1%, 0.4%]

Sum of the 13 single-band recoveries: 47.3%  (C separate experiments, not a decomposition)

(b2c) cumulative splice-in, descending single-band order
  k  band added  recovery             95% CI
---------------------------------------------
  1          B8    21.8% [ 19.6%, 23.9%]
  2         B12    27.2% [ 24.9%, 29.5%]
  3         B11    31.4% [ 29.0%, 33.8%]
  4          B7    39.5% [ 36.8%, 42.3%]
  5          B4    41.3% [ 38.5%, 44.1%]
  6          B3    49.9% [ 46.8%, 52.8%]
  7         B8A    88.7% [ 86.9%, 90.3%]
  8          B2    92.0% [ 90.5%, 93.2%]
  9          B6    96.5% [ 95.6%, 97.3%]
 10          B1    97.6% [ 96.9%, 98.2%]
 11          B5    98.8% [ 98.3%, 99.1%]
 12          B9    99.9% [ 99.8%, 99.9%]
 13         B10   100.0% [100.0%,100.0%]

(b4) blur test on REAL targets: D output, and how far it falls toward fake
 group  sigma  D(blurred real)   toward fake             95% CI
----------------------------------------------------------------
   ALL    0.5           0.8600         1.5% [  1.1%,  2.0%]
   ALL    1.0           0.6351        27.0% [ 24.4%, 29.7%]
   ALL    2.0           0.2722        68.1% [ 65.4%, 71.0%]
   VIS    0.5           0.8729         0.1% [  0.1%,  0.1%]
   VIS    1.0           0.8719         0.2% [  0.1%,  0.3%]
   VIS    2.0           0.8701         0.4% [  0.2%,  0.6%]
    RE    0.5           0.8712         0.3% [  0.2%,  0.4%]
    RE    1.0           0.8630         1.2% [  0.9%,  1.6%]
    RE    2.0           0.8576         1.8% [  1.3%,  2.4%]
   NIR    0.5           0.8717         0.2% [  0.2%,  0.3%]
   NIR    1.0           0.8622         1.3% [  1.0%,  1.7%]
   NIR    2.0           0.8354         4.3% [  3.4%,  5.4%]
  SWIR    0.5           0.8729         0.1% [  0.0%,  0.1%]
  SWIR    1.0           0.8717         0.2% [  0.1%,  0.4%]
  SWIR    2.0           0.8714         0.3% [  0.1%,  0.4%]
D map range   fake [-0.2066, 0.1520]   real [0.4206, 0.9785]