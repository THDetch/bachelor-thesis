### Saliency on NIR/SWIR, spread across bands
Saliency is not on B1/B9/B10 (4.8% combined) and not on VV/VH (6.5%). It sits on the land bands: B8A 17.8%, B8 13.8%, B7 13.4%, B11 11.6% — NIR/SWIR alone is 56.6%. The ablation confirms it isn't any single band: best recovery 10% (B12), all thirteen together 40%. That's the "spread evenly over the land bands → blur detection" branch of their question.

D barely uses VV/VH (6.5%). It works almost like an unconditional texture detector and largely ignores whether the output matches the SAR input. That is expected while blur is the easiest cue, so don't act on it yet. Re-check saliency once training is stable. If D still ignores SAR then, the "conditional" part isn't contributing.

