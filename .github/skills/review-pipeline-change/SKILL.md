---
name: review-pipeline-change
description: 'Adversarially review a code change made during the SAR-to-optical code review (see code_review_report.md). Use when asked to review, double-check, verify, or second-opinion a diff on the refine-pipeline branch, a training/evaluation/data-prep change, or an edit to the vendored pix2pix code. Verifies the change actually addresses its stated finding, hunts for regressions and silent behaviour changes, and checks whether previously reported results are invalidated.'
argument-hint: 'Optional: the finding ID or file under review, e.g. S2 or networks.py'
---

# Review a pipeline change

You are a second reviewer. Another agent made this change and claims it fixes something.
**Assume the claim is wrong until you verify it from the source.** Do not accept a summary
as evidence.

## Context

Repo: SAR→optical translation with pix2pix (Sentinel-1 → Sentinel-2, 13 bands).
Findings live in `code_review_report.md`:

| Prefix | Scope |
|---|---|
| `S1`…`S11` | Current pipeline — Part I |
| `C1`…`C6`, `M1`…`M8` | Paper-era / manuscript errata — Part II |
| `N0`…`N10` | Reviewer addendum — Part III |

Work happens on `refine-pipeline`. Commits follow the template in
[commit-review-fix](../commit-review-fix/SKILL.md).

## Procedure

1. **Get the diff.** `git diff` or `git show <sha>`. Read every hunk.
2. **Find the claim.** Which finding ID? Read that section of `code_review_report.md`.
3. **Verify the problem was real.** Re-read the *original* code. If the stated problem
   doesn't exist, the fix is noise or worse.
4. **Verify the fix addresses it.** Not "looks reasonable" — trace the actual code path.
5. **Hunt for regressions.** Walk the trap list below.
6. **Check the blast radius.** Does this invalidate previously reported metrics? Does any
   other script need the same change?
7. **Compile.** `python -m py_compile <each touched .py>`.
8. **Report** in the format at the bottom.

## Trap list — check every one

**Which script is this?** There are two training scripts. The paper's results came from
`train_pix2pix_lpips_ssim_generate_checkpoints.py` (winter, SSIM+LPIPS, BatchNorm,
20-epoch warm-up). The *current* pipeline is `train_pix2pix_lpips_ssim_generate_checkpoints_latest.py`
(all seasons, SSIM/LPIPS off, InstanceNorm, label smoothing). A change aimed at the wrong
one is a no-op.

**Vendored code.** Edits under `codebase/mymodels/pytorch-CycleGAN-and-pix2pix/` diverge from
upstream `junyanz/pytorch-CycleGAN-and-pix2pix`. Always flag them, and check the change
doesn't break other call sites in that repo.

**Ordering: `init_net` before `apply_spectral_norm`.** `spectral_norm` renames the parameter
to `weight_orig` and makes `weight` derived. Init applied after SN is silently discarded on
the next forward.

**`norm="none"` disables conv bias.** `get_norm_layer("none")` returns a plain function, not a
`functools.partial`, so `use_bias = norm_layer == nn.InstanceNorm2d` evaluates `False` in
`NLayerDiscriminator`. The convs end up with neither bias nor normalization.

**Auto-resume is unconditional.** `try_resume()` loads `checkpoint_last.pt` from `save_dir`
with no flag. A stale checkpoint where `epoch+1 >= epochs` makes `range(start_epoch, epochs)`
**empty** — the script "succeeds" having trained nothing. Any change to run config must also
change `save_dir` or disable resume.

**Shared `criterionGAN` leaks label smoothing into G.** If `target_real_label=0.9` and the same
object is used for `criterionGAN(pred_fake, True)` in the generator step, G is optimized toward
0.9, not 1.0. Smoothing belongs to D only.

**`val_total` degeneracy.** With `use_ssim=False` and `use_lpips=False`, `compute_val_total`
reduces to `lambda_L1 * val_L1`. "Best by val_total" and "best by val_L1" then select the same
epoch, and the `val_ssim`/`val_lpips` checkpoint branches never fire.

**Evaluation `data_range` is derived from the test set.** `global_max - global_min` of the test
ground truth feeds PSNR's MAX and the aggregate SSIM `data_range`, while per-band metrics use a
different per-band range. Two runs scored this way are not strictly comparable. Any metric change
must move toward fixed documented constants.

**SSIM: two different estimators.** Training loss uses torchmetrics (11×11 Gaussian,
`data_range=2.0`, on [−1,1]). The reported metric uses `skimage` (7×7 uniform, DN space). Both
are legitimate; conflating them is not.

**LPIPS backbones differ on purpose.** `vgg` for the training loss, `alex` for the metric. This
follows the library's own recommendation and avoids circularity. **Do not "fix" it** — flag it
if someone unifies them.

**Split must stay scene-disjoint.** `split_metadata.json` must contain `split_level: "scene"`,
and `prepare_data_sen_12_ms.py` must still assert the three scene sets are pairwise disjoint.
A patch-level split reintroduces ~50% pixel overlap between train and test.

**`init_net` hardcodes `net.to(0)`.** It ignores the configured `gpu_id`. Harmless at
`gpu_id = 0`, wrong otherwise.

**tanh can never equal a saturated target.** Target pixels at exactly ±1 (from clipping, or from
no-data mapped to the clip bound) are a permanent tell for D.

**Staging hygiene.** `.gitignore` excludes `*.npy`, `*.pt`, `*.pth`, `codebase/__pycache__/`.
Check nothing large was force-added, and that no absolute machine paths
(`/home/dll0505/...`, `/home/dll1801/...`) were introduced.

## Severity

| Level | Meaning |
|---|---|
| **BLOCKER** | Silently wrong results, or invalidates numbers without saying so |
| **MAJOR** | Fix doesn't address the finding, or introduces a regression |
| **MINOR** | Works, but unclear, undocumented, or incomplete |
| **NIT** | Style, naming, comments |

## Report format

```
## Verdict: APPROVE | APPROVE WITH CHANGES | REJECT

Finding claimed : <ID> - <title>
Problem real?   : yes / no  + evidence (file:line)
Fix correct?    : yes / no / partial + evidence (file:line)
Compiles        : yes / no

### Issues
| Severity | File:line | Issue | Suggested fix |

### Blast radius
- Previously reported metrics invalidated: yes / no / already stated in the commit
- Other scripts needing the same change: <list or none>

### Traps checked
<list the trap-list items that were relevant, and the outcome of each>
```

Be specific. "Looks fine" is not a review — cite file and line for every claim you make,
including the ones where you found nothing wrong.
