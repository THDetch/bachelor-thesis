---
name: commit-review-fix
description: 'Stage, commit and push changes made while working through the code review in code_review_report.md. Use when the user says commit the review changes, push the fix, save this finding, commit S1/S2/C4 etc., or after applying any fix traced to a review finding ID. Produces a compact commit message that links the change back to its finding, the verification evidence, and the impact on results.'
argument-hint: 'Optional: the finding ID being committed, e.g. S1 or C5'
---

# Commit a review fix

Commits made during the SAR→optical code review must stay traceable: months later it
must be obvious *which finding* a commit addresses, *what* it changed, and *whether
previously reported results are invalidated by it*.

## When to use

- After applying a fix that maps to a finding in [code_review_report.md](../../../code_review_report.md)
- When the user says "commit this", "push the changes", "commit S1", "save this fix"
- Before switching to the next review item

Do **not** use for unrelated work (manuscript edits, figure tweaks). Those get a normal commit.

## Finding IDs

| Prefix | Meaning | Lives in |
|---|---|---|
| `S1`…`S11` | Current pipeline findings | Report Part I |
| `C1`…`C6`, `M1`…`M8` | Paper-era findings / manuscript errata | Report Part II |
| `N0`…`N10` | Reviewer addendum items | Report Part III |

## Procedure

### 1. Inspect what changed

```powershell
git status --short
git diff --stat
```

If unrelated files are modified, stage only the review-related ones. Never `git add -A`
without looking first.

### 2. Decide the scope

**One commit per finding.** If two findings were fixed, make two commits. If a single
finding touched several files, that is still one commit.

### 3. Write the message

Use this template. Subject line ≤ 72 chars, body lines ≤ 80.

```
review(<ID>): <imperative one-line summary>

Finding : <short title of the finding>
Problem : <one line — what was actually wrong>
Change  : <bullet or semicolon list of the concrete edits>
Verify  : <evidence it works — numbers, assertions, compile check>
Impact  : <results invalidated / none / retrain required>
Report  : code_review_report.md §<section>
```

Rules:
- `Impact` is the most important line. If previously reported numbers become invalid,
  say so explicitly — that is the line future-you will search for.
- `Verify` must contain real evidence (measured numbers, an assertion that now exists,
  a passing syntax check), not "looks fine".
- Use `review(chore)` for pure cleanup with no finding ID.

Worked example:

```
review(S1): split SEN12-MS by scene instead of by patch

Finding : S1 — patch-level split leaked scenes across train/val/test
Problem : split key was md5(patch filename), so all 252 scenes appeared in all
          three splits; SEN12-MS tiles overlap 50%, so test patches shared
          pixels with training patches
Change  : group patches by (roi_group, season, roi_id); greedy per-season quota
          fill assigns whole scenes; assert splits are disjoint; record scene
          lists in split_metadata.json; output_dir -> dataset_all_seasons_scenesplit
Verify  : dry run on committed file lists -> 192/30/30 scenes,
          80.30/9.81/9.89% of tiles, disjointness assertions pass
Impact  : INVALIDATES all previously reported metrics; data must be regenerated
          and the model retrained before any number is quoted again
Report  : code_review_report.md §I.2
```

### 4. Commit

Write the message to a temp file to keep the formatting, then commit.

**Do not use `Set-Content -Encoding utf8`** — on Windows PowerShell 5.1 it writes a
byte-order mark, which ends up as `´╗┐` at the start of the commit subject.

```powershell
$msg = @'
review(S1): split SEN12-MS by scene instead of by patch

Finding : ...
'@
$f = [IO.Path]::GetTempFileName()
[IO.File]::WriteAllText($f, $msg, (New-Object Text.UTF8Encoding $false))
git commit -F $f
Remove-Item $f
git log --oneline -1          # confirm no BOM in the subject
```

Avoid chained `-m` flags — they collapse the body formatting.
Keep the message ASCII; `§` and similar characters survive badly through the shell.

### 5. Push — confirm first

**Always ask the user before pushing.** `paper` is a shared branch.

```powershell
git log --oneline -1
git push
```

If the user wants isolation instead, offer `git switch -c review-fixes` before the push.
Never use `--force`.

## Checks before committing

- [ ] Python files touched still compile: `python -m py_compile <file>`
- [ ] No large artifacts staged — `.gitignore` excludes `*.npy`, `*.pt`, `*.pth`; verify
      with `git status --short` that none slipped in via `-f`
- [ ] No absolute machine paths or credentials introduced
- [ ] The report section referenced in `Report:` actually exists
- [ ] `Impact:` honestly states whether published numbers are affected

## Tracing back later

```powershell
git log --grep="review(" --oneline          # every review commit
git log --grep="review(S1)" -p              # one finding, with its diff
git log --grep="INVALIDATES" --oneline      # commits that broke reported results
```
