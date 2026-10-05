# GEMSDOE42 — multiscale worming × topological persistence for the DOE GEMS Prize

**One-line:** a fault candidate must survive **two independent notions of scale** —
Hornby–Boschetti–Horowitz multiscale worming (upward continuation) **and**
scale-space topological persistence of gradient-magnitude ridges — on the
magnetic (RTP) and gravity (isostatic anomaly) layers. Per-pixel score =
(continuation steps survived) × (topological birth–death range), normalised
to [0,1], emitted as dotted off-catalogue submission.

**Primary submission (UNSCIRED — no organiser score exists for any file here):**
`docs/downloads/gemsdoe42-h42-1-wormpersist-20261005T230000Z-d1db2d77-zeros.tif`
— 30,363 dots, 0 on-catalogue, all 12,279,160 cells finite in [0,1].
Download it from the [site front page](docs/index.html) or the
[executive summary](docs/executive-summary.html).

## Core Values (read every session)

- **Maximize P(Win).** Every decision weighs tradeoffs, assesses risk, and
  chooses the path that maximizes the probability of winning. Every weekly
  submission slot is an experiment, not a lottery ticket.
- **Own the Outcome.** We own results end to end. Every claim carries an
  evidence class; defects are measured, published, and fixed — never quietly
  dropped. Failure and success are signals used to improve.

## Standing project prompt (read every session — this is what we are building)

1. **MUST GENERATE A UNIQUE TIF SUBMISSION.** Never copy a previous
   submission (prior artifacts are for learning/education only). The site must
   offer an easy one-click submission `.tif` exactly as the competition
   requires, obvious on the first screen.
2. **Method for this repo:** multiscale worming plus topological persistence,
   built from cross-scale stability, not any single-scale feature: apply
   Hornby, Boschetti & Horowitz (1999) worming to the magnetic and gravity
   layers, tracking which edges survive across upward continuation; separately
   compute persistent homology on the same layers' gradient-magnitude surfaces
   across a full sweep of smoothing scales; multiply into one persistence
   score per pixel (continuation steps × birth–death range); normalise to
   [0,1]; write the required format; confirm low correlation against every
   prior submission's raw output before download. Persistence output must be
   sparser and more spatially concentrated than a single-scale gradient map.
3. **Hypotheses:** before spending submission slots, generate 3–5 candidate
   geological hypotheses not tried yet, each naming layers, physical
   signature, why it catches off-catalogue faults, and how it differs from
   prior work; rank by expected DTI improvement vs implementation cost;
   validate the top candidate on the spatially-blocked holdout before touching
   a slot. External data must be free, official, and verified obtainable.
4. **No hallucinations; verify line by line** from official, verified, trusted
   sources; provide links for manual review; work autonomously; flag
   irregularities for review.
5. **Submission must satisfy the portal:** single-band float32 GeoTIFF,
   EPSG:32611, 100 m, 3730×3292, values in [0,1] — permanently fixing the
   `"Predicted values must be in range [0,1]"` error (both the float32
   sentinel mechanism and the NaN-nodata mechanism). Give every submission a
   unique name plus a short note for the form.
6. **Site:** clean, user-friendly GitHub Pages site with an executive-summary
   subpage explaining exactly how to submit, plus all evidence with official
   verified links. Target: beat the public leader (0.3262 on 2026-10-05).
7. **Multi-pass execution:** Pass 1 implement + verify; Pass 2 review for
   bugs/edge cases and fix; Pass 3 re-check against the request and polish.
   Then open a PR and merge to main, with suggestions and limitations noted.

## Quickstart

```bash
# 1. data (git mirrors, sha256-pinned; no DrivenData login needed)
bash scripts/restore_data.sh        # -> data/bridge/*, data/external/*

# 2. build the submission pair (caches intermediates in work/)
/home/user/pyenv42/bin/python scripts/build_submission.py --budget 40000

# 3. preregistered blocked holdout (validates the emission rule)
/home/user/pyenv42/bin/python scripts/validate_holdout.py

# 4. tests (metric vs published worked example + brute force, worming,
#    persistence, submission re-read audit, repo hygiene)
/home/user/pyenv42/bin/python -m pytest tests/ -q
```

Environment: Python 3.11 + `requirements.txt` (`numpy scipy rasterio
scikit-image matplotlib pytest`). GPU not needed — measured on CPU: full
pipeline ~53 s cold / ~19 s warm; holdout ~63 s; tests ~2 s.

## Competition (verified links)

- Overview: <https://www.drivendata.org/competitions/306/competition-doe-gems/>
- Problem / metric / format (p.967):
  <https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/>
- About / resources (p.968):
  <https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/>
- Leaderboard:
  <https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/>
- Rules PDF (NLR 96647): <https://docs.nlr.gov/docs/fy26osti/96647.pdf>
- Reference solution: <https://github.com/drivendataorg/gems-prize-reference-solution>
- GDR 1391 INGENIOUS regional compilation (DOI 10.15121/1881483):
  <https://gdr.openei.org/submissions/1391>

Metric: distance-weighted Tversky index, triangular kernel R=300 m,
α=0.2/β=0.8. Worked example TPw=3.00/FPw=1.89/FNw=2.00 → 0.6027 (page rounds
to 0.60) — reproduced in `tests/test_metric.py`.

## Repo map

| path | what |
|---|---|
| `src/gems42/grid.py` | grid facts (EPSG:32611, 3730×3292, footprint 5,167,373) + IO |
| `src/gems42/metric.py` | exact official DTI (soft + binary + brute-force) |
| `src/gems42/worming.py` | FFT upward continuation + worm picking + survival |
| `src/gems42/persistence.py` | scale-space maxima tracking with the elder rule |
| `src/gems42/emission.py` | fusion (worm × topo), [0,1], dotted thinning |
| `src/gems42/submission.py` | zeros/nan pair writer + 12-point audit |
| `scripts/build_submission.py` | end-to-end pipeline + uniqueness + sparsity gates |
| `scripts/validate_holdout.py` | preregistered 4-fold blocked holdout (H42-PREREG-1) |
| `scripts/restore_data.sh` | byte-pinned data restore via git mirrors |
| `docs/` | GitHub Pages site; `docs/downloads/` = submission files |
| `knowledge/` | literature + data-source research notes with links |
| `work/` | git-ignored intermediates, priors, receipts (local only) |

## Verification status (2026-10-05)

- [x] Metric: worked example + fast-vs-bruteforce agreement (`pytest tests/`)
- [x] Worming: synthetic-dyke tests (continuation smooths; worms hug contact)
- [x] Persistence: synthetic-blob tests (strong ridge outlives weak bump)
- [x] Submission: 12-point audit by re-reading written bytes; rebuild is
  bit-identical (sha256 `5d4913b0…` zeros, `1fea0f04…` nan)
- [x] Uniqueness: max Pearson 0.0053 / Jaccard 0.0061 vs 24 prior raw outputs
- [x] Sparsity: 30,363 dots vs 775,106 single-scale top-15% px (25× sparser)
- [x] Holdout H42-PREREG-1: PRIMARY (off-catalogue SGMC) +0.0125 mean, 4/4
  folds over single-scale baseline at matched mass → **PASS / slot-eligible**
- [ ] Organiser score: none — UNSCORED until submitted

## Limitations & next steps

- The holdout's catalogue truth cannot reward genuinely new faults
  (documented as IR-42-PROXY-01); the SGMC off-catalogue instrument mitigates
  but does not eliminate this.
- NW quadrant holds 59% of dots (enrichment 1.25× over footprint share) —
  plausible (densest structure) but flagged as observation OB-42-01.
- Random control beats concentrated arms on dense SGMC truth in 2/4 folds —
  an instrument property (dense truth favours spread), documented in
  `docs/research.html`; the hidden set is ~5× sparser per live inversion.
- Next hypotheses queued: H42-2 (tilt-angle worms on TMI), H42-3 ( worm
  intersection nodes × MT conductance), H42-4 (2 m temperature-probe mask),
  H42-5 (paleo-geothermal prior on tips/step-overs) — see
  `docs/hypotheses.html`.
