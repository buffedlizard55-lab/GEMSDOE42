# GEMSDOE42 — cross-scale worming × topological persistence

A unique, format-verified submission for the
[U.S. DOE GEMS Prize Challenge](https://www.drivendata.org/competitions/306/competition-doe-gems/)
(DrivenData competition 306) — geological fault prediction in the GeoDAWN study area of the
northwestern Great Basin, Nevada.

**The submission file, the "how to submit" walkthrough, and every number quoted below live on the
GitHub-Pages site:** [`docs/index.html`](docs/index.html). The site is generated from
`registry/*.json` by `scripts/build_site.py`, so it cannot drift from what was actually built.

---

## Read this first: what is and is not established

This repository is deliberately blunt about its own validation, because the honest answer changes
what you should do with the file.

**Established.** The shipped GeoTIFF passes 11 independent format checks, re-audited from disk
(`scripts/audit_shipped.py`): single-band `float32`, `EPSG:32611`, 100 m, 3730 × 3292, every value in
`[0, 1]`, NaN outside the survey footprint. It is genuinely unique — maximum |Pearson| against all
12 restored prior GEMSDOE submissions is **0.0248**. The metric implementation is checked against
the organiser's own worked example. The emission budget is chosen from a model fitted to 11 real
(score, size) pairs.

**Not established.** That this beats 0.2778 or the current leader's 0.3262. Two validation
instruments were built, and then both were **calibrated against 11 prior submissions whose official
public scores are known**. The result is the most important finding in this repository:

| instrument | Spearman vs official leaderboard score | p | n |
|---|---|---|---|
| spatially-blocked catalogue holdout | **+0.087** | 0.800 | 11 |
| off-catalogue USGS SGMC faults | **+0.305** | 0.361 | 11 |
| **emitted pixel count alone** | **−0.907** | **0.0001** | 11 |

Neither instrument predicts the live board. Emission *size* does. The structural reason is stated
by the organisers themselves: the private test set is faults **not** in the catalogue, so hiding
catalogue faults measures close to the opposite of the skill being tested
([forum thread 11516](https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516)).

At 60,069 emitted pixels this submission scores 0.0407 (holdout) and 0.0856 (SGMC), against 0.0945
and 0.0950 for the family's best prior at the same size. On the better of two weak instruments it is
**below** the best priors. It is a well-founded independent candidate, not a predicted winner.

---

## The submission

| | |
|---|---|
| file | `docs/downloads/gems42-xscale-worm-persistence-<stamp>-nan.tif` |
| name for the form | `gems42-xscale-worm-persistence-<stamp>` |
| emitted pixels | 60,069 of 5,167,373 in-footprint (1.16 %) |
| on known catalogue faults | 0 (the organiser masks these, so they are excluded on purpose) |
| values | 0.0 … 1.0 |
| max \|Pearson\| vs any prior | 0.0248 |

A `0`-filled twin (nulls as `0.0` rather than `NaN`) ships alongside in case the form rejects NaN.

---

## Method

Two independent notions of "this is a real structure, not noise", computed from the same two
potential-field layers (`rtp` reduced-to-pole magnetics and `iso_grav_anom` isostatic gravity
anomaly) and multiplied.

**Stage A — multiscale worming.** Following [Hornby, Boschetti & Horowitz (1999),
*GJI* 137(1):175–196](https://doi.org/10.1046/j.1365-246x.1999.00788.x) and
[Archibald, Gow & Boschetti (1999), *Exploration Geophysics* 30(1–2):38–44](https://doi.org/10.1071/EG999038),
using the operational recipe stated verbatim in
[Horowitz (2018), Stanford Geothermal Workshop](https://pangea.stanford.edu/ERE/pdf/IGAstandard/SGW/2018/Horowitz.pdf):
upward-continue the field to 0–4000 m, mark local maxima of the horizontal-gradient magnitude at
each height as multiscale edges, and score each pixel by how many consecutive levels it survives —
allowing ±2 px lateral migration, because a dipping structure's edge *moves* under continuation.

**Stage B — topological persistence.** For each layer and each smoothing scale σ ∈ {0.8, 1.6, 3.2,
6.4} px, threshold the gradient-magnitude surface at 32 levels and run a union-find sweep over the
superlevel-set filtration. Every pixel records the birth and death of its dim-0 homology class; the
shipped map is the birth–death range.

> **A trap worth knowing.** Naive H0 persistence assigns the *largest* range to the background —
> the component that never dies is the one seeded by the global maximum, and it eventually swallows
> the whole grid. Left that way the detector is exactly inverted. `src/gems42/persistence.py`
> closes the surviving component at the filtration floor, and `tests/test_core.py` pins it.

**Stage C — the product** of the two, each mapped to [0, 1] by a robust percentile.

---

## Reproducing it

```bash
bash scripts/download_competition_data.sh   # or: python3 scripts/restore_data.py
python3 scripts/prepare_data.py
python3 scripts/run_pipeline.py             # stages A -> C, cached under .cache/stage/
python3 scripts/calibrate_instruments.py    # score BOTH instruments against 11 known scores
python3 scripts/fit_emission_model.py       # pick the emission budget from those 11 scores
python3 scripts/ship_submission.py --budget 60069 --separation 2 --which persistence
python3 scripts/audit_shipped.py            # independent re-check of the file on disk
python3 scripts/make_previews.py && python3 scripts/build_site.py
python3 -m pytest tests/ -q                 # 19 tests
```

Data cannot be pulled from the DrivenData data page without credentials, and none are requested or
stored. `scripts/restore_data.py` restores the corpus from SHA-256-pinned mirrors; every byte is
recorded in `registry/data_manifest.json`.

---

## Layout

| path | what |
|---|---|
| `src/gems42/layers.py` | grid + band I/O; masks the finite `-3.4e38` nodata sentinel |
| `src/gems42/worming.py` | Stage A, FFT upward continuation + edge survival |
| `src/gems42/persistence.py` | Stage B, dim-0 persistent homology + a brute-force reference |
| `src/gems42/metric.py` | the competition's distance-weighted Tversky index |
| `src/gems42/pipeline.py` | stage orchestration, caching, emission selection |
| `src/gems42/holdout.py` | instrument 1: quadrant-blocked catalogue holdout |
| `src/gems42/sgmc.py` | instrument 2: off-catalogue USGS SGMC faults |
| `src/gems42/emission_model.py` | analytic DTI vs emission-size model |
| `src/gems42/submission.py` | writer + 11 format checks |
| `registry/` | every machine-readable result; the site reads these |
| `docs/` | the GitHub-Pages site, including the downloadable TIF |
| `tests/` | 19 tests, including a literal transcription of the metric |

---

## Known irregularities

Eleven are logged with evidence in [`registry/irregularities.json`](registry/irregularities.json).
The four that changed a decision:

- **IRR-01** — the catalogue holdout does not rank candidates the way the leaderboard does
  (ρ = +0.087, p = 0.80). It is retained but demoted; it never selects a candidate alone.
- **IRR-02** — emission size dominates the score spread across the whole family
  (ρ = −0.907, p = 0.0001).
- **IRR-04** — `training_features.tif`'s nodata value is `-3.4028234663852886e+38`, a **finite**
  float, so `np.isfinite` does not exclude it. 3,061 pixels inside the footprint carry it; left in,
  they corrupt every filter downstream.
- **IRR-05** — the brief's citation conflates two different 1999 papers. Both are cited where each
  is actually used.

---

## The task brief

Reproduced verbatim in [`PROMPT.md`](PROMPT.md) and on the site at
[`docs/prompt.html`](docs/prompt.html), as the brief itself required.

---

## Sources

Every external claim is recorded with what was actually verified from it in
[`registry/sources.json`](registry/sources.json). Key links:
[problem description](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/) ·
[leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/) ·
[scoring clarification (organiser)](https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516) ·
[forum](https://community.drivendata.org/c/gems-prize-challenge/111)
