# WPH-01 validation preregistration and run record

**Status: BLOCKED — not run.** This document records the holdout and promotion decision before candidate results exist. No official data, current-holdout incumbent raster, raw earlier submission TIFFs or prior fold report are in the checkout. No score, pass, correlation, or submission-slot decision is claimed.

## Candidate and frozen first-pass settings

- Candidate: `WPH-01`, potential-field worm survival multiplied by smoothing-scale 0D persistence of the gradient-magnitude surface, run separately on gravity and RTP magnetic layers and averaged after robust normalization.
- Potential-field continuation heights: 0, 100, 200, 400, 800, 1,600, 3,200 m.
- Gradient-surface Gaussian sigmas: 0.5, 1, 2, 4, 8, 16, 32 pixels (official grid resolution is 100 m, so these are nominal pixel scales, not geological depth estimates).
- Edge/worm definition: local maxima of the horizontal-gradient magnitude of the upward-continued field. Match across successive continuation levels within a 2-pixel Chebyshev neighborhood; record the longest consecutive survival run divided by the number of levels.
- Topology definition: 0D superlevel persistence on the 4-neighbor pixel graph of each smoothed gradient-magnitude surface. A local maximum's lifetime is its birth value minus its merge/death value; omit the essential component of each connected valid-mask component. Match persistence maxima between consecutive Gaussian scales within a 2-pixel neighborhood. The topology term is normalized lifetime times the longest consecutive scale-track fraction.
- Per-layer product: `worm_survival_fraction * topology_track_score`. Each product's positive in-footprint values are scaled by their 99.5th percentile (falling back to the positive maximum if needed); the two layers are averaged equally and clamped to `[0,1]`. No supervised label fitting is done by the detector. This positive-only scaling was fixed before any competition-data run so a sparse map does not collapse to all zeros merely because its full-footprint 99.5th percentile is zero.
- Magnetic transform caveat: use the documented RTP anomaly layer when available. This is an **RTP-field worming variant**, not an assertion that the full classic pseudogravity transform has been applied. A pseudogravity transform will not be invented without verified field/magnetization assumptions.

These settings are an initial preregistered configuration, not literature-prescribed optimal heights/scales. A scale/hyperparameter change after observing holdout results is a new experiment and must be recorded before rerunning.

## Spatial split and score protocol

1. Read the official `sample_submission.tif` grid, training feature grid and known-fault raster; require exact alignment or perform and document a deterministic reprojection to the sample grid. Record source checksums and metadata.
2. Freeze four contiguous spatial folds from equal-area blocks over the valid footprint. The exact pixel boundaries are derived once from the official sample raster and saved as a split file/hash. Exclude a 300 m (3-pixel) guard band at fold edges when evaluating so the 300 m metric neighborhood does not cross a fold boundary.
3. The WPH detector is unsupervised. Do not use held-out fault labels to choose feature bands, masks, thresholds, scales, normalization or parameters. For each fold, use the held-out known labels only to calculate proxy DTI. State clearly that these labels largely represent known faults and cannot measure recall of truly unmapped faults.
4. Freeze the current best holdout raster/report before candidate evaluation. Compare the candidate and incumbent at **matched emitted mass** within each held-out block; use top-ranked cells with equal `p>0` count (same count and same confidence convention). Also report unthresholded continuous-map DTI and a preregistered 0.5%, 1% and 2% of valid-fold-area support sweep; do not choose the best budget after reading the final fold results.
5. Use the competition's exact distance-weighted Tversky formula (`alpha=0.2`, `beta=0.8`, `R=300 m`). Save per-fold `TPw`, `FPw`, `FNw`, DTI, paired delta, number of predicted cells and score mass. Verify the independent reference example from the official problem page.
6. Promotion rule: candidate paired mean DTI delta must be positive; the candidate must beat the incumbent in at least 3 of 4 folds; and the 95% block-bootstrap interval for the paired mean must exclude zero. Report the interval and fold values, not only a pass/fail. Any results at other prediction masses are sensitivity analyses.
7. Report spatial concentration against a single-scale gradient-magnitude comparator from the same inputs: nonzero-support fraction, fraction of score mass in the top 1% and 5% of pixels, and Moran-like neighborhood concentration or an explicitly defined alternative. The requested “sparser and more concentrated” property is a measured hypothesis, not a claim made without output.
8. Before enabling a download, compare the new raw float TIFF against every raw prior submission on the exact sample-footprint mask. This repo policy defines “low correlation” as absolute Pearson and Spearman correlations both below 0.90 for every prior file; Pearson is accumulated exactly over all valid pixels, while Spearman and top-1%-rank overlap use a deterministic, reported systematic sample to bound memory. Positive-support Jaccard is exact over all valid pixels. These are internal distinctness thresholds, not DrivenData requirements. Missing prior bytes, mask mismatches or hash mismatches mean the audit fails closed.
9. Only after a reproducible positive holdout, format receipt and complete distinctness audit can a candidate be marked “eligible for human review.” A valid local candidate is not an organizer score. The participant must decide whether to use a submission slot and must choose the final submission through the authorized competition process.

## Implementation and review passes (2026-10-05 UTC)

1. **Algorithm/charter pass:** checked the WPH-01 implementation against the README prompt and official DTI definition. The detector reads only gravity/RTP features and sample footprint (no held-out labels); H0 lifetimes and upward continuation are explicit; score output is clipped to `[0,1]`; source, config, input and run hashes are recorded. The magnetic code is labelled as an RTP-field variant, not an invented pseudogravity transform. Matched support uses equal positive-pixel counts, and incomplete prior artifacts fail closed.
2. **Implementation/test pass:** the pure-Python H0 and DTI references are checked against production NumPy/SciPy/Numba routines; tests cover ties/masks, metric distances, synthetic WPH inference, spatial folds, GeoTIFF range/grid/mask, band selection, and identical/missing prior artifacts. **23 unit tests pass** (`unittest` and pytest); Ruff, static-site link/config checks, Python compile, shell syntax, and whitespace checks pass. No official competition data were used. A GeoTIFF tiling issue discovered during the first test run was fixed by selecting legal multiple-of-16 tile dimensions.
3. **Evidence/release pass:** checked static-page links/config consistency, fixed the malformed ComCat example URL, verified download/leaderboard status remains disabled/manual, and ran the local data guards. Both data checks correctly returned status 2 because official features, labels and sample template are absent. No candidate TIFF was generated, no correlation/holdout gate was run, and no submission slot was used.

## Run record

- Inputs: absent (`data/training_features.tif`, `data/labels.tif`, `data/sample_submission.tif` not present at session start).
- Current holdout-best raster / report: absent.
- Prior output TIFFs / complete manifest: absent; the 58-row local template has blank paths/hashes and cannot pass.
- Synthetic/reference testing: 23 unit tests pass; no official data were used.
- Candidate inference on competition inputs: not run.
- Four-fold DTI and paired deltas: not available.
- Low-correlation audit: not available.
- GeoTIFF format receipt: not available.
- Submission-slot use: none.

**Decision:** WPH-01 is the ranked top research candidate but is **not validated and not slot-eligible**. Do not download or upload any file from this repository as a competition submission until this record is completed with actual, hash-pinned evidence.
