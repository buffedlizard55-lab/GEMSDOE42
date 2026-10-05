# GEMSDOE42 — GEMS fault-discovery research and submission pipeline

> **Permanent project charter — read this README at the start of every session.**
> The objective is to build an auditable, reproducible way to identify previously unmapped geological faults in the DOE GEMS / GeoDAWN region, produce a legal single-band submission GeoTIFF, and compete effectively. The competition labels are faults, not geothermal vents themselves. Never promise a score or prize, misstate an owner-reported score as official, bypass competition login, fabricate data, copy an earlier submission raster, or spend a weekly submission slot on an idea that has not beaten the frozen spatial holdout incumbent.
>
> **Core Values**
> - **Maximize P(Win):** treat decisions and scarce submission slots as evidence-driven experiments. Optimize the expected result, not activity or optimism.
> - **Own the Outcome:** verify each claim, report failures and blockers, fix issues within our control, and preserve evidence so another person can reproduce the work.
>
> **Standing research prompt:** “Multiscale worming plus topological persistence. Build this from cross-scale stability, not any single-scale feature: apply Hornby, Boschetti, and Horowitz's multiscale ‘worming’ (1999) to the magnetic and gravity layers, tracking which edges survive across upward continuation, and separately compute persistent homology on the same layers' gradient-magnitude surfaces to track which ridge-like features survive across a full sweep of smoothing scales. Multiply the two into one persistence score per pixel — continuation steps survived times topological birth-death range — so a candidate needs corroboration across two independent notions of scale, not a threshold on either alone. Normalize to [0,1], write to the required format, and confirm low correlation against every prior submission's raw output before download — persistence-driven output should be sparser and more spatially concentrated than a single-scale gradient map.”
>
> Before promoting a candidate: register 3–5 distinct hypotheses (layers, physical signature, why it should find faults absent from the USGS/INGENIOUS catalogue, novelty versus prior work, expected DTI direction, cost, and data provenance); check any external data source is official, free/licensed, and actually obtainable; evaluate on frozen spatially blocked folds at matched emission mass against the recorded incumbent; test GeoTIFF format and range; compare the raw raster against every available prior output. If the evidence or raw rasters are missing, mark the gate **BLOCKED**, do not claim a pass, and do not spend a submission slot.
>
> Keep the download and submission instructions prominent on the site. Use a distinct method/date/hash name and a concise DrivenData note only for a validated candidate. Follow the official competition rules, including the one-final-entry selection and required generative-AI disclosure. Do not automate monitoring or copying of DrivenData pages: its current Terms of Use prohibit robots and automatic monitoring absent prior written consent. Keep research grounded in official sources, date-stamp leaderboard snapshots, flag irregularities, and rerun implementation checks in multiple passes.

## Current verified state — 2026-10-05 UTC

This checkout began with **only an 11-byte README**. It contained no raster data, source code, tests, trained model, spatial holdout, prior-output rasters, or submission artifact. The official competition data page sends unauthenticated access to DrivenData login. This environment has no competition account session and its shell cannot fetch external binaries. We will not work around authentication or invent georeferenced predictions.

Consequently, **there is no honest competition TIFF to download from this checkout yet**. The site intentionally disables the submission download until the exact competition template, feature raster, labels, a passing holdout report, and a complete prior-output correlation audit are present. This is a hard data/evidence gate, not a placeholder submission. A first WPH-01 inference implementation, TIFF writer/validator, spatial holdout comparator, concentration diagnostics, and prior-raster distinctness auditor now exist with unit tests. A repo-local `.venv` was provisioned; **22 unit tests pass** on synthetic/reference fixtures. None of these tools has been run on the official competition rasters, and their data-dependent validity and performance remain unverified.

The official public leaderboard page was read on **2026-10-05**. Its displayed leader was **nchuzhoy, 0.3262**; **0.3195** was displayed for DARD, not the leader. These are public-set scores, not the private test score or the final award result. The official page is live and may change; see the [dated snapshot and score-forensics note](docs/research/score-forensics.md).

The requested attribution “H33-H33-2-B2 = 0.2778” is **not verified**. The public GEMSDOE32 page itself labels that raster “UNSCORED” and gives a model projection, not an organizer score. The live DrivenData board shows a 0.2778 public score for participant `extradr19`; the page does not link that row to the named TIFF. Treating the two as the same submission would be an unsupported attribution. See [irregularities](docs/irregularities.md).

## Start here

1. Read this file first at each session. `AGENTS.md` records the same session-start rule.
2. Read [Executive Summary](docs/executive-summary.html) for the submission workflow, [hypotheses](docs/hypotheses.html) for the ranked tests, and [sources](docs/sources.html) for official references and verification status.
3. Inspect [score forensics](docs/research/score-forensics.md), [prior owner-reported results](docs/prior-results.csv), and [irregularities](docs/irregularities.md) before drawing conclusions from historic scores.
4. Check the data gate without attempting to bypass login:

   ```bash
   bash scripts/download_competition_data.sh
   ```

   The first command reports missing authorized inputs and the official download page; it does not ask for or store credentials. It is expected to stop until the user/team has lawfully downloaded the competition package.
5. Install the declared packages and run the tests. Once the authorized files exist, prepare the local input manifest, then generate a **quarantined research candidate** and evaluate it against the frozen incumbent and complete prior-output manifest:

   ```bash
   python -m venv .venv
   .venv/bin/python -m pip install -e '.[dev]'
   .venv/bin/python -m unittest discover -s tests -v
   .venv/bin/python scripts/prepare_data.py --data-dir data
   .venv/bin/python scripts/build_candidate.py --data-dir data
   .venv/bin/python scripts/evaluate_candidate.py \
     --candidate artifacts/candidates/<candidate-name>.tif \
     --incumbent data/holdout/incumbent-best.tif \
     --prior-manifest data/prior-output-manifest.csv \
     --data-dir data
   ```

   `build_candidate.py` never enables the public download. Evaluation fails closed if the incumbent or any hash-pinned prior raster is missing. The committed [`docs/prior-output-manifest-template.csv`](docs/prior-output-manifest-template.csv) is intentionally incomplete and **cannot pass**. Only a complete receipt can make a candidate eligible for human review; a separate human release/publication action and official portal decision remain required.

## Competition task and output contract

DrivenData #306 asks for fault-presence confidence/probability over the GeoDAWN study area. The official problem description specifies the distance-weighted Tversky index with `alpha=0.2`, `beta=0.8`, and a triangular distance kernel with 300 m support; it describes 100 m pixels and UTM zone 11N (EPSG:32611). A submitted raster must be a **single-band float32 GeoTIFF**, use the training grid's exact CRS, resolution, bounds, and transform, contain prediction values in `[0,1]`, and encode outside-footprint cells as null/NaN. Always derive shape, transform and footprint from the official sample raster; do not infer dimensions from another team's webpage.

The portal error `Predicted values must be in range [0, 1]` can result from a finite nodata sentinel (for example a large negative value) being treated as a prediction. The writer therefore uses the official sample raster as its grid template, checks only the in-footprint prediction cells for finite `[0,1]` values, writes NaN outside the footprint, and reopens the written bytes for an independent format receipt. This behavior is implemented but **not data-validated in this checkout**.

## Research decision: do not mistake leaderboard score for geological truth

The public leaderboard ranks submissions against a public subset of newly expert-labeled faults. Official rules state that a separate private subset is used for Phase 1 and that experts may expand labels from submitted predictions before Phase 2. Public-board optimization is therefore an incomplete proxy for discovery. The user-supplied historic scores are preserved as **unverified owner/user-reported claims**; a website title, a filename, and a score with the same number do not prove organizer attribution. Never copy an earlier raster. Rebuild predictions from the original features with a new, documented method and verify raw-output distinctness.

The proposed worm × topology method is a scientific hypothesis, not a demonstrated improvement. Previous project notes mention `wormrank` and `wormsurv-filter` attempts marked “zeros”; that is a reason for a careful implementation and audit, not evidence that a corrected scale-space PH product will work. A catalogue-only spatial holdout cannot establish recall on faults absent from that catalogue, so any proxy score must be labelled accordingly.

## Data access and current blockers

- Official competition files (`training_features.tif`, `labels.tif`, `sample_submission.tif`, and `1m_DEM_links.csv`) are served through the competition data page, which redirects unauthenticated access to login. The official rules say participants register to receive the training package. No authenticated package exists in this checkout.
- The user-provided Dropbox links and publicly available USGS/DOE records are useful sources to investigate, but a link alone is not proof of grid identity, band order, coverage, checksum, or permission to redistribute. No binary input was successfully placed in `data/` here.
- No baseline holdout report or raw TIFF from the prior GEMSDOE submissions is present. Therefore the candidate cannot yet be tested against a current holdout best or correlated against every prior raw output.
- The requested automatic live leaderboard feed conflicts with DrivenData's published Terms of Use, which prohibit robots/automatic access for monitoring/copying without prior written consent. This repository therefore keeps a dated snapshot and links to the official page; it does not run a crawler or submission bot.
- Official GEMS rules permit generative AI but require disclosure in the narrative of its extent and use. The submitter remains responsible for accuracy, authenticity, and authorship representations.

Full detail and the next steps are in [research sources](docs/sources.html), [ranked hypotheses](docs/hypotheses.html), and [limitations](docs/executive-summary.html#limitations).

## Development and verification

Python 3.11+ is expected. Raster processing uses NumPy, SciPy, Rasterio and Numba; dependencies are listed in `pyproject.toml`.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pytest -q
.venv/bin/ruff check src scripts tests
.venv/bin/python -m compileall -q src scripts tests
.venv/bin/python scripts/check_site.py
bash -n scripts/download_competition_data.sh
git diff --check
```

Once authorized data are present, follow the commands on the [Executive Summary](docs/executive-summary.html). Do **not** upload a candidate or consume a weekly slot until the spatially blocked holdout improves over its frozen incumbent on the preregistered folds and the prior-output correlation audit is complete.

## Sources for manual review

- [DrivenData problem description, metric, data description, and submission format](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/)
- [DrivenData public leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/)
- [DrivenData competition data page](https://www.drivendata.org/competitions/306/competition-doe-gems/data/) (login required)
- [DOE/NLR GEMS Prize Official Rules, September 2026 (PDF)](https://docs.nlr.gov/docs/fy26osti/96647.pdf)
- [DrivenData Terms of Use](https://www.drivendata.org/termsofuse/)
- [USGS GeoDAWN data release, DOI 10.5066/P93LGLVQ](https://doi.org/10.5066/P93LGLVQ)
- [Hornby, Boschetti & Horowitz (1999), Analysis of potential field data in the wavelet domain](https://doi.org/10.1046/j.1365-246x.1999.00788.x)

The complete evidence table and dated snapshots live under [`docs/`](docs/).
