# Local data directory

Competition rasters, prior raw outputs, fold incumbents, and generated predictions are intentionally excluded from Git. Place only files obtained through authorized sources here. Expected local paths are:

- `training_features.tif` — official multiband competition features
- `labels.tif` — official known-fault raster
- `sample_submission.tif` — exact output grid and valid footprint
- `holdout/incumbent-best.tif` — frozen current holdout incumbent (local, hash recorded)
- `prior-output-manifest.csv` — complete manifest for every prior output to audit, with local `artifact_path` and matching SHA-256
- `prior_submissions/` — the hash-pinned prior raw rasters referenced in that manifest
- `1m_DEM_links.csv` — optional, not used by WPH-01 first pass

Obtain the official competition package only through an authorized participant account at the [official data page](https://www.drivendata.org/competitions/306/competition-doe-gems/data/). This repository does not bypass that access control, save credentials, or download from third-party mirrors.

Run `bash scripts/download_competition_data.sh` to check whether the required files are present; it deliberately does not download them. After placing authorized files locally, run:

```bash
python scripts/prepare_data.py --data-dir data
```

The preflight checks file presence, exact alignment with the sample template, 100 m EPSG:32611 grid metadata, unambiguous or explicitly selected gravity/RTP bands, binary label encoding, and SHA-256 hashes. It writes only a local manifest under ignored `data/prepared/`.

Prior-output manifest template: [`../docs/prior-output-manifest-template.csv`](../docs/prior-output-manifest-template.csv). That public template has blank file paths and hashes and is not an audit receipt; evaluation intentionally fails until all required bytes are verified. Never commit contest inputs, prior submissions, or derived TIFFs.
