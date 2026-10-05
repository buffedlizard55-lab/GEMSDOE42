#!/usr/bin/env bash
# The official competition package requires the participant's authenticated account.
# This guard intentionally does not fetch, scrape, or store credentials.
set -euo pipefail

DATA_DIR="${1:-data}"
required=(
  "${DATA_DIR}/training_features.tif"
  "${DATA_DIR}/labels.tif"
  "${DATA_DIR}/sample_submission.tif"
)
missing=0
for path in "${required[@]}"; do
  if [[ -f "${path}" ]]; then
    printf 'FOUND  %s\n' "${path}"
  else
    printf 'MISSING %s\n' "${path}"
    missing=1
  fi
done

if (( missing )); then
  cat >&2 <<'EOF'
Competition files are not downloaded automatically. Sign in as an authorized
participant and obtain them through the official page:
https://www.drivendata.org/competitions/306/competition-doe-gems/data/

Place the original files in the data directory. This script does not bypass login,
contact a third-party file host, or save credentials. After setting up the repo-local
virtual environment as described in README.md, run:
  .venv/bin/python scripts/prepare_data.py --data-dir data
EOF
  exit 2
fi
printf '\nAll required local inputs are present. Run scripts/prepare_data.py to validate them.\n'
