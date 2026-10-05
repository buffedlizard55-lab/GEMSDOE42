#!/usr/bin/env bash
# Restore competition + external data into data/ via the owner's git mirrors.
# Every file is sha256-pinned (see data/manifest.json after prepare).
# Requires: git, network to github.com. No DrivenData login needed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/data/bridge" "$ROOT/data/external" "$ROOT/work/priors"

echo "== GEMSDOE mirror (training_features parts + bridge files) =="
if [ ! -d /tmp/gems42_g0 ]; then
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/buffedlizard55-lab/GEMSDOE.git /tmp/gems42_g0
fi
git -C /tmp/gems42_g0 sparse-checkout set data/bridge
cat /tmp/gems42_g0/data/bridge/gems-geodawn-numerical-features.tif.part-* \
  > "$ROOT/data/bridge/training_features.tif"
cp /tmp/gems42_g0/data/bridge/existing_faults.tif "$ROOT/data/bridge/" 2>/dev/null || true

echo "== GEMSDOE24 mirror (labels + sample + external) =="
if [ ! -d /tmp/gems42_g24 ]; then
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/buffedlizard55-lab/GEMSDOE24.git /tmp/gems42_g24
fi
git -C /tmp/gems42_g24 sparse-checkout set data/bridge data/external
cp /tmp/gems42_g24/data/bridge/labels.tif \
   /tmp/gems42_g24/data/bridge/existing_faults.tif \
   /tmp/gems42_g24/data/bridge/sample_submission.tif "$ROOT/data/bridge/"
cp /tmp/gems42_g24/data/external/derived_sgmc_faults_100m_u8.tif \
   /tmp/gems42_g24/data/external/gdr_wellspring_in_footprint.csv \
   "$ROOT/data/external/"

echo "== sha256 verification =="
sha256sum "$ROOT/data/bridge/training_features.tif" \
           "$ROOT/data/bridge/labels.tif" \
           "$ROOT/data/bridge/sample_submission.tif"
echo "expected training_features: 4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5"
echo "expected labels/existing:   7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093"
echo "expected sample:            2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc"

echo "== prior submissions for uniqueness correlation (GEMSDOE32) =="
if [ ! -d /tmp/gems42_g32 ]; then
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/buffedlizard55-lab/GEMSDOE32.git /tmp/gems42_g32
fi
git -C /tmp/gems42_g32 sparse-checkout set docs/downloads
cp /tmp/gems42_g32/docs/downloads/*-zeros.tif "$ROOT/work/priors/" 2>/dev/null || true
ls -lh "$ROOT/work/priors/" | head -20
echo DONE
