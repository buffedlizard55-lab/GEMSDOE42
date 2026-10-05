"""Repo hygiene: docs present, no large blobs tracked, data ignored."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_docs_exist():
    for p in ["README.md", "docs/index.html",
              "docs/executive-summary.html", "docs/hypotheses.html",
              "docs/research.html", "docs/sources.html",
              "docs/irregularities.html", "scripts/restore_data.sh",
              "scripts/build_submission.py", "scripts/validate_holdout.py"]:
        assert (ROOT / p).exists(), p


def test_no_large_files_tracked():
    out = subprocess.run(["git", "ls-files", "-s"], capture_output=True,
                         text=True, cwd=ROOT, check=True).stdout
    # sizes via cat-file; simpler: check working files that are tracked
    names = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                           cwd=ROOT, check=True).stdout.split()
    big = []
    for n in names:
        p = ROOT / n
        if p.exists() and p.stat().st_size > 5_000_000:
            big.append((n, p.stat().st_size))
    assert not big, f"large tracked files: {big}"


def test_competition_rasters_ignored():
    r = subprocess.run(["git", "check-ignore",
                        "data/bridge/training_features.tif",
                        "data/bridge/labels.tif"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, "competition rasters must be git-ignored"
