"""Path resolution for GEMSDOE42.

Data placement order (first hit wins):
  1. $GEMS_DATA_DIR
  2. <repo>/.cache/gems_data
  3. <repo>/data
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Files that must be present for the full pipeline.  Names are relative to the data dir.
CORE_FILES = ("training_features.tif", "labels.tif", "sample_submission.tif")


def candidate_dirs() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("GEMS_DATA_DIR")
    if env:
        out.append(Path(env).expanduser())
    out.append(ROOT / ".cache" / "gems_data")
    out.append(ROOT / "data")
    return out


def data_dir() -> Path:
    """Return the first candidate directory that holds every core file, else the first that exists."""
    existing = [d for d in candidate_dirs() if d.is_dir()]
    for d in existing:
        if all((d / f).is_file() for f in CORE_FILES):
            return d
    if existing:
        return existing[0]
    d = candidate_dirs()[1]
    d.mkdir(parents=True, exist_ok=True)
    return d


def external_dir() -> Path:
    return data_dir() / "external"


def downloads_dir() -> Path:
    d = ROOT / "docs" / "downloads"
    d.mkdir(parents=True, exist_ok=True)
    return d
