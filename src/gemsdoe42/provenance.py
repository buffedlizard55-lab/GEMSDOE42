"""Stable source-file hashes for reproducible candidate/evaluation receipts."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from .raster import sha256_file


def hash_source_bundle(module_names: Sequence[str]) -> tuple[str, dict[str, str]]:
    """Hash named modules in this package and their ordered bundle deterministically."""
    package_root = Path(__file__).resolve().parent
    module_hashes = {
        name: sha256_file(package_root / name)
        for name in module_names
    }
    digest = hashlib.sha256()
    for name, value in sorted(module_hashes.items()):
        digest.update(f"{name}:{value}\n".encode())
    return digest.hexdigest(), module_hashes


CANDIDATE_SOURCE_MODULES = ("pipeline.py", "worming.py", "topology.py", "raster.py")
EVALUATION_SOURCE_MODULES = (
    "metric.py",
    "holdout.py",
    "diagnostics.py",
    "distinctness.py",
    "raster.py",
)
