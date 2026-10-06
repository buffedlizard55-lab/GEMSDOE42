"""Second validation instrument: USGS SGMC faults that are NOT in the competition catalogue.

Why this exists
---------------
``gems42.holdout`` hides *catalogue* faults in unseen quadrants.  ``scripts/compare_candidates.py``
measures that instrument against 11 prior submissions whose official leaderboard scores are known
and finds **Spearman +0.087 (p = 0.80, n = 11)** - it does not rank candidates the way the real
leaderboard does.  The reason is structural, not accidental: the private test set is faults that
are *not* in the USGS/INGENIOUS catalogue, and hiding catalogue faults measures the opposite skill.

This module builds a different target.  ``derived_sgmc_faults_100m_u8.tif`` is rasterised from the
USGS State Geologic Map Compilation - an **independent** fault source.  The subset of SGMC fault
pixels lying more than 300 m from any catalogue fault (62,703 px, 1.21% of the footprint) is a
population of *real, mapped faults that the competition's own training labels do not contain*.
That is qualitatively the same thing the private test set is.

It is still only a proxy: SGMC faults were mapped from geologic maps, the private faults were
picked by NLR/USGS experts, and the organiser explicitly declined to say which data or fault types
those experts used (forum thread 11527, post 7).  So this instrument ranks candidates; it does not
predict scores.  Both instruments are reported side by side for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

from .layers import Grid, read_binary
from .metric import dti_exact

STRUCT8 = np.ones((3, 3), bool)


@dataclass
class SgmcTarget:
    truth: np.ndarray          # bool: SGMC fault pixels off the competition catalogue
    n_truth: int
    valid: np.ndarray          # bool: the domain a submission is judged on
    excluded_near_catalogue_px: int


def build_sgmc_target(ddir, grid: Grid, collar_px: int = 3) -> SgmcTarget:
    """SGMC fault pixels at least ``collar_px`` (300 m) from any competition catalogue fault."""
    labels = read_binary(ddir / "labels.tif", grid)
    sgmc = read_binary(ddir / "external" / "derived_sgmc_faults_100m_u8.tif", grid)
    near = binary_dilation(labels, structure=STRUCT8, iterations=collar_px)
    excluded = int((sgmc & near).sum())
    truth = sgmc & ~near & grid.footprint
    return SgmcTarget(truth=truth, n_truth=int(truth.sum()), valid=grid.footprint.copy(),
                      excluded_near_catalogue_px=excluded)


def evaluate(mask: np.ndarray, tgt: SgmcTarget) -> dict:
    """Exact DTI of a boolean emission against the off-catalogue SGMC population."""
    mask = np.asarray(mask, bool) & tgt.valid
    return dti_exact(mask.astype(np.float64), tgt.truth, valid=tgt.valid)


def evaluate_soft(score: np.ndarray, tgt: SgmcTarget) -> dict:
    """Exact DTI of a soft [0, 1] score surface against the off-catalogue SGMC population."""
    s = np.clip(np.nan_to_num(np.asarray(score, dtype=np.float64), nan=0.0), 0.0, 1.0)
    return dti_exact(s, tgt.truth, valid=tgt.valid)
