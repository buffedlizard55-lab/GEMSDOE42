"""Spatially-blocked holdout for GEMSDOE42.

Why this and not a random split
-------------------------------
The private test set is *new* faults - structures nobody has mapped.  A random split leaks: the
neighbouring pixel of a held-out fault carries almost the same signal, so a random holdout rewards
memorising the catalogue instead of finding structure.  Blocking by spatial quadrant, with a
1.5 km collar removed around the block boundary and a 1.2 km erosion of the block edge, keeps the
held-out faults genuinely out of the neighbourhood of anything the detector could have keyed on.

Construction
------------
1. Split the footprint into four quadrants at its median row and column (NW, NE, SW, SE).
2. For each quadrant, hide a target fraction of the *whole connected components* of the catalogue
   that touch that quadrant - hiding components rather than pixels, because half a fault trace is
   not a realistic stand-in for an unmapped fault.  A second, equal-sized set of components is
   hidden from the training side so the visible catalogue is realistic.
3. A candidate emission is scored with ``dti_exact`` inside the eroded quadrant, with the visible
   catalogue supplied as the ``known`` mask - the organiser's clarification is that known
   USGS/INGENIOUS pixels are excluded from evaluation entirely
   (https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516/2).
4. Two independent component draws per quadrant, so the reported number is a mean over 8 cells
   with a spread attached.

Caveat, stated plainly: the hidden truth is still *catalogue* faults, i.e. faults that were
mappable from public data.  The private test set is explicitly the faults that were NOT in that
catalogue, so this holdout measures "can the detector find mappable faults in a region it has not
seen", not "can it find faults no one has found".  It is the best instrument available without
the private labels and it is used here for *ranking candidates against each other*, never as a
claim about the leaderboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, label

from .metric import dti_exact

STRUCT8 = np.ones((3, 3), bool)
FOLD_NAMES = ("NW", "NE", "SW", "SE")
COLLAR_PX = 15        # 1.5 km buffer removed around each block boundary
DOMAIN_ERODE = 12     # 1.2 km erosion of the scored block edge
HIDE_FRAC = 0.20
DRAWS = (20, 21)


@dataclass
class Cell:
    key: str
    fold: int
    seed: int
    rows: slice
    cols: slice
    active: np.ndarray          # scored domain inside the block, visible catalogue removed
    truth: np.ndarray           # hidden catalogue inside the scored domain
    n_truth: int


@dataclass
class Holdout:
    footprint: np.ndarray
    labels: np.ndarray
    quadrants: np.ndarray
    cells: list[Cell] = field(default_factory=list)


def quadrant_ids(footprint: np.ndarray) -> np.ndarray:
    footprint = np.asarray(footprint, bool)
    yy, xx = np.nonzero(footprint)
    ym, xm = int(np.median(yy)), int(np.median(xx))
    H, W = footprint.shape
    gy, gx = np.ogrid[:H, :W]
    q = np.full((H, W), -1, np.int8)
    q[(gy < ym) & (gx < xm) & footprint] = 0
    q[(gy < ym) & (gx >= xm) & footprint] = 1
    q[(gy >= ym) & (gx < xm) & footprint] = 2
    q[(gy >= ym) & (gx >= xm) & footprint] = 3
    return q


def _pick(rng, comp_size: np.ndarray, ids: np.ndarray, target_px: float) -> np.ndarray:
    """Pick whole components until their combined size reaches ``target_px``."""
    if ids.size == 0:
        return ids
    perm = rng.permutation(ids)
    cum = np.cumsum(comp_size[perm])
    k = int(np.searchsorted(cum, target_px)) + 1
    return perm[: min(k, perm.size)]


def build_holdout(footprint: np.ndarray, labels: np.ndarray,
                  hide_frac: float = HIDE_FRAC) -> Holdout:
    footprint = np.asarray(footprint, bool)
    labels = np.asarray(labels, bool) & footprint
    quad = quadrant_ids(footprint)
    comp, n_comp = label(labels, structure=STRUCT8)
    comp_size = np.bincount(comp.ravel(), minlength=n_comp + 1)
    all_ids = np.arange(1, n_comp + 1)

    collars, domains, bboxes = {}, {}, {}
    for fold in range(4):
        q = quad == fold
        rows = np.flatnonzero(q.any(axis=1))
        cols = np.flatnonzero(q.any(axis=0))
        bboxes[fold] = (slice(max(0, rows[0] - 6), min(footprint.shape[0], rows[-1] + 7)),
                        slice(max(0, cols[0] - 6), min(footprint.shape[1], cols[-1] + 7)))
        collars[fold] = binary_dilation(q, structure=STRUCT8, iterations=COLLAR_PX) & footprint
        domains[fold] = binary_erosion(q, iterations=DOMAIN_ERODE)

    cells: list[Cell] = []
    for seed in DRAWS:
        for fold in range(4):
            rng = np.random.default_rng(10_000 * (seed + 1) + fold)
            q = quad == fold
            touch_collar = np.isin(all_ids, np.unique(comp[collars[fold] & labels]))
            in_test = np.isin(all_ids, np.unique(comp[q & labels]))
            hid_test = _pick(rng, comp_size, all_ids[in_test],
                             hide_frac * float((labels & q).sum()))
            hid_train = _pick(rng, comp_size, all_ids[~touch_collar],
                              hide_frac * float(comp_size[all_ids[~touch_collar]].sum()))
            hidden = np.isin(comp, hid_test)
            visible = labels & ~hidden & ~np.isin(comp, hid_train)

            rs, cs = bboxes[fold]
            active = domains[fold][rs, cs] & ~visible[rs, cs]
            truth = hidden[rs, cs] & domains[fold][rs, cs] & ~visible[rs, cs]
            cells.append(Cell(key=f"draw{seed}_{FOLD_NAMES[fold]}", fold=fold, seed=seed,
                              rows=rs, cols=cs, active=active, truth=truth,
                              n_truth=int(truth.sum())))
    return Holdout(footprint=footprint, labels=labels, quadrants=quad, cells=cells)


def evaluate(mask: np.ndarray, ho: Holdout) -> dict:
    """Score a boolean emission mask across all 8 blocked cells."""
    mask = np.asarray(mask, bool) & ho.footprint
    per_cell: dict[str, float] = {}
    for c in ho.cells:
        sub = mask[c.rows, c.cols] & c.active
        if c.n_truth == 0 or not sub.any():
            per_cell[c.key] = 0.0
            continue
        per_cell[c.key] = float(dti_exact(sub.astype(np.float64), c.truth, valid=c.active)["dti"])

    per_fold = {FOLD_NAMES[f]: float(np.mean([v for k, v in per_cell.items()
                                              if k.endswith(FOLD_NAMES[f])]))
                for f in range(4)}
    vals = np.asarray(list(per_cell.values()), dtype=np.float64)
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "min": float(vals.min()),
        "per_fold": per_fold,
        "per_cell": per_cell,
        "emitted_px": int(mask.sum()),
        "on_catalogue_px": int((mask & ho.labels).sum()),
    }
