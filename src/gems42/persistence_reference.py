"""Independent brute-force reference for dim-0 superlevel persistence (test-only, slow).

Deliberately written a different way from :mod:`gems42.persistence`:

* ``gems42.persistence`` recovers "which higher-level component sits inside which lower-level
  component" with a single vectorised ``scipy.ndimage.maximum`` pass.  That shortcut is only valid
  because superlevel sets are nested, and it is exactly the kind of clever step worth checking.
* This reference recovers the same parent/child relation by direct boolean mask comparison, level
  by level, with no nested-set shortcut at all, and asserts the nesting property as it goes.

Shared convention (stated so the cross-check is meaningful):

    A pixel is assigned to the first component it belonged to that dies.  The pixels of a dying
    component are taken as of the level *above* the merge, because that is the last level at which
    the component existed as an independent structure.  A pixel that first appears at the merge
    level itself was never independent, so it inherits the survivor's feature.

In a *superlevel* filtration the sweep descends, so the oldest component - the survivor of a
merge - is the one with the HIGHEST birth value.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label

STRUCT8 = np.ones((3, 3), bool)


def persistence_map_h0_reference(G: np.ndarray, valid: np.ndarray,
                                 levels: np.ndarray | None = None) -> dict:
    """Brute-force per-pixel ``(persistence, prominence, birth, death)`` plus a feature count."""
    G = np.asarray(G, dtype=np.float32).astype(np.float64)
    valid = np.asarray(valid, bool)
    if G.shape != valid.shape:
        raise ValueError("G and valid must share a shape")

    if levels is None:
        levels = np.unique(G[valid])[::-1]          # exact: every distinct value, descending
    levels = np.asarray(levels, dtype=np.float64)

    birth_px = np.zeros(G.shape, dtype=np.float64)
    death_px = np.zeros(G.shape, dtype=np.float64)
    done = np.zeros(G.shape, dtype=bool)

    node_birth: dict[int, float] = {}
    n_features = 0
    prev_masks: dict[int, np.ndarray] = {}
    last_masks: dict[int, np.ndarray] = {}

    for t in levels:
        mask = valid & (G >= t)
        lab, n = label(mask, structure=STRUCT8)
        masks = {lb: (lab == lb) for lb in range(1, n + 1)}
        births = {lb: float(G[m].max()) for lb, m in masks.items()}
        new_node_birth: dict[int, float] = {}

        if prev_masks:
            children: dict[int, list[int]] = {}
            for plb, pmask in prev_masks.items():
                parents = np.unique(lab[pmask])          # direct containment, no shortcut
                if parents.size != 1:
                    raise AssertionError(f"superlevel sets must be nested (got {parents})")
                children.setdefault(int(parents[0]), []).append(plb)

            for lb in range(1, n + 1):
                kids = children.get(lb, [])
                if len(kids) == 1:
                    new_node_birth[lb] = node_birth[kids[0]]
                elif len(kids) == 0:
                    new_node_birth[lb] = births[lb]
                else:
                    keep = max(kids, key=lambda c: node_birth[c])    # highest birth = oldest
                    new_node_birth[lb] = node_birth[keep]
                    for c in kids:
                        if c == keep:
                            continue
                        sel = prev_masks[c] & ~done
                        if sel.any():
                            birth_px[sel] = node_birth[c]
                            death_px[sel] = float(t)
                            done |= sel
                        n_features += 1
        else:
            for lb in range(1, n + 1):
                new_node_birth[lb] = births[lb]

        node_birth = new_node_birth
        last_masks = masks
        prev_masks = masks

    if last_masks:
        t_low = float(levels[-1])
        for lb, m in last_masks.items():
            sel = m & ~done
            if sel.any():
                birth_px[sel] = node_birth[lb]
                death_px[sel] = t_low
                done |= sel
                n_features += 1

    inside = valid & done
    persistence = np.zeros(G.shape, dtype=np.float64)
    prominence = np.zeros(G.shape, dtype=np.float64)
    persistence[inside] = np.maximum(birth_px[inside] - death_px[inside], 0.0)
    prominence[inside] = np.maximum(G[inside] - death_px[inside], 0.0)
    birth_px[~inside] = 0.0
    death_px[~inside] = 0.0
    return dict(persistence=persistence, prominence=prominence, birth=birth_px,
                death=death_px, n_features=n_features)
