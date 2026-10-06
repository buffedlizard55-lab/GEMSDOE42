#!/usr/bin/env python3
"""Calibrate the holdout instrument by scoring known prior submissions on it.

A holdout number means nothing on its own.  This script pushes every restored prior raster -
including several whose official leaderboard score is known - through the *same* spatially-blocked
holdout used to rank GEMSDOE42 candidates, and adds a random-emission baseline at matched budgets.
Only after that can a candidate's holdout DTI be read as good or bad.

    python3 scripts/compare_candidates.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems42.holdout import build_holdout, evaluate          # noqa: E402
from gems42.layers import load_grid, read_binary            # noqa: E402
from gems42.pipeline import select_emission                 # noqa: E402

KNOWN_SCORES = {
    "scored_h19_5": 0.1922,
    "scored_h19_4": 0.1894,
    "scored_h16_1": 0.1855,
    "scored_d15_scored": 0.2477,
    "scored_d28_unscored": None,
    "scored_gems27_tgc_v2_d15": 0.2449,
    "calib_13gems_20261001_r13-lattice-s5_v2_nan-ou": 0.0904,
    "calib_8GEMSDOE_Hedge-v2_submission": 0.1563,
    "calib_gems10-h25-ctx-ridge-20260927T2329477041": 0.1280,
    "calib_gems10-h28-dotted-ridge-20260928T0202562": 0.1839,
    "calib_gemsdoe-ens12-adopted-7f00890a": 0.1563,
    "calib_gemsdoe9-PLACEHOLDER-2314b599": 0.0107,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="12659,20000,40000,60000")
    ap.add_argument("--separations", default="0,1,2")
    ap.add_argument("--which", default="prominence,persistence")
    args = ap.parse_args()

    from gems42.paths import data_dir
    ddir = data_dir()
    grid = load_grid(ddir / "sample_submission.tif")
    labels = read_binary(ddir / "labels.tif", grid)
    ho = build_holdout(grid.footprint, labels)

    rows: list[dict] = []

    def add(name, mask, kind, official=None, extra=None):
        r = evaluate(np.asarray(mask, bool), ho)
        rows.append({"name": name, "kind": kind, "official": official,
                     "holdout_mean": r["mean"], "holdout_std": r["std"],
                     "emitted_px": r["emitted_px"], "on_catalogue_px": r["on_catalogue_px"],
                     **(extra or {})})
        print(f"  {name:<58} emitted={r['emitted_px']:>7,} "
              f"holdout={r['mean']:.4f}±{r['std']:.4f}"
              f"{'' if official is None else f'  official={official}'}", flush=True)
        return r

    # ------------------------------------------------------- random baselines at matched budgets
    print("[baseline] random emission, catalogue excluded", flush=True)
    rng = np.random.default_rng(1234)
    pool = np.flatnonzero((grid.footprint & ~labels).ravel())
    for b in [int(x) for x in args.budgets.split(",") if x.strip()]:
        pick = rng.choice(pool, size=min(b, pool.size), replace=False)
        m = np.zeros(grid.shape, bool).ravel()
        m[pick] = True
        add(f"random-{b}", m.reshape(grid.shape), "baseline")

    # ------------------------------------------------------- the priors, on the same instrument
    print("[prior] restored prior submissions", flush=True)
    man = json.loads((ROOT / "registry" / "data_manifest.json").read_text())
    dest = {f["id"]: f["dest"] for f in man["files"]}
    prior_masks: dict[str, np.ndarray] = {}
    for pid, official in KNOWN_SCORES.items():
        rel = dest.get(pid)
        if not rel or not (ddir / rel).exists():
            print(f"  [skip] {pid}: not restored", flush=True)
            continue
        with rasterio.open(ddir / rel) as ds:
            a = ds.read(1)
        if a.shape != grid.shape:
            print(f"  [skip] {pid}: shape {a.shape}", flush=True)
            continue
        m = np.nan_to_num(a, nan=0.0) > 0
        prior_masks[pid] = m
        add(pid, m, "prior", official)

    # ------------------------------------------------------- GEMSDOE42 variants
    cache = ROOT / ".cache" / "stage"
    worm = np.load(cache / "A_wormsurv_0-250-500-1000-2000-4000_p95.0.npy")
    score = np.load(cache / "C_score.npy")
    topo = {}
    for p in cache.glob("B_topo_*.npy"):
        key = p.stem
        which = key.split("_")[2]
        topo[which] = np.asarray(np.load(p), np.float32)
    print(f"[gems42] topo maps available: {sorted(topo)}", flush=True)

    print("[gems42] single-factor and product variants", flush=True)
    for name, surf in [("worm_only", worm), *[(f"topo_only[{k}]", v) for k, v in topo.items()],
                       ("product[prominence]", score)]:
        for sep in [int(x) for x in args.separations.split(",") if x.strip()]:
            for b in [int(x) for x in args.budgets.split(",") if x.strip()]:
                m = select_emission(np.asarray(surf, np.float32), grid, b,
                                    min_separation_px=sep, exclude=labels)
                if m.sum() == 0:
                    continue
                add(f"{name} sep{sep} b{b}", m, "gems42", None,
                    {"factor": name, "sep": sep, "budget": b})

    # ------------------------------------------------------- rank correlation with the official
    scored = [r for r in rows if r["kind"] == "prior" and r["official"] is not None]
    if len(scored) >= 4:
        from scipy.stats import spearmanr
        rho, pval = spearmanr([r["official"] for r in scored],
                              [r["holdout_mean"] for r in scored])
        print(f"\n[instrument] Spearman(official leaderboard, this holdout) = "
              f"{rho:+.3f} (p={pval:.3f}, n={len(scored)})", flush=True)
    else:
        rho, pval = float("nan"), float("nan")

    out = {"rows": rows,
           "instrument_spearman_vs_official": {"rho": None if rho != rho else float(rho),
                                               "p": None if pval != pval else float(pval),
                                               "n": len(scored)}}
    (ROOT / "registry" / "holdout_calibration.json").write_text(json.dumps(out, indent=1))
    print("[write] registry/holdout_calibration.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
