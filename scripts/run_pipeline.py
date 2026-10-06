#!/usr/bin/env python3
"""Build, validate and ship the GEMSDOE42 cross-scale-stability submission.

Stages
------
A  multiscale worming (Hornby/Boschetti/Horowitz) on the magnetic and gravity layers
B  dim-0 persistent homology of the same layers' gradient-magnitude surfaces, swept over scales
C  persistence score = continuation steps survived x topological birth-death range, in [0, 1]
D  emission budget chosen on the spatially-blocked holdout, then the GeoTIFF is written and
   independently verified against the organiser's format requirements

Usage
-----
    python3 scripts/run_pipeline.py --budgets 15000,25000,40000,60000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems42.holdout import build_holdout, evaluate                      # noqa: E402
from gems42.layers import Grid, load_grid, read_binary                  # noqa: E402
from gems42.pipeline import (StageConfig, stage_a_worming,              # noqa: E402
                             stage_b_persistence, stage_c_score, select_emission)
from gems42.submission import verify_submission, write_submission, zip_submission  # noqa: E402


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import rankdata
    # ravel FIRST: np.corrcoef treats each ROW of a 2-D input as a variable, so passing grids
    # straight in returns corr(row 0 of a, row 0 of b), not a global correlation.  This was a live
    # bug in ship_submission.py until the audit caught it - keep both copies honest.
    ra, rb = rankdata(np.asarray(a).ravel()), rankdata(np.asarray(b).ravel())
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="15000,25000,40000,60000,90000")
    ap.add_argument("--separation", type=int, default=2)
    ap.add_argument("--n-levels", type=int, default=32)
    ap.add_argument("--which", default="prominence", choices=["prominence", "persistence"])
    ap.add_argument("--worm-combine", default="max", choices=["max", "mean"])
    ap.add_argument("--topo-combine", default="max", choices=["max", "mean"])
    ap.add_argument("--exclude-catalogue", action="store_true", default=True)
    ap.add_argument("--keep-catalogue", dest="exclude_catalogue", action="store_false")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from gems42.paths import data_dir
    ddir = data_dir()
    features = ddir / "training_features.tif"
    template = ddir / "sample_submission.tif"
    print(f"[data] {ddir}", flush=True)

    cfg = StageConfig(n_levels=args.n_levels, which=args.which,
                      worm_combine=args.worm_combine, topo_combine=args.topo_combine)

    grid = load_grid(template)
    labels = read_binary(ddir / "labels.tif", grid)
    print(f"[grid] {grid.shape} {grid.crs}  footprint={int(grid.footprint.sum()):,} "
          f"catalogue={int(labels.sum()):,}", flush=True)

    t0 = time.time()
    print("[stage A] multiscale worming", flush=True)
    worm, worm_info = stage_a_worming(features, grid, cfg)
    print(f"[stage A] done in {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    print("[stage B] topological persistence", flush=True)
    topo, topo_info = stage_b_persistence(features, grid, cfg)
    print(f"[stage B] done in {time.time()-t0:.0f}s", flush=True)

    score = stage_c_score(worm, topo, grid)
    fp = grid.footprint
    print(f"[stage C] score: max={float(score[fp].max()):.4f} "
          f"p99.9={float(np.percentile(score[fp], 99.9)):.4f} "
          f"nonzero={int((score[fp] > 0).sum()):,}", flush=True)

    np.save(ROOT / ".cache" / "stage" / "C_score.npy", score)

    # ------------------------------------------------------------------ holdout budget sweep
    ho = build_holdout(grid.footprint, labels)
    print(f"[holdout] {len(ho.cells)} blocked cells; "
          f"hidden truth px/cell = {[c.n_truth for c in ho.cells]}", flush=True)

    exclude = labels if args.exclude_catalogue else None
    sweep = []
    for b in [int(x) for x in args.budgets.split(",") if x.strip()]:
        mask = select_emission(score, grid, b, min_separation_px=args.separation, exclude=exclude)
        res = evaluate(mask, ho)
        res["budget"] = b
        sweep.append(res)
        print(f"  budget={b:>7,}  emitted={res['emitted_px']:>7,}  "
              f"holdout DTI mean={res['mean']:.4f} +/- {res['std']:.4f}  min={res['min']:.4f}  "
              f"on_catalogue={res['on_catalogue_px']}", flush=True)

    best = max(sweep, key=lambda r: r["mean"])
    print(f"[holdout] best budget = {best['budget']:,} (mean {best['mean']:.4f})", flush=True)

    # ------------------------------------------------------------------ reference instruments
    instruments = {}
    for name in ("score", "worm", "topo"):
        arr = {"score": score, "worm": worm, "topo": topo}[name]
        instruments[name] = {
            "holdout_best_budget": best["budget"],
        }

    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()},
        "grid": {"shape": list(grid.shape), "crs": str(grid.crs),
                 "footprint_px": int(grid.footprint.sum()),
                 "catalogue_px": int(labels.sum())},
        "worm_info": worm_info,
        "topo_info": topo_info,
        "score_stats": {
            "max": float(score[fp].max()),
            "p99.9": float(np.percentile(score[fp], 99.9)),
            "p99": float(np.percentile(score[fp], 99)),
            "median": float(np.median(score[fp])),
            "nonzero_px": int((score[fp] > 0).sum()),
        },
        "holdout_sweep": sweep,
        "best_budget": best["budget"],
        "instruments": instruments,
    }
    (ROOT / "registry" / "pipeline_run.json").write_text(json.dumps(out, indent=1))
    print(f"[write] registry/pipeline_run.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
