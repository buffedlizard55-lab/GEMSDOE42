#!/usr/bin/env python3
"""Calibrate BOTH validation instruments against the 11 prior submissions with known public scores.

Writes registry/holdout_calibration.json, which is what the site's validation section reads.

The reason this exists: a holdout that cannot rank known-good submissions is worse than no holdout
at all, because it manufactures confidence. So every instrument in this repository is scored against
ground truth we actually have - the official public leaderboard - before it is allowed to rank
anything.

    python3 scripts/calibrate_instruments.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems42.holdout import build_holdout, evaluate as holdout_evaluate   # noqa: E402
from gems42.layers import load_grid, read_binary                        # noqa: E402
from gems42.metric import dti_exact                                     # noqa: E402
from gems42.paths import data_dir                                       # noqa: E402
from gems42.sgmc import build_sgmc_target, evaluate as sgmc_evaluate    # noqa: E402

# Official public leaderboard scores, keyed to a token in the restored filename.
# Source: https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/
# (fetched 2026-10-05) cross-referenced with registry/data_manifest.json provenance.
# Official public leaderboard scores, keyed by the manifest id so the file mapping is
# unambiguous.  Source: the owner-mirrored rasters' provenance in registry/data_manifest.json,
# cross-referenced with the live board fetched 2026-10-05.  `scored_d28_unscored` is excluded:
# it was never scored publicly, so it cannot calibrate anything.
OFFICIAL = {
    "scored_h19_5": 0.1922,
    "scored_h19_4": 0.1894,
    "scored_h16_1": 0.1855,
    "scored_d15_scored": 0.2477,
    "scored_gems27_tgc_v2_d15": 0.2449,
    "calib_8GEMSDOE_Hedge-v2_submission": 0.1563,
    "calib_gems10-h28-dotted-ridge-20260928T0202562": 0.1839,
    "calib_gems10-h25-ctx-ridge-20260927T2329477041": 0.1280,
    "calib_gemsdoe-ens12-adopted-7f00890a": 0.1563,
    "calib_13gems_20261001_r13-lattice-s5_v2_nan-ou": 0.0904,
    "calib_gemsdoe9-PLACEHOLDER-2314b599": 0.0107,
}


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


def main() -> int:
    ddir = data_dir()
    grid = load_grid(ddir / "sample_submission.tif")
    labels = read_binary(ddir / "labels.tif", grid)
    ho = build_holdout(grid.footprint, labels)
    tgt = build_sgmc_target(ddir, grid)
    print(f"[calib] holdout cells={len(ho.cells)}  sgmc truth px={tgt.n_truth:,}", flush=True)

    # Resolve each known score to a file on disk: the restored names are not identical to the
    # keys in the leaderboard table (suffixes differ), so match on a prefix rather than guessing.
    # Authoritative id -> filename mapping comes from registry/data_manifest.json, which records
    # the `dest` each restored prior was written to. Token-matching filenames is not reliable here:
    # two different leaderboard ids contain "d1-5".
    man = json.loads((ROOT / "registry" / "data_manifest.json").read_text())
    dest_by_id = {}
    for e in man["files"]:
        if str(e.get("dest", "")).startswith("scored/"):
            dest_by_id[e["id"]] = ddir / e["dest"]
    missing = [k for k in OFFICIAL if k not in dest_by_id]
    if missing:
        raise SystemExit(f"manifest has no scored/ entry for {missing}")

    rows = []
    for name, official in sorted(OFFICIAL.items()):
        p = dest_by_id[name]
        with rasterio.open(p) as ds:
            a = np.nan_to_num(ds.read(1).astype(np.float32), nan=0.0)
        mask = a > 0
        h = holdout_evaluate(mask, ho)
        s = sgmc_evaluate(mask, tgt)
        n = int(mask.sum())
        rows.append({
            "id": name, "file": p.name, "official_score": official, "emitted_px": n,
            "provenance": next(e.get("provenance", "") for e in man["files"]
                               if e.get("id") == name),
            "holdout_dti": float(h["mean"]), "holdout_std": float(h["std"]),
            "sgmc_dti": float(s["dti"]), "sgmc_coverage": float(s["coverage"]),
            "is_ours": False,
        })
        print(f"  {name[:52]:<52} N={n:>8,} official={official:.4f} "
              f"holdout={h['mean']:.4f} sgmc={s['dti']:.4f}", flush=True)

    # Score our own shipped submission on both instruments, so the site's candidate row is read
    # from this same record rather than quoted by hand.
    build_p = ROOT / "registry" / "submission_build.json"
    if build_p.exists():
        build = json.loads(build_p.read_text())
        mine = ROOT / "docs" / "downloads" / Path(build["files"]["primary_nan"]["path"]).name
        if mine.exists():
            with rasterio.open(mine) as ds:
                a = np.nan_to_num(ds.read(1).astype(np.float32), nan=0.0)
            mask = a > 0
            h = holdout_evaluate(mask, ho)
            s2 = sgmc_evaluate(mask, tgt)
            rows.append({
                "id": build["name"], "file": mine.name, "official_score": None,
                "emitted_px": int(mask.sum()),
                "holdout_dti": float(h["mean"]), "holdout_std": float(h["std"]),
                "sgmc_dti": float(s2["dti"]), "sgmc_coverage": float(s2["coverage"]),
                "is_ours": True, "provenance": "built by this repository; not yet submitted",
            })
            print(f"  {build['name'][:52]:<52} N={int(mask.sum()):>8,} official=  n/a  "
                  f"holdout={h['mean']:.4f} sgmc={s2['dti']:.4f}   <-- ours", flush=True)

    priors = [r for r in rows if not r["is_ours"]]
    off = np.array([r["official_score"] for r in priors])
    hh = np.array([r["holdout_dti"] for r in priors])
    ss = np.array([r["sgmc_dti"] for r in priors])
    nn = np.array([r["emitted_px"] for r in priors], dtype=float)

    out = {
        "n_priors": len(priors),
        "spearman_official_vs_holdout": dict(zip(("rho", "p"), spearman(off, hh))),
        "spearman_official_vs_sgmc": dict(zip(("rho", "p"), spearman(off, ss))),
        "spearman_size_vs_official": {
            "holdout": dict(zip(("rho", "p"), spearman(nn, hh))),
            "sgmc": dict(zip(("rho", "p"), spearman(nn, ss))),
            "official": dict(zip(("rho", "p"), spearman(nn, off))),
        },
        "spearman_holdout_vs_sgmc": dict(zip(("rho", "p"), spearman(hh, ss))),
        "sgmc_truth_px": tgt.n_truth,
        "priors": priors,
        "candidates": [r for r in rows if r["is_ours"]],
    }
    (ROOT / "registry" / "holdout_calibration.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")

    print("\n[calib] Spearman vs official leaderboard score:")
    print(f"  catalogue holdout      rho={out['spearman_official_vs_holdout']['rho']:+.3f} "
          f"p={out['spearman_official_vs_holdout']['p']:.3f}")
    print(f"  off-catalogue SGMC     rho={out['spearman_official_vs_sgmc']['rho']:+.3f} "
          f"p={out['spearman_official_vs_sgmc']['p']:.3f}")
    print(f"  emitted pixel count    rho={out['spearman_size_vs_official']['official']['rho']:+.3f} "
          f"p={out['spearman_size_vs_official']['official']['p']:.4f}")
    print(f"  holdout vs SGMC agree  rho={out['spearman_holdout_vs_sgmc']['rho']:+.3f} "
          f"p={out['spearman_holdout_vs_sgmc']['p']:.3f}")
    print("[write] registry/holdout_calibration.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
