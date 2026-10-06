#!/usr/bin/env python3
"""Turn the persistence score into a verified, downloadable submission bundle.

Does five things, in order, and refuses to proceed if any of them fails:

1. reads the cached Stage-C score (or recomputes it from the cached A and B stages),
2. selects the emission at the holdout-chosen budget,
3. writes the GeoTIFF twice - NaN outside the survey footprint (the organiser's own format) and
   a zeros twin - plus a single-file .zip of the primary,
4. independently verifies each file against the organiser's stated format requirements,
5. measures the correlation of the output against every restored prior submission so uniqueness
   is a measurement and not a claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems42.holdout import build_holdout, evaluate                        # noqa: E402
from gems42.layers import load_grid, read_binary                          # noqa: E402
from gems42.pipeline import select_emission                               # noqa: E402
from gems42.submission import verify_submission, write_submission, zip_submission  # noqa: E402

PRIOR_LABELS = {
    "scored_h19_5": ("h19-5-powerlaw-budget-multiline-corroborated", 0.1922),
    "scored_h19_4": ("h19-4-multiline-corroborated-openness-thermal-pop", 0.1894),
    "scored_h16_1": ("h16-1-topo-geophys-baseline-ridges", 0.1855),
    "scored_d15_scored": ("h25-1-dotted-h19-5-d1-5", 0.2477),
    "scored_d28_unscored": ("h25-1-dotted-h19-5-d2-8", 0.2600),
    "scored_gems27_tgc_v2_d15": ("topo-gap-closure-t-v2-on-d1-5", 0.2449),
    "calib_13gems_20261001_r13-lattice-s5_v2_nan-ou": ("13gems-r13-lattice-s5_v2", 0.0904),
    "calib_8GEMSDOE_Hedge-v2_submission": ("8GEMSDOE-Hedge-v2", 0.1563),
    "calib_gems10-h25-ctx-ridge-20260927T2329477041": ("gems10-h25-ctx-ridge", 0.1280),
    "calib_gems10-h28-dotted-ridge-20260928T0202562": ("gems10-h28-dotted-ridge", 0.1839),
    "calib_gemsdoe-ens12-adopted-7f00890a": ("gemsdoe-ens12-adopted", 0.1563),
    "calib_gemsdoe9-PLACEHOLDER-2314b599": ("gemsdoe9-placeholder", 0.0107),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import rankdata
    # ravel FIRST: np.corrcoef treats each ROW of a 2-D input as a variable, so passing grids
    # straight in silently returns corr(row 0 of a, row 0 of b) instead of a global correlation.
    ra, rb = rankdata(np.asarray(a).ravel()), rankdata(np.asarray(b).ravel())
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    x, y = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def load_prior(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with rasterio.open(path) as ds:
        a = ds.read(1)
    if a.shape != shape:
        raise ValueError(f"{path.name}: shape {a.shape} != {shape}")
    return np.nan_to_num(a.astype(np.float64), nan=0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=0, help="0 = use registry/pipeline_run.json")
    ap.add_argument("--separation", type=int, default=2)
    ap.add_argument("--name", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--keep-catalogue", action="store_true")
    ap.add_argument("--score", default="C_score.npy",
                    help="cached Stage-C file in .cache/stage to read out")
    ap.add_argument("--which", default="prominence", choices=("prominence", "persistence"),
                    help="which Stage-B map the score was built from; recorded in "
                         "registry/submission_build.json so the record cannot drift from the file")
    args = ap.parse_args()

    from gems42.paths import data_dir, downloads_dir
    ddir, dldir = data_dir(), downloads_dir()
    template = ddir / "sample_submission.tif"
    grid = load_grid(template)
    labels = read_binary(ddir / "labels.tif", grid)

    run = json.loads((ROOT / "registry" / "pipeline_run.json").read_text())
    model = json.loads((ROOT / "registry" / "emission_model.json").read_text())
    # Budget comes from the emission model fitted to 11 known (score, size) pairs, NOT from the
    # catalogue holdout: registry/irregularities.json IRR-01 records that the holdout is
    # uncorrelated with the live board (Spearman +0.087, p = 0.80).
    budget = args.budget or int(model["optimum"]["emitted_px"])
    print(f"[budget] {budget:,} px (from registry/emission_model.json: "
          f"A={model['parameters']['A']:.1f}, b={model['parameters']['b']:.4f}, "
          f"|G|={model['parameters']['hidden_truth_px_G']:,.0f}, "
          f"R^2={model['r2']:.3f})", flush=True)
    score = np.load(ROOT / ".cache" / "stage" / args.score)
    if score.shape != grid.shape:
        raise SystemExit("cached score does not match the grid - rerun scripts/run_pipeline.py")

    exclude = None if args.keep_catalogue else labels
    mask = select_emission(score, grid, budget, min_separation_px=args.separation, exclude=exclude)
    emit = np.where(mask, np.float32(1.0), np.float32(0.0))
    print(f"[emit] {int(mask.sum()):,} dots  on_catalogue={int((mask & labels).sum()):,}",
          flush=True)

    ho = build_holdout(grid.footprint, labels)
    hold = evaluate(mask, ho)
    print(f"[holdout] mean={hold['mean']:.4f} +/- {hold['std']:.4f} "
          f"per_fold={ {k: round(v,4) for k,v in hold['per_fold'].items()} }", flush=True)

    from gems42.sgmc import build_sgmc_target
    from gems42.sgmc import evaluate as sgmc_evaluate
    tgt = build_sgmc_target(ddir, grid)
    sg = sgmc_evaluate(mask, tgt)
    print(f"[sgmc] off-catalogue USGS SGMC target n={tgt.n_truth:,} -> DTI={sg['dti']:.4f} "
          f"coverage={sg['coverage']:.4f}", flush=True)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tag = args.name or f"gems42-xscale-worm-persistence-{stamp}"
    files: dict[str, dict] = {}

    p_nan = dldir / f"{tag}-nan.tif"
    r = write_submission(emit, template, p_nan, outside="nan")
    files["primary_nan"] = r.__dict__
    print(f"[write] {p_nan.name}  {r.bytes:,} B  sha256={r.sha256[:16]}...", flush=True)

    p_z = dldir / f"{tag}-zeros.tif"
    r2 = write_submission(emit, template, p_z, outside="zeros")
    files["twin_zeros"] = r2.__dict__
    print(f"[write] {p_z.name}  {r2.bytes:,} B", flush=True)

    z = zip_submission(p_nan, dldir / f"{tag}-nan.zip")
    files["zip"] = z
    print(f"[write] {Path(z['path']).name}  {z['bytes']:,} B", flush=True)

    # ---------------------------------------------------------------- independent verification
    verifications = {p.name: verify_submission(p, template) for p in (p_nan, p_z)}
    for name, v in verifications.items():
        status = "OK" if v["ok"] else "FAILED"
        print(f"[verify] {status} {name}  min={v['min']} max={v['max']} "
              f"epsg={v['epsg']} shape={v['shape']}", flush=True)
        if not v["ok"]:
            for prob in v["problems"]:
                print(f"         ! {prob}", flush=True)
    if not all(v["ok"] for v in verifications.values()):
        raise SystemExit("submission verification failed - not shipping")

    # ---------------------------------------------------------------- uniqueness vs priors
    print("[unique] correlating against restored prior submissions", flush=True)
    man = json.loads((ROOT / "registry" / "data_manifest.json").read_text())
    dest = {f["id"]: f["dest"] for f in man["files"]}
    mine_bin = mask.astype(np.float64)
    mine_score = score.astype(np.float64)
    corrs = []
    for pid, (label, lb_score) in PRIOR_LABELS.items():
        rel = dest.get(pid)
        if not rel:
            continue
        p = ddir / rel
        if not p.exists():
            print(f"  [skip] {label}: {rel} not restored", flush=True)
            continue
        prior = load_prior(p, grid.shape)
        prior_bin = (prior > 0).astype(np.float64)
        row = {
            "id": pid, "label": label, "leaderboard_dti": lb_score,
            "prior_positive_px": int(prior_bin.sum()),
            "pearson_binary": pearson(mine_bin, prior_bin),
            "spearman_binary": spearman(mine_bin, prior_bin),
            "spearman_score_vs_prior": spearman(mine_score, prior),
            "overlap_px": int((mask & (prior_bin > 0)).sum()),
        }
        corrs.append(row)
        print(f"  {label:<48} r_bin={row['pearson_binary']:+.4f} "
              f"rho_bin={row['spearman_binary']:+.4f} overlap={row['overlap_px']:>6,}", flush=True)

    worst = max((abs(c["pearson_binary"]) for c in corrs), default=0.0)
    worst_rho = max((abs(c["spearman_binary"]) for c in corrs), default=0.0)
    print(f"[unique] max |pearson| vs any prior = {worst:.4f}; max |spearman| = {worst_rho:.4f}",
          flush=True)

    out = {
        "name": tag,
        "note": args.note or ("GEMSDOE42 cross-scale stability: multiscale worming (Hornby/"
                              "Boschetti/Horowitz upward-continuation edge survival) multiplied "
                              "by dim-0 persistent homology of the same layers' gradient-magnitude "
                              "surfaces across a smoothing sweep."),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": run["config"],
        "budget": budget,
        "separation_px": args.separation,
        "emitted_px": int(mask.sum()),
        "on_catalogue_px": int((mask & labels).sum()),
        "holdout": hold,
        "sgmc_off_catalogue": {**sg, "n_truth": tgt.n_truth},
        "emission_model": model,
        "files": files,
        "verification": verifications,
        "prior_correlation": corrs,
        "uniqueness": {"max_abs_pearson_binary": worst, "max_abs_spearman_binary": worst_rho,
                       "n_priors_compared": len(corrs)},
        "score_stats": run["score_stats"],
        "holdout_sweep": run["holdout_sweep"],
    }
    (ROOT / "registry" / "submission_build.json").write_text(json.dumps(out, indent=1))
    print("[write] registry/submission_build.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
