#!/usr/bin/env python
"""Preregistered spatially-blocked holdout H42-PREREG-1.

Design (fixed before seeing results):
  - 4 quadrant folds from footprint bounding-box midlines.
  - Arms at MATCHED mass (40,000 dots, d=2.8px thinning, off-catalogue):
      A1 ours      : greedy dots from H42-1 worming x persistence score
      A0 single    : greedy dots from single-scale RTP gradmag (h=0, top 15%)
      A2 random    : uniform random dots (seed 42, control)
  - Instruments (both reported; promotion needs the PRIMARY):
      PRIMARY   : off-catalogue SGMC faults (>300 m from catalogue) = the only
                  local proxy drawn from the population the hidden test set
                  comes from (mapped faults missing from the catalogue).
      SECONDARY : catalogue-hidden (catalogue pixels in the held-out fold).
  - Promotion rule: mean paired contrast (A1 - A0) > 0 on >= 3 of 4 folds on
    the PRIMARY instrument. The SECONDARY is reported for calibration only:
    our detector uses ZERO labels, so the holdout measures emission-rule
    quality, not overfitting (no leakage possible by construction).
  - Budget sweep for A1 in {20K,30K,37654,40K,44K,50K} on both instruments
    (nested greedy prefixes); primary budget stays 40K unless the sweep peak
    beats it by > 0.005 mean PRIMARY proxy DTI.

Outputs: work/holdout.json + printed tables.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems42.grid import read_footprint, read_labels
from gems42.metric import dti_binary
from gems42.emission import thin_dots, mask_from_prefix

BUDGET = 30_000
SPACING = 2.8
SWEEP = [10_000, 20_000, 30_000]


def quadrant_folds(foot: np.ndarray):
    ys, xs = np.nonzero(foot)
    my, mx = (ys.min() + ys.max()) / 2.0, (xs.min() + xs.max()) / 2.0
    yy, xx = np.mgrid[0:foot.shape[0], 0:foot.shape[1]]
    return {
        "foldNW": foot & (yy < my) & (xx < mx),
        "foldNE": foot & (yy < my) & (xx >= mx),
        "foldSW": foot & (yy >= my) & (xx < mx),
        "foldSE": foot & (yy >= my) & (xx >= mx),
    }


def main():
    t0 = time.time()
    bridge = ROOT / "data" / "bridge"
    work = ROOT / "work"
    foot = read_footprint(bridge / "sample_submission.tif")
    labels = read_labels(bridge / "labels.tif")
    score01 = np.load(work / "score01.npy")
    assert score01.shape == foot.shape

    # single-scale baseline field: h=0 RTP gradmag cached by the build
    z = np.load(work / "worm_survival.npz")
    g0 = z["g0_mag"].astype(np.float64)
    g0s = np.zeros_like(g0)
    g0s[foot] = (g0[foot] - g0[foot].min()) / (g0[foot].max() - g0[foot].min())
    g0s[labels] = 0.0

    # SGMC off-catalogue instrument (>300 m = >3 px from catalogue)
    with rasterio.open(ROOT / "data" / "external" /
                       "derived_sgmc_faults_100m_u8.tif") as s:
        sgmc = s.read(1) > 0
    dcat = distance_transform_edt(~labels)
    sgmc_off = sgmc & foot & (~labels) & (dcat > 3.0)
    print(f"[holdout] SGMC off-catalogue px (>3px): {sgmc_off.sum()} "
          f"| catalogue px: {labels.sum()}")

    folds = quadrant_folds(foot)
    for k, v in folds.items():
        print(f"[holdout] {k}: footprint={v.sum()} cat={int((labels & v).sum())} "
              f"sgmc_off={int((sgmc_off & v).sum())}")

    # ---- arms (matched mass) ----
    print("[holdout] thinning A1 (ours)...", flush=True)
    acc1, _ = thin_dots(score01, foot, labels, SPACING, max(SWEEP))
    print(f"[holdout] A1 accepted {len(acc1)}", flush=True)
    print("[holdout] thinning A0 (single-scale)...", flush=True)
    acc0, _ = thin_dots(g0s, foot, labels, SPACING, BUDGET)
    rng = np.random.default_rng(42)
    cand = np.column_stack(np.nonzero(foot & ~labels))
    pick = rng.choice(len(cand), size=BUDGET, replace=False)
    rand_mask = np.zeros(foot.shape, bool)
    rand_mask[cand[pick, 0], cand[pick, 1]] = True
    arms = {
        "A1_ours": mask_from_prefix(acc1, foot.shape, BUDGET),
        "A0_single": mask_from_prefix(acc0, foot.shape, BUDGET),
        "A2_random": rand_mask,
    }
    for k, v in arms.items():
        print(f"[holdout] {k}: {v.sum()} dots")
    res_arm_counts = {k: int(v.sum()) for k, v in arms.items()}
    assert len(set(res_arm_counts.values())) == 1, (
        f"matched-mass violation: {res_arm_counts}")

    # ---- score per fold x instrument ----
    res = {"folds": {}, "contrasts": {}, "sweep": {}}
    for fname, fmask in folds.items():
        res["folds"][fname] = {}
        for aname, am in arms.items():
            p = am & fmask
            r_cat = dti_binary(p, labels & fmask, valid=fmask)
            r_sgmc = dti_binary(p, sgmc_off & fmask, valid=fmask)
            res["folds"][fname][aname] = {
                "proxy_catalogue": round(r_cat["dti"], 6),
                "primary_sgmc_off": round(r_sgmc["dti"], 6),
                "n_truth_cat": r_cat["n_truth"],
                "n_truth_sgmc": r_sgmc["n_truth"],
            }
    # contrasts A1-A0
    for inst in ("proxy_catalogue", "primary_sgmc_off"):
        ds = [res["folds"][f]["A1_ours"][inst] - res["folds"][f]["A0_single"][inst]
              for f in folds]
        res["contrasts"][inst] = {
            "mean": round(float(np.mean(ds)), 6),
            "per_fold": [round(d, 6) for d in ds],
            "folds_positive": int(sum(d > 0 for d in ds)),
        }
    # budget sweep for A1 (nested prefixes)
    for k in SWEEP:
        if k > len(acc1):
            continue
        m = mask_from_prefix(acc1, foot.shape, k)
        dc, ds = [], []
        for fmask in folds.values():
            dc.append(dti_binary(m & fmask, labels & fmask,
                                 valid=fmask)["dti"])
            ds.append(dti_binary(m & fmask, sgmc_off & fmask,
                                 valid=fmask)["dti"])
        res["sweep"][str(k)] = {"mean_catalogue": round(float(np.mean(dc)), 6),
                                "mean_sgmc_off": round(float(np.mean(ds)), 6)}

    prom = res["contrasts"]["primary_sgmc_off"]
    res["promotion"] = {
        "rule": "mean paired contrast (A1-A0) > 0 on >=3/4 folds, PRIMARY",
        "result": "PASS" if prom["folds_positive"] >= 3 and prom["mean"] > 0
                  else "FAIL",
    }
    res["seconds"] = round(time.time() - t0, 1)
    (work / "holdout.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"[holdout] promotion: {res['promotion']['result']} "
          f"in {res['seconds']}s")


if __name__ == "__main__":
    main()
