#!/usr/bin/env python
"""End-to-end H42-1 pipeline: worming x persistence -> submission GeoTIFF pair.

Stages (each cached under work/ so reruns are cheap):
  0. load footprint / labels / RTP / gravity
  1. worming per layer -> survival + h=0 gradmag
  2. scale-space persistence per layer on h=0 gradmag
  3. fuse -> raw -> [0,1] -> catalogue-masked score
  4. greedy dotted thinning to budget (nested acceptance order)
  5. write zeros+nan pair + 12-point audit -> docs/downloads/
  6. uniqueness: Pearson/Jaccard vs every prior zeros.tif in work/priors/
  7. sparsity check vs single-scale gradient map

Usage:  pyenv42/bin/python scripts/build_submission.py [--budget 40000]
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

from gems42.grid import (BAND_RTP, BAND_ISO_GRAV, read_footprint, read_labels,
                         read_band_filled)
from gems42.worming import worm_survival, HEIGHTS_M
from gems42.persistence import scale_space_persistence, SIGMAS_PX
from gems42.emission import fuse_scores, normalize01, thin_dots
from gems42.submission import write_pair

SLUG = "h42-1-wormpersist"
TIMESTAMP = "20261005T230000Z"
NOTE = ("GEMSDOE42 H42-1 worming x persistence | {dots} dots off-catalogue, "
        "RTP+gravity worms x topo persistence; sha {sha8}; UNSCORED")


def log(msg, t0=None):
    dt = f" (+{time.time()-t0:.1f}s)" if t0 else ""
    print(f"[build] {msg}{dt}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=40_000)
    ap.add_argument("--spacing", type=float, default=2.8)
    ap.add_argument("--max-budget", type=int, default=50_000)
    ap.add_argument("--fresh", action="store_true",
                    help="ignore cached intermediates")
    args = ap.parse_args()

    t_start = time.time()
    bridge = ROOT / "data" / "bridge"
    work = ROOT / "work"
    work.mkdir(exist_ok=True)
    t0 = time.time()

    # ---- stage 0: load ----
    log("stage 0: loading grid + bands")
    foot = read_footprint(bridge / "sample_submission.tif")
    labels = read_labels(bridge / "labels.tif")
    log(f"footprint={foot.sum()} catalogue={(labels).sum()}", t0)
    t0 = time.time()
    rtp = read_band_filled(bridge / "training_features.tif", BAND_RTP, foot)
    grav = read_band_filled(bridge / "training_features.tif", BAND_ISO_GRAV,
                            foot)
    log(f"bands loaded med_rtp={np.median(rtp[foot]):.1f} "
        f"med_grav={np.median(grav[foot]):.2f}", t0)

    # ---- stage 1: worming ----
    worm_cache = work / "worm_survival.npz"
    if worm_cache.exists() and not args.fresh:
        log("stage 1: loading cached worm survival")
        z = np.load(worm_cache)
        surv_mag, surv_grav = z["surv_mag"], z["surv_grav"]
        g0_mag, g0_grav = z["g0_mag"], z["g0_grav"]
    else:
        t0 = time.time()
        log(f"stage 1a: worming RTP over heights {HEIGHTS_M}")
        surv_mag, _, g0_mag = worm_survival(rtp, foot)
        log(f"RTP worms: survival histogram "
            f"{dict(zip(*np.unique(surv_mag[foot], return_counts=True)))}", t0)
        t0 = time.time()
        log("stage 1b: worming isostatic gravity")
        surv_grav, _, g0_grav = worm_survival(grav, foot)
        log(f"grav worms: survival histogram "
            f"{dict(zip(*np.unique(surv_grav[foot], return_counts=True)))}", t0)
        np.savez_compressed(worm_cache, surv_mag=surv_mag,
                             surv_grav=surv_grav, g0_mag=g0_mag,
                             g0_grav=g0_grav)
        log(f"cached {worm_cache} "
            f"({worm_cache.stat().st_size/1e6:.1f} MB)")

    # ---- stage 2: persistence ----
    pers_cache = work / "persistence.npz"
    if pers_cache.exists() and not args.fresh:
        log("stage 2: loading cached persistence")
        z = np.load(pers_cache)
        pers_mag, pers_grav = z["pers_mag"], z["pers_grav"]
    else:
        t0 = time.time()
        log(f"stage 2a: persistence on RTP gradmag, sigmas {SIGMAS_PX}")
        pers_mag, _, tr_mag, cnt_mag = scale_space_persistence(g0_mag, foot)
        log(f"RTP persistence: {len(tr_mag)} tracks, "
            f"maxima/scale={cnt_mag}, "
            f"steps>1 px={(pers_mag > 1).sum()}", t0)
        t0 = time.time()
        log("stage 2b: persistence on gravity gradmag")
        pers_grav, _, tr_grav, cnt_grav = scale_space_persistence(
            g0_grav, foot)
        log(f"grav persistence: {len(tr_grav)} tracks, "
            f"maxima/scale={cnt_grav}, "
            f"steps>1 px={(pers_grav > 1).sum()}", t0)
        np.savez_compressed(pers_cache, pers_mag=pers_mag,
                             pers_grav=pers_grav)
        log(f"cached {pers_cache}")

    # ---- stage 3: fuse ----
    t0 = time.time()
    log("stage 3: fusing worm x persistence")
    raw, comp = fuse_scores(surv_mag, surv_grav, pers_mag, pers_grav)
    score01 = normalize01(raw, foot)
    score01[labels] = 0.0
    np.save(work / "score01.npy", score01.astype(np.float32))
    nz = int((score01 > 0).sum())
    log(f"raw max={raw[foot].max():.0f} score01 max={score01.max():.4f} "
        f"nonzero={nz} ({100*nz/foot.sum():.2f}% of footprint)", t0)

    # ---- stage 4: thin ----
    t0 = time.time()
    log(f"stage 4: thinning to max_budget={args.max_budget} "
        f"spacing={args.spacing}")
    acc, _ = thin_dots(score01, foot, labels, args.spacing, args.max_budget)
    np.save(work / "accept_order.npy", acc)
    log(f"accepted {len(acc)} dots", t0)
    k = min(args.budget, len(acc))
    mask = np.zeros(foot.shape, bool)
    mask[acc[:k, 0], acc[:k, 1]] = True
    log(f"primary budget k={k}")

    # ---- stage 5: write + audit ----
    t0 = time.time()
    log("stage 5: writing submission pair + audit")
    bundle = write_pair(
        mask, foot, labels, bridge / "sample_submission.tif",
        ROOT / "docs" / "downloads", SLUG, TIMESTAMP,
        "H42-1 multiscale worming x topological persistence, "
        f"{k} dots at d={args.spacing}px, RTP + isostatic gravity",
        NOTE, extra_meta=dict(budget=k, spacing_px=args.spacing,
                              heights_m=list(HEIGHTS_M),
                              sigmas_px=list(SIGMAS_PX)))
    log(f"wrote {bundle['zeros_tif']['filename']} "
        f"sha={bundle['zeros_tif']['sha256'][:12]}...", t0)

    # ---- stage 6: uniqueness vs priors ----
    log("stage 6: uniqueness correlation vs priors")
    priors = sorted((work / "priors").glob("*.tif"))
    uniq = {}
    ours = mask[foot].astype(np.float64)
    for p in priors:
        with rasterio.open(p) as s:
            a = s.read(1)
        b = ((a > 0.5) & foot).astype(np.float64)[foot]
        if b.std() == 0 or ours.std() == 0:
            r = 0.0
        else:
            r = float(np.corrcoef(ours, b)[0, 1])
        jac = float((ours.astype(bool) & b.astype(bool)).sum()
                    / max(1, (ours.astype(bool) | b.astype(bool)).sum()))
        uniq[p.name] = dict(pearson=round(r, 4), jaccard=round(jac, 4),
                            n_prior=int(b.sum()))
        log(f"  {p.name[:60]:60s} pearson={r:.4f} jaccard={jac:.4f}")
    max_r = max([v["pearson"] for v in uniq.values()] or [0.0])
    max_j = max([v["jaccard"] for v in uniq.values()] or [0.0])
    log(f"max pearson={max_r:.4f} max jaccard={max_j:.4f} "
        f"over {len(priors)} priors")
    uniqueness_gate = (max_r < 0.50 and max_j < 0.40) or len(priors) == 0
    log(f"uniqueness gate (<0.50/<0.40): "
        f"{'PASS' if uniqueness_gate else 'FAIL'}")

    # ---- stage 7: sparsity vs single-scale ----
    log("stage 7: sparsity vs single-scale gradient map")
    single = int((((g0_mag >= np.percentile(g0_mag[foot], 85)) & foot).sum()))
    spars = dict(single_scale_top15_px=single, emitted_dots=k,
                 ratio=round(k / max(1, single), 4))
    log(f"single-scale top15%={single} vs emitted={k} "
        f"(ratio {spars['ratio']})")
    assert k < single, "persistence output must be sparser than single-scale"

    receipt = dict(slug=SLUG, timestamp=TIMESTAMP, budget=k,
                   spacing_px=args.spacing,
                   zeros_sha256=bundle["zeros_tif"]["sha256"],
                   nan_sha256=bundle["nan_tif"]["sha256"],
                   uniqueness=uniq, uniqueness_gate=uniqueness_gate,
                   sparsity=spars,
                   seconds=round(time.time() - t_start, 1))
    (work / "build_receipt.json").write_text(json.dumps(receipt, indent=2))
    log(f"DONE in {time.time()-t_start:.1f}s; receipt -> work/build_receipt.json")
    if not uniqueness_gate:
        raise SystemExit("UNIQUENESS GATE FAILED")
    print(json.dumps(receipt, indent=2)[:2000])


if __name__ == "__main__":
    main()
