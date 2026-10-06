#!/usr/bin/env python3
"""Render the site's preview PNGs from the shipped submission and the underlying stages.

Writes:
  docs/img/emission.png      the 60,069 emitted dots over the RTP magnetic field
  docs/img/stages.png        Stage A (worm survival), Stage B (topological persistence), product

Deliberately matplotlib-only and CPU-cheap so the site can be rebuilt in seconds.

    python3 scripts/make_previews.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems42.layers import Grid, load_grid, read_band, read_binary   # noqa: E402
from gems42.paths import data_dir                                    # noqa: E402

OUT = ROOT / "docs" / "img"


def _ax(ax, title: str) -> None:
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)


def main() -> int:
    ddir = data_dir()
    grid = load_grid(ddir / "sample_submission.tif")
    labels = read_binary(ddir / "labels.tif", grid)
    build = json.loads((ROOT / "registry" / "submission_build.json").read_text())
    prim = ROOT / "docs" / "downloads" / Path(build["files"]["primary_nan"]["path"]).name
    with rasterio.open(prim) as ds:
        emit = ds.read(1) > 0
    rtp = read_band(ddir / "training_features.tif", "rtp", grid)
    OUT.mkdir(parents=True, exist_ok=True)

    # --- emission over the magnetic field ----------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 8.2), dpi=150)
    lo, hi = np.nanpercentile(rtp, [1, 99])
    ax.imshow(np.where(grid.footprint, rtp, np.nan), cmap="gray_r", vmin=lo, vmax=hi,
              interpolation="nearest")
    ax.imshow(np.ma.masked_where(~labels, labels), cmap="autumn", alpha=0.85, vmin=0, vmax=1,
              interpolation="nearest")
    ax.imshow(np.ma.masked_where(~emit, emit), cmap="winter", alpha=0.9, vmin=0, vmax=1,
              interpolation="nearest")
    _ax(ax, "Shipped emission (green) over RTP magnetics; catalogue faults in red")
    fig.tight_layout()
    fig.savefig(OUT / "emission.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- stage panels ------------------------------------------------------------------
    worm = np.load(ROOT / ".cache" / "stage" / "A_wormsurv_0-250-500-1000-2000-4000_p88.0.npy")
    topo = np.load(ROOT / ".cache" / "stage" / "B_topo_persistence_0.8-1.6-3.2-6.4_L32_p99.5.npy")
    score = np.load(ROOT / ".cache" / "stage" / "C_score.npy")

    def show(v, pct=(0.0, 99.9)):
        out = np.where(grid.footprint, v, 0.0)
        return np.clip(out / max(float(np.percentile(out[grid.footprint & (out > 0)], pct[1])),
                                 1e-9), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.4), dpi=150)
    axes[0].imshow(show(worm), cmap="magma", vmin=0, vmax=1)
    _ax(axes[0], "Stage A - multiscale worm survival\n(0-4000 m continuation, p88)")
    axes[1].imshow(show(topo), cmap="viridis", vmin=0, vmax=1)
    _ax(axes[1], "Stage B - topological H0 persistence\n(birth - death range)")
    axes[2].imshow(show(score), cmap="inferno", vmin=0, vmax=1)
    _ax(axes[2], "Stage C - product, [0, 1]")
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "stages.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[preview] emitted={int(emit.sum()):,} px")
    print(f"[preview] wrote {OUT/'emission.png'} and {OUT/'stages.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
