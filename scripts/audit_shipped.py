#!/usr/bin/env python3
"""Independent audit of the shipped submission file.

Re-opens the file that is actually sitting in docs/downloads with rasterio, checks it against
sample_submission.tif, and re-runs the range check the organiser's form performs. Nothing here
trusts registry/submission_build.json - the point is to catch a case where the record and the file
disagree.

    python3 scripts/audit_shipped.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems42.layers import load_grid, read_binary            # noqa: E402
from gems42.submission import verify_submission             # noqa: E402


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    from gems42.paths import data_dir
    ddir = data_dir()
    template = ddir / "sample_submission.tif"
    grid = load_grid(template)
    labels = read_binary(ddir / "labels.tif", grid)
    build = json.loads((ROOT / "registry" / "submission_build.json").read_text())

    failures: list[str] = []

    for role in ("primary_nan", "twin_zeros"):
        rec = build["files"][role]
        p = ROOT / rec["path"].split("GEMSDOE42/")[-1] if "GEMSDOE42/" in rec["path"] \
            else Path(rec["path"])
        if not p.exists():
            p = ROOT / "docs" / "downloads" / Path(rec["path"]).name
        if not p.exists():
            failures.append(f"{role}: file missing at {rec['path']}")
            continue

        v = verify_submission(p, template)
        print(f"\n=== {role}: {p.name} ===")
        print(f"  bytes      {p.stat().st_size:,}   (record says {rec['bytes']:,})")
        if p.stat().st_size != rec["bytes"]:
            failures.append(f"{role}: size on disk != recorded size")
        got = sha256(p)
        print(f"  sha256     {got}")
        if got != rec["sha256"]:
            failures.append(f"{role}: sha256 on disk != recorded sha256")
        print(f"  verify     {'OK' if v['ok'] else 'FAILED'}")
        for k, ok in v["checks"].items():
            print(f"    {'PASS' if ok else 'FAIL'}  {k}")
            if not ok:
                failures.append(f"{role}: check {k} failed")
        for prob in v["problems"]:
            print(f"    ! {prob}")

        with rasterio.open(p) as ds:
            a = ds.read(1)
        inside = a[grid.footprint]
        print(f"  in-footprint  n={inside.size:,}  finite={int(np.isfinite(inside).sum()):,}  "
              f"min={float(np.nanmin(inside))}  max={float(np.nanmax(inside))}  "
              f"unique={np.unique(inside).tolist()[:6]}")
        print(f"  positive px   {int(((a > 0) & np.isfinite(a)).sum()):,}")
        print(f"  on catalogue  {int((((a > 0) & np.isfinite(a)) & labels).sum()):,}")
        print(f"  dtype={a.dtype}  crs={ds.crs}  epsg={ds.crs.to_epsg()}  "
              f"count={ds.count}  nodata={ds.nodata}")
        if a.dtype != np.float32:
            failures.append(f"{role}: dtype is {a.dtype}, not float32")
        if ds.count != 1:
            failures.append(f"{role}: {ds.count} bands, not 1")
        if not np.isfinite(inside).all():
            failures.append(f"{role}: non-finite values inside the footprint")
        if float(np.nanmin(inside)) < 0.0 or float(np.nanmax(inside)) > 1.0:
            failures.append(f"{role}: values outside [0, 1]")
        if tuple(np.round(ds.transform[:6], 6)) != tuple(np.round(grid.transform[:6], 6)):
            failures.append(f"{role}: geotransform mismatch")

    zp = ROOT / "docs" / "downloads" / Path(build["files"]["zip"]["path"]).name
    if zp.exists():
        with zipfile.ZipFile(zp) as z:
            names = z.namelist()
        print(f"\n=== zip: {zp.name} ===")
        print(f"  contains {names}")
        if len(names) != 1 or not names[0].endswith(".tif"):
            failures.append(f"zip must contain exactly one GeoTIFF, found {names}")
        if sha256(zp) != build["files"]["zip"]["sha256"]:
            failures.append("zip: sha256 on disk != recorded sha256")
    else:
        failures.append(f"zip missing at {zp}")

    print("\n" + ("AUDIT PASSED" if not failures else "AUDIT FAILED"))
    for f in failures:
        print(f"  ! {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
