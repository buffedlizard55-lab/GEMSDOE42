#!/usr/bin/env python3
"""Restore and SHA-256 verify every competition / external / historical file in registry/data_manifest.json.

Data placement is the single blocker to running the full pipeline.  The DrivenData data page is
login-walled, so this script pulls the owner-supplied integrity-pinned mirrors from the sibling
GEMSDOE* GitHub repositories and verifies each byte against the recorded SHA-256.

Pins prove *mirror consistency*, not organizer authentication.  Any mismatch aborts loudly.

Usage:
    python3 scripts/restore_data.py [--group core|external|scored|all] [--target-dir DIR] [--skip-large]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "registry" / "data_manifest.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gh_raw(repo: str, ref: str, path: str, dest: Path) -> None:
    """Stream a single file out of a GitHub repo via the Contents API raw media type."""
    if shutil.which("gh") is None:
        raise SystemExit("gh CLI not found and raw.githubusercontent.com is not reachable here")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    with tmp.open("wb") as out:
        subprocess.run(
            [
                "gh", "api",
                f"repos/{repo}/contents/{path}?ref={ref}",
                "-H", "Accept: application/vnd.github.raw",
            ],
            stdout=out,
            check=True,
        )
    tmp.replace(dest)


def restore_entry(entry: dict, root: Path, skip_large: bool) -> dict:
    dest = root / entry["dest"]
    nbytes = entry.get("bytes", 0)
    if skip_large and nbytes > 20_000_000 and not dest.exists():
        return {"id": entry["id"], "dest": entry["dest"], "group": entry["group"],
                "status": "skipped-large", "bytes": nbytes, "sha256": entry["sha256"]}
    if dest.exists() and dest.stat().st_size == nbytes and sha256_file(dest) == entry["sha256"]:
        return {"id": entry["id"], "dest": entry["dest"], "group": entry["group"],
                "status": "present", "bytes": nbytes, "sha256": entry["sha256"]}

    t0 = time.time()
    tmp = dest.with_suffix(dest.suffix + ".assembling")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        if "parts" in entry:
            with tmp.open("wb") as out:
                for part in entry["parts"]:
                    p = root / "raw_parts" / Path(part).name
                    gh_raw(entry["repo"], entry["ref"], part, p)
                    with p.open("rb") as src:
                        shutil.copyfileobj(src, out, 1 << 20)
                    p.unlink()
        else:
            gh_raw(entry["repo"], entry["ref"], entry["path"], tmp)
        got = sha256_file(tmp)
        if got != entry["sha256"]:
            tmp.unlink(missing_ok=True)
            raise SystemExit(
                f"SHA-256 MISMATCH for {entry['id']}\n  expected {entry['sha256']}\n  got      {got}"
            )
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return {"id": entry["id"], "dest": entry["dest"], "group": entry["group"],
            "status": "restored", "bytes": dest.stat().st_size, "sha256": got,
            "seconds": round(time.time() - t0, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all", choices=["core", "external", "scored", "all"])
    ap.add_argument("--target-dir", default=os.environ.get("GEMS_DATA_DIR", str(ROOT / ".cache" / "gems_data")))
    ap.add_argument("--skip-large", action="store_true", help="skip files >20 MB (fast smoke test)")
    ap.add_argument("--only", default="", help="comma-separated file ids")
    args = ap.parse_args()

    root = Path(args.target_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.load(MANIFEST.open())
    only = {s for s in args.only.split(",") if s}

    receipt = {"schema_version": 1, "target_dir": str(root),
               "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "files": []}
    failed = 0
    for entry in manifest["files"]:
        if args.group != "all" and entry["group"] != args.group:
            continue
        if only and entry["id"] not in only:
            continue
        try:
            rec = restore_entry(entry, root, args.skip_large)
        except subprocess.CalledProcessError as exc:
            rec = {"id": entry["id"], "dest": entry["dest"], "group": entry["group"],
                   "status": "FAILED", "error": f"exit {exc.returncode}"}
            failed += 1
        secs = f"  {rec['seconds']}s" if "seconds" in rec else ""
        print(f"[{rec['status']:>13}] {rec['id']:<48} {rec.get('bytes', 0):>12,} B{secs}", flush=True)
        receipt["files"].append(rec)

    receipt["all_verified"] = all(r["status"] in ("present", "restored", "skipped-large")
                                  for r in receipt["files"])
    out = ROOT / "data" / "restore_receipt.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=1))
    print(f"\nreceipt -> {out}")
    print(f"ALL_VERIFIED={receipt['all_verified']}  failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
