#!/usr/bin/env python3
"""Check local static-site links, JSON receipts and disabled download state."""

from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PREDICTION_ARTIFACT_SUFFIXES = {".tif", ".tiff", ".zip"}


def public_prediction_artifacts(docs: Path) -> list[Path]:
    """Return candidate raster/archive files that would be publicly served."""
    return sorted(
        path
        for path in docs.rglob("*")
        if path.is_file() and path.suffix.lower() in PREDICTION_ARTIFACT_SUFFIXES
    )


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        for name in ("href", "src"):
            if attributes.get(name):
                self.links.append(attributes[name])
        if attributes.get("id"):
            self.ids.add(attributes["id"])

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)


def main() -> int:
    errors = []
    html_files = sorted(DOCS.rglob("*.html"))
    parsers = {}
    for path in html_files:
        parser = LinkParser()
        try:
            parser.feed(path.read_text(encoding="utf-8"))
            parser.close()
        except Exception as error:  # noqa: BLE001 - report any malformed page as a check failure
            errors.append(f"HTML parse error in {path.relative_to(ROOT)}: {error}")
            continue
        parsers[path.resolve()] = parser

    for page, parser in parsers.items():
        for href in parser.links:
            parsed = urlsplit(href)
            if parsed.scheme or parsed.netloc or href.startswith(("data:", "mailto:", "javascript:")):
                continue
            target_path = unquote(parsed.path)
            target = (page.parent / target_path).resolve() if target_path else page
            if target.is_dir():
                target = target / "index.html"
            if not target.is_file():
                errors.append(f"Broken local link in {page.relative_to(ROOT)}: {href}")
                continue
            if parsed.fragment and target.suffix.lower() == ".html":
                target_parser = parsers.get(target)
                if target_parser is None:
                    target_parser = LinkParser()
                    target_parser.feed(target.read_text(encoding="utf-8"))
                if unquote(parsed.fragment) not in target_parser.ids:
                    errors.append(
                        f"Missing anchor in {page.relative_to(ROOT)}: {href}"
                    )

    for path in DOCS.rglob("*.html"):
        content = path.read_text(encoding="utf-8")
        if re.search(r"\[[^\]]+\]\(https?://", content):
            errors.append(f"Markdown link syntax inside HTML: {path.relative_to(ROOT)}")
        if "%26" in content:
            errors.append(f"Malformed escaped query separator (%26) in {path.relative_to(ROOT)}")

    try:
        canonical = json.loads((ROOT / "configs" / "wph01.json").read_text(encoding="utf-8"))
        public_copy = json.loads((DOCS / "wph01-preregistration.json").read_text(encoding="utf-8"))
        if canonical != public_copy:
            errors.append("docs/wph01-preregistration.json is out of sync with configs/wph01.json")
        status = json.loads((DOCS / "submission-status.json").read_text(encoding="utf-8"))
        if status.get("ready") is not True and status.get("ready") is not False:
            errors.append("submission-status.json requires a Boolean ready field")
        if status.get("ready") is True:
            format_receipt = status.get("format_receipt") or {}
            holdout_receipt = status.get("holdout_receipt") or {}
            correlation_receipt = status.get("correlation_receipt") or {}
            prior_rows = correlation_receipt.get("reports") or []
            prior_count = correlation_receipt.get("prior_count")
            correlations_pass = (
                isinstance(prior_count, int)
                and prior_count > 0
                and isinstance(prior_rows, list)
                and len(prior_rows) == prior_count
                and all(
                    row.get("correlation_gate_failed") is False
                    and abs(row.get("pearson_all_pixels", 1.0)) < 0.90
                    and abs(row.get("spearman_systematic_sample", 1.0)) < 0.90
                    for row in prior_rows
                )
            )
            holdout_interval = holdout_receipt.get("bootstrap_95_percent_interval")
            holdout_pass = (
                status.get("holdout_passed") is True
                and holdout_receipt.get("promotion_gate_passed") is True
                and holdout_receipt.get("positive_folds", 0) >= 3
                and holdout_receipt.get("mean_paired_delta", 0.0) > 0.0
                and isinstance(holdout_interval, list)
                and len(holdout_interval) == 2
                and holdout_interval[0] > 0.0
            )
            ready_fields = (
                isinstance(status.get("candidate_id"), str)
                and bool(status.get("candidate_id"))
                and isinstance(status.get("filename"), str)
                and bool(status.get("filename"))
                and isinstance(status.get("download_path"), str)
                and bool(status.get("download_path"))
                and isinstance(status.get("sha256"), str)
                and bool(re.fullmatch(r"[a-f0-9]{64}", status.get("sha256", ""), re.IGNORECASE))
                and holdout_pass
                and format_receipt.get("valid") is True
                and format_receipt.get("sha256") == status.get("sha256")
                and correlations_pass
                and correlation_receipt.get("all_priors_distinct") is True
            )
            if not ready_fields:
                errors.append("ready submission status lacks matching holdout/format/correlation receipts")
        elif (
            status.get("download_path") is not None
            or status.get("holdout_passed") is True
            or status.get("holdout_receipt") is not None
            or status.get("format_receipt") is not None
            or status.get("correlation_receipt") is not None
        ):
            errors.append("blocked submission status must not expose a download or passing receipts")
        if status.get("ready") is False:
            leaked = public_prediction_artifacts(DOCS)
            if leaked:
                names = ", ".join(str(path.relative_to(ROOT)) for path in leaked)
                errors.append(
                    "blocked submission status must not publish candidate raster/archive files: "
                    + names
                )
        snapshot = json.loads((DOCS / "leaderboard-snapshot-2026-10-05.json").read_text(encoding="utf-8"))
        if snapshot.get("automated_monitoring") is not False:
            errors.append("leaderboard snapshot must remain manual, not automated")
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, AttributeError) as error:
        errors.append(f"JSON/site-state validation failed: {type(error).__name__}: {error}")

    if errors:
        print("Static site checks failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"Static site checks passed: {len(html_files)} HTML pages, local links/anchors and JSON gates verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
