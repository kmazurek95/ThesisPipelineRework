#!/usr/bin/env python
"""CLI entry point for collecting Congressional Record transcripts from GovInfo.

USAGE AND QUICK REFERENCE

This script is a thin CLI wrapper around the collector implemented in
`interest_group_analysis.data_collection.govinfo.fetch_legislative_transcripts`.

Before running
- Set your GovInfo API key. The code loads environment variables and `.env` if
    python-dotenv is installed. Do NOT commit your real key.

    PowerShell (one-time for current session):
            $env:GOVINFO_API_KEY = "YOUR_REAL_KEY"

    Persist for your user (requires new shell):
            setx GOVINFO_API_KEY "YOUR_REAL_KEY"

    Or copy the template and edit it:
            Copy-Item .env.example .env
            # edit .env and replace GOVINFO_API_KEY=REPLACE_WITH_YOUR_KEY

Dry run (no network)
    --dry-run creates a tiny mock dataset and does not contact the API. Useful
    for smoke tests and verifying the pipeline wiring.

Presets
    --full-114
            Shortcut that sets congress=114, start_date=2015-01-06, end_date=2017-01-03,
            output=data/raw/govinfo_114 and increases page_size for higher throughput.

Examples (PowerShell, run from project root)
    # Full 114th Congress (uses GOVINFO_API_KEY from env or .env)
    & .\venv\Scripts\python.exe -m scripts.collect_govinfo --full-114 --no-raw

    # Explicit date range and output directory
    & .\venv\Scripts\python.exe -m scripts.collect_govinfo --congress 114 --start-date 2015-01-06 --end-date 2017-01-03 --output data\raw\govinfo_114 --no-raw

    # Collect two congresses and limit packages for quick test
    & .\venv\Scripts\python.exe -m scripts.collect_govinfo --congress 114 115 --output data\raw\govinfo_114_115 --max-packages 50 --no-raw

    # Dry run (no network, useful for CI or quick smoke tests)
    & .\venv\Scripts\python.exe -m scripts.collect_govinfo --dry-run --output data\raw\govinfo_dry

Where files are written
- Output directory (argument `--output` or auto from presets) will contain:
    - per-package CSVs (package_...csv)
    - `progress.json` (resume info)
    - `error_log.json` (errors encountered)
    - combined `legislative_transcripts.csv` (written at end)

Notes and troubleshooting
- If you see "GOVINFO_API_KEY is not set" set the environment variable or
    create a `.env` file. See commands above.
- If collection stops with network errors, try lowering `--workers` and
    `--page-size` or re-run; `progress.json` allows resuming.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root (folder containing interest_group_analysis) is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging

from interest_group_analysis.data_collection.govinfo import fetch_legislative_transcripts


def positive_int(value: str) -> int:
    iv = int(value)
    return iv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect CREC transcripts from GovInfo API")
    p.add_argument("--congress", nargs="*", type=int, default=None, help="One or more congress numbers (e.g. 114 115)")
    p.add_argument("--start-date", default="2015-01-06", help="Start date (YYYY-MM-DD or full ISO); default 2015-01-06")
    p.add_argument("--end-date", default=None, help="Optional end date (YYYY-MM-DD or full ISO)")
    p.add_argument("--output", required=False, default=None, help="Output directory (auto if --full-114)")
    p.add_argument("--page-size", type=int, default=500, help="Packages page size (default 500)")
    p.add_argument("--workers", type=int, default=8, help="Thread worker count for granules (default 8)")
    p.add_argument("--max-packages", type=positive_int, default=None, help="Max packages per congress (omit for no limit)")
    p.add_argument("--offset-mark", default="*", help="Initial offsetMark (default '*')")
    p.add_argument("--no-raw", action="store_true", help="Disable saving raw JSON responses")
    p.add_argument("--dry-run", action="store_true", help="Dry run: no network, tiny mock dataset")
    p.add_argument("--log-level", default="INFO", help="Logging level (default INFO)")
    p.add_argument("--full-114", action="store_true", help="Shortcut: collect full 114th Congress (2015-01-06 to 2017-01-03) into data/raw/govinfo_114")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    # Apply full-114 shortcut overrides
    if args.full_114:
        if not args.congress:
            args.congress = [114]
        if args.start_date == "2015-01-06":  # leave if user already changed
            args.start_date = "2015-01-06"
        if args.end_date is None:
            args.end_date = "2017-01-03"
        if args.output is None:
            args.output = "data/raw/govinfo_114"
        # Use larger page size unless user overrode
        if args.page_size == 500:
            args.page_size = 1000
        logging.info("Using --full-114 preset -> congress=%s start=%s end=%s output=%s page_size=%s", args.congress, args.start_date, args.end_date, args.output, args.page_size)

    if args.output is None and args.congress:
        args.output = "data/raw/govinfo_" + "_".join(str(c) for c in args.congress)
        logging.info("Auto output path set to %s", args.output)

    if args.output is None:
        raise SystemExit("--output is required (or use --full-114 for automatic path)")

    output_dir = Path(args.output)

    fetch_legislative_transcripts(
        output_dir=output_dir,
        congresses=args.congress,
        start_date=args.start_date,
        end_date=args.end_date,
        page_size=args.page_size,
        workers=args.workers,
        max_packages_per_congress=args.max_packages,
        initial_offset_mark=None if args.offset_mark.lower() == "none" else args.offset_mark,
        dry_run=bool(args.dry_run),
        save_raw=not args.no_raw,
    )


if __name__ == "__main__":
    main()
