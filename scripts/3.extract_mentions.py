"""Thin wrapper to run mention extraction over collected/normalized corpora.

Usage examples (PowerShell):
# Run on most recent raw directory
# & .\venv\Scripts\python.exe .\scripts\extract_mentions.py --preset last

# Run on a specific directory and output to a named folder
# & .\venv\Scripts\python.exe .\scripts\extract_mentions.py --source-dir data\raw\govinfo_114 --out-dir data\processed\mentions_114 --no-resume
"""
from __future__ import annotations

import sys
import argparse
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_most_recent_raw_dir(base: Path) -> Path | None:
    if not base.exists():
        return None
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def build_cmd(source_dir: Path, interest_csv: Path, out_dir: Path, threads: int, chunk_size: int, resume: bool, case_sensitive: bool, fuzzy_threshold: int | None, only_main: bool) -> list:
    cmd = [
        sys.executable,
        "-m",
        "interest_group_analysis.data_processing.mention_extraction",
        "--source-dir",
        str(source_dir),
        "--interest-csv",
        str(interest_csv),
        "--out-dir",
        str(out_dir),
        "--threads",
        str(threads),
        "--chunk-size",
        str(chunk_size),
    ]
    if not resume:
        cmd.append("--no-resume")
    if case_sensitive:
        cmd.append("--case-sensitive-acronyms")
    # FlashText removed
    if fuzzy_threshold is not None:
        cmd += ["--fuzzy-threshold", str(fuzzy_threshold)]
    if only_main:
        cmd.append("--only-main")
    return cmd


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_raw_base = repo_root / "data" / "raw"
    default_norm_base = repo_root / "data" / "normalized"
    default_interest_csv = repo_root / "data" / "Interest_group_names and acronyms.csv"

    parser = argparse.ArgumentParser(description="Run mention extraction (thin wrapper).")
    parser.add_argument("--source-dir", type=Path, help="Directory with raw/processed files")
    parser.add_argument("--interest-csv", type=Path, help="Interest group master CSV")
    parser.add_argument("--out-dir", type=Path, help="Output directory for mentions")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--case-sensitive", dest="case_sensitive", action="store_true")
    # FlashText removed
    parser.add_argument("--fuzzy-threshold", dest="fuzzy_threshold", type=int, default=None)
    parser.add_argument("--preset", choices=["last", "govinfo_114", "normalized_last", "normalized_govinfo_114"], help="Common presets")
    parser.add_argument("--only-main", action="store_true", help="Restrict to normalized main.csv or by_month shards if present")
    args = parser.parse_args(argv)

    source_dir = args.source_dir
    if args.preset in {"last", "normalized_last"} and not source_dir:
        base = default_norm_base if args.preset == "normalized_last" else default_raw_base
        found = find_most_recent_raw_dir(base)
        if not found:
            logger.error("No subdirectories found in %s", base)
            return 2
        source_dir = found
        logger.info("Auto-detected most recent dir: %s", source_dir)
    elif args.preset == "govinfo_114" and not source_dir:
        source_dir = default_raw_base / "govinfo_114"
    elif args.preset == "normalized_govinfo_114" and not source_dir:
        source_dir = default_norm_base / "govinfo_114"

    if not source_dir:
        logger.error("source-dir not provided and no preset used.")
        return 2
    source_dir = source_dir.resolve()

    interest_csv = args.interest_csv or default_interest_csv
    out_dir = args.out_dir or (Path(str(source_dir)).with_name(f"mentions_{source_dir.name}"))
    out_dir = out_dir.resolve()

    cmd = build_cmd(source_dir, interest_csv, out_dir, args.threads, args.chunk_size, args.resume, args.case_sensitive, args.use_flashtext, args.fuzzy_threshold, args.only_main)
    logger.info("Running mention extraction: %s", " ".join(map(str, cmd)))
    try:
        res = subprocess.run(cmd, check=False)
        return res.returncode
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130
    except Exception:
        logger.exception("Failed to run mention extraction subprocess")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
