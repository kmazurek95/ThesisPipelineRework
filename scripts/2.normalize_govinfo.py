r"""Thin wrapper for running the normalizer on collected govinfo data.

Usage examples (PowerShell):
# normalize most-recent raw directory (auto-detect)
# & .\venv\Scripts\python.exe .\scripts\normalize_govinfo.py --preset last

# normalize a specific collection
# & .\venv\Scripts\python.exe .\scripts\normalize_govinfo.py --raw-dir data\raw\govinfo_114 --out-dir data\normalized\govinfo_114 --clean
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


def build_cmd(raw_dir: Path, out_dir: Path, clean: bool, chunksize: int, split_by: str | None) -> list:
    cmd = [
        sys.executable,
        "-m",
        "interest_group_analysis.data_processing.process_and_normalize",
        "--raw-dir",
        str(raw_dir),
        "--out-dir",
        str(out_dir),
    ]
    if clean:
        cmd.append("--clean")
    if chunksize and chunksize > 0:
        cmd += ["--chunksize", str(chunksize)]
    if split_by:
        cmd += ["--split-by", split_by]
    return cmd


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_raw_base = repo_root / "data" / "raw"
    parser = argparse.ArgumentParser(description="Normalize govinfo raw data (thin wrapper).")
    parser.add_argument("--raw-dir", type=Path, help="Path to raw data directory")
    parser.add_argument("--out-dir", type=Path, help="Path to write normalized outputs")
    parser.add_argument("--clean", action="store_true", help="Remove existing normalized outputs first")
    parser.add_argument("--chunksize", type=int, default=10000, help="Chunksize passed to normalizer")
    parser.add_argument("--split-by", choices=["package"], help="Write outputs per packageId under by_package/<packageId>/")
    parser.add_argument("--preset", choices=["last", "govinfo_114"], help="Common presets")
    args = parser.parse_args(argv)

    # resolve raw dir
    raw_dir = args.raw_dir
    if args.preset == "last" and not raw_dir:
        found = find_most_recent_raw_dir(default_raw_base)
        if not found:
            logger.error("No raw subdirectories found in %s", default_raw_base)
            return 2
        raw_dir = found
        logger.info("Auto-detected most recent raw dir: %s", raw_dir)
    elif args.preset == "govinfo_114" and not raw_dir:
        raw_dir = default_raw_base / "govinfo_114"

    if not raw_dir:
        logger.error("raw-dir not provided and no preset used.")
        return 2
    raw_dir = raw_dir.resolve()

    # resolve out dir
    out_dir = args.out_dir or (Path(str(raw_dir)).with_name(f"normalized_{raw_dir.name}"))
    out_dir = out_dir.resolve()

    cmd = build_cmd(raw_dir, out_dir, args.clean, args.chunksize, args.split_by)
    logger.info("Running normalizer: %s", " ".join(map(str, cmd)))

    try:
        res = subprocess.run(cmd, check=False)
        return res.returncode
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130
    except Exception:
        logger.exception("Failed to run normalizer subprocess")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
