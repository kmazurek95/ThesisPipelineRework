#!/usr/bin/env python3
"""
Merge legacy prominence labels onto new analytic_windows.

This script matches old labeled data (from labeled_windows_2023.xlsx) with 
new pipeline output (analytic_windows) using ONLY paragraph_uuid.
The org_id and text are NOT used for matching because they may differ
between the old and new pipeline runs.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def read_windows(path: Path) -> pd.DataFrame:
    """
    Read analytic_windows from CSV or JSONL.
    """
    ext = path.suffix.lower()
    if ext in {".jsonl", ".ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    elif ext == ".csv":
        return pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"Unsupported windows file type: {path}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(asctime)s: %(message)s",
    )

    ap = argparse.ArgumentParser(
        description="Merge legacy prominence labels using paragraph UUID only."
    )
    ap.add_argument(
        "--windows",
        type=Path,
        required=True,
        help="Path to analytic_windows file (.csv or .jsonl) from data/processed/...",
    )
    ap.add_argument(
        "--legacy",
        type=Path,
        required=True,
        help="Path to labeled_windows_2023.xlsx (legacy labels).",
    )
    ap.add_argument(
        "--sheet",
        default="Final_Dataset",
        help="Sheet name in the legacy Excel file (default: Final_Dataset).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV path for merged labeled windows.",
    )
    args = ap.parse_args()

    # 1) Load analytic_windows
    LOGGER.info("Reading analytic_windows from: %s", args.windows)
    win_df = read_windows(args.windows)
    if win_df.empty:
        raise SystemExit(f"No rows in windows file: {args.windows}")
    LOGGER.info("Loaded %d rows from analytic_windows", len(win_df))

    if "paragraph_uuid" not in win_df.columns:
        raise SystemExit("analytic_windows is missing 'paragraph_uuid' column.")

    # 2) Load legacy labeled data (Excel)
    LOGGER.info("Reading legacy labels from: %s (sheet: %s)", args.legacy, args.sheet)
    legacy_df = pd.read_excel(args.legacy, sheet_name=args.sheet)
    if legacy_df.empty:
        raise SystemExit(f"No rows in legacy sheet {args.sheet} of {args.legacy}")
    LOGGER.info("Loaded %d rows from legacy Excel", len(legacy_df))

    if "uuid_paragraph" not in legacy_df.columns:
        raise SystemExit("Legacy file is missing 'uuid_paragraph' column.")

    if "prominence" not in legacy_df.columns:
        raise SystemExit("Legacy file does not contain a 'prominence' column.")

    # Keep only rows that actually have a prominence label
    legacy_df = legacy_df[legacy_df["prominence"].notna()].copy()
    LOGGER.info("Filtered to %d rows with non-null prominence labels", len(legacy_df))

    # 3) Normalize UUID columns to strings for matching
    win_df["paragraph_uuid"] = win_df["paragraph_uuid"].astype(str)
    legacy_df["uuid_paragraph"] = legacy_df["uuid_paragraph"].astype(str)

    # 4) Merge on paragraph_uuid == uuid_paragraph ONLY
    LOGGER.info("Merging on paragraph_uuid == uuid_paragraph ...")
    merged = win_df.merge(
        legacy_df[["uuid_paragraph", "prominence"]],
        left_on="paragraph_uuid",
        right_on="uuid_paragraph",
        how="inner",
    )

    if merged.empty:
        raise SystemExit(
            "Merge produced 0 rows. Check that paragraph_uuid matches legacy uuid_paragraph."
        )

    LOGGER.info("Successfully merged %d labeled windows", len(merged))

    # Rename label column to something model-friendly
    merged = merged.rename(columns={"prominence": "label_prominent"})

    # Drop the duplicate uuid_paragraph column from legacy
    if "uuid_paragraph" in merged.columns:
        merged = merged.drop(columns=["uuid_paragraph"])

    # 5) Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False, encoding="utf-8-sig")

    LOGGER.info("✅ Wrote merged labeled windows to: %s", args.out)
    LOGGER.info("Label distribution:")
    LOGGER.info("\n%s", merged["label_prominent"].value_counts().to_string())


if __name__ == "__main__":
    main()
