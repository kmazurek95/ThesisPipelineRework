#!/usr/bin/env python3
"""
Import legacy labeled data from combined_labeled.csv into the labeling database.

Usage:
    python scripts/import_legacy_labels.py
    python scripts/import_legacy_labels.py --input data/training/combined_labeled.csv
    python scripts/import_legacy_labels.py --clear  # Clear existing legacy labels
"""

import argparse
import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "training" / "combined_labeled.csv"
DB_PATH = PROJECT_ROOT / "data" / "labeling" / "labeling.db"


def import_legacy_labels(input_path: Path, clear: bool = False):
    """Import legacy labels from CSV into SQLite database."""
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        print("Run 'python scripts/init_labeling_db.py' first.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if clear:
        print("Clearing existing legacy labels...")
        cursor.execute("DELETE FROM legacy_labels")
        conn.commit()

    # Load CSV
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path, dtype=str)
    df = df.fillna("")

    print(f"Found {len(df):,} labeled records")

    # Import records
    print("Importing legacy labels...")
    imported = 0
    errors = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Importing"):
        try:
            cursor.execute("""
                INSERT INTO legacy_labels (
                    uuid_mention, uuid_paragraph, granuleId, org_id,
                    interest_group, variation, prominence,
                    paragraph_mention_count, ten_or_more_org_mentioned,
                    p1_original, source_file, imported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row.get("uuid_mention", ""),
                row.get("uuid_paragraph", ""),
                row.get("granuleId", ""),
                row.get("org_id", ""),
                row.get("interest_group", ""),
                row.get("variation", ""),
                int(row.get("prominence", 0)) if row.get("prominence", "").strip() else 0,
                int(row.get("paragraph_mention_count", 0)) if row.get("paragraph_mention_count", "").strip() else 0,
                int(row.get("10_or_more_org_mentioned", 0)) if row.get("10_or_more_org_mentioned", "").strip() else 0,
                row.get("p1_original", ""),
                row.get("source", str(input_path.name)),
                datetime.now().isoformat(),
            ))
            imported += 1

        except sqlite3.Error as e:
            errors += 1
            if errors <= 5:
                print(f"SQLite error: {e}")

    conn.commit()

    # Print summary
    print("\n" + "=" * 50)
    print("IMPORT SUMMARY")
    print("=" * 50)
    print(f"Total records:  {len(df):,}")
    print(f"Imported:       {imported:,}")
    print(f"Errors:         {errors:,}")

    # Verify count
    cursor.execute("SELECT COUNT(*) FROM legacy_labels")
    db_count = cursor.fetchone()[0]
    print(f"\nTotal in database: {db_count:,}")

    # Show prominence distribution
    cursor.execute("""
        SELECT
            prominence,
            COUNT(*) as count
        FROM legacy_labels
        GROUP BY prominence
        ORDER BY prominence
    """)
    print("\nProminence distribution:")
    for row in cursor.fetchall():
        label = "Prominent" if row[0] == 1 else "Passing"
        print(f"  {label}: {row[1]:,}")

    # Show top organizations
    cursor.execute("""
        SELECT interest_group, COUNT(*) as count
        FROM legacy_labels
        GROUP BY interest_group
        ORDER BY count DESC
        LIMIT 10
    """)
    print("\nTop 10 organizations in legacy data:")
    for row in cursor.fetchall():
        print(f"  {row[0][:50]}: {row[1]}")

    conn.close()
    print("\nImport complete!")


def main():
    parser = argparse.ArgumentParser(description="Import legacy labels into labeling database")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"Path to combined_labeled.csv (default: {DEFAULT_INPUT})")
    parser.add_argument("--clear", action="store_true",
                        help="Clear existing legacy labels before import")
    args = parser.parse_args()

    import_legacy_labels(args.input, clear=args.clear)


if __name__ == "__main__":
    main()
