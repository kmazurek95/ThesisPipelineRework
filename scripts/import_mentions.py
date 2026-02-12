#!/usr/bin/env python3
"""
Import mentions from mentions.jsonl into the labeling database.

Usage:
    python scripts/import_mentions.py
    python scripts/import_mentions.py --input data/processed/mentions/mentions.jsonl
    python scripts/import_mentions.py --clear  # Clear existing mentions before import
"""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from datetime import datetime

from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "mentions" / "mentions.jsonl"
DB_PATH = PROJECT_ROOT / "data" / "labeling" / "labeling.db"


def generate_mention_id(record: dict) -> str:
    """Generate a unique mention ID from org_id, granuleId, and char offsets."""
    key = f"{record.get('org_id', '')}|{record.get('granuleId', '')}|{record.get('mention_char_start', '')}|{record.get('mention_char_end', '')}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def import_mentions(input_path: Path, clear: bool = False):
    """Import mentions from JSONL file into SQLite database."""
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
        print("Clearing existing mentions...")
        cursor.execute("DELETE FROM mentions")
        conn.commit()

    # Count lines for progress bar
    print(f"Counting records in {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    print(f"Found {total_lines:,} records")

    # Import records
    print("Importing mentions...")
    imported = 0
    skipped = 0
    errors = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Importing"):
            try:
                record = json.loads(line.strip())
                mention_id = generate_mention_id(record)

                # Check if already exists
                cursor.execute("SELECT 1 FROM mentions WHERE mention_id = ?", (mention_id,))
                if cursor.fetchone():
                    skipped += 1
                    continue

                # Insert record
                cursor.execute("""
                    INSERT INTO mentions (
                        mention_id, org_id, interest_group, variation, match_text,
                        match_type, is_acronym, granuleId, packageId, date, title,
                        sentence, paragraph, sentence_index,
                        start_in_sentence, end_in_sentence,
                        mention_char_start, mention_char_end,
                        paragraph_char_start, paragraph_char_end,
                        speaker_raw, speaker_canonical, speaker_bioguide,
                        speaker_method, speaker_confidence, text_source,
                        source_file, imported_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    mention_id,
                    record.get("org_id"),
                    record.get("interest_group"),
                    record.get("variation"),
                    record.get("match_text"),
                    record.get("match_type"),
                    record.get("is_acronym"),
                    record.get("granuleId"),
                    record.get("packageId"),
                    record.get("date"),
                    record.get("title"),
                    record.get("sentence"),
                    record.get("paragraph"),
                    record.get("sentence_index"),
                    record.get("start_in_sentence"),
                    record.get("end_in_sentence"),
                    record.get("mention_char_start"),
                    record.get("mention_char_end"),
                    record.get("paragraph_char_start"),
                    record.get("paragraph_char_end"),
                    record.get("speaker_raw"),
                    record.get("speaker_canonical"),
                    record.get("speaker_bioguide"),
                    record.get("speaker_method"),
                    record.get("speaker_confidence"),
                    record.get("text_source"),
                    str(input_path.name),
                    datetime.now().isoformat(),
                ))
                imported += 1

                # Commit in batches
                if imported % 1000 == 0:
                    conn.commit()

            except json.JSONDecodeError as e:
                errors += 1
                if errors <= 5:
                    print(f"JSON error: {e}")
            except sqlite3.Error as e:
                errors += 1
                if errors <= 5:
                    print(f"SQLite error: {e}")

    conn.commit()

    # Print summary
    print("\n" + "=" * 50)
    print("IMPORT SUMMARY")
    print("=" * 50)
    print(f"Total records:   {total_lines:,}")
    print(f"Imported:        {imported:,}")
    print(f"Skipped (dupes): {skipped:,}")
    print(f"Errors:          {errors:,}")

    # Verify count
    cursor.execute("SELECT COUNT(*) FROM mentions")
    db_count = cursor.fetchone()[0]
    print(f"\nTotal in database: {db_count:,}")

    # Show sample stats
    cursor.execute("""
        SELECT
            COUNT(DISTINCT org_id) as unique_orgs,
            COUNT(DISTINCT granuleId) as unique_granules,
            MIN(date) as min_date,
            MAX(date) as max_date
        FROM mentions
    """)
    stats = cursor.fetchone()
    print(f"\nUnique organizations: {stats[0]:,}")
    print(f"Unique granules:      {stats[1]:,}")
    print(f"Date range:           {stats[2]} to {stats[3]}")

    conn.close()
    print("\nImport complete!")


def main():
    parser = argparse.ArgumentParser(description="Import mentions into labeling database")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"Path to mentions.jsonl (default: {DEFAULT_INPUT})")
    parser.add_argument("--clear", action="store_true",
                        help="Clear existing mentions before import")
    args = parser.parse_args()

    import_mentions(args.input, clear=args.clear)


if __name__ == "__main__":
    main()
