#!/usr/bin/env python3
"""
Initialize the labeling database schema.

Usage:
    python scripts/init_labeling_db.py
    python scripts/init_labeling_db.py --reset  # Drop and recreate all tables
"""

import argparse
import sqlite3
from pathlib import Path

# Database location
DB_DIR = Path(__file__).parent.parent / "data" / "labeling"
DB_PATH = DB_DIR / "labeling.db"

SCHEMA = """
-- ============================================================================
-- LABELING DATABASE SCHEMA
-- ============================================================================

-- Imported mentions (immutable reference from mentions.jsonl)
CREATE TABLE IF NOT EXISTS mentions (
    mention_id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    interest_group TEXT,
    variation TEXT,
    match_text TEXT,
    match_type TEXT,
    is_acronym BOOLEAN,
    granuleId TEXT NOT NULL,
    packageId TEXT,
    date TEXT,
    title TEXT,
    sentence TEXT,
    paragraph TEXT,
    sentence_index INTEGER,
    start_in_sentence INTEGER,
    end_in_sentence INTEGER,
    mention_char_start INTEGER,
    mention_char_end INTEGER,
    paragraph_char_start INTEGER,
    paragraph_char_end INTEGER,
    speaker_raw TEXT,
    speaker_canonical TEXT,
    speaker_bioguide TEXT,
    speaker_method TEXT,
    speaker_confidence REAL,
    text_source TEXT,
    source_file TEXT,
    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_mentions_org ON mentions(org_id);
CREATE INDEX IF NOT EXISTS idx_mentions_granule ON mentions(granuleId);
CREATE INDEX IF NOT EXISTS idx_mentions_date ON mentions(date);
CREATE INDEX IF NOT EXISTS idx_mentions_match_type ON mentions(match_type);
CREATE INDEX IF NOT EXISTS idx_mentions_is_acronym ON mentions(is_acronym);

-- Mention validity labels
CREATE TABLE IF NOT EXISTS mention_labels (
    label_id INTEGER PRIMARY KEY AUTOINCREMENT,
    mention_id TEXT NOT NULL REFERENCES mentions(mention_id),
    validity_label TEXT CHECK(validity_label IN (
        'true_mention', 'false_positive', 'ambiguous', 'needs_review', 'wrong_entity'
    )),
    correct_org_id TEXT,
    correct_org_name TEXT,
    prominence_label TEXT CHECK(prominence_label IN ('prominent', 'passing', 'unclear')),
    false_positive_type TEXT CHECK(false_positive_type IN (
        'person_name', 'location', 'different_org', 'partial_match',
        'procedural', 'abbreviation_clash', 'historical', 'other'
    )),
    confidence TEXT CHECK(confidence IN ('high', 'medium', 'low')),
    labeler_notes TEXT,
    labeler_name TEXT NOT NULL,
    labeled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(mention_id, labeler_name)
);

CREATE INDEX IF NOT EXISTS idx_labels_validity ON mention_labels(validity_label);
CREATE INDEX IF NOT EXISTS idx_labels_labeler ON mention_labels(labeler_name);
CREATE INDEX IF NOT EXISTS idx_labels_fp_type ON mention_labels(false_positive_type);
CREATE INDEX IF NOT EXISTS idx_labels_mention ON mention_labels(mention_id);

-- Speaker validation labels
CREATE TABLE IF NOT EXISTS speaker_labels (
    label_id INTEGER PRIMARY KEY AUTOINCREMENT,
    mention_id TEXT NOT NULL REFERENCES mentions(mention_id),
    speaker_validation TEXT CHECK(speaker_validation IN (
        'correct', 'incorrect', 'unknown', 'multiple'
    )),
    corrected_speaker_raw TEXT,
    corrected_speaker_canonical TEXT,
    corrected_speaker_bioguide TEXT,
    correction_reason TEXT,
    labeler_name TEXT NOT NULL,
    labeled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(mention_id, labeler_name)
);

CREATE INDEX IF NOT EXISTS idx_speaker_validation ON speaker_labels(speaker_validation);
CREATE INDEX IF NOT EXISTS idx_speaker_labeler ON speaker_labels(labeler_name);

-- Legacy labels (preserved from combined_labeled.csv)
CREATE TABLE IF NOT EXISTS legacy_labels (
    legacy_id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid_mention TEXT,
    uuid_paragraph TEXT,
    granuleId TEXT,
    org_id TEXT,
    interest_group TEXT,
    variation TEXT,
    prominence INTEGER,
    paragraph_mention_count INTEGER,
    ten_or_more_org_mentioned INTEGER,
    p1_original TEXT,
    source_file TEXT,
    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_legacy_org ON legacy_labels(org_id);
CREATE INDEX IF NOT EXISTS idx_legacy_granule ON legacy_labels(granuleId);
CREATE INDEX IF NOT EXISTS idx_legacy_prominence ON legacy_labels(prominence);

-- Labeling sessions for progress tracking
CREATE TABLE IF NOT EXISTS labeling_sessions (
    session_id TEXT PRIMARY KEY,
    labeler_name TEXT NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    description TEXT,
    target_count INTEGER,
    completed_count INTEGER DEFAULT 0,
    session_type TEXT CHECK(session_type IN (
        'mention_validity', 'speaker_validation', 'prominence', 'review', 'batch'
    ))
);

-- Export history
CREATE TABLE IF NOT EXISTS export_history (
    export_id INTEGER PRIMARY KEY AUTOINCREMENT,
    export_type TEXT CHECK(export_type IN ('training', 'validation', 'full', 'analytics')),
    file_path TEXT,
    mention_count INTEGER,
    filters_applied TEXT,
    exported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    exported_by TEXT
);

-- Views for common queries
CREATE VIEW IF NOT EXISTS unlabeled_mentions AS
SELECT m.*
FROM mentions m
LEFT JOIN mention_labels ml ON m.mention_id = ml.mention_id
WHERE ml.label_id IS NULL;

CREATE VIEW IF NOT EXISTS labeled_summary AS
SELECT
    m.org_id,
    m.interest_group,
    m.match_type,
    m.is_acronym,
    ml.validity_label,
    ml.prominence_label,
    ml.false_positive_type,
    COUNT(*) as count
FROM mentions m
JOIN mention_labels ml ON m.mention_id = ml.mention_id
GROUP BY m.org_id, m.interest_group, m.match_type, m.is_acronym,
         ml.validity_label, ml.prominence_label, ml.false_positive_type;

CREATE VIEW IF NOT EXISTS labeling_progress AS
SELECT
    COUNT(DISTINCT m.mention_id) as total_mentions,
    COUNT(DISTINCT ml.mention_id) as labeled_mentions,
    COUNT(DISTINCT m.mention_id) - COUNT(DISTINCT ml.mention_id) as unlabeled_mentions,
    ROUND(100.0 * COUNT(DISTINCT ml.mention_id) / NULLIF(COUNT(DISTINCT m.mention_id), 0), 2) as pct_complete
FROM mentions m
LEFT JOIN mention_labels ml ON m.mention_id = ml.mention_id;

CREATE VIEW IF NOT EXISTS false_positive_by_org AS
SELECT
    m.org_id,
    m.interest_group,
    COUNT(*) as total_labeled,
    SUM(CASE WHEN ml.validity_label = 'false_positive' THEN 1 ELSE 0 END) as false_positives,
    ROUND(100.0 * SUM(CASE WHEN ml.validity_label = 'false_positive' THEN 1 ELSE 0 END) / COUNT(*), 2) as fp_rate
FROM mentions m
JOIN mention_labels ml ON m.mention_id = ml.mention_id
GROUP BY m.org_id, m.interest_group
HAVING COUNT(*) >= 3
ORDER BY fp_rate DESC;

CREATE VIEW IF NOT EXISTS false_positive_by_match_type AS
SELECT
    CASE WHEN m.is_acronym THEN 'acronym' ELSE 'name' END as match_type,
    COUNT(*) as total_labeled,
    SUM(CASE WHEN ml.validity_label = 'false_positive' THEN 1 ELSE 0 END) as false_positives,
    ROUND(100.0 * SUM(CASE WHEN ml.validity_label = 'false_positive' THEN 1 ELSE 0 END) / COUNT(*), 2) as fp_rate
FROM mentions m
JOIN mention_labels ml ON m.mention_id = ml.mention_id
GROUP BY CASE WHEN m.is_acronym THEN 'acronym' ELSE 'name' END;
"""


def init_database(reset: bool = False):
    """Initialize the database with the schema."""
    # Ensure directory exists
    DB_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if reset:
        print("Dropping existing tables...")
        # Drop views first
        cursor.execute("DROP VIEW IF EXISTS unlabeled_mentions")
        cursor.execute("DROP VIEW IF EXISTS labeled_summary")
        cursor.execute("DROP VIEW IF EXISTS labeling_progress")
        cursor.execute("DROP VIEW IF EXISTS false_positive_by_org")
        cursor.execute("DROP VIEW IF EXISTS false_positive_by_match_type")
        # Drop tables
        cursor.execute("DROP TABLE IF EXISTS export_history")
        cursor.execute("DROP TABLE IF EXISTS labeling_sessions")
        cursor.execute("DROP TABLE IF EXISTS speaker_labels")
        cursor.execute("DROP TABLE IF EXISTS mention_labels")
        cursor.execute("DROP TABLE IF EXISTS legacy_labels")
        cursor.execute("DROP TABLE IF EXISTS mentions")
        conn.commit()

    print(f"Creating database at: {DB_PATH}")
    cursor.executescript(SCHEMA)
    conn.commit()

    # Verify tables were created
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()
    print(f"Created tables: {[t[0] for t in tables]}")

    cursor.execute("SELECT name FROM sqlite_master WHERE type='view' ORDER BY name")
    views = cursor.fetchall()
    print(f"Created views: {[v[0] for v in views]}")

    conn.close()
    print("Database initialization complete!")


def main():
    parser = argparse.ArgumentParser(description="Initialize the labeling database")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate all tables")
    args = parser.parse_args()

    init_database(reset=args.reset)


if __name__ == "__main__":
    main()
