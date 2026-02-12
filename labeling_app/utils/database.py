"""Database utilities for the labeling application."""

import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import pandas as pd

# Database path
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "labeling" / "labeling.db"


def get_connection() -> sqlite3.Connection:
    """Get a connection to the labeling database."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. "
            "Run 'python scripts/init_labeling_db.py' first."
        )
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_labeling_progress() -> Dict[str, Any]:
    """Get overall labeling progress statistics."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM labeling_progress")
    row = cursor.fetchone()

    if row:
        result = {
            "total_mentions": row["total_mentions"],
            "labeled_mentions": row["labeled_mentions"],
            "unlabeled_mentions": row["unlabeled_mentions"],
            "pct_complete": row["pct_complete"] or 0.0,
        }
    else:
        result = {
            "total_mentions": 0,
            "labeled_mentions": 0,
            "unlabeled_mentions": 0,
            "pct_complete": 0.0,
        }

    # Get label distribution
    cursor.execute("""
        SELECT validity_label, COUNT(*) as count
        FROM mention_labels
        GROUP BY validity_label
    """)
    result["label_distribution"] = {
        row["validity_label"]: row["count"] for row in cursor.fetchall()
    }

    # Get false positive type distribution
    cursor.execute("""
        SELECT false_positive_type, COUNT(*) as count
        FROM mention_labels
        WHERE validity_label = 'false_positive' AND false_positive_type IS NOT NULL
        GROUP BY false_positive_type
    """)
    result["fp_type_distribution"] = {
        row["false_positive_type"]: row["count"] for row in cursor.fetchall()
    }

    conn.close()
    return result


def get_unlabeled_mentions(
    limit: int = 100,
    offset: int = 0,
    org_id: Optional[str] = None,
    match_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> pd.DataFrame:
    """Get unlabeled mentions with optional filters."""
    conn = get_connection()

    query = """
        SELECT m.*
        FROM mentions m
        LEFT JOIN mention_labels ml ON m.mention_id = ml.mention_id
        WHERE ml.label_id IS NULL
    """
    params = []

    if org_id:
        query += " AND m.org_id = ?"
        params.append(org_id)

    if match_type == "acronym":
        query += " AND m.is_acronym = 1"
    elif match_type == "name":
        query += " AND (m.is_acronym = 0 OR m.is_acronym IS NULL)"

    if date_from:
        query += " AND m.date >= ?"
        params.append(date_from)

    if date_to:
        query += " AND m.date <= ?"
        params.append(date_to)

    query += " ORDER BY m.date, m.granuleId LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_mention_by_id(mention_id: str) -> Optional[Dict[str, Any]]:
    """Get a single mention by ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM mentions WHERE mention_id = ?", (mention_id,))
    row = cursor.fetchone()

    if row:
        result = dict(row)
    else:
        result = None

    conn.close()
    return result


def save_mention_label(
    mention_id: str,
    validity_label: str,
    labeler_name: str,
    prominence_label: Optional[str] = None,
    false_positive_type: Optional[str] = None,
    correct_org_id: Optional[str] = None,
    correct_org_name: Optional[str] = None,
    confidence: str = "high",
    labeler_notes: Optional[str] = None,
) -> bool:
    """Save a label for a mention."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Check if label already exists
        cursor.execute(
            "SELECT label_id FROM mention_labels WHERE mention_id = ? AND labeler_name = ?",
            (mention_id, labeler_name)
        )
        existing = cursor.fetchone()

        if existing:
            # Update existing label
            cursor.execute("""
                UPDATE mention_labels SET
                    validity_label = ?,
                    prominence_label = ?,
                    false_positive_type = ?,
                    correct_org_id = ?,
                    correct_org_name = ?,
                    confidence = ?,
                    labeler_notes = ?,
                    labeled_at = ?
                WHERE mention_id = ? AND labeler_name = ?
            """, (
                validity_label, prominence_label, false_positive_type,
                correct_org_id, correct_org_name, confidence, labeler_notes,
                datetime.now().isoformat(), mention_id, labeler_name
            ))
        else:
            # Insert new label
            cursor.execute("""
                INSERT INTO mention_labels (
                    mention_id, validity_label, prominence_label,
                    false_positive_type, correct_org_id, correct_org_name,
                    confidence, labeler_notes, labeler_name, labeled_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                mention_id, validity_label, prominence_label,
                false_positive_type, correct_org_id, correct_org_name,
                confidence, labeler_notes, labeler_name, datetime.now().isoformat()
            ))

        conn.commit()
        conn.close()
        return True

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        conn.close()
        return False


def get_organizations() -> pd.DataFrame:
    """Get list of unique organizations in the database."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT DISTINCT org_id, interest_group, COUNT(*) as mention_count
        FROM mentions
        GROUP BY org_id, interest_group
        ORDER BY mention_count DESC
    """, conn)
    conn.close()
    return df


def get_false_positive_stats() -> pd.DataFrame:
    """Get false positive statistics by organization."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT * FROM false_positive_by_org
        ORDER BY fp_rate DESC
    """, conn)
    conn.close()
    return df


def get_fp_stats_by_match_type() -> pd.DataFrame:
    """Get false positive statistics by match type."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT * FROM false_positive_by_match_type
    """, conn)
    conn.close()
    return df


def get_labeled_mentions(
    limit: int = 100,
    validity_label: Optional[str] = None,
) -> pd.DataFrame:
    """Get labeled mentions for export."""
    conn = get_connection()

    query = """
        SELECT
            m.mention_id,
            m.org_id,
            m.interest_group,
            m.variation,
            m.granuleId,
            m.date,
            m.paragraph as p1_original,
            ml.validity_label,
            ml.prominence_label,
            ml.false_positive_type,
            ml.labeler_name,
            ml.labeled_at
        FROM mentions m
        JOIN mention_labels ml ON m.mention_id = ml.mention_id
    """
    params = []

    if validity_label:
        query += " WHERE ml.validity_label = ?"
        params.append(validity_label)

    query += " ORDER BY ml.labeled_at DESC LIMIT ?"
    params.append(limit)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def export_training_data(
    include_true_only: bool = True,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Export labeled data in format compatible with text_classifier.py."""
    conn = get_connection()

    query = """
        SELECT
            m.org_id,
            m.paragraph as p1_original,
            CASE
                WHEN ml.prominence_label = 'prominent' THEN 1
                WHEN ml.prominence_label = 'passing' THEN 0
                ELSE NULL
            END as prominence,
            1 as paragraph_mention_count,
            0 as "10_or_more_org_mentioned",
            m.mention_id as uuid_mention,
            m.mention_id as uuid_paragraph,
            m.granuleId,
            'labeling_app' as source,
            m.interest_group,
            m.variation
        FROM mentions m
        JOIN mention_labels ml ON m.mention_id = ml.mention_id
    """

    if include_true_only:
        query += " WHERE ml.validity_label = 'true_mention'"

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Filter out rows without prominence labels
    df = df.dropna(subset=["prominence"])
    df["prominence"] = df["prominence"].astype(int)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} rows to {output_path}")

    return df
