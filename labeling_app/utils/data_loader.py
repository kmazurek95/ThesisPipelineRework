"""Data loading utilities with Streamlit caching."""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List

from .database import (
    get_connection,
    get_unlabeled_mentions,
    get_mention_by_id,
    get_organizations,
    get_labeling_progress,
    get_false_positive_stats,
    get_fp_stats_by_match_type,
)


@st.cache_data(ttl=60)
def load_mentions(
    limit: int = 100,
    offset: int = 0,
    org_id: Optional[str] = None,
    match_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    labeled: bool = False,
) -> pd.DataFrame:
    """Load mentions with caching."""
    if labeled:
        from .database import get_labeled_mentions
        return get_labeled_mentions(limit=limit)
    else:
        return get_unlabeled_mentions(
            limit=limit,
            offset=offset,
            org_id=org_id,
            match_type=match_type,
            date_from=date_from,
            date_to=date_to,
        )


def load_mention_by_id(mention_id: str) -> Optional[Dict[str, Any]]:
    """Load a single mention by ID (no caching for real-time updates)."""
    return get_mention_by_id(mention_id)


@st.cache_data(ttl=60)
def load_organizations() -> pd.DataFrame:
    """Load organizations with caching."""
    return get_organizations()


@st.cache_data(ttl=30)
def load_progress() -> Dict[str, Any]:
    """Load labeling progress with short cache."""
    return get_labeling_progress()


@st.cache_data(ttl=60)
def load_fp_stats() -> pd.DataFrame:
    """Load false positive statistics."""
    return get_false_positive_stats()


@st.cache_data(ttl=60)
def load_fp_by_match_type() -> pd.DataFrame:
    """Load false positive stats by match type."""
    return get_fp_stats_by_match_type()


def check_database_exists() -> bool:
    """Check if the labeling database exists."""
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DB_PATH = PROJECT_ROOT / "data" / "labeling" / "labeling.db"
    return DB_PATH.exists()


def get_date_range() -> tuple:
    """Get the min and max dates in the mentions table."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT MIN(date), MAX(date) FROM mentions")
        result = cursor.fetchone()
        conn.close()
        return result[0], result[1]
    except Exception:
        return None, None
