"""
General utility functions for data processing.

This module provides helper functions used across multiple stages of
data processing, such as text normalisation, duplicate detection, and
column renaming.  Additional helpers can be added here as needed.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd
from nltk.corpus import stopwords


def normalise_text(text: str) -> str:
    """Lowercase and remove non‑alphanumeric characters from a string."""
    if not isinstance(text, str):
        return ""
    lower = text.lower()
    # Replace non‑alphanumeric characters with spaces
    cleaned = re.sub(r"[^a-z0-9]+", " ", lower)
    # Collapse multiple spaces and strip
    return re.sub(r"\s+", " ", cleaned).strip()


def remove_stopwords(tokens: Iterable[str]) -> list[str]:
    """Remove English stopwords from a list of tokens."""
    stops = set(stopwords.words("english"))
    return [tok for tok in tokens if tok not in stops]


def safe_merge(
    df_left: pd.DataFrame, df_right: pd.DataFrame, on: str, how: str = "left"
) -> pd.DataFrame:
    """Merge two dataframes and log the number of resulting rows.

    Useful for debugging join operations.
    """
    before = len(df_left)
    merged = df_left.merge(df_right, on=on, how=how)
    after = len(merged)
    print(f"Merged {before} -> {after} rows on '{on}'")
    return merged
