"""File input/output helper functions."""

import json
import logging
from pathlib import Path

import pandas as pd


def read_json(path):
    """Read a JSON file and return the loaded object."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logging.error("Failed to read JSON file %s: %s", path, exc)
        raise


def write_json(data, path):
    """Write a Python object to a JSON file."""
    path = Path(path)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logging.error("Failed to write JSON file %s: %s", path, exc)
        raise


def read_csv(path):
    """Read a CSV file into a DataFrame."""
    path = Path(path)
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logging.error("Failed to read CSV file %s: %s", path, exc)
        raise


def write_csv(df, path):
    """Write a DataFrame to a CSV file."""
    path = Path(path)
    try:
        df.to_csv(path, index=False)
    except Exception as exc:
        logging.error("Failed to write CSV file %s: %s", path, exc)
        raise