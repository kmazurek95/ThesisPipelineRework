"""
Project configuration settings.

Edit the variables in this module to point to your data directories and
API keys.  Keeping configuration in one place makes it easy to
override default behaviour without modifying individual modules.
"""

from pathlib import Path
import os

try:  # optional dotenv load
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Base directory for storing input and output data.
BASE_DIR: Path = Path(__file__).resolve().parents[1]

###############################################################################
# API Keys
###############################################################################

# Replace the placeholder strings below with your actual API credentials.
# Prefer environment variable over hard-coded fallback. Remove the literal key once env is set.
GOVINFO_API_KEY: str | None = os.getenv("GOVINFO_API_KEY")
CONGRESS_API_KEY: str | None = None
GOOGLE_TRENDS_API_KEY: str | None = None

###############################################################################
# Directory paths
###############################################################################

# Raw data collected from external APIs will be stored here
RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"

# Processed/cleaned datasets will be stored here
PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"

# Intermediate outputs and results (predictions, models, etc.)
RESULTS_DIR: Path = BASE_DIR / "results"

# Create directories if they do not already exist
for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
