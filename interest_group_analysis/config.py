"""
Project configuration settings.

Edit the variables in this module to point to your data directories and
API keys. Keeping configuration in one place makes it easy to
override default behavior without modifying individual modules.
"""

from pathlib import Path
import os

try:  # optional dotenv load
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base directory for storing input and output data
BASE_DIR: Path = Path(__file__).resolve().parents[1]

###############################################################################
# API Keys
###############################################################################

# Prefer environment variables over hard-coded values
GOVINFO_API_KEY: str | None = os.environ.get("GOVINFO_API_KEY")
CONGRESS_API_KEY: str | None = os.environ.get("CONGRESS_API_KEY")

###############################################################################
# Directory paths
###############################################################################

# Raw data collected from external APIs will be stored here
RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"

# Normalized data after parsing raw API responses
NORMALIZED_DATA_DIR: Path = BASE_DIR / "data" / "normalized"

# Processed datasets will be stored here
PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"

# Classification models and predictions
CLASSIFIER_DIR: Path = BASE_DIR / "results" / "classifier"

# Integrated datasets with all features
RESULTS_DIR: Path = BASE_DIR / "results"

# Sample data for testing
SAMPLE_DATA_DIR: Path = BASE_DIR / "data" / "sample"

# Create directories if they don't exist
for _dir in (RAW_DATA_DIR, NORMALIZED_DATA_DIR, PROCESSED_DATA_DIR, CLASSIFIER_DIR, RESULTS_DIR, SAMPLE_DATA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

###############################################################################
# Policy Salience Configuration
###############################################################################

# Constant reference topic for Google Trends
CONSTANT_TOPIC = "Economy"

# Policy topics to track
POLICY_TOPICS = [
    'Civil Rights', 'Healthcare', 'Agriculture', 'Employment', 'Education Reform',
    'Climate Change', 'Energy', 'Immigration Policy', 'Infrastructure', 'Law Enforcement',
    'Welfare Policy', 'Affordable Housing', 'Trade Policy', 'National Security',
    'Innovation', 'International Trade', 'Foreign Policy', 'Public Administration',
    'National Parks', 'Arts and Culture'
]

# Topics group size for API calls
GROUP_SIZE = 4

# Delay between API calls (seconds)
API_SLEEP = 20

###############################################################################
# Congress Information
###############################################################################

# Mapping of Congress number to years
CONGRESS_WINDOWS = {
    114: (2015, 2017),
    115: (2017, 2019),
    116: (2019, 2021),
    117: (2021, 2023),
    118: (2023, 2025),
}

# Target congresses for analysis
TARGET_CONGRESSES = [114, 115]
