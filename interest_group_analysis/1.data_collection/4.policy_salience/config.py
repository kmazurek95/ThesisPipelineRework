"""Configuration settings for the policy salience pipeline."""
import os

# Paths
BASE_DIR = os.environ.get("BASE_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")

# Google Trends settings
CONSTANT_TOPIC = "Economy"
POLICY_TOPICS = [
    'Civil Rights', 'Healthcare', 'Agriculture', 'Employment', 'Education Reform',
    'Climate Change', 'Energy', 'Immigration Policy', 'Infrastructure', 'Law Enforcement',
    'Welfare Policy', 'Affordable Housing', 'Trade Policy', 'National Security',
    'Innovation', 'International Trade', 'Foreign Policy', 'Public Administration',
    'National Parks', 'Arts and Culture'
]
GROUP_SIZE = 4
API_SLEEP = 20  # seconds between API calls

# Data file paths
GRANULE_FILE = os.path.join(BASE_DIR, "g.graule_meta_data_CREC_114_AND_115.csv")
PROMINENCE_FILE = os.path.join(BASE_DIR, "paragraphs_NAME_114_115_EXPANDED_CLASSIFIED__UPDATED__4-29-2023____3B.json")