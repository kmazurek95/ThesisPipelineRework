#!/usr/bin/env python3
"""
Collect Congressional member data from Congress.gov API.

This script fetches member profiles, terms, and committee assignments
for members who appear in our mentions data.

Usage:
    python scripts/collect_members.py
    python scripts/collect_members.py --congress 114 115
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "members"
NORMALIZED_DIR = PROJECT_ROOT / "data" / "normalized_114"

# API Configuration
CONGRESS_API_KEY = os.environ.get("CONGRESS_API_KEY", "")
CONGRESS_API_BASE = "https://api.congress.gov/v3"
RATE_LIMIT_DELAY = 0.5  # Seconds between requests


def get_session() -> requests.Session:
    """Create a session with retry logic."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def extract_bioguide_ids_from_normalized() -> Set[str]:
    """Extract unique bioGuideIds from normalized member data."""
    logger.info("Extracting bioGuideIds from normalized data...")

    bioguide_ids = set()
    by_package_dir = NORMALIZED_DIR / "by_package"

    if not by_package_dir.exists():
        logger.warning(f"Normalized directory not found: {by_package_dir}")
        return bioguide_ids

    for pkg_dir in by_package_dir.iterdir():
        if not pkg_dir.is_dir():
            continue

        members_file = pkg_dir / "granule_members.csv"
        if members_file.exists():
            try:
                df = pd.read_csv(members_file)
                if 'bioGuideId' in df.columns:
                    ids = df['bioGuideId'].dropna().unique()
                    bioguide_ids.update(ids)
            except Exception as e:
                logger.warning(f"Error reading {members_file}: {e}")

    logger.info(f"Found {len(bioguide_ids)} unique bioGuideIds")
    return bioguide_ids


def fetch_member_profile(session: requests.Session, bioguide_id: str) -> Optional[Dict]:
    """Fetch member profile from Congress.gov API."""
    url = f"{CONGRESS_API_BASE}/member/{bioguide_id}"
    params = {"api_key": CONGRESS_API_KEY, "format": "json"}

    try:
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("member", {})
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error fetching {bioguide_id}: {e}")
        return None


def fetch_member_terms(member_data: Dict) -> List[Dict]:
    """Extract term information from member profile."""
    terms = []
    if not member_data:
        return terms

    for term in member_data.get("terms", []):
        terms.append({
            "bioGuideId": member_data.get("bioguideId"),
            "congress": term.get("congress"),
            "chamber": term.get("chamber"),
            "state": term.get("state"),
            "district": term.get("district"),
            "party": term.get("party"),
            "startDate": term.get("startDate"),
            "endDate": term.get("endDate"),
        })

    return terms


def process_member_data(member_data: Dict) -> Dict:
    """Process raw member data into flat record."""
    if not member_data:
        return {}

    # Extract party history
    party_history = member_data.get("partyHistory", [])
    current_party = party_history[0].get("party") if party_history else None

    return {
        "bioGuideId": member_data.get("bioguideId"),
        "fullName": member_data.get("directOrderName"),
        "firstName": member_data.get("firstName"),
        "lastName": member_data.get("lastName"),
        "middleName": member_data.get("middleName"),
        "birthYear": member_data.get("birthYear"),
        "currentParty": current_party,
        "state": member_data.get("state"),
        "officialWebsiteUrl": member_data.get("officialWebsiteUrl"),
        "updateDate": member_data.get("updateDate"),
    }


def main():
    parser = argparse.ArgumentParser(description="Collect Congressional member data")
    parser.add_argument(
        '--congress',
        nargs='+',
        type=int,
        default=[114, 115],
        help='Congress numbers to fetch (default: 114 115)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of members to fetch (for testing)'
    )
    args = parser.parse_args()

    if not CONGRESS_API_KEY:
        logger.error("CONGRESS_API_KEY not set. Please set it in .env or environment.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Congressional Member Data Collection")
    logger.info("=" * 60)
    logger.info(f"Congresses: {args.congress}")

    # Get bioGuideIds from normalized data
    bioguide_ids = extract_bioguide_ids_from_normalized()

    if not bioguide_ids:
        logger.error("No bioGuideIds found in normalized data")
        return

    if args.limit:
        bioguide_ids = set(list(bioguide_ids)[:args.limit])
        logger.info(f"Limited to {args.limit} members for testing")

    session = get_session()

    members = []
    terms = []
    errors = []

    logger.info(f"Fetching profiles for {len(bioguide_ids)} members...")

    for bioguide_id in tqdm(bioguide_ids):
        member_data = fetch_member_profile(session, bioguide_id)

        if member_data:
            members.append(process_member_data(member_data))
            terms.extend(fetch_member_terms(member_data))
        else:
            errors.append(bioguide_id)

        time.sleep(RATE_LIMIT_DELAY)

    # Save results
    members_df = pd.DataFrame(members)
    terms_df = pd.DataFrame(terms)

    members_path = OUTPUT_DIR / "congress_members.csv"
    terms_path = OUTPUT_DIR / "congress_member_terms.csv"

    members_df.to_csv(members_path, index=False)
    terms_df.to_csv(terms_path, index=False)

    logger.info(f"\nSaved {len(members_df)} members to {members_path}")
    logger.info(f"Saved {len(terms_df)} terms to {terms_path}")

    if errors:
        errors_path = OUTPUT_DIR / "fetch_errors.json"
        with open(errors_path, 'w') as f:
            json.dump(errors, f)
        logger.warning(f"Failed to fetch {len(errors)} members (see {errors_path})")

    # Summary
    logger.info("\n=== Summary ===")
    logger.info(f"Members fetched: {len(members_df)}")
    logger.info(f"Terms fetched: {len(terms_df)}")
    if not members_df.empty:
        logger.info(f"Party distribution: {members_df['currentParty'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
