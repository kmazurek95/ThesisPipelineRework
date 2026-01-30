#!/usr/bin/env python3
"""
Collect bill metadata and classify by CAP policy area.

This script extracts bill references from normalized CREC data and fetches
bill metadata from GovInfo API.

Usage:
    python scripts/collect_bills.py
    python scripts/collect_bills.py --limit 100  # Test with 100 bills
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "bills"
NORMALIZED_DIR = PROJECT_ROOT / "data" / "normalized_114"

# API Configuration
GOVINFO_API_KEY = os.environ.get("GOVINFO_API_KEY", "")
GOVINFO_API_BASE = "https://api.govinfo.gov"
RATE_LIMIT_DELAY = 0.35  # ~3 requests per second

# Bill type normalization
BILL_TYPE_MAP = {
    'HR': ['H.R.', 'HR', 'H R', 'HOUSE BILL'],
    'S': ['S.', 'S', 'SENATE BILL'],
    'HRES': ['H.RES.', 'H RES', 'HRES', 'HOUSE RESOLUTION'],
    'SRES': ['S.RES.', 'S RES', 'SRES', 'SENATE RESOLUTION'],
    'HJRES': ['H.J.RES.', 'HJRES', 'HOUSE JOINT RESOLUTION'],
    'SJRES': ['S.J.RES.', 'SJRES', 'SENATE JOINT RESOLUTION'],
    'HCONRES': ['H.CON.RES.', 'HCONRES'],
    'SCONRES': ['S.CON.RES.', 'SCONRES'],
}

# Bill version preference order
VERSION_PREFERENCE = ['enr', 'eh', 'rh', 'rfs', 'rds', 'ih', 'is']


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


def extract_bill_references_from_normalized() -> List[Dict]:
    """Extract bill references from normalized CREC data."""
    logger.info("Extracting bill references from normalized data...")

    references = []
    by_package_dir = NORMALIZED_DIR / "by_package"

    if not by_package_dir.exists():
        logger.warning(f"Normalized directory not found: {by_package_dir}")
        return references

    for pkg_dir in by_package_dir.iterdir():
        if not pkg_dir.is_dir():
            continue

        refs_file = pkg_dir / "granule_references.csv"
        if refs_file.exists():
            try:
                df = pd.read_csv(refs_file)
                for _, row in df.iterrows():
                    # Try both column naming conventions
                    number = row.get('contents__number') or row.get('number')
                    congress = row.get('contents__congress') or row.get('congress')
                    bill_type = row.get('contents__type') or row.get('type')

                    if pd.notna(number):
                        references.append({
                            'granuleId': row.get('granuleId'),
                            'congress': congress,
                            'type': bill_type,
                            'number': number,
                        })
            except Exception as e:
                logger.warning(f"Error reading {refs_file}: {e}")

    logger.info(f"Found {len(references)} bill references")
    return references


def normalize_bill_type(bill_type: str) -> Optional[str]:
    """Normalize bill type to standard format."""
    if not bill_type:
        return None

    bill_type = str(bill_type).upper().strip()

    for normalized, variants in BILL_TYPE_MAP.items():
        for variant in variants:
            if bill_type == variant.upper():
                return normalized

    return None


def build_bill_package_id(congress: int, bill_type: str, bill_number: int, version: str = 'ih') -> str:
    """Build GovInfo package ID for a bill."""
    return f"BILLS-{congress}{bill_type.lower()}{bill_number}{version}"


def fetch_bill_metadata(session: requests.Session, package_id: str) -> Optional[Dict]:
    """Fetch bill metadata from GovInfo API."""
    url = f"{GOVINFO_API_BASE}/packages/{package_id}/summary"
    params = {"api_key": GOVINFO_API_KEY}

    try:
        response = session.get(url, params=params, timeout=30)

        if response.status_code == 404:
            return None

        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.debug(f"Error fetching {package_id}: {e}")
        return None


def fetch_bill_with_fallback(
    session: requests.Session,
    congress: int,
    bill_type: str,
    bill_number: int
) -> Optional[Dict]:
    """Try fetching bill with multiple version fallbacks."""
    for version in VERSION_PREFERENCE:
        package_id = build_bill_package_id(congress, bill_type, bill_number, version)
        metadata = fetch_bill_metadata(session, package_id)

        if metadata:
            metadata['version'] = version
            metadata['packageId'] = package_id
            return metadata

        time.sleep(RATE_LIMIT_DELAY)

    return None


def deduplicate_bills(references: List[Dict]) -> List[Tuple[int, str, int]]:
    """Get unique (congress, type, number) tuples from references."""
    unique = set()

    for ref in references:
        congress = ref.get('congress')
        bill_type = normalize_bill_type(ref.get('type'))
        number = ref.get('number')

        if congress and bill_type and number:
            try:
                unique.add((int(congress), bill_type, int(number)))
            except (ValueError, TypeError):
                continue

    return list(unique)


def main():
    parser = argparse.ArgumentParser(description="Collect bill metadata")
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of bills to fetch (for testing)'
    )
    args = parser.parse_args()

    if not GOVINFO_API_KEY:
        logger.error("GOVINFO_API_KEY not set. Please set it in .env or environment.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Bill Data Collection")
    logger.info("=" * 60)

    # Extract bill references
    references = extract_bill_references_from_normalized()

    if not references:
        logger.error("No bill references found in normalized data")
        return

    # Deduplicate
    unique_bills = deduplicate_bills(references)
    logger.info(f"Unique bills to fetch: {len(unique_bills)}")

    if args.limit:
        unique_bills = unique_bills[:args.limit]
        logger.info(f"Limited to {args.limit} bills for testing")

    session = get_session()

    bills = []
    not_found = []

    logger.info(f"Fetching metadata for {len(unique_bills)} bills...")

    for congress, bill_type, number in tqdm(unique_bills):
        metadata = fetch_bill_with_fallback(session, congress, bill_type, number)

        if metadata:
            bills.append({
                'congress': congress,
                'billType': bill_type,
                'billNumber': number,
                'packageId': metadata.get('packageId'),
                'version': metadata.get('version'),
                'title': metadata.get('title'),
                'shortTitle': metadata.get('shortTitle', [{}])[0].get('title') if metadata.get('shortTitle') else None,
                'dateIssued': metadata.get('dateIssued'),
                'docClass': metadata.get('docClass'),
                'lastModified': metadata.get('lastModified'),
            })
        else:
            not_found.append((congress, bill_type, number))

        time.sleep(RATE_LIMIT_DELAY)

    # Save results
    bills_df = pd.DataFrame(bills)

    if not bills_df.empty:
        # Add bill number string for joining
        bills_df['billnumber_generated'] = (
            bills_df['billType'] + bills_df['billNumber'].astype(str)
        )

        bills_path = OUTPUT_DIR / "bill_metadata.csv"
        bills_df.to_csv(bills_path, index=False)
        logger.info(f"\nSaved {len(bills_df)} bills to {bills_path}")
    else:
        logger.warning("No bills fetched successfully")

    # Save reference-to-bill mapping
    refs_df = pd.DataFrame(references)
    refs_path = OUTPUT_DIR / "bill_references.csv"
    refs_df.to_csv(refs_path, index=False)
    logger.info(f"Saved {len(refs_df)} references to {refs_path}")

    if not_found:
        not_found_path = OUTPUT_DIR / "bills_not_found.json"
        with open(not_found_path, 'w') as f:
            json.dump([{"congress": c, "type": t, "number": n} for c, t, n in not_found], f)
        logger.info(f"Bills not found: {len(not_found)} (see {not_found_path})")

    # Summary
    logger.info("\n=== Summary ===")
    logger.info(f"Bills fetched: {len(bills_df)}")
    logger.info(f"Bills not found: {len(not_found)}")
    if not bills_df.empty:
        logger.info(f"Bill types: {bills_df['billType'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
