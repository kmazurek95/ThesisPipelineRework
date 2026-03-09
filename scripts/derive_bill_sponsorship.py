#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Derive Bill Sponsorship Counts for GLMM Replication

Fetches bill sponsorship counts per member from Congress.gov API
for the 114th-115th Congress.

Input:
    data/input/member_profiles_114_115.csv
    Congress.gov API (via CONGRESS_API_KEY in .env)

Output:
    data/input/member_bill_sponsorship.csv
    (updated analysis_dataset_replication.csv with bills_sponsored column)

Usage:
    python scripts/derive_bill_sponsorship.py
    python scripts/derive_bill_sponsorship.py --skip-api  # Use cached data only
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def fetch_sponsorship_counts(api_key: str, bioguide_ids: list[str],
                              cache_path: Path) -> dict[str, int]:
    """Fetch bill sponsorship counts from Congress.gov API."""
    import requests

    # Load cache
    cached = {}
    if cache_path.exists():
        cached_df = pd.read_csv(cache_path)
        cached = dict(zip(cached_df["bioguide_id"], cached_df["bills_sponsored"]))
        print(f"Loaded {len(cached)} cached members")

    results = dict(cached)
    to_fetch = [b for b in bioguide_ids if b not in cached]
    print(f"Need to fetch: {len(to_fetch)} members from API")

    for i, bio_id in enumerate(to_fetch):
        try:
            url = f"https://api.congress.gov/v3/member/{bio_id}/sponsored-legislation"
            r = requests.get(url, params={"api_key": api_key, "limit": 1}, timeout=30)

            if r.status_code == 200:
                data = r.json()
                count = data.get("pagination", {}).get("count", 0)
                results[bio_id] = count
            elif r.status_code == 429:
                print(f"\n  Rate limited at {i}. Waiting 60s...")
                time.sleep(60)
                # Retry
                r = requests.get(url, params={"api_key": api_key, "limit": 1}, timeout=30)
                if r.status_code == 200:
                    results[bio_id] = r.json().get("pagination", {}).get("count", 0)
                else:
                    results[bio_id] = np.nan
            else:
                results[bio_id] = np.nan

            if (i + 1) % 100 == 0:
                print(f"  Fetched {i + 1}/{len(to_fetch)}")
                # Save checkpoint
                _save_sponsorship(results, cache_path)

            time.sleep(0.5)

        except Exception as e:
            print(f"  Error for {bio_id}: {e}")
            results[bio_id] = np.nan
            time.sleep(2)

    # Final save
    _save_sponsorship(results, cache_path)
    return results


def _save_sponsorship(results: dict, path: Path):
    """Save sponsorship counts to CSV."""
    records = [{"bioguide_id": k, "bills_sponsored": v} for k, v in results.items()]
    pd.DataFrame(records).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-api", action="store_true",
                        help="Skip API calls, use cached data only")
    args = parser.parse_args()

    print("=" * 60)
    print("DERIVE BILL SPONSORSHIP")
    print("=" * 60)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("CONGRESS_API_KEY", "")

    # Load member profiles
    members = pd.read_csv(ROOT / "data" / "input" / "member_profiles_114_115.csv")
    unique_bios = members["bioguide_id"].unique().tolist()
    print(f"Unique members: {len(unique_bios)}")

    cache_path = ROOT / "data" / "input" / "member_bill_sponsorship.csv"

    if args.skip_api and cache_path.exists():
        print("Using cached sponsorship data only")
        cached_df = pd.read_csv(cache_path)
        sponsorship = dict(zip(cached_df["bioguide_id"], cached_df["bills_sponsored"]))
    elif api_key:
        sponsorship = fetch_sponsorship_counts(api_key, unique_bios, cache_path)
    else:
        print("No API key. Using bills_referenced as fallback.")
        sponsorship = {}

    if sponsorship:
        print(f"\nMembers with sponsorship data: {len(sponsorship)}")
        values = [v for v in sponsorship.values() if not np.isnan(v) if isinstance(v, (int, float))]
        if values:
            print(f"Bills sponsored: min={min(values):.0f}, max={max(values):.0f}, "
                  f"mean={np.mean(values):.1f}, median={np.median(values):.0f}")

    # Merge into analysis dataset
    analysis_path = ROOT / "data" / "output" / "analysis_dataset_replication.csv"
    df = pd.read_csv(analysis_path, low_memory=False)
    print(f"\nAnalysis dataset: {len(df):,} rows")

    if sponsorship:
        # Map bioGuideId -> bills_sponsored
        df["bills_sponsored"] = df["bioGuideId"].map(sponsorship)
        non_null = df["bills_sponsored"].notna().sum()
        print(f"bills_sponsored: {non_null:,} non-null")
        print(f"Distribution:\n{df['bills_sponsored'].describe()}")
    else:
        print("No sponsorship data. bills_referenced will serve as fallback.")
        df["bills_sponsored"] = np.nan

    # Save
    df.to_csv(analysis_path, index=False)
    print(f"\nSaved updated dataset to {analysis_path}")


if __name__ == "__main__":
    main()
