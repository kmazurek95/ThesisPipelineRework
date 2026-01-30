#!/usr/bin/env python3
"""
Collect policy salience data from Google Trends.

This script fetches Google Trends data for CAP policy topics and creates
a salience dataset that can be joined to mentions by year_week.

Usage:
    python scripts/collect_salience.py
    python scripts/collect_salience.py --skip-trends  # Use cached data
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

# CAP Policy Topics mapped to Google Trends search terms
CAP_TOPICS = {
    100: ("Macroeconomics", "economy budget deficit"),
    200: ("Civil Rights", "civil rights discrimination"),
    300: ("Health", "healthcare medical insurance"),
    400: ("Agriculture", "agriculture farming food policy"),
    500: ("Labor", "labor unions employment"),
    600: ("Education", "education schools"),
    700: ("Environment", "environment pollution climate"),
    800: ("Energy", "energy oil gas"),
    1000: ("Transportation", "transportation infrastructure"),
    1200: ("Law and Crime", "crime law enforcement justice"),
    1300: ("Social Welfare", "welfare social security"),
    1400: ("Housing", "housing mortgage"),
    1500: ("Domestic Commerce", "business commerce trade"),
    1600: ("Defense", "military defense security"),
    1700: ("Technology", "technology innovation science"),
    1900: ("International Affairs", "foreign policy international"),
    2000: ("Government Operations", "government reform"),
    2100: ("Public Lands", "public lands national parks"),
}

# Reference topic for normalization
CONSTANT_TOPIC = "news"


def get_date_range_from_mentions() -> tuple:
    """Extract date range from mentions data."""
    mentions_path = PROJECT_ROOT / "data" / "intermediate" / "mentions_114" / "mentions.jsonl"

    if not mentions_path.exists():
        logger.warning("Mentions file not found, using default date range")
        return "2015-01-01", "2016-12-31"

    import json
    dates = []
    with open(mentions_path) as f:
        for line in f:
            d = json.loads(line)
            if 'date' in d:
                dates.append(d['date'])

    if dates:
        dates = pd.to_datetime(dates)
        return str(dates.min().date()), str(dates.max().date())

    return "2015-01-01", "2016-12-31"


def fetch_google_trends(topics: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch Google Trends data for topics.

    Args:
        topics: List of search terms
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with weekly trend data
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        logger.error("pytrends not installed. Install with: pip install pytrends")
        return pd.DataFrame()

    pytrends = TrendReq(hl='en-US', tz=360)

    all_data = []
    timeframe = f"{start_date} {end_date}"

    # Process topics in groups of 4 (Google Trends limit is 5)
    group_size = 4
    topic_groups = [topics[i:i + group_size] for i in range(0, len(topics), group_size)]

    for i, group in enumerate(topic_groups):
        # Always include reference topic
        kw_list = [CONSTANT_TOPIC] + group

        logger.info(f"Fetching group {i+1}/{len(topic_groups)}: {group}")

        try:
            pytrends.build_payload(kw_list, timeframe=timeframe, geo='US')
            data = pytrends.interest_over_time()

            if not data.empty:
                # Normalize by reference topic
                for topic in group:
                    if topic in data.columns:
                        data[f"{topic}_normalized"] = data[topic] / (data[CONSTANT_TOPIC] + 1)

                all_data.append(data)

            # Rate limiting
            time.sleep(15)

        except Exception as e:
            logger.warning(f"Error fetching trends for {group}: {e}")
            time.sleep(30)

    if all_data:
        # Combine all groups
        result = pd.concat(all_data, axis=1)
        result = result.loc[:, ~result.columns.duplicated()]
        return result

    return pd.DataFrame()


def create_salience_long_format(trends_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert trends data to long format with year_week for joining.
    """
    if trends_df.empty:
        logger.warning("No trends data, creating placeholder salience")
        return create_placeholder_salience()

    # Reset index to get date column
    trends_df = trends_df.reset_index()
    trends_df = trends_df.rename(columns={'date': 'dateIssued'})

    # Add year_week
    trends_df['dateIssued'] = pd.to_datetime(trends_df['dateIssued'])
    trends_df['year'] = trends_df['dateIssued'].dt.year
    trends_df['week'] = trends_df['dateIssued'].dt.isocalendar().week
    trends_df['year_week'] = trends_df['year'].astype(str) + '_' + trends_df['week'].astype(str)

    # Melt to long format
    records = []
    for issue_number, (policy_name, search_term) in CAP_TOPICS.items():
        # Find matching column (either exact or normalized)
        col = None
        for c in trends_df.columns:
            if search_term in c or policy_name.lower() in c.lower():
                col = c
                break

        for _, row in trends_df.iterrows():
            salience = row[col] if col and col in row else 50.0
            records.append({
                'policy_area': policy_name,
                'issue_number': issue_number,
                'salience': salience,
                'dateIssued': row['dateIssued'],
                'year_week': row['year_week'],
            })

    return pd.DataFrame(records)


def create_placeholder_salience() -> pd.DataFrame:
    """Create placeholder salience data when Google Trends fails."""
    logger.info("Creating placeholder salience data...")

    # Generate weekly dates for 2015-2016
    dates = pd.date_range('2015-01-01', '2016-12-31', freq='W')

    records = []
    for date in dates:
        year_week = f"{date.year}_{date.isocalendar().week}"
        for issue_number, (policy_name, _) in CAP_TOPICS.items():
            records.append({
                'policy_area': policy_name,
                'issue_number': issue_number,
                'salience': 50.0,  # Neutral placeholder
                'dateIssued': date,
                'year_week': year_week,
            })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Collect policy salience data")
    parser.add_argument(
        '--skip-trends',
        action='store_true',
        help='Skip Google Trends API, use placeholder data'
    )
    parser.add_argument(
        '--output', '-o',
        default=str(OUTPUT_DIR / "issue_salience_long.csv"),
        help='Output file path'
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Policy Salience Collection")
    logger.info("=" * 60)

    # Get date range from mentions
    start_date, end_date = get_date_range_from_mentions()
    logger.info(f"Date range: {start_date} to {end_date}")

    if args.skip_trends:
        logger.info("Skipping Google Trends, using placeholder data")
        salience_df = create_placeholder_salience()
    else:
        # Extract search terms
        search_terms = [term for _, (_, term) in CAP_TOPICS.items()]

        logger.info(f"Fetching Google Trends for {len(search_terms)} policy topics...")
        trends_df = fetch_google_trends(search_terms, start_date, end_date)

        if trends_df.empty:
            logger.warning("Google Trends returned no data, using placeholder")
            salience_df = create_placeholder_salience()
        else:
            salience_df = create_salience_long_format(trends_df)

    # Save output
    output_path = Path(args.output)
    salience_df.to_csv(output_path, index=False)
    logger.info(f"Saved salience data to {output_path}")
    logger.info(f"  {len(salience_df):,} rows")
    logger.info(f"  {salience_df['issue_number'].nunique()} policy areas")
    logger.info(f"  {salience_df['year_week'].nunique()} weeks")

    # Summary stats
    logger.info("\nSalience summary by policy area:")
    summary = salience_df.groupby('policy_area')['salience'].mean().sort_values(ascending=False)
    for policy, sal in summary.head(10).items():
        logger.info(f"  {policy}: {sal:.1f}")


if __name__ == "__main__":
    main()
