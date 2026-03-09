#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collect Policy Salience Data for GLMM Replication

Attempts Google Trends collection via pytrends. If that fails (rate limiting),
derives salience from Congressional Record mention frequency as a proxy.

Output: data/input/policy_salience_scores.csv

Usage:
    python scripts/collect_policy_salience.py
    python scripts/collect_policy_salience.py --fallback  # Skip Google Trends
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# CAP Policy Topics -> Google Trends search terms (from existing collect_salience.py)
CAP_TOPICS = {
    100: ("Macroeconomics", ["economy", "budget deficit", "inflation"]),
    200: ("Civil Rights", ["civil rights", "discrimination", "voting rights"]),
    300: ("Health", ["healthcare", "medicare", "health insurance"]),
    400: ("Agriculture", ["agriculture", "farming", "farm bill"]),
    500: ("Labor", ["labor unions", "minimum wage", "workers rights"]),
    600: ("Education", ["education policy", "student loans"]),
    700: ("Environment", ["environment", "climate change", "EPA"]),
    800: ("Energy", ["energy policy", "oil prices", "renewable energy"]),
    1000: ("Transportation", ["transportation", "infrastructure"]),
    1200: ("Law and Crime", ["crime", "criminal justice"]),
    1300: ("Social Welfare", ["welfare", "food stamps", "poverty"]),
    1400: ("Housing", ["housing", "affordable housing"]),
    1500: ("Domestic Commerce", ["trade", "small business"]),
    1600: ("Defense", ["military", "defense spending", "national security"]),
    1700: ("Technology", ["technology", "cybersecurity"]),
    1900: ("International Affairs", ["foreign policy", "diplomacy"]),
    2000: ("Government Operations", ["government reform", "federal agencies"]),
    2100: ("Public Lands", ["public lands", "national parks"]),
}


def try_google_trends() -> pd.DataFrame | None:
    """Try to fetch Google Trends data. Returns None if it fails."""
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("pytrends not installed. Install with: pip install pytrends")
        return None

    print("Attempting Google Trends collection...")
    pytrends = TrendReq(hl="en-US", tz=360)
    timeframe = "2015-01-01 2019-01-03"

    results = {}
    for code, (name, terms) in CAP_TOPICS.items():
        try:
            # Use first 3 terms max
            kw_list = terms[:3]
            pytrends.build_payload(kw_list, timeframe=timeframe, geo="US")
            data = pytrends.interest_over_time()

            if not data.empty:
                # Average across terms for this policy area
                mean_score = data[kw_list].mean().mean()
                results[name] = {
                    "issue_number": code,
                    "salience_score": mean_score,
                }
                print(f"  {name}: {mean_score:.1f}")
            else:
                print(f"  {name}: empty response")
                results[name] = {"issue_number": code, "salience_score": np.nan}

            time.sleep(10)  # Conservative rate limiting

        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            # If rate-limited, bail out entirely
            if "429" in str(e) or "Too Many" in str(e):
                print("\nGoogle rate-limited. Falling back to CR-derived salience.")
                return None
            results[name] = {"issue_number": code, "salience_score": np.nan}
            time.sleep(15)

    if not results or all(np.isnan(v["salience_score"]) for v in results.values()):
        return None

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "policy_area"
    df = df.reset_index()
    return df


def derive_from_congressional_record() -> pd.DataFrame:
    """
    Derive salience proxy from Congressional Record mention frequency.

    Logic: policy areas that generate more mentions in floor speeches
    are more salient. This mirrors public attention measured by Google Trends
    since politicians respond to salient issues.
    """
    print("Deriving salience from Congressional Record mention frequency...")

    level1 = pd.read_csv(
        ROOT / "data" / "output" / "level1.csv.gz",
        compression="gzip",
        usecols=["issue_area_name", "issue_area", "granuleId", "year"],
        low_memory=False,
    )

    # Count mentions per policy area
    area_counts = (
        level1.dropna(subset=["issue_area_name"])
        .groupby("issue_area_name")
        .agg(
            mention_count=("granuleId", "size"),
            unique_speeches=("granuleId", "nunique"),
        )
        .reset_index()
    )

    # Normalize to 0-100 scale (like Google Trends)
    area_counts["salience_score"] = (
        area_counts["mention_count"]
        / area_counts["mention_count"].max()
        * 100
    ).round(1)

    # Map to issue numbers
    name_to_code = {name: code for code, (name, _) in CAP_TOPICS.items()}
    area_counts["issue_number"] = area_counts["issue_area_name"].map(name_to_code)

    result = area_counts.rename(columns={"issue_area_name": "policy_area"})
    result = result[["policy_area", "issue_number", "salience_score",
                      "mention_count", "unique_speeches"]]

    return result


def create_salience_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Add salience categories (low/medium/high) via tercile split."""
    df = df.copy()
    df["salience_category"] = pd.qcut(
        df["salience_score"].rank(method="first"),
        q=3,
        labels=["low", "medium", "high"],
    )
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fallback", action="store_true",
                        help="Skip Google Trends, use CR-derived proxy")
    args = parser.parse_args()

    print("=" * 60)
    print("POLICY SALIENCE COLLECTION")
    print("=" * 60)
    print()

    salience_df = None

    if not args.fallback:
        salience_df = try_google_trends()

    if salience_df is None:
        salience_df = derive_from_congressional_record()
        source = "Congressional Record frequency"
    else:
        source = "Google Trends"

    # Add categories
    salience_df = create_salience_categories(salience_df)

    # Save
    out_path = ROOT / "data" / "input" / "policy_salience_scores.csv"
    salience_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(f"Source: {source}")
    print(f"Rows: {len(salience_df)}")
    print()
    print(salience_df.to_string(index=False))

    # Validate against legacy if available
    legacy_path = ROOT / "data" / "output" / "data_legacy_thesis.txt"
    if legacy_path.exists():
        print("\n--- Legacy Comparison ---")
        legacy = pd.read_csv(legacy_path, sep="\t", low_memory=False)
        legacy_sal_col = None
        for c in legacy.columns:
            if "issue_area_salience" in c.lower():
                legacy_sal_col = c
                break
        if legacy_sal_col:
            # Legacy salience by issue area
            legacy_issue_col = None
            for c in legacy.columns:
                if c == "level1_issue_area" or c == "level1_issue_number":
                    legacy_issue_col = c
                    break
            if legacy_issue_col:
                legacy_avg = (
                    legacy.dropna(subset=[legacy_sal_col, legacy_issue_col])
                    .groupby(legacy_issue_col)[legacy_sal_col]
                    .mean()
                    .sort_values(ascending=False)
                )
                print(f"Legacy avg salience by {legacy_issue_col}:")
                print(legacy_avg.head(10).to_string())


if __name__ == "__main__":
    main()
