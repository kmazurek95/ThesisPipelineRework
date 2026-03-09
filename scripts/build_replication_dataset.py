#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build Full-Sample Replication Dataset

Creates a mention-level dataset matching the legacy thesis structure:
- One row per mention (org × granule)
- Plus one row per zero-mention org (from WRS dictionary)
- All variables needed for GLMM Models B and C

Input:
    data/output/level1.csv.gz          (53,892 mention rows)
    data/reference/interest_groups_list.csv  (5,441 WRS orgs)
    data/reference/washington_representatives_study.rda  (WRS metadata)
    data/input/member_profiles_114_115.csv  (member profiles)

Output:
    data/output/analysis_dataset_replication.csv

Usage:
    python scripts/build_replication_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_revamp() -> pd.DataFrame:
    """Load revamp level1 mention data."""
    path = ROOT / "data" / "output" / "level1.csv.gz"
    df = pd.read_csv(path, compression="gzip")
    print(f"Loaded revamp: {len(df):,} rows, {df.shape[1]} columns")
    return df


def load_wrs_dict() -> pd.DataFrame:
    """Load WRS organization dictionary."""
    path = ROOT / "data" / "reference" / "interest_groups_list.csv"
    df = pd.read_csv(path)
    print(f"Loaded WRS dict: {len(df):,} orgs")
    return df


def load_wrs_metadata() -> pd.DataFrame:
    """Load WRS metadata from .rda file, deduplicate to one row per org."""
    import pyreadr

    path = ROOT / "data" / "reference" / "washington_representatives_study.rda"
    rda = pyreadr.read_r(str(path))
    key = list(rda.keys())[0]
    meta = rda[key]
    print(f"Loaded WRS metadata: {len(meta):,} rows, {meta.shape[1]} columns")

    # Filter to IN2011=Yes (current orgs), then deduplicate
    meta_2011 = meta[meta["IN2011"].astype(str).str.contains("Yes")]
    meta_clean = meta_2011.drop_duplicates(subset="ORGIDNO", keep="last")
    print(f"After IN2011 filter + dedup: {len(meta_clean):,} rows")

    # Rename ORGIDNO to org_id for joining
    meta_clean = meta_clean.rename(columns={"ORGIDNO": "org_id"})
    return meta_clean


def load_member_profiles() -> pd.DataFrame:
    """Load member profiles for 114th-115th Congress."""
    path = ROOT / "data" / "input" / "member_profiles_114_115.csv"
    df = pd.read_csv(path)
    print(f"Loaded member profiles: {len(df):,} records")
    return df


def build_mention_rows(revamp: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """Build mention-level rows with member profile joins."""
    df = revamp.copy()

    # Derive key variables from existing columns
    df["is_democrat"] = (df["party"] == "D").astype(int)
    df["is_senate"] = (df["chamber"] == "Senate").astype(int)

    # Join member profiles for seniority and election timing
    # Members keyed on (bioguide_id, congress)
    member_cols = [
        "bioguide_id", "congress", "terms_served_before", "terms_served_total",
        "years_in_congress", "up_for_reelection", "senate_class",
    ]
    available_cols = [c for c in member_cols if c in members.columns]
    members_join = members[available_cols].copy()
    members_join = members_join.rename(columns={"bioguide_id": "bioGuideId"})

    df = df.merge(members_join, on=["bioGuideId", "congress"], how="left")
    matched = df["terms_served_before"].notna().sum()
    has_bio = df["bioGuideId"].notna().sum()
    print(f"Member profile join: {matched:,}/{has_bio:,} rows with bioGuideId matched "
          f"({matched / has_bio * 100:.1f}%)" if has_bio > 0 else "No bioGuideId rows")

    return df


def build_zero_mention_rows(wrs_dict: pd.DataFrame, mention_org_ids: set) -> pd.DataFrame:
    """Create one row per zero-mention org for full-sample design."""
    zero_orgs = wrs_dict[~wrs_dict["org_id"].isin(mention_org_ids)].copy()
    zero_orgs["mention_count"] = 0
    zero_orgs["prominence_prediction"] = 0
    zero_orgs["prominence_score"] = 0.0
    zero_orgs["is_zero_mention"] = 1
    print(f"Zero-mention orgs: {len(zero_orgs):,}")
    return zero_orgs


def derive_org_variables(
    df: pd.DataFrame, wrs_meta: pd.DataFrame, revamp: pd.DataFrame
) -> pd.DataFrame:
    """Derive organization-level variables and join WRS metadata."""

    # Compute policy scope: unique issue areas per org
    if "issue_area" in revamp.columns:
        policy_scope = (
            revamp.groupby("org_id")["issue_area"]
            .nunique()
            .reset_index()
            .rename(columns={"issue_area": "policy_scope"})
        )
        df = df.merge(policy_scope, on="org_id", how="left")
        df["policy_scope"] = df["policy_scope"].fillna(0).astype(int)
    else:
        df["policy_scope"] = 0

    # Join WRS metadata columns not already in revamp
    # The revamp level1 already has CATEGORY, LOBBYING11, FOUNDED, etc.
    # For zero-mention rows, we need to join from WRS metadata
    meta_cols = ["org_id", "CATEGORY", "LOBBYING11", "FOUNDED", "MSHIP_STATUS11",
                 "LOCATION", "IN2011"]
    available_meta = [c for c in meta_cols if c in wrs_meta.columns]
    wrs_subset = wrs_meta[available_meta].copy()

    # Fill missing WRS columns for zero-mention rows
    for col in ["CATEGORY", "LOBBYING11", "FOUNDED", "MSHIP_STATUS11", "LOCATION"]:
        if col in df.columns and col in wrs_subset.columns:
            # Only fill where missing
            fill_map = wrs_subset.set_index("org_id")[col]
            mask = df[col].isna()
            df.loc[mask, col] = df.loc[mask, "org_id"].map(fill_map)
        elif col in wrs_subset.columns and col not in df.columns:
            fill_map = wrs_subset.set_index("org_id")[col]
            df[col] = df["org_id"].map(fill_map)

    # Derive org_age
    df["FOUNDED_year"] = pd.to_numeric(df["FOUNDED"], errors="coerce")
    # Use 2015 for 114th Congress, 2017 for 115th
    if "congress" in df.columns:
        congress_year = df["congress"].map({114: 2015, 115: 2017})
        df["org_age"] = congress_year - df["FOUNDED_year"]
    else:
        df["org_age"] = 2016 - df["FOUNDED_year"]  # midpoint

    # Log lobbying expenditure
    df["log_lobbying"] = np.log1p(pd.to_numeric(df["LOBBYING11"], errors="coerce"))

    # Organization type dummies from CATEGORY
    if "CATEGORY" in df.columns:
        cat = df["CATEGORY"].astype(str)
        df["is_labor"] = cat.str.contains("labor|union", case=False, na=False).astype(int)
        df["is_single_issue"] = cat.str.contains("single.?issue|citizen", case=False, na=False).astype(int)
        df["is_trade"] = cat.str.contains("trade|business", case=False, na=False).astype(int)
        df["is_professional"] = cat.str.contains("professional", case=False, na=False).astype(int)

    # Membership status
    if "MSHIP_STATUS11" in df.columns:
        ms = df["MSHIP_STATUS11"].astype(str)
        df["is_membership_org"] = ms.str.contains("Association|membership", case=False, na=False).astype(int)

    return df


def main():
    print("=" * 60)
    print("BUILD FULL-SAMPLE REPLICATION DATASET")
    print("=" * 60)
    print()

    # Load all sources
    revamp = load_revamp()
    wrs_dict = load_wrs_dict()
    wrs_meta = load_wrs_metadata()
    members = load_member_profiles()
    print()

    # Step 1: Build mention rows with member profile joins
    print("--- Step 1: Build mention rows ---")
    mention_df = build_mention_rows(revamp, members)
    mention_df["is_zero_mention"] = 0
    print(f"Mention rows: {len(mention_df):,}")
    print()

    # Step 2: Build zero-mention rows
    print("--- Step 2: Build zero-mention rows ---")
    mention_org_ids = set(revamp["org_id"].unique())
    zero_df = build_zero_mention_rows(wrs_dict, mention_org_ids)
    print()

    # Step 3: Combine
    print("--- Step 3: Combine mention + zero-mention rows ---")
    # Align columns before concat
    full_df = pd.concat([mention_df, zero_df], ignore_index=True, sort=False)
    print(f"Combined: {len(full_df):,} rows")
    print()

    # Step 4: Derive org-level variables
    print("--- Step 4: Derive variables ---")
    full_df = derive_org_variables(full_df, wrs_meta, revamp)
    print()

    # Step 5: Select output columns
    # Keep all useful columns for GLMM modeling
    output_cols = [
        # Identifiers
        "org_id", "interest_group", "granuleId", "bioGuideId",
        # Structural
        "congress", "chamber", "party", "state",
        "issue_area", "issue_area_name",
        # DV
        "prominence_prediction", "prominence_score",
        # Organization characteristics
        "CATEGORY", "LOBBYING11", "log_lobbying", "FOUNDED", "FOUNDED_year",
        "org_age", "MSHIP_STATUS11", "LOCATION", "policy_scope",
        # Org type dummies
        "is_labor", "is_single_issue", "is_trade", "is_professional",
        "is_membership_org",
        # Speaker characteristics
        "is_democrat", "is_senate",
        # Member profiles
        "terms_served_before", "terms_served_total", "years_in_congress",
        "up_for_reelection", "senate_class",
        # Bill references
        "bills_referenced",
        # Full-sample indicator
        "is_zero_mention",
    ]

    # Only keep columns that exist
    final_cols = [c for c in output_cols if c in full_df.columns]
    missing_cols = [c for c in output_cols if c not in full_df.columns]
    if missing_cols:
        print(f"Note: {len(missing_cols)} requested columns not found: {missing_cols}")

    output_df = full_df[final_cols].copy()

    # Step 6: Save
    output_path = ROOT / "data" / "output" / "analysis_dataset_replication.csv"
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print()

    # Summary
    print("=" * 60)
    print("ANALYSIS DATASET BUILT")
    print("=" * 60)
    n_total = len(output_df)
    n_orgs = output_df["org_id"].nunique()
    n_mention = (output_df["is_zero_mention"] == 0).sum()
    n_zero = (output_df["is_zero_mention"] == 1).sum()
    n_with_mentions = output_df.loc[output_df["is_zero_mention"] == 0, "org_id"].nunique()
    print(f"Total rows:           {n_total:,}")
    print(f"Unique orgs:          {n_orgs:,}")
    print(f"  With mentions:      {n_with_mentions:,}")
    print(f"  Zero-mention:       {n_zero:,}")
    print(f"Mention rows:         {n_mention:,}")
    print(f"Columns:              {output_df.shape[1]}")
    print()

    print("Variable availability for GLMMs:")
    checks = {
        "DV (prominence)": "prominence_prediction",
        "Chamber": "is_senate",
        "Party": "is_democrat",
        "Seniority": "terms_served_before",
        "Election timing": "up_for_reelection",
        "Org type (CATEGORY)": "CATEGORY",
        "Org age": "org_age",
        "Log lobbying": "log_lobbying",
        "Policy scope": "policy_scope",
        "Policy salience": None,  # NOT USABLE
        "Policy overlap": None,  # NOT AVAILABLE
        "Bill references": "bills_referenced",
        "Membership status": "is_membership_org",
    }
    for label, col in checks.items():
        if col is None:
            print(f"  {label:25s} MISSING")
        elif col in output_df.columns:
            nn = output_df[col].notna().sum()
            print(f"  {label:25s} YES  ({nn:,} non-null)")
        else:
            print(f"  {label:25s} NO (column not found)")

    print()
    print("GLMM MODEL READINESS:")
    print("  Model A (Policy Salience):       CANNOT REPLICATE (salience = constant)")
    print("  Model B (Group-Politician):      PARTIALLY REPLICABLE")
    print("    Available: seniority, election timing, bills_referenced")
    print("    Missing: policy_overlap, bills_sponsored/cosponsored")
    print("  Model C (Group Characteristics): FULLY REPLICABLE")
    print("    Available: org_age, log_lobbying, policy_scope, CATEGORY, MSHIP_STATUS11")


if __name__ == "__main__":
    main()
