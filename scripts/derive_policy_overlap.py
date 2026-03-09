#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Derive Policy Overlap Variable for GLMM Replication

Policy overlap = 1 when the speaking member's committee assignments
overlap with the policy area of the organization being mentioned.

Uses the pipeline's intermediate granule_committees and granule_members
data to build member -> committee -> policy area mappings.

Input:
    data/intermediate/normalized_*/by_package/*/granule_committees.csv
    data/intermediate/normalized_*/by_package/*/granule_members.csv
    data/output/analysis_dataset_replication.csv

Output:
    data/input/member_committee_assignments.csv
    (updated analysis_dataset_replication.csv with policy_overlap column)

Usage:
    python scripts/derive_policy_overlap.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Committee name -> policy area (from committee_policy_linkage.py)
COMMITTEE_TO_POLICY = {
    "Committee on Agriculture": "Agriculture",
    "Committee on Agriculture, Nutrition, and Forestry": "Agriculture",
    "Committee on Appropriations": "Macroeconomics",
    "Committee on Armed Services": "Defense",
    "Committee on Banking, Housing, and Urban Affairs": "Housing",
    "Committee on the Budget": "Macroeconomics",
    "Committee on Commerce, Science, and Transportation": "Domestic Commerce",
    "Committee on Education and Labor": "Education",
    "Committee on Education and the Workforce": "Education",
    "Committee on Energy and Commerce": "Energy",
    "Committee on Energy and Natural Resources": "Energy",
    "Committee on Environment and Public Works": "Environment",
    "Committee on Ethics": "Government Operations",
    "Committee on Finance": "Macroeconomics",
    "Committee on Financial Services": "Domestic Commerce",
    "Committee on Foreign Affairs": "International Affairs",
    "Committee on Foreign Relations": "International Affairs",
    "Committee on Health, Education, Labor, and Pensions": "Health",
    "Committee on Homeland Security": "Defense",
    "Committee on Homeland Security and Governmental Affairs": "Government Operations",
    "Committee on House Administration": "Government Operations",
    "Committee on Indian Affairs": "Civil Rights",
    "Committee on the Judiciary": "Law and Crime",
    "Committee on Natural Resources": "Public Lands",
    "Committee on Oversight and Government Reform": "Government Operations",
    "Committee on Rules": "Government Operations",
    "Committee on Rules and Administration": "Government Operations",
    "Committee on Science, Space, and Technology": "Technology",
    "Committee on Small Business": "Domestic Commerce",
    "Committee on Small Business and Entrepreneurship": "Domestic Commerce",
    "Committee on Standards of Official Conduct": "Government Operations",
    "Committee on Transportation and Infrastructure": "Transportation",
    "Committee on Veterans' Affairs": "Social Welfare",
    "Committee on Ways and Means": "Macroeconomics",
    "Joint Committee on Taxation": "Macroeconomics",
    "Joint Economic Committee": "Macroeconomics",
    "Permanent Select Committee on Intelligence": "Defense",
    "Select Committee on Intelligence": "Defense",
    "Special Committee on Aging": "Social Welfare",
}


def build_member_committees() -> dict[str, set[str]]:
    """Build member -> set of policy areas from intermediate data."""
    print("Building member-committee links from intermediate data...")

    # Load granule_committees
    comms_files = glob.glob(
        str(ROOT / "data" / "intermediate" / "normalized_*" / "by_package" / "*" / "granule_committees.csv")
    )
    all_comms = []
    for f in comms_files:
        try:
            df = pd.read_csv(f, usecols=["granuleId", "committeeName"]).dropna()
            all_comms.append(df)
        except Exception:
            pass
    comms_df = pd.concat(all_comms, ignore_index=True) if all_comms else pd.DataFrame()

    # Load granule_members
    members_files = glob.glob(
        str(ROOT / "data" / "intermediate" / "normalized_*" / "by_package" / "*" / "granule_members.csv")
    )
    all_members = []
    for f in members_files:
        try:
            df = pd.read_csv(f)
            bio_col = [c for c in df.columns if "bioguide" in c.lower() or "bioGuide" in c]
            if bio_col:
                df = df[["granuleId", bio_col[0]]].dropna()
                df.columns = ["granuleId", "bioGuideId"]
                all_members.append(df)
        except Exception:
            pass
    members_df = pd.concat(all_members, ignore_index=True) if all_members else pd.DataFrame()

    print(f"  Committee-granule links: {len(comms_df):,}")
    print(f"  Member-granule links: {len(members_df):,}")

    # Join: which members spoke at which committees
    merged = members_df.merge(comms_df, on="granuleId", how="inner")
    print(f"  Joined links: {len(merged):,}")

    # Map committee names to policy areas
    def map_committee(name: str) -> str | None:
        if name in COMMITTEE_TO_POLICY:
            return COMMITTEE_TO_POLICY[name]
        # Try partial match
        for key, area in COMMITTEE_TO_POLICY.items():
            if key.lower() in name.lower() or name.lower() in key.lower():
                return area
        return None

    merged["policy_area"] = merged["committeeName"].apply(map_committee)

    # Build member -> set of policy areas
    member_areas: dict[str, set[str]] = {}
    for bio_id, group in merged.dropna(subset=["policy_area"]).groupby("bioGuideId"):
        member_areas[bio_id] = set(group["policy_area"].unique())

    print(f"  Members with policy area data: {len(member_areas)}")

    # Save committee assignments
    records = []
    for bio_id, group in merged.groupby("bioGuideId"):
        for _, row in group.drop_duplicates("committeeName").iterrows():
            records.append({
                "bioguide_id": bio_id,
                "committee_name": row["committeeName"],
                "policy_area": row.get("policy_area"),
            })
    cache_path = ROOT / "data" / "input" / "member_committee_assignments.csv"
    pd.DataFrame(records).to_csv(cache_path, index=False)
    print(f"  Saved committee assignments to {cache_path}")

    return member_areas


def compute_policy_overlap(df: pd.DataFrame, member_areas: dict[str, set[str]]) -> pd.Series:
    """Compute policy_overlap for each row."""
    def overlap_fn(row):
        bio_id = row.get("bioGuideId")
        issue_area = row.get("issue_area_name")

        if pd.isna(bio_id) or pd.isna(issue_area):
            return np.nan

        areas = member_areas.get(bio_id, set())
        if not areas:
            return np.nan

        return 1 if str(issue_area) in areas else 0

    return df.apply(overlap_fn, axis=1)


def main():
    print("=" * 60)
    print("DERIVE POLICY OVERLAP")
    print("=" * 60)
    print()

    # Build member -> policy areas mapping
    member_areas = build_member_committees()
    print()

    # Load analysis dataset
    analysis_path = ROOT / "data" / "output" / "analysis_dataset_replication.csv"
    df = pd.read_csv(analysis_path, low_memory=False)
    print(f"Analysis dataset: {len(df):,} rows")

    # Compute overlap
    print("Computing policy overlap...")
    df["policy_overlap"] = compute_policy_overlap(df, member_areas)

    non_null = df["policy_overlap"].notna().sum()
    if non_null > 0:
        overlap_rate = df["policy_overlap"].dropna().mean()
        print(f"policy_overlap: {non_null:,} non-null")
        print(f"Overlap rate: {overlap_rate:.3f} ({overlap_rate * 100:.1f}%)")
    print(f"Value counts:\n{df['policy_overlap'].value_counts(dropna=False)}")

    # Save
    df.to_csv(analysis_path, index=False)
    print(f"\nSaved updated dataset to {analysis_path}")


if __name__ == "__main__":
    main()
