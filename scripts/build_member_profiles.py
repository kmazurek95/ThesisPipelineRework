#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build Member Profiles for 114th-115th Congress

Parses the unitedstates/congress-legislators YAML files to derive
seniority and election timing variables needed for GLMM replication.

Input:
    data/input/legislators-current.yaml
    data/input/legislators-historical.yaml

Output:
    data/input/member_profiles_114_115.csv

Usage:
    python scripts/build_member_profiles.py
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "input"
OUTPUT_PATH = INPUT_DIR / "member_profiles_114_115.csv"

# Congress date ranges
CONGRESS_DATES = {
    114: (date(2015, 1, 6), date(2017, 1, 3)),
    115: (date(2017, 1, 3), date(2019, 1, 3)),
}


def load_legislators() -> list[dict[str, Any]]:
    """Load and combine current + historical legislator YAML files."""
    all_legs: list[dict[str, Any]] = []
    for fname in ["legislators-current.yaml", "legislators-historical.yaml"]:
        path = INPUT_DIR / fname
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        print(f"Loaded {len(data):,} legislators from {fname}")
        all_legs.extend(data)
    # Deduplicate by bioguide (a member might appear in both files)
    seen = set()
    unique = []
    for leg in all_legs:
        bio = leg["id"].get("bioguide")
        if bio and bio not in seen:
            seen.add(bio)
            unique.append(leg)
    print(f"Combined: {len(unique):,} unique legislators")
    return unique


def parse_date(d: Any) -> date | None:
    """Parse a date string or return date object as-is."""
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        parts = d.split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    return None


def term_overlaps_congress(term: dict, congress: int) -> bool:
    """Check if a term overlaps with a given congress's date range."""
    cstart, cend = CONGRESS_DATES[congress]
    tstart = parse_date(term["start"])
    tend = parse_date(term["end"])
    if tstart is None or tend is None:
        return False
    return tstart < cend and tend > cstart


def compute_seniority(terms: list[dict], current_term: dict, congress: int) -> dict:
    """Compute seniority metrics for a member in a given congress."""
    cstart = CONGRESS_DATES[congress][0]
    current_chamber = current_term["type"]  # "sen" or "rep"

    # Count prior terms in the same chamber
    prior_same_chamber = 0
    # Count prior terms in any chamber
    prior_any_chamber = 0
    earliest_start = parse_date(current_term["start"])

    for t in terms:
        tstart = parse_date(t["start"])
        if tstart is None:
            continue
        if tstart < earliest_start:
            earliest_start = tstart
        # Only count terms that started before this congress
        if tstart < cstart:
            prior_any_chamber += 1
            if t["type"] == current_chamber:
                prior_same_chamber += 1

    years_in_congress = (cstart - earliest_start).days / 365.25

    return {
        "terms_served_before": prior_same_chamber,
        "terms_served_total": prior_any_chamber,
        "years_in_congress": round(years_in_congress, 1),
        "first_term_start": earliest_start.isoformat(),
    }


def compute_election_timing(term: dict, congress: int) -> dict:
    """Compute election timing variables."""
    tend = parse_date(term["end"])
    chamber = term["type"]
    senate_class = term.get("class", None)

    # House members are always up for reelection
    if chamber == "rep":
        up_for_reelection = 1
    else:
        # Senators: up for reelection if their term ends near the congress end
        cend = CONGRESS_DATES[congress][1]
        if tend is not None:
            # Term ends within ~6 months of congress end = up for reelection
            days_diff = abs((tend - cend).days)
            up_for_reelection = 1 if days_diff < 180 else 0
        else:
            up_for_reelection = 0

    return {
        "up_for_reelection": up_for_reelection,
        "senate_class": senate_class,
    }


def build_profiles(legislators: list[dict[str, Any]]) -> list[dict]:
    """Build member profile records for 114th and 115th Congress."""
    records = []

    for leg in legislators:
        bioguide = leg["id"].get("bioguide")
        if not bioguide:
            continue

        name = leg.get("name", {})
        bio = leg.get("bio", {})
        terms = leg.get("terms", [])

        for congress in [114, 115]:
            # Find the term(s) that overlap this congress
            matching_terms = [t for t in terms if term_overlaps_congress(t, congress)]
            if not matching_terms:
                continue

            # Use the latest matching term (in case of mid-congress changes)
            term = matching_terms[-1]

            seniority = compute_seniority(terms, term, congress)
            election = compute_election_timing(term, congress)

            records.append({
                "bioguide_id": bioguide,
                "first_name": name.get("first", ""),
                "last_name": name.get("last", ""),
                "congress": congress,
                "chamber": "senate" if term["type"] == "sen" else "house",
                "party": term.get("party", ""),
                "state": term.get("state", ""),
                "district": term.get("district", ""),
                "terms_served_before": seniority["terms_served_before"],
                "terms_served_total": seniority["terms_served_total"],
                "years_in_congress": seniority["years_in_congress"],
                "first_term_start": seniority["first_term_start"],
                "up_for_reelection": election["up_for_reelection"],
                "senate_class": election["senate_class"] if election["senate_class"] else "",
                "gender": bio.get("gender", ""),
                "birthday": bio.get("birthday", ""),
            })

    return records


def main():
    print("Building member profiles for 114th-115th Congress...")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_PATH}\n")

    legislators = load_legislators()
    records = build_profiles(legislators)

    # Sort by congress, chamber, last name
    records.sort(key=lambda r: (r["congress"], r["chamber"], r["last_name"]))

    # Write CSV
    if not records:
        print("ERROR: No records produced")
        return

    fieldnames = list(records[0].keys())
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nWrote {len(records):,} records to {OUTPUT_PATH}")

    # Summary stats
    import collections
    congress_counts = collections.Counter(r["congress"] for r in records)
    chamber_counts = collections.Counter((r["congress"], r["chamber"]) for r in records)
    print(f"\nBy congress:")
    for c in sorted(congress_counts):
        print(f"  {c}: {congress_counts[c]:,} members")
    print(f"\nBy congress + chamber:")
    for (c, ch) in sorted(chamber_counts):
        print(f"  {c} {ch}: {chamber_counts[(c, ch)]:,}")

    # Seniority stats
    seniorities = [r["terms_served_before"] for r in records]
    print(f"\nSeniority (terms_served_before):")
    print(f"  Min: {min(seniorities)}, Max: {max(seniorities)}, "
          f"Median: {sorted(seniorities)[len(seniorities)//2]}, "
          f"Mean: {sum(seniorities)/len(seniorities):.1f}")

    # Election timing
    for congress in [114, 115]:
        up = sum(1 for r in records if r["congress"] == congress and r["up_for_reelection"] == 1)
        total = congress_counts[congress]
        print(f"\n  {congress}th: {up}/{total} up for reelection")


if __name__ == "__main__":
    main()
