#!/usr/bin/env python3
"""
Congressional member profile collection and linkage module.

This module provides tools to fetch member profiles from Congress.gov API,
manage time-varying attributes (terms, committees), and link speakers in
Congressional Record mentions to their canonical member profiles.
"""

# =============================================================================
# # Congressional Member Profiles and Linkage
# 
# This module provides functionality to:
# 
# 1. Fetch detailed member profiles from Congress.gov API
# 2. Track time-varying attributes (party changes, terms, committees)
# 3. Create snapshot profiles aligned to specific speech dates
# 4. Link mention speakers to member profiles with provenance tracking
# 
# ## Usage
# 
# ```python
# from pathlib import Path
# from interest_group_analysis.1.data_collection.6.members_linkage import run_end_to_end
# 
# # Full pipeline
# run_end_to_end(
#     output_dir=Path("data/members/"),
#     mentions_jsonl=Path("data/processed/mentions_with_speakers.jsonl"),
#     input_refs=Path("data/members/congress_member_refs.csv"),
#     enrich_committees=True, 
#     committee_congresses=[114, 115]
# )
# 
# # Or just fetch member profiles
# from interest_group_analysis.1.data_collection.6.members_linkage import fetch_congress_member_profiles
# 
# members_wide, terms_long, committees_df = fetch_congress_member_profiles(
#     output_dir=Path("data/members/"),
#     input_refs=Path("data/members/congress_member_refs.csv"),
#     enrich_committees=True
# )
# ```
# 
# ## Input/Output
# 
# - **Inputs**:
#   - Bioguide IDs in CSV format (columns: bioGuideId)
#   - Mentions JSONL with speaker information
# 
# - **Outputs**:
#   - `congress_members.parquet`: One row per member with stable IDs and attributes
#   - `congress_member_terms.parquet`: Time-varying service terms
#   - `congress_member_committees.parquet`: Committee assignments (optional)
#   - `mention_speaker_links.parquet`: Speaker-to-member linkage table
# 
# ## Authentication
# 
# Requires a Congress.gov API key, which can be set as `CONGRESS_API_KEY` in the
# config module or passed directly to the functions.
# =============================================================================

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import time
import datetime as dt

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

# =================== CONFIG ===================
CONGRESS_API_KEY = None  # override or import from your config
RATE_LIMIT_DELAY = 0.35
TIMEOUT = 30
INPUT_BIOGUIDE_CSV_NAME = "congress_member_refs.csv"  # any CSV having column 'bioGuideId'
MENTIONS_JSONL = "mentions_with_speakers.jsonl"

OUT_MEMBERS_WIDE = "congress_members.parquet"
OUT_TERMS_LONG = "congress_member_terms.parquet"
OUT_COMMITTEES = "congress_member_committees.parquet"
OUT_SPEAKER_LINKS = "mention_speaker_links.parquet"

# Default congress windows for alignment helpers (adjust if you need more)
CONGRESS_WINDOWS = [
    (114, dt.date(2015, 1, 3), dt.date(2017, 1, 3)),
    (115, dt.date(2017, 1, 3), dt.date(2019, 1, 3)),
    (116, dt.date(2019, 1, 3), dt.date(2021, 1, 3)),
    (117, dt.date(2021, 1, 3), dt.date(2023, 1, 3)),
    (118, dt.date(2023, 1, 3), dt.date(2025, 1, 3)),
]

# =================== HTTP =====================
def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    s.headers.update({"User-Agent": "ThesisRevamp/1.0"})
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def _get_json(session: requests.Session, url: str) -> Optional[dict]:
    r = session.get(url, timeout=TIMEOUT)
    if r.status_code == 200:
        return r.json()
    logging.error("GET %s -> %s", url, r.status_code)
    return None

# =================== HELPERS ==================
def _safe(dt_str: Optional[str]) -> Optional[dt.date]:
    if not dt_str:
        return None
    try:
        return dt.date.fromisoformat(dt_str[:10])
    except Exception:
        return None

def _infer_congress_from_date(d: Optional[dt.date]) -> Optional[int]:
    if not d:
        return None
    for cnum, start, end in CONGRESS_WINDOWS:
        if start <= d < end:
            return cnum
    return None

@dataclass(frozen=True)
class MemberKey:
    bioguide: str

# =================== CORE NORMALIZATION ===================
def _flatten_member_record(rec: dict) -> Tuple[dict, pd.DataFrame]:
    """
    Returns (wide_profile_row, terms_long_df)
    - wide_profile_row: one row per member with most recent known attributes
    - terms_long_df: one row per term (time-varying)
    """
    m = rec.get("member") or rec  # some responses are directly under 'member'
    # Identifiers
    identifiers = (m.get("identifiers") or {}) if isinstance(m, dict) else {}
    bio = identifiers.get("bioguideId")
    # Name
    name = m.get("name") or {}
    # Party history (keep latest)
    party_hist = m.get("partyHistory") or []
    current_party = None
    if isinstance(party_hist, list) and party_hist:
        try:
            current_party = party_hist[0].get("partyName")  # Congress.gov often desc by recency
        except Exception:
            current_party = None
    # Birth year
    birth_year = m.get("birthYear")
    # Terms (explode)
    terms = m.get("terms") or []
    terms_rows = []
    for t in terms:
        # Common fields (presence varies; guard each)
        term = {
            "bioGuideId": bio,
            "congress": t.get("congress"),
            "type": t.get("type"),                 # 'Senate'/'House'
            "state": t.get("state"),
            "party": t.get("party"),
            "district": t.get("district"),
            "class": t.get("class"),
            "chamber": t.get("chamber"),           # sometimes present
            "startDate": t.get("startDate"),
            "endDate": t.get("endDate"),
            "leadershipTitle": t.get("leadershipTitle"),
            "leadershipChamber": t.get("leadershipChamber"),
            "leadershipParty": t.get("leadershipParty"),
        }
        # normalized dates + tenure days
        sd = _safe(term["startDate"])
        ed = _safe(term["endDate"])
        term["startDate_parsed"] = sd
        term["endDate_parsed"] = ed
        if sd and ed and ed >= sd:
            term["term_days"] = (ed - sd).days
        else:
            term["term_days"] = None
        # add congress if missing and dates provided
        if not term.get("congress"):
            term["congress"] = _infer_congress_from_date(sd)
        terms_rows.append(term)

    # Wide record (latest known)
    wide = {
        "bioGuideId": bio,
        "fullName": name.get("officialFullName") or name.get("fullName"),
        "firstName": name.get("first"),
        "lastName": name.get("last"),
        "preferredName": name.get("preferred"),
        "currentParty": current_party or m.get("party"),
        "birthYear": birth_year,
        "icpsrId": (identifiers.get("icpsrId") if isinstance(identifiers.get("icpsrId"), (str,int)) else None),
        "govtrackId": identifiers.get("govtrackId"),
        "opensecretsId": identifiers.get("opensecretsId"),
        "thomasId": identifiers.get("thomasId"),
        "cspanId": identifiers.get("cspanId"),
        "lisId": identifiers.get("lisId"),
        "officialWebsite": m.get("officialWebsite"),
    }

    terms_df = pd.DataFrame(terms_rows) if terms_rows else pd.DataFrame(
        columns=[
            "bioGuideId","congress","type","state","party","district","class","chamber",
            "startDate","endDate","startDate_parsed","endDate_parsed",
            "leadershipTitle","leadershipChamber","leadershipParty","term_days"
        ]
    )
    return wide, terms_df

def _latest_known_chamber(terms_df: pd.DataFrame) -> Optional[str]:
    if terms_df.empty:
        return None
    # sort by endDate then startDate to guess most recent chamber
    t = terms_df.copy()
    t["endDate_parsed"] = pd.to_datetime(t["endDate_parsed"])
    t["startDate_parsed"] = pd.to_datetime(t["startDate_parsed"])
    t = t.sort_values(["endDate_parsed","startDate_parsed"], ascending=[False, False])
    for c in t["chamber"].tolist() + t["type"].tolist():
        if pd.notna(c):
            return str(c)
    return None

# =================== FETCHERS ===================
def fetch_member_core(
    output_dir: Path,
    refs_df: pd.DataFrame,
    enrich_committees: bool = False,
    committee_congresses: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Fetch member profiles for all bioguides in refs_df['bioGuideId'].
    Returns (members_wide_df, terms_long_df, committees_df [optional]).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = CONGRESS_API_KEY
    try:
        from .. import config as _cfg
        api_key = api_key or getattr(_cfg, "CONGRESS_API_KEY", None)
    except Exception:
        pass
    if not api_key:
        logging.warning("CONGRESS_API_KEY not set. Skipping.")
        return pd.DataFrame(), pd.DataFrame(), None

    bioguides = (
        refs_df["bioGuideId"].dropna().astype(str).str.strip().unique().tolist()
        if "bioGuideId" in refs_df.columns else []
    )
    if not bioguides:
        logging.warning("No bioGuideId values in refs_df.")
        return pd.DataFrame(), pd.DataFrame(), None

    base = "https://api.congress.gov/v3/member"
    sess = _session()

    wide_rows: List[dict] = []
    terms_all: List[pd.DataFrame] = []

    for bid in bioguides:
        url = f"{base}/{bid}?format=json&api_key={api_key}"
        data = _get_json(sess, url)
        time.sleep(RATE_LIMIT_DELAY)
        if not data:
            continue

        # Some responses nest under 'member', others return a top-level with 'member'
        payload = data.get("member") and data or {"member": data.get("member", data)}
        wide, terms_df = _flatten_member_record(payload)

        # derive recent chamber on wide
        recent_chamber = _latest_known_chamber(terms_df)
        wide["latestChamber"] = recent_chamber

        wide_rows.append(wide)
        if not terms_df.empty:
            terms_all.append(terms_df)

    members_wide = pd.DataFrame(wide_rows).drop_duplicates(subset=["bioGuideId"])
    terms_long = pd.concat(terms_all, ignore_index=True) if terms_all else pd.DataFrame()

    # Save core outputs
    if not members_wide.empty:
        members_wide.to_parquet(output_dir / OUT_MEMBERS_WIDE, index=False)
    if not terms_long.empty:
        # useful indexes for joins
        terms_long["startDate_parsed"] = pd.to_datetime(terms_long["startDate_parsed"])
        terms_long["endDate_parsed"] = pd.to_datetime(terms_long["endDate_parsed"])
        terms_long.to_parquet(output_dir / OUT_TERMS_LONG, index=False)

    committees_df = None
    if enrich_committees and not members_wide.empty:
        c_congs = committee_congresses or sorted({c for c,_,_ in CONGRESS_WINDOWS})
        committees_df = fetch_committee_memberships(
            output_dir=output_dir,
            bioguides=members_wide["bioGuideId"].dropna().astype(str).unique().tolist(),
            congresses=c_congs,
            api_key=api_key,
        )
        if committees_df is not None and not committees_df.empty:
            committees_df.to_parquet(output_dir / OUT_COMMITTEES, index=False)

    return members_wide, terms_long, committees_df

def fetch_committee_memberships(
    output_dir: Path,
    bioguides: List[str],
    congresses: List[int],
    api_key: str,
) -> Optional[pd.DataFrame]:
    """
    Pull member committee assignments by congress.
    """
    base = "https://api.congress.gov/v3/member"
    sess = _session()
    rows = []
    for bid in bioguides:
        for c in congresses:
            url = f"{base}/{bid}/committees?congress={c}&format=json&api_key={api_key}"
            data = _get_json(sess, url)
            time.sleep(RATE_LIMIT_DELAY)
            if not data:
                continue
            committees = (data.get("committees") or {}).get("committee", [])
            for cm in committees:
                rows.append({
                    "bioGuideId": bid,
                    "congress": c,
                    "committeeCode": cm.get("code"),
                    "committeeName": cm.get("name"),
                    "role": cm.get("role"),
                    "subcommitteeCode": cm.get("subcommitteeCode"),
                    "subcommitteeName": cm.get("subcommitteeName"),
                })
    if not rows:
        return None
    return pd.DataFrame(rows).drop_duplicates()

# =================== SNAPSHOT ALIGNMENT ===================
def snapshot_member_on_date(
    terms_long: pd.DataFrame,
    members_wide: pd.DataFrame,
    bioguide: str,
    on_date: dt.date,
) -> Optional[dict]:
    """
    Return the member's attributes (party, state, chamber, district) in force on on_date.
    """
    if members_wide.empty or terms_long.empty:
        return None
    t = terms_long[terms_long["bioGuideId"] == bioguide].copy()
    if t.empty:
        return None
    # select term covering on_date
    t["startDate_parsed"] = pd.to_datetime(t["startDate_parsed"])
    t["endDate_parsed"] = pd.to_datetime(t["endDate_parsed"])
    od = pd.to_datetime(on_date)
    active = t[(t["startDate_parsed"] <= od) & (t["endDate_parsed"] >= od)]
    if active.empty:
        # fall back to most recent past term before on_date
        active = t[t["startDate_parsed"] <= od].sort_values("startDate_parsed", ascending=False).head(1)
        if active.empty:
            return None
    row = active.iloc[0].to_dict()
    base = members_wide[members_wide["bioGuideId"] == bioguide].head(1).to_dict(orient="records")
    base0 = base[0] if base else {}
    return {
        "bioGuideId": bioguide,
        "party_on_date": row.get("party"),
        "state_on_date": row.get("state"),
        "district_on_date": row.get("district"),
        "chamber_on_date": row.get("chamber") or row.get("type"),
        "leadershipTitle_on_date": row.get("leadershipTitle"),
        "member_fullName": base0.get("fullName"),
        "member_currentParty": base0.get("currentParty"),
    }

# =================== MENTIONS JOIN ===================
def link_mentions_to_members(
    output_dir: Path,
    mentions_path: Path,
    members_wide: Optional[pd.DataFrame] = None,
    terms_long: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build a conservative mention→member link table using bioguide if present.
    Falls back to exact (normalized) name match within same congress if bioguide missing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load mentions JSONL
    mentions = []
    with mentions_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                mentions.append(json.loads(line))
            except Exception:
                continue
    if not mentions:
        logging.info("No mentions loaded from %s", mentions_path)
        return pd.DataFrame()

    mrows = []
    for m in mentions:
        mention_id = m.get("uuid_paragraph") or m.get("mention_id") or m.get("uuid")
        speaker = m.get("speaker") or {}
        bio = (speaker.get("bioGuideId") or speaker.get("bioguideId") or m.get("bioGuideId"))
        display = speaker.get("displayName") or speaker.get("name") or m.get("speaker_name")
        # date is crucial for time-varying snapshot
        date_str = m.get("date") or m.get("granule_date") or m.get("dateIssued")
        on_date = _safe(date_str)
        congress = m.get("congress") or _infer_congress_from_date(on_date)

        provenance = None
        confidence = 0.0
        resolved_bio = None
        snap = {}

        if bio:
            resolved_bio = str(bio).strip()
            provenance = "bioguide"
            confidence = 1.0
        else:
            # exact name match within same congress window (very conservative)
            if members_wide is not None and not members_wide.empty and display:
                # Normalize "LAST, First" → "First Last" if needed
                norm_disp = " ".join(str(display).replace(",", " ").split()).lower()
                # lookup candidates by lastName where possible
                cand = members_wide.assign(
                    _name=lambda d: (d["firstName"].fillna("") + " " + d["lastName"].fillna("")).str.strip().str.lower()
                )
                hits = cand[cand["_name"] == norm_disp]
                if not hits.empty:
                    resolved_bio = hits["bioGuideId"].iloc[0]
                    provenance = "name_exact"
                    confidence = 0.6

        if resolved_bio and terms_long is not None and on_date:
            snap = snapshot_member_on_date(terms_long, members_wide, resolved_bio, on_date) or {}

        mrows.append({
            "mention_id": mention_id,
            "speaker_display": display,
            "speaker_bioGuideId": resolved_bio,
            "provenance": provenance,
            "confidence": confidence,
            "on_date": on_date.isoformat() if on_date else None,
            "congress": congress,
            **{f"snapshot_{k}": v for k, v in snap.items()},
        })

    links = pd.DataFrame(mrows)
    links.to_parquet(output_dir / OUT_SPEAKER_LINKS, index=False)
    logging.info("Saved mention→member links to %s", output_dir / OUT_SPEAKER_LINKS)
    return links

# =================== RUNNERS ===================
def fetch_congress_member_profiles(
    output_dir: Path,
    input_refs: Optional[Path] = None,
    enrich_committees: bool = False,
    committee_congresses: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Backwards-compatible wrapper (keeps your original function name) with upgrades.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # get API key
    api_key = CONGRESS_API_KEY
    try:
        from .. import config as _cfg
        api_key = api_key or getattr(_cfg, "CONGRESS_API_KEY", None)
    except Exception:
        pass
    if not api_key:
        logging.warning("CONGRESS_API_KEY is not set. Skipping profile collection.")
        return pd.DataFrame(), pd.DataFrame(), None

    # locate refs csv if not provided
    if input_refs and input_refs.exists():
        refs_df = pd.read_csv(input_refs)
    else:
        refs_path = None
        for candidate in [output_dir / INPUT_BIOGUIDE_CSV_NAME] + list(output_dir.glob("*.csv")):
            try:
                test = pd.read_csv(candidate, nrows=1)
                if "bioGuideId" in test.columns:
                    refs_path = candidate
                    break
            except Exception:
                continue
        if refs_path is None:
            logging.warning("No CSV with 'bioGuideId' found in %s", output_dir)
            return pd.DataFrame(), pd.DataFrame(), None
        refs_df = pd.read_csv(refs_path)

    members_wide, terms_long, committees_df = fetch_member_core(
        output_dir=output_dir,
        refs_df=refs_df,
        enrich_committees=enrich_committees,
        committee_congresses=committee_congresses,
    )
    return members_wide, terms_long, committees_df

def run_end_to_end(
    output_dir: Path,
    mentions_jsonl: Optional[Path] = None,
    input_refs: Optional[Path] = None,
    enrich_committees: bool = False,
    committee_congresses: Optional[List[int]] = None,
) -> None:
    """
    Typical usage:
      run_end_to_end(
        Path("data/members/"),
        mentions_jsonl=Path("data/processed/mentions_with_speakers.jsonl"),
        input_refs=Path("data/members/congress_member_refs.csv"),
        enrich_committees=True, committee_congresses=[114,115]
      )
    """
    members_wide, terms_long, _ = fetch_congress_member_profiles(
        output_dir=output_dir,
        input_refs=input_refs,
        enrich_committees=enrich_committees,
        committee_congresses=committee_congresses,
    )
    if mentions_jsonl and mentions_jsonl.exists():
        link_mentions_to_members(
            output_dir=output_dir,
            mentions_path=mentions_jsonl,
            members_wide=members_wide,
            terms_long=terms_long,
        )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch and link Congressional member profiles")
    parser.add_argument("--output-dir", "-o", type=Path, required=True, 
                        help="Directory to save output files")
    parser.add_argument("--mentions", "-m", type=Path, 
                        help="Path to mentions JSONL file with speaker information")
    parser.add_argument("--refs", "-r", type=Path, 
                        help="Path to CSV file with bioGuideId column")
    parser.add_argument("--committees", "-c", action="store_true", 
                        help="Fetch committee memberships")
    parser.add_argument("--congress", "-n", type=int, nargs="*",
                        help="Congress numbers for committee fetching (default: all)")
    
    args = parser.parse_args()
    
    run_end_to_end(
        output_dir=args.output_dir,
        mentions_jsonl=args.mentions,
        input_refs=args.refs,
        enrich_committees=args.committees,
        committee_congresses=args.congress,
    )