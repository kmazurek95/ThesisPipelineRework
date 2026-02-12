#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build Analysis Dataset - Revamped ETL Pipeline

This script creates a unified analysis-ready dataset by:
1. Loading classified mentions (from run2: 52,825 mentions)
2. Consolidating normalized CREC data (committees, members, references, granules)
3. Loading/computing external enrichments:
   - Washington Representatives Study (interest group metadata)
   - Congress.gov member data (politician bio + voting)
   - Policy area assignments (committee-based CAP mapping)
   - Issue salience (Google Trends data)
4. Performing sequential merges to create df_prominence
5. Building multi-level aggregations

Output: data/output/
  - level1.csv: Individual mention level (base data)
  - level2_org.csv: Organization aggregation
  - level3_politician.csv: Politician aggregation
  - level4_policy.csv: Policy area aggregation

Usage:
    python -m interest_group_analysis.4_integration.build_analysis_dataset
    python -m interest_group_analysis.4_integration.build_analysis_dataset --dry-run
    python -m interest_group_analysis.4_integration.build_analysis_dataset --skip-collection
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
REFERENCE_DIR = DATA_DIR / "reference"
OUTPUT_DIR = DATA_DIR / "output"

# Input files - scan for all congress-specific mention directories
MENTIONS_DIRS = sorted(INTERMEDIATE_DIR.glob("mentions_*/"))
NORMALIZED_DIR = INTERMEDIATE_DIR / "normalized_114"
WRS_RDA = REFERENCE_DIR / "washington_representatives_study.rda"
INTEREST_GROUPS_CSV = REFERENCE_DIR / "interest_groups_list.csv"

# Newly collected data
CONGRESS_MEMBERS_CSV = RAW_DIR / "members" / "congress_members.csv"
CONGRESS_TERMS_CSV = RAW_DIR / "members" / "congress_member_terms.csv"
BILLS_METADATA_CSV = RAW_DIR / "bills" / "bill_metadata.csv"
BILL_REFERENCES_CSV = RAW_DIR / "bills" / "bill_references.csv"

# CAP (Comparative Agendas Project) Policy Area Mapping
# Maps congressional committees to 21 CAP policy domains
COMMITTEE_TO_CAP: Dict[str, Tuple[str, int]] = {
    # Budget/Economic
    'Committee on Appropriations': ('Macroeconomics', 100),
    'Committee on Finance': ('Macroeconomics', 100),
    'Committee on Ways and Means': ('Macroeconomics', 100),
    'Committee on the Budget': ('Macroeconomics', 100),
    'Joint Committee on Taxation': ('Macroeconomics', 100),
    'Joint Economic Committee': ('Macroeconomics', 100),

    # Civil Rights
    'Committee on Indian Affairs': ('Civil Rights', 200),

    # Health
    'Committee on Health, Education, Labor, and Pensions': ('Health', 300),

    # Agriculture
    'Committee on Agriculture, Nutrition, and Forestry': ('Agriculture', 400),
    'Committee on Agriculture': ('Agriculture', 400),

    # Labor
    'Committee on Education and the Workforce': ('Labor', 500),
    'Committee on Education and Labor': ('Labor', 500),

    # Education
    'Committee on Education': ('Education', 600),

    # Environment
    'Committee on Environment and Public Works': ('Environment', 700),

    # Energy
    'Committee on Energy and Natural Resources': ('Energy', 800),
    'Committee on Energy and Commerce': ('Energy', 800),

    # Transportation
    'Committee on Transportation and Infrastructure': ('Transportation', 1000),
    'Committee on Commerce, Science, and Transportation': ('Transportation', 1000),

    # Law and Crime
    'Committee on the Judiciary': ('Law and Crime', 1200),

    # Social Welfare
    'Special Committee on Aging': ('Social Welfare', 1300),
    'Select Committee on Aging': ('Social Welfare', 1300),

    # Housing
    'Committee on Banking, Housing, and Urban Affairs': ('Housing', 1400),

    # Commerce
    'Committee on Financial Services': ('Domestic Commerce', 1500),
    'Committee on Small Business and Entrepreneurship': ('Domestic Commerce', 1500),
    'Committee on Small Business': ('Domestic Commerce', 1500),
    'Committee on Commerce': ('Domestic Commerce', 1500),

    # Defense
    'Committee on Armed Services': ('Defense', 1600),
    'Permanent Select Committee on Intelligence': ('Defense', 1600),
    'Select Committee on Intelligence': ('Defense', 1600),
    'Committee on Homeland Security': ('Defense', 1600),
    'Committee on Homeland Security and Governmental Affairs': ('Defense', 1600),

    # Technology
    'Committee on Science, Space, and Technology': ('Technology', 1700),
    'Committee on Science': ('Technology', 1700),

    # International Affairs
    'Committee on Foreign Relations': ('International Affairs', 1900),
    'Committee on Foreign Affairs': ('International Affairs', 1900),

    # Government Operations
    "Committee on Veterans' Affairs": ('Government Operations', 2000),
    'Committee on Rules': ('Government Operations', 2000),
    'Committee on Rules and Administration': ('Government Operations', 2000),
    'Committee on Oversight and Government Reform': ('Government Operations', 2000),
    'Committee on House Administration': ('Government Operations', 2000),
    'Select Committee on Ethics': ('Government Operations', 2000),
    'Committee on Ethics': ('Government Operations', 2000),

    # Public Lands
    'Committee on Natural Resources': ('Public Lands', 2100),
    'Committee on Interior and Insular Affairs': ('Public Lands', 2100),
}

CAP_CODE_TO_NAME = {
    100: 'Macroeconomics',
    200: 'Civil Rights',
    300: 'Health',
    400: 'Agriculture',
    500: 'Labor',
    600: 'Education',
    700: 'Environment',
    800: 'Energy',
    1000: 'Transportation',
    1200: 'Law and Crime',
    1300: 'Social Welfare',
    1400: 'Housing',
    1500: 'Domestic Commerce',
    1600: 'Defense',
    1700: 'Technology',
    1900: 'International Affairs',
    2000: 'Government Operations',
    2100: 'Public Lands',
    2300: 'Culture',
}

# Noisy acronyms to filter (ambiguous matches)
ACRONYMS_TO_DROP = [
    "AACC", "AAI", "AAM", "AANP", "AANS", "AAP", "AAPA", "AAPC", "AAPD",
    "AAS", "ABS", "ACC", "ACCT", "ACM", "ACR", "ACS", "AED", "AFA", "AFC",
    "AFGE", "AFP", "AFSA", "AFT", "AHA", "AIA", "ALA", "AMA", "AMC", "AMSA",
    "ANA", "AOA", "APHSA", "API", "APLU", "ARA", "ARRA", "ASA", "ASC", "ASCO",
    "ASH", "ASN", "AST", "ASTA", "ATA", "ATC", "BOMA", "BSA", "CAMP", "CCE",
    "CCI", "CDA", "CEC", "CEF", "CFP", "CLC", "CLIA", "CNI", "COA", "CPA",
    "CRE", "CSF", "CSI", "CTA", "CUNA", "CURE", "EIA", "EMA", "EWG", "FBA",
    "FEA", "HLC", "ICA", "IDSA", "IFDA", "IHC", "IPA", "IRC", "ISA", "MCCA",
    "MSA", "NACA", "NAMA", "NAMI", "NASF", "NBCC", "NCA", "NCBA", "NCC",
    "NCCR", "NCEA", "NCOA", "NCPA", "NCPC", "NEMA", "NEON", "NFCA", "NHA",
    "NHC", "NIF", "NMA", "NMPA", "NOVA", "NPA", "NPC", "NPCA", "NPF", "NRLC",
    "NSBA", "NSTA", "NTCA", "OSMA", "PBA", "PHA", "PIA", "PMA", "PMC", "PMI",
    "PMPA", "PSI", "PVA", "PhRMA", "RAA", "SAF", "SCI", "SGA", "SIA", "SIIA",
    "SMA", "SUA", "USCCB", "WGA",
    # Common words/agencies that aren't interest groups
    "ACT", "Brady", "ACA", "AIR", "CARE", "NDAA", "CRA", "NRA", "CARA",
    "CAA", "AAA", "ADA", "COST", "FERC", "AF", "OPP", "SAFE", "IDEA",
    "NSA", "AA", "ISSA", "NETWORK", "ABA", "PAC", "NLRB", "AMS", "MAP",
    "SSA", "GSA", "NCAA", "AIM", "FAIR", "APS", "ABC", "NAS", "NSF", "ATF",
    "AMT", "SEA", "IFA", "AFB", "CPI", "OCC", "ESA", "ARC", "RPA", "CASE",
    "NBA", "PASS", "ASIA", "NOW", "CAP", "PRC", "AAR", "FRA", "NPR", "IHS",
    "NSC", "APA", "PER", "LISA", "ACE", "NGA", "CWA", "DAM", "CCA", "SMART",
    "MDA", "ATS", "FTA", "MSC", "GAP", "UNESCO", "CBA", "RISE", "ADS",
]


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_mentions(path: Path) -> pd.DataFrame:
    """Load mentions from JSONL file."""
    logger.info(f"Loading mentions from {path}")

    if not path.exists():
        raise FileNotFoundError(f"Mentions file not found: {path}")

    mentions = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                mentions.append(json.loads(line))

    df = pd.DataFrame(mentions)
    logger.info(f"Loaded {len(df):,} mentions")
    return df


def load_washington_representatives(path: Path) -> pd.DataFrame:
    """Load Washington Representatives Study from RDA file."""
    logger.info(f"Loading WRS data from {path}")

    try:
        import pyreadr
    except ImportError:
        logger.warning("pyreadr not installed. Install with: pip install pyreadr")
        return pd.DataFrame()

    if not path.exists():
        logger.warning(f"WRS file not found: {path}")
        return pd.DataFrame()

    result = pyreadr.read_r(str(path))

    # Get the first (and likely only) dataframe
    df_name = list(result.keys())[0]
    df = result[df_name]

    # Rename columns to match expected format
    df = df.rename(columns={
        'ORGIDNO': 'org_id',
        'ORGNAME': 'org_name',
    })

    # Convert org_id to string for consistent joins (handle float -> int -> str)
    df['org_id'] = df['org_id'].dropna().astype(int).astype(str)
    df = df[df['org_id'].notna()]

    logger.info(f"Loaded {len(df):,} interest groups from WRS")
    return df


def consolidate_normalized_data(normalized_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Consolidate all normalized CREC data from per-package directories.

    Returns dict with keys: 'granules', 'committees', 'members', 'references'
    """
    logger.info(f"Consolidating normalized data from {normalized_dir}")

    by_package_dir = normalized_dir / "by_package"
    if not by_package_dir.exists():
        logger.warning(f"Normalized data directory not found: {by_package_dir}")
        return {}

    # Collect all CSVs by type
    data = {
        'granules': [],
        'committees': [],
        'members': [],
        'references': [],
    }

    file_mapping = {
        'granules_core.csv': 'granules',
        'granule_committees.csv': 'committees',
        'granule_members.csv': 'members',
        'granule_references.csv': 'references',
    }

    package_dirs = list(by_package_dir.iterdir())
    logger.info(f"Found {len(package_dirs)} package directories")

    for pkg_dir in package_dirs:
        if not pkg_dir.is_dir():
            continue

        for csv_name, data_key in file_mapping.items():
            csv_path = pkg_dir / csv_name
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, low_memory=False)
                    data[data_key].append(df)
                except Exception as e:
                    logger.warning(f"Error reading {csv_path}: {e}")

    # Concatenate all dataframes
    result = {}
    for key, dfs in data.items():
        if dfs:
            result[key] = pd.concat(dfs, ignore_index=True)
            logger.info(f"Consolidated {key}: {len(result[key]):,} rows")
        else:
            result[key] = pd.DataFrame()
            logger.warning(f"No data found for {key}")

    return result


def load_or_create_salience(output_dir: Path) -> pd.DataFrame:
    """
    Load existing salience data or create placeholder.

    Returns DataFrame with columns: policy_area, issue_number, salience, year_week
    """
    salience_path = output_dir / "issue_salience_long.csv"

    if salience_path.exists():
        logger.info(f"Loading existing salience data from {salience_path}")
        return pd.read_csv(salience_path)

    # Create placeholder with all policy areas
    logger.warning("No salience data found. Creating placeholder.")
    logger.info("To populate: run `python scripts/collect_salience.py`")

    records = []
    for code, name in CAP_CODE_TO_NAME.items():
        records.append({
            'policy_area': name,
            'issue_number': code,
            'salience': 50.0,  # Default neutral salience
            'year_week': '2015_1',
        })

    return pd.DataFrame(records)


def load_congress_members() -> pd.DataFrame:
    """
    Load Congress member data from collected CSV files.

    Returns DataFrame with member biographical and term information.
    """
    if not CONGRESS_MEMBERS_CSV.exists():
        logger.warning(f"Congress members data not found: {CONGRESS_MEMBERS_CSV}")
        logger.info("To populate: run `python scripts/collect_members.py`")
        return pd.DataFrame()

    logger.info(f"Loading Congress member data from {CONGRESS_MEMBERS_CSV}")
    members = pd.read_csv(CONGRESS_MEMBERS_CSV)

    # Load terms data if available
    if CONGRESS_TERMS_CSV.exists():
        terms = pd.read_csv(CONGRESS_TERMS_CSV)
        logger.info(f"Loaded {len(terms):,} term records")

        # Get 114th Congress terms (2015-2017)
        terms_114 = terms[terms['congress'] == 114].copy()
        if not terms_114.empty:
            # Aggregate term info per member for 114th Congress
            agg_dict = {
                'chamber': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None,
            }
            # Add party if available
            if 'party' in terms_114.columns:
                agg_dict['party'] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None
            # Add start date if available
            if 'startDate' in terms_114.columns:
                agg_dict['startDate'] = 'min'

            term_agg = terms_114.groupby('bioGuideId').agg(agg_dict).reset_index()

            # Rename columns with 114 suffix
            rename_cols = {}
            for col in term_agg.columns:
                if col != 'bioGuideId':
                    rename_cols[col] = f'{col}_114'
            term_agg = term_agg.rename(columns=rename_cols)

            members = members.merge(term_agg, on='bioGuideId', how='left')

    logger.info(f"Loaded {len(members):,} Congress members")
    return members


def load_bill_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load bill metadata and references from collected CSV files.

    Returns:
        Tuple of (bill_metadata, bill_references) DataFrames
    """
    bills = pd.DataFrame()
    references = pd.DataFrame()

    if BILLS_METADATA_CSV.exists():
        logger.info(f"Loading bill metadata from {BILLS_METADATA_CSV}")
        bills = pd.read_csv(BILLS_METADATA_CSV)
        logger.info(f"Loaded {len(bills):,} bills")
    else:
        logger.warning(f"Bill metadata not found: {BILLS_METADATA_CSV}")
        logger.info("To populate: run `python scripts/collect_bills.py`")

    if BILL_REFERENCES_CSV.exists():
        logger.info(f"Loading bill references from {BILL_REFERENCES_CSV}")
        references = pd.read_csv(BILL_REFERENCES_CSV, low_memory=False)
        logger.info(f"Loaded {len(references):,} bill references")
    else:
        logger.warning(f"Bill references not found: {BILL_REFERENCES_CSV}")

    return bills, references


# =============================================================================
# Data Cleaning Functions
# =============================================================================

def clean_mentions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean mentions dataframe:
    - Filter out noisy acronyms
    - Standardize column names
    - Add derived columns
    """
    df = df.copy()  # Avoid SettingWithCopyWarning
    original_count = len(df)

    # Filter out noisy acronyms
    if 'variation' in df.columns:
        df = df[~df['variation'].isin(ACRONYMS_TO_DROP)].copy()

    # Ensure org_id is string
    if 'org_id' in df.columns:
        df['org_id'] = df['org_id'].astype(str)

    # Extract date components
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year
        df['week'] = df['date'].dt.isocalendar().week
        df['year_week'] = df['year'].astype(str) + '_' + df['week'].astype(str)

        # Assign congress based on date
        df['congress'] = np.where(
            df['date'] < pd.Timestamp('2017-01-03'),
            114,
            np.where(df['date'] < pd.Timestamp('2019-01-03'), 115, 116)
        )

    filtered_count = len(df)
    logger.info(f"Cleaned mentions: {original_count:,} -> {filtered_count:,} ({original_count - filtered_count:,} filtered)")

    return df


def assign_policy_areas(
    df: pd.DataFrame,
    committees_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign CAP policy areas to mentions based on committee associations.

    Uses mode-based assignment: for each granule, find the most common
    policy area among its associated committees.
    """
    if committees_df.empty:
        logger.warning("No committee data available for policy area assignment")
        df['issue_area'] = None
        df['issue_area_name'] = None
        return df

    # Map committees to policy areas
    committees_df = committees_df.copy()

    def get_policy_area(committee_name: str) -> Optional[Tuple[str, int]]:
        if pd.isna(committee_name):
            return None
        for key, val in COMMITTEE_TO_CAP.items():
            if key.lower() in committee_name.lower():
                return val
        return None

    committees_df['policy_tuple'] = committees_df['committeeName'].apply(get_policy_area)
    committees_df = committees_df[committees_df['policy_tuple'].notna()]
    committees_df['issue_area'] = committees_df['policy_tuple'].apply(lambda x: x[1] if x else None)
    committees_df['issue_area_name'] = committees_df['policy_tuple'].apply(lambda x: x[0] if x else None)

    # Get dominant policy area per granule (mode)
    if 'granuleId' in committees_df.columns:
        granule_policy = (
            committees_df.groupby('granuleId')['issue_area']
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
            .reset_index()
        )
        granule_policy_name = (
            committees_df.groupby('granuleId')['issue_area_name']
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
            .reset_index()
        )
        granule_policy = granule_policy.merge(granule_policy_name, on='granuleId')

        # Merge with mentions
        if 'granuleId' in df.columns:
            df = df.merge(granule_policy, on='granuleId', how='left')
        else:
            logger.warning("No granuleId column in mentions for policy area merge")

    assigned = df['issue_area'].notna().sum()
    logger.info(f"Policy area assignment: {assigned:,}/{len(df):,} mentions ({100*assigned/len(df):.1f}%)")

    return df


# =============================================================================
# Merge Pipeline
# =============================================================================

def build_level1(
    mentions: pd.DataFrame,
    normalized: Dict[str, pd.DataFrame],
    wrs: pd.DataFrame,
    salience: pd.DataFrame,
    congress_members: pd.DataFrame = None,
    bills: pd.DataFrame = None,
    bill_references: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Build Level 1 (individual mention) dataset by merging all sources.
    """
    logger.info("Building Level 1 dataset...")

    df = mentions.copy()

    # 1. Merge granule metadata
    if 'granules' in normalized and not normalized['granules'].empty:
        granules = normalized['granules'].copy()
        if 'granuleId' in granules.columns and 'granuleId' in df.columns:
            # Select relevant columns
            granule_cols = ['granuleId', 'packageId', 'congress', 'title']
            granule_cols = [c for c in granule_cols if c in granules.columns]
            df = df.merge(
                granules[granule_cols].drop_duplicates('granuleId'),
                on='granuleId',
                how='left',
                suffixes=('', '_granule')
            )
            logger.info(f"Merged granule metadata: {len(df):,} rows")

    # 2. Merge committee data and assign policy areas
    if 'committees' in normalized:
        df = assign_policy_areas(df, normalized['committees'])

    # 3. Merge member/speaker data
    if 'members' in normalized and not normalized['members'].empty:
        members = normalized['members'].copy()
        if 'granuleId' in members.columns and 'granuleId' in df.columns:
            # Get primary speaker per granule (first listed)
            speaker_cols = ['granuleId', 'bioGuideId', 'memberName', 'party', 'state', 'chamber']
            speaker_cols = [c for c in speaker_cols if c in members.columns]
            primary_speakers = members[speaker_cols].drop_duplicates('granuleId')

            df = df.merge(
                primary_speakers,
                on='granuleId',
                how='left',
                suffixes=('', '_speaker')
            )
            logger.info(f"Merged speaker data: {len(df):,} rows")

    # 4. Merge interest group metadata (WRS)
    if not wrs.empty and 'org_id' in df.columns:
        # Select relevant WRS columns
        wrs_cols = [
            'org_id', 'org_name', 'CATEGORY', 'LOCATION', 'FOUNDED',
            'IN2011', 'MSHIP_STATUS11', 'IN_HOUSE11', 'OUTSIDE11', 'LOBBYING11',
            'INHOUSEDUM11', 'OUTSIDEDUM11', 'LOBBYDUM11', 'ABBREVCAT',
        ]
        wrs_cols = [c for c in wrs_cols if c in wrs.columns]
        wrs_subset = wrs[wrs_cols].drop_duplicates('org_id')

        df = df.merge(wrs_subset, on='org_id', how='left')

        matched = df['CATEGORY'].notna().sum()
        logger.info(f"Merged WRS data: {matched:,}/{len(df):,} matched ({100*matched/len(df):.1f}%)")

    # 5. Merge salience data
    if not salience.empty and 'year_week' in df.columns and 'issue_area' in df.columns:
        salience_cols = ['issue_number', 'year_week', 'salience']
        salience_cols = [c for c in salience_cols if c in salience.columns]

        if salience_cols:
            df = df.merge(
                salience[salience_cols],
                left_on=['issue_area', 'year_week'],
                right_on=['issue_number', 'year_week'],
                how='left'
            )
            logger.info(f"Merged salience data")

    # 6. Merge Congress member biographical data
    if congress_members is not None and not congress_members.empty and 'bioGuideId' in df.columns:
        member_cols = [
            'bioGuideId', 'fullName', 'firstName', 'lastName', 'birthYear',
            'currentParty', 'state', 'chamber_114', 'party_114', 'startDate_114'
        ]
        member_cols = [c for c in member_cols if c in congress_members.columns]

        if member_cols:
            df = df.merge(
                congress_members[member_cols].drop_duplicates('bioGuideId'),
                on='bioGuideId',
                how='left',
                suffixes=('', '_bio')
            )
            matched = df['fullName'].notna().sum() if 'fullName' in df.columns else 0
            logger.info(f"Merged Congress member data: {matched:,}/{len(df):,} matched")

    # 7. Add bill reference counts per granule
    if bill_references is not None and not bill_references.empty and 'granuleId' in df.columns:
        # Count bills referenced per granule
        if 'granuleId' in bill_references.columns:
            bill_counts = (
                bill_references.groupby('granuleId')
                .size()
                .reset_index(name='bills_referenced')
            )
            df = df.merge(bill_counts, on='granuleId', how='left')
            df['bills_referenced'] = df['bills_referenced'].fillna(0).astype(int)
            logger.info(f"Added bill reference counts")

    # 8. Add derived features
    df['uuid_mention'] = df.index.astype(str)  # Simple unique ID

    logger.info(f"Level 1 complete: {len(df):,} rows, {len(df.columns)} columns")
    return df


def build_level2_org(df: pd.DataFrame) -> pd.DataFrame:
    """Build Level 2: Organization-level aggregation."""
    logger.info("Building Level 2 (organization) aggregation...")

    if 'org_id' not in df.columns:
        logger.warning("No org_id column for Level 2 aggregation")
        return pd.DataFrame()

    agg_dict = {
        'uuid_mention': 'count',  # Total mentions
    }

    # Add prominence if available
    if 'prominence_prediction' in df.columns:
        agg_dict['prominence_prediction'] = ['mean', 'sum']

    # Add speaker diversity
    if 'bioGuideId' in df.columns:
        agg_dict['bioGuideId'] = 'nunique'

    # Add policy area diversity
    if 'issue_area' in df.columns:
        agg_dict['issue_area'] = 'nunique'

    # Perform aggregation
    agg = df.groupby('org_id').agg(agg_dict)
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns]
    agg = agg.reset_index()

    # Rename columns
    agg = agg.rename(columns={
        'uuid_mention_count': 'total_mentions',
        'prominence_prediction_mean': 'avg_prominence',
        'prominence_prediction_sum': 'sum_prominence',
        'bioGuideId_nunique': 'unique_politicians',
        'issue_area_nunique': 'unique_policy_areas',
    })

    # Add org metadata if available
    if 'org_name' in df.columns:
        org_meta = df[['org_id', 'org_name', 'interest_group']].drop_duplicates('org_id')
        agg = agg.merge(org_meta, on='org_id', how='left')

    # Add WRS metadata if available
    wrs_cols = ['CATEGORY', 'LOCATION', 'FOUNDED', 'LOBBYING11', 'ABBREVCAT']
    wrs_cols = [c for c in wrs_cols if c in df.columns]
    if wrs_cols:
        org_wrs = df[['org_id'] + wrs_cols].drop_duplicates('org_id')
        agg = agg.merge(org_wrs, on='org_id', how='left')

    logger.info(f"Level 2 complete: {len(agg):,} organizations")
    return agg


def build_level3_politician(df: pd.DataFrame) -> pd.DataFrame:
    """Build Level 3: Politician-level aggregation."""
    logger.info("Building Level 3 (politician) aggregation...")

    if 'bioGuideId' not in df.columns:
        logger.warning("No bioGuideId column for Level 3 aggregation")
        return pd.DataFrame()

    # Filter to rows with valid bioGuideId
    df_valid = df[df['bioGuideId'].notna() & (df['bioGuideId'] != '')]

    if df_valid.empty:
        logger.warning("No valid bioGuideId values for Level 3")
        return pd.DataFrame()

    agg_dict = {
        'uuid_mention': 'count',
        'org_id': 'nunique',
    }

    if 'prominence_prediction' in df_valid.columns:
        agg_dict['prominence_prediction'] = 'mean'

    if 'issue_area' in df_valid.columns:
        agg_dict['issue_area'] = 'nunique'

    agg = df_valid.groupby('bioGuideId').agg(agg_dict)
    agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in agg.columns]
    agg = agg.reset_index()

    agg = agg.rename(columns={
        'uuid_mention_count': 'total_mentions',
        'uuid_mention': 'total_mentions',
        'org_id_nunique': 'unique_orgs',
        'org_id': 'unique_orgs',
        'prominence_prediction_mean': 'avg_prominence',
        'prominence_prediction': 'avg_prominence',
        'issue_area_nunique': 'unique_policy_areas',
        'issue_area': 'unique_policy_areas',
    })

    # Add politician metadata
    meta_cols = ['bioGuideId', 'memberName', 'party', 'state', 'chamber']
    meta_cols = [c for c in meta_cols if c in df_valid.columns]
    if meta_cols:
        pol_meta = df_valid[meta_cols].drop_duplicates('bioGuideId')
        agg = agg.merge(pol_meta, on='bioGuideId', how='left')

    logger.info(f"Level 3 complete: {len(agg):,} politicians")
    return agg


def build_level4_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Build Level 4: Policy area aggregation."""
    logger.info("Building Level 4 (policy area) aggregation...")

    if 'issue_area' not in df.columns:
        logger.warning("No issue_area column for Level 4 aggregation")
        return pd.DataFrame()

    # Filter to rows with valid issue_area
    df_valid = df[df['issue_area'].notna()]

    if df_valid.empty:
        logger.warning("No valid issue_area values for Level 4")
        return pd.DataFrame()

    agg_dict = {
        'uuid_mention': 'count',
        'org_id': 'nunique',
    }

    if 'bioGuideId' in df_valid.columns:
        agg_dict['bioGuideId'] = 'nunique'

    if 'prominence_prediction' in df_valid.columns:
        agg_dict['prominence_prediction'] = 'mean'

    if 'salience' in df_valid.columns:
        agg_dict['salience'] = 'mean'

    agg = df_valid.groupby('issue_area').agg(agg_dict)
    agg.columns = [col if isinstance(col, str) else '_'.join(col) for col in agg.columns]
    agg = agg.reset_index()

    agg = agg.rename(columns={
        'uuid_mention': 'total_mentions',
        'org_id': 'unique_orgs',
        'bioGuideId': 'unique_politicians',
        'prominence_prediction': 'avg_prominence',
        'salience': 'avg_salience',
    })

    # Add policy area names
    agg['issue_area_name'] = agg['issue_area'].map(CAP_CODE_TO_NAME)

    logger.info(f"Level 4 complete: {len(agg):,} policy areas")
    return agg


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(
    skip_collection: bool = False,
    dry_run: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Run the complete ETL pipeline.

    Args:
        skip_collection: If True, skip data collection steps (use existing data)
        dry_run: If True, just report what would be done

    Returns:
        Dict with 'level1', 'level2', 'level3', 'level4' DataFrames
    """
    logger.info("=" * 60)
    logger.info("Starting Analysis Dataset Build Pipeline")
    logger.info("=" * 60)

    if dry_run:
        logger.info("DRY RUN - No files will be written")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load mentions from all congress directories
    logger.info("\n[1/5] Loading classified mentions...")
    mention_frames = []
    for mdir in MENTIONS_DIRS:
        labeled_path = mdir / "labeled_mentions.jsonl"
        raw_path = mdir / "mentions.jsonl"
        if labeled_path.exists():
            df = load_mentions(labeled_path)
            logger.info(f"  Loaded {len(df):,} labeled mentions from {mdir.name}")
            mention_frames.append(df)
        elif raw_path.exists():
            df = load_mentions(raw_path)
            logger.warning(f"  Loaded {len(df):,} raw mentions from {mdir.name} (no labels)")
            mention_frames.append(df)
        else:
            logger.warning(f"  Skipping {mdir.name}: no mentions file found")

    if not mention_frames:
        raise FileNotFoundError(f"No mentions files found in {INTERMEDIATE_DIR}/mentions_*/")

    mentions = pd.concat(mention_frames, ignore_index=True)
    logger.info(f"Total mentions loaded: {len(mentions):,} from {len(mention_frames)} congress(es)")

    mentions = clean_mentions(mentions)

    # 2. Load normalized CREC data
    logger.info("\n[2/5] Loading normalized CREC data...")
    normalized = consolidate_normalized_data(NORMALIZED_DIR)

    # 3. Load Washington Representatives Study
    logger.info("\n[3/5] Loading interest group metadata...")
    wrs = load_washington_representatives(WRS_RDA)

    # 4. Load/create salience data
    logger.info("\n[4/7] Loading salience data...")
    salience = load_or_create_salience(OUTPUT_DIR)

    # 5. Load Congress member data
    logger.info("\n[5/7] Loading Congress member data...")
    congress_members = load_congress_members()

    # 6. Load bill data
    logger.info("\n[6/7] Loading bill data...")
    bills, bill_references = load_bill_data()

    # 7. Build multi-level datasets
    logger.info("\n[7/7] Building multi-level datasets...")

    level1 = build_level1(
        mentions, normalized, wrs, salience,
        congress_members=congress_members,
        bills=bills,
        bill_references=bill_references,
    )
    level2 = build_level2_org(level1)
    level3 = build_level3_politician(level1)
    level4 = build_level4_policy(level1)

    # Save outputs
    if not dry_run:
        logger.info("\nSaving outputs...")

        level1.to_csv(OUTPUT_DIR / "level1.csv", index=False)
        logger.info(f"Saved level1.csv: {len(level1):,} rows")

        level2.to_csv(OUTPUT_DIR / "level2_org.csv", index=False)
        logger.info(f"Saved level2_org.csv: {len(level2):,} rows")

        level3.to_csv(OUTPUT_DIR / "level3_politician.csv", index=False)
        logger.info(f"Saved level3_politician.csv: {len(level3):,} rows")

        level4.to_csv(OUTPUT_DIR / "level4_policy.csv", index=False)
        logger.info(f"Saved level4_policy.csv: {len(level4):,} rows")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"Level 1 (mentions):     {len(level1):>10,} rows x {len(level1.columns):>3} cols")
    logger.info(f"Level 2 (organizations): {len(level2):>10,} rows x {len(level2.columns):>3} cols")
    logger.info(f"Level 3 (politicians):   {len(level3):>10,} rows x {len(level3.columns):>3} cols")
    logger.info(f"Level 4 (policy areas):  {len(level4):>10,} rows x {len(level4.columns):>3} cols")
    logger.info(f"\nOutput directory: {OUTPUT_DIR}")

    return {
        'level1': level1,
        'level2': level2,
        'level3': level3,
        'level4': level4,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build analysis-ready multi-level dataset"
    )
    parser.add_argument(
        '--skip-collection',
        action='store_true',
        help='Skip data collection steps (use existing data)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Report what would be done without writing files'
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            skip_collection=args.skip_collection,
            dry_run=args.dry_run,
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
