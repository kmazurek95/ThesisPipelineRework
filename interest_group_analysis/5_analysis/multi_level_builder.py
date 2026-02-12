#!/usr/bin/env python3
"""
Multi-Level Data Builder for Interest Group Analysis

This module creates hierarchical datasets at multiple levels of aggregation:
- Level 1: Individual mention (base data)
- Level 2: Organization-level aggregation
- Level 3: Politician-level aggregation
- Level 4: Policy area-level aggregation

It also handles policy area assignment from committee data and computes
derived features (lags, specialization scores, etc.) needed for multi-level
regression models.

Usage:
    python -m interest_group_analysis.5_analysis.multi_level_builder

    # Or as a module:
    from interest_group_analysis.analysis.multi_level_builder import MultiLevelBuilder
    builder = MultiLevelBuilder(project_root)
    builder.build_all()
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("multi_level_builder")


# =============================================================================
# CAP Policy Area Mappings (Comparative Agendas Project)
# =============================================================================

POLICY_AREAS = {
    100: "Macroeconomics",
    200: "Civil Rights",
    300: "Health",
    400: "Agriculture",
    500: "Labor",
    600: "Education",
    700: "Environment",
    800: "Energy",
    900: "Immigration",
    1000: "Transportation",
    1200: "Law and Crime",
    1300: "Social Welfare",
    1400: "Housing",
    1500: "Domestic Commerce",
    1600: "Defense",
    1700: "Technology",
    1800: "Foreign Trade",
    1900: "International Affairs",
    2000: "Government Operations",
    2100: "Public Lands",
    2300: "Culture",
}

# Committee to Policy Area mapping (from your original code)
COMMITTEE_TO_POLICY: Dict[str, Tuple[str, int]] = {
    'Committee on Banking, Housing, and Urban Affairs': ('Housing', 1400),
    'Committee on the Judiciary': ('Law and Crime', 1200),
    'Committee on Health, Education, Labor, and Pensions': ('Health', 300),
    'Committee on Appropriations': ('Macroeconomics', 100),
    'Committee on Agriculture, Nutrition, and Forestry': ('Agriculture', 400),
    'Committee on Foreign Relations': ('International Affairs', 1900),
    'Committee on Rules': ('Government Operations', 2000),
    'Committee on Agriculture': ('Agriculture', 400),
    'Committee on Education and the Workforce': ('Education', 600),
    'Committee on Foreign Affairs': ('International Affairs', 1900),
    'Committee on Education and Labor': ('Education', 600),
    'Committee on Environment and Public Works': ('Environment', 700),
    "Committee on Veterans' Affairs": ('Government Operations', 2000),
    'Committee on Armed Services': ('Defense', 1600),
    'Committee on Energy and Natural Resources': ('Energy', 800),
    'Committee on Finance': ('Macroeconomics', 100),
    'Committee on Energy and Commerce': ('Energy', 800),
    'Committee on Ways and Means': ('Macroeconomics', 100),
    'Committee on Small Business and Entrepreneurship': ('Domestic Commerce', 1500),
    'Committee on Homeland Security and Governmental Affairs': ('Government Operations', 2000),
    'Committee on Financial Services': ('Domestic Commerce', 1500),
    'Joint Committee on Taxation': ('Macroeconomics', 100),
    'Committee on Natural Resources': ('Public Lands', 2100),
    'Permanent Select Committee on Intelligence': ('Defense', 1600),
    'Committee on Standards of Official Conduct': ('Government Operations', 2000),
    'Committee on Transportation and Infrastructure': ('Transportation', 1000),
    'Committee on Commerce, Science, and Transportation': ('Domestic Commerce', 1500),
    'Committee on Science, Space, and Technology': ('Technology', 1700),
    'Committee on Commerce': ('Domestic Commerce', 1500),
    'Committee on the Budget': ('Macroeconomics', 100),
    'Committee on House Administration': ('Government Operations', 2000),
    'Committee on Oversight and Government Reform': ('Government Operations', 2000),
    'Special Committee on Aging': ('Social Welfare', 1300),
    'Select Committee on Intelligence': ('Defense', 1600),
    'Committee on Homeland Security': ('Defense', 1600),
    'Committee on Indian Affairs': ('Civil Rights', 200),
    'Committee on Rules and Administration': ('Government Operations', 2000),
    'Select Committee on Ethics': ('Government Operations', 2000),
    'Committee on Ethics': ('Government Operations', 2000),
    'Joint Select Committee on Deficit Reduction': ('Macroeconomics', 100),
    'Committee on Small Business': ('Domestic Commerce', 1500),
    'Temporary Joint Committee on Deficit Reduction': ('Macroeconomics', 100),
    'Joint Committee on Printing': ('Government Operations', 2000),
    'Joint Committee on the Library': ('Culture', 2300),
    'Joint Economic Committee': ('Macroeconomics', 100),
    'Committee on International Relations': ('International Affairs', 1900),
}

# Tie-break priority (most salient first)
POLICY_PRIORITY = [
    "Macroeconomics", "Health", "Education", "Labor", "Housing",
    "Law and Crime", "Environment", "Energy", "Transportation",
    "Technology", "Social Welfare", "Defense", "Domestic Commerce",
    "International Affairs", "Government Operations", "Public Lands",
    "Agriculture", "Culture", "Civil Rights", "Immigration", "Foreign Trade"
]


# =============================================================================
# Policy Area Assignment
# =============================================================================

class PolicyAreaAssigner:
    """Assigns CAP policy domains to congressional granules based on committees."""

    def __init__(self, mapping: Optional[Dict[str, Tuple[str, int]]] = None):
        self.mapping = mapping or COMMITTEE_TO_POLICY
        self.priority = {name: i for i, name in enumerate(POLICY_PRIORITY)}

    def consolidate_committee_data(self, normalized_dir: Path) -> pd.DataFrame:
        """
        Consolidate all granule_committees.csv files from normalized packages.

        Args:
            normalized_dir: Path to normalized data (e.g., data/normalized_114_run2)

        Returns:
            DataFrame with all committee data
        """
        by_package_dir = normalized_dir / "by_package"
        if not by_package_dir.exists():
            logger.warning(f"by_package directory not found: {by_package_dir}")
            return pd.DataFrame()

        all_committees = []
        package_dirs = list(by_package_dir.iterdir())

        for pkg_dir in package_dirs:
            committees_file = pkg_dir / "granule_committees.csv"
            if committees_file.exists():
                try:
                    df = pd.read_csv(committees_file)
                    if not df.empty and "granuleId" in df.columns:
                        all_committees.append(df)
                except Exception as e:
                    logger.warning(f"Error reading {committees_file}: {e}")

        if not all_committees:
            logger.warning("No committee data found")
            return pd.DataFrame()

        combined = pd.concat(all_committees, ignore_index=True)
        logger.info(f"Consolidated {len(combined)} committee records from {len(all_committees)} packages")
        return combined

    def assign_policy_areas(
        self,
        df_committees: pd.DataFrame,
        method: str = "mode"
    ) -> pd.DataFrame:
        """
        Assign policy areas to granules based on committee associations.

        Args:
            df_committees: DataFrame with granuleId and committeeName columns
            method: "mode" (most frequent), "first", or "all"

        Returns:
            DataFrame with granuleId, policy_area_name, policy_area_code, confidence
        """
        if df_committees.empty:
            return pd.DataFrame(columns=[
                "granuleId", "policy_area_name", "policy_area_code",
                "confidence", "support", "total_committees"
            ])

        # Filter to rows with committee names
        df = df_committees[df_committees["committeeName"].notna()].copy()
        if df.empty:
            logger.warning("No committee names found in data")
            return pd.DataFrame()

        # Map committees to policy areas
        def get_policy(committee_name: str) -> Optional[Tuple[str, int]]:
            if pd.isna(committee_name):
                return None
            # Try exact match
            if committee_name in self.mapping:
                return self.mapping[committee_name]
            # Try case-insensitive match
            for key, val in self.mapping.items():
                if key.lower() == committee_name.lower():
                    return val
            return None

        df["policy_tuple"] = df["committeeName"].apply(get_policy)
        df = df[df["policy_tuple"].notna()]

        if df.empty:
            logger.warning("No committees matched to policy areas")
            return pd.DataFrame()

        df["policy_area_name"] = df["policy_tuple"].apply(lambda x: x[0] if x else None)
        df["policy_area_code"] = df["policy_tuple"].apply(lambda x: x[1] if x else None)

        # Aggregate by granule
        if method == "mode":
            return self._assign_by_mode(df)
        elif method == "first":
            return df.groupby("granuleId").first().reset_index()[
                ["granuleId", "policy_area_name", "policy_area_code"]
            ]
        else:
            raise ValueError(f"Unknown method: {method}")

    def _assign_by_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign policy area using mode (most frequent) with tie-breaking."""
        results = []

        for granule_id, group in df.groupby("granuleId"):
            counts = Counter(group["policy_area_name"])
            total = sum(counts.values())
            max_count = max(counts.values())

            # Find all areas with max count (potential tie)
            top_areas = [area for area, cnt in counts.items() if cnt == max_count]

            if len(top_areas) == 1:
                winner = top_areas[0]
                tie = False
            else:
                # Tie-break using priority
                top_areas.sort(key=lambda x: self.priority.get(x, 999))
                winner = top_areas[0]
                tie = True

            # Get code for winner
            code = None
            for _, row in group[group["policy_area_name"] == winner].iterrows():
                code = row["policy_area_code"]
                break

            results.append({
                "granuleId": granule_id,
                "policy_area_name": winner,
                "policy_area_code": code,
                "support": max_count,
                "total_committees": total,
                "confidence": max_count / total if total > 0 else 0,
                "tie": tie
            })

        return pd.DataFrame(results)

    def validate_coverage(self, df_assignments: pd.DataFrame, df_mentions: pd.DataFrame) -> Dict[str, Any]:
        """Validate policy area coverage against mentions data."""
        if df_assignments.empty or df_mentions.empty:
            return {"coverage": 0, "assigned": 0, "total": 0}

        granule_col = None
        for col in df_mentions.columns:
            if "granuleid" in col.lower():
                granule_col = col
                break

        if granule_col is None:
            return {"coverage": 0, "error": "No granuleId column found in mentions"}

        total_granules = df_mentions[granule_col].nunique()
        assigned_granules = df_assignments["granuleId"].nunique()
        matched = df_mentions[granule_col].isin(df_assignments["granuleId"]).sum()

        return {
            "coverage": matched / len(df_mentions) if len(df_mentions) > 0 else 0,
            "assigned_granules": assigned_granules,
            "total_granules": total_granules,
            "mentions_with_policy": matched,
            "total_mentions": len(df_mentions)
        }


# =============================================================================
# Multi-Level Aggregation Builder
# =============================================================================

class MultiLevelBuilder:
    """Builds hierarchical datasets at multiple levels of aggregation."""

    def __init__(self, project_root: Path):
        self.root = project_root
        self.data_dir = project_root / "data"
        self.merged_dir = self.data_dir / "output"
        self.output_dir = self.data_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.policy_assigner = PolicyAreaAssigner()

    def load_level1_data(self) -> pd.DataFrame:
        """Load the base level 1 data."""
        level1_path = self.merged_dir / "level1.csv"
        if not level1_path.exists():
            raise FileNotFoundError(f"Level 1 data not found: {level1_path}")

        logger.info(f"Loading level 1 data from {level1_path}")
        df = pd.read_csv(level1_path, low_memory=False)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        return df

    def _strip_prefix(self, df: pd.DataFrame, prefix: str = "level1_") -> pd.DataFrame:
        """Remove column prefix for cleaner processing."""
        df = df.copy()
        df.columns = [c.replace(prefix, "") if c.startswith(prefix) else c for c in df.columns]
        return df

    def _add_prefix(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Add column prefix to output data."""
        df = df.copy()
        df.columns = [f"{prefix}{c}" for c in df.columns]
        return df

    def build_level2_organization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build Level 2: Organization-level aggregation.

        Aggregates by org_id to create organization-level features:
        - Total mentions
        - Average prominence
        - Unique policy areas engaged
        - Unique politicians mentioning
        - Most common policy area
        """
        logger.info("Building Level 2: Organization aggregation")
        df = self._strip_prefix(df)

        # Find relevant columns
        org_col = "org_id"
        prominence_col = next(
            (c for c in ["prominence", "prominence_prediction"] if c in df.columns),
            None,
        )
        speaker_col = None
        for col in df.columns:
            if "bioguideid" in col.lower() and "speaker" in col.lower():
                speaker_col = col
                break
            elif "s.bioguideid" in col.lower():
                speaker_col = col
                break

        policy_col = "issue_area" if "issue_area" in df.columns else None

        agg_dict = {
            "total_mentions": (org_col, "count"),
        }

        if prominence_col:
            agg_dict["avg_prominence"] = (prominence_col, "mean")
            agg_dict["sum_prominence"] = (prominence_col, "sum")
            agg_dict["prominence_rate"] = (prominence_col, lambda x: x.mean() if len(x) > 0 else 0)

        if speaker_col:
            agg_dict["unique_politicians"] = (speaker_col, "nunique")

        if policy_col:
            agg_dict["unique_policy_areas"] = (policy_col, "nunique")
            agg_dict["most_common_policy"] = (policy_col, lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)

        # Add organization metadata columns to keep
        meta_cols = [c for c in df.columns if any(x in c.lower() for x in
                     ["lobbying", "founded", "category", "mship", "location", "abbrev"])]

        # Group and aggregate
        grouped = df.groupby(org_col)

        # Build aggregations
        result = pd.DataFrame({org_col: grouped.groups.keys()})

        for name, (col, func) in agg_dict.items():
            if col in df.columns:
                result[name] = grouped[col].agg(func).values

        # Add first value of metadata columns
        for col in meta_cols[:10]:  # Limit to avoid too many columns
            if col in df.columns:
                result[col] = grouped[col].first().values

        logger.info(f"Created Level 2 with {len(result)} organizations")
        return result

    def build_level3_politician(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build Level 3: Politician-level aggregation.

        Aggregates by speaker bioGuideId to create politician-level features:
        - Total mentions made
        - Unique organizations mentioned
        - Unique policy areas
        - Average prominence of mentions
        - Specialization score
        """
        logger.info("Building Level 3: Politician aggregation")
        df = self._strip_prefix(df)

        # Find speaker column
        speaker_col = None
        for col in df.columns:
            if "s.bioguideid" in col.lower():
                speaker_col = col
                break
            elif "bioguideid" in col.lower() and "speaker" in col.lower():
                speaker_col = col
                break

        if speaker_col is None:
            logger.warning("No speaker bioGuideId column found")
            return pd.DataFrame()

        # Filter to rows with valid speaker
        df = df[df[speaker_col].notna()]
        if df.empty:
            logger.warning("No rows with valid speaker IDs")
            return pd.DataFrame()

        prominence_col = next(
            (c for c in ["prominence", "prominence_prediction"] if c in df.columns),
            None,
        )
        org_col = "org_id" if "org_id" in df.columns else None
        policy_col = "issue_area" if "issue_area" in df.columns else None

        agg_dict = {
            "total_mentions": (speaker_col, "count"),
        }

        if org_col:
            agg_dict["unique_orgs_mentioned"] = (org_col, "nunique")

        if prominence_col:
            agg_dict["avg_prominence"] = (prominence_col, "mean")
            agg_dict["sum_prominence"] = (prominence_col, "sum")

        if policy_col:
            agg_dict["unique_policy_areas"] = (policy_col, "nunique")
            agg_dict["primary_policy_area"] = (policy_col, lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)

        # Politician metadata to keep
        meta_cols = [c for c in df.columns if any(x in c.lower() for x in
                     ["party", "chamber", "state", "seniority", "nominate", "ideal", "cook"])]

        grouped = df.groupby(speaker_col)
        result = pd.DataFrame({speaker_col: grouped.groups.keys()})

        for name, (col, func) in agg_dict.items():
            if col in df.columns:
                try:
                    result[name] = grouped[col].agg(func).values
                except Exception as e:
                    logger.warning(f"Error aggregating {name}: {e}")

        # Add politician metadata
        for col in meta_cols[:15]:
            if col in df.columns:
                result[col] = grouped[col].first().values

        # Compute specialization score (Herfindahl index of policy areas)
        if policy_col and policy_col in df.columns:
            def herfindahl(x):
                counts = x.value_counts(normalize=True)
                return (counts ** 2).sum()

            result["specialization_hhi"] = grouped[policy_col].agg(herfindahl).values

        logger.info(f"Created Level 3 with {len(result)} politicians")
        return result

    def build_level4_policy_area(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build Level 4: Policy area-level aggregation.

        Aggregates by policy area to create policy-level features:
        - Total mentions in this policy area
        - Unique organizations active
        - Unique politicians speaking
        - Average prominence
        - Average salience
        """
        logger.info("Building Level 4: Policy area aggregation")
        df = self._strip_prefix(df)

        policy_col = "issue_area" if "issue_area" in df.columns else None
        if policy_col is None:
            # Try alternative names
            for col in df.columns:
                if "issue_area" in col.lower() or "policy_area" in col.lower():
                    policy_col = col
                    break

        if policy_col is None:
            logger.warning("No policy area column found")
            return pd.DataFrame()

        # Filter to rows with valid policy area
        df = df[df[policy_col].notna()]
        if df.empty:
            logger.warning("No rows with valid policy areas")
            return pd.DataFrame()

        prominence_col = next(
            (c for c in ["prominence", "prominence_prediction"] if c in df.columns),
            None,
        )
        org_col = "org_id" if "org_id" in df.columns else None
        salience_col = None
        for col in df.columns:
            if "salience" in col.lower() or "saliency" in col.lower():
                salience_col = col
                break

        speaker_col = None
        for col in df.columns:
            if "s.bioguideid" in col.lower() or ("bioguideid" in col.lower() and "speaker" in col.lower()):
                speaker_col = col
                break

        agg_dict = {
            "total_mentions": (policy_col, "count"),
        }

        if org_col:
            agg_dict["unique_orgs"] = (org_col, "nunique")

        if speaker_col:
            agg_dict["unique_politicians"] = (speaker_col, "nunique")

        if prominence_col:
            agg_dict["avg_prominence"] = (prominence_col, "mean")
            agg_dict["prominence_rate"] = (prominence_col, lambda x: x.mean() if len(x) > 0 else 0)

        if salience_col:
            agg_dict["avg_salience"] = (salience_col, "mean")

        grouped = df.groupby(policy_col)
        result = pd.DataFrame({policy_col: grouped.groups.keys()})

        for name, (col, func) in agg_dict.items():
            if col in df.columns:
                try:
                    result[name] = grouped[col].agg(func).values
                except Exception as e:
                    logger.warning(f"Error aggregating {name}: {e}")

        # Add policy area name
        result["policy_area_name"] = result[policy_col].map(POLICY_AREAS)

        logger.info(f"Created Level 4 with {len(result)} policy areas")
        return result

    def compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived features for the level 1 data:
        - Lagged salience (prior week)
        - Prior mention counts
        - Specialization indicators
        """
        logger.info("Computing derived features")
        df = df.copy()

        # Find date column
        date_col = None
        for col in df.columns:
            if "date" in col.lower() and "issued" in col.lower():
                date_col = col
                break

        if date_col:
            # Sort by date for lag calculations
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.sort_values(date_col)

            # Add week number for temporal grouping
            df["year_week"] = df[date_col].dt.isocalendar().week.astype(str) + "_" + df[date_col].dt.year.astype(str)

        logger.info("Derived features computed")
        return df

    def build_all(self, save: bool = True) -> Dict[str, pd.DataFrame]:
        """Build all multi-level datasets."""
        logger.info("=" * 60)
        logger.info("BUILDING MULTI-LEVEL DATASETS")
        logger.info("=" * 60)

        results = {}

        # Load base data
        try:
            df_level1 = self.load_level1_data()
        except FileNotFoundError as e:
            logger.error(str(e))
            return results

        # Compute derived features
        df_level1 = self.compute_derived_features(df_level1)

        # Build aggregations
        results["level1"] = df_level1
        results["level2_org"] = self.build_level2_organization(df_level1)
        results["level3_politician"] = self.build_level3_politician(df_level1)
        results["level4_policy"] = self.build_level4_policy_area(df_level1)

        # Save outputs
        if save:
            for name, df in results.items():
                if not df.empty:
                    output_path = self.output_dir / f"{name}.csv"
                    df.to_csv(output_path, index=False)
                    logger.info(f"Saved {name} to {output_path} ({len(df)} rows)")

        # Print summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        for name, df in results.items():
            logger.info(f"  {name}: {len(df)} rows, {len(df.columns)} columns")

        return results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for multi-level data building."""
    builder = MultiLevelBuilder(PROJECT_ROOT)

    # First, check if we need to assign policy areas
    logger.info("Checking policy area coverage...")

    # Try to consolidate committee data
    normalized_dirs = [
        PROJECT_ROOT / "data" / "intermediate" / "normalized_114",
        PROJECT_ROOT / "data" / "normalized_114",
    ]

    committees_df = pd.DataFrame()
    for norm_dir in normalized_dirs:
        if norm_dir.exists():
            committees_df = builder.policy_assigner.consolidate_committee_data(norm_dir)
            if not committees_df.empty:
                logger.info(f"Found committee data in {norm_dir}")
                break

    if not committees_df.empty:
        # Assign policy areas
        policy_assignments = builder.policy_assigner.assign_policy_areas(committees_df)
        if not policy_assignments.empty:
            # Save policy assignments
            policy_path = builder.output_dir / "granule_policy_areas.csv"
            policy_assignments.to_csv(policy_path, index=False)
            logger.info(f"Saved policy area assignments to {policy_path}")

            # Validate coverage
            try:
                level1_df = builder.load_level1_data()
                coverage = builder.policy_assigner.validate_coverage(policy_assignments, level1_df)
                logger.info(f"Policy area coverage: {coverage['coverage']:.1%}")
                logger.info(f"  Mentions with policy: {coverage.get('mentions_with_policy', 0)}/{coverage.get('total_mentions', 0)}")
            except Exception as e:
                logger.warning(f"Could not validate coverage: {e}")

    # Build multi-level datasets
    results = builder.build_all(save=True)

    return results


if __name__ == "__main__":
    main()
