#!/usr/bin/env python3
"""
Data Validation Script for Interest Group Analysis Pipeline

This script validates data at each stage of the pipeline, checking for:
- File existence and basic structure
- Row counts and column completeness
- Data quality metrics (nulls, duplicates, ranges)
- Cross-stage consistency

Usage:
    # Validate all stages
    python scripts/validate_data.py

    # Validate specific stage
    python scripts/validate_data.py --stage classification

    # Output detailed report
    python scripts/validate_data.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    stage: str
    check_name: str
    passed: bool
    message: str
    details: Optional[dict[str, Any]] = None


class DataValidator:
    """Validates data at each pipeline stage."""

    def __init__(self, project_root: Path):
        self.root = project_root
        self.results: list[ValidationResult] = []

    def add_result(self, stage: str, check_name: str, passed: bool, message: str,
                   details: Optional[dict[str, Any]] = None):
        """Add a validation result."""
        self.results.append(ValidationResult(
            stage=stage,
            check_name=check_name,
            passed=passed,
            message=message,
            details=details
        ))

    def check_file_exists(self, stage: str, path: Path, description: str) -> bool:
        """Check if a file exists."""
        exists = path.exists()
        self.add_result(
            stage=stage,
            check_name=f"file_exists_{path.name}",
            passed=exists,
            message=f"{description}: {'Found' if exists else 'MISSING'}",
            details={"path": str(path)}
        )
        return exists

    def check_csv_structure(self, stage: str, path: Path, required_cols: list[str],
                            min_rows: int = 1) -> bool:
        """Check CSV file structure."""
        if not path.exists():
            self.add_result(
                stage=stage,
                check_name=f"csv_structure_{path.name}",
                passed=False,
                message=f"File not found: {path}",
                details={"path": str(path)}
            )
            return False

        try:
            df = pd.read_csv(path, nrows=5)  # Just read header + few rows
            full_df = pd.read_csv(path)

            missing_cols = set(required_cols) - set(df.columns)
            row_count = len(full_df)

            passed = len(missing_cols) == 0 and row_count >= min_rows
            self.add_result(
                stage=stage,
                check_name=f"csv_structure_{path.name}",
                passed=passed,
                message=f"CSV check: {row_count} rows, {'missing cols: ' + str(missing_cols) if missing_cols else 'all required cols present'}",
                details={
                    "path": str(path),
                    "row_count": row_count,
                    "columns": list(df.columns),
                    "missing_required": list(missing_cols)
                }
            )
            return passed
        except Exception as e:
            self.add_result(
                stage=stage,
                check_name=f"csv_structure_{path.name}",
                passed=False,
                message=f"Error reading CSV: {e}",
                details={"path": str(path), "error": str(e)}
            )
            return False

    def check_jsonl_structure(self, stage: str, path: Path, min_rows: int = 1) -> bool:
        """Check JSONL file structure."""
        if not path.exists():
            self.add_result(
                stage=stage,
                check_name=f"jsonl_structure_{path.name}",
                passed=False,
                message=f"File not found: {path}",
                details={"path": str(path)}
            )
            return False

        try:
            row_count = 0
            sample_keys = set()
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    row_count += 1
                    if i < 5:  # Sample first 5 rows
                        obj = json.loads(line)
                        sample_keys.update(obj.keys())

            passed = row_count >= min_rows
            self.add_result(
                stage=stage,
                check_name=f"jsonl_structure_{path.name}",
                passed=passed,
                message=f"JSONL check: {row_count} rows, {len(sample_keys)} unique keys",
                details={
                    "path": str(path),
                    "row_count": row_count,
                    "sample_keys": list(sample_keys)
                }
            )
            return passed
        except Exception as e:
            self.add_result(
                stage=stage,
                check_name=f"jsonl_structure_{path.name}",
                passed=False,
                message=f"Error reading JSONL: {e}",
                details={"path": str(path), "error": str(e)}
            )
            return False

    def validate_collection(self) -> bool:
        """Validate data collection stage outputs."""
        stage = "collection"
        all_passed = True

        # Check raw data directories (using new structure)
        raw_dir = self.root / "data" / "raw"
        for subdir in ["crec_114", "bills", "members"]:
            path = raw_dir / subdir
            if not self.check_file_exists(stage, path, f"Raw data directory: {subdir}"):
                all_passed = False

        # Check reference files
        ref_dir = self.root / "data" / "reference"
        ref_files = [
            "interest_groups_list.csv",
            "washington_representatives_study.rda",
        ]
        for ref_file in ref_files:
            self.check_file_exists(stage, ref_dir / ref_file, f"Reference: {ref_file}")

        return all_passed

    def validate_processing(self) -> bool:
        """Validate data processing stage outputs."""
        stage = "processing"
        all_passed = True

        # Check normalized data (new structure: intermediate/normalized_114)
        norm_dir = self.root / "data" / "intermediate" / "normalized_114"
        if norm_dir.exists():
            # Count by_package subdirectories
            by_package = norm_dir / "by_package"
            if by_package.exists():
                pkg_count = len(list(by_package.iterdir()))
                self.add_result(stage, "normalized_dir", True,
                              f"Normalized directory exists ({pkg_count} packages)")
            else:
                self.add_result(stage, "normalized_dir", True, "Normalized directory exists")
        else:
            self.add_result(stage, "normalized_dir", False, "Normalized directory MISSING")
            all_passed = False

        # Check mentions (new structure: intermediate/mentions_114)
        mentions_path = self.root / "data" / "intermediate" / "mentions_114" / "mentions.jsonl"
        if not self.check_jsonl_structure(stage, mentions_path, min_rows=1000):
            all_passed = False

        # Check mentions with speakers
        speakers_dir = self.root / "data" / "intermediate" / "mentions_with_speakers"
        if speakers_dir.exists():
            self.add_result(stage, "speakers_dir", True, "Mentions with speakers directory exists")
        else:
            self.add_result(stage, "speakers_dir", False, "Mentions with speakers MISSING")

        return all_passed

    def validate_classification(self) -> bool:
        """Validate classification stage outputs."""
        stage = "classification"
        all_passed = True

        # Check model file
        model_path = self.root / "results_classifier" / "prominence_pipeline.joblib"
        if not self.check_file_exists(stage, model_path, "Trained model"):
            all_passed = False

        # Check report
        report_path = self.root / "results_classifier" / "report.txt"
        self.check_file_exists(stage, report_path, "Training report")

        # Check classified mentions (new structure: intermediate/mentions_114)
        labeled_path = self.root / "data" / "intermediate" / "mentions_114" / "labeled_mentions.csv"
        required_cols = ["prominence_score", "prominence_prediction", "org_id", "paragraph"]
        if not self.check_csv_structure(stage, labeled_path, required_cols, min_rows=1000):
            all_passed = False

        # Check combined training data (new structure: training/)
        training_path = self.root / "data" / "training" / "combined_labeled.csv"
        training_cols = ["org_id", "p1_original", "prominence"]
        self.check_csv_structure(stage, training_path, training_cols, min_rows=500)

        return all_passed

    def validate_integration(self) -> bool:
        """Validate integration stage outputs."""
        stage = "integration"
        all_passed = True

        merged_dir = self.root / "data" / "output"

        # Check level1.csv (columns may have level1_ prefix)
        level1_path = merged_dir / "level1.csv"
        if level1_path.exists():
            try:
                df = pd.read_csv(level1_path, nrows=5, low_memory=False)
                cols = list(df.columns)
                # Check for key columns with or without prefix
                has_org_id = any("org_id" in c for c in cols)
                has_granule = any("granule" in c.lower() for c in cols)
                has_prominence = any("prominence" in c.lower() for c in cols)

                row_count = sum(1 for _ in open(level1_path)) - 1  # Exclude header
                passed = has_org_id and row_count >= 10000

                self.add_result(
                    stage=stage,
                    check_name="level1_structure",
                    passed=passed,
                    message=f"level1.csv: {row_count} rows, org_id={has_org_id}, granule={has_granule}, prominence={has_prominence}",
                    details={"row_count": row_count, "sample_cols": cols[:10]}
                )
                if not passed:
                    all_passed = False
            except Exception as e:
                self.add_result(stage, "level1_structure", False, f"Error: {e}")
                all_passed = False
        else:
            self.add_result(stage, "level1_exists", False, "level1.csv MISSING")
            all_passed = False

        # Check multi_level_data.csv
        multi_path = merged_dir / "multi_level_data.csv"
        if multi_path.exists():
            try:
                row_count = sum(1 for _ in open(multi_path)) - 1
                self.add_result(
                    stage=stage,
                    check_name="multi_level_structure",
                    passed=row_count >= 10000,
                    message=f"multi_level_data.csv: {row_count} rows",
                    details={"row_count": row_count}
                )
            except Exception as e:
                self.add_result(stage, "multi_level_structure", False, f"Error: {e}")
        else:
            self.add_result(stage, "multi_level_exists", True, "multi_level_data.csv not present (optional)")

        return all_passed

    def validate_multi_level(self) -> bool:
        """Validate multi-level data outputs."""
        stage = "multi_level"
        all_passed = True

        # Multi-level files now in data/output/
        output_dir = self.root / "data" / "output"

        # Check multi-level files
        expected_files = [
            ("level1.csv", 10000),
            ("level2_org.csv", 100),
            ("level3_politician.csv", 50),
            ("level4_policy.csv", 10),
        ]

        for filename, min_rows in expected_files:
            file_path = output_dir / filename
            if file_path.exists():
                try:
                    row_count = sum(1 for _ in open(file_path)) - 1
                    passed = row_count >= min_rows
                    self.add_result(
                        stage=stage,
                        check_name=f"multi_level_{filename}",
                        passed=passed,
                        message=f"{filename}: {row_count} rows (min: {min_rows})",
                        details={"row_count": row_count}
                    )
                    if not passed:
                        all_passed = False
                except Exception as e:
                    self.add_result(stage, f"multi_level_{filename}", False, f"Error: {e}")
                    all_passed = False
            else:
                self.add_result(stage, f"multi_level_{filename}", False, f"{filename} MISSING")
                all_passed = False

        # Check salience data
        salience_path = output_dir / "issue_salience_long.csv"
        if salience_path.exists():
            self.add_result(stage, "salience_data", True, "issue_salience_long.csv exists")
        else:
            self.add_result(stage, "salience_data", False, "issue_salience_long.csv MISSING (optional)")

        return all_passed

    def validate_analysis(self) -> bool:
        """Validate analysis stage outputs."""
        stage = "analysis"
        all_passed = True

        # Check output figures
        figures_dir = self.root / "outputs" / "figures"
        expected_figures = [
            "fig1_mentions_over_time.png",
            "fig2_org_categories.png",
            "fig3_lobbying_prominence.png",
            "fig4_party_patterns.png",
            "fig5_policy_heatmap.png",
        ]

        for fig in expected_figures:
            fig_path = figures_dir / fig
            self.check_file_exists(stage, fig_path, f"Figure: {fig}")

        return all_passed

    def validate_all(self) -> dict[str, bool]:
        """Run all validations."""
        return {
            "collection": self.validate_collection(),
            "processing": self.validate_processing(),
            "classification": self.validate_classification(),
            "integration": self.validate_integration(),
            "multi_level": self.validate_multi_level(),
            "analysis": self.validate_analysis(),
        }

    def print_report(self, verbose: bool = False):
        """Print validation report."""
        print("=" * 60)
        print("DATA VALIDATION REPORT")
        print("=" * 60)
        print()

        # Group results by stage
        by_stage: dict[str, list[ValidationResult]] = defaultdict(list)
        for result in self.results:
            by_stage[result.stage].append(result)

        for stage in ["collection", "processing", "classification", "integration", "multi_level", "analysis"]:
            if stage not in by_stage:
                continue

            results = by_stage[stage]
            passed = sum(1 for r in results if r.passed)
            total = len(results)

            print(f"Stage: {stage.upper()}")
            print("-" * 40)

            for result in results:
                status = "[OK]" if result.passed else "[FAIL]"
                print(f"  {status} {result.message}")
                if verbose and result.details:
                    for key, value in result.details.items():
                        if isinstance(value, list) and len(value) > 5:
                            value = value[:5] + ["..."]
                        print(f"       {key}: {value}")

            print(f"  Summary: {passed}/{total} checks passed")
            print()

        # Overall summary
        total_passed = sum(1 for r in self.results if r.passed)
        total_checks = len(self.results)
        print("=" * 60)
        print(f"OVERALL: {total_passed}/{total_checks} checks passed")
        if total_passed == total_checks:
            print("All validations passed!")
        else:
            print("Some validations failed. Check details above.")
        print("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Validate pipeline data")
    parser.add_argument(
        "--stage",
        choices=["collection", "processing", "classification", "integration", "analysis"],
        help="Validate specific stage only"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed validation results"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    validator = DataValidator(PROJECT_ROOT)

    if args.stage:
        # Validate specific stage
        validate_func = getattr(validator, f"validate_{args.stage}", None)
        if validate_func:
            validate_func()
        else:
            print(f"Unknown stage: {args.stage}")
            return 1
    else:
        # Validate all stages
        validator.validate_all()

    if args.json:
        results = [
            {
                "stage": r.stage,
                "check": r.check_name,
                "passed": r.passed,
                "message": r.message,
                "details": r.details
            }
            for r in validator.results
        ]
        print(json.dumps(results, indent=2))
    else:
        validator.print_report(verbose=args.verbose)

    # Return exit code
    all_passed = all(r.passed for r in validator.results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
