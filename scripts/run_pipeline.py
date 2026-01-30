#!/usr/bin/env python3
"""
ETL Pipeline Runner for Interest Group Analysis

This script provides a unified entry point for running the complete data pipeline
or individual stages. It reads configuration from config/pipeline_config.yaml
and orchestrates the execution of each stage.

Usage:
    # Run full pipeline
    python scripts/run_pipeline.py

    # Run specific stage
    python scripts/run_pipeline.py --stage classify

    # Run multiple stages
    python scripts/run_pipeline.py --stage process classify integrate

    # Dry run (show what would be executed)
    python scripts/run_pipeline.py --dry-run

    # Skip validation
    python scripts/run_pipeline.py --skip-validation
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """Load pipeline configuration from YAML file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "pipeline_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def resolve_path(path_str: str) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(config: dict[str, Any]) -> logging.Logger:
    """Configure logging based on config settings."""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    fmt = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create logs directory if needed
    log_file = log_config.get("file")
    if log_file:
        log_path = resolve_path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    return logging.getLogger("pipeline")


# =============================================================================
# Validation Functions
# =============================================================================

def validate_checkpoint(checkpoint: dict[str, Any], logger: logging.Logger) -> bool:
    """Run a single validation checkpoint."""
    check_type = checkpoint.get("check")
    path = resolve_path(checkpoint.get("path", ""))

    if check_type == "file_exists":
        if path.exists():
            logger.info(f"  [OK] File exists: {path}")
            return True
        else:
            logger.error(f"  [FAIL] File missing: {path}")
            return False

    elif check_type == "min_files":
        min_count = checkpoint.get("min_count", 1)
        if path.is_dir():
            count = len(list(path.iterdir()))
            if count >= min_count:
                logger.info(f"  [OK] Directory has {count} files (min: {min_count})")
                return True
            else:
                logger.error(f"  [FAIL] Directory has {count} files (min: {min_count})")
                return False
        else:
            logger.error(f"  [FAIL] Path is not a directory: {path}")
            return False

    elif check_type == "min_rows":
        min_count = checkpoint.get("min_count", 1)
        if path.exists():
            # Count lines (for CSV/JSONL)
            with open(path, "r", encoding="utf-8") as f:
                count = sum(1 for _ in f)
            if count >= min_count:
                logger.info(f"  [OK] File has {count} rows (min: {min_count})")
                return True
            else:
                logger.error(f"  [FAIL] File has {count} rows (min: {min_count})")
                return False
        else:
            logger.error(f"  [FAIL] File missing: {path}")
            return False

    else:
        logger.warning(f"  [SKIP] Unknown check type: {check_type}")
        return True


def run_validation(stage_name: str, config: dict[str, Any], logger: logging.Logger) -> bool:
    """Run all validation checkpoints for a stage."""
    validation_config = config.get("validation", {}).get("checkpoints", {})
    checkpoint_key = f"after_{stage_name}"
    checkpoints = validation_config.get(checkpoint_key, [])

    if not checkpoints:
        logger.info(f"No validation checkpoints for stage: {stage_name}")
        return True

    logger.info(f"Running validation for stage: {stage_name}")
    all_passed = True
    for checkpoint in checkpoints:
        if not validate_checkpoint(checkpoint, logger):
            all_passed = False

    return all_passed


# =============================================================================
# Pipeline Stage Functions
# =============================================================================

def run_collection(config: dict[str, Any], logger: logging.Logger, dry_run: bool = False) -> bool:
    """Run the data collection stage."""
    stage_config = config.get("stages", {}).get("collection", {})
    if not stage_config.get("enabled", True):
        logger.info("Collection stage is disabled, skipping...")
        return True

    logger.info("=" * 60)
    logger.info("STAGE 1: DATA COLLECTION")
    logger.info("=" * 60)

    if dry_run:
        logger.info("[DRY RUN] Would run data collection")
        return True

    try:
        from interest_group_analysis.pipelines import run_data_collection

        govinfo_config = stage_config.get("govinfo", {})
        run_data_collection(
            congresses=config.get("project", {}).get("target_congresses", [114, 115]),
            start_date=govinfo_config.get("start_date", "2015-01-06"),
            end_date=govinfo_config.get("end_date", "2017-01-03"),
            fetch_bills=stage_config.get("fetch_bills", True),
            fetch_members=stage_config.get("fetch_members", True),
            fetch_policy=stage_config.get("fetch_policy", True),
        )
        return True
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        return False


def run_processing(config: dict[str, Any], logger: logging.Logger, dry_run: bool = False) -> bool:
    """Run the data processing stage."""
    stage_config = config.get("stages", {}).get("processing", {})
    if not stage_config.get("enabled", True):
        logger.info("Processing stage is disabled, skipping...")
        return True

    logger.info("=" * 60)
    logger.info("STAGE 2: DATA PROCESSING")
    logger.info("=" * 60)

    if dry_run:
        logger.info("[DRY RUN] Would run data processing")
        return True

    try:
        from interest_group_analysis.pipelines import run_data_processing

        run_data_processing(
            clean=stage_config.get("normalize", {}).get("clean", False),
            emit_search_text=stage_config.get("normalize", {}).get("emit_search_text", True),
        )
        return True
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return False


def run_classification(config: dict[str, Any], logger: logging.Logger, dry_run: bool = False) -> bool:
    """Run the classification stage."""
    stage_config = config.get("stages", {}).get("classification", {})
    if not stage_config.get("enabled", True):
        logger.info("Classification stage is disabled, skipping...")
        return True

    logger.info("=" * 60)
    logger.info("STAGE 3: CLASSIFICATION")
    logger.info("=" * 60)

    paths_config = config.get("paths", {})

    if dry_run:
        logger.info("[DRY RUN] Would run classification")
        return True

    try:
        # Step 1: Prepare training data (if needed)
        combined_path = resolve_path(paths_config.get("labeled_data", "data/combined_labeled.csv"))
        if not combined_path.exists() or stage_config.get("use_combined_labels", True):
            logger.info("Preparing combined training data...")
            import subprocess
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "prepare_training_data.py")],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            if result.returncode != 0:
                logger.error(f"Training data preparation failed: {result.stderr}")
                return False
            logger.info("Training data prepared successfully")

        # Step 2: Train classifier
        logger.info("Training classifier...")
        classifier_script = PROJECT_ROOT / "interest_group_analysis" / "3.classification" / "text_classifier.py"
        import subprocess
        result = subprocess.run(
            [sys.executable, str(classifier_script)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        if result.returncode != 0:
            logger.error(f"Classifier training failed: {result.stderr}")
            return False
        logger.info("Classifier trained successfully")

        # Step 3: Apply classifier to mentions
        logger.info("Applying classifier to mentions...")
        classify_script = PROJECT_ROOT / "interest_group_analysis" / "3.classification" / "classify_mentions.py"
        result = subprocess.run(
            [sys.executable, str(classify_script)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        if result.returncode != 0:
            logger.error(f"Classification failed: {result.stderr}")
            return False
        logger.info("Classification complete")

        return True
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return False


def run_integration(config: dict[str, Any], logger: logging.Logger, dry_run: bool = False) -> bool:
    """Run the integration stage."""
    stage_config = config.get("stages", {}).get("integration", {})
    if not stage_config.get("enabled", True):
        logger.info("Integration stage is disabled, skipping...")
        return True

    logger.info("=" * 60)
    logger.info("STAGE 4: INTEGRATION")
    logger.info("=" * 60)

    if dry_run:
        logger.info("[DRY RUN] Would run integration")
        return True

    try:
        # Run complete_merge.py
        logger.info("Running complete merge pipeline...")
        merge_script = PROJECT_ROOT / "interest_group_analysis" / "4_integration" / "complete_merge.py"

        if merge_script.exists():
            import subprocess
            result = subprocess.run(
                [sys.executable, str(merge_script)],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            if result.returncode != 0:
                logger.error(f"Integration failed: {result.stderr}")
                return False
            logger.info("Integration complete")
        else:
            logger.warning(f"Merge script not found: {merge_script}")
            logger.info("Running fallback integration via pipelines module...")
            from interest_group_analysis.pipelines import run_integration as run_int
            run_int()

        return True
    except Exception as e:
        logger.error(f"Integration failed: {e}")
        return False


def run_analysis(config: dict[str, Any], logger: logging.Logger, dry_run: bool = False) -> bool:
    """Run the analysis stage."""
    stage_config = config.get("stages", {}).get("analysis", {})
    if not stage_config.get("enabled", True):
        logger.info("Analysis stage is disabled, skipping...")
        return True

    logger.info("=" * 60)
    logger.info("STAGE 5: ANALYSIS")
    logger.info("=" * 60)

    if dry_run:
        logger.info("[DRY RUN] Would run analysis")
        return True

    try:
        # Run multi-level data construction if available
        multi_level_script = PROJECT_ROOT / "interest_group_analysis" / "5_analysis" / "multi_level_construction.py"
        if multi_level_script.exists():
            logger.info("Running multi-level data construction...")
            import subprocess
            result = subprocess.run(
                [sys.executable, str(multi_level_script)],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            if result.returncode != 0:
                logger.warning(f"Multi-level construction had issues: {result.stderr}")

        # Run visualizations if available
        viz_script = PROJECT_ROOT / "interest_group_analysis" / "5_analysis" / "visualizations.py"
        if viz_script.exists():
            logger.info("Generating visualizations...")
            import subprocess
            result = subprocess.run(
                [sys.executable, str(viz_script)],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            if result.returncode != 0:
                logger.warning(f"Visualization generation had issues: {result.stderr}")

        logger.info("Analysis stage complete")
        return True
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return False


# =============================================================================
# Main Pipeline Orchestration
# =============================================================================

STAGE_FUNCTIONS: dict[str, Callable] = {
    "collect": run_collection,
    "process": run_processing,
    "classify": run_classification,
    "integrate": run_integration,
    "analyze": run_analysis,
}

STAGE_ORDER = ["collect", "process", "classify", "integrate", "analyze"]

STAGE_VALIDATION_MAP = {
    "collect": "collection",
    "process": "processing",
    "classify": "classification",
    "integrate": "integration",
    "analyze": "analysis",
}


def run_pipeline(
    stages: Optional[list[str]] = None,
    config: Optional[dict[str, Any]] = None,
    dry_run: bool = False,
    skip_validation: bool = False,
) -> bool:
    """
    Run the ETL pipeline.

    Args:
        stages: List of stages to run. If None, runs all stages.
        config: Configuration dict. If None, loads from default path.
        dry_run: If True, show what would be executed without running.
        skip_validation: If True, skip validation checkpoints.

    Returns:
        True if all stages completed successfully, False otherwise.
    """
    # Load config if not provided
    if config is None:
        config = load_config()

    # Setup logging
    logger = setup_logging(config)

    # Determine stages to run
    if stages is None:
        stages = STAGE_ORDER
    else:
        # Validate stage names
        invalid = set(stages) - set(STAGE_ORDER)
        if invalid:
            logger.error(f"Invalid stage names: {invalid}")
            return False
        # Sort by execution order
        stages = [s for s in STAGE_ORDER if s in stages]

    # Log pipeline start
    logger.info("=" * 60)
    logger.info("INTEREST GROUP ANALYSIS PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info(f"Stages to run: {stages}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"Skip validation: {skip_validation}")
    logger.info("")

    # Run each stage
    start_time = time.time()
    results: dict[str, bool] = {}

    for stage in stages:
        stage_start = time.time()
        logger.info(f"Starting stage: {stage}")

        # Run stage
        stage_func = STAGE_FUNCTIONS[stage]
        success = stage_func(config, logger, dry_run)
        results[stage] = success

        # Log stage result
        stage_duration = time.time() - stage_start
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"Stage {stage} {status} (duration: {stage_duration:.1f}s)")

        if not success:
            logger.error(f"Pipeline stopped due to failure in stage: {stage}")
            break

        # Run validation (unless skipped or dry run)
        if not skip_validation and not dry_run:
            validation_key = STAGE_VALIDATION_MAP.get(stage, stage)
            if not run_validation(validation_key, config, logger):
                logger.warning(f"Validation failed for stage: {stage}")
                # Continue anyway, just warn

    # Log summary
    total_duration = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Completed at: {datetime.now().isoformat()}")
    logger.info(f"Total duration: {total_duration:.1f}s")
    logger.info("")
    for stage, success in results.items():
        status = "OK" if success else "FAIL"
        logger.info(f"  {stage}: [{status}]")

    all_success = all(results.values())
    if all_success:
        logger.info("")
        logger.info("Pipeline completed successfully!")
    else:
        logger.error("")
        logger.error("Pipeline completed with errors.")

    return all_success


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for the pipeline runner."""
    parser = argparse.ArgumentParser(
        description="Run the Interest Group Analysis ETL Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py                    # Run full pipeline
  python scripts/run_pipeline.py --stage classify   # Run classification only
  python scripts/run_pipeline.py --stage process classify integrate
  python scripts/run_pipeline.py --dry-run          # Preview what would run
  python scripts/run_pipeline.py --skip-validation  # Skip validation checks
        """
    )

    parser.add_argument(
        "--stage",
        nargs="+",
        choices=STAGE_ORDER,
        help="Specific stage(s) to run. If not specified, runs all stages."
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file (default: config/pipeline_config.yaml)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without actually running"
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation checkpoints between stages"
    )

    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="List available pipeline stages and exit"
    )

    args = parser.parse_args()

    if args.list_stages:
        print("Available pipeline stages:")
        print("  collect   - Fetch data from external APIs (GovInfo, Congress)")
        print("  process   - Normalize data and extract mentions")
        print("  classify  - Train and apply prominence classifier")
        print("  integrate - Merge all data sources into final dataset")
        print("  analyze   - Run statistical models and generate visualizations")
        return 0

    # Load config
    config = load_config(args.config) if args.config else load_config()

    # Run pipeline
    success = run_pipeline(
        stages=args.stage,
        config=config,
        dry_run=args.dry_run,
        skip_validation=args.skip_validation,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
