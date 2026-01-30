"""
Pipeline Orchestration Module

This module provides high-level pipeline orchestration functions. Due to the
numbered folder naming convention used in this project (e.g., 1.data_collection),
direct imports aren't straightforward. Use these functions via subprocess calls
or run the individual stage scripts directly.

Recommended Usage:
    # Use the run_pipeline.py script for full control
    python scripts/run_pipeline.py --stage collect
    python scripts/run_pipeline.py --stage process
    python scripts/run_pipeline.py --stage classify
    python scripts/run_pipeline.py --stage integrate
    python scripts/run_pipeline.py --stage analyze

    # Or run individual modules directly
    python -m interest_group_analysis.4_integration.build_analysis_dataset
    python -m interest_group_analysis.5_analysis.regression_analysis

Pipeline Stages:
    1. Data Collection: Download raw data from GovInfo, Congress.gov APIs
    2. Data Processing: Normalize, extract mentions, attach speakers
    3. Classification: Train and apply ML prominence classifier
    4. Integration: Merge all data sources into analysis-ready datasets
    5. Analysis: Run regression models and generate visualizations
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from . import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _run_script(script_path: Path, *args, cwd: Optional[Path] = None) -> int:
    """Run a Python script as a subprocess.

    Args:
        script_path: Path to the Python script
        *args: Additional arguments to pass to the script
        cwd: Working directory for the subprocess

    Returns:
        Exit code from the subprocess
    """
    if cwd is None:
        cwd = config.BASE_DIR

    cmd = [sys.executable, str(script_path)] + list(args)
    logger.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=str(cwd))
    return result.returncode


def run_data_collection(
    congresses: List[int] = None,
    start_date: str = "2015-01-06",
    end_date: str = "2017-01-03",
) -> int:
    """Collect raw data from external sources.

    Orchestrates the following data collection steps:
    1. Download legislative transcripts from GovInfo API
    2. Fetch bill metadata from Congress.gov API
    3. Retrieve congress member profiles
    4. Collect policy salience data from Google Trends

    Args:
        congresses: List of congress numbers (default: [114])
        start_date: ISO format start date for collection
        end_date: ISO format end date for collection

    Returns:
        Exit code (0 = success)
    """
    if congresses is None:
        congresses = config.TARGET_CONGRESSES

    script_dir = config.BASE_DIR / "interest_group_analysis" / "1.data_collection"

    # Run GovInfo collection
    logger.info("Collecting legislative transcripts from GovInfo...")
    govinfo_script = script_dir / "1.govinfo.py"
    if govinfo_script.exists():
        _run_script(govinfo_script)

    # Run bills collection
    logger.info("Collecting bill metadata...")
    bills_script = script_dir / "2.bills_linkage.py"
    if bills_script.exists():
        _run_script(bills_script)

    # Run members collection
    logger.info("Collecting member profiles...")
    members_script = script_dir / "3.members_linkage.py"
    if members_script.exists():
        _run_script(members_script)

    logger.info("Data collection complete.")
    return 0


def run_data_processing() -> int:
    """Clean and prepare collected data for modeling.

    Orchestrates the following processing steps:
    1. Normalize raw Congressional Record XML
    2. Extract interest group mentions
    3. Attach speaker information to mentions
    4. Post-process mentions for analysis

    Returns:
        Exit code (0 = success)
    """
    script_dir = config.BASE_DIR / "interest_group_analysis" / "2.data_processing"

    # Run normalization
    logger.info("Normalizing raw data...")
    _run_script(script_dir / "1.process_and_normalize.py")

    # Extract mentions
    logger.info("Extracting mentions...")
    _run_script(script_dir / "2.mention_extraction.py")

    # Attach speakers
    logger.info("Attaching speakers...")
    _run_script(script_dir / "3.attach_speakers.py")

    # Post-process
    logger.info("Post-processing...")
    _run_script(script_dir / "4.mentions_postprocess.py")

    logger.info("Data processing complete.")
    return 0


def run_classification() -> int:
    """Train and apply the prominence classifier.

    Uses TF-IDF + Logistic Regression to classify mention prominence.
    Training data is located in data/training/combined_labeled.csv.

    Returns:
        Exit code (0 = success)
    """
    script_dir = config.BASE_DIR / "interest_group_analysis" / "3.classification"

    logger.info("Training classifier...")
    _run_script(script_dir / "text_classifier.py")

    logger.info("Applying classifier to mentions...")
    _run_script(script_dir / "classify_mentions.py")

    logger.info("Classification complete.")
    return 0


def run_integration() -> int:
    """Merge all data sources into analysis-ready datasets.

    Creates multi-level datasets:
    - level1.csv: Individual mentions
    - level2_org.csv: Organization aggregations
    - level3_politician.csv: Politician aggregations
    - level4_policy.csv: Policy area aggregations

    Returns:
        Exit code (0 = success)
    """
    script_dir = config.BASE_DIR / "interest_group_analysis" / "4_integration"

    logger.info("Running integration pipeline...")
    return _run_script(script_dir / "build_analysis_dataset.py")


def run_analysis() -> int:
    """Run statistical analysis and generate visualizations.

    Produces:
    - Regression results (outputs/tables/)
    - Visualizations (outputs/figures/)

    Returns:
        Exit code (0 = success)
    """
    script_dir = config.BASE_DIR / "interest_group_analysis" / "5_analysis"

    logger.info("Running descriptive analysis...")
    _run_script(script_dir / "descriptive_analysis.py")

    logger.info("Running regression analysis...")
    _run_script(script_dir / "regression_analysis.py")

    logger.info("Analysis complete.")
    return 0


def run_full_pipeline() -> int:
    """Run the complete analysis pipeline from collection to analysis.

    Returns:
        Exit code (0 = success, non-zero = failure at some stage)
    """
    logger.info("Starting full pipeline...")

    stages = [
        ("Data Collection", run_data_collection),
        ("Data Processing", run_data_processing),
        ("Classification", run_classification),
        ("Integration", run_integration),
        ("Analysis", run_analysis),
    ]

    for stage_name, stage_func in stages:
        logger.info(f"Running stage: {stage_name}")
        result = stage_func()
        if result != 0:
            logger.error(f"Stage '{stage_name}' failed with code {result}")
            return result

    logger.info("Full pipeline complete.")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Interest Group Analysis Pipeline")
    parser.add_argument(
        "--stage",
        choices=["collect", "process", "classify", "integrate", "analyze", "all"],
        default="all",
        help="Pipeline stage to run"
    )
    args = parser.parse_args()

    stage_map = {
        "collect": run_data_collection,
        "process": run_data_processing,
        "classify": run_classification,
        "integrate": run_integration,
        "analyze": run_analysis,
        "all": run_full_pipeline,
    }

    sys.exit(stage_map[args.stage]())
