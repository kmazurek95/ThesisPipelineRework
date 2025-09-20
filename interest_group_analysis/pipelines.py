"""
High‑level pipeline orchestration functions.

Each function in this module coordinates a distinct stage of the
analysis.  The functions call into lower‑level modules defined in
`data_collection`, `data_processing`, `classification`, `integration`,
and `analysis`.  Use these functions from the command line or import
them into your own scripts/notebooks.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .data_processing.Archived import api_results, expand_nested, expander, members, speaker_assignment, stats

from . import config
from .data_collection import (
    govinfo,
    bill_metadata,
    congress_members,
    policy_salience,
)
from .data_processing import (
    mention_extraction,
    utils as processing_utils,
)
from .classification import text_classifier
from .integration import merge_datasets
from .analysis import regression, visualizations


def run_data_collection() -> None:
    """Collect raw data from external sources.

    This function orchestrates the following sub‑steps:

    1. Download legislative transcripts from GovInfo.
    2. Fetch metadata for bills.
    3. Retrieve congress member profiles.
    4. Pull policy salience metrics from Google Trends.

    Each sub‑function writes its output to `config.RAW_DATA_DIR`.  It
    is safe to call this function multiple times; existing files
    should be overwritten or skipped based on your implementation.
    """
    logging.info("Collecting legislative transcripts…")
    govinfo.fetch_legislative_transcripts(output_dir=config.RAW_DATA_DIR)

    logging.info("Collecting bill metadata…")
    bill_metadata.fetch_bill_metadata(output_dir=config.RAW_DATA_DIR)

    logging.info("Collecting congress member profiles…")
    congress_members.fetch_congress_member_profiles(output_dir=config.RAW_DATA_DIR)

    logging.info("Collecting policy salience metrics…")
    policy_salience.fetch_policy_salience(output_dir=config.RAW_DATA_DIR)


def run_data_processing() -> None:
    """Clean and prepare collected data for modelling.

    This stage reads raw files from `config.RAW_DATA_DIR`, performs
    cleaning and transformations, and writes processed outputs to
    `config.PROCESSED_DATA_DIR`.  The specific operations include:

    - Parsing API responses into structured tables.
    - Assigning speaker information to transcripts.
    - Extracting and highlighting interest group mentions.
    - Expanding nested JSON structures.

    You can customise or extend the pipeline by editing the
    corresponding functions in `data_processing/`.
    """
    logging.info("Processing API results…")
    api_results.process_results(
        input_dir=config.RAW_DATA_DIR,
        output_dir=config.PROCESSED_DATA_DIR,
    )

    logging.info("Assigning speakers to transcripts…")
    speaker_assignment.assign_speakers(
        input_dir=config.PROCESSED_DATA_DIR,
        output_dir=config.PROCESSED_DATA_DIR,
    )

    logging.info("Extracting interest group mentions…")
    mention_extraction.extract_mentions(
        input_dir=config.PROCESSED_DATA_DIR,
        output_dir=config.PROCESSED_DATA_DIR,
    )

    logging.info("Expanding nested structures (legacy expander)…")
    # keep calling legacy expander if present
    try:
        expander.expand_nested_data(
            input_dir=config.PROCESSED_DATA_DIR,
            output_dir=config.PROCESSED_DATA_DIR,
        )
    except Exception:
        logging.debug("Legacy expander not available or failed; continuing.", exc_info=True)

    # Pick a processed JSON/JSONL input to drive downstream flattening steps.
    # Prefer a canonical file name if present, otherwise use the first JSON/JSONL file.
    def _find_processed_input(directory: Path) -> Path | None:
        candidates = [
            directory / "api_results.jsonl",
            directory / "api_results.json",
            directory / "final_result.jsonl",
            directory / "final_result.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        # fallback: first .jsonl or .json in the directory
        for ext in ("*.jsonl", "*.json"):
            found = sorted(directory.glob(ext))
            if found:
                return found[0]
        return None

    input_json = _find_processed_input(config.PROCESSED_DATA_DIR)
    if input_json is None:
        logging.warning("No processed JSON input found in %s — skipping members/expand/stats steps", config.PROCESSED_DATA_DIR)
        return

    logging.info("Creating members-wide table from %s …", input_json.name)
    try:
        members_out = config.PROCESSED_DATA_DIR / "members_wide.csv"
        members.run_members_wide(input_json, members_out)
    except Exception:
        logging.exception("members.run_members_wide failed for %s", input_json)

    logging.info("Expanding nested structures to wide CSV from %s …", input_json.name)
    try:
        expanded_out = config.PROCESSED_DATA_DIR / "expanded_nested.csv"
        expand_nested.run_expand_nested(input_json, expanded_out)
    except Exception:
        logging.exception("expand_nested.run_expand_nested failed for %s", input_json)

    logging.info("Writing basic counts/statistics for %s …", input_json.name)
    try:
        stats_out = config.PROCESSED_DATA_DIR / "basic_counts.txt"
        stats.write_basic_counts(input_json, stats_out)
    except Exception:
        logging.exception("stats.write_basic_counts failed for %s", input_json)


def run_classification() -> None:
    """Train supervised models to classify prominence and predict labels.

    This stage reads cleaned datasets from `config.PROCESSED_DATA_DIR`,
    splits labeled data into training/validation/test sets, trains
    multiple models, evaluates them, and labels unlabeled data.  The
    labelled predictions and trained model artefacts are written to
    `config.RESULTS_DIR`.
    """
    logging.info("Running text classification pipeline…")
    text_classifier.run_pipeline(
        processed_data_dir=config.PROCESSED_DATA_DIR,
        results_dir=config.RESULTS_DIR,
    )


def run_integration() -> None:
    """Merge processed and classified datasets and handle deduplication.

    This function orchestrates dataset merging, duplicate detection,
    and additional feature engineering (e.g. deriving congressional
    session from dates).  The resulting integrated dataset is saved
    into `config.RESULTS_DIR`.
    """
    logging.info("Integrating and deduplicating datasets…")
    merge_datasets.integrate_datasets(
        processed_data_dir=config.PROCESSED_DATA_DIR,
        classification_results_dir=config.RESULTS_DIR,
        output_dir=config.RESULTS_DIR,
    )


def run_analysis() -> None:
    """Perform statistical analysis and generate visualisations.

    This function runs regression models on the integrated dataset
    stored in `config.RESULTS_DIR` and produces plots describing
    variable distributions and model outcomes.  Results are saved
    back to `config.RESULTS_DIR`.
    """
    integrated_dataset: Path = config.RESULTS_DIR / "integrated_dataset.csv"
    if not integrated_dataset.exists():
        logging.error("Integrated dataset not found at %s", integrated_dataset)
        return

    logging.info("Running regression analysis…")
    regression.run_regression_analysis(
        input_path=integrated_dataset,
        output_dir=config.RESULTS_DIR,
    )

    logging.info("Generating visualisations…")
    visualizations.generate_plots(
        input_path=integrated_dataset,
        output_dir=config.RESULTS_DIR,
    )
