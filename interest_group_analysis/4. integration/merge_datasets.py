"""
Integration and deduplication of datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


# Placeholder for integrate_datasets function.  A detailed
# implementation will be added later.
def integrate_datasets(
    processed_data_dir: Path,
    classification_results_dir: Path,
    output_dir: Path,
) -> None:
    """Merge processed datasets with classification predictions.

    Parameters
    ----------
    processed_data_dir : Path
        Directory containing cleaned/processed CSV files.
    classification_results_dir : Path
        Directory containing classifier outputs (predictions).
    output_dir : Path
        Directory where the integrated dataset will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Locate classification results
    classified_path = classification_results_dir / "labeled_unlabeled_data.csv"
    if not classified_path.exists():
        logging.error("Classification results not found at %s", classified_path)
        return
    try:
        classified_df = pd.read_csv(classified_path)
    except Exception as exc:
        logging.error("Failed to load classification results: %s", exc)
        return

    # Gather all processed CSV files
    processed_files = list(processed_data_dir.glob("*.csv"))
    if not processed_files:
        logging.error("No processed CSV files found in %s", processed_data_dir)
        return
    frames = []
    for pf in processed_files:
        try:
            frames.append(pd.read_csv(pf))
        except Exception as exc:
            logging.error("Failed to load processed file %s: %s", pf, exc)
    if not frames:
        logging.error("No processed data loaded; aborting integration.")
        return
    processed_df = pd.concat(frames, ignore_index=True)

    # TODO: determine join keys; here we simply concatenate along columns
    logging.info("Merging processed data with classification results…")
    merged_df = pd.concat(
        [processed_df.reset_index(drop=True), classified_df.reset_index(drop=True)],
        axis=1,
    )

    # Save integrated dataset
    out_path = output_dir / "integrated_dataset.csv"
    merged_df.to_csv(out_path, index=False)