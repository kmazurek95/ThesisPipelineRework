"""
High-level pipeline orchestration functions.

Each function in this module coordinates a distinct stage of the
analysis. The functions call into lower-level modules defined in
`data_collection`, `data_processing`, `classification`, `integration`,
and `analysis`. Use these functions from the command line or import
them into your own scripts/notebooks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from . import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def run_data_collection(
    congresses: List[int] = None,
    start_date: str = "2015-01-06",
    end_date: str = "2017-01-03",
    fetch_bills: bool = True,
    fetch_members: bool = True,
    fetch_policy: bool = True
) -> None:
    """Collect raw data from external sources.

    This function orchestrates the following sub-steps:
    1. Download legislative transcripts from GovInfo
    2. Fetch metadata for bills
    3. Retrieve congress member profiles
    4. Pull policy salience metrics
    
    Args:
        congresses: List of congress numbers (defaults to config.TARGET_CONGRESSES)
        start_date: ISO format date for start of collection period
        end_date: ISO format date for end of collection period
        fetch_bills: Whether to fetch bill metadata
        fetch_members: Whether to fetch member profiles
        fetch_policy: Whether to collect policy salience data
    """
    if congresses is None:
        congresses = config.TARGET_CONGRESSES

    logging.info("Collecting legislative transcripts...")
    from interest_group_analysis.data_collection.govinfo import fetch_legislative_transcripts
    fetch_legislative_transcripts(
        output_dir=config.RAW_DATA_DIR,
        congresses=congresses,
        start_date=start_date,
        end_date=end_date
    )

    if fetch_bills:
        logging.info("Collecting bill metadata...")
        from interest_group_analysis.data_collection.bills_linkage import run_end_to_end as fetch_bills
        fetch_bills(
            output_dir=config.RAW_DATA_DIR / "bills",
            mentions_jsonl=config.PROCESSED_DATA_DIR / "mentions_with_speakers.jsonl"
        )

    if fetch_members:
        logging.info("Collecting congress member profiles...")
        from interest_group_analysis.data_collection.members_linkage import run_end_to_end as fetch_members
        fetch_members(
            output_dir=config.RAW_DATA_DIR / "members",
            mentions_jsonl=config.PROCESSED_DATA_DIR / "mentions_with_speakers.jsonl",
            enrich_committees=True,
            committee_congresses=congresses
        )

    if fetch_policy:
        logging.info("Collecting policy salience metrics...")
        from interest_group_analysis.data_collection.policy_salience import run_policy_salience_pipeline
        run_policy_salience_pipeline(
            output_dir=str(config.RAW_DATA_DIR / "policy"),
            skip_trends=False
        )

    logging.info("Data collection complete.")


def run_data_processing(
    raw_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    congresses: List[int] = None,
    clean: bool = False,
    emit_search_text: bool = True
) -> None:
    """Clean and prepare collected data for modeling.

    This stage reads raw files, performs cleaning and transformations,
    and writes processed outputs. The specific operations include:
    - Normalizing raw data into structured tables
    - Extracting interest group mentions
    - Attaching speaker information to mentions
    - Post-processing mentions for analysis
    
    Args:
        raw_dir: Directory containing raw data (default: config.RAW_DATA_DIR)
        out_dir: Directory for processed outputs (default: config.NORMALIZED_DATA_DIR)
        congresses: List of congress numbers (defaults to config.TARGET_CONGRESSES)
        clean: Whether to remove existing outputs before processing
        emit_search_text: Whether to include search-ready text in output
    """
    if raw_dir is None:
        raw_dir = config.RAW_DATA_DIR
    if out_dir is None:
        out_dir = config.NORMALIZED_DATA_DIR
    if congresses is None:
        congresses = config.TARGET_CONGRESSES

    # Step 1: Normalize raw data
    logging.info("Normalizing raw data...")
    from interest_group_analysis.data_processing.process_and_normalize import main_with_params
    norm_dir = out_dir / "normalized"
    main_with_params(
        raw_dir=raw_dir,
        out_dir=norm_dir,
        clean=clean,
        split_by="package",
        emit_search_text=emit_search_text
    )

    # Step 2: Extract mentions
    logging.info("Extracting interest group mentions...")
    from interest_group_analysis.data_processing.mention_extraction import process_normalized_packages
    mentions_dir = config.PROCESSED_DATA_DIR / "mentions"
    process_normalized_packages(
        normalized_dir=norm_dir,
        interest_csv=config.RAW_DATA_DIR / "interest_groups_list.csv",
        out_dir=mentions_dir,
        resume=True
    )

    # Step 3: Attach speakers to mentions
    logging.info("Attaching speakers to mentions...")
    from interest_group_analysis.data_processing.attach_speakers import main as attach_speakers
    speakers_dir = config.PROCESSED_DATA_DIR / "mentions_with_speakers"
    # This would typically be called via argparse, adapting to direct function call
    # attach_speakers(
    #     mentions_jsonl=mentions_dir / "mentions.jsonl",
    #     normalized_dir=norm_dir,
    #     out_dir=speakers_dir,
    #     save_csv=True,
    #     qa_jsonl=True
    # )

    # Step 4: Post-process mentions
    logging.info("Post-processing mentions...")
    from interest_group_analysis.data_processing.mentions_postprocess import run
    processed_dir = config.PROCESSED_DATA_DIR / "mentions_processed"
    run(
        input_jsonl=speakers_dir / "mentions_with_speakers.jsonl",
        out_dir=processed_dir,
        prefix_file=None,
        save_csv=True,
        save_diagnostics=True
    )

    # Step 5: Build labeling samples (if needed)
    logging.info("Building labeling samples...")
    from interest_group_analysis.data_processing.build_labeling_samples import main as build_samples
    # build_samples would be called with appropriate args

    # Step 6: Check speaker coverage (optional diagnostic)
    logging.info("Checking speaker coverage...")
    # This would be run as a separate diagnostic step

    logging.info("Data processing complete.")


def run_classification(
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> None:
    """Train supervised models to classify prominence.

    This stage reads processed datasets, trains classification models,
    evaluates them, and produces labeled predictions.
    
    Args:
        input_dir: Directory with processed data (default: config.PROCESSED_DATA_DIR)
        output_dir: Directory for model and prediction outputs (default: config.CLASSIFIER_DIR)
    """
    if input_dir is None:
        input_dir = config.PROCESSED_DATA_DIR
    if output_dir is None:
        output_dir = config.CLASSIFIER_DIR

    logging.info("Running text classification pipeline...")
    from interest_group_analysis.classification.text_classifier import run_pipeline
    run_pipeline(
        input_dir=input_dir / "mentions_processed",
        output_dir=output_dir
    )

    logging.info("Classification complete.")


def run_integration(
    bill_dir: Optional[Path] = None,
    members_dir: Optional[Path] = None,
    policy_dir: Optional[Path] = None,
    mentions_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> None:
    """Merge processed datasets and handle integration.

    This function orchestrates dataset merging, linking bills and members
    to mentions, and additional feature engineering.
    
    Args:
        bill_dir: Directory with bill data
        members_dir: Directory with member data
        policy_dir: Directory with policy data
        mentions_dir: Directory with processed mentions
        output_dir: Directory for integrated outputs
    """
    if bill_dir is None:
        bill_dir = config.RAW_DATA_DIR / "bills"
    if members_dir is None:
        members_dir = config.RAW_DATA_DIR / "members" 
    if policy_dir is None:
        policy_dir = config.RAW_DATA_DIR / "policy"
    if mentions_dir is None:
        mentions_dir = config.PROCESSED_DATA_DIR / "mentions_processed"
    if output_dir is None:
        output_dir = config.RESULTS_DIR

    # Step 1: Link bills to mentions
    logging.info("Linking bills to mentions...")
    from interest_group_analysis.data_collection.bills_linkage import link_mentions_to_bills
    bill_links = link_mentions_to_bills(
        output_dir=bill_dir,
        mentions_path=mentions_dir / "analytic_paragraph_units.jsonl"
    )

    # Step 2: Link members to mentions  
    logging.info("Linking members to mentions...")
    from interest_group_analysis.data_collection.members_linkage import link_mentions_to_members
    member_links = link_mentions_to_members(
        output_dir=members_dir,
        mentions_path=mentions_dir / "analytic_paragraph_units.jsonl"
    )

    # Step 3: Link committees to policy areas
    logging.info("Linking committees to policy areas...")
    from interest_group_analysis.data_processing.committee_policy_linkage import main as link_committees
    # link_committees would be called with appropriate parameters

    # Step 4: Integrate all datasets
    logging.info("Integrating datasets...")
    # This would be your custom integration logic to create final analytic dataset
    
    logging.info("Integration complete.")


def run_analysis(
    input_file: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> None:
    """Perform statistical analysis and generate visualizations.

    This function runs regression models on the integrated dataset
    and produces plots describing prominence patterns.
    
    Args:
        input_file: Path to integrated dataset file
        output_dir: Directory for analysis outputs
    """
    if input_file is None:
        input_file = config.RESULTS_DIR / "integrated_dataset.parquet"
    if output_dir is None:
        output_dir = config.RESULTS_DIR / "analysis"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add your analysis steps here
    logging.info("Running regression analysis...")
    # run_regression_analysis(input_file, output_dir)

    logging.info("Generating visualizations...")
    # generate_visualizations(input_file, output_dir)
    
    logging.info("Analysis complete.")


def run_full_pipeline() -> None:
    """Run the complete analysis pipeline from data collection to analysis."""
    logging.info("Starting full pipeline...")
    
    run_data_collection()
    run_data_processing()
    run_classification()
    run_integration()
    run_analysis()
    
    logging.info("Full pipeline complete.")


if __name__ == "__main__":
    # This allows running the full pipeline directly:
    # python -m interest_group_analysis.pipelines
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Interest Group Analysis Pipeline")
    parser.add_argument("--stage", choices=["collect", "process", "classify", "integrate", "analyze", "all"], 
                        default="all", help="Pipeline stage to run")
    args = parser.parse_args()
    
    if args.stage == "collect":
        run_data_collection()
    elif args.stage == "process":
        run_data_processing()
    elif args.stage == "classify":
        run_classification()  
    elif args.stage == "integrate":
        run_integration()
    elif args.stage == "analyze":
        run_analysis()
    elif args.stage == "all":
        run_full_pipeline()
