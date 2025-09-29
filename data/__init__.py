"""
Data folder structure for ThesisPipelineRework

This file documents and provides programmatic access to the organized data pipeline folders.
Each subfolder corresponds to a specific stage or output in the analysis pipeline.

Structure:

raw/                # Raw data from APIs and external sources
  congresses/       # Congress-specific raw data
    114/            # Data for 114th Congress
      YYYY-MM-DD/   # Date-stamped runs
    115/
      YYYY-MM-DD/
  interest_groups/  # Interest group reference data
    YYYY-MM-DD/

normalized/         # Output from process_and_normalize.py
  114/
    YYYY-MM-DD/
      by_package/
      manifest.json
      summary.json
    latest -> YYYY-MM-DD/
  115/
    ...

mentions/           # Output from mention_extraction.py
  114/
    YYYY-MM-DD/
      mentions.jsonl
      by_package/
    latest -> YYYY-MM-DD/
  115/
    ...

speakers/           # Output from attach_speakers.py
  114/
    YYYY-MM-DD/
      mentions_with_speakers.jsonl
      speaker_qc.jsonl
    latest -> YYYY-MM-DD/
  115/
    ...

processed/          # Output from mentions_postprocess.py
  114/
    YYYY-MM-DD/
      analytic_paragraph_units.jsonl
      analytic_windows.jsonl
      diagnostics/
    latest -> YYYY-MM-DD/
  115/
    ...

members/            # Output from members_linkage.py
  congress_members.parquet
  congress_member_terms.parquet
  mention_speaker_links.parquet

bills/              # Output from bills_linkage.py
  bill_metadata.parquet
  mention_bill_links.parquet

policies/           # Output from committee_policy_linkage.py
  granule_dominant_policy.parquet

labeling/           # Output from build_labeling_samples.py
  YYYY-MM-DD_batch1/
  YYYY-MM-DD_batch2/

Use the DataPaths class below to programmatically access key folders.
"""

from pathlib import Path

class DataPaths:
    """Convenience class for accessing key data folders."""
    BASE = Path(__file__).parent.resolve()

    RAW = BASE / "raw"
    RAW_CONGRESSES = RAW / "congresses"
    RAW_INTEREST_GROUPS = RAW / "interest_groups"

    NORMALIZED = BASE / "normalized"
    MENTIONS = BASE / "mentions"
    SPEAKERS = BASE / "speakers"
    PROCESSED = BASE / "processed"
    MEMBERS = BASE / "members"
    BILLS = BASE / "bills"
    POLICIES = BASE / "policies"
    LABELING = BASE / "labeling"

    @staticmethod
    def congress_folder(stage: str, congress: int, date: str = None):
        """
        Get the folder for a given stage (normalized, mentions, speakers, processed), congress, and date.
        Example: DataPaths.congress_folder('normalized', 114, '2025-09-29')
        """
        base = getattr(DataPaths, stage.upper(), None)
        if base is None:
            raise ValueError(f"Unknown stage: {stage}")
        folder = base / str(congress)
        if date:
            folder = folder / date
        return folder

    @staticmethod
    def latest_congress_folder(stage: str, congress: int):
        """
        Get the symlink to the latest run for a given stage and congress.
        """
        base = getattr(DataPaths, stage.upper(), None)
        if base is None:
            raise ValueError(f"Unknown stage: {stage}")
        return base / str(congress) / "latest"

# Example usage:
# DataPaths.RAW_CONGRESSES / "114" / "2025-09-29"
# DataPaths.congress_folder('normalized', 114, '2025-09-29')
# DataPaths.latest_congress_folder('mentions', 115)
