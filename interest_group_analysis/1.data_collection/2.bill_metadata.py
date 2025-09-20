"""
Functions to collect metadata for congressional bills.

This module corresponds to the original script `fetch_bill_data.py`.
It should build request URLs for each bill, fetch the bill text and
metadata from the GovInfo API, and write the results to disk.  At
present this module provides a function stub to be implemented.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .. import config


def fetch_bill_metadata(output_dir: Path) -> None:
    """Download bill metadata and texts from the GovInfo API.

    This function reads a CSV file of bill references (containing at
    minimum ``type``, ``congress`` and ``number`` columns), constructs
    package IDs and API links for each bill, fetches summary data and
    bill texts, and writes the results to ``output_dir``.  If no
    suitable reference file is found in ``output_dir``, the function
    logs a warning and exits.

    Parameters
    ----------
    output_dir : Path
        Directory where input reference files are stored and where
        outputs should be written.  This directory will be created if
        necessary.

    Notes
    -----
    - Requires a valid GovInfo API key in ``config.GOVINFO_API_KEY``.
    - The reference file is expected to be named ``references.csv`` or to
      contain columns ``type``, ``congress`` and ``number``.  You can
      place the reference file in ``output_dir`` before running this
      function.
    - Outputs include ``bill_metadata.csv`` and ``bill_metadata.json``
      containing the fetched metadata, and ``bill_references_augmented.csv``
      which adds ``packageId`` and ``link`` columns to the original
      references.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = config.GOVINFO_API_KEY
    if not api_key:
        logging.warning(
            "GOVINFO_API_KEY is not set. Skipping bill metadata collection."
        )
        return
    import pandas as pd
    import requests
    from urllib.parse import urlsplit
    # Locate reference CSV
    references_path: Path | None = None
    for candidate in [output_dir / "references.csv"] + list(output_dir.glob("*.csv")):
        try:
            df = pd.read_csv(candidate)
        except Exception:
            continue
        # Check for required columns
        if {"type", "congress", "number"}.issubset(df.columns):
            references_path = candidate
            break
    if references_path is None:
        logging.warning(
            "No reference CSV with columns 'type', 'congress', 'number' found in %s",
            output_dir,
        )
        return
    # Load references
    refs_df = pd.read_csv(references_path)
    # Construct packageId and link
    def construct(row: pd.Series) -> tuple[str, str]:
        bill_version = "is" if row["type"] in ["S", "SRES"] else "ih"
        package_id = f"BILLS-{row['congress']}{row['type'].lower()}{row['number']}{bill_version}"
        api_link = f"https://api.govinfo.gov/packages/{package_id}/summary"
        return package_id, api_link
    refs_df[["packageId", "link"]] = refs_df.apply(construct, axis=1, result_type="expand")
    # Save augmented references
    augmented_path = output_dir / "bill_references_augmented.csv"
    refs_df.to_csv(augmented_path, index=False)
    logging.info("Saved augmented reference file to %s", augmented_path)
    # Deduplicate links
    unique_links = refs_df["link"].dropna().unique().tolist()
    # Fetch metadata
    responses: list[dict] = []
    logging.info("Fetching metadata for %s unique bills", len(unique_links))
    for link in unique_links:
        try:
            url = f"{link}?api_key={api_key}"
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                logging.error("Request failed for %s with status %s", link, resp.status_code)
                continue
            item = resp.json()
            # Fetch bill text if available
            txt_link = item.get("download", {}).get("txtLink")
            if txt_link:
                txt_resp = requests.get(f"{txt_link}?api_key={api_key}", timeout=30)
                if txt_resp.status_code == 200:
                    item["billText"] = txt_resp.text
            responses.append(item)
        except Exception as exc:  # noqa: BLE001
            logging.error("Error fetching bill data for %s: %s", link, exc)
    if not responses:
        logging.warning("No bill metadata retrieved.")
        return
    # Convert responses to DataFrame and save
    meta_df = pd.json_normalize(responses)
    meta_csv = output_dir / "bill_metadata.csv"
    meta_json = output_dir / "bill_metadata.json"
    meta_df.to_csv(meta_csv, index=False)
    meta_df.to_json(meta_json, orient="records", lines=True)
    logging.info("Saved bill metadata to %s", meta_csv)
