"""
Functions to retrieve detailed profiles of congressional members.

This module wraps the functionality of the original
`congress_member_data_collector.py`.  It should query the Congress
API or another suitable endpoint using member bioguide IDs, normalise
the JSON responses into flat tables, and write the results to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .. import config


def fetch_congress_member_profiles(output_dir: Path) -> None:
    """Retrieve member profiles from the Congress API and save them to disk.

    This function wraps the original ``congress_member_data_collector.py``.
    It reads a CSV containing Bioguide IDs, fetches detailed member
    information via the Congress API, processes nested JSON into a flat
    table, and writes both the raw JSON and the cleaned CSV to
    ``output_dir``.

    Parameters
    ----------
    output_dir : Path
        Directory where input member reference files reside and where
        outputs should be stored.

    Notes
    -----
    - Requires ``config.CONGRESS_API_KEY``.  Without it the function
      logs a warning and returns.
    - The input file is expected to contain a column called
      ``bioGuideId``.  If multiple CSV files are present in
      ``output_dir``, the first with this column is used.
    - Outputs include ``congress_members_raw.json`` (pretty‑printed
      JSON responses) and ``congress_members.csv`` (flattened table).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = config.CONGRESS_API_KEY
    if not api_key:
        logging.warning(
            "CONGRESS_API_KEY is not set. Skipping congress member profile collection."
        )
        return
    import json
    import pandas as pd
    import requests
    from tqdm import tqdm  # type: ignore
    # Determine input CSV containing bioguide IDs
    input_path: Path | None = None
    for candidate in list(output_dir.glob("*.csv")):
        try:
            df = pd.read_csv(candidate)
        except Exception:
            continue
        if "bioGuideId" in df.columns:
            input_path = candidate
            break
    if input_path is None:
        logging.warning(
            "No CSV containing 'bioGuideId' found in %s; cannot fetch member profiles.",
            output_dir,
        )
        return
    # Load bioguide IDs
    refs_df = pd.read_csv(input_path)
    unique_ids = refs_df["bioGuideId"].dropna().unique().tolist()
    if not unique_ids:
        logging.warning("No bioguide IDs found in %s", input_path)
        return
    base_url = "https://api.congress.gov/v3/member"
    raw_responses: list[dict] = []
    # Fetch each member's data
    for bioguide_id in tqdm(unique_ids, desc="Fetching member profiles"):
        url = f"{base_url}/{bioguide_id}?format=json&api_key={api_key}"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                raw_responses.append(resp.json())
            else:
                logging.error(
                    "Failed to fetch member %s: status %s", bioguide_id, resp.status_code
                )
        except Exception as exc:  # noqa: BLE001
            logging.error("Exception fetching member %s: %s", bioguide_id, exc)
    if not raw_responses:
        logging.warning("No member data retrieved.")
        return
    # Save raw JSON responses
    raw_json_path = output_dir / "congress_members_raw.json"
    with raw_json_path.open("w", encoding="utf-8") as jf:
        json.dump(raw_responses, jf, indent=2)
    logging.info("Saved raw member responses to %s", raw_json_path)
    # Flatten JSON structure
    df = pd.json_normalize(raw_responses)
    # Extract fields similar to the original script
    if 'member.birthYear' in df.columns:
        df['birthYear'] = df['member.birthYear']
    if 'member.cosponsoredLegislation.count' in df.columns:
        df['cosponsoredLegislationCount'] = df['member.cosponsoredLegislation.count']
    if 'member.partyHistory' in df.columns:
        df['partyHistory'] = df['member.partyHistory'].apply(
            lambda x: x[0]['partyName'] if isinstance(x, list) and x else None
        )
    # Explode terms and normalise
    if 'member.terms' in df.columns:
        logging.info("Exploding and normalising terms data …")
        df = df.explode('member.terms').reset_index(drop=True)
        terms_df = pd.json_normalize(df['member.terms'])
        df = pd.concat([df, terms_df], axis=1).drop(columns=['member.terms'])
    # Filter by congress if present
    if 'congress' in df.columns:
        df = df[df['congress'].isin([114, 115])]
    # Rename identifiers
    if 'member.identifiers.bioguideId' in df.columns:
        df = df.rename(columns={'member.identifiers.bioguideId': 'bioGuideId'})
    # Save cleaned CSV
    out_csv = output_dir / "congress_members.csv"
    df.to_csv(out_csv, index=False)
    logging.info("Saved cleaned member data to %s", out_csv)
