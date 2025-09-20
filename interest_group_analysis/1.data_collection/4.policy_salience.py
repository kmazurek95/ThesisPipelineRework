"""
Functions to collect policy salience data from Google Trends.

This module corresponds to `policy_salience_pipeline.py` from the
original repository.  It should accept a list of policy topics,
download time series of search interest for each topic using the
Google Trends API (via the `pytrends` library), aggregate the data
by year and policy area, and save the results to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from .. import config


def fetch_policy_salience(
    output_dir: Path,
    topics: Iterable[str] | None = None,
    geo: str = "US",
    timeframe: str = "2005-01-01 2022-12-31",
) -> None:
    """Collect and save Google Trends data for a list of topics.

    Parameters
    ----------
    output_dir : Path
        Directory where trends and aggregated salience data should be
        written.
    topics : Iterable[str] or None, optional
        A collection of search terms representing policy areas.  If
        `None`, a default list of topics should be defined based on
        the research context.
    geo : str
        The geographic location code for Google Trends (e.g. "US").
    timeframe : str
        The date range for the trends query in the format
        "YYYY-MM-DD YYYY-MM-DD".

    Notes
    -----
    - `pytrends` does not require an API key but it does make HTTP
      requests to Google.  Be mindful of request limits and consider
      sleeping between calls.
    - After downloading trends data, aggregate it by year and by
      topic to create salience metrics.  For example, compute the
      average interest score per year for each topic.
    """
    from pytrends.request import TrendReq  # noqa: WPS433 (import inside function)

    output_dir.mkdir(parents=True, exist_ok=True)

    if topics is None:
        topics = [
            "health care",
            "education",
            "environment",
            "tax policy",
            "immigration",
        ]

    logging.info("Fetching Google Trends data for %d topics", len(topics))
    pytrends = TrendReq(hl="en-US", tz=360)

    all_data: list[pd.DataFrame] = []
    for topic in topics:
        logging.debug("Querying trend for '%s'", topic)
        try:
            pytrends.build_payload([topic], geo=geo, timeframe=timeframe)
            df = pytrends.interest_over_time()
            if df.empty:
                logging.warning("No trends data returned for topic '%s'", topic)
                continue
            df["topic"] = topic
            all_data.append(df.reset_index())
        except Exception as exc:  # noqa: BLE001
            logging.error("Error retrieving trends for topic '%s': %s", topic, exc)
            continue

    if not all_data:
        logging.warning("No trends data retrieved.  Check your internet connection or topics.")
        return

    combined = pd.concat(all_data, ignore_index=True)
    combined.to_csv(output_dir / "raw_policy_trends.csv", index=False)

    # Aggregate by year and topic
    combined["year"] = combined["date"].dt.year
    agg = (
        combined.groupby(["year", "topic"])[topic]
        .mean()
        .reset_index(name="avg_interest")
    )
    agg.to_csv(output_dir / "aggregated_policy_salience.csv", index=False)
