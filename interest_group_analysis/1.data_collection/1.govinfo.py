"""
Functions to fetch legislative transcripts from the GovInfo API.

This module downloads transcripts and associated metadata for multiple sessions of Congress.
It now extracts and saves BOTH readability-lxml and BeautifulSoup text fields for each document.
"""

from __future__ import annotations

# Long descriptive note moved here from an accidental inline insertion.
# Keep this at module top so it's easy to find and edit.
MODULE_NOTE = (
    """
    Download and process legislative transcripts from the GovInfo API, saving extracted text and metadata to disk.

    This function retrieves Congressional Record transcripts for specified congress sessions, downloads and parses the transcript text using both readability-lxml and BeautifulSoup extraction methods, and saves the results as CSV and JSON files. It supports resuming from previous progress, parallel downloading, and error logging.

    Args:
        output_dir (Path): Directory where output files and progress will be saved.
        congresses (list[int] | None, optional): List of congress numbers to process. Defaults to [114, 115] if None.
        start_date (str, optional): ISO8601 date string to start collecting from. Defaults to "2015-01-01T00:00:00Z".
        page_size (int, optional): Number of packages to fetch per API request. Defaults to 1000.
        workers (int, optional): Number of threads for parallel granule downloads. Defaults to 8.
        max_packages_per_congress (int | None, optional): Maximum number of packages to process per congress. If None, no limit.

    Returns:
        None

    Side Effects:
        - Creates output_dir and subdirectories as needed.
        - Writes CSV and JSON files with extracted transcript data.
        - Maintains a progress file to support resuming.
        - Logs errors to an error log file.
        - Saves raw API responses for debugging.

    Notes:
        - Requires GOVINFO_API_KEY to be set in the config module.
        - Uses both readability-lxml and BeautifulSoup for text extraction, saving both results.
        - Handles API pagination and retries on network errors.
    """
)

import gzip
import logging
from pathlib import Path
from typing import Any, Iterable
import random
import time
import os
import json

import requests

from .. import config

# No hard-coded package limit: pass max_packages_per_congress=None to collect everything


def fetch_legislative_transcripts(
    output_dir: Path,
    congresses: list[int] | None = None,
    start_date: str = "2015-01-06T00:00:00Z",
    end_date: str | None = None,
    page_size: int = 1000,
    workers: int = 8,
    max_packages_per_congress: int | None = None,
    initial_offset_mark: str | None = "*",
    dry_run: bool = False,
    save_raw: bool = True,
) -> None:
    """Download Congressional Record transcripts (CREC) and metadata.

    Key features:
      - Uses /published/{start}/{end}?collection=CREC&congress=... with nextPage pagination.
      - Optional dry_run mode returns a tiny synthetic dataset (no network) for CI/testing.
      - Parallel granule fetching with text extraction (readability + bs4 fallback).
      - Progress resume via progress.json (per-congress package index).
      - Optional raw JSON response saving (gzip) for audit/debug.
      - Rate limit handling: backs off on HTTP 429 and certain 5xx responses with jitter.
      - Integrity summary after completion (counts packages / granules / rows).

    Args:
        output_dir: Destination directory for outputs.
        congresses: List of congress numbers; defaults to [114, 115] if None.
        start_date: ISO8601 start (YYYY-MM-DD[THH:MM:SSZ]). Only date part is used for range path.
        end_date: Optional ISO8601 end. If None, inferred from congress span; else truncated to date.
        page_size: Package page size.
        workers: Thread pool size for granule fetches.
        max_packages_per_congress: Positive int limit; None or <=0 means no limit.
        initial_offset_mark: Starting offsetMark (“*” for first page). Ignored after first request.
        dry_run: If True, skip network calls and emit deterministic mock data.
        save_raw: If True save raw JSON responses (can be large).
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    api_key: str | None = config.GOVINFO_API_KEY
    if not api_key:
        logging.warning(
            "GOVINFO_API_KEY is not set. Skipping transcript collection."
        )
        return
    if congresses is None:
        congresses = [114, 115]

    congress_year_spans = {114:(2015,2017),115:(2017,2019),116:(2019,2021),117:(2021,2023),118:(2023,2025)}
    try:
        start_year = int(start_date[:4])
        for c in congresses:
            span = congress_year_spans.get(c)
            if span and not (span[0] <= start_year < span[1]):
                logging.warning(
                    "start_date %s (year %s) outside typical span %s-%s for congress %s",
                    start_date, start_year, span[0], span[1]-1, c
                )
    except Exception:
        logging.debug("Could not parse start_date %s", start_date)

    # Validate that provided date span does not obviously cross multiple congress spans if list is single
    if end_date and congresses and len(congresses) == 1:
        c = congresses[0]
        span = congress_year_spans.get(c)
        try:
            s_year = int(start_date[:4])
            e_year = int(end_date[:4])
            if span and not (span[0] <= s_year < span[1]) or not (span and span[0] <= e_year <= span[1]):
                logging.warning(
                    "Date range %s -> %s extends outside typical congress %s span %s-%s",
                    start_date, end_date, c, span[0], span[1]-1 if span else "?",
                )
        except Exception:
            pass
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from bs4 import BeautifulSoup  # type: ignore
    from retrying import retry  # type: ignore
    import pandas as pd
    from tqdm import tqdm  # type: ignore

    progress_path = output_dir / "progress.json"
    # Load progress if available
    if progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as pf:
            progress = json.load(pf)
    else:
        progress = {}

    if dry_run:
        logging.info("Dry run enabled: skipping API preflight.")
    else:
        # QUICK: validate key before heavy loop
        try:
            test_resp = requests.get(
                "https://api.govinfo.gov/collections",
                params={"api_key": api_key},
                timeout=10,
            )
            if test_resp.status_code != 200:
                logging.error(
                    "Preflight /collections check failed (status=%s). Body: %s",
                    test_resp.status_code,
                    test_resp.text[:300],
                )
                return
            if "collections" not in test_resp.text:
                logging.error(
                    "Preflight succeeded but unexpected body (no 'collections' key). Body: %s",
                    test_resp.text[:300],
                )
                return
            logging.info("API key preflight OK.")
        except Exception as pre_exc:
            logging.exception("Preflight request error: %s", pre_exc)
            return

    from functools import wraps
    # Rate-limit aware request with limited retries & jitter.
    def make_request(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if dry_run:
            return {}
        attempt = 0
        backoff = 0.5
        while True:
            attempt += 1
            start_t = time.time()
            logging.info("GET %s params=%s", url, params)
            resp = requests.get(url, params=params, timeout=30)
            elapsed = (time.time() - start_t) * 1000
            status = resp.status_code
            if status == 429 or 500 <= status < 600:
                if attempt <= 5:
                    sleep_for = backoff + random.uniform(0, 0.5)
                    logging.warning(
                        "HTTP %s from %s (%.0f ms); backing off %.2fs (attempt %s)",
                        status,
                        url,
                        elapsed,
                        sleep_for,
                        attempt,
                    )
                    time.sleep(sleep_for)
                    backoff *= 2
                    continue
            if status >= 400:
                logging.warning(
                    "HTTP %s from %s (%.0f ms): %s",
                    status,
                    url,
                    elapsed,
                    resp.text[:300],
                )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "message" in data and "packages" not in data:
                raise RuntimeError(f"API error payload: {data.get('message')}")
            if save_raw:
                try:
                    save_raw_response(data, output_dir, f"response_{url.split('/')[-1]}")
                except Exception as _save_exc:  # noqa: BLE001
                    logging.debug("Raw save failed for %s: %s", url, _save_exc)
            return data

    def extract_text_readability(html: str) -> str:
        """Extract main body text using readability-lxml. Returns empty string on failure."""
        try:
            from readability import Document  # type: ignore
            from bs4 import BeautifulSoup as _BS

            doc = Document(html)
            summary_html = doc.summary() or ""
            text = _BS(summary_html, "lxml").get_text(separator=" ", strip=True)
            return " ".join(text.split()) if text else ""
        except Exception:
            return ""

    def extract_text_bs(html: str) -> str:
        """Extract full-page text using BeautifulSoup with basic boilerplate removal."""
        try:
            try:
                soup = BeautifulSoup(html, "lxml")
            except Exception:
                soup = BeautifulSoup(html, "html.parser")

            for selector in ("script", "style", "header", "footer", "nav", "aside"):
                for el in soup.select(selector):
                    el.decompose()

            text = soup.get_text(separator=" ", strip=True)
            return " ".join(text.split()) if text else ""
        except Exception:
            return ""

    def save_raw_response(response: dict[str, Any], output_dir: Path, filename: str) -> None:
        """Save raw JSON response to a compressed file."""
        if not save_raw:
            return
        import re
        raw_dir = output_dir / "raw_responses"
        raw_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r'[^A-Za-z0-9._-]', '_', filename)[:200]
        file_path = raw_dir / f"{safe}.json.gz"
        try:
            with gzip.open(file_path, "wt", encoding="utf-8") as f:
                json.dump(response, f)
            logging.info("Saved raw response to %s", file_path)
        except Exception as exc:
            logging.exception("Could not save raw response to %s: %s", file_path, exc)

    def log_error_details(error: str, output_dir: Path) -> None:
        """Log error details to a separate file; tolerate malformed files."""
        error_log_path = output_dir / "error_log.json"
        try:
            if error_log_path.exists():
                with error_log_path.open("r", encoding="utf-8") as ef:
                    try:
                        error_log = json.load(ef)
                        if not isinstance(error_log, list):
                            error_log = []
                    except json.JSONDecodeError:
                        # Corrupted file — back it up and start a fresh list
                        try:
                            backup = error_log_path.with_suffix(".corrupt.json")
                            error_log_path.replace(backup)
                            logging.warning("Corrupt error_log.json moved to %s", backup)
                        except Exception:
                            logging.exception("Unable to back up corrupt error_log.json")
                        error_log = []
            else:
                error_log = []
        except Exception:
            logging.exception("Unexpected error reading error_log.json; starting fresh")
            error_log = []

        error_log.append({"error": error, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")})
        try:
            with error_log_path.open("w", encoding="utf-8") as ef:
                json.dump(error_log, ef, indent=2)
            logging.info("Logged error to %s", error_log_path)
        except Exception:
            logging.exception("Could not write to %s", error_log_path)

    def fetch_granule_details(link: str) -> dict[str, Any] | None:
        """Retrieve metadata and plain text for a single granule, saving both text extraction results."""
        try:
            detail_json = make_request(link, params={"api_key": api_key})
            save_raw_response(detail_json, output_dir, f"granule_{link.split('/')[-1]}")
            download_link = detail_json.get("download", {}).get("txtLink")
            if download_link:
                dl_params = None
                if "api.govinfo.gov" in download_link and "api_key=" not in download_link:
                    dl_params = {"api_key": api_key}
                txt_resp = requests.get(download_link, params=dl_params, timeout=30)
                if txt_resp.status_code == 200:
                    ctype = (txt_resp.headers.get("Content-Type") or "").lower()
                    body = txt_resp.text or ""
                    is_html_link = ("/htm" in download_link.lower()) or download_link.lower().endswith(".htm")
                    is_html_ct = any(k in ctype for k in ("text/html", "application/xhtml+xml", "html"))
                    is_json = "application/json" in ctype or body.strip().startswith("{")

                    if (is_html_ct or is_html_link) and not is_json:
                        # Save BOTH extraction methods as separate fields
                        detail_json["text_readability"] = extract_text_readability(body)
                        detail_json["text_bs4"] = extract_text_bs(body)
                        # Optional: for backward compatibility, you may keep "parsed_text" as preferred
                        detail_json["parsed_text"] = detail_json["text_readability"] or detail_json["text_bs4"]
                    elif is_json:
                        logging.debug("txtLink returned JSON/error for %s: %s", download_link, body[:200])
                    else:
                        logging.debug("txtLink not HTML for %s (Content-Type=%s); skipping parsed_text", download_link, ctype)
            return detail_json
        except Exception as exc:  # noqa: BLE001
            error_message = f"Error fetching granule details for {link}: {exc}"
            logging.error(error_message)
            log_error_details(error_message, output_dir)
            return None

    def save_progress(progress_obj: dict, path: Path) -> None:
        """Atomically save progress to disk and fsync to ensure durability."""
        tmp_path = path.with_suffix(".tmp")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8") as pf:
            json.dump(progress_obj, pf, indent=2)
            pf.flush()
            os.fsync(pf.fileno())
        tmp_path.replace(path)
        logging.info("Saved progress to %s", path)

    # Dry run short-circuit producing deterministic mock artifacts.
    if dry_run:
        mock = [{
            "granuleId": "TEST-GRANULE-1",
            "text_readability": "Sample readability text.",
            "text_bs4": "Sample bs4 text.",
            "parsed_text": "Sample readability text.",
        }]
        df = pd.DataFrame(mock)
        df.to_csv(output_dir / "mock_transcripts.csv", index=False)
        logging.info("Dry run complete: wrote mock_transcripts.csv")
        return

    all_dataframes: list[pd.DataFrame] = []
    for congress in congresses:
        start_index = progress.get(str(congress), 0)
        # Interpret max_packages_per_congress: None or <=0 => no limit, otherwise integer limit
        if max_packages_per_congress is None:
            effective_max = None
        else:
            try:
                effective_max = int(max_packages_per_congress)
            except Exception:
                logging.warning("Invalid max_packages_per_congress=%r; treating as no limit", max_packages_per_congress)
                effective_max = None

        if effective_max is not None and effective_max <= 0:
            # Treat non-positive values as 'no limit' for convenience.
            logging.info("max_packages_per_congress=%s interpreted as no limit", effective_max)
            effective_max = None

        no_limit = effective_max is None

        if (not no_limit) and start_index >= effective_max:
            logging.info("Already reached max_packages (%s) for congress %s; skipping", effective_max, congress)
            continue
        # ------------------------------------------------------------------
        # NEW: Use the /published/{start}/{end} endpoint which proved stable
        # for CREC with congress + offsetMark pagination.
        # ------------------------------------------------------------------
        start_day = start_date[:10]
        if end_date:
            end_day = end_date[:10]
        else:
            # Derive an end_day from congress span if available, else same day.
            span = congress_year_spans.get(congress)
            if span:
                # Convention: Jan 03 of the start year of the NEXT congress boundary
                # (example: congress 114 spans 2015-2017, choose 2017-01-03)
                end_day = f"{span[1]}-01-03"
            else:
                end_day = start_day
        published_url = f"https://api.govinfo.gov/published/{start_day}/{end_day}"
        params_local: dict[str, Any] | None = {
            "api_key": api_key,
            "collection": "CREC",
            "congress": congress,
            "pageSize": page_size,
        }
        if initial_offset_mark:
            params_local["offsetMark"] = initial_offset_mark
        next_page = published_url
        packages: list[dict[str, Any]] = []
        logging.info(
            "Fetching packages for congress %s via /published window %s -> %s (initial offsetMark=%s)",
            congress,
            start_day,
            end_day,
            initial_offset_mark,
        )

        while next_page:
            try:
                resp_json = make_request(next_page, params=params_local)
                # After first call, rely on fully-qualified nextPage URLs (no params)
                params_local = None
                new_pkgs = resp_json.get("packages", [])
                if new_pkgs:
                    packages.extend(new_pkgs)
                else:
                    logging.debug("No packages in current page for congress %s", congress)
                if not no_limit and len(packages) >= (effective_max - start_index):
                    logging.info("Hit requested max_packages limit while fetching packages (have %s)", len(packages))
                    break
                next_page = resp_json.get("nextPage")
            except Exception as exc:  # noqa: BLE001
                logging.error("Error fetching packages page for congress %s: %s", congress, exc)
                break

        if not packages:
            logging.warning("No packages returned for congress %s", congress)
            continue
        for idx, package in enumerate(packages[start_index:], start=start_index):
            if (not no_limit) and idx >= effective_max:
                logging.info("Reached max_packages (%s) for congress %s", effective_max, congress)
                break
            try:
                logging.info(
                    "Processing package %s/%s for congress %s", idx + 1, len(packages), congress
                )
                package_data = make_request(package["packageLink"], params={"api_key": api_key})
                save_raw_response(package_data, output_dir, f"package_{congress}_{idx}")
                granules_link = package_data.get("granulesLink")
                if not granules_link:
                    logging.debug("No granules found for package %s", package.get("packageLink"))
                    continue
                granule_links: list[str] = []
                next_gpage: str | None = granules_link
                params_local = {"api_key": api_key, "pageSize": 100}
                while next_gpage:
                    try:
                        use_params = None
                        if params_local is not None:
                            use_params = params_local
                        elif "api_key=" not in (next_gpage or ""):
                            use_params = {"api_key": api_key}

                        granules_json = make_request(next_gpage, params=use_params)
                        params_local = None
                        granule_links.extend(
                            [g["granuleLink"] for g in granules_json.get("granules", [])]
                        )
                        next_gpage = granules_json.get("nextPage")
                    except Exception as exc:  # noqa: BLE001
                        logging.error("Error fetching granules: %s", exc)
                        break
                if not granule_links:
                    continue
                granule_results: list[dict[str, Any]] = []
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(fetch_granule_details, link): link for link in granule_links}
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Granules", leave=False):
                        result = future.result()
                        if result:
                            granule_results.append(result)
                if granule_results:
                    df = pd.DataFrame(granule_results)
                    # Normalize whitespace in text columns
                    for col in ["text_readability", "text_bs4", "parsed_text"]:
                        if col in df.columns:
                            df[col] = (
                                df[col]
                                .astype(str)
                                .str.replace(r"\s+", " ", regex=True)
                                .str.strip()
                            )
                    out_csv = output_dir / f"package_{congress}_{idx}.csv"
                    df.to_csv(out_csv, index=False)
                    all_dataframes.append(df)
                progress[str(congress)] = idx + 1
                save_progress(progress, progress_path)
            except Exception as exc:  # noqa: BLE001
                error_message = f"Error processing package {idx + 1} for congress {congress}: {exc}"
                logging.error(error_message)
                log_error_details(error_message, output_dir)
                continue
    if all_dataframes:
        combined = pd.concat(all_dataframes, ignore_index=True)
        for col in ["text_readability", "text_bs4", "parsed_text"]:
            if col in combined.columns:
                combined[col] = (
                    combined[col]
                    .astype(str)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                )
        combined_csv = output_dir / "legislative_transcripts.csv"
        combined_json = output_dir / "legislative_transcripts.json"
        combined.to_csv(combined_csv, index=False)
        combined.to_json(combined_json, orient="records", lines=True)
        logging.info("Saved combined transcripts to %s", combined_csv)
        # Integrity summary
        pkg_files = list(output_dir.glob("package_*_*.csv"))
        logging.info(
            "Integrity summary: %s package CSV files, %s combined rows.",
            len(pkg_files),
            len(combined),
        )
