#!/usr/bin/env python3
"""
Bill linkage and metadata collection module.

This module provides tools to link Congressional Record mentions to 
referenced bills, extract bill metadata from GovInfo, and create a 
reliable linkage dataset between mentions and bill references.
"""

# =============================================================================
# # Bill Linkage and Metadata Collection
#
# This module provides functionality to:
#
# 1. Extract and normalize bill references from Congressional Record text
# 2. Build canonical bill identifiers that work with the GovInfo API
# 3. Fetch bill metadata and text with robust error handling and caching
# 4. Create a link table between mentions and referenced bills
#
# ## Usage
#
# ```python
# from pathlib import Path
# from interest_group_analysis.1.data_collection.6.bills_linkage import run_end_to_end
#
# # Full pipeline with existing bill references
# run_end_to_end(
#     output_dir=Path("data/bills"),
#     references_csv=Path("data/bills/references.csv"),
#     mentions_jsonl=Path("data/processed/mentions_with_speakers.jsonl"),
# )
#
# # Or just extract bill references from mentions
# from interest_group_analysis.1.data_collection.6.bills_linkage import link_mentions_to_bills
# 
# links = link_mentions_to_bills(
#     output_dir=Path("data/bills"),
#     mentions_path=Path("data/processed/mentions_with_speakers.jsonl")
# )
# ```
#
# ## Input/Output
#
# - **Inputs**: 
#   - Mentions JSONL file
#   - (Optional) Bill references CSV with columns: congress, type, number
# 
# - **Outputs**:
#   - `bill_metadata.parquet`: Structured bill metadata and text
#   - `bill_metadata.jsonl`: Raw JSON responses for inspection
#   - `mention_bill_links.parquet`: Link table connecting mentions to bills
#
# ## Authentication
#
# Requires a GovInfo API key, which can be set as `GOVINFO_API_KEY` in the
# config module or passed directly to the functions.
# =============================================================================

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

# ---- CONFIG ----
GOVINFO_API_KEY = None  # override or import from your config module
RATE_LIMIT_DELAY = 0.35  # ~3 req/sec
TIMEOUT = 30
BILL_REF_CSV_NAME = "references.csv"  # if you already have a curated bill list
MENTIONS_JSONL = "mentions_with_speakers.jsonl"  # or your mentions file path
OUTPUT_META = "bill_metadata.parquet"
OUTPUT_META_JSONL = "bill_metadata.jsonl"
OUTPUT_LINKS = "mention_bill_links.parquet"

# Order matters: we try these versions in sequence until one exists.
# Adjust to your corpus/version needs.
PREFERRED_VERSIONS = {
    "HR":  ["ih", "rh", "rfs", "rds", "eh", "enr", "pap", "fph"],
    "HRES": ["ih", "rh", "rfs", "rds", "eh", "enr"],
    "HJRES": ["ih", "rh", "eh", "enr"],
    "HCONRES": ["ih", "rh", "enr"],
    "S":   ["is", "rs", "rfs", "rds", "es", "enr", "pap", "fph"],
    "SRES": ["is", "rs", "es", "enr"],
    "SJRES": ["is", "rs", "es", "enr"],
    "SCONRES": ["is", "rs", "enr"],
}

BILL_TYPE_ALIASES = {
    # Normalized → accepted inputs
    "HR": {"H.R.", "HR", "H R", "H- R", "HOUSE BILL"},
    "S": {"S.", "S", "S "},
    "HRES": {"H. RES.", "H.RES.", "H RES", "HRES", "HOUSE RES"},
    "SRES": {"S. RES.", "S.RES.", "S RES", "SRES"},
    "HJRES": {"H. J. RES.", "H.J.RES.", "HJRES", "HOUSE JOINT RES"},
    "SJRES": {"S. J. RES.", "S.J.RES.", "SJRES"},
    "HCONRES": {"H. CON. RES.", "H.CON.RES.", "HCONRES", "HOUSE CONCURRENT RES"},
    "SCONRES": {"S. CON. RES.", "S.CON.RES.", "SCONRES"},
}

# Compiled once; matches canonical formats in debate text
BILL_CITATION_RE = re.compile(
    r"\b(?:(H\.?\s*R\.?|S\.?|H\.?\s*RES\.?|S\.?\s*RES\.?|H\.?\s*J\.?\s*RES\.?|S\.?\s*J\.?\s*RES\.?|H\.?\s*CON\.?\s*RES\.?|S\.?\s*CON\.?\s*RES\.?))\s*\.?\s*(\d{1,5})\b",
    flags=re.IGNORECASE,
)

@dataclass(frozen=True)
class CanonicalBillID:
    congress: int
    bill_type: str  # normalized e.g., HR, S, HRES ...
    number: int

    def key(self) -> str:
        # stable join key for your warehouse
        return f"{self.congress}-{self.bill_type}{self.number}"

    def package_id(self, version: str) -> str:
        # GovInfo package format e.g., BILLS-114hr1234ih
        return f"BILLS-{self.congress}{self.bill_type.lower()}{self.number}{version}"

def _requests_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",)
    )
    s.headers.update({"User-Agent": "ThesisRevamp/1.0"})
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def _normalize_bill_type(raw: str) -> Optional[str]:
    if not raw:
        return None
    r = raw.strip().upper().replace("-", " ").replace("_", " ")
    for norm, alts in BILL_TYPE_ALIASES.items():
        if r in alts or r == norm:
            return norm
    # fallbacks for plain tokens
    token = r.replace(".", "").replace(" ", "")
    mapping = {
        "HR": "HR", "HRES": "HRES", "HJRES": "HJRES", "HCONRES": "HCONRES",
        "S": "S", "SRES": "SRES", "SJRES": "SJRES", "SCONRES": "SCONRES",
    }
    return mapping.get(token)

def parse_bill_citations(text: str) -> List[Tuple[str, int]]:
    hits = []
    if not text:
        return hits
    for m in BILL_CITATION_RE.finditer(text):
        btype_raw, num = m.group(1), m.group(2)
        btype = _normalize_bill_type(btype_raw)
        if btype:
            hits.append((btype, int(num)))
    return hits

def _try_package(session: requests.Session, package_id: str, api_key: str) -> Optional[dict]:
    url = f"https://api.govinfo.gov/packages/{package_id}/summary?api_key={api_key}"
    r = session.get(url, timeout=TIMEOUT)
    if r.status_code == 200:
        return r.json()
    return None

def resolve_package(session: requests.Session, bill: CanonicalBillID, api_key: str) -> Tuple[Optional[str], Optional[dict]]:
    versions = PREFERRED_VERSIONS.get(bill.bill_type, [])
    for v in versions:
        pid = bill.package_id(v)
        data = _try_package(session, pid, api_key)
        if data:
            return pid, data
        time.sleep(RATE_LIMIT_DELAY)
    return None, None

def fetch_txt(session: requests.Session, summary_json: dict, api_key: str) -> Optional[str]:
    txt_link = summary_json.get("download", {}).get("txtLink")
    if not txt_link:
        return None
    r = session.get(f"{txt_link}?api_key={api_key}", timeout=TIMEOUT)
    if r.status_code == 200:
        return r.text
    return None

# ---------- PIPELINE STEPS ----------

def build_candidate_bills_from_references(refs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expect columns: congress, type, number (any casing).
    Returns a normalized DataFrame with columns: congress, bill_type, number, bill_key
    """
    need = {"congress", "type", "number"}
    assert need.issubset(set(refs_df.columns)), f"refs_df must have {need}"
    df = refs_df.copy()
    df["bill_type"] = df["type"].map(_normalize_bill_type)
    df["number"] = df["number"].astype(int)
    df["congress"] = df["congress"].astype(int)
    df = df.dropna(subset=["bill_type"])
    df["bill_key"] = df.apply(lambda r: CanonicalBillID(int(r["congress"]), r["bill_type"], int(r["number"])).key(), axis=1)
    return df[["congress", "bill_type", "number", "bill_key"]].drop_duplicates()

def build_candidate_bills_from_mentions(mentions: Iterable[dict]) -> pd.DataFrame:
    """
    Read your mentions JSONL (one dict per line) and recover bill references from:
      - structured fields if present (e.g., 'billType','billNumber','congress')
      - raw text citation fallback (regex)
    Returns rows with: mention_id, congress, bill_type, number, link_source ('structured'|'regex'), confidence
    """
    rows = []
    for m in mentions:
        mention_id = m.get("uuid_paragraph") or m.get("mention_id") or m.get("uuid")  # adapt to your schema
        congress = m.get("congress") or m.get("congressNumber")  # adapt if you store congress on mention
        # structured first
        stype, snum = m.get("billType"), m.get("billNumber")
        if stype and snum and congress:
            btype = _normalize_bill_type(str(stype))
            try:
                num = int(snum)
            except Exception:
                num = None
            if btype and num:
                rows.append({
                    "mention_id": mention_id, "congress": int(congress),
                    "bill_type": btype, "number": num,
                    "link_source": "structured", "confidence": 1.0
                })
        # regex fallback from text/window
        raw_text = m.get("context") or m.get("p1_original") or m.get("paragraph") or m.get("text") or ""
        if raw_text:
            hits = parse_bill_citations(raw_text)
            for (btype, num) in hits:
                # If congress is missing, you can infer from granule date (optional)
                # Here we skip inference to keep it conservative:
                if congress:
                    rows.append({
                        "mention_id": mention_id, "congress": int(congress),
                        "bill_type": btype, "number": int(num),
                        "link_source": "regex", "confidence": 0.7
                    })
    if not rows:
        return pd.DataFrame(columns=["mention_id","congress","bill_type","number","link_source","confidence"])
    df = pd.DataFrame(rows).drop_duplicates()
    df = df.dropna(subset=["congress","bill_type","number"])
    df["bill_key"] = df.apply(lambda r: CanonicalBillID(int(r["congress"]), r["bill_type"], int(r["number"])).key(), axis=1)
    return df

def fetch_bill_metadata(output_dir: Path, references_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch (or refresh) bill metadata for all candidate bills discovered.
    Returns (bill_meta_df, links_df) where links_df is empty here (used when building from mentions).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = GOVINFO_API_KEY
    try:
        from .. import config as _cfg
        api_key = api_key or getattr(_cfg, "GOVINFO_API_KEY", None)
    except Exception:
        pass

    if not api_key:
        logging.warning("GOVINFO_API_KEY is not set. Skipping.")
        return pd.DataFrame(), pd.DataFrame()

    # Locate references.csv if not provided
    if references_df is None:
        refs_path = None
        for candidate in [output_dir / BILL_REF_CSV_NAME] + list(output_dir.glob("*.csv")):
            try:
                test = pd.read_csv(candidate)
                if {"type", "congress", "number"}.issubset(test.columns):
                    refs_path = candidate
                    break
            except Exception:
                continue
        if refs_path is None:
            logging.warning("No bill reference CSV found in %s", output_dir)
            return pd.DataFrame(), pd.DataFrame()
        references_df = pd.read_csv(refs_path)

    candidates = build_candidate_bills_from_references(references_df)
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Deduplicate by bill_key
    seen = set()
    records: List[dict] = []

    session = _requests_session()
    for _, r in candidates.iterrows():
        bill = CanonicalBillID(int(r["congress"]), r["bill_type"], int(r["number"]))
        if bill.key() in seen:
            continue
        pid, summary = resolve_package(session, bill, api_key)
        time.sleep(RATE_LIMIT_DELAY)
        if not summary:
            logging.info("No package found for %s", bill.key())
            seen.add(bill.key())
            continue
        txt = fetch_txt(session, summary, api_key)
        time.sleep(RATE_LIMIT_DELAY)

        rec = {
            "bill_key": bill.key(),
            "congress": bill.congress,
            "bill_type": bill.bill_type,
            "number": bill.number,
            "packageId": pid,
            "summary": summary,
            "billText": txt,
            # common pull-ups to flatten the JSON a bit:
            "title": summary.get("title"),
            "docClass": summary.get("docClass"),
            "dateIssued": summary.get("dateIssued"),
            "download_txt": summary.get("download", {}).get("txtLink"),
            "download_pdf": summary.get("download", {}).get("pdfLink"),
        }
        records.append(rec)
        seen.add(bill.key())

    if not records:
        logging.warning("No bill metadata retrieved.")
        return pd.DataFrame(), pd.DataFrame()

    meta_df = pd.DataFrame(records)
    meta_df.to_parquet(output_dir / OUTPUT_META, index=False)
    meta_df.to_json(output_dir / OUTPUT_META_JSONL, lines=True, orient="records")
    logging.info("Saved %s and %s", OUTPUT_META, OUTPUT_META_JSONL)
    return meta_df, pd.DataFrame()

def link_mentions_to_bills(output_dir: Path,
                           mentions_path: Path,
                           meta_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Build a mention↔bill link table and (optionally) fetch missing bill metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # read mentions JSONL
    mentions = []
    with mentions_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                mentions.append(json.loads(line))
            except Exception:
                continue

    links = build_candidate_bills_from_mentions(mentions)
    if links.empty:
        logging.info("No bill links discovered from mentions.")
        links.to_parquet(output_dir / OUTPUT_LINKS, index=False)
        return links

    # Ensure we have metadata for all links
    needed = links[["congress", "bill_type", "number"]].drop_duplicates()
    refs_df = needed.rename(columns={"bill_type": "type"})
    if meta_df is None or meta_df.empty:
        meta_df, _ = fetch_bill_metadata(output_dir, references_df=refs_df)

    # join availability flag
    have = set(meta_df["bill_key"]) if meta_df is not None and "bill_key" in meta_df.columns else set()
    links["has_metadata"] = links["bill_key"].apply(lambda k: k in have)

    links.to_parquet(output_dir / OUTPUT_LINKS, index=False)
    logging.info("Saved mention↔bill links to %s", OUTPUT_LINKS)
    return links

# ---- Convenience runner (optional) ----
def run_end_to_end(output_dir: Path,
                   mentions_jsonl: Optional[Path] = None,
                   references_csv: Optional[Path] = None) -> None:
    """
    Typical usage:
      run_end_to_end(Path("data/bills/"), Path("data/processed/mentions_with_speakers.jsonl"),
                     Path("data/bills/references.csv"))
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    refs_df = pd.read_csv(references_csv) if references_csv and references_csv.exists() else None
    meta_df, _ = fetch_bill_metadata(output_dir, references_df=refs_df)

    if mentions_jsonl and mentions_jsonl.exists():
        link_mentions_to_bills(output_dir, mentions_jsonl, meta_df=meta_df)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract bill references and fetch metadata")
    parser.add_argument("--output-dir", type=Path, default=Path("data/bills"),
                       help="Directory to save bill metadata and links")
    parser.add_argument("--mentions", type=Path, default=None,
                       help="Path to mentions JSONL file")
    parser.add_argument("--references", type=Path, default=None,
                       help="Path to bill references CSV (columns: congress, type, number)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="GovInfo API key")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if args.api_key:
        global GOVINFO_API_KEY
        GOVINFO_API_KEY = args.api_key
    
    # Run the pipeline
    run_end_to_end(
        output_dir=args.output_dir,
        mentions_jsonl=args.mentions,
        references_csv=args.references
    )