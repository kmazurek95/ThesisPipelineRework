r"""
Speaker Attribution CLI — How to run

This script attaches speakers to extracted mentions from Congressional Record text.
It uses normalized granule text and optional member metadata to attribute the likely
speaker for each mention (by character offset).

Prerequisites
- Python 3.9+ (use the repo's .venv if available)
- Packages: pandas, tqdm, psutil (installed via requirements.txt)

Quick setup (PowerShell)
        # From repo root
        python -m venv .venv
        .\.venv\Scripts\Activate.ps1
        python -m pip install --upgrade pip
        pip install -r requirements.txt

Inputs
- --mentions-jsonl: JSONL with at least granuleId and mention_char_start
- --normalized-dir: directory containing by_package/**/granules_core.csv
- --members-csv (optional): member metadata (first_name,last_name,bioguide_id,chamber,party,...)

Basic run
        $PY = ".\.venv\Scripts\python.exe"
        $mentions = "data\processed\mentions_114_run2\mentions.jsonl"
        $norm     = "data\normalized\normalized_114_run2"
        $out      = "data\processed\mentions_and_speaker_114"

        $PY .\interest_group_analysis\2.data_processing\3.attach_speakers.py `
            --mentions-jsonl $mentions `
            --normalized-dir $norm `
            --out-dir $out `
            --save-csv `
            --qa-jsonl

Faster run (parallel + resume)
        $PY .\interest_group_analysis\2.data_processing\3.attach_speakers.py `
            --mentions-jsonl $mentions `
            --normalized-dir $norm `
            --out-dir $out `
            --save-csv `
            --qa-jsonl `
            --parallel `
            --workers 6 `
            --resume

Useful flags
- --only-granules-with-members: keep only granules with member metadata
- --drop-unknown-speaker: exclude unresolved attributions

Outputs (in --out-dir)
- mentions_with_speakers.jsonl (always)
- mentions_with_speakers.csv (if --save-csv)
- speaker_qc.jsonl (if --qa-jsonl)
- processed_granules.jsonl (resume manifest when --resume)

Monitoring & performance
- INFO logs include timing and memory snapshots (psutil)
- Progress bars show granule attribution progress and ETA
- Reduce memory by using --parallel with reasonable --workers or lowering batch sizes
"""

from __future__ import annotations
import re
def load_granule_members(normalized_dir: Path) -> dict[str, list[dict]]:
    """
    Return: { granuleId: [ {first_name, last_name, bioguide_id, chamber, party}, ... ] }
    Searches for all granule_members.csv files under normalized_dir/by_package/*/
    """
    by_pkg = normalized_dir / "by_package"
    if not by_pkg.exists():
        LOGGER.warning("by_package directory not found in %s", normalized_dir)
        return {}
    member_files = list(by_pkg.glob("**/granule_members.csv"))
    if not member_files:
        LOGGER.warning("No granule_members.csv found under %s/by_package/", normalized_dir)
        return {}
    LOGGER.info("Found %d granule_members.csv files to process", len(member_files))
    idx: dict[str, list[dict]] = {}
    for file_path in tqdm(member_files, desc="Loading granule members"):
        try:
            for chunk in pd.read_csv(file_path, dtype=str, keep_default_na=False, chunksize=10000):
                chunk = chunk.fillna("")
                for _, r in chunk.iterrows():
                    gid = str(r.get("granuleId", "")).strip()
                    if not gid:
                        continue
                    if r.get("role") != "SPEAKING":
                        continue
                    rec = {
                        "first_name": str(r.get("first_name", "")).strip(),
                        "last_name": str(r.get("name__authority-lnf", "")).split(",")[0].strip() if r.get("name__authority-lnf") else "",
                        "bioguide_id": str(r.get("bioGuideId", "")).strip() or None,
                        "chamber": str(r.get("chamber", "")).strip().upper() or None,
                        "party": str(r.get("party", "")).strip() or None,
                    }
                    if r.get("name__parsed"):
                        parsed = str(r.get("name__parsed")).strip()
                        if parsed:
                            parts = parsed.split()
                            if len(parts) > 1 and parts[0].lower() in ["mr.", "ms.", "mrs.", "dr.", "rep.", "sen."]:
                                rec["first_name"] = " ".join(parts[1:]).strip()
                    if rec["bioguide_id"] or (rec["last_name"] and rec["chamber"]):
                        idx.setdefault(gid, []).append(rec)
        except Exception as e:
            LOGGER.warning("Failed reading %s: %s", file_path, e)
    LOGGER.info("Loaded speaker data for %d granules with %d total members", len(idx), sum(len(members) for members in idx.values()))
    return idx

def granule_chamber_from_id(granule_id: str) -> Optional[str]:
    """Extract chamber information from granuleId"""
    gid = (granule_id or "").upper()
    if "PGH" in gid: return "H"
    if "PGS" in gid: return "S"
    if "PGE" in gid: return "E"
    return None

def refine_spans_with_members(spans, gid, members_by_gid):
    """
    Enhance speaker spans using granule member information
    """
    members = members_by_gid.get(gid, [])
    if not members:
        return spans
    g_chamber = granule_chamber_from_id(gid)
    for span in spans:
        if span.canonical_name and span.bioguide_id:
            continue
        m = re.search(r"\b([A-Z][A-Z\-\']+)\b", span.raw_label.upper())
        if not m:
            continue
        last_name = m.group(1)
        candidates = [m for m in members if m.get("last_name", "").upper() == last_name]
        if len(candidates) == 1:
            c = candidates[0]
            span.canonical_name = f"{c.get('first_name', '').title()} {c.get('last_name', '').title()}".strip()
            span.bioguide_id = c.get("bioguide_id")
            continue
        if len(candidates) > 1 and g_chamber:
            chamber_matches = [c for c in candidates if c.get("chamber") == g_chamber]
            if len(chamber_matches) == 1:
                c = chamber_matches[0]
                span.canonical_name = f"{c.get('first_name', '').title()} {c.get('last_name', '').title()}".strip()
                span.bioguide_id = c.get("bioguide_id")
                continue
    return spans
# (shebang removed; file is invoked via python interpreter)

import argparse
import json
import logging
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
import csv  # NEW
try:
    import psutil  # For memory tracking
except Exception:  # pragma: no cover
    psutil = None

from speaker_attribution import (
    build_member_patterns,
    iter_speaker_spans,
    assign_speaker_for_offset,
)

LOGGER = logging.getLogger("attach_speakers")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def pick_text_row(r: Dict[str, str]) -> tuple[str, str]:
    for k in (
        "text_for_speaker",
        "text_bs4",
        "parsed_text",
        "text_readability",
        "text",
        "parsed_content_text",
    ):
        v = r.get(k)
        if isinstance(v, str) and v.strip():
            return v, k
    return "", ""


def log_memory_usage():
    """Log current memory usage of the process"""
    if psutil is None:
        LOGGER.info("Memory usage: psutil not installed; skipping snapshot")
        return
    process = psutil.Process()
    memory_info = process.memory_info()
    LOGGER.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")

def parse_args():
    p = argparse.ArgumentParser(description="Attach speaker attribution to mentions using char offsets.")
    p.add_argument("--mentions-jsonl", type=Path, required=True, help="Input mentions JSONL with char offsets.")
    p.add_argument("--normalized-dir", type=Path, required=True, help="Path to normalized dir (contains by_package/*/granules_core.csv).")
    p.add_argument("--members-csv", type=Path, required=False, help="Optional members CSV for canonicalization (first_name,last_name,bioguide_id,...)")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    p.add_argument("--save-csv", action="store_true")
    p.add_argument("--qa-jsonl", action="store_true", help="Write per-granule QA jsonl with spans and summary counts.")
    p.add_argument("--only-granules-with-members", action="store_true", help="Skip mentions from granules that have no members metadata")
    p.add_argument("--drop-unknown-speaker", action="store_true", help="Drop mentions when speaker attribution remains UNKNOWN")
    p.add_argument("--parallel", action="store_true", help="Use multiprocessing to speed up speaker attribution")
    p.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1), help="Number of worker processes to use if --parallel is enabled")
    p.add_argument("--batch-size", type=int, default=10000, help="Number of mentions to process in each batch")
    p.add_argument("--main-csv", type=Path, default=None,
                   help="Flat normalized CSV (e.g. data/normalized/main.csv). "
                        "Used instead of by_package/*/granules_core.csv for loading granule text.")
    # NEW:
    p.add_argument("--resume", action="store_true", help="Resume using processed_granules.jsonl manifest; append outputs")
    p.add_argument("--manifest", type=Path, default=None, help="Path to processed manifest (defaults to out_dir/processed_granules.jsonl)")
    return p.parse_args()


def read_jsonl(path: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows)


def stream_granules_text(
    normalized_dir: Path,
    needed: set[str],
    package_ids: Optional[set[str]] = None,
    main_csv: Optional[Path] = None,
) -> tuple[Dict[str, str], Dict[str, Optional[str]]]:
    """
    Load the text for each granuleId we need.
    Fast path: if main_csv provided, read directly from the flat CSV.
    Otherwise: if package_ids provided, only read those by_package/*/granules_core.csv files.
    Falls back to scanning all by_package if package_ids is None or no matches found.
    """
    # Log memory usage at the start
    log_memory_usage()
    wanted_cols = ["granuleId", "text_for_speaker", "text_bs4", "parsed_text", "text_readability"]
    out: Dict[str, str] = {}
    sources: Dict[str, Optional[str]] = {}

    def _pick_text_row(row: pd.Series) -> tuple[str, str]:
        for k in ("text_for_speaker", "text_bs4", "parsed_text", "text_readability"):
            v = row.get(k)
            if isinstance(v, str) and v.strip():
                return v, k
        return "", ""

    # Fast path: read from flat main.csv directly
    if main_csv and main_csv.exists():
        LOGGER.info("Reading granule texts from flat CSV: %s", main_csv)
        remaining = set(needed)
        chunk_num = 0
        for chunk in pd.read_csv(
            main_csv, chunksize=5000, dtype=str,
            keep_default_na=False,
            usecols=lambda c: c in wanted_cols,
        ):
            chunk_num += 1
            if "granuleId" not in chunk.columns:
                continue
            sub = chunk[chunk["granuleId"].astype(str).isin(remaining)]
            for _, r in sub.iterrows():
                gid = str(r["granuleId"])
                if gid in out:
                    continue
                txt, src = _pick_text_row(r)
                out[gid] = txt or ""
                sources[gid] = src
                remaining.discard(gid)
            if chunk_num % 20 == 0:
                LOGGER.info("Read %d chunks, found %d/%d granule texts...",
                           chunk_num, len(out), len(needed))
            if not remaining:
                LOGGER.info("Collected all %d requested granule texts from main CSV.", len(needed))
                break
        if remaining:
            LOGGER.warning("Missing %d granule texts from main CSV. Example: %s",
                          len(remaining), next(iter(remaining), None))
        return out, sources

    by_pkg = normalized_dir / "by_package"
    if not by_pkg.exists():
        # NEW: allow passing the by_package folder itself or any folder containing granules_core.csv
        if normalized_dir.name == "by_package":
            by_pkg = normalized_dir
        elif list(normalized_dir.rglob("granules_core.csv")):
            by_pkg = normalized_dir
        else:
            raise FileNotFoundError(f"Expected {by_pkg} to exist with granules_core.csv files")

    # Build list of CSVs to open
    csv_paths: List[Path] = []
    if package_ids:
        # Map packageId -> dir by reading package_meta.json (robust to directory naming scheme)
        for d in by_pkg.iterdir():
            if not d.is_dir():
                continue
            meta_path = d / "package_meta.json"
            try:
                if meta_path.exists():
                    with meta_path.open("r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                    pkg = str(meta.get("packageId", ""))
                    if pkg in package_ids:
                        cand = d / "granules_core.csv"
                        if cand.exists():
                            csv_paths.append(cand)
                else:
                    # Fallback: heuristic directory name contains __<packageId>__
                    for pkg in package_ids:
                        if f"__{pkg}__" in d.name:
                            cand = d / "granules_core.csv"
                            if cand.exists():
                                csv_paths.append(cand)
                            break
            except Exception:
                continue
        if not csv_paths:
            LOGGER.warning("No granules_core.csv found for given packageIds; falling back to full scan")
    if not csv_paths:
        # slow path
        csv_paths = list(by_pkg.glob("**/granules_core.csv"))

    LOGGER.info("Scanning %d granules_core.csv files for %d granules...", len(csv_paths), len(needed))

    remaining = set(needed)
    for core in tqdm(csv_paths, desc="granules_core.csv files"):
        try:
            for chunk in pd.read_csv(
                core,
                chunksize=5000,
                dtype=str,
                keep_default_na=False,
                usecols=lambda c: c in wanted_cols,
            ):
                if "granuleId" not in chunk.columns:
                    continue
                sub = chunk[chunk["granuleId"].astype(str).isin(remaining)]
                if sub.empty:
                    continue
                for _, r in sub.iterrows():
                    gid = str(r["granuleId"])  # safe str
                    if gid in out:
                        continue
                    txt, src = _pick_text_row(r)
                    out[gid] = txt or ""
                    sources[gid] = src
                    remaining.discard(gid)
                if not remaining:
                    LOGGER.info("Collected all requested granule texts.")
                    break
        except Exception as e:
            LOGGER.warning("Failed reading %s: %s", core, e)
        if not remaining:
            break

    if remaining:
        LOGGER.warning(
            "Missing %d granule texts (no text found). Example: %s",
            len(remaining),
            next(iter(remaining), None),
        )
    return out, sources


def save_jsonl(df: pd.DataFrame, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# NEW: streaming append helpers and resume manifest
def append_jsonl(records: List[Dict], path: Path):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def write_csv_append(df: pd.DataFrame, path: Path):
    header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", index=False, encoding="utf-8-sig", header=header)

def load_processed_set(manifest: Path) -> set[str]:
    if not manifest.exists():
        return set()
    done = set()
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                gid = obj.get("granuleId")
                if gid:
                    done.add(str(gid))
            except Exception:
                # also accept plain text lines
                done.add(line)
    return done

def append_processed(manifest: Path, gid: str):
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"granuleId": str(gid)}) + "\n")


def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    LOGGER.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")

def main():
    start_time = time.time()
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (args.out_dir / "processed_granules.jsonl")
    log_memory_usage()  # Initial memory snapshot

    # Prepare output targets and reset if not resuming
    out_jsonl = args.out_dir / "mentions_with_speakers.jsonl"
    out_csv = args.out_dir / "mentions_with_speakers.csv"
    qa_path = args.out_dir / "speaker_qc.jsonl"
    if not args.resume:
        try:
            if out_jsonl.exists():
                out_jsonl.unlink()
            if out_csv.exists() and args.save_csv:
                out_csv.unlink()
            if qa_path.exists() and args.qa_jsonl:
                qa_path.unlink()
            if manifest_path.exists():
                manifest_path.unlink()
        except Exception:
            pass
    
    # Load already processed granules if resuming
    processed_granules = set()
    if args.resume and manifest_path.exists():
        processed_granules = load_processed_set(manifest_path)
        LOGGER.info("Resuming processing. Skipping %d already processed granules.", len(processed_granules))
    
    df = read_jsonl(args.mentions_jsonl)
    # Filter out already processed granules
    if args.resume and processed_granules:
        before_count = len(df)
        df = df[~df["granuleId"].astype(str).isin(processed_granules)]
        LOGGER.info("Filtered out %d mentions from %d already processed granules",
                   before_count - len(df), len(processed_granules))
        if df.empty:
            LOGGER.error("No mentions remain after filtering for already processed granules")
            return

    required_cols = {"granuleId", "mention_char_start"}
    if not required_cols.issubset(df.columns):
        LOGGER.error("Input mentions lack required columns %s. Re-run extractor with offsets.", required_cols)
        return

    # Coerce types for safety
    if "granuleId" in df.columns:
        df["granuleId"] = df["granuleId"].astype(str)
    if "mention_char_start" in df.columns:
        df["mention_char_start"] = pd.to_numeric(df["mention_char_start"], errors="coerce")
    if "mention_char_end" in df.columns:
        df["mention_char_end"] = pd.to_numeric(df["mention_char_end"], errors="coerce")

    # Load granule members data
    members_by_gid = load_granule_members(args.normalized_dir)

    # Optional filter by members
    if args.only_granules_with_members and members_by_gid:
        before_count = len(df)
        df = df[df["granuleId"].astype(str).isin(set(members_by_gid.keys()))]
        LOGGER.info("Filtered from %d to %d mentions (keeping only granules with member data)",
                   before_count, len(df))
        if df.empty:
            LOGGER.error("No mentions remain after filtering for granules with members")
            return

    needed = set(df["granuleId"].dropna().astype(str))
    pkg_ids = set(df["packageId"].dropna().astype(str)) if "packageId" in df.columns else None
    LOGGER.info("Loading texts for %d granules from %s packages...", len(needed), (len(pkg_ids) if pkg_ids else "ALL"))
    text_by_gid, source_by_gid = stream_granules_text(
        args.normalized_dir, needed, pkg_ids,
        main_csv=getattr(args, "main_csv", None),
    )
    LOGGER.info("Loaded %d granule texts", len(text_by_gid))
    log_memory_usage()

    if args.members_csv and Path(args.members_csv).exists():
        try:
            df_members = pd.read_csv(args.members_csv, dtype=str)
        except Exception as e:
            LOGGER.warning("Could not read members CSV (%s): proceeding without canonicalization", e)
            df_members = pd.DataFrame()
    else:
        df_members = pd.DataFrame()
    members_idx = build_member_patterns(df_members)

    # Precompute speaker spans per granule
    spans_cache: Dict[str, List] = {}
    total_granules = len(text_by_gid)
    LOGGER.info(f"Starting span precomputation for {total_granules} granules...")
    log_memory_usage()
    
    # Track stats for reporting
    spans_found = 0
    spans_from_bs4 = 0
    single_speaker_defaults = 0
    
    # Use tqdm with position=0 to ensure it appears at the bottom of the terminal
    with tqdm(text_by_gid.items(), total=total_granules, desc="Precomputing spans", position=0, leave=True) as pbar:
        for i, (gid, txt) in enumerate(pbar):
            spans = iter_speaker_spans(txt or "", members_idx)
            # NEW: Enhance spans with granule member information
            spans = refine_spans_with_members(spans, gid, members_by_gid)
            
            # Belt-and-suspenders: try a liney alternative from normalized CSV (text_bs4) when available
            # Skip this fallback when using --main-csv (text already loaded from flat CSV)
            if not spans and not getattr(args, "main_csv", None):
                try:
                    by_pkg = args.normalized_dir / "by_package"
                    for core in by_pkg.glob("**/granules_core.csv"):
                        for chunk in pd.read_csv(core, chunksize=5000, dtype=str, keep_default_na=False):
                            cand = chunk.loc[chunk["granuleId"].astype(str) == str(gid)]
                            if not cand.empty:
                                raw_bs4 = cand.iloc[0].get("text_bs4", "")
                                if isinstance(raw_bs4, str) and raw_bs4.strip() and raw_bs4 != txt:
                                    spans = iter_speaker_spans(raw_bs4, members_idx)
                                    spans = refine_spans_with_members(spans, gid, members_by_gid)
                                    if spans:
                                        spans_from_bs4 += 1
                                break
                        if spans:
                            break
                except Exception:
                    pass
            
            # If no spans but we have granule members, create a default span for single-speaker granules
            if not spans and gid in members_by_gid and len(members_by_gid[gid]) == 1:
                member = members_by_gid[gid][0]
                name = f"{member.get('first_name', '').title()} {member.get('last_name', '').title()}".strip()
                spans = [type('SpeakerSpan', (), {})()]
                spans[0].start = 0
                spans[0].end = len(txt or "")
                spans[0].raw_label = name
                spans[0].canonical_name = name
                spans[0].bioguide_id = member.get("bioguide_id")
                single_speaker_defaults += 1
            
            spans_cache[gid] = spans
            if spans:
                spans_found += 1
                
            # Update progress bar description with some stats
            if (i + 1) % 10 == 0:
                pbar.set_postfix(found=f"{spans_found}/{i+1}", bs4=spans_from_bs4, defaults=single_speaker_defaults)
            
            if (i + 1) % 500 == 0:
                LOGGER.info("Precomputed spans for %d/%d granules (%.1f%%): found=%d, from_bs4=%d, defaults=%d", 
                           i + 1, total_granules, 100*(i+1)/total_granules, spans_found, spans_from_bs4, single_speaker_defaults)
                log_memory_usage()

    # Log span precomputation summary
    spans_with_speakers = sum(1 for spans in spans_cache.values() if any(span.canonical_name for span in spans))
    LOGGER.info(f"Span precomputation complete. Stats:")
    LOGGER.info(f"  - Total granules: {len(text_by_gid)}")
    LOGGER.info(f"  - Granules with any spans: {sum(1 for spans in spans_cache.values() if spans)}")
    LOGGER.info(f"  - Granules with identified speakers: {spans_with_speakers}")
    LOGGER.info(f"  - Coverage: {spans_with_speakers/len(text_by_gid)*100:.1f}%")
    log_memory_usage()

    # Progress over granules with ETA
    granule_keys = [str(k) for k in df["granuleId"].astype(str).unique()]
    
    # Initialize counters for tracking attribution quality
    unknown_speakers = 0
    resolved_speakers = 0
    
    LOGGER.info(f"Processing {len(granule_keys)} unique granules with {len(df)} total mentions")
    log_memory_usage()
    
    # We'll stream results per granule instead of holding everything in memory
    
    # Process granules in parallel or sequentially based on args
    if args.parallel:
        LOGGER.info(f"Using parallel processing with {args.workers} workers")
        # Create a partial function with the precomputed data
        process_func = partial(
            process_granule, 
            text_by_gid=text_by_gid, 
            spans_cache=spans_cache,
            members_by_gid=members_by_gid
        )
        
        # Group by granuleId and convert to list of (granule_id, dataframe) tuples
        granule_groups = [(gid, g) for gid, g in df.groupby("granuleId")]
        
        # Use ProcessPoolExecutor to parallelize over granules
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            iterator = executor.map(process_func, granule_groups)
            for (gid, _), batch_results in tqdm(zip(granule_groups, iterator), total=len(granule_groups),
                                               desc="Attributing (granules)", unit="granule"):
                # Optionally drop unknowns before writing
                if args.drop_unknown_speaker:
                    batch_results = [r for r in batch_results if r.get("speaker_raw") != "UNKNOWN"]

                # Update counters
                for row in batch_results:
                    if row.get("speaker_raw") == "UNKNOWN":
                        unknown_speakers += 1
                    else:
                        resolved_speakers += 1

                # Stream writes
                append_jsonl(batch_results, out_jsonl)
                if args.save_csv and batch_results:
                    write_csv_append(pd.DataFrame(batch_results), out_csv)

                # Stream QA per granule
                if args.qa_jsonl:
                    spans = spans_cache.get(str(gid), [])
                    counts: Dict[str, int] = {}
                    for r in batch_results:
                        m = r.get("speaker_method")
                        counts[m] = counts.get(m, 0) + 1
                    span_snap = [
                        {"start": getattr(s, 'start', None), "end": getattr(s, 'end', None),
                         "raw": getattr(s, 'raw_label', None), "canon": getattr(s, 'canonical_name', None)}
                        for s in spans[:5]
                    ]
                    first_cue = spans[0].raw_label if spans else None
                    last_cue = spans[-1].raw_label if spans else None
                    common_src = source_by_gid.get(str(gid))
                    with qa_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "granuleId": str(gid),
                            "counts": counts,
                            "n": int(len(batch_results)),
                            "spans_sample": span_snap,
                            "n_spans": len(spans),
                            "first_cue": first_cue,
                            "last_cue": last_cue,
                            "text_source": common_src,
                        }, ensure_ascii=False) + "\n")

                # Update manifest per granule
                if args.resume:
                    append_processed(manifest_path, str(gid))
    else:
        # Sequential processing (original logic)
        pbar = tqdm(total=len(granule_keys), desc="Attributing (granules)", unit="granule")
        
        for gid, g in df.groupby("granuleId"):
            gid = str(gid)
            pbar.set_postfix_str(gid[-22:])  # show tail of current gid
            pbar.update(1)
            
            batch_results = process_granule((gid, g), text_by_gid, spans_cache, members_by_gid)
            if args.drop_unknown_speaker:
                batch_results = [r for r in batch_results if r.get("speaker_raw") != "UNKNOWN"]
            
            # Update counters
            for row in batch_results:
                if row.get("speaker_raw") == "UNKNOWN":
                    unknown_speakers += 1
                else:
                    resolved_speakers += 1
            
            # Update manifest for successful granule if resuming
            if args.resume:
                append_processed(manifest_path, gid)

            # Stream writes
            append_jsonl(batch_results, out_jsonl)
            if args.save_csv and batch_results:
                write_csv_append(pd.DataFrame(batch_results), out_csv)

            # Stream QA per granule
            if args.qa_jsonl:
                spans = spans_cache.get(str(gid), [])
                counts: Dict[str, int] = {}
                for r in batch_results:
                    m = r.get("speaker_method")
                    counts[m] = counts.get(m, 0) + 1
                span_snap = [
                    {"start": getattr(s, 'start', None), "end": getattr(s, 'end', None),
                     "raw": getattr(s, 'raw_label', None), "canon": getattr(s, 'canonical_name', None)}
                    for s in spans[:5]
                ]
                first_cue = spans[0].raw_label if spans else None
                last_cue = spans[-1].raw_label if spans else None
                common_src = source_by_gid.get(str(gid))
                with qa_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "granuleId": str(gid),
                        "counts": counts,
                        "n": int(len(batch_results)),
                        "spans_sample": span_snap,
                        "n_spans": len(spans),
                        "first_cue": first_cue,
                        "last_cue": last_cue,
                        "text_source": common_src,
                    }, ensure_ascii=False) + "\n")
                
        pbar.close()
        
    LOGGER.info(f"Speaker attribution complete: {resolved_speakers} resolved, {unknown_speakers} unknown")
    log_memory_usage()

    # We streamed outputs per granule; log where files are
    if out_jsonl.exists():
        try:
            n_lines = sum(1 for _ in out_jsonl.open("r", encoding="utf-8"))
            LOGGER.info("Wrote %s (%d rows)", out_jsonl, n_lines)
        except Exception:
            LOGGER.info("Wrote %s", out_jsonl)
    if args.save_csv and out_csv.exists():
        LOGGER.info("Wrote %s (CSV)", out_csv)
    if args.qa_jsonl and qa_path.exists():
        LOGGER.info("Wrote QA summaries to %s", qa_path)

    # Report final timing and results
    total_mentions = len(df)
    elapsed = time.time() - start_time
    mentions_per_sec = total_mentions/elapsed if elapsed > 0 else 0
    
    LOGGER.info("Processed %d mentions in %.2f seconds (%.1f mentions/sec)",
               total_mentions, elapsed, mentions_per_sec)
    LOGGER.info("Speaker attribution results: %d resolved, %d unknown (%.1f%%)",
               resolved_speakers, unknown_speakers, 
               100 * resolved_speakers / total_mentions if total_mentions > 0 else 0)


def process_granule(granule_data: Tuple[str, pd.DataFrame], text_by_gid: Dict[str, str], 
                spans_cache: Dict[str, List], members_by_gid: Dict[str, List]) -> List[Dict]:
    """Process a single granule for parallel execution"""
    gid, g = granule_data  # Unpack the tuple (granule_id, dataframe_slice)
    gid = str(gid)
    
    speaker_rows = []
    spans = spans_cache.get(gid, [])
    text_missing = gid not in text_by_gid or not text_by_gid.get(gid)
    single_member = None
    if not spans and gid in members_by_gid and len(members_by_gid[gid]) == 1:
        member = members_by_gid[gid][0]
        single_member = f"{member.get('first_name', '').title()} {member.get('last_name', '').title()}".strip()
    
    for _, r in g.iterrows():
        rec = r.to_dict()
        if text_missing:
            rec.update({
                "speaker_raw": "UNKNOWN",
                "speaker_canonical": None,
                "speaker_bioguide": None,
                "speaker_method": "no_text",
                "speaker_confidence": 0.0,
            })
            speaker_rows.append(rec)
            continue
        try:
            offset = int(r.get("mention_char_start"))
        except Exception:
            rec.update({
                "speaker_raw": "UNKNOWN",
                "speaker_canonical": None,
                "speaker_bioguide": None,
                "speaker_method": "no_offset",
                "speaker_confidence": 0.0,
            })
            speaker_rows.append(rec)
            continue
        
        # If we have spans, use them to attribute speaker
        if spans:
            s, method, conf = assign_speaker_for_offset(offset, spans)
            rec.update({
                "speaker_raw": s.raw_label,
                "speaker_canonical": s.canonical_name,
                "speaker_bioguide": s.bioguide_id,
                "speaker_method": method,
                "speaker_confidence": conf,
            })
        # If no spans but single member, use that
        elif single_member:
            member = members_by_gid[gid][0]
            rec.update({
                "speaker_raw": single_member,
                "speaker_canonical": single_member,
                "speaker_bioguide": member.get("bioguide_id"),
                "speaker_method": "sole_granule_member",
                "speaker_confidence": 0.7,
            })
        else:
            rec.update({
                "speaker_raw": "UNKNOWN",
                "speaker_canonical": None,
                "speaker_bioguide": None,
                "speaker_method": "no_speaker_cues",
                "speaker_confidence": 0.0,
            })
        speaker_rows.append(rec)
    return speaker_rows

def process_in_batches(df, batch_size=10000):
    """Process a dataframe in batches to conserve memory"""
    total_rows = len(df)
    results = []
    
    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        LOGGER.info(f"Processing batch {start//batch_size + 1} ({start}-{end} of {total_rows})")
        batch_df = df.iloc[start:end]
        batch_results = []
        
        # Process each granule in the batch
        for gid, g in batch_df.groupby("granuleId"):
            batch_results.extend(process_granule((gid, g), {}, {}, {}))
            
        # Save intermediate results
        results.extend(batch_results)
        
    return results


if __name__ == "__main__":
    main()
