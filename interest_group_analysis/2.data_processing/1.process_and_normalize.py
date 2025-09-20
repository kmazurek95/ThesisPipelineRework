"""Stream raw govinfo results and produce normalized CSV tables.

This script iterates over files in data/raw (JSON, JSONL, .gz). For JSONL files it
uses pandas.read_json(..., lines=True, chunksize=10000) to stream. For JSON arrays
or single JSON files it loads the entire file (pd.json_normalize) and processes
it as a single chunk.

For each DataFrame chunk the script:
- ensures a `granuleId` column exists
- builds a canonical `main` rowset and computes text length fields
- extracts and flattens `references`, `committees`, and `members` into separate
    CSVs under data/normalized
- appends to CSV files using mode='a' and writes headers only if the file did not
    exist

At the end a small JSON summary is written to data/normalized/summary.json.
"""

from __future__ import annotations

import gzip
import json
import ast
import os
import glob
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import argparse
import logging
import shutil
import re
from datetime import datetime

import pandas as pd

SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/normalized")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# default chunk size for pandas streaming
CHUNKSIZE = 10000

LOG = logging.getLogger(__name__)


# Optional split mode (None or "package"). When enabled, instead of appending
# to aggregate CSVs, data will be written per-package under
# by_package/<packageId>/{main,references,committees,members}.csv.
SPLIT_MODE: Optional[str] = None
BY_PACKAGE_DIR: Path = OUT_DIR / "by_package"

# --- New CLI knobs ---
ONLY_CONGRESS: Optional[int] = None          # e.g., 114
EMIT_SEARCH_TEXT: bool = False               # create search-ready text column
PREFER_TEXT_ORDER = ["text_readability", "parsed_text", "text_bs4"]


def _maybe_clean_out_dir(out_dir: Path):
    """Remove CSVs and summary.json from out_dir to allow a fresh run."""
    patterns = ["*.csv", "summary.json"]
    for pat in patterns:
        for p in out_dir.glob(pat):
            try:
                p.unlink()
            except Exception:
                LOG.warning("Could not remove %s", p)
    # Also remove split directories if present
    for d in (out_dir / "by_package",):
        try:
            if d.exists():
                shutil.rmtree(d)
        except Exception:
            LOG.warning("Could not remove directory %s", d)


def parse_json_field(v: Any):
    """If v is a JSON string, try to parse it. Otherwise return as-is."""
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, str):
        v = v.strip()
        if not v:
            return None
        try:
            return json.loads(v)
        except Exception:
            # not a strict JSON string; try Python literal_eval (single quotes)
            try:
                return ast.literal_eval(v)
            except Exception:
                # not a JSON or Python literal; return original
                return v
    return v


def ensure_granule_id(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if "granuleId" in df.columns:
        return df
    # try common alternatives
    for alt in ("id", "granule_id", "granuleId_str"):
        if alt in df.columns:
            df["granuleId"] = df[alt]
            return df
    # fallback: synthesize from source filename + row index
    df = df.reset_index(drop=True)
    df["granuleId"] = df.index.map(lambda i: f"{source_name}__{i}")
    return df


def write_append_csv(df: pd.DataFrame, path: Path):
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header)



def write_split_by_package(df_main: pd.DataFrame,
                           df_refs: pd.DataFrame,
                           df_comm: pd.DataFrame,
                           df_mem: pd.DataFrame):
    """When SPLIT_MODE=='package', write tables under descriptive per-package folders."""
    global SPLIT_MODE, BY_PACKAGE_DIR
    if SPLIT_MODE != "package":
        return

    # Pick a non-empty frame to compute directory name
    basis = next((x for x in [df_main, df_refs, df_comm, df_mem] if x is not None and not x.empty), None)
    if basis is None or "packageId" not in basis.columns:
        return

    # One subfolder per unique packageId present in this chunk
    for pkg_id, grp in basis.groupby("packageId"):
        # Build a dir name using all rows for this package across inputs
        # Gather per-package slices
        m = df_main[df_main["packageId"] == pkg_id] if df_main is not None and not df_main.empty else pd.DataFrame()
        r = df_refs[df_refs["packageId"] == pkg_id] if df_refs is not None and not df_refs.empty else pd.DataFrame()
        c = df_comm[df_comm["packageId"] == pkg_id] if df_comm is not None and not df_comm.empty else pd.DataFrame()
        me = df_mem[df_mem["packageId"] == pkg_id] if df_mem is not None and not df_mem.empty else pd.DataFrame()

        # Use whatever slice is non-empty to craft folder name
        basis_pkg = next((x for x in [m, r, c, me] if not x.empty), pd.DataFrame({"packageId":[pkg_id]}))
        pkg_dir_name = build_pkg_dir_name(basis_pkg)
        pkg_dir = BY_PACKAGE_DIR / pkg_dir_name
        pkg_dir.mkdir(parents=True, exist_ok=True)

        # File names: succinct + descriptive
        if not m.empty:
            write_append_csv(m, pkg_dir / "granules_core.csv")
        if not r.empty:
            write_append_csv(r, pkg_dir / "granule_references.csv")
        if not c.empty:
            write_append_csv(c, pkg_dir / "granule_committees.csv")
        if not me.empty:
            write_append_csv(me, pkg_dir / "granule_members.csv")

        # Lightweight meta file to aid downstream tools
        meta = {
            "packageId": str(pkg_id),
            "congress": (int(m["congress"].dropna().iloc[0]) if "congress" in m.columns and not m["congress"].dropna().empty else None),
            "date": (str(m["date"].dropna().iloc[0]) if "date" in m.columns and not m["date"].dropna().empty else None),
            "chambers": chambers_from(basis_pkg),
            "counts": {
                "granules_core": int(len(m)) if not m.empty else 0,
                "granule_references": int(len(r)) if not r.empty else 0,
                "granule_committees": int(len(c)) if not c.empty else 0,
                "granule_members": int(len(me)) if not me.empty else 0,
            }
        }
        with open(pkg_dir / "package_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def parse_pkg_filename(name: str) -> tuple[Optional[int], Optional[int]]:
    """
    Parse 'package_114_123.csv' → (114, 123). Returns (None, None) if not matched.
    """
    m = re.match(r"package_(\d+)_(\d+)\.csv$", name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def build_main_table(df: pd.DataFrame) -> pd.DataFrame:
    # canonical columns to preserve if present
    canonical = [
        "granuleId",
        "packageId",
        "package_number",
        "congress",
        "date",
        "url",
        "title",
        "subject",
        "type",
        "summary",
        "parsed_text",
        "text_readability",
        "text_bs4",
    ]
    cols = [c for c in canonical if c in df.columns]
    main = df[cols].copy()

    # add provenance
    if "__source_file" in df.columns:
        main["__source_file"] = df["__source_file"]

    # compute text lengths
    def _len_col(s):
        return s.fillna("").astype(str).map(len)

    if "parsed_text" in main.columns:
        main["parsed_text_len"] = _len_col(main["parsed_text"])
    if "text_readability" in main.columns:
        main["text_readability_len"] = _len_col(main["text_readability"])
    if "text_bs4" in main.columns:
        main["text_bs4_len"] = _len_col(main["text_bs4"])

    # Optional: emit a single search-ready text column for fast dictionary matching
    if EMIT_SEARCH_TEXT:
        # pick first non-empty among preferred text fields
        def pick_text(row):
            for col in PREFER_TEXT_ORDER:
                if col in row and isinstance(row[col], str) and row[col].strip():
                    return row[col]
            return ""
        text = main.apply(pick_text, axis=1)
        # normalize for matching: lowercase + compress whitespace
        text = text.astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        main["search_text"] = text
        main["search_text_len"] = main["search_text"].map(len)

    return main


def explode_and_normalize(df: pd.DataFrame, field: str, parent_key: str, extra_parent_cols=None) -> pd.DataFrame:
    """
    Deeply flatten/explode `field` into a 1-row-per-atomic-item table.

    - Parses JSON strings.
    - Explodes lists repeatedly until none remain.
    - Flattens dicts repeatedly until none remain.
    - Carries through parent keys (e.g., granuleId) and any extra parent cols.
    - Normalizes nested keys with `sep`.
    """
    if field not in df.columns and not (field == "references" and "contents" in df.columns):
        return pd.DataFrame()

    # Accept 'contents' as references fallback
    source_field = field if field in df.columns else "contents"

    # Prepare base rows: each parent row contributes zero or more child items
    def to_list(x):
        x = parse_json_field(x)
        if x is None:
            return []
        if isinstance(x, list):
            return x
        if isinstance(x, dict):
            return [x]
        # strings or scalars: keep as a single-item list
        return [x]

    items_per_row = df[source_field].apply(to_list)
    if items_per_row.map(len).sum() == 0:
        return pd.DataFrame()

    base_rows = []
    keep_cols = [c for c in ([parent_key] + (extra_parent_cols or [])) if c in df.columns]

    for idx, items in items_per_row.items():
        parent_vals = {c: df.at[idx, c] for c in keep_cols}
        for it in items:
            # normalize dict keys a bit (hyphens/spaces -> underscores) before deep flatten
            if isinstance(it, dict):
                it = {str(k).replace("-", "_").replace(" ", "_"): v for k, v in it.items()}
            base_rows.append({**parent_vals, field: it})

    if not base_rows:
        return pd.DataFrame()

    # Start with a shallow normalize—this turns dicts under `field` into columns
    flat = pd.json_normalize(base_rows, sep="__")

    # If the dict was directly under `field`, columns look like `field__key`.
    # Bring those up to top-level by stripping the 'field__' prefix for readability.
    field_prefix = f"{field}__"
    renamed = {}
    for c in flat.columns:
        if c.startswith(field_prefix):
            renamed[c] = c[len(field_prefix):]
    flat = flat.rename(columns=renamed)

    # Iteratively explode list columns and expand dict columns until none remain
    def has_lists(s: pd.Series) -> bool:
        # fast check: treat strings as not lists even if they look like '[...]'
        return s.apply(lambda x: isinstance(x, list)).any()

    def has_dicts(s: pd.Series) -> bool:
        return s.apply(lambda x: isinstance(x, dict)).any()

    passes = 0
    max_passes = 10
    while passes < max_passes:
        passes += 1
        changed = False

        # 1) explode every list column present
        list_cols = [c for c in flat.columns if has_lists(flat[c])]
        for c in list_cols:
            flat = flat.explode(c, ignore_index=True)
            changed = True

        # 2) expand any dict columns into prefixed columns
        dict_cols = [c for c in flat.columns if has_dicts(flat[c])]
        for c in dict_cols:
            # normalize this column's dicts to columns
            expanded = pd.json_normalize(flat[c], sep="__")
            expanded.columns = [f"{c}__{sub}" for sub in expanded.columns]
            flat = pd.concat([flat.drop(columns=[c]).reset_index(drop=True),
                              expanded.reset_index(drop=True)], axis=1)
            changed = True

        if not changed:
            break

    # Final safety: convert any lingering non-scalar (list/dict) values to JSON strings
    for c in flat.columns:
        flat[c] = flat[c].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x
        )

    # Reorder to keep parent keys first
    parent_first = [c for c in keep_cols if c in flat.columns]
    rest = [c for c in flat.columns if c not in parent_first]
    flat = flat[parent_first + rest]

    return flat


def process_chunk(df: pd.DataFrame, source_name: str, counters: Dict[str, int]):
    # annotate provenance
    df["__source_file"] = source_name
    df = ensure_granule_id(df, source_name)

    main = build_main_table(df)
    counters["main_rows"] += len(main)
    if not main.empty and SPLIT_MODE != "package":
        write_append_csv(main, OUT_DIR / "main.csv")
        # Monthly partitioning removed

    # explode references
    refs = explode_and_normalize(df, "references", "granuleId", extra_parent_cols=["packageId", "date"])
    counters["references_rows"] += len(refs)
    if not refs.empty:
        assert_fully_scalar(refs, "references")
        if SPLIT_MODE != "package":
            write_append_csv(refs, OUT_DIR / "references.csv")

    comm = explode_and_normalize(df, "committees", "granuleId", extra_parent_cols=["packageId", "date"])
    counters["committees_rows"] += len(comm)
    if not comm.empty:
        if SPLIT_MODE != "package":
            write_append_csv(comm, OUT_DIR / "committees.csv")

    mem = explode_and_normalize(df, "members", "granuleId", extra_parent_cols=["packageId", "date"])
    counters["members_rows"] += len(mem)
    if not mem.empty:
        assert_fully_scalar(mem, "members")
        if SPLIT_MODE != "package":
            write_append_csv(mem, OUT_DIR / "members.csv")

    # In split-by-package mode write per-package shards now
    if SPLIT_MODE == "package":
        write_split_by_package(main, refs, comm, mem)


def assert_fully_scalar(df: pd.DataFrame, table_name: str):
    """Raise if any column contains list or dict values (used as sanity check)."""
    offenders = []
    for c in df.columns:
        if df[c].apply(lambda x: isinstance(x, (list, dict))).any():
            offenders.append(c)
    if offenders:
        raise ValueError(f"{table_name} still has nested columns: {', '.join(offenders)}")


def iter_raw_files(path: Path) -> Iterable[Path]:
    # Also include CSVs produced by the collector (package_*.csv)
    patterns = ["**/*.json", "**/*.jsonl", "**/*.gz", "**/*.csv"]
    seen = set()
    for pat in patterns:
        for p in path.glob(pat):
            if p.is_file() and p not in seen:
                # If user asked for a specific congress, filter filenames like package_114_*.csv
                if ONLY_CONGRESS is not None:
                    c, _ = parse_pkg_filename(p.name)
                    if c is None or c != ONLY_CONGRESS:
                        continue
                seen.add(p)
                yield p


def is_json_array(path: Path) -> bool:
    # peek into file (gz or plain) to see if it starts with '[' after whitespace
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
            for _ in range(10):
                ch = fh.read(1)
                if not ch:
                    return False
                if ch.isspace():
                    continue
                return ch == "["
    except Exception:
        return False


def process_file(path: Path, counters: Dict[str, int]):
    name = path.name
    print(f"Processing {path}...")
    # JSONL streaming
    try:
        if path.suffix == ".jsonl" or (path.suffix == ".gz" and ".jsonl" in path.name):
            # use pandas streaming
            it = pd.read_json(path, lines=True, chunksize=CHUNKSIZE, compression=("gzip" if path.suffix == ".gz" else "infer"))
            for chunk in it:
                counters["processed_chunks"] += 1
                process_chunk(chunk.astype(object), name, counters)
            counters["files_processed"] += 1
            return

        # CSV streaming (collector may produce package_*.csv files)
        if path.suffix == ".csv":
            it = pd.read_csv(path, chunksize=CHUNKSIZE)
            for chunk in it:
                counters["processed_chunks"] += 1
                # ensure a packageId (some collector CSVs already include it)
                if "packageId" not in chunk.columns:
                    c, i = parse_pkg_filename(name)
                    synth = f"CREC-{c}-{i}" if (c is not None and i is not None) else f"CREC-{name}"
                    chunk["packageId"] = synth
                if "congress" not in chunk.columns:
                    c, _ = parse_pkg_filename(name)
                    if c is not None:
                        chunk["congress"] = c
                process_chunk(chunk.astype(object), name, counters)
            counters["files_processed"] += 1
            return

        # For gz without .jsonl or plain json/json
        if path.suffix == ".gz":
            # Decide if json array or jsonl by peeking
            if is_json_array(path):
                with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
                    data = json.load(fh)
                df = pd.json_normalize(data)
                counters["processed_chunks"] += 1
                process_chunk(df.astype(object), name, counters)
                counters["files_processed"] += 1
                return
            else:
                # treat as JSONL
                it = pd.read_json(path, lines=True, chunksize=CHUNKSIZE, compression="gzip")
                for chunk in it:
                    counters["processed_chunks"] += 1
                    process_chunk(chunk.astype(object), name, counters)
                counters["files_processed"] += 1
                return

        # plain .json or .jsonl
        if path.suffix == ".json":
            # peek for array
            if is_json_array(path):
                with open(path, "rt", encoding="utf-8", errors="ignore") as fh:
                    data = json.load(fh)
                df = pd.json_normalize(data)
                counters["processed_chunks"] += 1
                process_chunk(df.astype(object), name, counters)
            else:
                # treat as single-record JSON or object
                with open(path, "rt", encoding="utf-8", errors="ignore") as fh:
                    try:
                        obj = json.load(fh)
                        # if it's a dict that's not a list, try to normalize
                        df = pd.json_normalize(obj)
                        counters["processed_chunks"] += 1
                        process_chunk(df.astype(object), name, counters)
                    except Exception:
                        # fallback to pandas read_json
                        it = pd.read_json(path, lines=True, chunksize=CHUNKSIZE)
                        for chunk in it:
                            counters["processed_chunks"] += 1
                            process_chunk(chunk.astype(object), name, counters)
            counters["files_processed"] += 1
            return

        # unknown suffix: try reading as jsonl
        try:
            it = pd.read_json(path, lines=True, chunksize=CHUNKSIZE)
            for chunk in it:
                counters["processed_chunks"] += 1
                process_chunk(chunk.astype(object), name, counters)
            counters["files_processed"] += 1
            return
        except Exception:
            print(f"Skipped {path}: unrecognized format")
            counters["skipped_files"] += 1
            return

    except Exception as exc:
        print(f"Error processing {path}: {exc}")
        counters["failed_files"] += 1


def safe_slug(s: str) -> str:
    return SAFE_CHARS.sub("_", str(s)).strip("_")


def yyyymmdd(date_str: str | None) -> str:
    if not date_str:
        return "unknown"
    # Accept "YYYY-MM-DD" or full ISO
    try:
        return datetime.fromisoformat(date_str[:10]).strftime("%Y%m%d")
    except Exception:
        return "unknown"


def chambers_from(df: pd.DataFrame) -> str:
    """
    Heuristic: derive chambers present from granuleId or granuleClass if available.
    - PgH… → H, PgS… → S, PgE… → E
    - fall back to 'granuleClass' values containing 'HOUSE'/'SENATE'/'EXT'
    """
    ch = set()
    if "granuleId" in df.columns:
        vals = df["granuleId"].astype(str).fillna("")
        if vals.str.contains(r"PgH", regex=True).any(): ch.add("H")
        if vals.str.contains(r"PgS", regex=True).any(): ch.add("S")
        if vals.str.contains(r"PgE", regex=True).any(): ch.add("E")
    if not ch and "granuleClass" in df.columns:
        g = df["granuleClass"].astype(str).str.upper().fillna("")
        if g.str.contains("HOUSE").any(): ch.add("H")
        if g.str.contains("SENATE").any(): ch.add("S")
        if g.str.contains("EXT").any() or g.str.contains("EXTENSIONS").any(): ch.add("E")
    return "".join(sorted(ch)) or "UNK"


def build_pkg_dir_name(df_any: pd.DataFrame) -> str:
    pkg = df_any.get("packageId")
    congress = df_any.get("congress")
    date_col = df_any.get("date")
    # coerce scalars from first non-null
    pkgid = safe_slug(str(pkg.dropna().iloc[0])) if isinstance(pkg, pd.Series) else safe_slug(str(pkg))
    cong = str(int(congress.dropna().iloc[0])) if isinstance(congress, pd.Series) and not congress.dropna().empty else (str(congress) if congress else "UNK")
    date_str = date_col.dropna().iloc[0] if isinstance(date_col, pd.Series) and not date_col.dropna().empty else None
    chambers = chambers_from(df_any if isinstance(df_any, pd.DataFrame) else pd.DataFrame())
    return f"{cong}C__{pkgid}__{yyyymmdd(date_str)}__{chambers}"


def main():
    parser = argparse.ArgumentParser(description="Stream raw govinfo results and produce normalized CSV tables")
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw input files")
    parser.add_argument("--out-dir", default="data/normalized", help="Path to write normalized CSVs")
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE, help="pandas read_json chunksize")
    parser.add_argument("--clean", action="store_true", help="Remove existing CSVs/summary before running")
    parser.add_argument("--split-by", choices=["package"], help="Write outputs per packageId under by_package/<packageId>/")
    parser.add_argument("--only-congress", type=int, help="Filter input files to this congress (e.g., 114)")
    parser.add_argument("--emit-search-text", action="store_true",
                        help="Create normalized lowercased search_text (+ length) in main.csv")
    args = parser.parse_args()
    main_with_params(Path(args.raw_dir), Path(args.out_dir), args.chunksize,
                     clean=args.clean, split_by=args.split_by,
                     only_congress=args.only_congress,
                     emit_search_text=args.emit_search_text)

def main_with_params(raw_dir: Path, out_dir: Path, chunksize: int = CHUNKSIZE,
                     clean: bool = False, split_by: Optional[str] = None,
                     only_congress: Optional[int] = None,
                     emit_search_text: bool = False) -> Dict[str, int]:
    """Process files under raw_dir and write normalized CSVs to out_dir.

    Returns the counters dict.
    """
    global RAW_DIR, OUT_DIR, CHUNKSIZE, SPLIT_MODE, BY_PACKAGE_DIR, ONLY_CONGRESS, EMIT_SEARCH_TEXT
    RAW_DIR = Path(raw_dir)
    OUT_DIR = Path(out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKSIZE = chunksize
    SPLIT_MODE = split_by
    BY_PACKAGE_DIR = OUT_DIR / "by_package"
    ONLY_CONGRESS = only_congress
    EMIT_SEARCH_TEXT = emit_search_text
    # monthly partitioning removed

    if clean:
        _maybe_clean_out_dir(OUT_DIR)

    combined = RAW_DIR / "legislative_granules.jsonl"
    # Always process all files in the raw dir; if a combined file exists, skip it to avoid duplication
    paths = list(iter_raw_files(RAW_DIR))
    if combined.exists():
        paths = [p for p in paths if p.resolve() != combined.resolve()]
    if not paths:
        LOG.info("No raw files found in %s", RAW_DIR)
        return {}

    counters = {
        "files_processed": 0,
        "processed_chunks": 0,
        "main_rows": 0,
        "references_rows": 0,
        "committees_rows": 0,
        "members_rows": 0,
        "skipped_files": 0,
        "failed_files": 0,
    }

    for p in paths:
        process_file(p, counters)

    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(counters, fh, indent=2)

    LOG.info("Processing complete. Summary: %s", json.dumps(counters))
    return counters


if __name__ == "__main__":
    main()
