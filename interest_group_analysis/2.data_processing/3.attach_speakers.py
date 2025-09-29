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
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm
import csv  # NEW

from speaker_attribution import (
    build_member_patterns,
    iter_speaker_spans,
    assign_speaker_for_offset,
)

LOGGER = logging.getLogger("attach_speakers")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

r"""
# How to Run This Script

This script attaches speakers to mentions in legislative data.

## Example Command:
```powershell
python .\interest_group_analysis\2.data_processing\3.attach_speakers.py `
  --mentions-jsonl data\processed\mentions_114_run2\mentions.jsonl `
  --normalized-dir data\normalized\normalized_114_run2 `
  --out-dir data\processed\mentions_and_speaker_114 `
  --save-csv `
  --qa-jsonl
```

## Arguments:
- `--mentions-jsonl`: Path to the JSONL file containing mentions.
- `--normalized-dir`: Directory containing normalized data.
- `--out-dir`: Directory to save the output.
- `--save-csv`: Save the output in CSV format.
- `--qa-jsonl`: Save QA results in JSONL format.
"""


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
) -> tuple[Dict[str, str], Dict[str, Optional[str]]]:
    """
    Load the text for each granuleId we need.
    Fast path: if package_ids provided, only read those by_package/*/granules_core.csv files.
    Falls back to scanning all by_package if package_ids is None or no matches found.
    """
    wanted_cols = ["granuleId", "text_for_speaker", "text_bs4", "parsed_text", "text_readability"]
    out: Dict[str, str] = {}
    sources: Dict[str, Optional[str]] = {}

    def _pick_text_row(row: pd.Series) -> tuple[str, str]:
        for k in ("text_for_speaker", "text_bs4", "parsed_text", "text_readability"):
            v = row.get(k)
            if isinstance(v, str) and v.strip():
                return v, k
        return "", ""

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


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (args.out_dir / "processed_granules.jsonl")

    df = read_jsonl(args.mentions_jsonl)
    if df.empty:
        LOGGER.error("No mentions in input: %s", args.mentions_jsonl)
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
    text_by_gid, source_by_gid = stream_granules_text(args.normalized_dir, needed, pkg_ids)

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
    for gid, txt in text_by_gid.items():
        spans = iter_speaker_spans(txt or "", members_idx)
        # NEW: Enhance spans with granule member information
        spans = refine_spans_with_members(spans, gid, members_by_gid)
        # Belt-and-suspenders: if no spans, try a liney alternative from normalized CSV (text_bs4) when available
        if not spans:
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
        spans_cache[gid] = spans

    # Progress over granules with ETA
    granule_keys = [str(k) for k in df["granuleId"].astype(str).unique()]
    pbar = tqdm(total=len(granule_keys), desc="Attributing (granules)", unit="granule")

    speaker_rows: List[Dict] = []
    for gid, g in df.groupby("granuleId"):
        gid = str(gid)
        pbar.set_postfix_str(gid[-22:])  # show tail of current gid
        pbar.update(1)
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
    pbar.close()

    out_df = pd.DataFrame(speaker_rows)
    # Optional: filter out rows with unknown speakers
    if args.drop_unknown_speaker and not out_df.empty:
        before_count = len(out_df)
        out_df = out_df[out_df["speaker_raw"] != "UNKNOWN"]
        LOGGER.info("Filtered from %d to %d mentions (dropping unknown speakers)",
                   before_count, len(out_df))
    if out_df.empty:
        LOGGER.error("No output rows produced. Check inputs.")
        return

    out_jsonl = args.out_dir / "mentions_with_speakers.jsonl"
    save_jsonl(out_df, out_jsonl)
    if args.save_csv:
        out_csv = args.out_dir / "mentions_with_speakers.csv"
        out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    LOGGER.info("Wrote %s (%d rows)", out_jsonl, len(out_df))

    if args.qa_jsonl:
        qa_path = args.out_dir / "speaker_qc.jsonl"
        with qa_path.open("w", encoding="utf-8") as f:
            for gid, g in out_df.groupby("granuleId"):
                counts = g["speaker_method"].value_counts(dropna=False).to_dict()
                spans = spans_cache.get(str(gid), [])
                span_snap = [
                    {"start": getattr(s, 'start', None), "end": getattr(s, 'end', None), "raw": getattr(s, 'raw_label', None), "canon": getattr(s, 'canonical_name', None)}
                    for s in spans[:5]
                ]
                first_cue = spans[0].raw_label if spans else None
                last_cue = spans[-1].raw_label if spans else None
                common_src = source_by_gid.get(str(gid))
                f.write(json.dumps({
                    "granuleId": gid,
                    "counts": counts,
                    "n": int(len(g)),
                    "spans_sample": span_snap,
                    "n_spans": len(spans),
                    "first_cue": first_cue,
                    "last_cue": last_cue,
                    "text_source": common_src,
                }, ensure_ascii=False) + "\n")
        LOGGER.info("Wrote QA summary to %s", qa_path)


if __name__ == "__main__":
    main()
