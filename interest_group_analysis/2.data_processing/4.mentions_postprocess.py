#!/usr/bin/env python3
"""
Postprocess matcher output so the analysis unit is (org_id × paragraph).

Outputs:
- analytic_paragraph_units.(jsonl|csv): one row per (org_id × paragraph)
- diagnostics_mentions_expanded.(jsonl|csv): optional per-mention rows for QA

Behavior:
- Clean paragraph prefixes (configurable).
- Dedupe on (org_id, cleaned_paragraph).
- Aggregate all variations for the group within the paragraph.
- Count mentions without inflating rows; per-mention indices only in diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
import uuid
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm

# ---------------------------- I/O utils ---------------------------- #

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # skip bad line but keep going
                continue


def write_jsonl(df: pd.DataFrame, path: Path):
    with path.open('w', encoding='utf-8') as fh:
        for rec in df.to_dict(orient='records'):
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')


# -------------------------- Cleaning / matching -------------------------- #

def load_prefixes(path: Optional[Path]) -> List[re.Pattern]:
    patterns: List[re.Pattern] = []
    if not path:
        raw = [r"^.*?www\.gpo\.gov", r"^https?://\S+\s*", r"^\[Page \d+\]\s*"]
        for p in raw:
            patterns.append(re.compile(p, flags=re.IGNORECASE))
        return patterns
    try:
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            patterns.append(re.compile(line, flags=re.IGNORECASE))
    except Exception:
        pass
    return patterns


def clean_paragraph(paragraph: str, prefixes: List[re.Pattern]) -> str:
    if not isinstance(paragraph, str) or not paragraph:
        return paragraph or ""
    for pat in prefixes:
        m = pat.search(paragraph)
        if m:
            return paragraph[m.end():].lstrip()
    return paragraph


def make_pattern(word: str) -> str:
    """Return a pattern string that tolerates variable whitespace between tokens."""
    if not isinstance(word, str):
        word = str(word or "")
    tokens = re.split(r"\s+", word.strip()) if word else []
    if not tokens:
        return r"(?!x)x"  # never matches
    return r"\s+".join(map(re.escape, tokens))


def compile_union(variations: List[str]) -> Optional[re.Pattern]:
    alts = [make_pattern(v) for v in variations if isinstance(v, str) and v.strip()]
    if not alts:
        return None
    return re.compile(r"(?:%s)" % "|".join(alts), flags=re.IGNORECASE)


def ensure_variation(df: pd.DataFrame) -> pd.DataFrame:
    if 'variation' not in df.columns:
        df['variation'] = None
    for cand in ['surface_form', 'alias', 'name', 'canonical_name', 'term', 'needle', 'match_text']:
        if cand in df.columns:
            df['variation'] = df['variation'].fillna(df[cand])
    df['variation'] = df['variation'].astype(str).str.strip()
    df.loc[df['variation'].isin(['', 'nan', 'None', 'NaN']), 'variation'] = None
    return df


def number_mentions(text: str, variations: List[str]):
    """
    Replace every occurrence of any variation with ****(k)match****,
    where k = 1,2,3... in left-to-right order. Returns:
      out_text, count, spans  (spans = [(k, start, end), ...])  -- start/end in original text
    """
    if not isinstance(text, str) or not text or not variations:
        return text, 0, []
    pat = compile_union(variations)
    if pat is None:
        return text, 0, []

    i = 0
    spans = []

    def _repl(m: re.Match):
        nonlocal i
        i += 1
        spans.append((i, m.start(), m.end()))
        return f"****({i}){m.group(0)}****"

    out = pat.sub(_repl, text)
    return out, i, spans


# -------------------------- Core pipeline -------------------------- #

def _first_or_none(series: pd.Series):
    try:
        return next((x for x in series if pd.notna(x)), None)
    except Exception:
        return None


def build_analytic_units(raw: pd.DataFrame, prefix_file: Optional[Path]) -> pd.DataFrame:
    """Return one row per (org_id × cleaned_paragraph)."""
    raw = raw.copy()

    # Preserve or create paragraph UUIDs
    if 'uuid_paragraph' not in raw.columns:
        raw['uuid_paragraph'] = [str(uuid.uuid4()) for _ in range(len(raw))]

    # Ensure variation exists
    raw = ensure_variation(raw)

    # Ensure paragraph text
    if 'paragraph' not in raw.columns:
        raw['paragraph'] = raw.get('sentence', '')
    
    # Ensure sentence exists for aggregation, otherwise reuse paragraph
    if 'sentence' not in raw.columns:
        raw['sentence'] = raw['paragraph']
    
    raw['paragraph_original'] = raw['paragraph']

    # Clean paragraph text
    prefixes = load_prefixes(prefix_file)
    raw['cleaned_paragraph'] = raw['paragraph'].apply(lambda p: clean_paragraph(p or '', prefixes))

    # Check if cleaned_paragraph exists
    if 'cleaned_paragraph' not in raw.columns or raw['cleaned_paragraph'].isnull().all():
        raise ValueError("The 'cleaned_paragraph' column could not be created. Ensure the 'paragraph' column exists in the input data.")

    # ----- group to one row per (org_id × cleaned_paragraph)
    key_cols = ['org_id', 'cleaned_paragraph']
    grouped = (
        raw.groupby(key_cols, dropna=False)
           .agg(
               uuid_paragraphs=('uuid_paragraph', lambda s: list(dict.fromkeys(s))),
               variations=('variation', lambda s: sorted({v for v in s if isinstance(v, str) and v.strip()})),
               paragraph=('paragraph_original', _first_or_none),
               packageId=('packageId', _first_or_none) if 'packageId' in raw.columns else ('uuid_paragraph', 'size'),
               granuleId=('granuleId', _first_or_none) if 'granuleId' in raw.columns else ('uuid_paragraph', 'size'),
               date=('date', _first_or_none) if 'date' in raw.columns else ('uuid_paragraph', 'size'),
               title=('title', _first_or_none) if 'title' in raw.columns else ('uuid_paragraph', 'size'),
               sentence_examples=('sentence', lambda s: list(dict.fromkeys([x for x in s if isinstance(x, str)]))[:3])
           )
           .reset_index()
    )

    # Stable primary uuid per unit (first of the originals)
    grouped['primary_uuid_paragraph'] = grouped['uuid_paragraphs'].apply(lambda xs: xs[0] if xs else None)

    # Numbered and all-highlight views, plus counts
    out_rows = []
    for _, r in tqdm(grouped.iterrows(), total=len(grouped), desc="Building analytic units"):
        para_text = r['paragraph'] or ""
        clean_text = r['cleaned_paragraph'] or ""
        vars_ = r['variations'] or []

        para_num, n_raw, _ = number_mentions(para_text, vars_)
        clean_num, n_clean, _ = number_mentions(clean_text, vars_)

        rec = {
            'org_id': r['org_id'],
            'primary_uuid_paragraph': r['primary_uuid_paragraph'],
            'uuid_paragraphs': r['uuid_paragraphs'],
            'paragraph': para_text,
            'cleaned_paragraph': clean_text,
            'variations': vars_,
            # analysis count: do NOT multiply rows
            'mentions_n': int(max(n_clean, n_raw)),
            # QA strings with numbering
            'paragraph_numbered': para_num,
            'cleaned_paragraph_numbered': clean_num,
            # optional “bold-only” views (no indices)
            'paragraph_highlighted': re.sub(r"\(\d+\)", "", para_num),
            'cleaned_paragraph_highlighted': re.sub(r"\(\d+\)", "", clean_num),
            'sentence_examples': r['sentence_examples'],
        }
        for col in ['packageId', 'granuleId', 'date', 'title']:
            if col in grouped.columns:
                rec[col] = r[col]
        out_rows.append(rec)

    analytic = pd.DataFrame(out_rows)
    # one row per (org_id × paragraph)
    return analytic


def build_diagnostics_mentions(raw: pd.DataFrame,
                               analytic: pd.DataFrame,
                               prefix_file: Optional[Path]) -> pd.DataFrame:
    """Optional: explode to per-mention rows for QA (indices (1..k) & context strings)."""
    raw = raw.copy()
    raw = ensure_variation(raw)

    # ensure base text columns
    if 'uuid_paragraph' not in raw.columns:
        raw['uuid_paragraph'] = [str(uuid.uuid4()) for _ in range(len(raw))]
    if 'paragraph' not in raw.columns:
        raw['paragraph'] = raw.get('sentence', '')

    # Build cleaned_paragraph exactly like in analytic builder
    prefixes = load_prefixes(prefix_file)
    raw['cleaned_paragraph'] = raw['paragraph'].apply(lambda p: clean_paragraph(p or '', prefixes))

    # Join to harmonize cleaned text and variations per (org_id × cleaned_paragraph)
    j = raw.merge(
        analytic[['org_id', 'cleaned_paragraph', 'primary_uuid_paragraph', 'variations', 'paragraph', 'packageId', 'granuleId']],
        on=['org_id', 'cleaned_paragraph'],
        how='inner',
        suffixes=('', '_unit')
    )
    if j.empty:
        print("Diagnostics join produced 0 rows. Check that org_id and cleaned_paragraph align between raw and analytic.")
        return pd.DataFrame(columns=[
            'org_id','primary_uuid_paragraph','uuid_mention','mention_index',
            'variations','paragraph','cleaned_paragraph',
            'paragraph_mention','cleaned_paragraph_mention'
        ])

    def explode_mentions(row, which: str):
        text = row['cleaned_paragraph'] if which == 'cleaned_paragraph' else row['paragraph']
        out = []
        numbered, count, spans = number_mentions(text or "", row['variations'] or [])
        for i, start, end in spans:
            rec = {
                'org_id': row['org_id'],
                'primary_uuid_paragraph': row['primary_uuid_paragraph'],
                'uuid_mention': f"{row['primary_uuid_paragraph']}:{i}",
                'mention_index': i,
                'variations': row['variations'],
                'paragraph': row['paragraph'],
                'cleaned_paragraph': row['cleaned_paragraph'],
                f'{which}_mention': numbered
            }
            out.append(rec)
        return out

    rows: List[Dict[str, Any]] = []
    for _, r in tqdm(j.iterrows(), total=len(j), desc="Building diagnostics (mentions)"):
        rows += explode_mentions(r, 'paragraph')
        rows += explode_mentions(r, 'cleaned_paragraph')

    if not rows:
        return pd.DataFrame(columns=[
            'org_id','primary_uuid_paragraph','uuid_mention','mention_index',
            'variations','paragraph','cleaned_paragraph',
            'paragraph_mention','cleaned_paragraph_mention'
        ])

    diag = pd.DataFrame(rows)
    for col in ['paragraph_mention', 'cleaned_paragraph_mention']:
        if col not in diag.columns:
            diag[col] = ""

    # Convert unhashable columns to hashable types before deduplication
    diag['variations'] = diag['variations'].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    diag = diag[
        [
            'org_id', 'primary_uuid_paragraph', 'uuid_mention', 'mention_index',
            'variations', 'paragraph', 'cleaned_paragraph',
            'paragraph_mention', 'cleaned_paragraph_mention'
        ]
    ].drop_duplicates()
    return diag


# Add helper functions for mention enumeration and window processing
import hashlib, unicodedata

# Normalize text for hashing
def normalize_for_hash(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s.strip())


# Regular expression for sentence splitting
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences_with_spans(text: str):
    sents, spans = [], []
    start = 0
    for m in SENT_SPLIT_RE.finditer(text):
        end = m.start() + 1
        sents.append(text[start:end]); spans.append((start, end))
        start = m.end()
    if start < len(text):
        sents.append(text[start:]); spans.append((start, len(text)))
    return [(s, a, b) for (s, (a, b)) in zip(sents, spans)]


# Generate stable window ID
def stable_window_id(org_id: str, source_block_id: str, window_text: str) -> str:
    key = f"{org_id}||{source_block_id}||{normalize_for_hash(window_text)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


# Generate stable mention ID
def stable_mention_id(org_id: str, source_block_id: str, para_uuid: str, start: int, end: int) -> str:
    key = f"{org_id}||{source_block_id}||{para_uuid}||{start}||{end}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


# Merge overlapping ranges
def merge_overlapping_ranges(ranges):
    if not ranges: return []
    ranges = sorted(ranges, key=lambda x: (x[0], x[1]))
    merged = [ranges[0]]
    for lo, hi in ranges[1:]:
        mlo, mhi = merged[-1]
        if lo <= mhi:
            merged[-1] = (mlo, max(mhi, hi))
        else:
            merged.append((lo, hi))
    return merged


# Compile variations into a regex pattern (simple version for window processing)
def compile_union_simple(variations: list[str]) -> Optional[re.Pattern]:
    alts = [re.escape(v) for v in variations if isinstance(v, str) and v.strip()]
    if not alts:
        return None
    return re.compile(r"(?:%s)" % "|".join(alts), flags=re.IGNORECASE)


# Enumerate mentions in a paragraph
def enumerate_mentions(paragraph: str, variations: list[str], sent_spans: list[tuple[str,int,int]]):
    out = []
    if not paragraph or not variations:
        return out
    pat = compile_union_simple(variations)
    if pat is None:
        return out
    idx = 0
    for m in pat.finditer(paragraph):
        idx += 1
        start, end = m.start(), m.end()
        sent_idx = next((i for i, (_, a, b) in enumerate(sent_spans) if a <= start < b), -1)
        out.append({
            "mention_index": idx,
            "start": start,
            "end": end,
            "sent_idx": sent_idx,
            "variation": m.group(0),
            "matched_text": paragraph[start:end],
        })
    return out


# Mask organization mentions in text
def mask_org_mentions(text: str, forms: list[str]) -> str:
    forms = sorted({f for f in forms if f}, key=len, reverse=True)
    for f in forms:
        text = re.sub(rf"\b{re.escape(f)}\b", "[ORG]", text, flags=re.IGNORECASE)
    return text


# Build windows with mentions
def build_windows_with_mentions(raw: pd.DataFrame, prefix_file: Optional[Path], context_radius: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw.copy()
    if 'paragraph' not in df.columns:
        df['paragraph'] = df.get('sentence', '')

    if 'uuid_paragraph' not in df.columns:
        df['uuid_paragraph'] = [str(uuid.uuid4()) for _ in range(len(df))]
    
    # Treat each paragraph as its own block
    df['source_block_id'] = df['uuid_paragraph']

    prefixes = load_prefixes(prefix_file)
    df['cleaned_paragraph'] = df['paragraph'].apply(lambda p: clean_paragraph(p or '', prefixes))
    df = ensure_variation(df)

    block_sents = (df.groupby('source_block_id')['paragraph'].first()
                     .apply(lambda t: split_sentences_with_spans(str(t))).to_dict())
    block_para = df.groupby('source_block_id')['paragraph'].first().to_dict()
    block_para_uuid = df.groupby('source_block_id')['uuid_paragraph'].first().to_dict()

    windows_rows, mention_rows = [], []

    for (org, blk), g in df.groupby(['org_id','source_block_id'], dropna=False):
        para = block_para.get(blk, "")
        sent_spans = block_sents.get(blk, [])
        if not para or not sent_spans:
            continue

        block_vars = sorted({str(v) for v in g['variation'].dropna().astype(str)})
        mentions = enumerate_mentions(para, block_vars, sent_spans)
        if not mentions:
            continue

        n_sent = len(sent_spans)
        ranges = []
        for m in mentions:
            si = max(0, min(n_sent-1, int(m['sent_idx'])))
            ranges.append((max(0, si-context_radius), min(n_sent-1, si+context_radius)))
        merged = merge_overlapping_ranges(ranges)

        # Updated `build_windows_with_mentions` to include numbered/highlighted windows
        for lo, hi in merged:
            # 1) Build window by slicing the ORIGINAL paragraph to preserve char offsets
            win_start = sent_spans[lo][1]   # start char of first sentence in window
            win_end   = sent_spans[hi][2]   # end char of last sentence in window
            window_text = para[win_start:win_end]

            # 2) Mentions inside this window (by sentence index)
            m_inside = [m for m in mentions if lo <= m['sent_idx'] <= hi]
            m_vars   = sorted({m['variation'] for m in m_inside if m.get('variation')})
            m_indices = [m['mention_index'] for m in m_inside]

            # 3) Build a numbered version of the window using the paragraph-level indices
            numbered = window_text
            repls = []
            for m in m_inside:
                s_local = max(0, m['start'] - win_start)
                e_local = max(0, m['end']   - win_start)
                if 0 <= s_local < e_local <= len(window_text):
                    repls.append((s_local, e_local, f"****({m['mention_index']}){window_text[s_local:e_local]}****"))

            for s_local, e_local, rep in sorted(repls, key=lambda x: x[0], reverse=True):
                numbered = numbered[:s_local] + rep + numbered[e_local:]

            # 4) Highlighted (no indices) version
            highlighted = re.sub(r"\(\d+\)", "", numbered)

            # 5) Masked version for modeling
            masked = mask_org_mentions(window_text, m_vars or block_vars)

            # 6) Stable window id based on the (normalized) window_text
            wid = stable_window_id(org, blk, window_text)

            # 7) Emit row
            windows_rows.append({
                "window_id": wid,
                "org_id": org,
                "source_block_id": blk,
                "paragraph_uuid": block_para_uuid.get(blk),
                "text_for_labeler": window_text,
                "text_for_model": masked,
                "window_numbered": numbered,          # <-- NEW
                "window_highlighted": highlighted,    # <-- NEW
                "variations_in_window": m_vars or block_vars,
                "window_lo_idx": lo,
                "window_hi_idx": hi,
                "sentences_in_window": (hi - lo + 1),
                "mentions_in_window_n": len(m_inside),
                "mention_indices_in_window": m_indices,
            })

    windows_df = pd.DataFrame(windows_rows).drop_duplicates(subset=["window_id"]).reset_index(drop=True)
    return windows_df, pd.DataFrame()


def run(input_jsonl: Path,
        out_dir: Path,
        prefix_file: Optional[Path],
        save_csv: bool,
        save_diagnostics: bool):

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_records = list(read_jsonl(input_jsonl))
    if not raw_records:
        print(f"No records in {input_jsonl}")
        return
    raw = pd.DataFrame(raw_records)

    # Build analytic units (one row per group × paragraph)
    analytic = build_analytic_units(raw, prefix_file)

    # Save analytic outputs
    analytic_jsonl = out_dir / "analytic_paragraph_units.jsonl"
    write_jsonl(analytic, analytic_jsonl)
    if save_csv:
        analytic_csv = out_dir / "analytic_paragraph_units.csv"
        analytic.to_csv(analytic_csv, index=False, quoting=csv.QUOTE_MINIMAL, encoding='utf-8-sig')
    print(f"Wrote analytic units to {analytic_jsonl}")

    # Optional diagnostics (per-mention)
    if save_diagnostics:
        diagnostics = build_diagnostics_mentions(raw, analytic, prefix_file)
        diag_jsonl = out_dir / "diagnostics_mentions_expanded.jsonl"
        write_jsonl(diagnostics, diag_jsonl)
        if save_csv:
            diag_csv = out_dir / "diagnostics_mentions_expanded.csv"
            diagnostics.to_csv(diag_csv, index=False, quoting=csv.QUOTE_MINIMAL, encoding='utf-8-sig')
        print(f"Wrote diagnostics to {diag_jsonl}")

    windows_df, _ = build_windows_with_mentions(raw, prefix_file, context_radius=3)

    win_path = out_dir / "analytic_windows.jsonl"
    write_jsonl(windows_df, win_path)
    if save_csv:
        windows_df.to_csv(out_dir / "analytic_windows.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote merged windows to {win_path}")

    # Removed `windows_mentions.*` output in `run`
    # Commented out the block that writes `windows_mentions.jsonl` and `.csv`
    # if not win_mentions_df.empty:
    #     winm_path = out_dir / "windows_mentions.jsonl"
    #     write_jsonl(win_mentions_df, winm_path)
    #     if save_csv:
    #         win_mentions_df.to_csv(out_dir / "windows_mentions.csv", index=False, encoding='utf-8-sig')
    #     print(f"Wrote window-mention map to {winm_path}")


# ---------------------------- CLI ---------------------------- #

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="Produce paragraph-level units (org × paragraph) and optional per-mention diagnostics."
    )
    ap.add_argument('--input-jsonl', required=True, help='Matcher output (JSONL: one record per line)')
    ap.add_argument('--out-dir', required=True, help='Output directory')
    ap.add_argument('--clean-prefix-file', default=None, help='Regex lines to strip leading boilerplate')
    ap.add_argument('--save-csv', action='store_true', help='Also write CSV beside JSONL')
    ap.add_argument('--save-diagnostics', action='store_true', help='Write per-mention diagnostics tables')
    args = ap.parse_args()

    run(
        input_jsonl=Path(args.input_jsonl),
        out_dir=Path(args.out_dir),
        prefix_file=Path(args.clean_prefix_file) if args.clean_prefix_file else None,
        save_csv=args.save_csv,
        save_diagnostics=args.save_diagnostics
    )
