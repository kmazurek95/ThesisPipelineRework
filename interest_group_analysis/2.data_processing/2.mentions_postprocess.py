"""Postprocess mention JSONL produced by the matcher into expanded, deduped, highlighted outputs.

Features:
- paragraph cleaning (prefix removal) with original/clean fields
- paragraph-level deduplication and overlap aggregation
- per-mention row generation with uuid_mention and mention_index
- in-text highlighting (sentence and paragraph multi-highlight)
- CSV export option

This is intentionally separate from the matcher to keep responsibilities small and make re-runs safe.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

import pandas as pd


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def load_prefixes(path: Optional[Path]) -> List[re.Pattern]:
    patterns: List[re.Pattern] = []
    if not path:
        # default builtins
        raw = [r"^.*?www\\.gpo\\.gov", r"^https?://\\S+\\s*", r"^\[Page \d+\]\\s*"]
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
    if not paragraph:
        return paragraph
    for pat in prefixes:
        m = pat.search(paragraph)
        if m:
            # remove everything up to and including the match
            return paragraph[m.end():].lstrip()
    return paragraph


def highlight_by_spans(text: str, spans: List[Dict[str, Any]], primary_marker: str, secondary_marker: str) -> str:
    """Replace spans in text with markers. Spans list must contain dicts with start,end,primary(bool)."""
    if not spans:
        return text
    # sort spans by start descending so replacements don't shift earlier indices
    spans_sorted = sorted(spans, key=lambda s: s['start'], reverse=True)
    out = text
    for s in spans_sorted:
        start = s['start']
        end = s['end']
        marker = primary_marker if s.get('primary') else secondary_marker
        out = out[:start] + marker + out[start:end] + marker + out[end:]
    return out


def find_non_overlapping_occurrences(text: str, pattern: str) -> List[re.Match]:
    # finditer will produce non-overlapping matches naturally
    return list(re.finditer(pattern, text, flags=re.IGNORECASE))


def process_mentions_jsonl(
    input_jsonl: Path,
    out_dir: Path,
    save_csv: bool = False,
    csv_path: Optional[Path] = None,
    clean_prefix_file: Optional[Path] = None,
    use_clean: bool = True,
    multi_highlight: bool = True,
    primary_marker: str = '****',
    secondary_marker: str = '*****',
):
    out_dir.mkdir(parents=True, exist_ok=True)
    prefixes = load_prefixes(clean_prefix_file)

    records = list(read_jsonl(input_jsonl))
    if not records:
        print('No records found in', input_jsonl)
        return

    df = pd.DataFrame(records)

    # ensure uuid_paragraph exists per row (some matchers may not provide it)
    if 'uuid_paragraph' not in df.columns:
        df['uuid_paragraph'] = [str(uuid.uuid4()) for _ in range(len(df))]

    # keep original paragraph text fields if present
    if 'paragraph' not in df.columns:
        # attempt to assemble paragraph from sentence or sentence context
        df['paragraph'] = df.get('sentence', '')

    # store original paragraph
    df['paragraph_original'] = df['paragraph']

    # cleaning
    if use_clean:
        df['paragraph_clean'] = df['paragraph'].apply(lambda p: clean_paragraph(p or '', prefixes))
    else:
        df['paragraph_clean'] = df['paragraph']

    # group by org_id + paragraph_clean to dedupe and aggregate overlap ids
    group_cols = []
    if 'org_id' in df.columns:
        group_cols.append('org_id')
    group_cols.append('paragraph_clean')

    grouped = df.groupby(group_cols).agg({'uuid_paragraph': lambda s: list(dict.fromkeys(s)), 'sentence': lambda s: list(dict.fromkeys(s))}).reset_index()
    grouped = grouped.rename(columns={'uuid_paragraph': 'overlap_ids', 'sentence': 'sample_sentences'})
    grouped['overlap_count'] = grouped['overlap_ids'].apply(len)
    # choose primary uuid_paragraph
    grouped['primary_uuid_paragraph'] = grouped['overlap_ids'].apply(lambda ids: ids[0] if ids else None)

    # merge aggregated info back to df on org_id + paragraph_clean
    df = df.merge(grouped[[*group_cols, 'overlap_ids', 'overlap_count', 'primary_uuid_paragraph']], on=group_cols, how='left')

    # expand to per-mention rows (some rows may already be per-mention, ensure unique mention ids)
    expanded_rows: List[Dict[str, Any]] = []
    # we'll assign mention_index per paragraph (by order of occurrence in sentence if possible)
    by_para = df.groupby(['primary_uuid_paragraph'])
    for para_id, sub in by_para:
        # order sub by sentence_index and start_in_sentence if available
        sub = sub.copy()
        if 'sentence_index' in sub.columns and 'start_in_sentence' in sub.columns:
            sub = sub.sort_values(['sentence_index', 'start_in_sentence'], na_position='last')
        else:
            sub = sub.sort_values('timestamp', na_position='last')
        for idx, (_, row) in enumerate(sub.iterrows(), start=1):
            uuid_par = row.get('uuid_paragraph') or str(uuid.uuid4())
            uuid_mention = f"{uuid_par}:{idx}"
            mention_index = idx
            # prepare highlighted sentence
            highlighted_sentence = None
            sent = row.get('sentence') or ''
            if isinstance(row.get('start_in_sentence'), (int, float)) and isinstance(row.get('end_in_sentence'), (int, float)):
                try:
                    s = int(row.get('start_in_sentence'))
                    e = int(row.get('end_in_sentence'))
                    highlighted_sentence = (sent[:s] + primary_marker + sent[s:e] + primary_marker + sent[e:])
                except Exception:
                    highlighted_sentence = re.sub(re.escape(str(row.get('variation') or '')), primary_marker + (row.get('variation') or '') + primary_marker, sent, flags=re.IGNORECASE)
            else:
                # best-effort: replace first occurrence of variation in sentence
                var = row.get('variation') or ''
                if var:
                    try:
                        highlighted_sentence = re.sub(re.escape(var), primary_marker + var + primary_marker, sent, count=1, flags=re.IGNORECASE)
                    except Exception:
                        highlighted_sentence = sent
                else:
                    highlighted_sentence = sent

            expanded = dict(row)
            expanded.update({
                'uuid_paragraph': uuid_par,
                'uuid_mention': uuid_mention,
                'mention_index': mention_index,
                'highlighted_sentence': highlighted_sentence,
            })
            expanded_rows.append(expanded)

    df_expanded = pd.DataFrame(expanded_rows)

    # paragraph multi-highlight: for each paragraph, find all mention variations and mark primary vs secondary
    paragraph_multi: Dict[str, str] = {}
    for para_id, sub in df_expanded.groupby('primary_uuid_paragraph'):
        paragraph_text = (sub['paragraph_clean'].iloc[0]) if 'paragraph_clean' in sub.columns else ''
        if not paragraph_text:
            paragraph_multi[para_id] = ''
            continue
        # build spans: find occurrences for each mention record in this paragraph
        spans: List[Dict[str, Any]] = []
        # we'll find all occurrences for each variation, then mark first as primary
        for _, r in sub.iterrows():
            var = r.get('variation') or ''
            if not var:
                continue
            try:
                occs = list(re.finditer(re.escape(var), paragraph_text, flags=re.IGNORECASE))
            except Exception:
                occs = []
            for i, m in enumerate(occs):
                spans.append({'start': m.start(), 'end': m.end(), 'variation': var, 'primary': i == 0})
        # collapse overlapping spans by keeping earliest primary choice
        if not spans:
            paragraph_multi[para_id] = paragraph_text
            continue
        # mark primary as first occurrence of each variation; secondary for others
        # if spans overlap, we'll still do replacements from end->start
        paragraph_multi[para_id] = highlight_by_spans(paragraph_text, spans, primary_marker, secondary_marker)

    # attach paragraph_multi_highlighted to df_expanded
    df_expanded['paragraph_multi_highlighted'] = df_expanded['primary_uuid_paragraph'].map(paragraph_multi)

    # save outputs
    mentions_out = out_dir / 'mentions_expanded.jsonl'
    paragraphs_out = out_dir / 'paragraphs_agg.jsonl'
    with mentions_out.open('w', encoding='utf-8') as fh:
        for rec in df_expanded.to_dict(orient='records'):
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')

    with paragraphs_out.open('w', encoding='utf-8') as pf:
        for rec in grouped.to_dict(orient='records'):
            pf.write(json.dumps(rec, ensure_ascii=False) + '\n')

    if save_csv:
        csv_target = csv_path or (out_dir / 'mentions_expanded.csv')
        # ensure proper quoting
        df_expanded.to_csv(str(csv_target), index=False, quoting=csv.QUOTE_MINIMAL, encoding='utf-8-sig')

    print('Wrote', mentions_out, 'and', paragraphs_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Postprocess mention JSONL into expanded/deduped highlights and CSV')
    parser.add_argument('--input-jsonl', required=True, help='Input mentions.jsonl produced by matcher')
    parser.add_argument('--out-dir', default='data/processed/mentions', help='Output directory')
    parser.add_argument('--save-csv', action='store_true', help='Also save CSV of expanded mentions')
    parser.add_argument('--csv-path', default=None, help='Path to save CSV')
    parser.add_argument('--clean-prefix-file', default=None, help='Path to file with prefix regex lines to remove')
    parser.add_argument('--no-clean', dest='use_clean', action='store_false', help='Do not apply paragraph cleaning')
    parser.add_argument('--no-multi-highlight', dest='multi_highlight', action='store_false', help='Do not produce multi-highlighted paragraph')
    parser.add_argument('--primary-marker', default='****', help='Primary highlight marker')
    parser.add_argument('--secondary-marker', default='*****', help='Secondary highlight marker')
    args = parser.parse_args()

    process_mentions_jsonl(Path(args.input_jsonl), Path(args.out_dir), save_csv=args.save_csv, csv_path=Path(args.csv_path) if args.csv_path else None, clean_prefix_file=Path(args.clean_prefix_file) if args.clean_prefix_file else None, use_clean=args.use_clean, multi_highlight=args.multi_highlight, primary_marker=args.primary_marker, secondary_marker=args.secondary_marker)
