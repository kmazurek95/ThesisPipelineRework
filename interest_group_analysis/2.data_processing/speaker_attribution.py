#!/usr/bin/env python3
r"""
Speaker attribution primitives

This module provides the core functions to detect and segment speaker turns in
Congressional Record text and to map those turns to canonical member identities.

What it does:
- Defines regexes to find speaker cues (e.g., "Mr. SMITH.", "The PRESIDING OFFICER:").
- Builds a lightweight member index from a members table for last‑name matching.
- Produces SpeakerSpan segments that cover text ranges between cues.
- Assigns a speaker to an arbitrary character offset within the text.

Primary functions:
- build_member_patterns(df_members) -> Dict: prepare candidates bucketed by last name.
- iter_speaker_spans(text, members_idx) -> List[SpeakerSpan]: segment text by cues.
- assign_speaker_for_offset(offset, spans, granule_single_speaker=None) -> tuple.

How to use (normally via the CLI that calls this module):

        # Attach speakers to extracted mentions
        python .\interest_group_analysis\2.data_processing\3.attach_speakers.py `
            --mentions-jsonl data\processed\mentions_114_run2\mentions.jsonl `
            --normalized-dir data\normalized\normalized_114_run2 `
            --out-dir data\processed\mentions_and_speaker_114 `
            --save-csv `
            --qa-jsonl

Note: This file is a library (no CLI). Import its functions or use the
3.attach_speakers.py script shown above.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# --- Compile once ---
HONORIFICS = r"(Mr|Mrs|Ms|Miss|Mx|Dr|Senator|Sen|Representative|Rep)\."
# Role cues can appear as e.g. "The PRESIDING OFFICER.", "The SPEAKER:"
# Also include address-style role cues like "Mr. Speaker:" or "Madam President."
ROLES = r"(?:The\s+(?:SPEAKER(?:\s+pro\s+tempore)?|PRESIDING\s+OFFICER|CHAIR|ACTING\s+PRESIDENT\s+pro\s+tempore)|(?:Mr\.\s+Speaker|Mr\.\s+President|Madam\s+Speaker|Madam\s+President))"
# Examples: "Mr. SMITH.", "Ms. JOHNSON of California.", "The PRESIDING OFFICER:", "Madam Speaker:"
SENTENCE_END_PUNCT = r"[\.:]"
# fixed-width lookbehinds; whitespace consumed outside
BOUNDARY = r"(?:(?<=^)|(?<=\n)|(?<=\.)|(?<=\?)|(?<=!)|(?<=\)))\s*"
# Note: We compile with IGNORECASE so roles/honorifics are case-insensitive, but we wrap
# the captured NAME group in (?-i:...) to still require uppercase surnames on cue lines.
SPEAKER_CUE_RE = re.compile(
    rf"{BOUNDARY}(?:{HONORIFICS}\s+(?P<name>(?-i:[A-Z][A-Z\-\.'\s]+?))(?:\s+of\s+[A-Z][A-Za-z\.\s]+)?{SENTENCE_END_PUNCT})"
    rf"|{BOUNDARY}(?P<role>{ROLES}){SENTENCE_END_PUNCT}",
    flags=re.IGNORECASE,
)

# Salutation like "Mr. Speaker," inside a speech—not a turn cue
IN_SPEECH_SALUTATION_RE = re.compile(r"\bMr\.?\s+Speaker,|\bMadam\s+Speaker,|\bMr\.?\s+President,", re.I)

@dataclass
class SpeakerSpan:
    start: int
    end: int
    raw_label: str                  # e.g., "Mr. SMITH."
    canonical_name: Optional[str]   # e.g., "Steve Scalise"
    bioguide_id: Optional[str]      # if resolvable


def build_member_patterns(df_members) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    if df_members is None or getattr(df_members, 'empty', False):
        return out
    for _, r in df_members.iterrows():
        last = str(r.get("last_name", "")).upper().strip()
        if not last:
            continue
        cand = {
            "bioguide_id": r.get("bioguide_id"),
            "first": str(r.get("first_name", "")).upper().strip(),
            "last": last,
            "state": str(r.get("state", "")).upper().strip(),
            "chamber": str(r.get("chamber", "")).strip(),
            "party": r.get("party"),
        }
        out.setdefault(last, {"candidates": []})["candidates"].append(cand)
    return out


def canonicalize_from_raw(raw_label: str, members_idx: Dict[str, Dict]) -> Tuple[Optional[str], Optional[str]]:
    rl = raw_label.strip().strip(".:")
    if rl.upper().startswith("THE "):
        return ("PRESIDING_OFFICER", None)
    m = re.search(r"\b([A-Z][A-Z\-\']+)(?:\s+of\s+[A-Za-z\.\s]+)?$", rl)
    if not m:
        return (None, None)
    last = m.group(1).upper()
    bucket = members_idx.get(last)
    if not bucket:
        return (None, None)
    cands = bucket["candidates"]
    if len(cands) == 1:
        c = cands[0]
        name = f"{c['first'].title()} {c['last'].title()}".strip()
        return (name, c.get("bioguide_id"))
    return (None, None)


def iter_speaker_spans(text: str, members_idx: Dict[str, Dict]) -> List[SpeakerSpan]:
    spans: List[SpeakerSpan] = []
    matches = list(SPEAKER_CUE_RE.finditer(text or ""))
    if not matches:
        return spans
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw = m.group(0).strip()
        canon, bioguide = canonicalize_from_raw(raw, members_idx)
        spans.append(SpeakerSpan(start=start, end=end, raw_label=raw, canonical_name=canon, bioguide_id=bioguide))
    return spans

def assign_speaker_for_offset(offset: int, speaker_spans: List[SpeakerSpan], granule_single_speaker: Optional[str] = None):
    for s in speaker_spans:
        if s.start <= offset < s.end:
            method = "segment"
            conf = 1.0 if (s.canonical_name or s.raw_label) else 0.8
            return s, method, conf
    PREV_WINDOW = 2000
    prev = [s for s in speaker_spans if s.start <= offset and (offset - s.start) <= PREV_WINDOW]
    if prev:
        s = prev[-1]
        return s, "preceding", 0.6
    if granule_single_speaker:
        pseudo = SpeakerSpan(start=0, end=10**12, raw_label=granule_single_speaker, canonical_name=granule_single_speaker, bioguide_id=None)
        return pseudo, "single_speaker", 0.4
    return SpeakerSpan(0, 0, "UNKNOWN", None, None), "unknown", 0.0
