# 3.mention_extraction.py
# Mention extraction with strict matching, incremental processing, and JSONL outputs.

from __future__ import annotations

import json
import re
import logging
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any

import pandas as pd
from tqdm import tqdm

# Sentence tokenizer
try:
    import nltk
    nltk.data.find("tokenizers/punkt")
except Exception:
    try:
        import nltk
        nltk.download("punkt", quiet=True)
    except Exception:
        nltk = None

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class InterestGroup:
    org_id: str
    canonical_name: str
    alt_names: List[str]
    acronyms: List[str]


# Stop words and filters
STOP_WORDS = {"the", "a", "an", "of", "and", "for", "to", "in", "on", "at"}

# Very common single words that spur false positives if treated as org names
COMMON_SINGLE_WORDS = {
    "oregon", "washington", "california", "texas", "florida", "new", "york",
    "tax", "taxation", "energy", "transportation", "education", "health",
    "business", "agriculture", "farm", "union", "council", "committee",
    "association", "institute", "foundation", "coalition", "alliance",
    # extra guards to reduce generic matches when present as lone 'names'
    "leadership", "secretary", "tourism", "ability", "america", "states",
    "now", "cars", "era", "all", "who", "air", "big", "rise",
}

# Common multiword phrases that should never be treated as organization names
BLOCKLIST_PHRASES = {s.lower() for s in [
    "United States",
    "White House",
]}

# If you truly have legit single-word orgs, put them here (lowercased)
WHITELIST_SINGLE_WORDS = {
    "aei", "opec", "nato", "amnesty", "greenpeace", "emily’s list", "emily's list"
}


def normalize_name_piece(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    m = re.match(r"^(.*?),\s*(The)$", s, flags=re.IGNORECASE)
    if m:
        core, _ = m.groups()
        s = f"The {core.strip()}"
    return re.sub(r"\s+", " ", s)


def keep_name(s: str) -> bool:
    if not s:
        return False
    sl = s.strip().lower()
    if sl in STOP_WORDS or sl in BLOCKLIST_PHRASES:
        return False
    # Require 2+ tokens; allow single-word only if whitelisted
    tokens = re.findall(r"[A-Za-z0-9]+", s)
    if len(tokens) < 2:
        return sl in WHITELIST_SINGLE_WORDS
    return True


def keep_acronym(s: str) -> bool:
    if not s:
        return False
    # Allow typical acronym tokens like "UN", "AFL-CIO", "U.N.", "OAS"
    return re.fullmatch(r"[A-Za-z][A-Za-z&.\-]{1,9}", s) is not None


DEFAULT_INTEREST_CSV = Path(r"data/interest_groups_list.csv")


def load_interest_groups(csv_path: Path, *_, **__) -> List[InterestGroup]:
    """
    Load interest groups from a 3-column CSV produced by interest_group_prep.py:
      - org_id
      - interest_group  (the canonical name to match)
      - acronym         (may be blank; may contain ';' or '|' separated values)
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_exc = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, encoding=enc).fillna("")
            break
        except Exception as exc:
            last_exc = exc
    else:
        raise last_exc

    required = {"org_id", "interest_group", "acronym"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Interest CSV missing columns: {missing}. Found: {list(df.columns)}")

    groups: List[InterestGroup] = []
    for _, r in df.iterrows():
        gid = (r.get("org_id") or "").strip() or str(uuid.uuid4())

        name = normalize_name_piece((r.get("interest_group") or "").strip())
        canonical = name if keep_name(name) else gid

        acr_raw = (r.get("acronym") or "").strip()
        acrs: List[str] = []
        if acr_raw:
            # Support multiple acronyms separated by ';' or '|'
            for part in re.split(r"[;|]", acr_raw):
                p = part.strip()
                if p and keep_acronym(p):
                    acrs.append(p)

        groups.append(InterestGroup(
            org_id=gid,
            canonical_name=canonical,
            alt_names=[],       # no alts in the new schema
            acronyms=list(dict.fromkeys(acrs)),
        ))

    LOGGER.info("Loaded %d interest groups from %s (columns = org_id, interest_group, acronym)",
                len(groups), csv_path)
    return groups


def _sentence_tokenize(text: str) -> List[str]:
    if not text:
        return []
    if "nltk" in globals() and nltk is not None:
        try:
            return nltk.tokenize.sent_tokenize(text)
        except Exception:
            pass
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def custom_word_boundary(text: str, acronym: bool = False) -> str:
    if acronym:
        # strict token boundaries, case-sensitive by default (set flags at compile time)
        return r'(?:(?<=\s)|(?<=^)|(?<=\W))' + re.escape(text) + r'(?:(?=\s)|(?=$)|(?=\W))'
    else:
        # case-insensitive whole word
        return r'(?i)\b' + re.escape(text) + r'\b'


def _make_name_pattern(name: str) -> re.Pattern:
    return re.compile(custom_word_boundary(name, acronym=False))


def _make_acronym_pattern(acr: str) -> re.Pattern:
    # Always case-sensitive for acronyms to prevent false positives like now/NOW
    return re.compile(custom_word_boundary(acr, acronym=True))


def _read_jsonl_or_json(file_path: Path) -> Iterable[dict]:
    # CSV handled separately upstream; this is line-delimited JSON or JSON arrays (optionally .gz)
    open_fn = open
    if file_path.suffix.lower().endswith('.gz'):
        import gzip
        def open_fn(p, mode='rt', encoding='utf-8'):
            return gzip.open(p, mode='rt', encoding=encoding)

    with open_fn(file_path, 'r', encoding='utf-8', errors='ignore') as fh:
        first = fh.read(1)
        if not first:
            return
        rest = fh.read()
        content = (first + rest).strip()
        if content.startswith('['):
            try:
                for obj in json.loads(content):
                    yield obj
                return
            except Exception:
                pass
        # fallback: jsonl
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def find_source_files(source_dir: Path) -> List[Path]:
    patterns = ["**/*.json", "**/*.jsonl", "**/*.gz", "**/*.csv"]
    files: List[Path] = []
    for p in patterns:
        files.extend([f for f in source_dir.glob(p) if f.is_file()])
    return sorted(files)


def _safe_make_name_pattern(name: str) -> Optional[re.Pattern]:
    if not keep_name(name):
        return None
    return _make_name_pattern(name)


def process_files(
    source_files: List[Path],
    interest_csv: Path,
    out_dir: Path,
    num_threads: int = 4,
    chunk_size: int = 500,
    resume: bool = True,
    only_canonical_names: bool = False,  # NEW
    strict_current_acronym: bool = False,  # NEW
):
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = load_interest_groups(
        interest_csv,
        strict_current_acronym=strict_current_acronym,
        only_canonical_names=only_canonical_names,
    )

    # Map org_id -> interest_group (canonical_name)
    org_name_by_id: Dict[str, str] = {g.org_id: g.canonical_name for g in groups}

    # Build patterns (names + acronyms only)
    name_patterns: List[Tuple[str, str, re.Pattern]] = []
    acronym_patterns: List[Tuple[str, str, re.Pattern]] = []

    for g in tqdm(groups, desc="Build patterns", unit="org", dynamic_ncols=True, leave=False):
        if g.canonical_name:
            pat = _safe_make_name_pattern(g.canonical_name)
            if pat:
                name_patterns.append((g.org_id, g.canonical_name, pat))
        for acr in g.acronyms:
            acronym_patterns.append((g.org_id, acr, _make_acronym_pattern(acr)))

    LOGGER.info("Pattern summary: names=%d acronyms=%d", len(name_patterns), len(acronym_patterns))

    processed_manifest = out_dir / 'processed_files.jsonl'
    processed_seen: set[str] = set()
    if resume and processed_manifest.exists():
        with processed_manifest.open('r', encoding='utf-8') as mf:
            for line in mf:
                try:
                    obj = json.loads(line)
                    if obj.get('file'):
                        processed_seen.add(obj['file'])
                except Exception:
                    continue

    mentions_out = out_dir / 'mentions.jsonl'
    mentions_out_fd = mentions_out.open('a', encoding='utf-8')

    def _process_record(rec: dict) -> List[Dict[str, Any]]:
        granule = rec.get('granuleId') or rec.get('granule_id') or rec.get('id') or ""
        txt_link = ''
        if isinstance(rec.get('txt_link'), str):
            m = re.search(r'href="([^"]+)"', rec.get('txt_link'))
            txt_link = m.group(1) if m else rec.get('txt_link')

        text = (
            rec.get('parsed_content_text')
            or rec.get('parsed_text')
            or rec.get('paragraph')
            or rec.get('text')
            or ''
        )
        out: List[Dict[str, Any]] = []
        if not text:
            return out

        sentences = _sentence_tokenize(text)
        for si, sent in enumerate(sentences):
            sent_text = sent.strip()
            found: List[Tuple[str, str, bool, int, int]] = []

            # Single regex pass (names + acronyms)
            for org_id, literal, pat in name_patterns:
                m = pat.search(sent_text)
                if m:
                    found.append((org_id, literal, False, m.start(), m.end()))
            for org_id, literal, pat in acronym_patterns:
                m = pat.search(sent_text)
                if m:
                    found.append((org_id, literal, True, m.start(), m.end()))

            for org_id, variation, is_acro, start, end in found:
                p_start, p_end = max(0, si - 3), min(len(sentences), si + 4)
                out.append({
                    'org_id': org_id,
                    'interest_group': org_name_by_id.get(org_id, ""),  # <-- Add this
                    'variation': variation,
                    'is_acronym': bool(is_acro),
                    'match_text': sent_text[start:end],
                    'match_type': 'acronym' if is_acro else 'name',
                    'score': None,
                    'granuleId': granule,
                    'txt_link': txt_link or '',
                    'sentence': sent_text,
                    'sentence_index': si,
                    'start_in_sentence': start,
                    'end_in_sentence': end,
                    'paragraph': " ".join(sentences[p_start:p_end]),
                    'timestamp': time.time(),
                })
        return out

    total_names = 0
    total_acros = 0

    for file_path in tqdm(source_files, desc="Files", unit="file", dynamic_ncols=True):
        if resume and str(file_path) in processed_seen:
            tqdm.write(f"Skipping already processed file: {file_path}")
            continue

        records_iter = _read_jsonl_or_json(file_path)
        rec_pbar = tqdm(total=0, desc=file_path.name, unit="rec", leave=False, dynamic_ncols=True)

        buffer: List[dict] = []
        for rec in records_iter:
            buffer.append(rec)
            rec_pbar.update(1)
            if len(buffer) >= chunk_size:
                with ThreadPoolExecutor(max_workers=num_threads) as exc:
                    for mentions in tqdm(
                        exc.map(_process_record, buffer),
                        total=len(buffer),
                        desc="Chunk",
                        unit="rec",
                        leave=False,
                        dynamic_ncols=True,
                    ):
                        for m in mentions:
                            mentions_out_fd.write(json.dumps(m, ensure_ascii=False) + "\n")
                            if m.get('is_acronym'):
                                total_acros += 1
                            else:
                                total_names += 1
                buffer = []

        if buffer:
            with ThreadPoolExecutor(max_workers=num_threads) as exc:
                for mentions in tqdm(
                    exc.map(_process_record, buffer),
                    total=len(buffer),
                    desc="Chunk",
                    unit="rec",
                    leave=False,
                    dynamic_ncols=True,
                ):
                    for m in mentions:
                        mentions_out_fd.write(json.dumps(m, ensure_ascii=False) + "\n")
                        if m.get('is_acronym'):
                            total_acros += 1
                        else:
                            total_names += 1

        rec_pbar.close()

        # mark file processed
        with processed_manifest.open('a', encoding='utf-8') as pf:
            pf.write(json.dumps({'file': str(file_path), 'processed_at': time.time()}) + "\n")

    mentions_out_fd.close()
    LOGGER.info("Total name mentions: %d, acronyms: %d", total_names, total_acros)
    return dict(names=total_names, acronyms=total_acros)


def iter_normalized_granules(normalized_dir: Path) -> Iterable[Tuple[str, dict]]:
    by_pkg = normalized_dir / "by_package"
    if by_pkg.exists():
        for pkg_dir in sorted(by_pkg.glob("*")):
            core = pkg_dir / "granules_core.csv"
            if not core.exists():
                continue
            for chunk in pd.read_csv(core, chunksize=10000, dtype=str, keep_default_na=False):
                for _, row in chunk.iterrows():
                    rowd = row.to_dict()
                    pkg = rowd.get("packageId", "")
                    yield pkg, rowd
        return

    main_csv = normalized_dir / "main.csv"
    if not main_csv.exists():
        raise FileNotFoundError(f"Could not find {by_pkg}/**/granules_core.csv or {main_csv}")
    for chunk in pd.read_csv(main_csv, chunksize=10000, dtype=str, keep_default_na=False):
        for _, row in chunk.iterrows():
            rowd = row.to_dict()
            pkg = rowd.get("packageId", "")
            yield pkg, rowd


def process_normalized_packages(
    normalized_dir: Path,
    interest_csv: Path,
    out_dir: Path,
    num_threads: int = 4,
    chunk_size: int = 500,
    resume: bool = True,
    only_package: Optional[str] = None,
    only_canonical_names: bool = False,  # NEW
    strict_current_acronym: bool = False,  # NEW
):
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = load_interest_groups(
        interest_csv,
        strict_current_acronym=strict_current_acronym,
        only_canonical_names=only_canonical_names,  # NEW
    )

    # Map org_id -> interest_group (canonical_name)
    org_name_by_id: Dict[str, str] = {g.org_id: g.canonical_name for g in groups}

    name_patterns: List[Tuple[str, str, re.Pattern]] = []
    acronym_patterns: List[Tuple[str, str, re.Pattern]] = []

    include_alts = (not strict_current_acronym) and (not only_canonical_names)  # NEW

    for g in tqdm(groups, desc="Build patterns", unit="org", dynamic_ncols=True, leave=False):
        if g.canonical_name and keep_name(g.canonical_name):
            name_patterns.append((g.org_id, g.canonical_name, _make_name_pattern(g.canonical_name)))
        for acr in g.acronyms:
            acronym_patterns.append((g.org_id, acr, _make_acronym_pattern(acr)))

    LOGGER.info("Pattern summary (normalized): names=%d acronyms=%d",
                len(name_patterns), len(acronym_patterns))

    processed_manifest = out_dir / 'processed_packages.jsonl'
    processed_seen: set[str] = set()
    if resume and processed_manifest.exists():
        with processed_manifest.open('r', encoding='utf-8') as mf:
            for line in mf:
                try:
                    obj = json.loads(line)
                    if obj.get('packageId'):
                        processed_seen.add(obj['packageId'])
                except Exception:
                    continue

    global_mentions = out_dir / 'mentions.jsonl'
    global_fd = global_mentions.open('a', encoding='utf-8')

    current_pkg: Optional[str] = None
    buffer: List[Tuple[str, dict]] = []
    granule_pbar = tqdm(total=0, desc="Granules", unit="row", dynamic_ncols=True)

    def _process_row(row: dict) -> List[Dict[str, Any]]:
        granule = row.get('granuleId') or row.get('granule_id') or row.get('id') or ""
        pkg = row.get('packageId') or ""
        date = row.get('date') or ""
        title = row.get('title') or ""
        text = row.get('text_bs4') or ''
        out: List[Dict[str, Any]] = []
        if not text:
            return out

        sentences = _sentence_tokenize(text)
        for si, sent in enumerate(sentences):
            sent_text = sent.strip()
            found: List[Tuple[str, str, bool, int, int]] = []

            for org_id, literal, pat in name_patterns:
                m = pat.search(sent_text)
                if m:
                    found.append((org_id, literal, False, m.start(), m.end()))
            for org_id, literal, pat in acronym_patterns:
                m = pat.search(sent_text)
                if m:
                    found.append((org_id, literal, True, m.start(), m.end()))

            for org_id, variation, is_acro, start, end in found:
                p_start, p_end = max(0, si - 3), min(len(sentences), si + 4)
                out.append({
                    'org_id': org_id,
                    'interest_group': org_name_by_id.get(org_id, ""),  # <-- Add this line
                    'variation': variation,
                    'is_acronym': bool(is_acro),
                    'match_text': sent_text[start:end],
                    'match_type': 'acronym' if is_acro else 'name',
                    'score': None,
                    'packageId': pkg,
                    'granuleId': granule,
                    'date': date,
                    'title': title,
                    'sentence': sent_text,
                    'sentence_index': si,
                    'start_in_sentence': start,
                    'end_in_sentence': end,
                    'paragraph': " ".join(sentences[p_start:p_end]),
                    'timestamp': time.time(),
                })
        return out

    def flush(pkg: str, rows: List[dict]):
        if not rows:
            return
        pkg_dir = out_dir / "by_package" / re.sub(r'[^A-Za-z0-9._-]', '_', pkg or "UNKNOWN")
        pkg_dir.mkdir(parents=True, exist_ok=True)
        pkg_mentions_path = pkg_dir / "mentions.jsonl"
        with pkg_mentions_path.open('a', encoding='utf-8') as pkg_fd, ThreadPoolExecutor(max_workers=num_threads) as exc:
            for mentions in tqdm(
                exc.map(_process_row, rows),
                total=len(rows),
                desc=f"Package {pkg}",
                unit="row",
                leave=False,
                dynamic_ncols=True,
            ):
                granule_pbar.update(1)
                for m in mentions:
                    line = json.dumps(m, ensure_ascii=False)
                    pkg_fd.write(line + "\n")
                    global_fd.write(line + "\n")

        with processed_manifest.open('a', encoding='utf-8') as pf:
            pf.write(json.dumps({'packageId': pkg, 'processed_at': time.time()}) + "\n")

    for pkg, rowd in iter_normalized_granules(normalized_dir):
        if only_package and pkg != only_package:
            continue
        if resume and pkg and pkg in processed_seen:
            continue
        if current_pkg is None:
            current_pkg = pkg
        if pkg != current_pkg:
            flush(current_pkg, [r for p, r in buffer if p == current_pkg])
            buffer = [(p, r) for (p, r) in buffer if p != current_pkg]
            current_pkg = pkg
        buffer.append((pkg, rowd))
        if len(buffer) >= chunk_size:
            flush(current_pkg, [r for p, r in buffer if p == current_pkg])
            buffer = [(p, r) for (p, r) in buffer if p != current_pkg]

    if current_pkg is not None:
        flush(current_pkg, [r for p, r in buffer if p == current_pkg])

    granule_pbar.close()
    global_fd.close()


def main(
    source_dir: Optional[Path] = None,
    interest_csv: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    num_threads: int = 4,
    chunk_size: int = 500,
    resume: bool = True,
    only_main: bool = False,
    strict_current_acronym: bool = False,  # NEW
    only_canonical_names: bool = False,    # NEW
):
    source_dir = Path(source_dir or "data/raw")
    interest_csv = Path(interest_csv or DEFAULT_INTEREST_CSV)
    out_dir = Path(out_dir or "data/processed/mentions")

    if not interest_csv.exists():
        raise FileNotFoundError(f"Interest group master CSV not found: {interest_csv}")

    files = find_source_files(source_dir)
    if only_main:
        by_month_dir = source_dir / "by_month"
        if by_month_dir.exists() and by_month_dir.is_dir():
            month_files = sorted([f for f in by_month_dir.glob("main_*.csv") if f.is_file()])
            files = month_files or [f for f in files if f.suffix.lower() == ".csv" and f.stem.startswith("main")]
        else:
            files = [f for f in files if f.suffix.lower() == ".csv" and f.stem.startswith("main")]

    if not files:
        LOGGER.warning("No source files found in %s", source_dir)
        return

    res = process_files(
        files,
        interest_csv,
        out_dir,
        num_threads=num_threads,
        chunk_size=chunk_size,
        resume=resume,
        only_canonical_names=only_canonical_names,      # NEW
        strict_current_acronym=strict_current_acronym,  # NEW
    )
    LOGGER.info("Processing complete: %s", res)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog="mention_extractor", description="Extract interest group mentions")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Normalized mode
    p_norm = sub.add_parser("normalized", help="Scan normalized outputs (prefers by_package/*)")
    p_norm.add_argument("--normalized-dir", default="data/normalized")
    p_norm.add_argument("--interest-csv", type=Path, default=DEFAULT_INTEREST_CSV)
    p_norm.add_argument("--out-dir", default="data/processed/mentions")
    p_norm.add_argument("--threads", type=int, default=4)
    p_norm.add_argument("--chunk-size", type=int, default=1000)
    p_norm.add_argument("--resume", action="store_true")
    p_norm.add_argument("--only-package", type=str, default=None, help="Process only this packageId")
    p_norm.add_argument("--only-canonical-names", action="store_true",
                    help="Restrict matches to canonical names and acronyms only (no alt/partial names)")
    p_norm.add_argument(
        "--strict-current-acronym",
        action="store_true",
        help="Match ONLY current_name and acronym (no alts)"
    )
    # Raw/legacy mode
    p_raw = sub.add_parser("raw", help="Scan raw or generic files (json/jsonl/gz/csv)")
    p_raw.add_argument("--source-dir", default="data/raw")
    p_raw.add_argument("--interest-csv", type=Path, default=DEFAULT_INTEREST_CSV)
    p_raw.add_argument("--out-dir", default="data/processed/mentions")
    p_raw.add_argument("--threads", type=int, default=4)
    p_raw.add_argument("--chunk-size", type=int, default=500)
    p_raw.add_argument("--no-resume", dest="resume", action="store_false", help="Do not skip already processed files")
    p_raw.add_argument("--only-main", action="store_true")
    p_raw.add_argument(
        "--strict-current-acronym",
        action="store_true",
        help="Match ONLY current_name and acronym (no alts)"
    )
    p_raw.add_argument(  # NEW
        "--only-canonical-names",
        action="store_true",
        help="Restrict matches to current_name + acronym(s) only (no alt/partial names)"
    )

    args = parser.parse_args()

    if args.cmd == "normalized":
        process_normalized_packages(
            Path(args.normalized_dir),
            Path(args.interest_csv),
            Path(args.out_dir),
            num_threads=args.threads,
            chunk_size=args.chunk_size,
            resume=args.resume,
            only_package=args.only_package,
            only_canonical_names=args.only_canonical_names,
            strict_current_acronym=args.strict_current_acronym,
        )
    else:
        main(
            source_dir=Path(args.source_dir),
            interest_csv=Path(args.interest_csv),
            out_dir=Path(args.out_dir),
            num_threads=args.threads,
            chunk_size=args.chunk_size,
            resume=getattr(args, "resume", True),
            only_main=args.only_main,
            strict_current_acronym=args.strict_current_acronym,
            only_canonical_names=getattr(args, "only_canonical_names", False),
        )
