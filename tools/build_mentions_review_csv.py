# tools/build_mentions_review_csv.py
from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Dict, Tuple, Iterable, List

import pandas as pd

# --- CLI-ish knobs: edit for your run ---
MENTIONS_JSONL = Path(r"data\processed\mentions_and_speaker_114\mentions_with_speakers.jsonl")
BY_PACKAGE_DIR = Path(r"data\normalized\normalized_114_run2\by_package")
OUT_CSV        = Path(r"data\processed\mentions_and_speaker_114\mentions_review.csv")
CTX = 120  # characters of context on each side

TEXT_ORDER = ["text_for_speaker", "text_bs4", "parsed_text", "text_readability"]

def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("mention_char_start","mention_char_end","speaker_confidence"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["granuleId"] = df["granuleId"].astype(str)
    return df

def iter_core_csvs(root: Path) -> Iterable[Path]:
    return root.rglob("granules_core.csv")

def preferred_text(row: pd.Series) -> Tuple[str, str]:
    for c in TEXT_ORDER:
        t = row.get(c)
        if isinstance(t, str) and t.strip():
            return t, c
    return "", ""

def build_granule_index(df_mentions: pd.DataFrame) -> Dict[str, Dict]:
    needed = set(df_mentions["granuleId"].dropna().astype(str))
    idx: Dict[str, Dict] = {}
    missing_cols_any = False
    if not BY_PACKAGE_DIR.exists():
        raise FileNotFoundError(f"Not found: {BY_PACKAGE_DIR}")

    # Don’t over-prune with usecols — read all headers, select what we need.
    wanted_meta = ["granuleId","packageId","date","title","chamber","detailsLink","txtLink","pdfLink"]
    for core in iter_core_csvs(BY_PACKAGE_DIR):
        try:
            for chunk in pd.read_csv(core, dtype=str, chunksize=4000, keep_default_na=False):
                if "granuleId" not in chunk.columns:
                    continue
                sub = chunk[chunk["granuleId"].astype(str).isin(needed)]
                if sub.empty:
                    continue
                # track column availability once
                if not missing_cols_any:
                    needed_cols = set(TEXT_ORDER + wanted_meta)
                    missing = [c for c in needed_cols if c not in sub.columns]
                    if missing:
                        missing_cols_any = True
                        print(f"[WARN] Missing columns in {core.name}: {missing}")
                for _, r in sub.iterrows():
                    gid = str(r["granuleId"])
                    if gid in idx:
                        continue
                    text, src = preferred_text(r)
                    idx[gid] = {
                        "text": text or "",
                        "text_source": src,
                        "packageId": r.get("packageId",""),
                        "date": r.get("date",""),
                        "title": r.get("title",""),
                        "chamber": r.get("chamber",""),
                        "detailsLink": r.get("detailsLink",""),
                        "txtLink": r.get("txtLink",""),
                        "pdfLink": r.get("pdfLink",""),
                        # also keep the raw text columns if present
                        "text_for_speaker": r.get("text_for_speaker",""),
                        "text_bs4": r.get("text_bs4",""),
                        "parsed_text": r.get("parsed_text",""),
                        "text_readability": r.get("text_readability",""),
                    }
        except Exception as e:
            print(f"[WARN] Failed reading {core}: {e}")
    return idx

def slice_context(text: str, start, end) -> Tuple[str, str, str]:
    if not isinstance(start, (int,float)) or pd.isna(start):
        return "", "", ""
    start = max(0, int(start))
    if not isinstance(end, (int,float)) or pd.isna(end) or end <= start:
        # try to end at next word-ish boundary
        end = min(len(text), start + 60)
        for sep in (" ", "\n", "\r", "\t", ".", ",", ";", ":", ")"):
            p = text.find(sep, start+1, min(len(text), start+180))
            if p != -1:
                end = p
                break
    end = int(min(max(end, start+1), len(text)))

    left  = text[max(0, start-CTX): start]
    mid   = text[start:end]
    right = text[end: min(len(text), end+CTX)]
    squash = lambda s: s.replace("\r"," ").replace("\n"," ").replace("\t"," ").strip()
    return squash(left), squash(mid), squash(right)

def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    mentions = read_jsonl(MENTIONS_JSONL)
    if mentions.empty:
        print(f"[ERROR] No rows in {MENTIONS_JSONL}")
        return

    gidx = build_granule_index(mentions)
    need = set(mentions["granuleId"])
    print(f"Granules matched: {len(gidx)} / {len(need)}")

    cols = [
        "granuleId","packageId","date","chamber","title",
        "mention","mention_char_start","mention_char_end",
        "speaker_raw","speaker_canonical","speaker_bioguide","speaker_method","speaker_confidence",
        "text_source","context_left","mention_exact","context_right",
        # full text columns (BIG!)
        "text_for_speaker","text_bs4","parsed_text","text_readability"
    ]

    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()

        for gid, g in mentions.groupby(mentions["granuleId"].astype(str)):
            meta = gidx.get(gid, {})
            text = meta.get("text","")

            for _, r in g.iterrows():
                left, mid, right = ("","","")
                if text:
                    left, mid, right = slice_context(text, r.get("mention_char_start"), r.get("mention_char_end"))
                w.writerow({
                    "granuleId": gid,
                    "packageId": meta.get("packageId",""),
                    "date": meta.get("date",""),
                    "chamber": meta.get("chamber",""),
                    "title": meta.get("title",""),
                    "mention": r.get("mention",""),
                    "mention_char_start": r.get("mention_char_start",""),
                    "mention_char_end": r.get("mention_char_end",""),
                    "speaker_raw": r.get("speaker_raw",""),
                    "speaker_canonical": r.get("speaker_canonical",""),
                    "speaker_bioguide": r.get("speaker_bioguide",""),
                    "speaker_method": r.get("speaker_method",""),
                    "speaker_confidence": r.get("speaker_confidence",""),
                    "text_source": meta.get("text_source",""),
                    "context_left": left,
                    "mention_exact": mid,
                    "context_right": right,
                    # full text columns
                    "text_for_speaker": meta.get("text_for_speaker",""),
                    "text_bs4": meta.get("text_bs4",""),
                    "parsed_text": meta.get("parsed_text",""),
                    "text_readability": meta.get("text_readability",""),
                })

    print(f"Wrote {OUT_CSV}")

if __name__ == "__main__":
    main()
