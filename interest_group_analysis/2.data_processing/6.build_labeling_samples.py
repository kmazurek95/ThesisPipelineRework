#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, logging, re, hashlib, unicodedata, datetime as dt
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd

LOGGER = logging.getLogger("labeling_sampler")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Sample merged windows for labeling (no extra processing).")
    p.add_argument("--data", type=Path, required=True, help="Input analytic_windows file (.jsonl|.csv).")
    p.add_argument("--labeled", type=Path, help="Existing labeled file to de-dup against (CSV/JSONL).")
    p.add_argument("--out", type=Path, required=True, help="Output dir.")
    p.add_argument("--n", type=int, default=700, help="Total rows to sample.")
    p.add_argument("--seed", type=int, default=98, help="Random seed.")
    p.add_argument("--stratify-by", nargs="*", default=[], help="Optional columns to approximate stratified sampling (e.g., chamber year).")
    p.add_argument("--csv-only", action="store_true", help="Write only CSV + JSONL (no Parquet).")
    p.add_argument("--mask-if-missing", action="store_true", help="Create text_for_model by masking variations if missing.")
    return p.parse_args()

# ---------- Utils ----------
def read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in {".jsonl", ".ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    elif ext == ".csv":
        return pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"Unsupported input: {path}")

def save_jsonl(df: pd.DataFrame, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def normalize_for_hash(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"\s+", " ", s.strip())

def compute_window_id(row: pd.Series) -> str:
    # Prefer existing text columns in this order
    txt = row.get("window_text")
    if not isinstance(txt, str) or not txt:
        txt = row.get("text_for_labeler") or row.get("paragraph") or ""
    key = f"{row.get('org_id','')}||{row.get('source_block_id','')}||{normalize_for_hash(txt)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

def mask_org_mentions(text: str, forms: List[str]) -> str:
    if not isinstance(text, str):
        return ""
    forms = sorted({f for f in forms if isinstance(f,str) and f.strip()}, key=len, reverse=True)
    for f in forms:
        text = re.sub(rf"\b{re.escape(f)}\b", "[ORG]", text, flags=re.IGNORECASE)
    return text

def ensure_text_for_model(df: pd.DataFrame, mask_if_missing: bool) -> pd.DataFrame:
    df = df.copy()
    if "text_for_model" in df.columns and df["text_for_model"].notna().any():
        return df
    if not mask_if_missing:
        df["text_for_model"] = df.get("window_text", df.get("text_for_labeler", ""))
        return df
    # Try masking using variations_in_window if present
    forms_col = "variations_in_window" if "variations_in_window" in df.columns else None
    def _mk(row):
        base = row.get("window_text") or row.get("text_for_labeler") or ""
        forms = row.get(forms_col) if forms_col else []
        if isinstance(forms, str):
            try:
                forms = json.loads(forms)
            except Exception:
                forms = [forms]
        return mask_org_mentions(base, forms or [])
    df["text_for_model"] = df.apply(_mk, axis=1)
    return df

def remove_overlap(df: pd.DataFrame, labeled_path: Optional[Path]) -> pd.DataFrame:
    if not labeled_path or not labeled_path.exists():
        return df
    try:
        labeled = read_table(labeled_path)
    except Exception:
        LOGGER.warning("Could not read labeled file; skipping overlap removal.")
        return df
    if "window_id" in labeled.columns and "window_id" in df.columns:
        before = len(df)
        df = df[~df["window_id"].astype(str).isin(labeled["window_id"].astype(str).dropna())]
        LOGGER.info("Overlap removal by window_id: %d -> %d", before, len(df))
        return df
    LOGGER.info("No window_id in labeled; overlap removal skipped.")
    return df

def stratified_sample(df: pd.DataFrame, n: int, by: List[str], seed: int) -> pd.DataFrame:
    if not by or not set(by).issubset(df.columns):
        return df.sample(n=min(n, len(df)), random_state=seed) if len(df) > n else df
    # proportional by groups, simple & deterministic
    g = df.groupby(by, dropna=False)
    sizes = (g.size() / len(df) * n).round().astype(int).clip(lower=1)
    parts = []
    for key, grp in g:
        take = int(min(len(grp), sizes.loc[key]))
        parts.append(grp.sample(n=take, random_state=seed))
    out = pd.concat(parts, axis=0)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    return out

# ---------- Main ----------
def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # 1) Load windows
    df = read_table(args.data)
    if df.empty:
        LOGGER.error("No rows in input.")
        return

    # 2) Ensure stable window_id (if upstream didn’t write it)
    if "window_id" not in df.columns:
        df["window_id"] = df.apply(compute_window_id, axis=1)

    # 3) Ensure labeler/model text
    # prefer existing columns:
    if "window_text" not in df.columns and "text_for_labeler" in df.columns:
        df["window_text"] = df["text_for_labeler"]
    elif "window_text" not in df.columns and "paragraph" in df.columns:
        df["window_text"] = df["paragraph"]
    df = ensure_text_for_model(df, mask_if_missing=args.mask_if_missing)

    # 4) Remove overlap with already labeled items
    df = remove_overlap(df, args.labeled)

    # 5) Sample (optionally stratified)
    take = min(args.n, len(df))
    sampled = stratified_sample(df, take, args.stratify_by, args.seed)

    # 6) Save
    stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base = f"LABELING_WINDOWS__{stamp}"
    out_csv  = args.out / f"{base}.csv"
    out_json = args.out / f"{base}.jsonl"

    sampled.to_csv(out_csv, index=False, encoding="utf-8-sig")
    save_jsonl(sampled, out_json)

    meta = {
        "generated_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds")+"Z",
        "seed": args.seed,
        "input_file": str(args.data),
        "labeled_file": str(args.labeled) if args.labeled else None,
        "n_available": int(len(df)),
        "n_sampled": int(len(sampled)),
        "stratify_by": args.stratify_by,
        "outputs": {"csv": str(out_csv), "jsonl": str(out_json)},
    }
    (args.out / f"{base}__meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOGGER.info("✅ Wrote %s and %s", out_csv, out_json)

if __name__ == "__main__":
    main()