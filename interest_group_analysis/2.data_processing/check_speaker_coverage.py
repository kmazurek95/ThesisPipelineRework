r"""
Speaker coverage audit

This helper scans normalized Congressional Record data to check whether speaker
cues in the text align with recorded members per granule. It produces three CSVs:

- speaker_audit_all.csv: one row per granule with counts of detected speaker cues
    and number of members listed.
- speaker_audit_noMembers_butCues.csv: granules where no members are listed but
    the text contains speaker cues (potential metadata gap).
- speaker_audit_members_butNoCues.csv: granules where members are listed but
    no speaker cues are found in the text (potential extraction gap).

How to run (PowerShell):

        python .\interest_group_analysis\2.data_processing\check_speaker_coverage.py `
            --norm-dir data\normalized\normalized_114_run2

Notes:
- The --norm-dir should be the normalized directory that contains a by_package/
    subfolder with per-package granules_core.csv and granule_members.csv files.
- Adjust the path to match your congress/output layout.
"""

from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd

# --- same cue pattern family as in speaker_attribution.py ---
HONORIFICS = r"(Mr|Mrs|Ms|Miss|Mx|Dr|Senator|Sen|Representative|Rep)\."
ROLES = r"(?:The\s+(?:SPEAKER(?:\s+pro\s+tempore)?|PRESIDING\s+OFFICER|CHAIR|ACTING\s+PRESIDENT\s+pro\s+tempore)|(?:Mr\.\s+Speaker|Mr\.\s+President|Madam\s+Speaker|Madam\s+President))"
SENTENCE_END_PUNCT = r"[\.:]"
BOUNDARY = r"(?:(?<=^)|(?<=\n)|(?<=\.)|(?<=\?)|(?<=!)|(?<=\)))\s*"
SPEAKER_CUE_RE = re.compile(
    rf"{BOUNDARY}(?:{HONORIFICS}\s+(?P<name>(?-i:[A-Z][A-Z\-\.'\s]+?))(?:\s+of\s+[A-Z][A-Za-z\.\s]+)?{SENTENCE_END_PUNCT})"
    rf"|{BOUNDARY}(?P<role>{ROLES}){SENTENCE_END_PUNCT}",
    flags=re.IGNORECASE,
)

def find_granule_member_file(norm_dir: Path) -> list[Path]:
    # aggregated file?
    agg = list(norm_dir.rglob("granule_members.csv"))
    if agg:
        return agg
    # per-package fallback
    return list((norm_dir / "by_package").rglob("granule_members.csv"))

def iter_granules_core(norm_dir: Path):
    for p in (norm_dir / "by_package").rglob("granules_core.csv"):
        yield p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm-dir", required=True, help="normalized dir (the one that contains by_package/)")
    ap.add_argument("--out-prefix", default="speaker_audit", help="prefix for output CSVs")
    args = ap.parse_args()
    norm_dir = Path(args.norm_dir)

    # 1) Load members and count members per granule
    mem_paths = find_granule_member_file(norm_dir)
    if not mem_paths:
        raise SystemExit(f"No granule_members.csv found under {norm_dir}")
    df_mem_list = []
    for p in mem_paths:
        try:
            for chunk in pd.read_csv(p, dtype=str, chunksize=100000, keep_default_na=False):
                if "granuleId" in chunk.columns:
                    df_mem_list.append(chunk[["granuleId"]].assign(_one=1))
        except Exception as e:
            print(f"Skipping {p}: {e}")
    df_mem = pd.concat(df_mem_list, ignore_index=True) if df_mem_list else pd.DataFrame(columns=["granuleId","_one"])
    members_per_gid = df_mem.groupby("granuleId")["_one"].sum().rename("n_members").reset_index()

    # 2) Scan text for speaker cues per granule (use text_for_speaker, then fallback)
    rows = []
    for core_csv in iter_granules_core(norm_dir):
        for chunk in pd.read_csv(core_csv, dtype=str, chunksize=20000, keep_default_na=False):
            if "granuleId" not in chunk.columns:
                continue
            # choose text column
            def pick_text(row):
                for c in ("text_for_speaker", "text_bs4", "parsed_text", "text_readability"):
                    v = row.get(c)
                    if isinstance(v, str) and v.strip():
                        return v, c
                return "", None
            for _, r in chunk.iterrows():
                gid = str(r["granuleId"])
                txt, src = pick_text(r)
                if not txt:
                    rows.append({"granuleId": gid, "packageId": r.get("packageId"), "text_src": src, "n_cues": 0, "first_cue": None})
                    continue
                matches = list(SPEAKER_CUE_RE.finditer(txt))
                first = matches[0].group(0).strip() if matches else None
                rows.append({"granuleId": gid, "packageId": r.get("packageId"), "text_src": src, "n_cues": len(matches), "first_cue": first})

    cues = pd.DataFrame(rows)
    if cues.empty:
        raise SystemExit("No granules_core.csv or no rows found.")

    # 3) Join and compute flags
    audit = cues.merge(members_per_gid, how="left", on="granuleId")
    audit["n_members"] = audit["n_members"].fillna(0).astype(int)

    # A) Cases you asked about: no members listed, but the text shows speaker cues
    suspect = audit[(audit["n_members"] == 0) & (audit["n_cues"] > 0)].copy()

    # B) (Optional) Opposite: members listed, but no cues found in text
    silent = audit[(audit["n_members"] > 0) & (audit["n_cues"] == 0)].copy()

    # Save reports
    out_all    = Path(f"{args.out_prefix}_all.csv")
    out_sus    = Path(f"{args.out_prefix}_noMembers_butCues.csv")
    out_silent = Path(f"{args.out_prefix}_members_butNoCues.csv")
    audit.to_csv(out_all, index=False)
    suspect.to_csv(out_sus, index=False)
    silent.to_csv(out_silent, index=False)

    # Console summary
    print("\n=== Speaker coverage audit ===")
    print(f"Granules scanned: {len(audit):,}")
    print(f"  No members but cues present: {len(suspect):,}  -> {out_sus}")
    print(f"  Members present but no cues: {len(silent):,}  -> {out_silent}")
    print(f"Full audit saved to: {out_all}")

if __name__ == "__main__":
    main()
