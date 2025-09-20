# interest_group_prep.py
from __future__ import annotations
from pathlib import Path
import sys
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Fill current_name_2 from original_name_2 when blank, keep 3 cols, and export CSV."
    )
    parser.add_argument("--in", dest="in_path", type=Path, default=None,
                        help="Path to Interest_groups_manually_validated.xlsx")
    parser.add_argument("--out", dest="out_path", type=Path, default=None,
                        help="Path to write interest_groups_list.csv")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    script_dir  = script_path.parent

    # Resolve input
    if args.in_path is not None:
        in_path = args.in_path.expanduser().resolve()
    else:
        try:
            project_root = script_dir.parents[2]  # 1.data_collection -> interest_group_analysis -> project root
        except IndexError:
            print("ERROR: Could not compute project root from script location:", script_dir)
            sys.exit(1)
        in_path = (project_root / "data" / "Interest_groups_manually_validated.xlsx").resolve()

    # Resolve output
    out_path = (args.out_path.expanduser().resolve()
                if args.out_path is not None
                else (in_path.parent / "interest_groups_list.csv").resolve())

    print(f"Script: {script_path}")
    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")

    if not in_path.exists():
        print(f"ERROR: Input file not found at:\n  {in_path}")
        sys.exit(1)

    # Load and validate
    df = pd.read_excel(in_path, dtype=str)
    required_cols = {"org_id", "original_name_2", "current_name_2", "acronym_2"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: Missing required column(s): {', '.join(sorted(missing))}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Normalize and fill
    df["original_name_2"] = df["original_name_2"].astype("string")
    df["current_name_2"]  = df["current_name_2"].astype("string")
    df["acronym_2"]       = df["acronym_2"].astype("string")

    blank_curr = df["current_name_2"].isna() | (df["current_name_2"].str.strip().fillna("") == "")
    src = df["original_name_2"].str.strip()
    can_copy = blank_curr & src.notna() & (src != "")
    n_updates = int(can_copy.sum())
    df.loc[can_copy, "current_name_2"] = src[can_copy]

    # Keep only the requested columns and rename them
    out_df = (
        df[["org_id", "current_name_2", "acronym_2"]]
        .rename(columns={"current_name_2": "interest_group", "acronym_2": "acronym"})
        .assign(
            interest_group=lambda d: d["interest_group"].astype("string").str.strip(),
            acronym=lambda d: d["acronym"].astype("string").str.strip(),
        )
    )

    # Write CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Updated rows (filled current_name_2): {n_updates}")
    print(f"Wrote CSV with columns: {list(out_df.columns)}")

if __name__ == "__main__":
    main()
