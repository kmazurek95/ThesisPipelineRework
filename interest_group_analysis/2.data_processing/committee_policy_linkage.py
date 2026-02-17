# 4.committee_policy_linkage.py
from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# =========================== CONFIG ===========================
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = _PROJECT_ROOT / "data" / "reference"
COMMITTEES_CSV = BASE_DIR / "g.committees_CREC_114_AND_115.csv"
PROMINENCE_CSV = BASE_DIR / "df_interest_groups_prominence.csv"  # optional join target
OUT_DIR = _PROJECT_ROOT / "data" / "output" / "policy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# If you want a CSV mapping instead of the in-code dict, set this path:
# Expected columns: committee_canonical, policy_area_name, policy_area_code
OPTIONAL_MAPPING_CSV: Optional[Path] = None

# Deterministic tie-break priority for policy areas (most global-salient first, edit as needed)
POLICY_PRIORITY = [
    "Macroeconomics","Health","Education","Labor","Housing","Law and Crime","Environment",
    "Energy","Transportation","Technology","Social Welfare","Defense","Domestic Commerce",
    "International Affairs","Government Operations","Public Lands","Agriculture","Culture"
]

# =========================== STATIC MAPPING (fallback) ===========================
# (canonical committee name -> [policy_area_name, policy_area_code])
COMMITTEE_TO_POLICY: Dict[str, Tuple[str, int]] = {
    'Committee on Banking, Housing, and Urban Affairs': ('Housing', 1400),
    'Committee on the Judiciary': ('Law and Crime', 1200),
    'Committee on Health, Education, Labor, and Pensions': ('Health', 300),
    'Committee on Appropriations': ('Macroeconomics', 100),
    'Committee on Agriculture, Nutrition, and Forestry': ('Agriculture', 400),
    'Committee on Foreign Relations': ('International Affairs', 1900),
    'Committee on Rules': ('Government Operations', 2000),
    'Committee on Agriculture': ('Agriculture', 400),
    'Committee on Education and the Workforce': ('Education', 600),
    'Committee on Foreign Affairs': ('International Affairs', 1900),
    'Committee on Education and Labor': ('Education', 600),
    'Committee on Environment and Public Works': ('Environment', 700),
    "Committee on Veterans' Affairs": ('Government Operations', 2000),
    'Committee on Armed Services': ('Defense', 1600),
    'Committee on Energy and Natural Resources': ('Energy', 800),
    'Committee on Finance': ('Macroeconomics', 100),
    'Committee on Energy and Commerce': ('Energy', 800),
    'Committee on Ways and Means': ('Macroeconomics', 100),
    'Committee on Small Business and Entrepreneurship': ('Domestic Commerce', 1500),
    'Committee on Homeland Security and Governmental Affairs': ('Government Operations', 2000),
    'Committee on Financial Services': ('Domestic Commerce', 1500),
    'Joint Committee on Taxation': ('Macroeconomics', 100),
    'Committee on Natural Resources': ('Public Lands', 2100),
    'Permanent Select Committee on Intelligence': ('Defense', 1600),
    'Committee on Standards of Official Conduct': ('Government Operations', 2000),
    'Committee on Transportation and Infrastructure': ('Transportation', 1000),
    'Committee on Commerce, Science, and Transportation': ('Domestic Commerce', 1500),
    'Committee on Science, Space, and Technology': ('Technology', 1700),
    'Committee on Commerce': ('Domestic Commerce', 1500),
    'Committee on the Budget': ('Macroeconomics', 100),
    'Committee on House Administration': ('Government Operations', 2000),
    'Committee on Oversight and Government Reform': ('Government Operations', 2000),
    'Special Committee on Aging': ('Social Welfare', 1300),
    'Select Committee on Intelligence': ('Defense', 1600),
    'Committee on Homeland Security': ('Defense', 1600),
    'Committee on Indian Affairs': ('Civil Rights', 200),
    'Committee on Rules and Administration': ('Government Operations', 2000),
    'Select Committee on Ethics': ('Government Operations', 2000),
    'Committee on Ethics': ('Government Operations', 2000),
    'Joint Select Committee on Deficit Reduction': ('Macroeconomics', 100),
    'Committee on Small Business': ('Domestic Commerce', 1500),
    'Temporary Joint Committee on Deficit Reduction': ('Macroeconomics', 100),
    'Joint Committee on Printing': ('Government Operations', 2000),
    'Select Committee on the Events Surrounding the 2012 Terrorist Attack in Benghazi': ('International Affairs', 1900),
    'Select Committee on Assassinations': ('Government Operations', 2000),
    'Committee on Education': ('Education', 600),
    'Committee on Banking and Currency': ('Domestic Commerce', 1500),
    'Committee on Oversight': ('Government Operations', 2000),
    'Joint Committee on the Library': ('Culture', 2300),
    'Committee on Public Works and Transportation': ('Transportation', 1000),
    'Committee on Interior and Insular Affairs': ('Public Lands', 2100),
    'Committee on Labor and Human Resources': ('Labor', 500),
    'Committee on Government Operations': ('Government Operations', 2000),
    'Committee on Science and Technology': ('Technology', 1700),
    'Committee on Governmental Affairs': ('Government Operations', 2000),
    'Select Committee on Energy Independence and Global Warming': ('Energy', 800),
    'Committee on Public Works': ('Public Lands', 2100),
    'Select Committee on Hunger': ('Social Welfare', 1300),
    'Select Committee on Presidential Campaign Activities': ('Government Operations', 2000),
    'Ad Hoc Committee on Energy': ('Energy', 800),
    'Committee on Merchant Marine and Fisheries': ('Domestic Commerce', 1500),
    'Committee on Public Lands': ('Public Lands', 2100),
    'Committee on Banking, Finance, and Urban Affairs': ('Domestic Commerce', 1500),
    'Joint Economic Committee': ('Macroeconomics', 100),
    'Committee on International Relations': ('International Affairs', 1900),
    'Committee on District of Columbia': ('Government Operations', 2000),
    'Select Committee on Standards and Conduct': ('Government Operations', 2000),
    'Committee on Science': ('Technology', 1700),
    'Committee on Agriculture and Forestry': ('Agriculture', 400),
    'Committee on Government Reform': ('Government Operations', 2000),
    'Joint Committee on Internal Revenue Taxation': ('Macroeconomics', 100),
    'Select Investigative Panel of the Committee on Energy and Commerce': ('Energy', 800),
    'Committee on Labor and Public Welfare': ('Social Welfare', 1300),
    'United States Senate Caucus on International Narcotics Control': ('International Affairs', 1900),
    'Committee on Internal Security': ('Law and Crime', 1200),
    'Joint Select Committee on Solvency of Multiemployer Pension Plans': ('Labor', 500),
    'Committee on National Security': ('Defense', 1600),
    'Select Committee on Aging': ('Social Welfare', 1300),
    'Joint Committee on the Organization of Congress': ('Government Operations', 2000),
}

# Aliases → canonical committee name (deterministic; no fuzzy)
ALIASES: Dict[str, str] = {
    # Example aliasing; extend over time using the unmapped export
    "house committee on energy and commerce": "Committee on Energy and Commerce",
    "senate committee on energy and natural resources": "Committee on Energy and Natural Resources",
    "house committee on foreign affairs": "Committee on Foreign Affairs",
    "house committee on ways and means": "Committee on Ways and Means",
    "senate committee on the judiciary": "Committee on the Judiciary",
    "house committee on appropriations": "Committee on Appropriations",
    "senate finance committee": "Committee on Finance",
    "senate armed services committee": "Committee on Armed Services",
    "senate foreign relations committee": "Committee on Foreign Relations",
    # add more as you discover them
}

# =========================== HELPERS ===========================
def _norm(s: str) -> str:
    """Deterministic normalization: lowercase, collapse spaces, strip punctuation-like chars."""
    if pd.isna(s) or s is None:
        return ""
    t = str(s).strip().lower()
    for ch in [".", ",", ";", ":", "'", '"', "’", "“", "”"]:
        t = t.replace(ch, "")
    t = " ".join(t.split())
    return t

def load_df(path: Path, kind: str = "csv") -> pd.DataFrame:
    try:
        if kind == "csv":
            return pd.read_csv(path)
        elif kind == "jsonl":
            return pd.read_json(path, lines=True)
        else:
            raise ValueError("kind must be 'csv' or 'jsonl'")
    except FileNotFoundError:
        logging.error("File not found: %s", path)
        return pd.DataFrame()

def build_mapping() -> Dict[str, Tuple[str, int]]:
    """
    Build final mapping dict. If OPTIONAL_MAPPING_CSV is provided, it overrides/extends COMMITTEE_TO_POLICY.
    """
    mapping = {k: v for k, v in COMMITTEE_TO_POLICY.items()}
    if OPTIONAL_MAPPING_CSV and OPTIONAL_MAPPING_CSV.exists():
        m = pd.read_csv(OPTIONAL_MAPPING_CSV)
        need = {"committee_canonical", "policy_area_name", "policy_area_code"}
        if not need.issubset(m.columns):
            raise ValueError(f"Mapping CSV must have columns {need}")
        for _, r in m.iterrows():
            mapping[str(r["committee_canonical"])] = (str(r["policy_area_name"]), int(r["policy_area_code"]))
    return mapping

# =========================== CORE PIPELINE ===========================
def build_committee_links(df_committees: pd.DataFrame,
                          mapping: Dict[str, Tuple[str, int]]) -> pd.DataFrame:
    """
    Returns many-to-many link rows:
    granuleId, committeeName_original, committeeName_canonical, policy_area_name, policy_area_code, provenance
    """
    if df_committees.empty:
        return pd.DataFrame(columns=[
            "granuleId","committeeName_original","committeeName_canonical",
            "policy_area_name","policy_area_code","provenance"
        ])

    # Ensure required columns
    if "granuleId" not in df_committees.columns or "committeeName" not in df_committees.columns:
        raise ValueError("df_committees must contain 'granuleId' and 'committeeName' columns.")

    # Normalize + alias → canonical
    work = df_committees[["granuleId","committeeName"]].copy()
    work["committeeName_original"] = work["committeeName"]
    work["committee_norm"] = work["committeeName"].map(_norm)
    work["committeeName_canonical"] = work["committeeName"]  # default passthrough

    # alias resolution
    work.loc[work["committee_norm"].isin(ALIASES.keys()), "committeeName_canonical"] = \
        work.loc[work["committee_norm"].isin(ALIASES.keys()), "committee_norm"].map(ALIASES)

    # apply mapping
    work["policy_area_tuple"] = work["committeeName_canonical"].map(mapping)

    # provenance flags
    work["provenance"] = work.apply(
        lambda r: "alias+dict" if r["committee_norm"] in ALIASES and r["policy_area_tuple"] is not None
        else ("dict" if r["policy_area_tuple"] is not None else "unmapped"),
        axis=1
    )

    links = work.rename(columns={"committeeName": "committeeName_input"}).copy()
    links[["policy_area_name","policy_area_code"]] = pd.DataFrame(
        links["policy_area_tuple"].tolist(), index=links.index
    )
    links = links.drop(columns=["policy_area_tuple","committee_norm"])

    # persist unmapped to help you extend ALIASES / mapping
    unmapped = links[links["provenance"] == "unmapped"] \
        .groupby("committeeName_original").size().sort_values(ascending=False).reset_index(name="count")
    if not unmapped.empty:
        unmapped.to_csv(OUT_DIR / "unmapped_committees.csv", index=False)

    # keep only mapped rows for downstream modeling (but we’ve already exported unmapped)
    links_mapped = links[links["provenance"] != "unmapped"].drop_duplicates()
    links_mapped.to_parquet(OUT_DIR / "committee_policy_links.parquet", index=False)
    links_mapped.to_csv(OUT_DIR / "committee_policy_links.csv", index=False)
    return links_mapped

def derive_dominant_policy_per_granule(links: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to one policy area per granule with transparent tie-handling and confidence.
    confidence = share of committee occurrences for the chosen area within that granule.
    """
    if links.empty:
        return pd.DataFrame(columns=[
            "granuleId","policy_area_name","policy_area_code","support","total_committees",
            "confidence","tie","tie_candidates"
        ])

    counts = (links.groupby(["granuleId","policy_area_name","policy_area_code"])
                   .size().reset_index(name="support"))
    totals = counts.groupby("granuleId")["support"].sum().reset_index(name="total_committees")
    merged = counts.merge(totals, on="granuleId", how="left")
    merged["confidence"] = merged["support"] / merged["total_committees"]

    # rank within granule by support, then by global POLICY_PRIORITY
    prio = {name: i for i, name in enumerate(POLICY_PRIORITY)}
    merged["prio"] = merged["policy_area_name"].map(lambda n: prio.get(n, len(POLICY_PRIORITY)))

    # find winners per granule
    merged = merged.sort_values(["granuleId","support","confidence"], ascending=[True, False, False])
    winners = merged.groupby("granuleId").head(2)  # inspect top-2 for tie detection

    out_rows = []
    for gid, g in winners.groupby("granuleId"):
        g = g.sort_values(["support","confidence","prio"], ascending=[False, False, True])
        top = g.iloc[0]
        tie_flag = False
        tie_cands: List[str] = []
        if len(g) > 1 and g.iloc[1]["support"] == top["support"]:
            # true tie on support: break with priority list but mark tie + record candidates
            tie_flag = True
            tie_cands = g[g["support"] == top["support"]]["policy_area_name"].tolist()
        out_rows.append({
            "granuleId": gid,
            "policy_area_name": top["policy_area_name"],
            "policy_area_code": int(top["policy_area_code"]) if pd.notna(top["policy_area_code"]) else None,
            "support": int(top["support"]),
            "total_committees": int(top["total_committees"]),
            "confidence": float(top["confidence"]),
            "tie": bool(tie_flag),
            "tie_candidates": "; ".join(sorted(set(tie_cands))) if tie_flag else "",
        })

    dom = pd.DataFrame(out_rows)
    dom.to_parquet(OUT_DIR / "granule_dominant_policy.parquet", index=False)
    dom.to_csv(OUT_DIR / "granule_dominant_policy.csv", index=False)
    return dom

def attach_policy_to_prominence(dom: pd.DataFrame, prominence_csv: Path) -> pd.DataFrame:
    """
    Optional: left-join dominant policy area onto your paragraph/prominence table by granuleId.
    """
    prom = load_df(prominence_csv, kind="csv")
    if prom.empty:
        logging.info("Prominence CSV empty or missing; skipping attach.")
        return pd.DataFrame()
    # ensure key exists
    if "granuleId" not in prom.columns:
        # attempt fallback
        if "granule_id" in prom.columns:
            prom = prom.rename(columns={"granule_id": "granuleId"})
        else:
            raise ValueError("prominence file must have 'granuleId'")
    enriched = prom.merge(dom, on="granuleId", how="left")
    enriched.to_parquet(OUT_DIR / "prominence_with_policy.parquet", index=False)
    enriched.to_csv(OUT_DIR / "prominence_with_policy.csv", index=False)
    return enriched

# =========================== RUNNER ===========================
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df_committees = load_df(COMMITTEES_CSV, kind="csv")
    if df_committees.empty:
        logging.error("Committees DataFrame is empty. Check path or data integrity.")
        sys.exit(1)

    mapping = build_mapping()
    links = build_committee_links(df_committees, mapping)
    if links.empty:
        logging.warning("No mapped committee links were produced.")
        sys.exit(0)

    dom = derive_dominant_policy_per_granule(links)

    # optional: attach to prominence table
    if PROMINENCE_CSV.exists():
        attach_policy_to_prominence(dom, PROMINENCE_CSV)

    # quick coverage report
    coverage = (len(dom), df_committees["granuleId"].nunique())
    logging.info("Granules with dominant policy: %d / %d", coverage[0], coverage[1])

if __name__ == "__main__":
    main()
