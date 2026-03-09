#!/usr/bin/env python
"""Merge policy salience scores into the analysis dataset."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Load salience scores
sal = pd.read_csv(ROOT / "data" / "input" / "policy_salience_scores.csv")
print(f"Salience scores: {len(sal)} policy areas")

# Create mapping: issue_area_name -> salience
sal_map = dict(zip(sal["policy_area"], sal["salience_score"]))
cat_map = dict(zip(sal["policy_area"], sal["salience_category"]))

# Load analysis dataset
df = pd.read_csv(ROOT / "data" / "output" / "analysis_dataset_replication.csv",
                 low_memory=False)
print(f"Analysis dataset: {len(df):,} rows")

# Check what issue area column exists
if "issue_area_name" in df.columns:
    ia_col = "issue_area_name"
elif "issue_area" in df.columns:
    ia_col = "issue_area"
else:
    print("ERROR: No issue_area column found")
    exit(1)

print(f"Using '{ia_col}' for salience join")
print(f"  Non-null: {df[ia_col].notna().sum():,}")

# Map salience
df["salience_score"] = df[ia_col].map(sal_map)
df["salience_category"] = df[ia_col].map(cat_map)

# Drop the old constant salience column if it exists
if "salience" in df.columns:
    df = df.drop(columns=["salience"])

matched = df["salience_score"].notna().sum()
print(f"Salience matched: {matched:,} / {len(df):,}")
print(f"\nSalience distribution:")
print(df["salience_category"].value_counts(dropna=False))

# Save
df.to_csv(ROOT / "data" / "output" / "analysis_dataset_replication.csv", index=False)
print(f"\nSaved updated dataset")
