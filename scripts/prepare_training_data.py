"""
Prepare combined training data for prominence classifier.

This script combines labeled data from:
1. data/Labeled_Data.csv (366 rows) - current labeled data
2. data/legacy/labeled_windows_2023.xlsx - legacy labeled data with multiple sheets

The output is a deduplicated, cleaned dataset ready for training the
prominence classifier.

Usage:
    python scripts/prepare_training_data.py

Output:
    data/combined_labeled.csv - Combined training data
"""

from pathlib import Path
import pandas as pd
import numpy as np


# Required columns for the classifier
REQUIRED_COLS = ["org_id", "p1_original", "prominence"]
OPTIONAL_COLS = ["paragraph_mention_count", "10_or_more_org_mentioned"]
ID_COLS = ["uuid_mention", "uuid_paragraph", "granuleId"]


def load_current_labeled(path: Path) -> pd.DataFrame:
    """Load the current labeled data CSV."""
    df = pd.read_csv(path)
    df["source"] = "Labeled_Data.csv"
    return df


def load_legacy_labeled(path: Path) -> pd.DataFrame:
    """Load and combine legacy labeled data from Excel sheets."""
    xlsx = pd.ExcelFile(path)

    frames = []

    # Combined_Version____2 has 823 labeled rows (most useful)
    df_v2 = pd.read_excel(xlsx, sheet_name="Combined_Version____2")
    df_v2 = df_v2[df_v2["prominence"].notna()]
    df_v2["source"] = "Combined_Version____2"
    frames.append(df_v2)

    # Final_Dataset has additional labeled rows not in Combined_Version____2
    df_final = pd.read_excel(xlsx, sheet_name="Final_Dataset")
    df_final = df_final[df_final["prominence"].notna()]
    df_final["source"] = "Final_Dataset"
    frames.append(df_final)

    combined = pd.concat(frames, ignore_index=True)
    return combined


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column types and values."""
    df = df.copy()

    # Standardize prominence to int (0/1)
    if df["prominence"].dtype == bool:
        df["prominence"] = df["prominence"].astype(int)
    else:
        df["prominence"] = df["prominence"].astype(float).astype(int)

    # Ensure org_id is int
    df["org_id"] = df["org_id"].astype(int)

    # Fill optional columns with defaults
    for col in OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0).astype(int)

    # Clean p1_original (text column)
    df["p1_original"] = df["p1_original"].fillna("").astype(str)

    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows based on text content or unique identifiers."""
    original_len = len(df)

    # Try deduplication by uuid_mention first (most precise)
    if "uuid_mention" in df.columns:
        df = df.drop_duplicates(subset=["uuid_mention"], keep="first")
    else:
        # Fall back to text-based deduplication
        df = df.drop_duplicates(subset=["org_id", "p1_original"], keep="first")

    removed = original_len - len(df)
    if removed > 0:
        print(f"  Removed {removed} duplicate rows")

    return df


def validate_data(df: pd.DataFrame) -> bool:
    """Validate that data meets requirements for training."""
    issues = []

    # Check required columns
    for col in REQUIRED_COLS:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")

    # Check for empty text
    empty_text = (df["p1_original"] == "").sum()
    if empty_text > 0:
        issues.append(f"{empty_text} rows have empty p1_original")

    # Check class balance
    class_counts = df["prominence"].value_counts()
    if len(class_counts) < 2:
        issues.append("Only one class present in data")
    else:
        minority_pct = class_counts.min() / len(df) * 100
        if minority_pct < 20:
            print(f"  Warning: Minority class is only {minority_pct:.1f}% of data")

    if issues:
        for issue in issues:
            print(f"  ERROR: {issue}")
        return False

    return True


def main():
    base_dir = Path(__file__).parent.parent

    # Paths
    current_path = base_dir / "data" / "Labeled_Data.csv"
    legacy_path = base_dir / "data" / "legacy" / "labeled_windows_2023.xlsx"
    output_path = base_dir / "data" / "combined_labeled.csv"

    print("=" * 60)
    print("Preparing Combined Training Data")
    print("=" * 60)

    # Load current labeled data
    print(f"\n1. Loading current labeled data: {current_path}")
    df_current = load_current_labeled(current_path)
    print(f"   Loaded {len(df_current)} rows")

    # Load legacy labeled data
    print(f"\n2. Loading legacy labeled data: {legacy_path}")
    df_legacy = load_legacy_labeled(legacy_path)
    print(f"   Loaded {len(df_legacy)} rows from legacy sheets")

    # Combine
    print("\n3. Combining datasets...")
    df_combined = pd.concat([df_current, df_legacy], ignore_index=True)
    print(f"   Combined total: {len(df_combined)} rows")

    # Standardize
    print("\n4. Standardizing columns...")
    df_combined = standardize_columns(df_combined)

    # Deduplicate
    print("\n5. Removing duplicates...")
    df_combined = deduplicate(df_combined)
    print(f"   After deduplication: {len(df_combined)} rows")

    # Validate
    print("\n6. Validating data...")
    if not validate_data(df_combined):
        print("\n   Data validation failed. Please check the issues above.")
        return

    # Report class distribution
    print("\n7. Class distribution:")
    class_dist = df_combined["prominence"].value_counts()
    for cls, count in class_dist.items():
        pct = count / len(df_combined) * 100
        label = "Prominent" if cls == 1 else "Not Prominent"
        print(f"   {label} ({cls}): {count} ({pct:.1f}%)")

    # Report source distribution
    print("\n8. Source distribution:")
    source_dist = df_combined["source"].value_counts()
    for src, count in source_dist.items():
        print(f"   {src}: {count}")

    # Select columns to save
    cols_to_save = REQUIRED_COLS + OPTIONAL_COLS + ID_COLS + ["source", "interest_group", "variation"]
    cols_to_save = [c for c in cols_to_save if c in df_combined.columns]
    df_out = df_combined[cols_to_save]

    # Save
    print(f"\n9. Saving to: {output_path}")
    df_out.to_csv(output_path, index=False)
    print(f"   Saved {len(df_out)} rows with {len(cols_to_save)} columns")

    print("\n" + "=" * 60)
    print("Done! Ready to train classifier with:")
    print(f"  python -m interest_group_analysis.classification.text_classifier \\")
    print(f"    --labeled-path {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
