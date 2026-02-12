#!/usr/bin/env python3
"""
Classify interest group mentions using the trained text classifier.

This script reads a JSONL file containing interest group mentions,
extracts relevant features, applies the trained classifier to predict
prominence, and saves the results as a new JSONL file.
"""

# =============================================================================
# # Interest Group Mentions Classifier
#
# This script applies the trained prominence classifier to interest group mentions
# extracted from Congressional Record text. It:
#
# 1. Loads mentions data from a JSONL file
# 2. Prepares text and numerical features for each mention
# 3. Applies a pre-trained classifier to predict prominence
# 4. Adds prominence scores and predictions to each mention
# 5. Saves the classified mentions as both JSONL and CSV
#
# ## Usage
#
# ```powershell
# # Run from the project root directory
# cd "C:\Users\kaleb\OneDrive\Desktop\ThesisPipelineRework"
# python classify_mentions.py
# ```
#
# ## Input/Output
# 
# - **Input**: JSONL file with mentions (default: `data/processed/mentions_114/mentions.jsonl`)
# - **Model**: Pre-trained classifier (default: `results_classifier/prominence_pipeline.joblib`)
# - **Output**: JSONL and CSV files with added prominence predictions
#   - `data/processed/mentions_114/labeled_mentions.jsonl`
#   - `data/processed/mentions_114/labeled_mentions.csv`
#
# ## Requirements
#
# - Python 3.x
# - pandas, joblib, tqdm
# - A trained classifier model (created with `train_classifier.py` or the module version)
# =============================================================================
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd
import joblib
from tqdm import tqdm

def normalise_text(text: str) -> str:
    """Lowercase and remove non‑alphanumeric characters from a string."""
    if not isinstance(text, str):
        return ""
    lower = text.lower()
    # Replace non‑alphanumeric characters with spaces
    cleaned = re.sub(r"[^a-z0-9]+", " ", lower)
    # Collapse multiple spaces and strip
    return re.sub(r"\s+", " ", cleaned).strip()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger("classify_mentions")

# Required columns in the JSONL input
TEXT_COL = "paragraph"
MENTION_START_COL = "paragraph_char_start"
MENTION_END_COL = "paragraph_char_end"
SENTENCE_COL = "sentence"

def load_mentions_from_jsonl(jsonl_path: str) -> pd.DataFrame:
    """Load mentions from a JSONL file into a DataFrame."""
    LOGGER.info(f"Loading mentions from {jsonl_path}")
    mentions = []
    
    # Read the JSONL file line by line
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            try:
                mention = json.loads(line.strip())
                mentions.append(mention)
            except json.JSONDecodeError:
                LOGGER.warning(f"Failed to parse line: {line[:100]}...")
                continue
    
    df = pd.DataFrame(mentions)
    LOGGER.info(f"Loaded {len(df)} mentions from {jsonl_path}")
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for classification."""
    LOGGER.info("Preparing features for classification")
    
    # Create features needed by the classifier
    features = pd.DataFrame()
    
    # Use paragraph as the main text feature (p1_original)
    features["p1_original"] = df[TEXT_COL]
    
    # Calculate additional features
    
    # 1. Count of mentions in the same paragraph (by org_id)
    if "org_id" in df.columns and TEXT_COL in df.columns:
        # Group by org_id and paragraph text to count mentions per paragraph per organization
        mention_counts = df.groupby(["org_id", TEXT_COL]).size().reset_index(name="paragraph_mention_count")
        df = pd.merge(df, mention_counts, on=["org_id", TEXT_COL], how="left")
        features["paragraph_mention_count"] = df["paragraph_mention_count"]
    else:
        features["paragraph_mention_count"] = 1
    
    # 2. Flag if 10 or more organizations are mentioned in the same text
    if "org_id" in df.columns and TEXT_COL in df.columns:
        # Count unique org_ids per paragraph
        unique_orgs_per_paragraph = df.groupby(TEXT_COL)["org_id"].nunique().reset_index(name="unique_orgs")
        df = pd.merge(df, unique_orgs_per_paragraph, on=TEXT_COL, how="left")
        features["10_or_more_org_mentioned"] = (df["unique_orgs"] >= 10).astype(int)
    else:
        features["10_or_more_org_mentioned"] = 0
    
    LOGGER.info(f"Prepared features for {len(features)} mentions")
    return features

def classify_mentions(df: pd.DataFrame, features: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """Apply the trained classifier to mentions."""
    LOGGER.info(f"Loading model from {model_path}")
    model_bundle = joblib.load(model_path)
    pipeline = model_bundle["pipeline"]
    threshold = model_bundle["threshold"]
    
    LOGGER.info(f"Classifying {len(features)} mentions (using threshold {threshold:.3f})")
    
    # Get predictions
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        probas = pipeline.predict_proba(features)[:, 1]
    else:
        scores = pipeline.decision_function(features)
        # Normalize scores to 0..1 via min-max for simple thresholding
        smin, smax = scores.min(), scores.max()
        probas = (scores - smin) / (smax - smin + 1e-12)
    
    # Apply the threshold
    predictions = (probas >= threshold).astype(int)
    
    # Add predictions to the original DataFrame
    result_df = df.copy()
    result_df["prominence_score"] = probas
    result_df["prominence_prediction"] = predictions
    
    # Log prediction statistics
    positive_count = predictions.sum()
    LOGGER.info(f"Classified {len(predictions)} mentions: {positive_count} prominent ({positive_count/len(predictions):.1%})")
    
    return result_df

def save_jsonl(df: pd.DataFrame, output_path: str) -> None:
    """Save DataFrame as JSONL with one JSON object per line."""
    LOGGER.info(f"Saving {len(df)} classified mentions to {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            f.write(json.dumps(row.to_dict()) + "\n")
    
    LOGGER.info(f"Saved classified mentions to {output_path}")

def main():
    """Run the classification pipeline."""
    import argparse

    project_root = Path(__file__).resolve().parent.parent.parent

    parser = argparse.ArgumentParser(description="Classify interest group mentions")
    parser.add_argument(
        "--mentions", "-m",
        default=str(project_root / "data" / "intermediate" / "mentions_114" / "mentions.jsonl"),
        help="Path to mentions JSONL file"
    )
    parser.add_argument(
        "--model", "-M",
        default=str(project_root / "results_classifier" / "prominence_pipeline.joblib"),
        help="Path to trained model"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path (default: same dir as mentions with 'labeled_' prefix)"
    )
    args = parser.parse_args()

    # Define paths
    mentions_path = args.mentions
    model_path = args.model
    if args.output:
        output_path = args.output
    else:
        # Default: labeled_mentions.jsonl in same directory
        mentions_dir = Path(mentions_path).parent
        output_path = str(mentions_dir / "labeled_mentions.jsonl")
    
    # Create results directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load mentions
    mentions_df = load_mentions_from_jsonl(mentions_path)
    
    # Check for required columns
    required_cols = [TEXT_COL, MENTION_START_COL, MENTION_END_COL]
    missing = [col for col in required_cols if col not in mentions_df.columns]
    if missing:
        LOGGER.error(f"Missing required columns: {missing}")
        return
    
    # Prepare features
    features_df = prepare_features(mentions_df)
    
    # Classify mentions
    classified_df = classify_mentions(mentions_df, features_df, model_path)
    
    # Save results
    save_jsonl(classified_df, output_path)
    
    # Also save a CSV version for easier inspection
    csv_output = str(output_path).replace('.jsonl', '.csv')
    classified_df.to_csv(csv_output, index=False)
    LOGGER.info(f"Also saved CSV version to {csv_output}")
    
    LOGGER.info("Classification complete!")

if __name__ == "__main__":
    main()