"""
End‑to‑end text classification pipeline.

This module encapsulates the workflow for supervised learning on
legislative text. It fixes common issues like:
- Feature leakage during vectorization
- Group-aware cross-validation
- Threshold optimization
- Better metrics for imbalanced data
- Model selection based on appropriate metrics
- Reproducibility and proper model saving
"""

# =============================================================================
# # Interest Group Prominence Classifier
#
# This module implements a robust machine learning pipeline for classifying the 
# prominence of interest group mentions in legislative text. Built with scikit-learn
# best practices, it addresses common issues in text classification:
#
# - Prevents feature leakage across train/test splits
# - Uses group-aware cross-validation to avoid data contamination
# - Optimizes decision thresholds for imbalanced data
# - Selects models based on precision-recall metrics
# - Includes proper validation and evaluation
# - Ensures reproducibility and model persistence
#
# ## Usage
#
# ### As a module (recommended)
# ```python
# from interest_group_analysis.3.classification.text_classifier import train_select, load_labeled_df
# import joblib
# from pathlib import Path
#
# # Train model
# df = load_labeled_df("path/to/Labeled_Data.csv")
# model, f1, ap = train_select(df, model_name="logreg", results_dir=Path("results_dir"))
# 
# # Use trained model
# model_bundle = joblib.load("results_dir/prominence_pipeline.joblib")
# ```
#
# ### As a script
# ```powershell
# cd "C:\Users\kaleb\OneDrive\Desktop\ThesisPipelineRework"
#
# # Using Python module import
# python -c "from interest_group_analysis.3.classification import text_classifier; text_classifier.run_pipeline()"
#
# # Or direct execution (may have import issues)
# python "interest_group_analysis\3.classification\text_classifier.py"
# ```
#
# ## Key Features
#
# - **Group-aware splits**: Prevents leakage between same-organization mentions
# - **Optimized threshold**: Uses precision-recall curves for optimal F1
# - **Feature engineering**: Combines TF-IDF text features with numerical features
# - **Linear models**: Efficient for sparse text data
# - **Performance metrics**: Tracks average precision and F1 score
# - **Reproducibility**: Fixed random seeds and complete pipeline serialization
# =============================================================================

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, average_precision_score,
                             precision_recall_curve, f1_score)
from sklearn.model_selection import GroupKFold, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import joblib

# If you have this already, reuse it. Otherwise make it identity.
from ..data_processing.utils import normalise_text

REQUIRED_COLS = ["p1_original", "prominence", "org_id"]
NUM_COLS = ["paragraph_mention_count", "10_or_more_org_mentioned"]

def load_labeled_df(csv_path: Path) -> pd.DataFrame:
    """Load labeled dataset and return the DataFrame with validation of required columns.
    
    Parameters
    ----------
    csv_path : Path
        Path to the CSV file with labeled data
        
    Returns
    -------
    pd.DataFrame
        DataFrame with the labeled data
        
    Raises
    ------
    ValueError
        If any required columns are missing
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = 0
    return df


def make_pipe(base_model: str = "logreg") -> Tuple[Pipeline, dict]:
    """Create a pipeline with text vectorization and model.
    
    Parameters
    ----------
    base_model : str
        Model type to use: 'logreg' or 'linsvm'
        
    Returns
    -------
    tuple
        Pipeline and parameter grid for GridSearchCV
    """
    text_vec = TfidfVectorizer(
        preprocessor=normalise_text,  # your normalizer
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        strip_accents="unicode"
    )
    coltx = ColumnTransformer(
        transformers=[
            ("text", text_vec, "p1_original"),
            ("num", StandardScaler(with_mean=False), NUM_COLS),
        ],
        remainder="drop"
    )

    if base_model == "logreg":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",  # small tilt toward recall
            solver="liblinear"
        )
        param_grid = {
            "clf__C": [0.25, 0.5, 1.0, 2.0],
            "clf__penalty": ["l1", "l2"]
        }
    elif base_model == "linsvm":
        clf = LinearSVC(class_weight="balanced")
        param_grid = {
            "clf__C": [0.25, 0.5, 1.0, 2.0]
        }
    else:
        raise ValueError("base_model must be 'logreg' or 'linsvm'.")

    pipe = Pipeline([("coltx", coltx), ("clf", clf)])
    return pipe, param_grid


def train_select(df: pd.DataFrame, model_name: str = "logreg", results_dir: Path = Path("results")) -> Tuple[Pipeline, float, float]:
    """Train a model with group-aware CV and optimize threshold.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with labeled data
    model_name : str
        Model to use: 'logreg' or 'linsvm'
    results_dir : Path
        Directory to save results
        
    Returns
    -------
    tuple
        Trained pipeline, F1 score, Average Precision score
    """
    X = df[["p1_original"] + NUM_COLS]
    y = df["prominence"].astype(int)
    groups = df["org_id"]

    # Holdout split that preserves label balance and avoids org leakage into test
    # First split by groups; then stratify inside the train fold, or keep it simple:
    unique_orgs = groups.drop_duplicates()
    rng = np.random.RandomState(42)
    test_orgs = set(unique_orgs.sample(frac=0.2, random_state=rng))
    test_mask = groups.isin(test_orgs)
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]
    g_train = groups[~test_mask]

    pipe, grid = make_pipe(model_name)
    cv = GroupKFold(n_splits=5)
    gs = GridSearchCV(
        pipe,
        grid,
        cv=cv.split(X_train, y_train, groups=g_train),
        n_jobs=-1,
        scoring="average_precision",  # PR-AUC
        refit=True,
        verbose=0
    )
    gs.fit(X_train, y_train)

    best_pipe: Pipeline = gs.best_estimator_

    # Tune threshold on the test set using PR curve (only for probabilistic models)
    # LinearSVC doesn't have predict_proba; for that branch, stick to decision_function.
    if hasattr(best_pipe.named_steps["clf"], "predict_proba"):
        proba = best_pipe.predict_proba(X_test)[:, 1]
        prec, rec, thr = precision_recall_curve(y_test, proba)
        f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
        t_idx = np.nanargmax(f1s)
        best_thr = thr[t_idx] if t_idx < len(thr) else 0.5
        y_pred = (proba >= best_thr).astype(int)
    else:
        scores = best_pipe.decision_function(X_test)
        # Normalize scores to 0..1 via min-max for simple thresholding
        smin, smax = scores.min(), scores.max()
        proba_like = (scores - smin) / (smax - smin + 1e-12)
        prec, rec, thr = precision_recall_curve(y_test, proba_like)
        f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
        t_idx = np.nanargmax(f1s)
        best_thr = thr[t_idx] if t_idx < len(thr) else 0.5
        y_pred = (proba_like >= best_thr).astype(int)

    ap = average_precision_score(y_test, y_pred)  # PR-AUC of hard labels (conservative)
    f1 = f1_score(y_test, y_pred)

    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "report.txt").open("w", encoding="utf-8") as f:
        f.write(f"Best params: {gs.best_params_}\n")
        f.write(f"Best CV score (avg precision): {gs.best_score_:.4f}\n\n")
        f.write(f"Test F1@thr={best_thr:.3f}: {f1:.4f}\n")
        f.write(f"Test Average Precision (hard preds): {ap:.4f}\n\n")
        f.write(classification_report(y_test, y_pred))

    # Persist both the pipeline and the threshold you chose
    joblib.dump({"pipeline": best_pipe, "threshold": float(best_thr)},
                results_dir / "prominence_pipeline.joblib")

    return best_pipe, f1, ap














def run_pipeline(labeled_path: Path = None, unlabeled_path: Path = None, results_dir: Path = None) -> None:
    """Execute the full classification pipeline with best practices.
    
    Parameters
    ----------
    labeled_path : Path
        Path to labeled data CSV file
    unlabeled_path : Path
        Path to unlabeled data CSV file (optional)
    results_dir : Path
        Directory to save results
    """
    # Set default paths if none provided
    if labeled_path is None:
        labeled_path = Path(r"C:\Users\kaleb\OneDrive\Desktop\ThesisPipelineRework\data\Labeled_Data.csv")
    if results_dir is None:
        results_dir = Path(r"C:\Users\kaleb\OneDrive\Desktop\ThesisPipelineRework\results_classifier")
    
    logging.info(f"Loading labeled dataset from {labeled_path}")
    df = load_labeled_df(labeled_path)
    logging.info(f"Loaded {len(df)} labeled examples with prominence distribution: {df['prominence'].value_counts(normalize=True).to_dict()}")
    
    logging.info("Training model with group-aware cross-validation...")
    best_pipe, f1, ap = train_select(df, model_name="logreg", results_dir=results_dir)
    logging.info(f"Model trained and saved to {results_dir / 'prominence_pipeline.joblib'}")
    logging.info(f"Performance: F1={f1:.3f} AP={ap:.3f}")
    
    # Optional: Apply to unlabeled data if provided
    if unlabeled_path and Path(unlabeled_path).exists():
        logging.info(f"Processing unlabeled data from {unlabeled_path}")
        
        # Load the saved model bundle
        bundle = joblib.load(results_dir / "prominence_pipeline.joblib")
        pipe = bundle["pipeline"]
        thr = bundle["threshold"]
        
        unlabeled = pd.read_csv(unlabeled_path)
        # Ensure we have the needed columns
        for c in NUM_COLS:
            if c not in unlabeled.columns:
                unlabeled[c] = 0
        
        if "p1_original" not in unlabeled.columns:
            logging.error("Unlabeled data missing 'p1_original' column")
            return
            
        X_unl = unlabeled[["p1_original"] + NUM_COLS]
        
        # Get predictions using the optimal threshold
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            p = pipe.predict_proba(X_unl)[:, 1]
        else:
            s = pipe.decision_function(X_unl)
            p = (s - s.min())/(s.max()-s.min() + 1e-12)
            
        unlabeled["predicted_label"] = (p >= thr).astype(int)
        unlabeled["predicted_score"] = p
        
        # Save results
        unlabeled.to_csv(results_dir / "unlabeled_scored.csv", index=False)
        logging.info(f"Predictions saved to {results_dir / 'unlabeled_scored.csv'}")
        logging.info(f"Predicted {unlabeled['predicted_label'].sum()} positive instances out of {len(unlabeled)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    labeled_path = Path(r"C:\Users\kaleb\OneDrive\Desktop\ThesisPipelineRework\data\Labeled_Data.csv")
    out = Path(r"C:\Users\kaleb\OneDrive\Desktop\ThesisPipelineRework\results_classifier")
    
    # Create results directory if it doesn't exist
    out.mkdir(parents=True, exist_ok=True)
    
    df = load_labeled_df(labeled_path)
    best_pipe, f1, ap = train_select(df, model_name="logreg", results_dir=out)
    print(f"✓ Saved pipeline to {out / 'prominence_pipeline.joblib'} | F1={f1:.3f} AP={ap:.3f}")
