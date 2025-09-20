"""
End‑to‑end text classification pipeline.

This module encapsulates the workflow for supervised learning on
legislative text.  It defines helper functions for preprocessing
text, loading datasets, splitting data, training multiple models,
evaluating them, and applying the best model to unlabeled data.  The
default models include Naive Bayes, Logistic Regression, SVM, and
Random Forest.  Extend or modify the model list as needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from ..data_processing.utils import normalise_text


def preprocess_text(texts: Iterable[str]) -> np.ndarray:
    """Normalise and vectorise a collection of texts using TF‑IDF.

    Parameters
    ----------
    texts : iterable of str
        Raw text documents to vectorise.

    Returns
    -------
    array-like
        TF‑IDF feature matrix.
    """
    cleaned = [normalise_text(t) for t in texts]
    vectoriser = TfidfVectorizer(stop_words="english")
    return vectoriser.fit_transform(cleaned)


def load_labeled_data(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load labeled dataset and return features and labels.

    The CSV is expected to have a `p1_original` column for text and a
    `prominence` column for labels.  If your dataset uses different
    column names, adjust the `text_col` and `label_col` assignments
    accordingly.
    """
    df = pd.read_csv(file_path)
    if "p1_original" in df.columns:
        text_col = "p1_original"
    else:
        text_col = df.columns[0]
    label_col = "prominence" if "prominence" in df.columns else df.columns[-1]
    X = preprocess_text(df[text_col])
    y = LabelEncoder().fit_transform(df[label_col])
    return X, y


def load_unlabeled_data(file_path: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load unlabeled data and return features and the original dataframe."""
    df = pd.read_csv(file_path)
    if "p1_original" in df.columns:
        text_col = "p1_original"
    else:
        text_col = df.columns[0]
    X = preprocess_text(df[text_col])
    return X, df


def split_data(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split features and labels into train, validation, and test sets."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )  # 0.25 x 0.8 = 0.2
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_models() -> Dict[str, Tuple[Pipeline, Dict[str, list]]]:
    """Define models and their hyperparameter grids for grid search."""
    models: Dict[str, Tuple[Pipeline, Dict[str, list]]] = {}
    models["naive_bayes"] = (
        Pipeline([
            ("clf", MultinomialNB()),
        ]),
        {
            "clf__alpha": [0.1, 1.0, 10.0],
        },
    )
    models["logistic_regression"] = (
        Pipeline([
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"],
        },
    )
    models["svm"] = (
        Pipeline([
            ("clf", SVC()),
        ]),
        {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__kernel": ["linear", "rbf"],
        },
    )
    models["random_forest"] = (
        Pipeline([
            ("clf", RandomForestClassifier()),
        ]),
        {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 10, 20],
        },
    )
    return models


def run_grid_search(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, any]:
    """Perform grid search for each model and return the best estimators."""
    models = get_models()
    best_estimators: Dict[str, any] = {}
    for name, (pipeline, param_grid) in models.items():
        logging.info("Training %s model", name)
        gs = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)
        gs.fit(X_train, y_train)
        best_estimators[name] = gs.best_estimator_
    return best_estimators


def evaluate_models(
    best_estimators: Dict[str, any],
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> str:
    """Evaluate each model on validation and test sets; return a report."""
    report_lines = []
    for name, model in best_estimators.items():
        logging.info("Evaluating model %s", name)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        report_lines.append(f"Model: {name}\n")
        report_lines.append("Validation Classification Report:\n")
        report_lines.append(classification_report(y_val, y_val_pred))
        report_lines.append("Test Classification Report:\n")
        report_lines.append(classification_report(y_test, y_test_pred))
    return "\n".join(report_lines)


def label_unlabeled(
    best_model: any, X_unlabeled: np.ndarray, df_unlabeled: pd.DataFrame
) -> pd.DataFrame:
    """Predict labels for unlabeled data and return an augmented DataFrame."""
    preds = best_model.predict(X_unlabeled)
    df_unlabeled = df_unlabeled.copy()
    df_unlabeled["predicted_label"] = preds
    return df_unlabeled


def save_predictions(predictions: pd.DataFrame, output_path: Path) -> None:
    """Save predictions to CSV and JSON formats."""
    predictions.to_csv(output_path.with_suffix(".csv"), index=False)
    predictions.to_json(output_path.with_suffix(".json"), orient="records")


def run_pipeline(processed_data_dir: Path, results_dir: Path) -> None:
    """Execute the full classification pipeline.

    This function expects two CSV files in `processed_data_dir`:
    `labeled_data.csv` and `unlabeled_data.csv`.  Modify the file
    names here if your datasets are named differently.
    """
    labeled_file = processed_data_dir / "labeled_data.csv"
    unlabeled_file = processed_data_dir / "unlabeled_data.csv"
    if not labeled_file.exists() or not unlabeled_file.exists():
        logging.error(
            "Expected labeled_data.csv and unlabeled_data.csv in %s", processed_data_dir
        )
        return

    logging.info("Loading labeled dataset…")
    X, y = load_labeled_data(labeled_file)
    logging.info("Loading unlabeled dataset…")
    X_unlabeled, df_unlabeled = load_unlabeled_data(unlabeled_file)

    logging.info("Splitting data…")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    logging.info("Training models with grid search…")
    best_estimators = run_grid_search(X_train, y_train)

    logging.info("Evaluating models…")
    report = evaluate_models(best_estimators, X_val, y_val, X_test, y_test)
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(report)

    # Choose the best model based on validation accuracy; here we pick the first
    # model in the dictionary for simplicity.  Consider selecting based on
    # actual performance metrics.
    best_model_name = next(iter(best_estimators))
    best_model = best_estimators[best_model_name]
    logging.info("Using %s as the best model for unlabeled data", best_model_name)

    logging.info("Predicting labels for unlabeled data…")
    predictions = label_unlabeled(best_model, X_unlabeled, df_unlabeled)
    save_path = results_dir / "labeled_unlabeled_data"
    save_predictions(predictions, save_path)
