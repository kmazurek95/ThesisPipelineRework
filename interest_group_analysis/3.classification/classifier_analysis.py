"""
Classifier Analysis and Visualization Module

This module provides functions for analyzing and visualizing the performance
of the prominence classifier. It includes:
- Confusion matrix visualization
- ROC and Precision-Recall curves
- Feature importance extraction
- SHAP value computation
- Error analysis utilities
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.calibration import CalibrationDisplay
import joblib


def load_trained_model(model_path: Path) -> Tuple[Any, float]:
    """Load trained model bundle with pipeline and threshold.

    Parameters
    ----------
    model_path : Path
        Path to the joblib model file

    Returns
    -------
    tuple
        (pipeline, threshold)
    """
    bundle = joblib.load(model_path)
    return bundle['pipeline'], bundle['threshold']


def get_predictions(pipe, X: pd.DataFrame, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Get predictions and probabilities from the pipeline.

    Parameters
    ----------
    pipe : Pipeline
        Trained sklearn pipeline
    X : pd.DataFrame
        Input features
    threshold : float
        Decision threshold

    Returns
    -------
    tuple
        (y_pred, y_prob)
    """
    if hasattr(pipe.named_steps['clf'], 'predict_proba'):
        y_prob = pipe.predict_proba(X)[:, 1]
    else:
        scores = pipe.decision_function(X)
        y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = ['Passing', 'Prominent'],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """Create publication-ready confusion matrix visualization.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    labels : List[str]
        Class labels for display
    output_path : Path, optional
        Path to save the figure
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Raw counts
    cm = confusion_matrix(y_true, y_pred)
    disp1 = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp1.plot(ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title('Confusion Matrix (Counts)')

    # Normalized
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    disp2 = ConfusionMatrixDisplay(cm_norm, display_labels=labels)
    disp2.plot(ax=axes[1], cmap='Blues', colorbar=False, values_format='.2%')
    axes[1].set_title('Confusion Matrix (Normalized)')

    plt.suptitle('Prominence Classification: Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """Create ROC and Precision-Recall curve visualization.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_prob : np.ndarray
        Predicted probabilities
    threshold : float
        Decision threshold to mark on PR curve
    output_path : Path, optional
        Path to save the figure
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ROC Curve
    roc_auc = roc_auc_score(y_true, y_prob)
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=axes[0], name='Classifier')
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0].set_title(f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[0].legend(loc='lower right')

    # Precision-Recall Curve
    ap = average_precision_score(y_true, y_prob)
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=axes[1], name='Classifier')
    axes[1].axhline(y=y_true.mean(), color='k', linestyle='--', label=f'Baseline ({y_true.mean():.2f})')
    axes[1].axvline(x=threshold, color='r', linestyle=':', alpha=0.7, label=f'Threshold = {threshold:.2f}')
    axes[1].set_title(f'Precision-Recall Curve (AP = {ap:.3f})')
    axes[1].legend(loc='lower left')

    plt.suptitle('Prominence Classification: Performance Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def get_feature_importance(
    pipe,
    num_features: List[str] = None,
    top_n: int = 20
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract feature importance from the trained pipeline.

    Parameters
    ----------
    pipe : Pipeline
        Trained sklearn pipeline
    num_features : List[str], optional
        Names of numeric features
    top_n : int
        Number of top features to return

    Returns
    -------
    tuple
        (top_positive, top_negative) DataFrames
    """
    if num_features is None:
        num_features = ['paragraph_mention_count', '10_or_more_org_mentioned']

    # Get components
    coltx = pipe.named_steps['coltx']
    clf = pipe.named_steps['clf']

    # Get TF-IDF feature names
    tfidf = coltx.transformers_[0][1]
    tfidf_features = list(tfidf.get_feature_names_out())

    # Combine all features
    all_features = tfidf_features + num_features
    coefficients = clf.coef_[0]

    # Create dataframe
    feature_df = pd.DataFrame({
        'feature': all_features,
        'coefficient': coefficients
    })
    feature_df['abs_coef'] = feature_df['coefficient'].abs()
    feature_df = feature_df.sort_values('abs_coef', ascending=False)

    # Get top positive and negative
    top_positive = feature_df.nlargest(top_n, 'coefficient')[['feature', 'coefficient']]
    top_negative = feature_df.nsmallest(top_n, 'coefficient')[['feature', 'coefficient']]

    return top_positive, top_negative


def plot_feature_importance(
    top_positive: pd.DataFrame,
    top_negative: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """Visualize top positive and negative features.

    Parameters
    ----------
    top_positive : pd.DataFrame
        Top features predicting prominence
    top_negative : pd.DataFrame
        Top features predicting passing
    output_path : Path, optional
        Path to save the figure
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Positive features
    axes[0].barh(range(len(top_positive)), top_positive['coefficient'].values, color='#4ecdc4')
    axes[0].set_yticks(range(len(top_positive)))
    axes[0].set_yticklabels(top_positive['feature'].values)
    axes[0].set_xlabel('Coefficient')
    axes[0].set_title('Top Features → PROMINENCE', fontweight='bold')
    axes[0].invert_yaxis()

    # Negative features
    axes[1].barh(range(len(top_negative)), top_negative['coefficient'].values, color='#ff6b6b')
    axes[1].set_yticks(range(len(top_negative)))
    axes[1].set_yticklabels(top_negative['feature'].values)
    axes[1].set_xlabel('Coefficient')
    axes[1].set_title('Top Features → PASSING', fontweight='bold')
    axes[1].invert_yaxis()

    plt.suptitle('Feature Importance (Logistic Regression Coefficients)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def analyze_errors(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    text_col: str = 'p1_original',
    org_col: str = 'interest_group'
) -> Dict[str, pd.DataFrame]:
    """Analyze classification errors.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe with text
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_prob : np.ndarray
        Predicted probabilities
    text_col : str
        Column name for text
    org_col : str
        Column name for organization

    Returns
    -------
    dict
        Dictionary with 'false_positives' and 'false_negatives' DataFrames
    """
    result_df = df.copy()
    result_df['y_true'] = y_true
    result_df['y_pred'] = y_pred
    result_df['y_prob'] = y_prob
    result_df['correct'] = result_df['y_true'] == result_df['y_pred']

    # False positives and negatives
    fp = result_df[(result_df['y_pred'] == 1) & (result_df['y_true'] == 0)]
    fn = result_df[(result_df['y_pred'] == 0) & (result_df['y_true'] == 1)]

    return {
        'false_positives': fp,
        'false_negatives': fn,
        'summary': {
            'total': len(result_df),
            'correct': result_df['correct'].sum(),
            'fp_count': len(fp),
            'fn_count': len(fn),
            'accuracy': result_df['correct'].mean()
        }
    }


def compute_metrics_at_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray = None
) -> pd.DataFrame:
    """Compute precision, recall, F1 at different thresholds.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_prob : np.ndarray
        Predicted probabilities
    thresholds : np.ndarray, optional
        Thresholds to evaluate

    Returns
    -------
    pd.DataFrame
        Metrics at each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)

    results = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        results.append({
            'threshold': thresh,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        })

    return pd.DataFrame(results)


def generate_full_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path = None
) -> Dict[str, Any]:
    """Generate comprehensive classification report.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_prob : np.ndarray
        Predicted probabilities
    output_dir : Path, optional
        Directory to save outputs

    Returns
    -------
    dict
        Dictionary with all metrics and reports
    """
    report = {
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'average_precision': average_precision_score(y_true, y_prob),
        'accuracy': (y_true == y_pred).mean(),
        'classification_report': classification_report(y_true, y_pred,
                                                       target_names=['Passing', 'Prominent'],
                                                       output_dict=True)
    }

    # Confusion matrix
    report['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_df = pd.DataFrame([{
            'Metric': k.replace('_', ' ').title(),
            'Value': f'{v:.4f}' if isinstance(v, float) else str(v)
        } for k, v in report.items() if k not in ['classification_report', 'confusion_matrix']])
        metrics_df.to_csv(output_dir / 'classifier_metrics.csv', index=False)

    return report


if __name__ == "__main__":
    # Example usage
    print("Classifier Analysis Module")
    print("=" * 50)
    print("\nAvailable functions:")
    print("  - load_trained_model(model_path)")
    print("  - get_predictions(pipe, X, threshold)")
    print("  - plot_confusion_matrix(y_true, y_pred)")
    print("  - plot_roc_pr_curves(y_true, y_prob)")
    print("  - get_feature_importance(pipe)")
    print("  - plot_feature_importance(top_pos, top_neg)")
    print("  - analyze_errors(df, y_true, y_pred, y_prob)")
    print("  - compute_metrics_at_thresholds(y_true, y_prob)")
    print("  - generate_full_report(y_true, y_pred, y_prob)")
