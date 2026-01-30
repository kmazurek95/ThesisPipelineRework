#!/usr/bin/env python3
"""
Regression Analysis of Interest Group Prominence

This script runs multi-level regression models to test hypotheses about
what factors predict interest group prominence in congressional speech.

Models:
1. Mention-level: What predicts high prominence mentions?
2. Organization-level: What organizational characteristics predict prominence?
3. Politician-level: Which politicians give more prominent mentions?

Output: outputs/tables/ (regression results)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "output"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load datasets for regression analysis."""
    level1 = pd.read_csv(DATA_DIR / "level1.csv", low_memory=False)
    level2 = pd.read_csv(DATA_DIR / "level2_org.csv")
    level3 = pd.read_csv(DATA_DIR / "level3_politician.csv")

    logger.info(f"Level 1: {len(level1):,} mentions")
    logger.info(f"Level 2: {len(level2):,} organizations")
    logger.info(f"Level 3: {len(level3):,} politicians")

    return level1, level2, level3


def prepare_mention_level(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare mention-level data for regression."""
    # Create analysis variables
    df = df.copy()

    # Binary outcomes
    df['prominent'] = df['prominence_prediction'].astype(int)

    # Organization characteristics
    df['has_lobbying'] = (df['LOBBYING11'] > 0).astype(int)
    df['log_lobbying'] = np.log1p(df['LOBBYING11'].fillna(0))
    df['has_inhouse'] = (df['IN_HOUSE11'] > 0).astype(int)
    df['has_outside'] = (df['OUTSIDE11'] > 0).astype(int)

    # Speaker characteristics
    df['is_democrat'] = (df['party'] == 'D').astype(int)
    df['is_republican'] = (df['party'] == 'R').astype(int)
    df['is_senate'] = (df['chamber'].isin(['S', 'Senate'])).astype(int)

    # Simplify category - extract main type
    df['category_simple'] = df['CATEGORY'].str.extract(r'^\((\d+)\)')[0].fillna('0').astype(int)

    # Category dummies for common types
    df['is_trade_assoc'] = df['category_simple'].isin([204, 205]).astype(int)
    df['is_labor'] = df['category_simple'].isin([301, 302, 303]).astype(int)
    df['is_single_issue'] = df['category_simple'].isin([1101, 1102, 1103, 1104]).astype(int)
    df['is_identity'] = df['category_simple'].isin([1401, 1402, 1403]).astype(int)

    # Policy area dummies
    df['is_defense'] = (df['issue_area_name'] == 'Defense').astype(int)
    df['is_health'] = (df['issue_area_name'] == 'Health').astype(int)
    df['is_econ'] = (df['issue_area_name'] == 'Macroeconomics').astype(int)

    # Salience
    df['salience_z'] = (df['salience'] - df['salience'].mean()) / df['salience'].std()
    df['salience_z'] = df['salience_z'].fillna(0)

    return df


def run_mention_level_model(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Model 1: What predicts high prominence at the mention level?

    DV: prominence_prediction (0/1)
    IVs: Organization characteristics, speaker characteristics, context
    """
    logger.info("Running Model 1: Mention-level logistic regression")

    # Filter to complete cases - use simpler model to avoid collinearity
    vars_needed = [
        'prominent', 'log_lobbying', 'has_lobbying',
        'is_democrat', 'is_senate',
        'is_trade_assoc', 'is_labor', 'is_single_issue'
    ]

    # Keep only mentions with speaker info
    df_model = df[df['party'].notna()][vars_needed].dropna()
    logger.info(f"Analysis sample: {len(df_model):,} mentions")

    # Check for variation in all variables
    for var in vars_needed:
        unique_vals = df_model[var].nunique()
        logger.info(f"  {var}: {unique_vals} unique values")

    # Formula - simplified to avoid collinearity
    formula = """
    prominent ~ log_lobbying +
                is_democrat + is_senate +
                is_trade_assoc + is_labor + is_single_issue
    """

    try:
        # Fit logit model with regularization to handle potential issues
        model = smf.logit(formula, data=df_model).fit(disp=0, method='bfgs', maxiter=100)

        print("\n" + "=" * 70)
        print("MODEL 1: Mention-Level Logistic Regression")
        print("DV: High Prominence (0/1)")
        print("=" * 70)
        print(model.summary2().tables[1].to_string())

        # Marginal effects
        try:
            mfx = model.get_margeff(at='mean')
            print("\nMarginal Effects (at means):")
            print(mfx.summary().tables[0].to_string())
        except Exception as e:
            logger.warning(f"Could not compute marginal effects: {e}")

        return model

    except Exception as e:
        logger.error(f"Logit model failed: {e}")
        logger.info("Falling back to Linear Probability Model (OLS)")

        # Fallback to OLS (Linear Probability Model)
        model = smf.ols(formula, data=df_model).fit()

        print("\n" + "=" * 70)
        print("MODEL 1: Mention-Level Linear Probability Model (OLS)")
        print("DV: High Prominence (0/1)")
        print("=" * 70)
        print(model.summary2().tables[1].to_string())

        return model


def run_org_level_model(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Model 2: What organizational characteristics predict prominence?

    DV: avg_prominence (average prominence score)
    IVs: Lobbying, category, membership status
    """
    logger.info("Running Model 2: Organization-level OLS")

    df = df.copy()

    # Use actual column names from level2
    lobbying_col = 'LOBBYING11' if 'LOBBYING11' in df.columns else 'total_lobbying_2011'
    mentions_col = 'total_mentions' if 'total_mentions' in df.columns else 'mention_count'
    prominence_col = 'avg_prominence' if 'avg_prominence' in df.columns else 'high_prominence_pct'
    category_col = 'CATEGORY' if 'CATEGORY' in df.columns else 'primary_category'

    # Create variables
    df['log_lobbying'] = np.log1p(df[lobbying_col].fillna(0))
    df['log_mentions'] = np.log1p(df[mentions_col])

    # Category dummies
    if category_col in df.columns:
        df['category_code'] = df[category_col].str.extract(r'^\((\d+)\)')[0].fillna('0').astype(int)
        df['is_trade'] = df['category_code'].isin([204, 205]).astype(int)
        df['is_labor'] = df['category_code'].isin([301, 302, 303]).astype(int)
        df['is_single_issue'] = df['category_code'].isin([1101, 1102, 1103, 1104]).astype(int)
    else:
        df['is_trade'] = 0
        df['is_labor'] = 0
        df['is_single_issue'] = 0

    # Filter
    df_model = df[df[mentions_col] >= 5].copy()  # At least 5 mentions
    df_model['prominence_dv'] = df_model[prominence_col]
    logger.info(f"Analysis sample: {len(df_model):,} organizations")

    # Formula
    formula = """
    prominence_dv ~ log_lobbying + log_mentions +
                    is_trade + is_labor + is_single_issue
    """

    # OLS (proportion DV)
    model = smf.ols(formula, data=df_model).fit()

    print("\n" + "=" * 70)
    print("MODEL 2: Organization-Level OLS")
    print("DV: Average Prominence Score")
    print("=" * 70)
    print(model.summary2().tables[1].to_string())

    return model


def run_politician_level_model(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Model 3: Which politicians give more prominent mentions?

    DV: avg_prominence (average prominence score)
    IVs: Party, chamber, mention volume
    """
    logger.info("Running Model 3: Politician-level OLS")

    df = df.copy()

    # Use actual column names from level3
    mentions_col = 'total_mentions' if 'total_mentions' in df.columns else 'mention_count'
    prominence_col = 'avg_prominence' if 'avg_prominence' in df.columns else 'high_prominence_pct'

    # Create variables
    df['is_democrat'] = (df['party'] == 'D').astype(int)
    df['is_senate'] = (df['chamber'].isin(['S', 'Senate'])).astype(int)
    df['log_mentions'] = np.log1p(df[mentions_col])

    # Filter to politicians with enough mentions
    df_model = df[df[mentions_col] >= 5].copy()
    df_model['prominence_dv'] = df_model[prominence_col]
    logger.info(f"Analysis sample: {len(df_model):,} politicians")

    # Formula
    formula = """
    prominence_dv ~ is_democrat + is_senate + log_mentions
    """

    model = smf.ols(formula, data=df_model).fit()

    print("\n" + "=" * 70)
    print("MODEL 3: Politician-Level OLS")
    print("DV: Average Prominence Score")
    print("=" * 70)
    print(model.summary2().tables[1].to_string())

    return model


def export_results(
    model1: sm.regression.linear_model.RegressionResultsWrapper,
    model2: sm.regression.linear_model.RegressionResultsWrapper,
    model3: sm.regression.linear_model.RegressionResultsWrapper,
    output_dir: Path
):
    """Export regression results to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract coefficients and SEs for each model
    results = []

    for model_name, model in [('Model 1: Mention', model1),
                               ('Model 2: Organization', model2),
                               ('Model 3: Politician', model3)]:
        for var in model.params.index:
            results.append({
                'Model': model_name,
                'Variable': var,
                'Coefficient': model.params[var],
                'Std Error': model.bse[var],
                'z/t': model.tvalues[var],
                'P-value': model.pvalues[var],
                'Significant': '***' if model.pvalues[var] < 0.001 else
                              '**' if model.pvalues[var] < 0.01 else
                              '*' if model.pvalues[var] < 0.05 else ''
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "regression_results.csv", index=False)
    logger.info(f"Results saved to {output_dir / 'regression_results.csv'}")

    # Model fit statistics
    fit_stats = pd.DataFrame([
        {
            'Model': 'Model 1: Mention',
            'N': model1.nobs,
            'Pseudo R2': model1.prsquared,
            'AIC': model1.aic,
            'BIC': model1.bic,
        },
        {
            'Model': 'Model 2: Organization',
            'N': model2.nobs,
            'R2': model2.rsquared,
            'Adj R2': model2.rsquared_adj,
            'F-stat': model2.fvalue,
        },
        {
            'Model': 'Model 3: Politician',
            'N': model3.nobs,
            'R2': model3.rsquared,
            'Adj R2': model3.rsquared_adj,
            'F-stat': model3.fvalue,
        },
    ])
    fit_stats.to_csv(output_dir / "model_fit_statistics.csv", index=False)

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run regression analysis")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model", choices=['1', '2', '3', 'all'], default='all',
                       help="Which model to run")
    args = parser.parse_args()

    # Load data
    logger.info("Loading data...")
    level1, level2, level3 = load_data()

    # Prepare data
    level1_prepped = prepare_mention_level(level1)

    # Run models
    model1 = model2 = model3 = None

    if args.model in ['1', 'all']:
        model1 = run_mention_level_model(level1_prepped)

    if args.model in ['2', 'all']:
        model2 = run_org_level_model(level2)

    if args.model in ['3', 'all']:
        model3 = run_politician_level_model(level3)

    # Export results
    if args.model == 'all' and all([model1, model2, model3]):
        export_results(model1, model2, model3, args.output_dir)

    logger.info("\nAnalysis complete!")


if __name__ == "__main__":
    main()
