"""
Regression analysis on the integrated dataset.

This module defines functions to fit simple logistic regression models
to the integrated dataset.  The original thesis used more complex
mixed‑effects models in R; here we provide a basic example that can
be extended or replaced as needed.
"""

import logging
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def run_regression_analysis(input_path, output_dir):
    """Run a logistic regression on the integrated dataset.

    Parameters
    ----------
    input_path : Path or str
        Path to the integrated CSV dataset.
    output_dir : Path or str
        Directory where model summaries should be saved.

    Notes
    -----
    - The function uses a simple logistic regression as an example.
      Modify the formula to include the predictors present in your
      dataset.
    - To implement mixed‑effects models, consider using R via
      `rpy2` or Python packages that support such models.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        logging.error("Failed to load integrated dataset: %s", exc)
        return

    # Determine outcome variable
    outcome = None
    for candidate in ["prominence", "predicted_label"]:
        if candidate in df.columns:
            outcome = candidate
            break
    if outcome is None:
        logging.error(
            "Integrated dataset must contain 'prominence' or 'predicted_label' column."
        )
        return

    # Example predictors.  Update these names based on your dataset.
    possible_predictors = [
        "avg_interest",
        "lobbying_expenditure",
        "years_existed",
        "speaker_seniority",
    ]
    present_predictors = [col for col in possible_predictors if col in df.columns]
    if not present_predictors:
        logging.error("No recognised predictors found in the dataset.  Aborting regression.")
        return
    formula = outcome + " ~ " + " + ".join(present_predictors)
    logging.info("Fitting logistic regression: %s", formula)
    try:
        model = smf.logit(formula=formula, data=df).fit(disp=False)
    except Exception as exc:
        logging.error("Error fitting regression model: %s", exc)
        return

    summary_str = model.summary().as_text()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "regression_summary.txt").open("w", encoding="utf-8") as f:
        f.write(summary_str)
