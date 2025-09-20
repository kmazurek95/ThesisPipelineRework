"""Visualisation utilities for the integrated dataset."""

import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def generate_plots(input_path, output_dir):
    """Generate a set of exploratory plots from the integrated dataset.

    Parameters
    ----------
    input_path : Path or str
        Path to the integrated CSV dataset.
    output_dir : Path or str
        Directory where figures should be saved.

    Notes
    -----
    - The function creates histograms and boxplots for numeric
      variables found in the dataset.  You can modify the list of
      variables or add custom plots.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        logging.error("Failed to load data for visualisation: %s", exc)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        df[col].dropna().hist(bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        fig_path = output_dir / f"hist_{col}.png"
        plt.savefig(fig_path)
        plt.close()

        plt.figure(figsize=(4, 4))
        df[col].dropna().plot.box()
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        fig_path = output_dir / f"box_{col}.png"
        plt.savefig(fig_path)
        plt.close()