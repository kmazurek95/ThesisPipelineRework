# Analysis Notebooks

Interactive notebooks documenting the analysis pipeline and findings.

## Notebook Guide

| Notebook | Description | Key Outputs |
|----------|-------------|-------------|
| [Analysis_Showcase.ipynb](Analysis_Showcase.ipynb) | Executive summary: regression models, key visualizations, and findings across all four data levels | 5 figures, 3 regression models |
| [Classification_Analysis.ipynb](Classification_Analysis.ipynb) | ML classifier deep-dive: EDA, model comparison (NB/SVM/RF/LR), SHAP analysis, error analysis, threshold optimization | 10 figures, 2 metric tables |
| [Exploratory Analysis](../interest_group_analysis/5_analysis/Exploratory%20Analysis.ipynb) | Initial data exploration and descriptive statistics | Exploratory plots |

## R Analysis

| File | Description |
|------|-------------|
| [Multilevel_Analysis.Rmd](../R_analysis/Multilevel_Analysis.Rmd) | Mixed-effects regression with lme4: empty model (ICC), issue salience model, politician characteristics, organizational characteristics, model comparison | Forest plots, coefficient tables |

## Setup

1. Install Python dependencies: `pip install -r requirements.txt`
2. Ensure pipeline data exists in `data/output/` (run `python scripts/run_pipeline.py --stage integrate` if needed)
3. For the Classification notebook, also install: `pip install wordcloud shap`
4. For the R analysis, see [R_analysis/README.md](../R_analysis/README.md)

## Data Requirements

All notebooks read from:
- `data/output/level1.csv` — 53,892 mention-level records
- `data/output/level2_org.csv` — 2,260 organization aggregates
- `data/output/level3_politician.csv` — 490 politician aggregates
- `data/output/level4_policy.csv` — 18 policy area aggregates
- `data/training/combined_labeled.csv` — 1,222 labeled training examples (Classification notebook only)
- `results_classifier/prominence_pipeline.joblib` — trained model (Classification notebook only)
