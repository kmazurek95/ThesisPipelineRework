# Replication guide

This document explains how to reproduce the analysis. You'll need Python 3.10+ and optionally R 4.0+ for the multilevel models. If you want to re-collect data from scratch (most people won't), you'll also need API keys from GovInfo and Congress.gov.

## Setup

```bash
git clone https://github.com/kmazurek95/ThesisPipelineRework.git
cd ThesisPipelineRework
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For R analysis, install the required packages:

```r
install.packages(c('tidyverse', 'lme4', 'broom.mixed', 'sjPlot', 'performance', 'knitr', 'rmarkdown'))
```

Verify the installation with `pytest tests/ -v` and `python scripts/validate_data.py`.

## Reproducing the analysis (recommended path)

The repository includes pre-processed output data in `data/output/`. The mention-level file is gzip compressed (`level1.csv.gz`, ~14MB). The organization, politician, and policy area aggregates are in `level2_org.csv`, `level3_politician.csv`, and `level4_policy.csv`. No additional data acquisition is needed.

To generate the analysis outputs:

```bash
# Descriptive statistics and figures
python -m interest_group_analysis.5_analysis.descriptive_analysis

# Regression models and coefficient tables
python -m interest_group_analysis.5_analysis.regression_analysis

# R multilevel models (from R_analysis/ directory)
cd R_analysis
Rscript run_analysis.R
# Or render the full report:
Rscript -e "rmarkdown::render('Multilevel_Analysis.Rmd')"
```

Figures go to `outputs/figures/`, tables to `outputs/tables/`, and R output to `R_analysis/outputs/`.

The interactive notebooks provide a walkthrough of the classification pipeline (`notebooks/Classification_Analysis.ipynb`) and key findings (`notebooks/Analysis_Showcase.ipynb`).

## Reproducing from raw data (full pipeline)

If you want to re-collect and process everything from scratch, copy `.env.example` to `.env` and add your API keys, then run the pipeline stages in order:

```bash
python -m interest_group_analysis.1_data_collection.govinfo_collector
python scripts/collect_bills.py
python scripts/collect_members.py
python -m interest_group_analysis.2_data_processing.normalize_speeches
python -m interest_group_analysis.2_data_processing.extract_mentions
python -m interest_group_analysis.3_classification.classify_mentions
python -m interest_group_analysis.4_integration.build_analysis_dataset
```

The GovInfo collection step takes several hours due to API rate limits. Everything else runs in minutes.

## Common issues

If you get import errors for `interest_group_analysis`, run `pip install -e .` from the project root. The mention-level data is stored compressed as `level1.csv.gz`, not `level1.csv`. R packages need to be installed separately from Python dependencies.

---

*See [METHODOLOGY.md](METHODOLOGY.md) for research design and statistical model specifications.*
