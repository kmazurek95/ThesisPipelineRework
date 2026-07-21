# Notebooks

## Core analysis

- `Analysis_Showcase.ipynb` - Executive summary: regression models, key visualizations, and findings across all four data levels
- `Classification_Analysis.ipynb` - ML classifier deep-dive: EDA, model comparison (NB/SVM/RF/LR), SHAP analysis, threshold optimization
- `Exploratory_Analysis.ipynb` - Initial data exploration and descriptive statistics
- `legacy_replication.ipynb` - Filter waterfall analysis: applies legacy filtering logic to revamp output (53,892 to 42,303)
- `legacy_record_match.ipynb` - Record-level comparison between legacy and revamp datasets (72.9% corrected match rate)
- `fill_replication_gaps.ipynb` - Documents gap-filling for policy salience, policy overlap, and bill sponsorship
- `replication_glmm.ipynb` - GLMM replication results and comparison to legacy thesis coefficients
- `member_profile_integration.ipynb` - Validates seniority and election timing data from congress-legislators

## Diagnostics (planning and validation)

- `diagnostics/replication_feasibility.ipynb` - Column-by-column audit of legacy vs revamp datasets
- `diagnostics/validate_partial_columns.ipynb` - Deep dive on salience and bills_referenced columns
- `diagnostics/policy_area_diagnostic.ipynb` - Policy area coverage analysis (26.7% committee-based)

## R analysis

- `../scripts/run_glmm_replication.R` - GLMM replication of three legacy thesis models (A, B, C) on revamp full-sample dataset
- `../R_analysis/Multilevel_Analysis.Rmd` - Mixed-effects regression with lme4: empty model (ICC), issue salience model, politician characteristics, organizational characteristics

## Setup

1. Install Python dependencies: `pip install -r requirements.txt`
2. Ensure pipeline data exists in `data/output/` (run `python scripts/run_pipeline.py --stage integrate` if needed)
3. For the Classification notebook, also install: `pip install wordcloud shap`
4. For the R analysis, see `R_analysis/README.md`
