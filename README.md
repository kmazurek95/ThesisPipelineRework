# Interest Group Prominence in Congressional Speech

Pipeline rebuild of my MSc thesis (University of Amsterdam, 2023). This repo is a ground-up rewrite of the data pipeline with an improved classifier, modular code, and reproducible workflows.

**Original thesis repository:** [MastersThesis_InterestGroupAnalysis](https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis)

**Interactive dashboard:** [Streamlit app](https://thesispipelinerework-emngd3hbxghtkfzbe9secw.streamlit.app/)

## What this project does

Analyzes 53,892 interest group mentions in the 114th and 115th U.S. Congress (2015-2019) to understand which organizations receive prominent vs. passing mentions in floor speeches, and what predicts prominence.

## Key findings

| Finding | Evidence |
|---------|----------|
| Lobbying predicts prominence | +7.4% higher odds per log unit increase (p < 0.001) |
| Senators > Representatives | +45% higher odds of prominent mentions (p < 0.001) |
| Democrats give fewer prominent mentions | -23% compared to Republicans (p < 0.001) |
| Single-issue groups get noticed | +41% higher prominence rate (p < 0.001) |

## Classification

The prominence classifier distinguishes substantive discussion of an organization from passing references.

- **Model:** Logistic Regression (L2, C=2.0) with TF-IDF features
- **Performance:** F1 = 0.91, Cohen's kappa = 0.82 (classifier-human agreement on held-out test set)
- **Training data:** 1,222 manually labeled mentions, coded following Fraussen et al. (2018)
- **Cross-validation:** Group-aware 5-fold (all mentions of a given org in the same fold)
- **Threshold:** Optimized on held-out test set (0.558)

## Statistical models

Three levels of analysis using mixed-effects logistic regression (R/lme4):

- **Mention-level:** Logistic regression with lobbying, party, chamber, org type
- **Organization-level:** OLS on aggregated prominence rates
- **Politician-level:** OLS on member-level prominence patterns

Random effects: Organization ID and Policy Area (crossed). Full model specifications and results in `R_analysis/Multilevel_Analysis.Rmd`.

## What changed from the original thesis

The original thesis pipeline (SVM classifier, F1 = 0.79) produced ~20,699 mentions. This rebuild produces 53,892 mentions from the same corpus. The difference comes from three sources: character-offset extraction instead of paragraph-level matching, less aggressive early filtering, and a more inclusive acronym policy. Full comparison in `docs/METHODOLOGY.md`.

## Data

| File | Level | Rows | Description |
|------|-------|------|-------------|
| `level1.csv` | Mention | 53,892 | Individual mention-level data |
| `level2_org.csv` | Organization | 2,260 | Aggregated by interest group |
| `level3_politician.csv` | Politician | 490 | Aggregated by Congress member |
| `level4_policy.csv` | Policy | 18 | Aggregated by policy area |

Sources: Congressional Record (GovInfo API), Washington Representatives Study (2011), Congress.gov API.

## Repository structure

```
interest_group_analysis/     Core Python package (collection, processing, classification, integration, analysis)
R_analysis/                  Multilevel models in R (lme4)
notebooks/                   Analysis notebooks (showcase, classifier deep-dive)
dashboard/                   Streamlit interactive dashboard
scripts/                     Data validation and API collection utilities
docs/                        Methodology and replication guide
data/                        Reference data, training data, and output datasets
outputs/                     Figures and regression tables
tests/                       Unit and integration tests
```

## Quick start

```bash
git clone https://github.com/kmazurek95/ThesisPipelineRework.git
cd ThesisPipelineRework
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Validate data
python scripts/validate_data.py

# Run analysis
jupyter notebook notebooks/Analysis_Showcase.ipynb

# R multilevel models
cd R_analysis && Rscript run_analysis.R

# Dashboard
streamlit run dashboard/Home.py
```

## Methodology

See `docs/METHODOLOGY.md` for detailed documentation of the classification pipeline, statistical models, validation strategy, and comparison with the original thesis pipeline.

## Citation

```bibtex
@software{mazurek2025interest,
  author = {Mazurek, Kaleb},
  title = {Interest Group Prominence in Congressional Speech},
  year = {2025},
  url = {https://github.com/kmazurek95/ThesisPipelineRework}
}
```

## License

MIT License - see [LICENSE](LICENSE).
