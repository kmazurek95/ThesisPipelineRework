# Interest Group Prominence in Congressional Speech

Analyzing what drives substantive vs. passing mentions of interest groups in U.S. Congressional floor debates.

---

## Overview

This repository is a ground-up rebuild of the data pipeline I developed for my master's thesis, "Beyond Policy Influence: A Deeper Dive into the Factors Driving Advocacy Group Prominence" (University of Amsterdam, 2023). The original thesis analyzed 20,699 mentions using an SVM classifier (F1=0.79). This pipeline extends that work with a larger corpus (53,892 mentions across two Congresses), an improved classifier (Logistic Regression, F1=0.91), and a fully reproducible architecture. The statistical findings in this repository reflect the expanded dataset and may differ from the original thesis results.

## Key Results

| Metric | Value |
|--------|-------|
| Total mentions extracted | 53,892 |
| Unique organizations | 2,260 |
| Classifier F1 score | 0.91 |
| Prominence rate | 35.5% |

| Finding | Effect Size | p-value |
|---------|-------------|---------|
| Lobbying expenditure predicts prominence | +7.4% odds per log-unit | < 0.001 |
| Senators give more prominent mentions | +45% odds vs. House | < 0.001 |
| Democrats give fewer prominent mentions | -23% odds vs. Republicans | < 0.001 |
| Single-issue groups get more prominence | +41% odds vs. multi-issue | < 0.001 |

## Repository Structure

```
ThesisPipelineRework/
├── interest_group_analysis/     # Core Python package (5 pipeline stages)
│   ├── 1.data_collection/       # GovInfo API, Congress.gov, Google Trends
│   ├── 2.data_processing/       # Normalization, mention extraction, speaker attribution
│   ├── 3.classification/        # TF-IDF + Logistic Regression prominence classifier
│   ├── 4_integration/           # Multi-level dataset assembly
│   └── 5_analysis/              # Regression models and visualizations
│
├── scripts/                     # Pipeline runner, data validation, API collectors
├── R_analysis/                  # Multilevel GLMER models (lme4)
├── dashboard/                   # Streamlit interactive dashboard
├── notebooks/                   # Analysis and classification notebooks
├── data/                        # Reference data, training labels, output datasets
├── outputs/                     # Figures and regression tables
├── docs/                        # Methodology, findings summary, replication guide
└── tests/                       # Pytest suite
```

## Quick Start

```bash
# Clone and set up
git clone https://github.com/kmazurek95/ThesisPipelineRework.git
cd ThesisPipelineRework
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Validate data integrity
python scripts/validate_data.py

# Build analysis datasets (from pre-processed data)
python -m interest_group_analysis.4_integration.build_analysis_dataset

# Generate figures and regression tables
python -m interest_group_analysis.5_analysis.descriptive_analysis
python -m interest_group_analysis.5_analysis.regression_analysis

# Run R multilevel models
cd R_analysis && Rscript run_analysis.R
```

To run the full pipeline from scratch (requires GovInfo and Congress.gov API keys in `.env` — see `.env.example`):

```bash
python scripts/run_pipeline.py          # all stages
python scripts/run_pipeline.py --dry-run  # preview without running
```

## Dashboard

The interactive dashboard is deployed on Streamlit Cloud:

**[View Live Dashboard](https://thesispipelinerework-emngd3hbxghtkfzbe9secw.streamlit.app/)**

Run locally:

```bash
streamlit run dashboard/Home.py
```

The dashboard includes methodology explanations, five organization case studies (AARP, AFL-CIO, ACLU, NAM, AMA), and a technical appendix with model diagnostics.

## Documentation

| Document | Description |
|----------|-------------|
| [Findings Summary](docs/FINDINGS_SUMMARY.md) | One-page research overview with key numbers |
| [Thesis Extension Notes](docs/THESIS_EXTENSION_NOTES.md) | How this pipeline extends the original thesis — what changed, where findings align/diverge |
| [Known Limitations](docs/KNOWN_LIMITATIONS.md) | Honest assessment of where the pipeline falls short and what I'd improve |
| [Methodology](docs/METHODOLOGY.md) | Research design, classification approach, statistical models |
| [Replication Guide](docs/REPLICATION.md) | Step-by-step instructions to reproduce the analysis |

## Skills Demonstrated

- **Data Engineering**: Multi-source ETL pipeline with GovInfo and Congress.gov APIs, incremental processing, data validation
- **NLP / Text Classification**: TF-IDF feature extraction, logistic regression, SHAP interpretability, group-aware cross-validation
- **Statistical Modeling**: Multilevel logistic regression, OLS, mixed-effects GLMER (lme4), variance decomposition
- **Reproducible Research**: Configuration-driven pipeline, pinned dependencies, documented methodology
- **Visualization**: Streamlit dashboard, Plotly interactives, publication-quality matplotlib/seaborn/ggplot2 figures

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

MIT License. See [LICENSE](LICENSE) for details.

---

<!-- GitHub repo topics to set manually: political-science, text-classification, multilevel-models, nlp, computational-social-science, congressional-record -->
