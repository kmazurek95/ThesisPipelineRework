# Interest Group Prominence in Congressional Speech

Analyzing what drives substantive vs. passing mentions of interest groups in U.S. Congressional floor debates.

---

## Research Context

Why do some advocacy organizations receive substantive attention from U.S. legislators while others are mentioned only in passing? This project investigates **prominence**, the degree to which politicians invoke an organization as a credible voice or useful resource during floor speeches, as a distinct form of political attention that is more tractable than measuring policy influence directly. The theoretical framework draws on Halpin & Fraussen (2017) for the conceptualization of prominence and the "audience dynamic," Grossman (2012) for organizational-level predictors, and Ibenskas & Bunea (2021) for the politician-interest group linkage.

The work has two phases. My MSc thesis at the University of Amsterdam (2023) developed the original research design: collecting Congressional Record documents via the GovInfo API, extracting mentions of 5,447 national advocacy organizations, training a supervised classifier to distinguish prominent from routine mentions, and running multilevel regression models to test hypotheses about what drives prominence. The full thesis argument, hypothesis tests, and results are in [THESIS_FINDINGS_2023.md](docs/THESIS_FINDINGS_2023.md) and the [thesis PDF](docs/Thesis_UvA_Kaleb_Mazurek.pdf).

After graduating, I rebuilt the pipeline from scratch, not to correct the thesis but to extend the methodology with a larger corpus, an improved classifier, and a fully reproducible architecture. This repository is that rebuild. The statistical findings here reflect the expanded dataset and should be read on their own terms; see [PIPELINE_FINDINGS.md](docs/PIPELINE_FINDINGS.md) for the full results.

## Key Findings

Lobbying expenditure predicts not just visibility but the *quality* of attention organizations receive, and institutional factors (chamber, party) shape prominence patterns independently of organization characteristics.

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

## Project Evolution

| | Original Thesis (2023) | Pipeline Rebuild |
|---|---|---|
| Corpus | 114th Congress (20,699 mentions) | 114th + 115th Congress (53,892 mentions) |
| Classifier | SVM (F1=0.79) | Logistic Regression (F1=0.91) |
| Infrastructure | Ad hoc scripts | 5-stage reproducible pipeline with CI/CD, tests, dashboard |

For a detailed comparison of where findings align and diverge, see [THESIS_EXTENSION_NOTES.md](docs/THESIS_EXTENSION_NOTES.md).

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
├── docs/                        # Methodology, findings, replication guide, thesis PDF
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

To run the full pipeline from scratch (requires GovInfo and Congress.gov API keys in `.env` (see `.env.example`)):

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
| [Thesis Findings (2023)](docs/THESIS_FINDINGS_2023.md) | Original thesis: theoretical framework, hypothesis tests, and all results |
| [Pipeline Findings](docs/PIPELINE_FINDINGS.md) | Pipeline rebuild: expanded dataset results and new findings |
| [Thesis Extension Notes](docs/THESIS_EXTENSION_NOTES.md) | How this pipeline extends the original thesis: where findings align and diverge |
| [Known Limitations](docs/KNOWN_LIMITATIONS.md) | Honest assessment of where the pipeline falls short and what I'd improve |
| [Methodology](docs/METHODOLOGY.md) | Research design, classification approach, statistical models |
| [Replication Guide](docs/REPLICATION.md) | Step-by-step instructions to reproduce the analysis |
| [Thesis PDF](docs/Thesis_UvA_Kaleb_Mazurek.pdf) | Full MSc thesis (University of Amsterdam, 2023) |

## Skills Demonstrated

- **Data Engineering**: Built a multi-source ETL pipeline pulling from GovInfo and Congress.gov APIs, with incremental processing and automated data validation
- **NLP / Text Classification**: Trained a TF-IDF + logistic regression classifier with group-aware cross-validation and SHAP-based feature attribution
- **Statistical Modeling**: Ran multilevel logistic regressions, OLS models, and mixed-effects GLMER (R/lme4) with variance decomposition
- **Reproducible Research**: Pipeline is configuration-driven with pinned dependencies and documented methodology
- **Visualization**: Interactive Streamlit dashboard (deployed), plus Plotly, matplotlib, seaborn, and ggplot2 figures

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
