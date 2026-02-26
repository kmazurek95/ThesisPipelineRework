# Changelog

All notable changes to this project are documented in this file.

## [2.0.0] - 2025

### Added
- Extended analysis to 115th Congress (2017-2019), doubling the dataset to 53,892 mentions
- Interactive Streamlit dashboard with live deployment on Streamlit Cloud
- R multilevel models (lme4 GLMER with crossed random effects)
- Data validation with pytest
- CI/CD pipeline with GitHub Actions
- Pipeline findings summary (PIPELINE_FINDINGS.md)
- Original thesis findings summary (THESIS_FINDINGS_2023.md)
- Classification Analysis notebook with SHAP explanations and model comparison
- Organization case studies dashboard page (5 curated organizations)

### Changed
- Expanded dataset: 53,892 mentions (from 25,106), 2,260 organizations (from 1,679)
- Expanded training set: 1,222 hand-labeled examples (from 907)
- Compressed large data files (level1.csv.gz) for GitHub compatibility
- Dashboard redesigned as story-driven portfolio showcase
- All documentation updated for 114th-115th Congress coverage

## [1.0.0] - 2024

### Added
- Initial production pipeline implementation
- 114th Congress analysis (25,106 mentions across 1,679 organizations)
- TF-IDF + Logistic Regression classifier (F1 = 0.91)
- Mention-level, organization-level, and politician-level regression models
- GovInfo and Congress.gov API integration
- Modular Python package structure
- Methodology and replication documentation
