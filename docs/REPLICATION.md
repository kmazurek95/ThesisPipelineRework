# Replication Guide

Step-by-step instructions to reproduce the analysis in this project.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Data Acquisition](#data-acquisition)
4. [Running the Pipeline](#running-the-pipeline)
5. [Reproducing Results](#reproducing-results)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Core pipeline |
| R | 4.0+ | Multilevel models |
| Git | 2.0+ | Version control |
| pip | 21.0+ | Package management |

### Hardware Requirements

- **RAM:** 8GB minimum (16GB recommended for full pipeline)
- **Storage:** 2GB for code and data
- **CPU:** Any modern processor (no GPU required)

### API Keys (Optional)

For data collection from scratch:
- GovInfo API key: https://api.govinfo.gov/docs/
- Congress.gov API key: https://api.congress.gov/

---

## Environment Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/kmazurek95/ThesisPipelineRework.git
cd ThesisPipelineRework
```

### Step 2: Create Python Virtual Environment

```bash
# Create environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### Step 3: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install R Dependencies (Optional)

For multilevel model analysis:

```bash
cd R_analysis
Rscript -e "install.packages(c('tidyverse', 'lme4', 'broom.mixed', 'sjPlot', 'performance', 'knitr', 'rmarkdown'))"
```

Or use the requirements file:

```r
# In R console
packages <- readLines("requirements-r.txt")
packages <- packages[!grepl("^#", packages) & packages != ""]
install.packages(packages)
```

### Step 5: Verify Installation

```bash
# Run tests
pytest tests/ -v

# Check data validation
python scripts/validate_data.py
```

---

## Data Acquisition

### Option A: Use Provided Output Data (Recommended)

The repository includes pre-processed output data in `data/output/`:

| File | Size | Description |
|------|------|-------------|
| `level1.csv.gz` | ~14MB | Mention-level data (gzip compressed) |
| `level2_org.csv` | ~500KB | Organization aggregates |
| `level3_politician.csv` | ~100KB | Politician aggregates |
| `level4_policy.csv` | ~10KB | Policy area aggregates |

**No additional data acquisition needed for analysis replication.**

### Option B: Full Pipeline from Scratch

To reproduce from raw data collection:

#### 1. Set Up API Keys

```bash
# Create .env file
cp .env.example .env

# Edit with your keys
# GOVINFO_API_KEY=your_key_here
# CONGRESS_API_KEY=your_key_here
```

#### 2. Collect Congressional Record

```bash
python -m interest_group_analysis.1_data_collection.govinfo_collector
```

**Note:** This may take several hours due to API rate limits.

#### 3. Collect Bill/Member Metadata

```bash
python scripts/collect_bills.py
python scripts/collect_members.py
```

#### 4. Process and Integrate

```bash
# Run full pipeline
python -m interest_group_analysis.2_data_processing.normalize_speeches
python -m interest_group_analysis.2_data_processing.extract_mentions
python -m interest_group_analysis.3_classification.classify_mentions
python -m interest_group_analysis.4_integration.build_analysis_dataset
```

---

## Running the Pipeline

### Validate Data Integrity

Before any analysis, verify data integrity:

```bash
python scripts/validate_data.py
```

Expected output:
```
Validating data/output/level1.csv...
  ✓ File exists
  ✓ Required columns present
  ✓ No duplicate mention IDs
  ✓ Valid prominence values (0/1)
  ✓ Valid party values (D/R/I)

All validations passed!
```

### Generate Analysis Outputs

#### Descriptive Statistics

```bash
python -m interest_group_analysis.5_analysis.descriptive_analysis
```

**Outputs:**
- `outputs/figures/fig1_mentions_over_time.png`
- `outputs/figures/fig2_org_categories.png`
- `outputs/tables/summary_statistics.csv`

#### Regression Analysis

```bash
python -m interest_group_analysis.5_analysis.regression_analysis
```

**Outputs:**
- `outputs/figures/fig3_lobbying_prominence.png`
- `outputs/tables/regression_results.csv`
- `outputs/tables/model_fit_stats.csv`

#### R Multilevel Models

```bash
cd R_analysis
Rscript run_analysis.R
```

Or render the R Markdown report:

```bash
Rscript -e "rmarkdown::render('Multilevel_Analysis.Rmd')"
```

**Outputs:**
- `R_analysis/Multilevel_Analysis.html`
- `R_analysis/outputs/coefficient_plot.png`
- `R_analysis/outputs/model_comparison.csv`

---

## Reproducing Results

### Key Tables

#### Table 1: Regression Results

Location: `outputs/tables/regression_results.csv`

Reproduce:
```bash
python -m interest_group_analysis.5_analysis.regression_analysis
```

#### Table 2: Model Fit Statistics

Location: `outputs/tables/model_fit_stats.csv`

Reproduce:
```bash
python -m interest_group_analysis.5_analysis.regression_analysis
```

### Key Figures

#### Figure 1: Mentions Over Time

Location: `outputs/figures/fig1_mentions_over_time.png`

Reproduce:
```bash
python -m interest_group_analysis.5_analysis.descriptive_analysis
```

#### Figure 3: Lobbying vs Prominence

Location: `outputs/figures/fig3_lobbying_prominence.png`

Reproduce:
```bash
python -m interest_group_analysis.5_analysis.regression_analysis
```

### Jupyter Notebooks

#### Classification Analysis

```bash
jupyter notebook notebooks/Classification_Analysis.ipynb
```

Run all cells to reproduce:
- Confusion matrix
- ROC/PR curves
- SHAP analysis
- Error analysis

#### Analysis Showcase

```bash
jupyter notebook notebooks/Analysis_Showcase.ipynb
```

Interactive summary of key findings.

---

## Troubleshooting

### Common Issues

#### Import Errors

```
ModuleNotFoundError: No module named 'interest_group_analysis'
```

**Solution:** Install package in development mode:
```bash
pip install -e .
```

#### Missing Data Files

```
FileNotFoundError: data/output/level1.csv (or level1.csv.gz)
```

**Solution:** The mention-level data is stored as `level1.csv.gz` (gzip compressed). Ensure you're in the project root:
```bash
cd ThesisPipelineRework
ls data/output/
```

#### R Package Errors

```
Error in library(lme4) : there is no package called 'lme4'
```

**Solution:** Install R packages:
```r
install.packages("lme4")
```

#### Memory Errors

```
MemoryError: Unable to allocate array
```

**Solution:** Process data in chunks or increase available RAM. The full Level 1 dataset requires ~1GB RAM.

### Getting Help

1. Check existing issues: https://github.com/kmazurek95/ThesisPipelineRework/issues
2. Run diagnostic script:
   ```bash
   python scripts/diagnose_environment.py
   ```
3. Open a new issue with:
   - Error message
   - Python/R version
   - Operating system

---

## Verification Checklist

After running the pipeline, verify:

- [ ] `outputs/figures/` contains 5+ PNG files
- [ ] `outputs/tables/regression_results.csv` has 3+ rows
- [ ] `python scripts/validate_data.py` passes
- [ ] `pytest tests/` shows all tests passing
- [ ] Dashboard runs: `streamlit run dashboard/Home.py`

---

## Citation

If you use this code or reproduce the analysis, please cite:

```bibtex
@software{mazurek2025interest,
  author = {Mazurek, Kaleb},
  title = {Interest Group Prominence in Congressional Speech},
  year = {2025},
  url = {https://github.com/kmazurek95/ThesisPipelineRework}
}
```

---

*For methodology details, see [METHODOLOGY.md](METHODOLOGY.md).*
