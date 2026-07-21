# R analysis: multilevel models

This directory contains R scripts for multilevel/mixed-effects regression analysis of interest group prominence.

## Why R?

R's `lme4` package is the standard tool for fitting mixed-effects models (GLMER with crossed random effects), which Python's `statsmodels` does not support.

## Files

| File | Purpose |
|------|---------|
| `Multilevel_Analysis.Rmd` | Main analysis with all models |
| `utils.R` | Helper functions |
| `run_analysis.R` | Script to knit the report |
| `README.md` | This file |

## Requirements

Install required packages:

```r
install.packages(c(
  "tidyverse",   # Data manipulation
  "lme4",        # Mixed-effects models
  "broom.mixed", # Tidy model outputs
  "sjPlot",      # Visualization & tables
  "performance", # Model diagnostics
  "ggeffects",   # Marginal effects
  "knitr",       # Report generation
  "kableExtra"   # Table formatting
))
```

## Running the analysis

### Option 1: In RStudio

1. Open `Multilevel_Analysis.Rmd` in RStudio
2. Click "Knit" → "Knit to HTML"

### Option 2: Command line

```bash
Rscript run_analysis.R
```

### Option 3: From R console

```r
rmarkdown::render("Multilevel_Analysis.Rmd", output_dir = "../outputs/reports")
```

## Model specifications

### Model structure

All models use crossed random effects:

```
prominence ~ fixed_effects + (1 | org_id) + (1 | issue_area)
```

### Models

| Model | Fixed Effects | Purpose |
|-------|---------------|---------|
| M0 | (intercept only) | Null model for ICC |
| M1 | + lobbying, org_type | Organizational effects |
| M2 | + party, chamber | Full model with politician effects |

## Expected outputs

After running the analysis:

```
outputs/
├── figures/
│   ├── R_model2_forest_plot.png
│   ├── R_coefficient_comparison.png
│   ├── R_lobbying_effect.png
│   └── R_party_chamber_effect.png
├── tables/
│   ├── R_multilevel_summary.html
│   └── R_model2_coefficients.csv
├── models/
│   ├── R_model_null.rds
│   ├── R_model_org.rds
│   └── R_model_full.rds
└── reports/
    └── Multilevel_Analysis.html
```

## Key results

From the full model (M2):

| Variable | Odds Ratio | p-value | Interpretation |
|----------|------------|---------|----------------|
| Log Lobbying | 1.07 | <0.001 | +7% per log-unit |
| Single-Issue | 1.41 | <0.001 | +41% vs. other orgs |
| Labor Union | 1.15 | <0.01 | +15% vs. other orgs |
| Democrat | 0.77 | <0.001 | -23% vs. Republicans |
| Senate | 1.45 | <0.001 | +45% vs. House |

## Comparison with Python

The Python analysis (`regression_analysis.py`) uses:
- `statsmodels` for basic OLS/logit
- No random effects (treats data as independent)

The R analysis adds:
- Proper multilevel structure with `lme4::glmer()`
- Random intercepts for organizations and issue areas
- Accounts for clustering/non-independence

Both approaches yield similar fixed-effect estimates, validating the findings.
