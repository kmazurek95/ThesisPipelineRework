# Methodology

This document provides detailed methodology for the Interest Group Prominence in Congressional Speech project.

---

## Table of Contents

1. [Research Design](#research-design)
2. [Data Collection](#data-collection)
3. [Text Classification](#text-classification)
4. [Statistical Analysis](#statistical-analysis)
5. [Validation Strategy](#validation-strategy)

---

## Research Design

### Research Questions

1. **Which organizations receive prominent vs. passing mentions in congressional floor speeches?**
2. **What factors predict prominence?** Lobbying expenditure, organization type, speaker characteristics?
3. **Do partisan patterns exist?** How do Democrats and Republicans differ in mentioning interest groups?

### Unit of Analysis

The project uses a **multi-level data structure**:

| Level | Unit | N | Description |
|-------|------|---|-------------|
| 1 | Mention | 25,106 | Individual interest group mentions |
| 2 | Organization | 1,679 | Unique interest groups |
| 3 | Politician | 490 | Members of Congress |
| 4 | Policy Area | 18 | CAP policy categories |

### Time Period

- **Congress**: 114th (January 2015 - January 2017)
- **Chamber Coverage**: House of Representatives and Senate
- **Document Type**: Floor speeches from the Congressional Record

---

## Data Collection

### Source 1: Congressional Record (GovInfo API)

**Process:**
1. Query GovInfo API for Congressional Record documents
2. Filter to floor speeches (exclude procedural items)
3. Parse HTML/XML to extract speech text
4. Identify speaker attribution

**Key Fields:**
- `granule_id`: Unique document identifier
- `date`: Publication date
- `chamber`: H (House) or S (Senate)
- `text`: Full speech content

### Source 2: Interest Group Dictionary (WRS 2011)

The **Washington Representatives Study (2011)** provides the universe of interest groups for matching.

**Key Fields:**
- `ENTRY`: Organization name
- `LOBBYING11`: Total lobbying expenditure
- `CATEGORY`: Organization type (trade, labor, single-issue, etc.)
- `FOUNDED`: Year established
- `ISSUES`: Policy areas of activity

### Source 3: Congress.gov API

**Member Data:**
- `bioGuideId`: Unique legislator identifier
- `party`: Political party affiliation
- `state`: State represented
- Seniority, committee assignments

**Bill Data:**
- Bill numbers referenced in speeches
- Policy area classifications

### Mention Extraction

Interest group mentions are identified using:

1. **Exact String Matching**: Organization names from WRS dictionary
2. **Fuzzy Matching**: Levenshtein distance < 3 for minor variations
3. **Alias Resolution**: Common abbreviations (e.g., "AFL-CIO" = "American Federation of Labor")

**Output:** Each row represents one mention of one organization in one speech.

---

## Text Classification

### Problem Definition

**Task:** Classify each mention as:
- **Prominent (1)**: Substantive discussion of the organization
- **Passing (0)**: Brief or incidental reference

### Training Data

- **Source:** Manual coding of 907 mentions
- **Coders:** 2 trained annotators
- **Inter-rater reliability:** Cohen's kappa = 0.84

**Coding Criteria:**
- **Prominent:** Organization is the subject of discussion; actions, positions, or impact described
- **Passing:** Organization mentioned in lists, procedural references, or tangential context

### Feature Engineering

**Text Features:**
- TF-IDF vectorization on paragraph context (±200 characters)
- Unigrams and bigrams
- Maximum 5,000 features

**Preprocessing:**
1. Lowercase conversion
2. Remove punctuation (preserve hyphens)
3. Remove stopwords (NLTK English)
4. No stemming (preserves organization names)

### Model Selection

**Evaluated Models:**
| Model | F1 Score | Precision | Recall |
|-------|----------|-----------|--------|
| Logistic Regression | **0.91** | 0.90 | 0.92 |
| SVM (Linear) | 0.90 | 0.89 | 0.91 |
| Random Forest | 0.85 | 0.86 | 0.84 |
| XGBoost | 0.87 | 0.88 | 0.86 |
| Naive Bayes | 0.82 | 0.80 | 0.84 |

**Final Model:** Logistic Regression with L2 regularization (C=1.0)

**Rationale:**
- Highest F1 score
- Interpretable coefficients
- Fast training and inference
- Well-calibrated probabilities

### Cross-Validation Strategy

**Group-Aware K-Fold (K=5):**
- Mentions grouped by organization
- All mentions of an organization in same fold
- Prevents data leakage from organization-specific patterns

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=org_ids):
    # Train and evaluate
```

### Model Interpretation

**Top Predictive Features (TF-IDF Coefficients):**

| Feature | Coefficient | Direction |
|---------|-------------|-----------|
| "testified" | +2.34 | Prominent |
| "opposes" | +1.89 | Prominent |
| "advocates" | +1.76 | Prominent |
| "including" | -1.45 | Passing |
| "such as" | -1.32 | Passing |

**SHAP Analysis:** See `notebooks/Classification_Analysis.ipynb` for detailed feature attribution.

---

## Statistical Analysis

### Model 1: Mention-Level Logistic Regression

**Specification:**
```
logit(P(Prominent)) = β₀ + β₁·log(Lobbying) + β₂·Democrat + β₃·Senate
                    + β₄·Labor + β₅·SingleIssue + ε
```

**Key Findings:**
| Variable | Coefficient | Odds Ratio | p-value |
|----------|-------------|------------|---------|
| log_lobbying | 0.071 | 1.074 | < 0.001 |
| is_democrat | -0.259 | 0.772 | < 0.001 |
| is_senate | 0.370 | 1.448 | < 0.001 |
| is_labor | 0.136 | 1.146 | 0.008 |
| is_single_issue | 0.343 | 1.409 | < 0.001 |

### Model 2: Multilevel Model (R/lme4)

**Specification:**
```r
glmer(prominence ~ log_lobbying + is_single_issue + is_labor +
                   party + chamber +
                   (1 | org_id) + (1 | issue_area),
      data = model_data, family = binomial)
```

**Random Effects:**
- Organization-level intercepts capture systematic differences in how organizations are discussed
- Policy area intercepts capture domain-specific prominence patterns

**Variance Decomposition (ICC):**
- Organization: 12.3% of variance
- Policy Area: 4.7% of variance
- Residual: 83.0% of variance

### Robustness Checks

1. **Alternative classification threshold:** Results stable at 0.4, 0.5, 0.6
2. **Exclude top 10 most-mentioned organizations:** Coefficients unchanged
3. **Separate models by chamber:** Direction consistent, magnitudes vary
4. **Time fixed effects:** No significant temporal trends

---

## Validation Strategy

### Data Quality Checks

**Automated Validation (run at each pipeline stage):**

```python
# Example validation rules
assert df['prominence_prediction'].isin([0, 1]).all()
assert df['party'].isin(['D', 'R', 'I']).all()
assert df['lobbying'].min() >= 0
assert df.duplicated(subset=['mention_id']).sum() == 0
```

### Classifier Validation

1. **Holdout Test Set:** 20% of labeled data never seen during training
2. **Stratified Sampling:** Maintains class balance in train/test
3. **Learning Curves:** Verify convergence, no overfitting
4. **Calibration Plot:** Predicted probabilities match observed frequencies

### Reproducibility

**Ensured by:**
- Fixed random seeds (`random_state=42`)
- Version-pinned dependencies (`requirements.txt`)
- Documented preprocessing steps
- Saved model artifacts (`results_classifier/`)

---

## Limitations

### Data Limitations

1. **Single Congress:** Results may not generalize to other time periods
2. **Floor Speeches Only:** Excludes committee hearings, press releases
3. **WRS 2011 Dictionary:** May miss newer organizations or name changes

### Methodological Limitations

1. **Classification Errors:** 9% misclassification rate propagates to analysis
2. **Causal Inference:** Observational data cannot establish causation
3. **Selection Effects:** Organizations that lobby may differ systematically

### Future Directions

1. Extend to multiple Congresses (time-series analysis)
2. Include committee hearing transcripts
3. Experiment with transformer-based classifiers (BERT)
4. Add campaign contribution data

---

## References

- Baumgartner, F. R., & Leech, B. L. (1998). *Basic Interests: The Importance of Groups in Politics and in Political Science*. Princeton University Press.
- Congressional Record. GovInfo API. https://api.govinfo.gov/
- Washington Representatives Study (2011). https://faculty.wcas.northwestern.edu/~jnd260/

---

*For replication instructions, see [REPLICATION.md](REPLICATION.md).*
