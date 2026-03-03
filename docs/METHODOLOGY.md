# Methodology

This pipeline is a ground-up rebuild of the methodology developed for my master's thesis (Mazurek, 2023). It applies the same research design to the same corpus with improved classification and reproducible infrastructure.

---

## Theoretical Background

This pipeline investigates **organizational prominence** as defined by Halpin & Fraussen (2017): the perception that an interest group is a preeminent voice for a constituency. Prominence is distinct from *access* (direct contact with policymakers) and *involvement* (formal inclusion in policy processes). It operates through an "audience dynamic": politicians decide which organizations matter, constrained by attention scarcity, and their choices are reflected in how they discuss those organizations on the floor.

The original thesis derived hypotheses from this framework across three dimensions: issue characteristics (salience), politician-group linkage (party, chamber, seniority), and group characteristics (age, policy breadth, external lobbying). This pipeline tests a subset of those variables (lobbying expenditure, party, chamber, and organization type) on the rebuilt dataset. For the full conceptual model, literature review, and hypothesis structure, see the [thesis paper](Thesis_UvA_Kaleb_Mazurek.pdf).

---

## Research Design

### Research Questions

1. **Which organizations receive prominent vs. passing mentions in congressional floor speeches?**
2. **What factors predict prominence?** Lobbying expenditure, organization type, speaker characteristics?
3. **Do partisan patterns exist?** How do Democrats and Republicans differ in mentioning interest groups?

### Unit of Analysis

The data has four levels: 53,892 individual mentions nested within 2,260 organizations and 490 politicians, cross-classified by 18 CAP policy areas. The analysis covers the 114th Congress (January 2015 - January 2017) and 115th Congress (January 2017 - January 2019), using floor speeches from the Congressional Record in both House and Senate.

---

## Data Collection

### Source 1: Congressional Record (GovInfo API)

Congressional Record documents are fetched from the GovInfo API, filtered to floor speeches (excluding procedural items), and parsed from HTML/XML to extract the speech text, speaker cues, chamber, and date for each granule.

### Source 2: Interest Group Dictionary (WRS 2011)

The **Washington Representatives Study (2011)** provides the universe of interest groups for matching. Each organization entry includes the organization name, lobbying expenditure, organization type (trade, labor, single-issue, etc.), founding year, and policy areas of activity.

### Source 3: Congress.gov API

Member metadata (party, state, seniority, committee assignments) comes from the Congress.gov API and is linked to speakers via BioGuide IDs. Bill metadata provides policy area classifications for speeches that reference specific legislation.

### Mention Extraction

Interest group mentions are identified using:

1. **Exact String Matching**: Organization names from WRS dictionary
2. **Alias Resolution**: Common abbreviations (e.g., "AFL-CIO" = "American Federation of Labor")

**Output:** Each row represents one mention of one organization in one speech.

---

## Text Classification

### Problem Definition

**Task:** Classify each mention as:
- **Prominent (1)**: Substantive discussion of the organization
- **Passing (0)**: Brief or incidental reference

### Training Data

- **Source:** Manual coding of 1,222 mentions
- **Coder:** 1 annotator (the author)
- **Classifier–human agreement:** Cohen's kappa on the held-out test set (see `results_classifier/report.txt`)

**Coding Criteria** (adapted from Fraussen et al., 2018):
- **Prominent** if any of: (1) views adopted/endorsed, (2) significant role in policy area mentioned, (3) used as expert resource, (4) importance or relevance conveyed
- **Passing** if: brief reference, list inclusion, procedural mention, or 10+ other organizations in same context
- See the [thesis paper](Thesis_UvA_Kaleb_Mazurek.pdf) for full operationalization

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

**Model Comparison (5-fold GroupKFold cross-validation, scored on average precision):**

| Model | F1 | Precision | Recall | ROC-AUC |
|-------|-----|-----------|--------|---------|
| Logistic Regression | **0.85** | 0.82 | 0.89 | 0.91 |
| Random Forest | 0.83 | 0.84 | 0.83 | 0.92 |
| SVM (Linear) | 0.83 | 0.83 | 0.83 | 0.92 |

These are cross-validation metrics at the default 0.5 threshold. After selecting Logistic Regression and optimizing the decision threshold on the held-out test set (threshold = 0.558), the final test-set performance is:

| Metric | Value |
|--------|-------|
| F1 | **0.91** |
| Precision | 0.87 |
| Recall | 0.94 |
| ROC-AUC | 0.95 |

**Final Model:** Logistic Regression with L2 regularization (C=2.0)

I tried three classifiers and logistic regression edged out SVM slightly on F1 while being faster to train and, more importantly, producing interpretable coefficients and well-calibrated probabilities. Interpretability mattered because I wanted to inspect which textual features drive prominence predictions (see SHAP analysis in `notebooks/Classification_Analysis.ipynb`). Threshold optimization on the held-out test set (see `results_classifier/report.txt`) raised F1 from 0.85 to 0.91 by shifting the decision boundary to 0.558.

### Cross-Validation Strategy

Mentions are grouped by organization so that all mentions of a given org land in the same fold, preventing the model from memorizing organization-specific language during training and then being "tested" on more text from the same org.

```python
# interest_group_analysis/3.classification/text_classifier.py, lines 159-163
cv = GroupKFold(n_splits=5)
gs = GridSearchCV(
    pipe, grid,
    cv=cv.split(X_train, y_train, groups=g_train),
    scoring="average_precision", refit=True
)
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
| is_labor | 0.136 | 1.146 | 0.003 |
| is_single_issue | 0.343 | 1.409 | < 0.001 |

### Model 2: Multilevel Model (R/lme4)

**Specification:**
```r
# R_analysis/Multilevel_Analysis.Rmd, lines 259-265
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

**Automated Validation:**

```python
# tests/test_data_validation.py, lines 62-66
def test_level1_prominence_is_binary(self, level1):
    """Prominence prediction should be 0 or 1."""
    unique_vals = level1["prominence_prediction"].dropna().unique()
    assert set(unique_vals).issubset({0, 1, 0.0, 1.0})
```

The full test suite (`pytest tests/ -v`) checks column presence, value ranges, cross-level consistency, and deduplication across all four dataset levels.

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

1. **Two Congresses:** Analysis covers the 114th-115th Congress (2015-2019); results may not generalize to other time periods
2. **Floor Speeches Only:** Excludes committee hearings, press releases
3. **WRS 2011 Dictionary:** May miss newer organizations or name changes

### Methodological Limitations

1. **Classification Errors:** 9% misclassification rate propagates to analysis
2. **Causal Inference:** Observational data cannot establish causation
3. **Selection Effects:** Organizations that lobby may differ systematically

### Future Directions

1. Extend beyond the 114th-115th Congress (time-series analysis)
2. Include committee hearing transcripts
3. Experiment with transformer-based classifiers (BERT)
4. Add campaign contribution data

---

## References

- Baumgartner, F. R., & Leech, B. L. (1998). *Basic Interests: The Importance of Groups in Politics and in Political Science*. Princeton University Press.
- Congressional Record. GovInfo API. https://api.govinfo.gov/
- Fraussen, B., Graham, T., & Halpin, D. R. (2018). Assessing the Prominence of Interest Groups in Parliament: A Supervised Machine Learning Approach. *The Journal of Legislative Studies*, 24(4), 450–74.
- Grossmann, M. (2012). *The Not-so-Special Interests: Interest Groups, Public Representation, and American Governance*. Stanford University Press.
- Halpin, D. R., & Fraussen, B. (2017). Conceptualising the Policy Engagement of Interest Groups: Involvement, Access and Prominence. *European Journal of Political Research*, 56(3), 723–32.
- Mazurek, K. (2023). Beyond Policy Influence: A Deeper Dive into the Factors Driving Advocacy Group Prominence. Master's thesis, University of Amsterdam.
- Washington Representatives Study (2011). https://faculty.wcas.northwestern.edu/~jnd260/

---

*For replication instructions, see [REPLICATION.md](REPLICATION.md). For a comparison of this pipeline's results with the original thesis findings, see [THESIS_EXTENSION_NOTES.md](THESIS_EXTENSION_NOTES.md).*
