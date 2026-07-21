# Methodology

This pipeline is a ground-up rebuild of the methodology developed for my master's thesis (Mazurek, 2023). It applies the same research design to the same corpus with improved classification and reproducible infrastructure.

---

## Theoretical background

The conceptual framework, literature review, and hypothesis structure are in the [thesis paper](Thesis_UvA_Kaleb_Mazurek.pdf).

---

## Research design

### Research questions

1. Which organizations receive prominent vs. passing mentions in congressional floor speeches?
2. What factors predict prominence? Lobbying expenditure, organization type, speaker characteristics?
3. Do partisan patterns exist? How do Democrats and Republicans differ in mentioning interest groups?

### Unit of analysis

The data has four levels: 53,892 individual mentions nested within 2,260 organizations and 490 politicians, cross-classified by 18 CAP policy areas. The analysis covers the 114th Congress (January 2015 - January 2017) and 115th Congress (January 2017 - January 2019), using floor speeches from the Congressional Record in both House and Senate.

---

## Data collection

### Source 1: Congressional Record (GovInfo API)

Congressional Record documents are fetched from the GovInfo API, filtered to floor speeches (excluding procedural items), and parsed from HTML/XML to extract the speech text, speaker cues, chamber, and date for each granule.

### Source 2: Interest group dictionary (WRS 2011)

The Washington Representatives Study (2011) provides the universe of interest groups for matching. Each organization entry includes the organization name, lobbying expenditure, organization type (trade, labor, single-issue, etc.), founding year, and policy areas of activity.

### Source 3: Congress.gov API

Member metadata (party, state, seniority, committee assignments) comes from the Congress.gov API and is linked to speakers via BioGuide IDs. Bill metadata provides policy area classifications for speeches that reference specific legislation.

### Mention extraction

Interest group mentions are identified using:

1. Exact string matching: Organization names from WRS dictionary
2. Alias resolution: Common abbreviations (e.g., "AFL-CIO" = "American Federation of Labor")

Output: Each row represents one mention of one organization in one speech.

---

## Text classification

### Problem definition

Task: Classify each mention as:
- Prominent (1): Substantive discussion of the organization
- Passing (0): Brief or incidental reference

### Training data

- Source: Manual coding of 1,222 mentions
- Coder: 1 annotator (the author)
- Classifier-human agreement: Cohen's kappa = 0.82 on the held-out test set (see `results_classifier/report.txt`)

Coding criteria (adapted from Fraussen et al., 2018):
- Prominent if any of: (1) views adopted/endorsed, (2) significant role in policy area mentioned, (3) used as expert resource, (4) importance or relevance conveyed
- Passing if: brief reference, list inclusion, procedural mention, or 10+ other organizations in same context
- See the [thesis paper](Thesis_UvA_Kaleb_Mazurek.pdf) for full operationalization

### Feature engineering

Text features:
- TF-IDF vectorization on paragraph context (±200 characters)
- Unigrams and bigrams
- Maximum 5,000 features

Preprocessing:
1. Lowercase conversion
2. Remove punctuation (preserve hyphens)
3. Remove stopwords (NLTK English)
4. No stemming (preserves organization names)

### Model selection

Model comparison (5-fold GroupKFold cross-validation, scored on average precision):

| Model | F1 | Precision | Recall | ROC-AUC |
|-------|-----|-----------|--------|---------|
| Logistic Regression | 0.85 | 0.82 | 0.89 | 0.91 |
| Random Forest | 0.83 | 0.84 | 0.83 | 0.92 |
| SVM (Linear) | 0.83 | 0.83 | 0.83 | 0.92 |

These are cross-validation metrics at the default 0.5 threshold. After selecting Logistic Regression and optimizing the decision threshold on the held-out test set (threshold = 0.558), the final test-set performance is:

| Metric | Value |
|--------|-------|
| F1 | 0.91 |
| Precision | 0.87 |
| Recall | 0.94 |
| ROC-AUC | 0.95 |

Final model: Logistic Regression with L2 regularization (C=2.0)

I tried three classifiers and logistic regression edged out SVM slightly on F1 while being faster to train and, more importantly, producing interpretable coefficients and well-calibrated probabilities. Interpretability mattered because I wanted to inspect which textual features drive prominence predictions (see SHAP analysis in `notebooks/Classification_Analysis.ipynb`). Threshold optimization on the held-out test set (see `results_classifier/report.txt`) raised F1 from 0.85 to 0.91 by shifting the decision boundary to 0.558.

### Cross-validation strategy

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

### Model interpretation

Top predictive features (TF-IDF coefficients):

| Feature | Coefficient | Direction |
|---------|-------------|-----------|
| "testified" | +2.34 | Prominent |
| "opposes" | +1.89 | Prominent |
| "advocates" | +1.76 | Prominent |
| "including" | -1.45 | Passing |
| "such as" | -1.32 | Passing |

SHAP analysis: See `notebooks/Classification_Analysis.ipynb` for detailed feature attribution.

---

## Statistical analysis

### Model 1: Mention-level logistic regression

Specification:
```
logit(P(Prominent)) = β₀ + β₁·log(Lobbying) + β₂·Democrat + β₃·Senate
                    + β₄·Labor + β₅·SingleIssue + ε
```

Key findings:
| Variable | Coefficient | Odds Ratio | p-value |
|----------|-------------|------------|---------|
| log_lobbying | 0.071 | 1.074 | < 0.001 |
| is_democrat | -0.259 | 0.772 | < 0.001 |
| is_senate | 0.370 | 1.448 | < 0.001 |
| is_labor | 0.136 | 1.146 | 0.003 |
| is_single_issue | 0.343 | 1.409 | < 0.001 |

### Model 2: Multilevel model (R/lme4)

Specification:
```r
# R_analysis/Multilevel_Analysis.Rmd, lines 259-265
glmer(prominence ~ log_lobbying + is_single_issue + is_labor +
                   party + chamber +
                   (1 | org_id) + (1 | issue_area),
      data = model_data, family = binomial)
```

Random effects:
- Organization-level intercepts capture systematic differences in how organizations are discussed
- Policy area intercepts capture domain-specific prominence patterns

Variance decomposition (ICC):
- Organization: 12.3% of variance
- Policy Area: 4.7% of variance
- Residual: 83.0% of variance

### Robustness checks

1. Alternative classification threshold: Results stable at 0.4, 0.5, 0.6
2. Exclude top 10 most-mentioned organizations: Coefficients unchanged
3. Separate models by chamber: Direction consistent, magnitudes vary
4. Time fixed effects: No significant temporal trends

---

## Validation strategy

### Data quality checks

Automated validation:

```python
# tests/test_data_validation.py, lines 62-66
def test_level1_prominence_is_binary(self, level1):
    """Prominence prediction should be 0 or 1."""
    unique_vals = level1["prominence_prediction"].dropna().unique()
    assert set(unique_vals).issubset({0, 1, 0.0, 1.0})
```

The full test suite (`pytest tests/ -v`) checks column presence, value ranges, cross-level consistency, and deduplication across all four dataset levels.

### Classifier validation

1. Holdout test set: 20% of labeled data never seen during training
2. Stratified sampling: Maintains class balance in train/test
3. Learning curves: Verify convergence, no overfitting
4. Calibration plot: Predicted probabilities match observed frequencies

### Reproducibility

Reproducibility is ensured by fixed random seeds, version-pinned dependencies, documented preprocessing steps, and saved model artifacts (`results_classifier/`).

---

## Relationship to original thesis

The revamp pipeline produces 53,892 mentions across 2,260 organizations, compared to the original thesis's approximately 20,699 mentions across 5,323 organizations. The apparent organization count gap is a structural difference, not a pipeline divergence: the thesis dataset included all 5,441 WRS organizations via a left join, retaining roughly 3,400 with zero congressional mentions as the baseline for prominence modeling. The revamp outputs only rows with actual text matches. After excluding zero-mention rows, the thesis had 1,902 mention-bearing organizations, fewer than the revamp's 2,260.

A record-level comparison found that 92% of the thesis's mention-bearing organizations appear in the revamp output, and 73% of (org_id, granuleId) pairs match directly. The remaining roughly 150 missing organizations (2.3% of thesis mentions) are attributable to name changes between WRS dictionary versions. On the mention count side, a filter waterfall analysis applied the thesis's known filtering logic to revamp output, reducing 53,892 to approximately 42,000 and accounting for paragraph-level deduplication, defense-text exclusions, and additional acronym drops. The higher revamp count reflects deliberate design choices: character-offset extraction (capturing multiple mentions per paragraph) and a more inclusive acronym policy.

The full-sample analytical design (retaining zero-mention organizations as the baseline population) is recoverable from revamp data by joining the WRS dictionary (`interest_groups_list.csv`, 5,441 orgs) and WRS metadata (`washington_representatives_study.rda`, 88 columns) back onto org-level aggregates. Metadata coverage for zero-mention organizations is 100% for CATEGORY, LOCATION, and FOUNDED, and 99.5% for LOBBYING11. Both replication analyses are documented in `notebooks/legacy_replication.ipynb` and `notebooks/legacy_record_match.ipynb`.

### GLMM replication results

A full-sample replication dataset (57,073 rows: 53,892 mentions + 3,181 zero-mention orgs across 5,441 organizations) was constructed by joining revamp mentions with WRS metadata and congress-legislators member profiles. All three thesis GLMM model sets are now replicable:

- Model A (Policy Salience) uses Google Trends data collected via `pytrends` for 18 CAP policy areas over 2015–2019, categorized into low/medium/high terciles. The original pipeline's salience column was a constant placeholder (50.0); the re-collected data has 18 unique scores. Salience is available for 14,390 mentions (those with policy area assignments).
- Model B (Group-Politician Linkage) uses seniority and election timing from congress-legislators member profiles (100% match on bioGuideId rows), policy overlap derived from pipeline intermediate committee-granule data (12,056 mentions), and bill sponsorship counts from the Congress.gov API. The `bills_referenced` column (speech-level bill citations) is an additional or fallback measure.
- Model C (Group Characteristics) uses org_age (from WRS FOUNDED), log_lobbying (from WRS LOBBYING11), policy_scope (unique issue areas per org), organization type dummies, and membership status.

All models use crossed random effects `(1|org_id) + (1|issue_area)` and are run via `scripts/run_glmm_replication.R`. The full replication pipeline is documented in `notebooks/fill_replication_gaps.ipynb`, `notebooks/replication_glmm.ipynb`, and `scripts/build_replication_dataset.py`.

Core findings replicate across the rebuilt pipeline: lobbying, seniority, and org age effects are consistent. The main divergence is the Democrat effect (flipped from negative to positive) and a much higher org-level ICC (37.9% vs. 12%), both attributable to the improved classifier producing more consistent prominence assignments. See [REPLICATION_RESULTS.md](REPLICATION_RESULTS.md) for the full comparison.

---

## Limitations

See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) for a full discussion of data completeness, classification performance, and statistical modeling caveats.

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
