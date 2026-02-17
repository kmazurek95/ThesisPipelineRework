# Known Limitations & Future Work

Every research pipeline involves tradeoffs — these are mine, and what I'd change in the next iteration.

---

## 1. Speaker Attribution

This is the single largest data completeness issue in the pipeline.

### Current methodology

Speaker attribution uses character-offset span detection. The system scans each Congressional Record granule for speaker cues (patterns like "Mr. SMITH.", "The PRESIDING OFFICER:", "Madam Speaker:"), builds spans from one speaker cue to the next, then maps each mention's character offset into the appropriate span. Speakers are canonicalized by looking up last names against a member metadata index and disambiguating by chamber when multiple members share a surname.

### Coverage and confidence gaps

**58.7% of mentions have no speaker attribution.** The `bioGuideId` field is missing for the majority of the dataset. It means the mention-level regression (which requires party and chamber data) uses only 22,248 of 53,892 mentions — about 41% of the data. Every finding about party or chamber effects comes from this subset, and if the unattributed mentions differ systematically from attributed ones — which they likely do, since speaker cue formatting varies by document type — the regression results may not generalize to the full dataset.

The speaker attribution gaps come from several sources:

- **Inserted documents**: Letters, reports, and external documents read into the record get attributed to the reading member, not the author. These often contain organization mentions but the "speaker" is really just the person who entered the document.
- **Role designations**: "The PRESIDING OFFICER" and similar institutional roles can't be resolved to a specific member without additional metadata.
- **Ambiguous names**: When multiple members share a last name and chamber disambiguation isn't sufficient, the system returns no canonical match.
- **Missing speaker cues**: Some granules (especially extensions of remarks and daily digest sections) lack the standard "Mr./Ms. LASTNAME." formatting.

### Potential fixes

- Cross-reference granule-level member metadata more aggressively — if only one member appears in the granule's metadata with `role="SPEAKING"`, assign that member even without a text-level speaker cue
- For inserted documents, flag them separately rather than attributing mentions to the reading member
- Use the Congress.gov API member data to resolve ambiguous names by checking committee assignments or state delegation
- Accept that some mentions will never have reliable speaker attribution and report results both with and without the unattributed subset

---

## 2. Training Data

### Current scope and balance

The classifier was trained on 1,222 hand-labeled examples (2 annotators, Cohen's kappa = 0.84). Class split: 448 prominent (36.7%), 774 passing (63.3%), which roughly mirrors the dataset-wide prominence rate of 35.5%.

The training data covers 593 unique organizations, drawn from three labeling batches: 823 from an earlier combined dataset, 365 from a separate labeling round, and 34 from a final dataset. All labels come from the original thesis work on the 114th Congress.

### Limitations

**Small sample relative to the prediction task.** 1,222 labels are being used to classify 53,892 mentions. The held-out test set is only 192 examples, which means the per-class F1 scores (0.91 prominent, 0.92 passing) have meaningful confidence intervals — a handful of misclassifications shift the metrics noticeably.

**Temporal concentration.** All training labels come from the 114th Congress (2015-2017). The classifier is then applied to both the 114th and 115th Congress (2017-2019). If the language used to discuss organizations shifted between Congresses — different policy priorities, different rhetorical styles — the classifier may perform differently on 115th Congress text than the training metrics suggest.

**No false-positive validation in training.** The classifier only sees mentions that the extraction step already identified. If the extraction step produces a false positive (e.g., "Brady" matching a person name), the classifier has no mechanism to flag it as "not actually an organization mention." The labeled data doesn't include a validity label — it only distinguishes prominent from passing among assumed-valid mentions.

**The labeling app exists but new labels aren't integrated.** A Streamlit-based labeling interface was built with 3,716 mentions loaded for annotation. This app supports richer labels (validity, false positive type, speaker validation) but its outputs haven't been fed back into the classifier training pipeline yet.

### Next steps

- Complete the labeling app workflow: label at least 500 additional mentions from the 115th Congress and retrain
- Add a validity classification step upstream of prominence classification — a two-stage pipeline where stage 1 filters false positives and stage 2 classifies prominence
- Use active learning to prioritize labeling mentions where the classifier is least confident
- Report confidence intervals on test metrics given the small test set size

---

## 3. Classification Performance

### Current metrics

| Metric | Class 0 (Passing) | Class 1 (Prominent) | Overall |
|--------|-------------------|---------------------|---------|
| Precision | 0.95 | 0.87 | 0.91 |
| Recall | 0.89 | 0.94 | 0.91 |
| F1 | 0.92 | 0.91 | 0.91 |

Test set: 192 examples. ROC-AUC: 0.95. Optimized decision threshold: 0.558.

The model uses TF-IDF (unigrams + bigrams) on paragraph context plus two numerical features (paragraph mention count, 10+ orgs mentioned flag), fed into a Logistic Regression with L2 regularization (C=2.0). Cross-validation is group-aware (GroupKFold by org_id, 5 folds) to prevent data leakage.

### Error patterns

**Precision gap on prominent class (0.87 vs. 0.95).** The classifier is more likely to incorrectly call something "prominent" than to incorrectly call something "passing." This matters because false positives on the prominent side inflate the prominence rate, which could bias the downstream regression coefficients upward.

**CV-to-test gap.** The best cross-validation average precision was 0.76, but the test F1 was 0.91. This gap is partly explained by threshold optimization on the test set (threshold = 0.558 instead of default 0.5). While the train/test split is proper (group-aware holdout), the threshold was tuned on the test data, which could be considered a mild form of overfitting to the evaluation set.

**Context dependency.** The classifier relies on paragraph context (roughly 7 sentences). Very short granules or mentions near the beginning/end of a document may have truncated context, which could degrade performance in ways not captured by the test set.

### Future directions

- Hold out a separate validation set for threshold tuning (train/validation/test three-way split)
- Report bootstrap confidence intervals on F1 given the small test set
- Experiment with transformer-based models (DistilBERT fine-tuning) — the TF-IDF approach works well but likely has a ceiling
- Analyze errors by organization type and policy area to identify systematic blind spots

---

## 4. Statistical Analysis

### Model specifications

The analysis runs three regression models plus R multilevel models:

1. **Mention-level logistic regression** (N=22,248): prominence ~ log(lobbying) + party + chamber + org type
2. **Organization-level OLS** (N=1,245): avg prominence ~ log(lobbying) + log(mentions) + org type
3. **Politician-level OLS** (N=390): avg prominence ~ party + chamber + log(mentions)
4. **R multilevel GLMER** (lme4): crossed random effects for organizations and policy areas

### Data completeness

**The mention-level model uses only 41% of the data.** Because it requires speaker attribution (party, chamber), 58.7% of mentions are excluded. If the unattributed mentions differ systematically from attributed ones — and they likely do, since speaker cue formatting varies by document type — the regression results may not generalize to the full dataset.

**73.3% of mentions lack issue area coding.** Policy area assignment depends on bill references in the Congressional Record text. Mentions in speeches that don't reference specific bills get no policy area, which severely limits the multilevel model's cross-classification by issue area.

**Lobbying data temporal mismatch.** The WRS 2011 lobbying expenditure data predates the speech data (2015-2018) by 4-7 years. Lobbying spending is moderately stable for large organizations but can shift for smaller or newer groups. This introduces measurement error in the key independent variable.

### Model fit

The mention-level pseudo-R-squared is 0.018. This is normal for a binary logit with many observations — most of the variation in whether a specific mention is prominent is driven by idiosyncratic factors (the specific speech, the specific context) rather than the organizational or speaker characteristics in the model. The ICC decomposition (organizations: 12.3%, policy areas: 4.7%, residual: 83%) confirms this.

The low R-squared doesn't mean the findings are wrong — the coefficients are precisely estimated and highly significant. It means the model explains *which factors shift the odds* of prominence, not *which specific mentions will be prominent.* This is the right question for the research design.

### Recommended improvements

- Impute or model speaker attribution for the 58.7% of unattributed mentions, or at least run sensitivity analyses comparing attributed vs. full-sample results
- Supplement policy area coding with a text-based topic model (LDA or similar) to assign policy areas to mentions without bill references
- Use more recent lobbying data (OpenSecrets has annual data through 2023) to reduce temporal mismatch
- Add Congress fixed effects and interaction terms (e.g., lobbying x party) to the models
- Report the full multilevel model results from R alongside the Python regressions for comparison
- The original thesis tested additional variables — issue salience, organization age, seniority, and legislative activity — that are available in the pipeline's data but not yet included in the current regressions. The data architecture supports extending the regression specifications to match the original thesis models as future work (see [THESIS_EXTENSION_NOTES.md](THESIS_EXTENSION_NOTES.md) for details)

---

## 5. Mention Extraction

The extraction pipeline uses compiled regex patterns with word-boundary matching for full names and flexible separator patterns for acronyms (so "A.F.L.-C.I.O." and "AFL-CIO" both match). There's a stop-word blocklist, a minimum-token requirement, and deduplication by character span. Match type breakdown: 68% full name, 32% acronym.

### Known false positive patterns

Short acronyms (2-3 letters) are the biggest risk — "ACT" (American College Testing) matches the word "act" in legislative text, and the Congressional Record's all-caps formatting makes case-based disambiguation unreliable. Person name collisions are another problem: "Brady" matches both "Brady: United Against Gun Violence" and "Rep. Brady (PA)." The WRS 2011 dictionary also misses organizations founded after 2011 or that changed names, and partial name overlap between organizations (e.g., "National Bar Association" vs. "American Bar Association") can cause misattribution.

### What I'd change

- Add a context-aware disambiguation step: when an acronym or short name matches, check surrounding text for semantic cues (e.g., "Brady" near "gun" vs. near committee membership language)
- Build a false-positive blocklist for known problem org names — short acronyms that collide with common English words, surnames that are also organization names
- Update the interest group dictionary to a more recent edition of the WRS or supplement with OpenSecrets/lobbying disclosure data
