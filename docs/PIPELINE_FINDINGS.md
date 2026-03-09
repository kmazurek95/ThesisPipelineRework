# Pipeline Rebuild Findings

Kaleb Mazurek | MSc Political Science | 114th-115th U.S. Congress (2015-2019)

---

## Research Question

What factors explain variation in organizational prominence in U.S. Congressional floor debates? This is the same question I investigated in my master's thesis (Mazurek, 2023), applied here to a rebuilt dataset. I wanted to know whether interest groups that spend more on lobbying get discussed more substantively, or just mentioned more often, and how speaker characteristics like party and chamber shape those patterns.

## Data and Method

The original thesis analyzed 20,699 mentions using an SVM classifier (F1=0.79). This pipeline rebuilds the analysis from scratch. It scans roughly 78,000 Congressional Record documents from the 114th and 115th Congresses (2015-2019) via the GovInfo API, extracting every mention of organizations listed in the Washington Representatives Study. This produced 53,892 mentions across 2,260 unique interest groups.

Each mention was classified as "prominent" or "passing" using a Logistic Regression classifier (F1=0.91) trained on 1,222 hand-labeled examples. Cross-validation splits by organization, so the test set only contains groups the model has never seen. I linked mentions to speaker metadata from Congress.gov and organizational characteristics from the WRS, then ran regressions at the mention, organization, and politician levels, plus multilevel models in R with crossed random effects. See [METHODOLOGY.md](METHODOLOGY.md) for full details.

## Key Findings

Lobbying expenditure: +7.4% higher odds of a prominent mention per log-unit increase (OR=1.074, p < 0.001). The question is not whether well-funded groups get mentioned more often (they do), but whether those mentions are substantive. The rebuilt data says yes. The original thesis hypothesized a null relationship here, but the data contradicted that in both the thesis and this pipeline.

Senate vs. House: Senators have 45% higher odds of giving a prominent mention (OR=1.448, p < 0.001), consistent with longer floor speeches and more deliberative norms in the Senate.

Party: Democrats give 23% fewer prominent mentions than Republicans (OR=0.772, p < 0.001), controlling for organization type and lobbying expenditure. The original thesis found no significant party effect.

Organization type: Single-issue groups receive 41% more prominent mentions than multi-issue groups (OR=1.409, p < 0.001). Labor unions show a smaller premium at +15% (OR=1.146, p = 0.003).

## Relationship to Original Thesis

The original thesis (Mazurek, 2023) covered 20,699 mentions across 1,903 organizations with full hypothesis testing across three theoretical dimensions (issue salience, politician-group linkage, group characteristics). This pipeline expands to 53,892 mentions and 2,260 organizations with an improved classifier (F1=0.91 vs. 0.79). For the original findings, see [THESIS_FINDINGS_2023.md](THESIS_FINDINGS_2023.md) or the [thesis PDF](Thesis_UvA_Kaleb_Mazurek.pdf). For where results align and diverge, see [THESIS_EXTENSION_NOTES.md](THESIS_EXTENSION_NOTES.md).

## Technical Implementation

The pipeline runs in five stages: data collection, processing, classification, integration, and analysis. Speaker attribution uses character-offset span detection to map mentions to legislators within Congressional Record granules. The classifier uses TF-IDF features (unigrams and bigrams on paragraph context) with threshold optimization, and the train/test split is by organization to prevent leakage. An interactive Streamlit dashboard provides organization-level case studies and regression visualizations.

## Limitations and Extensions

The biggest limitation is speaker attribution: 58.7% of mentions have no reliable speaker data, so the regressions use only 41% of the dataset (N=22,248). The training data is small (1,222 examples, 192 in the test set) and drawn entirely from the 114th Congress. The interest group dictionary and lobbying figures are from 2011, creating a temporal mismatch with the 2015-2019 speech data. For the full list, see [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).
