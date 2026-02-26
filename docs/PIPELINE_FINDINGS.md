# Pipeline Rebuild Findings

**Kaleb Mazurek** | MSc Political Science | 114th-115th U.S. Congress (2015-2018)

---

## Research Question

What factors explain variation in organizational prominence in U.S. Congressional floor debates? This is the same question I investigated in my master's thesis (Mazurek, 2023), now applied to a larger dataset. I wanted to know whether interest groups that spend more on lobbying actually get discussed more substantively, or just mentioned more often, and how speaker characteristics like party and chamber shape those patterns.

## Data and Method

The original thesis (Mazurek, 2023) analyzed 20,699 mentions from the 114th Congress using an SVM classifier (F1=0.79). This pipeline rebuilds the analysis from scratch with a larger corpus and improved methods. It scans roughly 78,000 Congressional Record documents from the 114th and 115th Congresses (2015-2018) via the GovInfo API, extracting every mention of organizations listed in the Washington Representatives Study. This produced 53,892 individual mentions across 2,260 unique interest groups.

Each mention was classified as "prominent" or "passing" using a Logistic Regression classifier (F1=0.91) trained on 1,222 hand-labeled examples. Cross-validation splits by organization rather than randomly, so the test set only contains groups the model has never seen. I then linked mentions to speaker metadata from Congress.gov and organizational characteristics from the WRS, and ran regressions at the mention, organization, and politician levels, plus multilevel models in R with crossed random effects for organizations and policy areas. See [METHODOLOGY.md](METHODOLOGY.md) for full details.

## Key Findings

**Lobbying predicts prominence, not just visibility.** The question wasn't whether well-funded groups get mentioned more (they do, almost mechanically, because they're more active). The more interesting question is whether those mentions are *substantive.* The expanded dataset shows that each log-unit increase in lobbying expenditure is associated with 7.4% higher odds of a mention being prominent rather than passing (OR=1.074, p < 0.001). This matters because it suggests lobbying buys not just airtime but the quality of attention. The original thesis hypothesized a null relationship here; the data contradicted that in both the thesis and the pipeline.

Senators have 45% higher odds of giving a prominent mention compared to House members (OR=1.448, p < 0.001), which tracks with the Senate's longer floor speeches and more deliberative norms.

Democrats give 23% fewer prominent mentions than Republicans (OR=0.772, p < 0.001). This holds after controlling for organization type and lobbying expenditure, and it's a new finding. The original thesis found no significant party effect.

Organizations focused on a single policy area receive 41% more prominent mentions than multi-issue groups (OR=1.409, p < 0.001). Labor unions show a smaller premium (+15%, OR=1.146, p = 0.003).

## Relationship to Original Thesis

**Original thesis (Mazurek, 2023):** 20,699 mentions, 1,903 orgs, SVM classifier (F1=0.79), full hypothesis testing across three theoretical dimensions (issue salience, politician-group linkage, group characteristics)

**This pipeline:** 53,892 mentions, 2,260 orgs, Logistic Regression (F1=0.91), tests lobbying expenditure, party, chamber, and organization type on the expanded dataset

For the complete theoretical argument and original findings, see the [thesis findings summary](THESIS_FINDINGS_2023.md) or the full [thesis paper](Thesis_UvA_Kaleb_Mazurek.pdf). For a detailed comparison of where results align and diverge, see [THESIS_EXTENSION_NOTES.md](THESIS_EXTENSION_NOTES.md).

## Technical Implementation

The pipeline is organized into five stages (data collection, processing, classification, integration, and analysis), each independently runnable and validated. Speaker attribution uses character-offset span detection to map mentions to specific legislators within Congressional Record granules. The classifier uses TF-IDF features (unigrams and bigrams on paragraph context) with threshold optimization, and the train/test split is done by organization (not random) to prevent leakage. Results are served through an interactive Streamlit dashboard with organization-level case studies and regression coefficient visualizations.

## Limitations and Extensions

The biggest limitation is speaker attribution coverage: 58.7% of mentions lack reliable speaker data, so the regressions use only 41% of the dataset. The training data comes entirely from the 114th Congress and is small (1,222 examples, 192-example test set). The interest group dictionary and lobbying data are from 2011, creating a temporal mismatch with the 2015-2018 speech data. For a full accounting, see [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md). A next iteration would prioritize extending the labeling dataset (particularly for the 115th Congress), experimenting with transformer-based classifiers, and supplementing with more recent lobbying data from OpenSecrets.

