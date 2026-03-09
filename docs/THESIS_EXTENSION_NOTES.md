# From Thesis to Pipeline: Extension Notes

**Kaleb Mazurek** | MSc Political Science | University of Amsterdam, 2023

## Context

My master's thesis, "Beyond Policy Influence: A Deeper Dive into the Factors Driving Advocacy Group Prominence," investigated what drives substantive mentions of interest groups in U.S. Congressional floor debates. The thesis analyzed 20,699 mentions from the 114th and 115th Congress (2015-2019), classified each as prominent or passing using an SVM text classifier, and ran multilevel regressions testing three groups of hypotheses derived from Halpin & Fraussen's (2017) theoretical framework (for the full argument, see the [thesis findings summary](THESIS_FINDINGS_2023.md) or the [thesis PDF](Thesis_UvA_Kaleb_Mazurek.pdf)).

This repository is a ground-up rebuild of that pipeline. The goal was not to re-run the thesis but to improve the methodology: build a better classifier, automate the extraction process, and make the whole pipeline reproducible. The research question is the same. The data, classifier, and infrastructure are different. The statistical findings reflect the rebuilt dataset and should be read on their own terms, not as a replication of the thesis results.

## What Changed

| Dimension | Original Thesis (2023) | Pipeline Rebuild |
|-----------|----------------------|-----------------|
| Corpus | 114th + 115th Congress | 114th + 115th Congress |
| Documents scanned | ~78,000 | ~78,000 |
| Mentions extracted | 20,699 | 53,892 |
| Unique organizations | 1,903 | 2,260 |
| Classifier | SVM (F1=0.79, ROC-AUC=0.72) | Logistic Regression (F1=0.91, ROC-AUC=0.95) |
| Training labels | 1,222 hand-labeled examples | Same 1,222 labels |
| Cross-validation | Standard k-fold | Group-aware k-fold (by org_id) |
| Speaker attribution | Regex-based speaker cue matching | Character-offset spans with confidence tiers (1.0/0.8/0.7/0.6/0.0) |
| Statistical models | 3 GLMER models testing H1, H2, H3 separately | Logistic regression + OLS + GLMER (combined variable set) |
| Infrastructure | Ad hoc scripts | Config-driven 5-stage pipeline with validation, tests, dashboard |

The training data is the same 1,222 hand-labeled examples. The classification improvement comes from the modeling choices, not more labels.

## Where Findings Align

Several patterns are consistent across both the thesis and the expanded pipeline:

**Chamber effect.** The thesis found that both House (OR=3.83) and Senate (OR=3.92) floor speeches produced more prominent mentions relative to the reference category, with Senate slightly higher. The pipeline finds Senate members have 45% higher odds of giving a prominent mention compared to House members (OR=1.448, p<0.001). The direction is the same: Senators give more prominent mentions, consistent with the Senate's longer floor speeches and more deliberative norms.

**Single-issue groups.** The pipeline finds that single-issue organizations receive 41% more prominent mentions than multi-issue groups (OR=1.409, p<0.001). The thesis tested policy breadth through a different operationalization but found a similar directional pattern.

**Labor unions.** The pipeline finds a modest premium for labor organizations (OR=1.146, p=0.003). This aligns with the thesis's finding that unions, as well-known constituency representatives, tend to receive more substantive discussion.

**Lobbying expenditure.** The thesis's third model tested external lobbying and found a positive association with prominence, contradicting the initial null hypothesis. The pipeline confirms this: each log-unit increase in lobbying expenditure is associated with 7.4% higher odds of a prominent mention (OR=1.074, p<0.001). Lobbying predicts not just visibility but the substantiveness of discussion.

## Where Results Diverge or Are Not Directly Comparable

**Party effects.** The pipeline finds that Democrats give 23% fewer prominent mentions than Republicans (OR=0.772, p<0.001). The thesis's Full Model 1 found a non-significant party effect (Democrat OR=1.06, p=0.33). This divergence could reflect the larger sample, differences in extraction methodology, or differences in model specification (the pipeline includes lobbying expenditure and organization type controls that the thesis's Model 1 did not).

**Issue salience.** The thesis found medium-salience policy areas had more prominent mentions (OR=1.49) but high-salience areas actually had fewer (OR=0.70), partially contradicting the hypothesis. The pipeline does not currently include salience in its regressions, though the variable exists in the data.

**Seniority.** The thesis found a significant negative effect of seniority on prominence (estimate=-0.017, p<0.001 in Models 2 and 3), contradicting the hypothesis that more senior members would give more prominent mentions. The pipeline does not currently test seniority, though the data to compute it is available.

**Model specifications differ.** The thesis tested three hypothesis groups in separate models; the pipeline combines variables into a single mention-level model plus separate organization-level and politician-level OLS models. This means direct coefficient comparisons should be made cautiously. Different controls in the model change the interpretation of individual coefficients.

## What the Pipeline Doesn't (Yet) Test

The thesis tested several variables that are available in the pipeline's data but not included in the current regressions:

- **Issue salience**: coded in the `salience` column, derived from policy area classifications
- **Organization age**: the `FOUNDED` column contains founding year for most organizations
- **Seniority**: computable from `startDate_114` in the member metadata
- **Legislative activity**: `bills_referenced` captures bill mentions; bills sponsored data would need computation
- **Issue overlap**: the thesis measured maximal issue overlap between politician committee assignments and organization policy areas; this would require additional data joining

The pipeline rebuild prioritized infrastructure, classification, and reproducibility over replicating every model specification from the thesis. The data architecture supports extending the regression models to include these variables. Running the full thesis model specifications on the expanded dataset is a natural next step and would enable direct coefficient comparisons.

## What the Pipeline Adds

Beyond the improved classifier, the pipeline adds group-aware cross-validation that splits by organization rather than randomly, speaker attribution with confidence tiers instead of binary assignment, an interactive Streamlit dashboard with case studies and model diagnostics, and a fully reproducible architecture with configuration-driven stages and data validation.

## Next Steps

The most natural next step is running the full thesis model specifications on the pipeline's rebuilt dataset. The thesis tested issue salience, organization age, seniority, and legislative activity, all of which are available in the pipeline's data but not yet in the current regressions. Adding them would enable direct coefficient comparisons across both datasets and answer whether the thesis's findings (like the unexpected negative seniority effect) hold up on the rebuilt dataset with a better classifier. That's the extension that would generate the most analytical value for the least additional work.

Beyond that: experimenting with transformer-based classifiers (DistilBERT fine-tuning on the 1,222 labels) to test whether the TF-IDF approach has hit its ceiling; updating the lobbying data beyond WRS 2011 using OpenSecrets annual figures; extending the time series to the 116th-117th Congress, which the pipeline architecture already supports; and labeling 500+ additional mentions from the 115th Congress, particularly targeting cases where the classifier is least confident.
