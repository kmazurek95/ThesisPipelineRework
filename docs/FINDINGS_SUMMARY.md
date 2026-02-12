# Research Findings Summary

## Interest Group Prominence in Congressional Speech

**Author:** Kaleb Mazurek | **Data:** 114th-115th U.S. Congress (2015-2018)

---

### Research Question

What determines whether interest groups receive **substantive** versus **passing** mentions in congressional floor speeches? This project examines how organizational characteristics (lobbying expenditure, group type) and speaker attributes (party, chamber) predict prominence.

### Dataset

| Measure | Value |
|---------|-------|
| Total mentions analyzed | 53,892 |
| Unique organizations | 2,260 |
| Congress members | 490 |
| Policy areas | 18 |
| Congresses | 114th (2015-2017), 115th (2017-2018) |
| Prominence rate | 35.5% |

### Classification

Interest group mentions are classified as **prominent** (substantive discussion) or **passing** (brief reference) using a Logistic Regression + TF-IDF pipeline trained on 1,222 hand-labeled examples.

| Metric | Value |
|--------|-------|
| F1 Score | 0.91 |
| Precision | 0.87 |
| Recall | 0.94 |
| Cross-validation | 5-fold group-aware (prevents org leakage) |

### Key Findings

| Finding | Evidence |
|---------|----------|
| **Lobbying predicts prominence** | +7.4% higher odds per log-unit increase in expenditure (p < 0.001) |
| **Senators > Representatives** | +45% higher odds of giving prominent mentions (p < 0.001) |
| **Partisan gap** | Democrats give 23% fewer prominent mentions than Republicans (p < 0.001) |
| **Single-issue groups stand out** | +41% higher prominence rate compared to multi-issue organizations (p < 0.001) |

### Methodology

The analysis employs a **multi-level data architecture**:

1. **Mention-level** (N=53,892) — Logistic regression: prominence ~ lobbying + party + chamber + org type
2. **Organization-level** (N=2,260) — OLS: avg prominence ~ lobbying + mentions + org type
3. **Politician-level** (N=490) — OLS: avg prominence ~ party + chamber + mentions
4. **R multilevel models** — GLMER with crossed random effects for organizations and policy areas

### Further Reading

- [Full methodology](METHODOLOGY.md)
- [Replication guide](REPLICATION.md)
- [Classification deep-dive](../notebooks/Classification_Analysis.ipynb)
- [R multilevel analysis](../R_analysis/Multilevel_Analysis.Rmd)
- [Interactive dashboard](../dashboard/Home.py)
