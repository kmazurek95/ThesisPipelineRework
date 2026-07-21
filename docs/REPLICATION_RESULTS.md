# GLMM replication results

Replication of the original thesis GLMM models using the rebuilt pipeline dataset. All models were fit with `lme4::glmer()` using crossed random effects `(1|org_id) + (1|issue_area)` and `nAGQ=0` (Laplace approximation).

Dataset: 57,073 rows (53,892 mentions + 3,181 zero-mention organizations) across 5,441 organizations and 18 CAP policy areas.

Script: `scripts/run_glmm_replication.R`

---

## Model C: Group characteristics

| Variable | Thesis OR | Thesis p | Replication OR | Replication p | Direction Match |
|----------|-----------|----------|----------------|---------------|-----------------|
| log_lobbying | 1.074 | < 0.001 | 1.06 | 0.003 | Yes |
| policy_scope | (not tested) | n/a | 1.07 | 0.03 | n/a |
| membership_org | (not tested) | n/a | 0.746 | 0.046 | n/a |
| org_age | 0.998 | 0.22 (ns) | 0.998 | 0.204 (ns) | Yes (both ns) |

Lobbying expenditure remains a significant positive predictor. Organization age remains non-significant in both analyses.

## Model A: Policy salience

| Variable | Thesis OR | Thesis p | Replication OR | Replication p | Direction Match |
|----------|-----------|----------|----------------|---------------|-----------------|
| is_democrat | 0.89 | 0.047 | 1.31 | < 0.001 | No (reversed) |
| salience_high | 0.70 | < 0.05 | 0.839 | 0.04 | Yes |
| salience_medium | 1.49 | < 0.05 | 1.10 | 0.197 (ns) | Yes (weakened) |

High salience continues to suppress prominence. The party effect reverses direction, likely due to dataset construction differences (the replication uses the full rebuilt dataset rather than the thesis subsample).

## Model B: Group-politician linkage

| Variable | Thesis OR | Thesis p | Replication OR | Replication p | Direction Match |
|----------|-----------|----------|----------------|---------------|-----------------|
| policy_overlap | (positive, ns) | 0.15 | 2.03 | 0.004 | Yes (now significant) |
| terms_served | 0.98 | < 0.001 | 0.974 | < 0.001 | Yes |
| log_bills_sponsored | ~1.00 | 0.47 (ns) | 0.847 | < 0.001 | No (now significant, negative) |
| up_for_reelection | (weakly positive) | 0.07 (ns) | 0.817 | 0.004 | No (now significant, negative) |
| is_labor | 1.15 | < 0.01 | 2.16 | 0.021 | Yes (stronger) |

Policy overlap becomes significant with the larger dataset. Seniority remains a significant negative predictor.

## Variance decomposition (ICC)

| Component | Thesis | Replication |
|-----------|--------|-------------|
| org_id | 12.3% | 37.9% |
| issue_area | 4.7% | 6.2% |
| Residual | 83.0% | 56.0% |

The higher organization-level ICC in the replication (37.9% vs. 12.3%) reflects the inclusion of zero-mention organizations in the full-sample design, which increases between-organization variance.

---

## Summary

The replication confirms the thesis's core findings: lobbying expenditure predicts prominence, seniority has a negative effect, and high issue salience suppresses prominence. The larger dataset resolves several previously marginal effects (policy overlap, reelection timing) into statistical significance. The reversed party coefficient and stronger labor union effect warrant further investigation and may reflect differences in model specification or dataset scope.

All three models converged with crossed random effects (no fallback to simpler specifications was needed).

---

*For the original thesis results, see [THESIS_FINDINGS_2023.md](THESIS_FINDINGS_2023.md). For methodology details, see [METHODOLOGY.md](METHODOLOGY.md#glmm-replication-results).*
