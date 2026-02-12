"""
Technical Appendix — Full model results, classifier diagnostics, and limitations.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils.config import configure_page, apply_custom_css
from dashboard.utils.data_loader import (
    load_regression_results, load_classifier_metrics,
    load_classifier_comparison, get_figure_path,
)

configure_page(title="Technical Appendix", icon="📖")
apply_custom_css()

st.title("Technical Appendix")
st.markdown("Full model results, classifier diagnostics, data dictionary, and limitations.")

# ── Data Dictionary ────────────────────────────────────────────────
st.header("Data Dictionary")

with st.expander("Level 1 — Mention-Level (53,892 records)", expanded=False):
    st.markdown("""
    | Column | Description |
    |--------|-------------|
    | `org_id` | Unique organization identifier |
    | `org_name` | Organization name |
    | `sentence` | Sentence containing the mention |
    | `prominence_score` | Model confidence (0-1) |
    | `prominence_prediction` | Binary classification (0=passing, 1=prominent) |
    | `party` | Speaker's party (D/R/I) — available for ~41% of mentions |
    | `chamber` | Senate (S) or House (H) |
    | `date` | Date of Congressional Record entry |
    | `LOBBYING11` | Organization's 2011 lobbying expenditure ($K) |
    | `CATEGORY` | WRS organization type |
    | `ABBREVCAT` | Abbreviated category |
    """)

with st.expander("Level 2 — Organization-Level (2,260 records)", expanded=False):
    st.markdown("""
    | Column | Description |
    |--------|-------------|
    | `org_id` | Unique organization identifier |
    | `total_mentions` | Total mentions across both congresses |
    | `prominence_rate` | Proportion of mentions classified as prominent |
    | `avg_prominence` | Mean prominence score |
    | `LOBBYING11` | Lobbying expenditure ($K, 2011) |
    | `ABBREVCAT` | Abbreviated organization category |
    """)

with st.expander("Level 3 — Politician-Level (490 records)", expanded=False):
    st.markdown("""
    | Column | Description |
    |--------|-------------|
    | `bioGuideId` | Biographical Directory ID |
    | `fullName` | Legislator name |
    | `party` | Political party |
    | `chamber` | Senate or House |
    | `total_mentions` | Total interest group mentions by this legislator |
    | `prominence_rate` | Proportion classified as prominent |
    """)

st.markdown("---")

# ── Full Regression Results ───────────────────────────────────────
st.header("Full Regression Results")

regression = load_regression_results()
if not regression.empty:
    for model_name in regression['Model'].unique():
        model_df = regression[regression['Model'] == model_name].copy()
        st.subheader(model_name)

        display = model_df[['Variable', 'Coefficient', 'Std Error', 'P-value', 'Significant']].copy()
        display['Coefficient'] = display['Coefficient'].apply(lambda x: f"{x:.4f}")
        display['Std Error'] = display['Std Error'].apply(lambda x: f"{x:.4f}")
        display['P-value'] = display['P-value'].apply(
            lambda x: "< 0.001" if x < 0.001 else f"{x:.4f}")
        st.dataframe(display, use_container_width=True, hide_index=True)
else:
    st.warning("Regression results not available.")

st.markdown("---")

# ── Classifier Diagnostics ────────────────────────────────────────
st.header("Classifier Diagnostics")

metrics = load_classifier_metrics()
comparison = load_classifier_comparison()

if metrics:
    st.subheader("Final Model Performance")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("F1 Score", f"{metrics.get('F1 Score', 0):.3f}")
    mc2.metric("Precision", f"{metrics.get('Precision', 0):.3f}")
    mc3.metric("Recall", f"{metrics.get('Recall', 0):.3f}")
    mc4.metric("ROC-AUC", f"{metrics.get('ROC-AUC', 0):.3f}")
    mc5.metric("Accuracy", f"{metrics.get('Accuracy', 0):.3f}")

if not comparison.empty:
    st.subheader("Model Comparison")
    display_comp = comparison[['Model', 'F1', 'Precision', 'Recall', 'ROC-AUC']].copy()
    for col in ['F1', 'Precision', 'Recall', 'ROC-AUC']:
        display_comp[col] = display_comp[col].apply(lambda x: f"{x:.3f}")
    st.dataframe(display_comp, use_container_width=True, hide_index=True)

# All classifier figures
st.subheader("Diagnostic Plots")

figure_names = [
    ("classifier_confusion_matrix.png", "Confusion Matrix"),
    ("classifier_roc_pr_curves.png", "ROC & PR Curves"),
    ("classifier_calibration.png", "Calibration Plot"),
    ("classifier_feature_importance.png", "Feature Importance"),
    ("classifier_shap_summary.png", "SHAP Summary"),
    ("classifier_model_comparison.png", "Model Comparison"),
    ("classifier_class_distribution.png", "Class Distribution"),
    ("classifier_text_length.png", "Text Length Distribution"),
    ("classifier_accuracy_by_confidence.png", "Accuracy by Confidence"),
    ("classifier_threshold_optimization.png", "Threshold Optimization"),
    ("classifier_wordclouds.png", "Word Clouds"),
]

# Display in 2-column grid
for i in range(0, len(figure_names), 2):
    cols = st.columns(2)
    for j, col in enumerate(cols):
        idx = i + j
        if idx < len(figure_names):
            fname, caption = figure_names[idx]
            fpath = get_figure_path(fname)
            if fpath.exists():
                with col:
                    st.image(str(fpath), caption=caption, use_container_width=True)

st.markdown("---")

# ── Limitations ───────────────────────────────────────────────────
st.header("Limitations & Caveats")

st.markdown("""
<div class="warning-box">
<strong>This analysis has important limitations that should be considered when interpreting results:</strong>
</div>
""", unsafe_allow_html=True)

st.markdown("""
1. **Speaker attribution coverage (41%):** Only 41.3% of mentions have an identified
   speaker (via Congress.gov member data). The remaining 58.7% are legislative text
   segments (amendments, bill text, inserted letters) where the speaker is not directly
   identified. Partisan and chamber analyses are limited to the attributed subset.

2. **Model-predicted prominence:** Prominence labels are machine-predicted (Logistic
   Regression, F1=0.91), not human-verified for all 53,892 mentions. The 1,222
   hand-labeled training examples may not capture all edge cases.

3. **Organization matching:** Interest group mentions are identified via dictionary
   matching against 2,260 organization names and aliases. Some false positives may
   remain (e.g., acronym collisions, legislative amendment text). The curated
   case studies use well-known organizations to minimize this issue.

4. **Lobbying data vintage:** Lobbying expenditure data comes from the Washington
   Representatives Study 2011 wave — the latest available. Lobbying patterns may
   have shifted between 2011 and the 114th-115th Congress (2015-2018).

5. **Observational design:** This is a correlational study. The relationship between
   lobbying expenditure and prominence does not establish causation — organizations
   that lobby more may differ in other ways that affect how they are mentioned.
""")

st.markdown("---")

# ── Links ─────────────────────────────────────────────────────────
st.header("Project Links")
st.markdown("""
- [GitHub Repository](https://github.com/kmazurek95/ThesisPipelineRework)
- [Methodology Documentation](https://github.com/kmazurek95/ThesisPipelineRework/blob/main/docs/METHODOLOGY.md)
- [Replication Guide](https://github.com/kmazurek95/ThesisPipelineRework/blob/main/docs/REPLICATION.md)
- [R Multilevel Analysis](https://github.com/kmazurek95/ThesisPipelineRework/tree/main/R_analysis)
""")
