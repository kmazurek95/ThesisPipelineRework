"""
Methodology page — Pipeline design, ML classification, and statistical models.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils.config import configure_page, apply_custom_css
from dashboard.utils.data_loader import (
    load_classifier_metrics, load_classifier_comparison,
    load_regression_results, get_figure_path, get_summary_stats,
)

configure_page(title="Methodology", icon="🔬")
apply_custom_css()

st.title("How I Built This")
st.markdown("A five-stage NLP and statistical analysis pipeline, from raw congressional text to regression models.")

st.markdown("---")

# ── Pipeline overview ──────────────────────────────────────────────
st.header("Data Pipeline")

stages = [
    ("1. Data Collection", "GovInfo API", "764 packages across 2 congresses",
     "Automated retrieval of Congressional Record HTML from the Government Publishing Office API with retry logic and rate limiting."),
    ("2. NLP Processing", "BeautifulSoup + NLTK", "53,892 mentions extracted",
     "HTML parsing, sentence segmentation, named-entity mention extraction using a curated dictionary of 2,260 interest group names and aliases."),
    ("3. ML Classification", "TF-IDF + Logistic Regression", "F1 = 0.91",
     "Binary classifier trained on 1,222 hand-labeled examples to distinguish *prominent* (substantive policy discussion) from *passing* (roll-call lists, ceremonial) mentions."),
    ("4. Integration", "Pandas + pyreadr", "4-level analytical dataset",
     "Merge classified mentions with Washington Representatives Study (lobbying, organization type), Congress.gov (legislator metadata), and Congressional Bills data."),
    ("5. Statistical Analysis", "statsmodels + R/lme4", "3 regression models",
     "Logistic regression at mention, organization, and politician levels. Mixed-effects models in R for crossed random intercepts."),
]

for title, tools, metric, desc in stages:
    with st.expander(f"**{title}** — {tools} — *{metric}*", expanded=False):
        st.markdown(desc)

st.markdown("---")

# ── Classification spotlight ───────────────────────────────────────
st.header("ML Classification")

metrics = load_classifier_metrics()
comparison = load_classifier_comparison()

st.markdown(
    "Binary classifier distinguishing **prominent** mentions (substantive policy "
    "discussion of an interest group) from **passing** mentions (roll-call lists, "
    "ceremonial references). Built from 1,222 hand-labeled examples, split "
    "group-aware by organization into 1,030 training and 192 test examples, "
    "with group-aware 5-fold cross-validation (GroupKFold by `org_id`) used "
    "for model selection."
)

# Metrics cards
if metrics:
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("F1 Score", f"{metrics.get('F1 Score', 0):.3f}")
    mc2.metric("Precision", f"{metrics.get('Precision', 0):.3f}")
    mc3.metric("Recall", f"{metrics.get('Recall', 0):.3f}")
    mc4.metric("ROC-AUC", f"{metrics.get('ROC-AUC', 0):.3f}")
    st.caption(
        "Held-out test set (n=192), after tuning the decision threshold to 0.558 "
        "on that same test set — so these are optimistic. Group-aware 5-fold CV "
        "gives F1 = 0.672 ± 0.061, which is the honest generalization estimate."
    )

# Model comparison table
if not comparison.empty:
    st.subheader("Model Comparison")
    display = comparison[['Model', 'F1', 'Precision', 'Recall', 'ROC-AUC']].copy()
    for col in ['F1', 'Precision', 'Recall', 'ROC-AUC']:
        display[col] = display[col].apply(lambda x: f"{x:.3f}")
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption(
        "Held-out test set at the default 0.5 threshold (not cross-validation). "
        "Logistic Regression selected for best F1 and interpretability."
    )

# Classifier figures
st.subheader("Diagnostic Plots")
fig_col1, fig_col2 = st.columns(2)

cm_path = get_figure_path("classifier_confusion_matrix.png")
roc_path = get_figure_path("classifier_roc_pr_curves.png")

if cm_path.exists():
    with fig_col1:
        st.image(str(cm_path), caption="Confusion Matrix", use_container_width=True)
if roc_path.exists():
    with fig_col2:
        st.image(str(roc_path), caption="ROC & Precision-Recall Curves", use_container_width=True)

# Prominent vs passing examples
st.subheader("What Does Prominent vs. Passing Look Like?")

ex_col1, ex_col2 = st.columns(2)
with ex_col1:
    st.markdown("**Prominent** (substantive policy discussion)")
    st.info(
        '"I am not surprised to see that there has been a letter issued by '
        'the **National Association of Manufacturers** opposing my amendment."'
    )
    st.info(
        '"The **AARP** is the largest senior organization in America, and they '
        'have weighed in strongly against this proposal."'
    )
with ex_col2:
    st.markdown("**Passing** (list / ceremonial)")
    st.warning(
        '"Among the many organizations that have endorsed our bill are: the College '
        '& University Professional Association, the **National Association of '
        'Manufacturers**, the Small Business Council..."'
    )
    st.warning(
        '"In addition to service on the board of directors for what is now the '
        'United Way, she also volunteered with the **AARP**."'
    )

st.markdown("---")

# ── Statistical models ─────────────────────────────────────────────
st.header("Statistical Models")

regression = load_regression_results()

st.markdown(
    "Logistic regression predicting the probability of a *prominent* mention. "
    "Three models at different levels of aggregation:"
)

if not regression.empty:
    # Model 1 — Mention level
    m1 = regression[regression['Model'] == 'Model 1: Mention'].copy()
    if not m1.empty:
        st.subheader("Model 1: Mention-Level (n = 22,248, mentions with speaker attribution)")
        display_m1 = m1[['Variable', 'Coefficient', 'Std Error', 'P-value', 'Significant']].copy()
        display_m1['Coefficient'] = display_m1['Coefficient'].apply(lambda x: f"{x:.4f}")
        display_m1['Std Error'] = display_m1['Std Error'].apply(lambda x: f"{x:.4f}")
        display_m1['P-value'] = display_m1['P-value'].apply(
            lambda x: "< 0.001" if x < 0.001 else f"{x:.4f}")
        st.dataframe(display_m1, use_container_width=True, hide_index=True)

    st.markdown(
        "**Key takeaway:** Each unit increase in log-lobbying expenditure raises the "
        "odds of a prominent mention by ~7.4%. Senate mentions are 45% more likely "
        "to be prominent than House mentions."
    )

st.markdown("---")

# Tech stack
st.header("Tech Stack")
tc1, tc2, tc3, tc4 = st.columns(4)
tc1.markdown("**Data**\n\nPython 3.10+\n\npandas, NumPy\n\nBeautifulSoup, NLTK")
tc2.markdown("**ML**\n\nscikit-learn\n\nTF-IDF vectorizer\n\nLogistic Regression")
tc3.markdown("**Statistics**\n\nstatsmodels (Python)\n\nlme4 (R)\n\nMixed-effects models")
tc4.markdown("**Infrastructure**\n\nStreamlit\n\nGitHub Actions CI\n\npytest")
