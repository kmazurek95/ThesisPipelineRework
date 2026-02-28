"""
Home page — Story-driven landing page for portfolio showcase.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils.config import (
    configure_page, apply_custom_css,
    HERO_ORG_COLORS, HERO_ORG_META,
)
from dashboard.utils.data_loader import (
    load_level1_data, load_level2_data,
    load_classifier_metrics, load_regression_results,
    get_summary_stats, HERO_ORG_IDS,
)


def build_hero_scatter(level2: pd.DataFrame, level1: pd.DataFrame) -> go.Figure:
    """Lobbying vs Prominence scatter with hero orgs highlighted."""
    names = level1.groupby('org_id')['org_name'].first().reset_index()
    df = level2.merge(names, on='org_id', how='left')

    df = df[(df['total_mentions'] >= 50) & (df['LOBBYING11'] > 0)].copy()
    df['log_lobbying'] = np.log10(df['LOBBYING11'].clip(lower=1))
    df['is_hero'] = df['org_id'].isin(HERO_ORG_IDS.keys())

    bg = df[~df['is_hero']]
    heroes = df[df['is_hero']]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=bg['log_lobbying'], y=bg['prominence_rate'] * 100,
        mode='markers',
        marker=dict(size=6, color='#d3d3d3', opacity=0.4),
        text=bg['org_name'],
        hovertemplate='%{text}<br>Lobbying: $%{customdata:,.0f}K<br>Prominence: %{y:.1f}%<extra></extra>',
        customdata=bg['LOBBYING11'],
        name='Other organizations', showlegend=True,
    ))

    for _, row in heroes.iterrows():
        name = HERO_ORG_IDS.get(row['org_id'], row['org_name'])
        fig.add_trace(go.Scatter(
            x=[row['log_lobbying']], y=[row['prominence_rate'] * 100],
            mode='markers+text',
            marker=dict(size=14, color=HERO_ORG_COLORS.get(name, '#333'),
                        line=dict(width=2, color='white')),
            text=[name], textposition='top center',
            textfont=dict(size=12, color=HERO_ORG_COLORS.get(name, '#333')),
            hovertemplate=f'{name}<br>Lobbying: ${row["LOBBYING11"]:,.0f}K<br>'
                          f'Prominence: {row["prominence_rate"]*100:.1f}%<extra></extra>',
            name=name, showlegend=True,
        ))

    valid = df.dropna(subset=['log_lobbying', 'prominence_rate'])
    if len(valid) > 2:
        z = np.polyfit(valid['log_lobbying'], valid['prominence_rate'] * 100, 1)
        x_range = np.linspace(valid['log_lobbying'].min(), valid['log_lobbying'].max(), 50)
        fig.add_trace(go.Scatter(
            x=x_range, y=np.polyval(z, x_range),
            mode='lines', line=dict(color='rgba(102,126,234,0.5)', dash='dash', width=2),
            showlegend=False, hoverinfo='skip',
        ))

    fig.update_layout(
        title=None,
        xaxis_title='Lobbying Expenditure ($K, log scale)',
        yaxis_title='Prominence Rate (%)',
        xaxis=dict(tickvals=[1, 2, 3, 4, 5],
                   ticktext=['$10', '$100', '$1K', '$10K', '$100K']),
        plot_bgcolor='white', height=500,
        font=dict(family="Arial, sans-serif"),
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5),
        margin=dict(t=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    return fig


# --- Page ---
configure_page(title="Interest Group Prominence", icon="🏛️")
apply_custom_css()

stats = get_summary_stats()
metrics = load_classifier_metrics()
regression = load_regression_results()

# Hero
st.markdown("""
<div class="hero">
    <h1>Does Money Buy Voice?</h1>
    <p class="subtitle">How Interest Groups Shape Congressional Debate in the U.S. Congress</p>
</div>
""", unsafe_allow_html=True)

# Metrics row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mentions Analyzed", f"{stats.get('total_mentions', 0):,}")
c2.metric("Organizations", f"{stats.get('total_orgs', 0):,}")
c3.metric("Congresses", "2 (114th-115th)")
f1 = metrics.get('F1 Score', 0)
c4.metric("Classifier F1", f"{f1:.2f}" if f1 else "N/A")

st.markdown("---")

# Scatter
st.subheader("Lobbying Expenditure vs. Prominence in Congressional Record")
st.caption(
    "Each dot is an organization with 50+ mentions. "
    "Highlighted: 5 case-study organizations spanning elderly advocacy, "
    "labor, business, civil liberties, and healthcare."
)

level1 = load_level1_data()
level2 = load_level2_data()
if not level2.empty and not level1.empty:
    fig = build_hero_scatter(level2, level1)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Findings
st.subheader("Key Findings")

lobbying_coef = senate_coef = party_coef = None
if not regression.empty:
    m1 = regression[regression['Model'] == 'Model 1: Mention']
    for _, row in m1.iterrows():
        if row['Variable'] == 'log_lobbying':
            lobbying_coef = row['Coefficient']
        elif row['Variable'] == 'is_senate':
            senate_coef = row['Coefficient']
        elif row['Variable'] == 'is_democrat':
            party_coef = row['Coefficient']

f1_col, f2_col, f3_col = st.columns(3)

with f1_col:
    st.markdown('<div class="stat-box"><h3>Lobbying Predicts Prominence</h3></div>',
                unsafe_allow_html=True)
    if lobbying_coef is not None:
        st.metric("Log-Lobbying Coefficient", f"{lobbying_coef:.3f}", "p < 0.001")
    st.markdown(
        "A 10x increase in lobbying expenditure is associated with a "
        "**7.1 percentage-point** increase in the probability of receiving "
        "a prominent mention."
    )

with f2_col:
    st.markdown('<div class="stat-box"><h3>Senate Amplifies Voice</h3></div>',
                unsafe_allow_html=True)
    if senate_coef is not None:
        pct = (np.exp(senate_coef) - 1) * 100
        st.metric("Senate vs. House Effect", f"+{pct:.0f}%", "p < 0.001")
    st.markdown(
        "Interest groups receive significantly more prominent mentions "
        "in Senate floor debates compared to the House."
    )

with f3_col:
    st.markdown('<div class="stat-box"><h3>Partisan Divide</h3></div>',
                unsafe_allow_html=True)
    if party_coef is not None:
        pct = (np.exp(party_coef) - 1) * 100
        st.metric("Democrat vs. Republican", f"{pct:.0f}%", "p < 0.001")
    st.markdown(
        "Democrats are less likely than Republicans to give interest groups "
        "prominent mentions, controlling for lobbying expenditure."
    )

st.markdown("---")

# Navigation
st.subheader("Explore Further")
n1, n2, n3 = st.columns(3)
with n1:
    st.page_link("pages/1_Methodology.py", label="Methodology", icon="🔬")
    st.caption("Pipeline design, ML classification, and statistical models")
with n2:
    st.page_link("pages/2_Case_Studies.py", label="Case Studies", icon="🏛️")
    st.caption("Deep dives into AARP, AFL-CIO, ACLU, NAM, and AMA")
with n3:
    st.page_link("pages/3_Technical_Appendix.py", label="Technical Appendix", icon="📖")
    st.caption("Full model results, classifier diagnostics, and limitations")

# Disclaimer
st.markdown("---")
st.caption(
    "Curated portfolio demonstration showcasing 5 representative organizations. "
    "Full pipeline processes {:,} mentions across {:,} organizations from the "
    "114th-115th U.S. Congress (2015-2018).".format(
        stats.get('total_mentions', 0), stats.get('total_orgs', 0))
)
