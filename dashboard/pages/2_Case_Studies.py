"""
Case Studies page — Deep dives into 5 curated organizations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils.config import (
    configure_page, apply_custom_css,
    PARTY_COLORS, HERO_ORG_COLORS, HERO_ORG_META, PROMINENCE_COLORS,
)
from dashboard.utils.data_loader import (
    load_hero_mentions, load_level2_data, HERO_ORG_IDS,
)

configure_page(title="Case Studies", icon="🏛️")
apply_custom_css()

st.title("Organization Case Studies")
st.markdown(
    "Deep dives into 5 representative organizations spanning elderly advocacy, "
    "labor, business, civil liberties, and healthcare."
)

# Load data
hero_mentions = load_hero_mentions()
level2 = load_level2_data()

if hero_mentions.empty:
    st.error("Could not load mention data.")
    st.stop()

# ── Org selector ──────────────────────────────────────────────────
org_options = {v: k for k, v in HERO_ORG_IDS.items()}
selected_name = st.selectbox(
    "Select an organization",
    options=list(org_options.keys()),
    index=0,
)
selected_id = org_options[selected_name]
meta = HERO_ORG_META[selected_id]

org_data = hero_mentions[hero_mentions['org_id'] == selected_id].copy()

# ── Profile card ──────────────────────────────────────────────────
st.markdown("---")

pc1, pc2, pc3, pc4, pc5 = st.columns(5)
pc1.metric("Sector", meta['sector'])
pc2.metric("Total Mentions", f"{len(org_data):,}")
prom_rate = org_data['prominence_prediction'].mean() * 100
pc3.metric("Prominence Rate", f"{prom_rate:.1f}%")
pc4.metric("Lobbying ($K)", f"${meta['lobbying_k']:,}")

# Party breakdown (where speaker is known)
with_party = org_data[org_data['party'].notna()]
if not with_party.empty:
    top_party = with_party['party'].value_counts().index[0]
    pc5.metric("Top Party (speakers)", top_party)
else:
    pc5.metric("Speaker Data", "Limited")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────
tab_examples, tab_trends, tab_partisan, tab_compare = st.tabs([
    "Text Examples", "Time Trends", "Partisan Patterns", "Cross-Org Comparison"
])

# ── Text Examples ─────────────────────────────────────────────────
with tab_examples:
    st.subheader("Prominent vs. Passing Mentions")
    st.caption("Showing only high-confidence predictions (score > 0.7 or < 0.3).")

    ex1, ex2 = st.columns(2)

    prominent = org_data[org_data['prominence_score'] > 0.7].head(3)
    passing = org_data[org_data['prominence_score'] < 0.3].head(3)

    with ex1:
        st.markdown("**Prominent** (substantive discussion)")
        if prominent.empty:
            st.info("No high-confidence prominent examples available.")
        for _, row in prominent.iterrows():
            sentence = str(row.get('sentence', ''))[:300]
            score = row.get('prominence_score', 0)
            st.success(f'"{sentence}..." \n\n*Confidence: {score:.2f}*')

    with ex2:
        st.markdown("**Passing** (list / ceremonial)")
        if passing.empty:
            st.info("No high-confidence passing examples available.")
        for _, row in passing.iterrows():
            sentence = str(row.get('sentence', ''))[:300]
            score = row.get('prominence_score', 0)
            st.warning(f'"{sentence}..." \n\n*Confidence: {1 - score:.2f}*')

# ── Time Trends ───────────────────────────────────────────────────
with tab_trends:
    st.subheader("Mentions Over Time")

    if 'date' in org_data.columns:
        org_data['quarter'] = org_data['date'].dt.to_period('Q').astype(str)
        quarterly = org_data.groupby(['quarter', 'prominence_prediction']).size().reset_index(name='count')
        quarterly['Type'] = quarterly['prominence_prediction'].map({1: 'Prominent', 0: 'Passing'})

        fig = px.bar(
            quarterly, x='quarter', y='count', color='Type',
            color_discrete_map=PROMINENCE_COLORS,
            title=f"{selected_name}: Mentions by Quarter",
            labels={'count': 'Mentions', 'quarter': 'Quarter'},
        )
        fig.update_layout(
            plot_bgcolor='white', barmode='stack',
            font=dict(family="Arial, sans-serif"),
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Date information not available.")

    # Chamber split
    st.subheader("Chamber Distribution")
    if 'chamber' in org_data.columns:
        chamber_counts = org_data['chamber'].value_counts()
        ch1, ch2 = st.columns(2)
        ch1.metric("Senate", f"{chamber_counts.get('S', 0):,}")
        ch2.metric("House", f"{chamber_counts.get('H', 0):,}")

# ── Partisan Patterns ─────────────────────────────────────────────
with tab_partisan:
    st.subheader("Who Mentions This Organization?")

    if not with_party.empty:
        party_prom = with_party.groupby('party').agg(
            mentions=('prominence_prediction', 'count'),
            prominence_rate=('prominence_prediction', 'mean'),
        ).reset_index()
        party_prom['prominence_rate'] = party_prom['prominence_rate'] * 100

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=party_prom['party'], y=party_prom['mentions'],
            marker_color=[PARTY_COLORS.get(p, '#999') for p in party_prom['party']],
            text=party_prom['mentions'], textposition='auto',
        ))
        fig.update_layout(
            title=f"{selected_name}: Mentions by Party (speaker-attributed only)",
            xaxis_title='Party', yaxis_title='Number of Mentions',
            plot_bgcolor='white', showlegend=False,
            font=dict(family="Arial, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Prominence by party
        st.markdown("**Prominence rate by party:**")
        for _, row in party_prom.iterrows():
            party_label = {'D': 'Democrat', 'R': 'Republican', 'I': 'Independent'}.get(
                row['party'], row['party'])
            st.markdown(f"- **{party_label}**: {row['prominence_rate']:.1f}% prominent ({row['mentions']} mentions)")

        st.caption(
            f"Note: Only {len(with_party):,} of {len(org_data):,} mentions "
            f"({len(with_party)/len(org_data)*100:.0f}%) have identified speakers."
        )
    else:
        st.info("No speaker-attributed mentions available for this organization.")

# ── Cross-Org Comparison ──────────────────────────────────────────
with tab_compare:
    st.subheader("All 5 Organizations Compared")

    compare_data = []
    for oid, name in HERO_ORG_IDS.items():
        org_df = hero_mentions[hero_mentions['org_id'] == oid]
        m = HERO_ORG_META[oid]
        compare_data.append({
            'Organization': name,
            'Mentions': len(org_df),
            'Prominence Rate (%)': round(org_df['prominence_prediction'].mean() * 100, 1),
            'Lobbying ($K)': m['lobbying_k'],
            'Sector': m['sector'],
        })
    compare_df = pd.DataFrame(compare_data)

    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    # Grouped bar: prominence rate
    fig = px.bar(
        compare_df, x='Organization', y='Prominence Rate (%)',
        color='Organization', color_discrete_map=HERO_ORG_COLORS,
        title="Prominence Rate by Organization",
    )
    fig.update_layout(
        plot_bgcolor='white', showlegend=False,
        font=dict(family="Arial, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Lobbying vs prominence scatter
    fig2 = px.scatter(
        compare_df, x='Lobbying ($K)', y='Prominence Rate (%)',
        text='Organization', size='Mentions', color='Organization',
        color_discrete_map=HERO_ORG_COLORS,
        title="Lobbying Expenditure vs. Prominence Rate",
        log_x=True,
    )
    fig2.update_traces(textposition='top center')
    fig2.update_layout(
        plot_bgcolor='white', showlegend=False,
        font=dict(family="Arial, sans-serif"),
    )
    st.plotly_chart(fig2, use_container_width=True)
