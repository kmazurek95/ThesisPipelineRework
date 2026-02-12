"""Analytics page for investigating false positive patterns."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import configure_page, apply_custom_css, VALIDITY_COLORS, FP_TYPE_COLORS
from utils.data_loader import (
    load_progress,
    load_fp_stats,
    load_fp_by_match_type,
    load_mentions,
    check_database_exists,
)
from utils.database import get_connection


def load_detailed_fp_analysis():
    """Load detailed false positive analysis data."""
    conn = get_connection()

    # FP by organization with variation details
    query = """
        SELECT
            m.org_id,
            m.interest_group,
            m.variation,
            m.is_acronym,
            ml.false_positive_type,
            COUNT(*) as count
        FROM mentions m
        JOIN mention_labels ml ON m.mention_id = ml.mention_id
        WHERE ml.validity_label = 'false_positive'
        GROUP BY m.org_id, m.interest_group, m.variation, m.is_acronym, ml.false_positive_type
        ORDER BY count DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_problematic_patterns():
    """Identify problematic matching patterns."""
    conn = get_connection()

    # Find variations with high FP rates
    query = """
        SELECT
            m.org_id,
            m.interest_group,
            m.variation,
            m.is_acronym,
            COUNT(*) as total_labeled,
            SUM(CASE WHEN ml.validity_label = 'false_positive' THEN 1 ELSE 0 END) as fp_count,
            ROUND(100.0 * SUM(CASE WHEN ml.validity_label = 'false_positive' THEN 1 ELSE 0 END) / COUNT(*), 1) as fp_rate
        FROM mentions m
        JOIN mention_labels ml ON m.mention_id = ml.mention_id
        GROUP BY m.org_id, m.interest_group, m.variation, m.is_acronym
        HAVING total_labeled >= 3
        ORDER BY fp_rate DESC, total_labeled DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_fp_examples(org_id=None, variation=None, fp_type=None, limit=10):
    """Load example false positives for investigation."""
    conn = get_connection()

    query = """
        SELECT
            m.mention_id,
            m.org_id,
            m.interest_group,
            m.variation,
            m.is_acronym,
            m.sentence,
            m.paragraph,
            m.date,
            m.granuleId,
            ml.false_positive_type,
            ml.labeler_notes
        FROM mentions m
        JOIN mention_labels ml ON m.mention_id = ml.mention_id
        WHERE ml.validity_label = 'false_positive'
    """
    params = []

    if org_id:
        query += " AND m.org_id = ?"
        params.append(org_id)

    if variation:
        query += " AND m.variation = ?"
        params.append(variation)

    if fp_type:
        query += " AND ml.false_positive_type = ?"
        params.append(fp_type)

    query += " ORDER BY ml.labeled_at DESC LIMIT ?"
    params.append(limit)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def main():
    """Main analytics page."""
    configure_page(title="Analytics", icon="📊")
    apply_custom_css()

    st.title("📊 False Positive Analytics")

    if not check_database_exists():
        st.error("Database not found. Please initialize first.")
        return

    # Load progress for overview
    progress = load_progress()

    # Overview metrics
    st.header("Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Labeled", f"{progress['labeled_mentions']:,}")

    with col2:
        fp_count = progress.get('label_distribution', {}).get('false_positive', 0)
        st.metric("False Positives", f"{fp_count:,}")

    with col3:
        true_count = progress.get('label_distribution', {}).get('true_mention', 0)
        st.metric("True Mentions", f"{true_count:,}")

    with col4:
        if progress['labeled_mentions'] > 0:
            fp_rate = fp_count / progress['labeled_mentions'] * 100
        else:
            fp_rate = 0
        st.metric("Overall FP Rate", f"{fp_rate:.1f}%")

    st.divider()

    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "By Organization",
        "By Match Type",
        "Problematic Patterns",
        "FP Examples"
    ])

    with tab1:
        st.subheader("False Positive Rate by Organization")

        try:
            fp_stats = load_fp_stats()

            if fp_stats.empty:
                st.info("No labeled data yet. Start labeling to see organization statistics.")
            else:
                # Filter controls
                min_labels = st.slider(
                    "Minimum labels per organization",
                    min_value=1,
                    max_value=50,
                    value=5,
                    key="org_min_labels"
                )

                filtered = fp_stats[fp_stats['labeled_count'] >= min_labels]

                if not filtered.empty:
                    # Bar chart of FP rates
                    fig = px.bar(
                        filtered.head(20),
                        x='org_id',
                        y='fp_rate',
                        color='fp_rate',
                        color_continuous_scale='RdYlGn_r',
                        labels={'fp_rate': 'FP Rate (%)', 'org_id': 'Organization'},
                        title=f'Top 20 Organizations by FP Rate (min {min_labels} labels)',
                        hover_data=['labeled_count', 'fp_count'],
                    )
                    fig.update_layout(xaxis_tickangle=-45, height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    # Detailed table
                    st.subheader("Detailed Statistics")
                    display_df = filtered[['org_id', 'labeled_count', 'fp_count', 'fp_rate']].copy()
                    display_df.columns = ['Organization', 'Total Labeled', 'False Positives', 'FP Rate (%)']
                    st.dataframe(display_df, use_container_width=True, height=400)
                else:
                    st.info(f"No organizations have at least {min_labels} labels yet.")

        except Exception as e:
            st.error(f"Error loading organization stats: {e}")

    with tab2:
        st.subheader("False Positive Rate by Match Type")

        try:
            fp_by_type = load_fp_by_match_type()

            if fp_by_type.empty:
                st.info("No labeled data yet.")
            else:
                col1, col2 = st.columns(2)

                for idx, row in fp_by_type.iterrows():
                    match_type = "Acronyms" if row.get('is_acronym') else "Full Names"
                    labeled = row.get('labeled_count', 0)
                    fp_count = row.get('fp_count', 0)
                    fp_rate = (row.get('fp_rate', 0) or 0) * 100  # Convert to percentage

                    with col1 if idx == 0 else col2:
                        st.markdown(f"""
                        <div class="card">
                            <h3>{match_type}</h3>
                            <p><strong>Total Labeled:</strong> {labeled:,}</p>
                            <p><strong>False Positives:</strong> {fp_count:,}</p>
                            <p><strong>FP Rate:</strong> {fp_rate:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)

                # Comparison chart
                if len(fp_by_type) >= 2:
                    fig = go.Figure(data=[
                        go.Bar(
                            name='True Mentions',
                            x=['Acronyms' if r.get('is_acronym') else 'Full Names' for _, r in fp_by_type.iterrows()],
                            y=[r.get('labeled_count', 0) - r.get('fp_count', 0) for _, r in fp_by_type.iterrows()],
                            marker_color='#2ca02c',
                        ),
                        go.Bar(
                            name='False Positives',
                            x=['Acronyms' if r.get('is_acronym') else 'Full Names' for _, r in fp_by_type.iterrows()],
                            y=[r.get('fp_count', 0) for _, r in fp_by_type.iterrows()],
                            marker_color='#d62728',
                        ),
                    ])
                    fig.update_layout(
                        barmode='stack',
                        title='Label Distribution by Match Type',
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # FP type breakdown
                st.subheader("False Positive Types Distribution")
                fp_dist = progress.get('fp_type_distribution', {})
                if fp_dist:
                    fig = px.pie(
                        names=[t.replace('_', ' ').title() for t in fp_dist.keys()],
                        values=list(fp_dist.values()),
                        title='Distribution of False Positive Types',
                        color_discrete_sequence=list(FP_TYPE_COLORS.values()),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No false positive type data yet.")

        except Exception as e:
            st.error(f"Error loading match type stats: {e}")

    with tab3:
        st.subheader("Problematic Patterns")
        st.markdown("""
        This analysis identifies **variations** (specific name/acronym matches) that have
        high false positive rates. These patterns are candidates for blocklist additions
        or special handling.
        """)

        try:
            patterns = load_problematic_patterns()

            if patterns.empty:
                st.info("Not enough labeled data to identify patterns yet.")
            else:
                # Filter controls
                col1, col2 = st.columns(2)
                with col1:
                    min_fp_rate = st.slider(
                        "Minimum FP Rate (%)",
                        min_value=0,
                        max_value=100,
                        value=50,
                        key="pattern_fp_rate"
                    )
                with col2:
                    show_acronyms_only = st.checkbox("Acronyms only", key="pattern_acronyms")

                filtered = patterns[patterns['fp_rate'] >= min_fp_rate]
                if show_acronyms_only:
                    filtered = filtered[filtered['is_acronym'] == 1]

                if not filtered.empty:
                    # Highlight high-risk patterns
                    st.markdown("### High False Positive Patterns")

                    for _, row in filtered.head(15).iterrows():
                        fp_rate = row['fp_rate']
                        if fp_rate >= 80:
                            color = "#f8d7da"
                            border = "#dc3545"
                            icon = "🚨"
                        elif fp_rate >= 60:
                            color = "#fff3cd"
                            border = "#ffc107"
                            icon = "⚠️"
                        else:
                            color = "#e7f3ff"
                            border = "#0d6efd"
                            icon = "ℹ️"

                        st.markdown(f"""
                        <div style="background-color: {color}; border-left: 4px solid {border};
                                    padding: 10px 15px; margin: 10px 0; border-radius: 0 8px 8px 0;">
                            <strong>{icon} {row['variation']}</strong>
                            {'(Acronym)' if row['is_acronym'] else '(Full Name)'}
                            <br>
                            <small>
                                Organization: {row['interest_group']} ({row['org_id']})<br>
                                FP Rate: <strong>{row['fp_rate']:.0f}%</strong>
                                ({row['fp_count']} / {row['total_labeled']} labeled)
                            </small>
                        </div>
                        """, unsafe_allow_html=True)

                    # Blocklist suggestions
                    st.subheader("Suggested Blocklist Additions")
                    high_fp = filtered[filtered['fp_rate'] >= 80]
                    if not high_fp.empty:
                        st.code("\n".join(high_fp['variation'].tolist()), language=None)
                        st.caption("These patterns have 80%+ false positive rates and may be candidates for blocklisting.")
                    else:
                        st.info("No patterns with 80%+ FP rate yet.")

                else:
                    st.info(f"No patterns with {min_fp_rate}%+ FP rate found.")

        except Exception as e:
            st.error(f"Error loading pattern analysis: {e}")

    with tab4:
        st.subheader("False Positive Examples")
        st.markdown("Investigate specific false positive cases to understand patterns.")

        try:
            # Filters
            col1, col2, col3 = st.columns(3)

            with col1:
                fp_types = [
                    'All Types',
                    'person_name',
                    'location',
                    'different_org',
                    'partial_match',
                    'procedural',
                    'abbreviation_clash',
                    'historical',
                    'other',
                ]
                selected_fp_type = st.selectbox(
                    "FP Type",
                    fp_types,
                    format_func=lambda x: x.replace('_', ' ').title(),
                )
                fp_type_filter = None if selected_fp_type == 'All Types' else selected_fp_type

            with col2:
                # Get unique org_ids with FPs
                conn = get_connection()
                orgs = pd.read_sql_query("""
                    SELECT DISTINCT m.org_id
                    FROM mentions m
                    JOIN mention_labels ml ON m.mention_id = ml.mention_id
                    WHERE ml.validity_label = 'false_positive'
                    ORDER BY m.org_id
                """, conn)
                conn.close()

                org_options = ['All Organizations'] + orgs['org_id'].tolist()
                selected_org = st.selectbox("Organization", org_options)
                org_filter = None if selected_org == 'All Organizations' else selected_org

            with col3:
                num_examples = st.slider("Number of examples", 5, 50, 10)

            # Load and display examples
            examples = load_fp_examples(
                org_id=org_filter,
                fp_type=fp_type_filter,
                limit=num_examples,
            )

            if examples.empty:
                st.info("No false positive examples match your filters.")
            else:
                for _, row in examples.iterrows():
                    with st.expander(
                        f"**{row['variation']}** - {row['interest_group']} "
                        f"({'Acronym' if row['is_acronym'] else 'Name'}) - "
                        f"{row['false_positive_type'] or 'unspecified'}"
                    ):
                        st.markdown(f"**Date:** {row['date']} | **Granule:** {row['granuleId'][:40]}...")

                        if row['sentence']:
                            st.markdown("**Context:**")
                            st.markdown(f"""
                            <div class="mention-context">
                                {row['sentence']}
                            </div>
                            """, unsafe_allow_html=True)

                        if row['labeler_notes']:
                            st.markdown(f"**Notes:** {row['labeler_notes']}")

        except Exception as e:
            st.error(f"Error loading examples: {e}")

    # Sidebar
    with st.sidebar:
        st.header("Analytics Info")
        st.markdown("""
        Use this page to:

        - **Identify patterns** in false positives
        - **Find problematic** name/acronym matches
        - **Generate blocklist** suggestions
        - **Investigate specific** false positive cases

        ### Recommended Actions

        1. Focus on variations with **50%+** FP rates
        2. Consider blocklisting patterns with **80%+** FP rates
        3. Review **acronyms** first (typically higher FP rates)
        """)

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()


if __name__ == "__main__":
    main()
