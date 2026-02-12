"""Labeling Interface Dashboard - Home Page."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import configure_page, apply_custom_css, VALIDITY_COLORS, FP_TYPE_COLORS
from utils.data_loader import (
    load_progress,
    load_fp_stats,
    load_fp_by_match_type,
    check_database_exists,
)


def main():
    """Main dashboard page."""
    configure_page(title="Labeling Dashboard", icon="🏷️")
    apply_custom_css()

    st.title("🏷️ Mention Labeling Dashboard")

    # Check if database exists
    if not check_database_exists():
        st.error(
            "Database not found! Please run the initialization scripts first:\n\n"
            "```bash\n"
            "python scripts/init_labeling_db.py\n"
            "python scripts/import_mentions.py\n"
            "```"
        )
        return

    # Load progress data
    try:
        progress = load_progress()
    except Exception as e:
        st.error(f"Error loading progress data: {e}")
        return

    # Progress Overview Section
    st.header("Progress Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Mentions",
            f"{progress['total_mentions']:,}",
        )

    with col2:
        st.metric(
            "Labeled",
            f"{progress['labeled_mentions']:,}",
            delta=f"{progress['pct_complete']:.1f}%",
        )

    with col3:
        st.metric(
            "Remaining",
            f"{progress['unlabeled_mentions']:,}",
        )

    with col4:
        # Calculate labels today (would need session tracking for real implementation)
        st.metric(
            "Completion",
            f"{progress['pct_complete']:.1f}%",
        )

    # Progress bar
    st.progress(progress['pct_complete'] / 100 if progress['pct_complete'] else 0)

    # Two-column layout for charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Label Distribution")

        label_dist = progress.get('label_distribution', {})
        if label_dist:
            # Create pie chart
            labels = list(label_dist.keys())
            values = list(label_dist.values())
            colors = [VALIDITY_COLORS.get(label, '#999999') for label in labels]

            fig = go.Figure(data=[go.Pie(
                labels=[l.replace('_', ' ').title() for l in labels],
                values=values,
                marker_colors=colors,
                hole=0.4,
            )])
            fig.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                height=300,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No labels recorded yet. Start labeling to see distribution.")

    with col_right:
        st.subheader("False Positive Types")

        fp_dist = progress.get('fp_type_distribution', {})
        if fp_dist:
            # Create bar chart
            fp_types = list(fp_dist.keys())
            fp_counts = list(fp_dist.values())
            colors = [FP_TYPE_COLORS.get(t, '#999999') for t in fp_types]

            fig = go.Figure(data=[go.Bar(
                x=[t.replace('_', ' ').title() for t in fp_types],
                y=fp_counts,
                marker_color=colors,
            )])
            fig.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                height=300,
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No false positives recorded yet.")

    # False Positive Patterns Section
    st.header("False Positive Patterns")

    try:
        fp_stats = load_fp_stats()
        if not fp_stats.empty:
            st.subheader("Organizations with Highest False Positive Rates")

            # Filter to orgs with at least 5 labeled mentions
            fp_stats_filtered = fp_stats[fp_stats['labeled_count'] >= 5].head(10)

            if not fp_stats_filtered.empty:
                # Create bar chart
                fig = px.bar(
                    fp_stats_filtered,
                    x='org_id',
                    y='fp_rate',
                    color='fp_rate',
                    color_continuous_scale='Reds',
                    labels={'fp_rate': 'FP Rate', 'org_id': 'Organization'},
                    title='Top 10 Organizations by False Positive Rate (min 5 labels)',
                )
                fig.update_layout(
                    xaxis_tickangle=-45,
                    height=400,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Table view
                with st.expander("View detailed table"):
                    st.dataframe(
                        fp_stats_filtered[['org_id', 'labeled_count', 'fp_count', 'fp_rate']].rename(columns={
                            'org_id': 'Organization',
                            'labeled_count': 'Labeled',
                            'fp_count': 'False Positives',
                            'fp_rate': 'FP Rate',
                        }),
                        use_container_width=True,
                    )
            else:
                st.info("Not enough labeled data yet. Label at least 5 mentions per organization to see patterns.")
        else:
            st.info("No false positive data available yet.")
    except Exception as e:
        st.warning(f"Could not load false positive stats: {e}")

    # Match Type Analysis
    try:
        fp_by_type = load_fp_by_match_type()
        if not fp_by_type.empty:
            st.subheader("False Positive Rate by Match Type")

            col1, col2 = st.columns(2)

            for idx, row in fp_by_type.iterrows():
                with col1 if idx % 2 == 0 else col2:
                    match_type = "Acronyms" if row.get('is_acronym') else "Full Names"
                    labeled = row.get('labeled_count', 0)
                    fp_count = row.get('fp_count', 0)
                    fp_rate = row.get('fp_rate', 0) or 0

                    st.markdown(f"""
                    <div class="info-box">
                        <h4>{match_type}</h4>
                        <p>Labeled: <strong>{labeled:,}</strong></p>
                        <p>False Positives: <strong>{fp_count:,}</strong></p>
                        <p>FP Rate: <strong>{fp_rate:.1%}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load match type stats: {e}")

    # Quick Actions Section
    st.header("Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📋 Start Labeling", use_container_width=True, type="primary"):
            st.switch_page("pages/1_📋_Mention_Labeling.py")

    with col2:
        if st.button("📊 View Analytics", use_container_width=True):
            st.switch_page("pages/2_📊_Analytics.py")

    with col3:
        if st.button("📤 Export Data", use_container_width=True):
            st.switch_page("pages/3_📤_Export.py")

    # Sidebar with additional info
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This interface allows you to:

        - **Label mentions** as true/false positives
        - **Classify prominence** (prominent vs. passing)
        - **Identify patterns** in false positives
        - **Export training data** for the classifier

        ### Labeling Guidelines

        **True Mention**: The text actually refers to the interest group

        **False Positive**: The text does NOT refer to the interest group (e.g., person's name, different org, procedural text)

        **Prominent**: The interest group is a main subject of the text

        **Passing**: Brief or tangential mention
        """)

        st.divider()

        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()


if __name__ == "__main__":
    main()
