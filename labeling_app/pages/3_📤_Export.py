"""Export page for generating training data."""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import io

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import configure_page, apply_custom_css
from utils.data_loader import load_progress, check_database_exists
from utils.database import get_connection, export_training_data, get_labeled_mentions


def load_export_preview(
    include_true_only: bool = True,
    include_prominent_only: bool = False,
    limit: int = 100,
):
    """Load preview of export data."""
    conn = get_connection()

    query = """
        SELECT
            m.org_id,
            m.interest_group,
            m.variation,
            m.paragraph as p1_original,
            ml.validity_label,
            ml.prominence_label,
            ml.false_positive_type,
            m.granuleId,
            m.date,
            ml.confidence,
            ml.labeler_name,
            ml.labeled_at
        FROM mentions m
        JOIN mention_labels ml ON m.mention_id = ml.mention_id
        WHERE 1=1
    """
    params = []

    if include_true_only:
        query += " AND ml.validity_label = 'true_mention'"

    if include_prominent_only:
        query += " AND ml.prominence_label = 'prominent'"

    query += " ORDER BY ml.labeled_at DESC LIMIT ?"
    params.append(limit)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def generate_classifier_export(
    include_true_only: bool = True,
    include_prominent_only: bool = False,
):
    """Generate export in format compatible with text_classifier.py."""
    conn = get_connection()

    query = """
        SELECT
            m.org_id,
            m.paragraph as p1_original,
            CASE
                WHEN ml.prominence_label = 'prominent' THEN 1
                WHEN ml.prominence_label = 'passing' THEN 0
                ELSE NULL
            END as prominence,
            1 as paragraph_mention_count,
            0 as "10_or_more_org_mentioned",
            m.mention_id as uuid_mention,
            m.mention_id as uuid_paragraph,
            m.granuleId,
            'labeling_app' as source,
            m.interest_group,
            m.variation
        FROM mentions m
        JOIN mention_labels ml ON m.mention_id = ml.mention_id
        WHERE 1=1
    """

    if include_true_only:
        query += " AND ml.validity_label = 'true_mention'"

    if include_prominent_only:
        query += " AND ml.prominence_label = 'prominent'"

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Filter out rows without prominence labels for classifier training
    if include_true_only:
        df = df.dropna(subset=["prominence"])
        df["prominence"] = df["prominence"].astype(int)

    return df


def generate_full_export():
    """Generate full export with all labeling data."""
    conn = get_connection()

    query = """
        SELECT
            m.mention_id,
            m.org_id,
            m.interest_group,
            m.variation,
            m.match_text,
            m.is_acronym,
            m.granuleId,
            m.date,
            m.sentence,
            m.paragraph,
            m.speaker_canonical,
            m.speaker_bioguide,
            ml.validity_label,
            ml.prominence_label,
            ml.false_positive_type,
            ml.correct_org_id,
            ml.correct_org_name,
            ml.confidence,
            ml.labeler_notes,
            ml.labeler_name,
            ml.labeled_at
        FROM mentions m
        JOIN mention_labels ml ON m.mention_id = ml.mention_id
        ORDER BY ml.labeled_at DESC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def main():
    """Main export page."""
    configure_page(title="Export Data", icon="📤")
    apply_custom_css()

    st.title("📤 Export Training Data")

    if not check_database_exists():
        st.error("Database not found. Please initialize first.")
        return

    # Load progress for stats
    progress = load_progress()

    # Stats overview
    st.header("Labeling Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Labeled", f"{progress['labeled_mentions']:,}")

    with col2:
        true_count = progress.get('label_distribution', {}).get('true_mention', 0)
        st.metric("True Mentions", f"{true_count:,}")

    with col3:
        fp_count = progress.get('label_distribution', {}).get('false_positive', 0)
        st.metric("False Positives", f"{fp_count:,}")

    with col4:
        ambiguous = progress.get('label_distribution', {}).get('ambiguous', 0)
        needs_review = progress.get('label_distribution', {}).get('needs_review', 0)
        st.metric("Ambiguous/Review", f"{ambiguous + needs_review:,}")

    st.divider()

    # Export options
    st.header("Export Options")

    # Tabs for different export types
    tab1, tab2, tab3 = st.tabs([
        "Classifier Training Data",
        "Full Labeled Data",
        "False Positives Only"
    ])

    with tab1:
        st.subheader("Export for Text Classifier")
        st.markdown("""
        Export data in the format expected by `text_classifier.py`.
        This format matches `combined_labeled.csv` with columns:
        - `org_id`, `p1_original`, `prominence`
        - `paragraph_mention_count`, `10_or_more_org_mentioned`
        - `uuid_mention`, `uuid_paragraph`, `granuleId`
        - `source`, `interest_group`, `variation`
        """)

        col1, col2 = st.columns(2)

        with col1:
            include_true_only = st.checkbox(
                "Include only true mentions",
                value=True,
                help="Exclude false positives and ambiguous labels"
            )

        with col2:
            include_prominent_only = st.checkbox(
                "Include only prominent mentions",
                value=False,
                help="Exclude passing mentions"
            )

        # Preview
        st.subheader("Preview")
        try:
            preview_df = load_export_preview(
                include_true_only=include_true_only,
                include_prominent_only=include_prominent_only,
                limit=10,
            )

            if preview_df.empty:
                st.warning("No data matches your filter criteria.")
            else:
                st.dataframe(preview_df, use_container_width=True)
                st.caption(f"Showing first 10 of {len(preview_df)} rows (preview limited)")
        except Exception as e:
            st.error(f"Error loading preview: {e}")

        st.divider()

        # Export button
        if st.button("Generate Export", type="primary", key="classifier_export"):
            with st.spinner("Generating export..."):
                try:
                    export_df = generate_classifier_export(
                        include_true_only=include_true_only,
                        include_prominent_only=include_prominent_only,
                    )

                    if export_df.empty:
                        st.error("No data to export with current filters.")
                    else:
                        # Generate CSV
                        csv_buffer = io.StringIO()
                        export_df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()

                        # Download button
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"labeled_training_data_{timestamp}.csv"

                        st.download_button(
                            label=f"Download {filename}",
                            data=csv_data,
                            file_name=filename,
                            mime="text/csv",
                            type="primary",
                        )

                        st.success(f"Export ready! {len(export_df):,} rows generated.")

                        # Show stats
                        st.markdown("**Export Statistics:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Rows", len(export_df))
                        with col2:
                            if 'prominence' in export_df.columns:
                                prominent = (export_df['prominence'] == 1).sum()
                                st.metric("Prominent", prominent)
                        with col3:
                            if 'prominence' in export_df.columns:
                                passing = (export_df['prominence'] == 0).sum()
                                st.metric("Passing", passing)

                except Exception as e:
                    st.error(f"Export failed: {e}")

    with tab2:
        st.subheader("Export All Labeled Data")
        st.markdown("""
        Export complete labeling data including:
        - All validity labels (true, false positive, ambiguous, etc.)
        - Prominence labels
        - False positive types and reasons
        - Labeler notes and confidence
        - Full context (sentence and paragraph)
        """)

        if st.button("Generate Full Export", type="primary", key="full_export"):
            with st.spinner("Generating full export..."):
                try:
                    export_df = generate_full_export()

                    if export_df.empty:
                        st.error("No labeled data to export.")
                    else:
                        # Generate CSV
                        csv_buffer = io.StringIO()
                        export_df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"full_labeled_export_{timestamp}.csv"

                        st.download_button(
                            label=f"Download {filename}",
                            data=csv_data,
                            file_name=filename,
                            mime="text/csv",
                            type="primary",
                        )

                        st.success(f"Full export ready! {len(export_df):,} rows.")

                        # Label distribution
                        st.markdown("**Label Distribution in Export:**")
                        label_counts = export_df['validity_label'].value_counts()
                        st.dataframe(label_counts.reset_index().rename(
                            columns={'index': 'Label', 'validity_label': 'Count'}
                        ))

                except Exception as e:
                    st.error(f"Export failed: {e}")

    with tab3:
        st.subheader("Export False Positives Only")
        st.markdown("""
        Export only false positive cases for analysis.
        Useful for:
        - Pattern analysis
        - Blocklist generation
        - Improving matching algorithms
        """)

        if st.button("Generate FP Export", type="primary", key="fp_export"):
            with st.spinner("Generating false positive export..."):
                try:
                    conn = get_connection()
                    query = """
                        SELECT
                            m.org_id,
                            m.interest_group,
                            m.variation,
                            m.match_text,
                            m.is_acronym,
                            m.sentence,
                            m.paragraph,
                            m.granuleId,
                            m.date,
                            ml.false_positive_type,
                            ml.labeler_notes,
                            ml.labeler_name,
                            ml.labeled_at
                        FROM mentions m
                        JOIN mention_labels ml ON m.mention_id = ml.mention_id
                        WHERE ml.validity_label = 'false_positive'
                        ORDER BY m.org_id, ml.labeled_at DESC
                    """
                    export_df = pd.read_sql_query(query, conn)
                    conn.close()

                    if export_df.empty:
                        st.warning("No false positives labeled yet.")
                    else:
                        # Generate CSV
                        csv_buffer = io.StringIO()
                        export_df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"false_positives_{timestamp}.csv"

                        st.download_button(
                            label=f"Download {filename}",
                            data=csv_data,
                            file_name=filename,
                            mime="text/csv",
                            type="primary",
                        )

                        st.success(f"FP export ready! {len(export_df):,} false positives.")

                        # FP type distribution
                        st.markdown("**False Positive Types:**")
                        fp_types = export_df['false_positive_type'].value_counts()
                        st.dataframe(fp_types.reset_index().rename(
                            columns={'index': 'Type', 'false_positive_type': 'Count'}
                        ))

                except Exception as e:
                    st.error(f"Export failed: {e}")

    # Sidebar info
    with st.sidebar:
        st.header("Export Info")
        st.markdown("""
        ### Export Formats

        **Classifier Training Data**
        - Compatible with `text_classifier.py`
        - Only true mentions with prominence labels
        - Matches `combined_labeled.csv` format

        **Full Labeled Data**
        - Complete labeling information
        - All label types included
        - For backup and analysis

        **False Positives Only**
        - Just the FP cases
        - For pattern analysis
        - Blocklist generation

        ### Tips
        - Export regularly to backup your work
        - Use classifier export for model training
        - Use FP export to improve matching
        """)

        st.divider()

        # Quick stats
        st.markdown("### Current Data")
        try:
            label_dist = progress.get('label_distribution', {})
            for label, count in label_dist.items():
                st.markdown(f"- **{label.replace('_', ' ').title()}**: {count:,}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
