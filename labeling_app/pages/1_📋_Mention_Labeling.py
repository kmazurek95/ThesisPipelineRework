"""Core mention labeling workflow page."""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import configure_page, apply_custom_css, VALIDITY_COLORS
from utils.data_loader import (
    load_mentions,
    load_mention_by_id,
    load_organizations,
    load_progress,
    check_database_exists,
)
from utils.database import save_mention_label
from utils.text_display import highlight_mention, format_context_html


def init_session_state():
    """Initialize session state variables."""
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'mentions_df' not in st.session_state:
        st.session_state.mentions_df = None
    if 'labeler_name' not in st.session_state:
        st.session_state.labeler_name = ""
    if 'filters_applied' not in st.session_state:
        st.session_state.filters_applied = False
    if 'last_label_saved' not in st.session_state:
        st.session_state.last_label_saved = None


def load_filtered_mentions(
    org_id=None,
    match_type=None,
    date_from=None,
    date_to=None,
    limit=100,
):
    """Load mentions with applied filters."""
    return load_mentions(
        limit=limit,
        org_id=org_id,
        match_type=match_type,
        date_from=date_from,
        date_to=date_to,
        labeled=False,
    )


def display_mention_card(mention: dict):
    """Display a mention with its context."""
    # Organization info
    st.markdown(f"""
    <div class="card">
        <div class="org-name">{mention.get('interest_group', 'Unknown Organization')}</div>
        <div class="metadata">
            <strong>Org ID:</strong> {mention.get('org_id', 'N/A')} |
            <strong>Match:</strong> <span class="match-text">{mention.get('match_text') or mention.get('variation', 'N/A')}</span> |
            <strong>Type:</strong> {'Acronym' if mention.get('is_acronym') else 'Full Name'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Document info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Date:** {mention.get('date', 'N/A')}")
    with col2:
        st.markdown(f"**Granule:** {mention.get('granuleId', 'N/A')[:30]}...")
    with col3:
        speaker = mention.get('speaker_canonical') or mention.get('speaker_raw') or 'Unknown'
        st.markdown(f"**Speaker:** {speaker}")

    # Context display
    st.subheader("Context")

    paragraph = mention.get('paragraph', '')
    sentence = mention.get('sentence', '')
    match_text = mention.get('match_text') or mention.get('variation', '')
    start = mention.get('mention_char_start')
    end = mention.get('mention_char_end')

    # Show sentence with highlighting
    if sentence:
        highlighted = highlight_mention(sentence, match_text, start, end)
        st.markdown(f"""
        <div class="mention-context">
            <p>{highlighted}</p>
        </div>
        """, unsafe_allow_html=True)

    # Expandable full paragraph
    if paragraph and paragraph != sentence:
        with st.expander("View Full Paragraph"):
            highlighted_para = highlight_mention(paragraph, match_text)
            st.markdown(f"""
            <div class="mention-context" style="font-size: 0.95em;">
                {highlighted_para}
            </div>
            """, unsafe_allow_html=True)

    # Speaker confidence warning
    speaker_conf = mention.get('speaker_confidence')
    if speaker_conf is not None and speaker_conf < 0.7:
        st.warning(f"Low speaker confidence: {speaker_conf:.0%}")


def display_labeling_form(mention: dict, labeler_name: str):
    """Display the labeling form and handle submission."""
    st.subheader("Label This Mention")

    # Main validity label
    validity_options = {
        'true_mention': 'True Mention - Correctly identifies the interest group',
        'false_positive': 'False Positive - Does NOT refer to this interest group',
        'ambiguous': 'Ambiguous - Cannot determine with certainty',
        'needs_review': 'Needs Review - Requires additional context or expertise',
        'wrong_entity': 'Wrong Entity - Refers to a different organization',
    }

    # Quick label buttons in a row
    st.markdown("**Quick Labels:**")
    col1, col2, col3, col4 = st.columns(4)

    validity_label = None

    with col1:
        if st.button("✅ True", use_container_width=True, type="primary"):
            validity_label = 'true_mention'

    with col2:
        if st.button("❌ False Positive", use_container_width=True):
            validity_label = 'false_positive'

    with col3:
        if st.button("❓ Ambiguous", use_container_width=True):
            validity_label = 'ambiguous'

    with col4:
        if st.button("🚩 Needs Review", use_container_width=True):
            validity_label = 'needs_review'

    st.divider()

    # Detailed form with expander
    with st.expander("Detailed Label Options", expanded=validity_label is None):
        # Validity selection
        selected_validity = st.radio(
            "Validity Label",
            options=list(validity_options.keys()),
            format_func=lambda x: validity_options[x],
            horizontal=False,
            key="validity_radio",
        )

        # Conditional fields based on validity
        prominence_label = None
        false_positive_type = None
        correct_org_id = None
        correct_org_name = None

        if selected_validity == 'true_mention':
            st.markdown("**Prominence:**")
            prominence_label = st.radio(
                "How prominent is this mention?",
                options=['prominent', 'passing', 'unclear'],
                format_func=lambda x: {
                    'prominent': 'Prominent - Main subject of discussion',
                    'passing': 'Passing - Brief or tangential mention',
                    'unclear': 'Unclear - Cannot determine prominence',
                }[x],
                horizontal=True,
                key="prominence_radio",
            )

        elif selected_validity == 'false_positive':
            st.markdown("**False Positive Type:**")
            false_positive_type = st.selectbox(
                "Why is this a false positive?",
                options=[
                    'person_name',
                    'location',
                    'different_org',
                    'partial_match',
                    'procedural',
                    'abbreviation_clash',
                    'historical',
                    'other',
                ],
                format_func=lambda x: {
                    'person_name': "Person's name matches organization",
                    'location': 'Geographic location match',
                    'different_org': 'Different organization with similar name',
                    'partial_match': 'Partial text match (substring)',
                    'procedural': 'Procedural/legislative text',
                    'abbreviation_clash': 'Acronym refers to something else',
                    'historical': 'Historical reference to defunct org',
                    'other': 'Other reason',
                }[x],
                key="fp_type_select",
            )

        elif selected_validity == 'wrong_entity':
            st.markdown("**Correct Organization:**")
            correct_org_name = st.text_input(
                "What organization does this actually refer to?",
                key="correct_org_input",
            )

        # Confidence level
        confidence = st.select_slider(
            "Confidence in this label",
            options=['low', 'medium', 'high'],
            value='high',
            key="confidence_slider",
        )

        # Notes
        notes = st.text_area(
            "Notes (optional)",
            placeholder="Add any relevant notes about this labeling decision...",
            key="notes_area",
        )

        # Submit button
        if st.button("Submit Label", type="primary", use_container_width=True):
            validity_label = selected_validity

    # Handle submission (either quick button or form)
    if validity_label:
        # Get form values if using detailed form
        if 'validity_radio' in st.session_state:
            selected_validity = st.session_state.validity_radio
            if validity_label in ['true_mention', 'false_positive', 'ambiguous', 'needs_review']:
                # Quick button was used, get prominence/fp_type from form if available
                prominence_label = st.session_state.get('prominence_radio') if validity_label == 'true_mention' else None
                false_positive_type = st.session_state.get('fp_type_select') if validity_label == 'false_positive' else None
        else:
            prominence_label = None
            false_positive_type = None

        confidence = st.session_state.get('confidence_slider', 'high')
        notes = st.session_state.get('notes_area', '')
        correct_org_name = st.session_state.get('correct_org_input', '')

        success = save_mention_label(
            mention_id=mention['mention_id'],
            validity_label=validity_label,
            labeler_name=labeler_name,
            prominence_label=prominence_label,
            false_positive_type=false_positive_type,
            correct_org_name=correct_org_name if correct_org_name else None,
            confidence=confidence,
            labeler_notes=notes if notes else None,
        )

        if success:
            st.session_state.last_label_saved = validity_label
            st.success(f"Label saved: {validity_label.replace('_', ' ').title()}")
            # Move to next mention
            st.session_state.current_index += 1
            # Clear caches to refresh data
            st.cache_data.clear()
            st.rerun()
        else:
            st.error("Failed to save label. Please try again.")


def main():
    """Main labeling page."""
    configure_page(title="Mention Labeling", icon="📋")
    apply_custom_css()

    init_session_state()

    st.title("📋 Mention Labeling")

    # Check database
    if not check_database_exists():
        st.error("Database not found. Please initialize first.")
        return

    # Sidebar filters
    with st.sidebar:
        st.header("Filters")

        # Labeler name (required)
        labeler_name = st.text_input(
            "Your Name (required)",
            value=st.session_state.labeler_name,
            placeholder="Enter your name",
        )
        if labeler_name:
            st.session_state.labeler_name = labeler_name

        st.divider()

        # Organization filter
        try:
            orgs_df = load_organizations()
            org_options = ['All Organizations'] + orgs_df['org_id'].tolist()
            selected_org = st.selectbox("Organization", org_options)
            org_filter = None if selected_org == 'All Organizations' else selected_org
        except Exception:
            org_filter = None
            st.warning("Could not load organizations")

        # Match type filter
        match_type = st.radio(
            "Match Type",
            options=['all', 'acronym', 'name'],
            format_func=lambda x: {'all': 'All', 'acronym': 'Acronyms Only', 'name': 'Full Names Only'}[x],
            horizontal=True,
        )
        match_type_filter = None if match_type == 'all' else match_type

        # Date range
        st.markdown("**Date Range:**")
        col1, col2 = st.columns(2)
        with col1:
            date_from = st.date_input("From", value=None)
        with col2:
            date_to = st.date_input("To", value=None)

        date_from_str = date_from.isoformat() if date_from else None
        date_to_str = date_to.isoformat() if date_to else None

        # Batch size
        batch_size = st.slider("Batch Size", min_value=10, max_value=500, value=100, step=10)

        # Apply filters button
        if st.button("Apply Filters", use_container_width=True):
            st.session_state.current_index = 0
            st.session_state.filters_applied = True
            st.cache_data.clear()
            st.rerun()

        st.divider()

        # Progress display
        try:
            progress = load_progress()
            st.metric("Total Labeled", f"{progress['labeled_mentions']:,}")
            st.metric("Remaining", f"{progress['unlabeled_mentions']:,}")
            st.progress(progress['pct_complete'] / 100 if progress['pct_complete'] else 0)
        except Exception:
            pass

    # Main content
    if not labeler_name:
        st.warning("Please enter your name in the sidebar to start labeling.")
        return

    # Load mentions
    try:
        mentions_df = load_filtered_mentions(
            org_id=org_filter if 'org_filter' in dir() else None,
            match_type=match_type_filter if 'match_type_filter' in dir() else None,
            date_from=date_from_str if 'date_from_str' in dir() else None,
            date_to=date_to_str if 'date_to_str' in dir() else None,
            limit=batch_size if 'batch_size' in dir() else 100,
        )
    except Exception as e:
        st.error(f"Error loading mentions: {e}")
        return

    if mentions_df.empty:
        st.success("🎉 No unlabeled mentions match your filters! Try different filters or you may be done.")
        return

    # Ensure index is valid
    if st.session_state.current_index >= len(mentions_df):
        st.session_state.current_index = 0
        st.cache_data.clear()
        st.rerun()

    # Navigation
    total_in_batch = len(mentions_df)
    current = st.session_state.current_index

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("◀ Previous", disabled=current == 0):
            st.session_state.current_index -= 1
            st.rerun()

    with col2:
        st.markdown(f"<h3 style='text-align: center;'>Mention {current + 1} of {total_in_batch}</h3>", unsafe_allow_html=True)

    with col3:
        if st.button("Skip ▶", disabled=current >= total_in_batch - 1):
            st.session_state.current_index += 1
            st.rerun()

    st.divider()

    # Get current mention
    mention_row = mentions_df.iloc[current]
    mention = mention_row.to_dict()

    # Display mention card
    display_mention_card(mention)

    st.divider()

    # Display labeling form
    display_labeling_form(mention, labeler_name)

    # Show last saved label
    if st.session_state.last_label_saved:
        st.info(f"Last label: {st.session_state.last_label_saved.replace('_', ' ').title()}")


if __name__ == "__main__":
    main()
