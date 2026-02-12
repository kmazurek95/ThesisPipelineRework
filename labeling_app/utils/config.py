"""Configuration and styling for the labeling application."""

import streamlit as st

# Color schemes
VALIDITY_COLORS = {
    "true_mention": "#2ca02c",      # Green
    "false_positive": "#d62728",    # Red
    "ambiguous": "#ff7f0e",         # Orange
    "needs_review": "#9467bd",      # Purple
    "wrong_entity": "#8c564b",      # Brown
}

PROMINENCE_COLORS = {
    "prominent": "#1f77b4",         # Blue
    "passing": "#7f7f7f",           # Gray
    "unclear": "#bcbd22",           # Yellow-green
}

FP_TYPE_COLORS = {
    "person_name": "#e377c2",       # Pink
    "location": "#17becf",          # Cyan
    "different_org": "#bcbd22",     # Yellow-green
    "partial_match": "#8c564b",     # Brown
    "procedural": "#7f7f7f",        # Gray
    "abbreviation_clash": "#9467bd", # Purple
    "historical": "#e7ba52",        # Gold
    "other": "#aec7e8",             # Light blue
}


def configure_page(title: str = "Labeling Interface", icon: str = "🏷️", layout: str = "wide"):
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded"
    )


def apply_custom_css():
    """Apply custom CSS styling to the app."""
    st.markdown("""
    <style>
    /* Main container */
    .main {
        padding-top: 1rem;
    }

    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 1rem;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
    }

    /* Mention context box */
    .mention-context {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        font-family: Georgia, serif;
        font-size: 1.1em;
        line-height: 1.6;
    }

    /* Highlighted mention */
    .mention-highlight {
        background-color: #fff3cd;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
        border: 1px solid #ffc107;
    }

    /* Label buttons */
    .stButton > button {
        width: 100%;
        margin: 2px 0;
    }

    /* True mention button */
    .true-mention-btn > button {
        background-color: #d4edda !important;
        border-color: #28a745 !important;
    }

    /* False positive button */
    .false-positive-btn > button {
        background-color: #f8d7da !important;
        border-color: #dc3545 !important;
    }

    /* Progress bar custom */
    .stProgress > div > div > div {
        background-color: #1f77b4;
    }

    /* Info box */
    .info-box {
        background-color: #e7f3ff;
        border: 1px solid #b6d4fe;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }

    /* Warning box */
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }

    /* Success box */
    .success-box {
        background-color: #d4edda;
        border: 1px solid #28a745;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }

    /* Table styling */
    .dataframe {
        font-size: 0.9em;
    }

    /* Navigation buttons */
    .nav-buttons {
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Card styling */
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }

    /* Organization name styling */
    .org-name {
        font-size: 1.2em;
        font-weight: bold;
        color: #1f77b4;
    }

    /* Match text styling */
    .match-text {
        font-family: monospace;
        background-color: #f0f0f0;
        padding: 2px 6px;
        border-radius: 3px;
    }

    /* Metadata styling */
    .metadata {
        color: #666;
        font-size: 0.9em;
    }

    /* Pattern alert */
    .pattern-alert {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }

    /* High FP rate warning */
    .high-fp-warning {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    </style>
    """, unsafe_allow_html=True)


def format_number(n: int) -> str:
    """Format a number with comma separators."""
    return f"{n:,}"


def get_validity_color(label: str) -> str:
    """Get the color for a validity label."""
    return VALIDITY_COLORS.get(label, "#999999")


def get_prominence_color(label: str) -> str:
    """Get the color for a prominence label."""
    return PROMINENCE_COLORS.get(label, "#999999")


def get_fp_type_color(fp_type: str) -> str:
    """Get the color for a false positive type."""
    return FP_TYPE_COLORS.get(fp_type, "#999999")
