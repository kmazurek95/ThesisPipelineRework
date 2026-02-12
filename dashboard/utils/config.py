"""
Configuration and styling utilities for Streamlit dashboard
"""

import streamlit as st


def configure_page(title: str, icon: str = "📊", layout: str = "wide"):
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded"
    )


def apply_custom_css():
    """Apply custom CSS styling to the dashboard"""
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --background-color: #ffffff;
        --text-color: #262730;
    }
    
    /* Hero section */
    .hero {
        padding: 2rem 0;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .hero h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Better spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Table styling */
    table {
        width: 100%;
    }
    
    /* Filter section */
    .filter-section {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Statistic boxes */
    .stat-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .stat-box h3 {
        margin-top: 0;
        color: #667eea;
    }
    
    /* Code blocks */
    code {
        background-color: #f0f2f6;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 5px;
        border: 2px solid #667eea;
        color: #667eea;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #667eea;
        color: white;
    }
    
    /* Download buttons */
    .stDownloadButton>button {
        background-color: #28a745;
        color: white;
        border: none;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
    }
    
    /* Hide streamlit branding in footer */
    footer {
        visibility: hidden;
    }
    
    /* Custom info boxes */
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


# Color schemes for consistent visualizations
PARTY_COLORS = {
    'D': '#1f77b4',  # Democratic Blue
    'R': '#d62728',  # Republican Red
    'I': '#9467bd',  # Independent Purple
}

CATEGORY_COLORS = {
    'Business': '#1f77b4',
    'Labor': '#ff7f0e',
    'Public Interest': '#2ca02c',
    'Professional': '#d62728',
    'Government': '#9467bd',
    'Other': '#8c564b'
}

PROMINENCE_COLORS = {
    'Prominent': '#2ca02c',
    'Passing': '#d62728',
}

HERO_ORG_COLORS = {
    'AARP': '#1f77b4',
    'AFL-CIO': '#ff7f0e',
    'NAM': '#2ca02c',
    'ACLU': '#d62728',
    'AMA': '#9467bd',
}

HERO_ORG_META = {
    12:   {'name': 'AARP',    'sector': 'Elderly Advocacy',       'lobbying_k': 37_848},
    59:   {'name': 'AFL-CIO', 'sector': 'Labor',                  'lobbying_k': 6_280},
    238:  {'name': 'ACLU',    'sector': 'Civil Liberties',         'lobbying_k': 2_579},
    2215: {'name': 'NAM',     'sector': 'Business/Manufacturing',  'lobbying_k': 16_810},
    391:  {'name': 'AMA',     'sector': 'Healthcare/Professional', 'lobbying_k': 44_055},
}
