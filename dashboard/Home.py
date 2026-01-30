"""
Interest Group Prominence in Congressional Speech - Dashboard
Main landing page with key findings and project overview
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils.data_loader import load_summary_stats
from dashboard.utils.config import configure_page, apply_custom_css

# Page configuration
configure_page(
    title="Interest Group Prominence Dashboard",
    icon="🏛️",
    layout="wide"
)

apply_custom_css()

# Hero Section
st.markdown("""
<div class="hero">
    <h1>🏛️ Interest Group Prominence in Congressional Speech</h1>
    <p class="subtitle">
        Analyzing 25,000+ mentions across the 114th U.S. Congress (2015-2017)
    </p>
</div>
""", unsafe_allow_html=True)

# Key Findings Banner
st.markdown("---")
st.header("🔑 Key Findings")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Mentions Analyzed",
        value="25,000+",
        delta="91% Classification Accuracy"
    )

with col2:
    st.metric(
        label="Lobbying Impact",
        value="+7.1%",
        delta="per log unit increase",
        help="Lobbying expenditure predicts higher prominence (p < 0.001)"
    )

with col3:
    st.metric(
        label="Senate Effect",
        value="+45%",
        delta="vs. House",
        help="Senators give more prominent mentions than Representatives (exp(0.37) = 1.45)"
    )

with col4:
    st.metric(
        label="Partisan Gap",
        value="-23%",
        delta="Democrats vs Republicans",
        help="Democrats give fewer prominent mentions (exp(-0.26) = 0.77)"
    )

# Project Overview
st.markdown("---")
col_left, col_right = st.columns([3, 2])

with col_left:
    st.header("📖 About This Project")
    st.markdown("""
    This dashboard showcases a **complete data science pipeline** for analyzing how interest groups 
    are mentioned in U.S. Congressional floor speeches. 
    
    ### Research Questions
    - **Which organizations** receive prominent vs. passing mentions?
    - **What predicts prominence?** Lobbying, organization type, speaker characteristics?
    - **Partisan patterns:** Do Democrats and Republicans mention groups differently?
    
    ### Technical Highlights
    - ✅ **ETL Pipeline:** Modular stages for collection, processing, classification
    - ✅ **ML Classification:** TF-IDF + Logistic Regression (F1 = 0.91)
    - ✅ **Multi-level Data:** Hierarchical structure for nested analysis
    - ✅ **API Integration:** GovInfo, Congress.gov
    - ✅ **Production-Ready:** Validated, tested, documented codebase
    
    ### Data Sources
    - **Congressional Record:** GovInfo API (25,000+ parsed speeches)
    - **Interest Group Data:** Washington Representatives Study 2011
    - **Legislator Info:** Congress.gov API
    - **Lobbying Data:** Center for Responsive Politics
    """)

with col_right:
    st.header("🎯 Quick Navigation")
    st.markdown("""
    ### Explore the Dashboard
    
    **📊 Overview**  
    Executive summary, methodology, key statistics
    
    **🔍 Explore Data**  
    Interactive filters, search, dynamic visualizations
    
    **📈 Statistical Models**  
    Regression results, model comparison, diagnostics
    
    **🏛️ Organizations**  
    Organization-level analysis, lobbying patterns
    
    ---
    
    ### Project Links
    - 📦 [GitHub Repository](https://github.com/kmazurek95/ThesisPipelineRework)
    - 📄 [Original Thesis](https://github.com/kmazurek95/MastersThesis_InterestGroupAnalysis)
    - 👔 [LinkedIn](#)
    
    ---
    
    ### Master's Thesis Revamp
    This is a **complete rewrite** of my Master's thesis pipeline, 
    transforming research scripts into production-ready Python code with:
    - Automated ML classification
    - Comprehensive testing & validation
    - Professional documentation
    - Reproducible workflows
    """)

# Key Findings Table
st.markdown("---")
st.header("📊 Summary of Findings")

findings_data = {
    "Finding": [
        "Lobbying predicts prominence",
        "Senators > Representatives",
        "Democrats give fewer prominent mentions",
        "Single-issue groups get noticed",
        "Labor unions receive substantive attention"
    ],
    "Evidence": [
        "+7.4% higher odds per log unit increase (p < 0.001)",
        "+45% higher odds of prominent mentions (p < 0.001)",
        "-23% compared to Republicans (p < 0.001)",
        "+41% higher prominence rate (p < 0.001)",
        "+15% higher prominence rate (p < 0.01)"
    ],
    "Level": [
        "Level 1 (Mentions)",
        "Level 1 (Mentions)",
        "Level 1 (Mentions)",
        "Level 1 (Mentions)",
        "Level 1 (Mentions)"
    ]
}

st.table(findings_data)

# Data Pipeline Flowchart
st.markdown("---")
st.header("🔄 Data Pipeline Architecture")

st.markdown("""
```mermaid
graph LR
    A[Congressional Record<br/>GovInfo API] --> B[Normalize & Parse]
    C[Congress.gov APIs<br/>Bills, Members] --> B
    B --> D[Extract Mentions]
    D --> E[Attribute Speakers]
    E --> F[ML Classification<br/>F1=0.91]
    F --> G[Multi-level Integration]
    H[Interest Group Data<br/>WRS 2011] --> G
    G --> I[Level 1: Mentions]
    G --> J[Level 2: Organizations]
    G --> K[Level 3: Politicians]
    G --> L[Level 4: Policy Areas]
    I --> M[Statistical Analysis]
    J --> M
    K --> M
    L --> M
    M --> N[Visualizations & Reports]
```
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Built with Streamlit • Python 3.10+ • Data Science Pipeline</p>
    <p>© 2026 • MIT License • For Academic & Professional Use</p>
</div>
""", unsafe_allow_html=True)
