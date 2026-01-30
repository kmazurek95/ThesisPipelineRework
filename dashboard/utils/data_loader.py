"""
Data loading utilities with caching for Streamlit dashboard
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "output"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"


@st.cache_data(ttl=3600)
def load_level1_data() -> pd.DataFrame:
    """Load Level 1 (mention-level) data with caching"""
    try:
        df = pd.read_csv(DATA_DIR / "level1.csv", low_memory=False)
        
        # Parse dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.to_period('M').astype(str)
        
        logger.info(f"Loaded Level 1: {len(df):,} mentions")
        return df
    except Exception as e:
        logger.error(f"Error loading Level 1 data: {e}")
        st.error(f"Error loading mention-level data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_level2_data() -> pd.DataFrame:
    """Load Level 2 (organization-level) data with caching"""
    try:
        df = pd.read_csv(DATA_DIR / "level2_org.csv")
        logger.info(f"Loaded Level 2: {len(df):,} organizations")
        return df
    except Exception as e:
        logger.error(f"Error loading Level 2 data: {e}")
        st.error(f"Error loading organization-level data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_level3_data() -> pd.DataFrame:
    """Load Level 3 (politician-level) data with caching"""
    try:
        df = pd.read_csv(DATA_DIR / "level3_politician.csv")
        logger.info(f"Loaded Level 3: {len(df):,} politicians")
        return df
    except Exception as e:
        logger.error(f"Error loading Level 3 data: {e}")
        st.error(f"Error loading politician-level data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_level4_data() -> pd.DataFrame:
    """Load Level 4 (policy area-level) data with caching"""
    try:
        df = pd.read_csv(DATA_DIR / "level4_policy.csv")
        logger.info(f"Loaded Level 4: {len(df):,} policy areas")
        return df
    except Exception as e:
        logger.error(f"Error loading Level 4 data: {e}")
        st.error(f"Error loading policy-level data: {e}")
        return pd.DataFrame()


@st.cache_data
def load_all_data() -> dict:
    """Load all levels of data at once"""
    return {
        'level1': load_level1_data(),
        'level2': load_level2_data(),
        'level3': load_level3_data(),
        'level4': load_level4_data(),
    }


def get_summary_stats() -> dict:
    """Calculate summary statistics across all levels"""
    data = load_all_data()
    
    level1 = data['level1']
    level2 = data['level2']
    level3 = data['level3']
    
    stats = {
        'total_mentions': len(level1),
        'total_orgs': len(level2),
        'total_politicians': len(level3),
        'prominent_rate': 0,
        'avg_prominence_score': 0,
    }
    
    if not level1.empty and 'prominence_prediction' in level1.columns:
        stats['prominent_rate'] = (level1['prominence_prediction'] == 1).mean() * 100
    
    if not level1.empty and 'prominence_score' in level1.columns:
        stats['avg_prominence_score'] = level1['prominence_score'].mean()
    
    return stats


def load_summary_stats():
    """Wrapper for cached summary stats"""
    return get_summary_stats()


def get_figure_path(figure_name: str) -> Path:
    """Get path to a figure file"""
    return FIGURES_DIR / figure_name


def check_data_availability() -> dict:
    """Check which data files are available"""
    availability = {
        'level1': (DATA_DIR / "level1.csv").exists(),
        'level2': (DATA_DIR / "level2_org.csv").exists(),
        'level3': (DATA_DIR / "level3_politician.csv").exists(),
        'level4': (DATA_DIR / "level4_policy.csv").exists(),
    }
    return availability


@st.cache_data(ttl=3600)
def load_regression_results() -> pd.DataFrame:
    """Load regression results from outputs/tables"""
    try:
        df = pd.read_csv(TABLES_DIR / "regression_results.csv")
        logger.info(f"Loaded regression results: {len(df)} coefficients")
        return df
    except Exception as e:
        logger.error(f"Error loading regression results: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_model_fit_stats() -> pd.DataFrame:
    """Load model fit statistics from outputs/tables"""
    try:
        df = pd.read_csv(TABLES_DIR / "model_fit_statistics.csv")
        logger.info(f"Loaded model fit stats: {len(df)} models")
        return df
    except Exception as e:
        logger.error(f"Error loading model fit stats: {e}")
        return pd.DataFrame()
