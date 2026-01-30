#!/usr/bin/env python3
"""
Descriptive Analysis of Interest Group Congressional Mentions

This script generates descriptive statistics and visualizations for the
thesis on interest group prominence in congressional speech.

Output: outputs/figures/ and outputs/tables/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "output"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all multi-level datasets."""
    level1 = pd.read_csv(DATA_DIR / "level1.csv", low_memory=False)
    level2 = pd.read_csv(DATA_DIR / "level2_org.csv")
    level3 = pd.read_csv(DATA_DIR / "level3_politician.csv")
    level4 = pd.read_csv(DATA_DIR / "level4_policy.csv")

    # Parse dates
    level1['date'] = pd.to_datetime(level1['date'])
    level1['month'] = level1['date'].dt.to_period('M')

    logger.info(f"Loaded Level 1: {len(level1):,} rows")
    logger.info(f"Loaded Level 2: {len(level2):,} organizations")
    logger.info(f"Loaded Level 3: {len(level3):,} politicians")
    logger.info(f"Loaded Level 4: {len(level4):,} policy areas")

    return level1, level2, level3, level4


# =============================================================================
# Table 1: Summary Statistics
# =============================================================================

def generate_summary_table(level1: pd.DataFrame, level2: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics table."""

    # Mention-level stats
    mention_stats = {
        'Total Mentions': len(level1),
        'High Prominence (%)': 100 * level1['prominence_prediction'].mean(),
        'With Speaker Attribution (%)': 100 * level1['bioGuideId'].notna().mean(),
        'With Policy Area (%)': 100 * level1['issue_area'].notna().mean(),
        'Unique Dates': level1['date'].nunique(),
    }

    # Organization-level stats
    org_stats = {
        'Unique Organizations': len(level2),
        'Mean Mentions per Org': level2['total_mentions'].mean(),
        'Median Mentions per Org': level2['total_mentions'].median(),
        'Orgs with 10+ Mentions': (level2['total_mentions'] >= 10).sum(),
    }

    # Combine into table
    summary = pd.DataFrame([
        {'Category': 'Mentions', 'Metric': k, 'Value': v}
        for k, v in mention_stats.items()
    ] + [
        {'Category': 'Organizations', 'Metric': k, 'Value': v}
        for k, v in org_stats.items()
    ])

    return summary


# =============================================================================
# Figure 1: Mentions Over Time
# =============================================================================

def plot_mentions_over_time(level1: pd.DataFrame) -> plt.Figure:
    """Plot weekly mention counts over time."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Weekly counts
    weekly = level1.groupby('year_week').agg({
        'org_id': 'count',
        'prominence_prediction': 'mean'
    }).reset_index()
    weekly.columns = ['year_week', 'mention_count', 'pct_high_prominence']
    weekly['pct_high_prominence'] *= 100

    # Parse year_week for plotting
    weekly['year'] = weekly['year_week'].str.split('_').str[0].astype(int)
    weekly['week'] = weekly['year_week'].str.split('_').str[1].astype(int)
    weekly['date'] = pd.to_datetime(weekly['year'].astype(str) + '-W' + weekly['week'].astype(str).str.zfill(2) + '-1', format='%Y-W%W-%w')
    weekly = weekly.sort_values('date')

    # Plot 1: Total mentions
    axes[0].plot(weekly['date'], weekly['mention_count'], linewidth=1.5, color='steelblue')
    axes[0].fill_between(weekly['date'], weekly['mention_count'], alpha=0.3, color='steelblue')
    axes[0].set_ylabel('Weekly Mentions')
    axes[0].set_title('Interest Group Mentions in Congressional Record (114th Congress)')

    # Plot 2: Prominence percentage
    axes[1].plot(weekly['date'], weekly['pct_high_prominence'], linewidth=1.5, color='coral')
    axes[1].axhline(y=level1['prominence_prediction'].mean() * 100, color='gray', linestyle='--', alpha=0.7, label='Mean')
    axes[1].set_ylabel('% High Prominence')
    axes[1].set_xlabel('Date')
    axes[1].legend()

    plt.tight_layout()
    return fig


# =============================================================================
# Figure 2: Organization Categories
# =============================================================================

def plot_org_categories(level1: pd.DataFrame) -> plt.Figure:
    """Plot mentions by organization category."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Category counts
    cat_counts = level1.groupby('CATEGORY').agg({
        'org_id': 'count',
        'prominence_prediction': 'mean'
    }).reset_index()
    cat_counts.columns = ['category', 'mentions', 'pct_prominent']
    cat_counts = cat_counts.sort_values('mentions', ascending=True).tail(15)

    # Clean category names
    cat_counts['category_short'] = cat_counts['category'].str.extract(r'\) (.+)$')[0]

    # Plot 1: Mention counts
    axes[0].barh(cat_counts['category_short'], cat_counts['mentions'], color='steelblue')
    axes[0].set_xlabel('Number of Mentions')
    axes[0].set_title('Mentions by Organization Category')

    # Plot 2: Prominence by category
    cat_counts_sorted = cat_counts.sort_values('pct_prominent', ascending=True)
    colors = plt.cm.RdYlGn(cat_counts_sorted['pct_prominent'])
    axes[1].barh(cat_counts_sorted['category_short'], cat_counts_sorted['pct_prominent'] * 100, color=colors)
    axes[1].axvline(x=level1['prominence_prediction'].mean() * 100, color='black', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('% High Prominence')
    axes[1].set_title('Prominence Rate by Category')

    plt.tight_layout()
    return fig


# =============================================================================
# Figure 3: Lobbying vs Prominence
# =============================================================================

def plot_lobbying_prominence(level2: pd.DataFrame) -> plt.Figure:
    """Plot lobbying expenditure vs prominence."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Filter to orgs with lobbying data and multiple mentions
    # Use column names that exist in level2
    lobbying_col = 'LOBBYING11' if 'LOBBYING11' in level2.columns else 'total_lobbying_2011'
    mentions_col = 'total_mentions' if 'total_mentions' in level2.columns else 'mention_count'
    prominence_col = 'avg_prominence' if 'avg_prominence' in level2.columns else 'high_prominence_pct'

    df = level2[
        (level2[lobbying_col] > 0) &
        (level2[mentions_col] >= 5)
    ].copy()

    if len(df) == 0:
        logger.warning("No organizations with lobbying data and 5+ mentions")
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        return fig

    # Log transform lobbying
    df['log_lobbying'] = np.log10(df[lobbying_col] + 1)

    # Scatter plot
    scatter = ax.scatter(
        df['log_lobbying'],
        df[prominence_col] * 100,
        s=df[mentions_col] * 2,  # Size by mentions
        alpha=0.6,
        c=df[mentions_col],
        cmap='viridis'
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Mention Count')

    # Trend line
    z = np.polyfit(df['log_lobbying'], df[prominence_col] * 100, 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['log_lobbying'].min(), df['log_lobbying'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.8, label=f'Trend (slope={z[0]:.2f})')

    ax.set_xlabel('Log10(Lobbying Expenditure 2011)')
    ax.set_ylabel('% High Prominence Mentions')
    ax.set_title('Lobbying Investment vs Congressional Prominence')
    ax.legend()

    # Add correlation
    corr = df['log_lobbying'].corr(df[prominence_col])
    ax.annotate(f'r = {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=12, verticalalignment='top')

    return fig


# =============================================================================
# Figure 4: Party Patterns
# =============================================================================

def plot_party_patterns(level1: pd.DataFrame) -> plt.Figure:
    """Plot mention patterns by party."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Filter to mentions with party info
    df = level1[level1['party'].notna()].copy()

    # Party mention counts by category
    party_cat = df.groupby(['party', 'CATEGORY']).size().unstack(fill_value=0)

    # Top categories
    top_cats = df['CATEGORY'].value_counts().head(8).index
    party_cat_top = party_cat[top_cats].T

    # Clean names
    party_cat_top.index = party_cat_top.index.str.extract(r'\) (.+)$')[0].values

    # Plot 1: Stacked bar by party
    party_cat_top[['D', 'R']].plot(kind='barh', stacked=True, ax=axes[0],
                                    color=['steelblue', 'coral'])
    axes[0].set_xlabel('Number of Mentions')
    axes[0].set_title('Organization Categories by Party')
    axes[0].legend(title='Party')

    # Plot 2: Prominence by party and category
    party_prom = df.groupby(['party', 'CATEGORY'])['prominence_prediction'].mean().unstack(fill_value=0)
    party_prom_top = party_prom[top_cats].T
    party_prom_top.index = party_prom_top.index.str.extract(r'\) (.+)$')[0].values

    x = np.arange(len(party_prom_top))
    width = 0.35

    axes[1].bar(x - width/2, party_prom_top['D'] * 100, width, label='Democrats', color='steelblue')
    axes[1].bar(x + width/2, party_prom_top['R'] * 100, width, label='Republicans', color='coral')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(party_prom_top.index, rotation=45, ha='right')
    axes[1].set_ylabel('% High Prominence')
    axes[1].set_title('Prominence by Party and Category')
    axes[1].legend()

    plt.tight_layout()
    return fig


# =============================================================================
# Figure 5: Policy Area Heatmap
# =============================================================================

def plot_policy_heatmap(level1: pd.DataFrame) -> plt.Figure:
    """Plot heatmap of organization categories by policy area."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Filter to mentions with policy area
    df = level1[level1['issue_area_name'].notna()].copy()

    # Cross-tabulation
    cross_tab = pd.crosstab(
        df['CATEGORY'].str.extract(r'\) (.+)$')[0],
        df['issue_area_name'],
        normalize='columns'  # Normalize within policy area
    ) * 100

    # Top categories and policy areas
    top_cats = df['CATEGORY'].value_counts().head(12).index
    top_cats_clean = [c.split(') ')[1] if ') ' in c else c for c in top_cats]
    cross_tab = cross_tab.loc[cross_tab.index.isin(top_cats_clean)]

    # Plot heatmap
    sns.heatmap(cross_tab, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': '% of Policy Area Mentions'})
    ax.set_xlabel('Policy Area')
    ax.set_ylabel('Organization Category')
    ax.set_title('Organization Types by Policy Domain')

    plt.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate descriptive analysis")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()

    # Create output directories
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    level1, level2, level3, level4 = load_data()

    # Generate tables
    if not args.figures_only:
        logger.info("Generating summary table...")
        summary = generate_summary_table(level1, level2)
        summary.to_csv(TABLES_DIR / "table1_summary_stats.csv", index=False)
        print("\n=== Summary Statistics ===")
        print(summary.to_string(index=False))

    # Generate figures
    logger.info("Generating Figure 1: Mentions over time...")
    fig1 = plot_mentions_over_time(level1)
    fig1.savefig(FIGURES_DIR / "fig1_mentions_over_time.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)

    logger.info("Generating Figure 2: Organization categories...")
    fig2 = plot_org_categories(level1)
    fig2.savefig(FIGURES_DIR / "fig2_org_categories.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)

    logger.info("Generating Figure 3: Lobbying vs prominence...")
    fig3 = plot_lobbying_prominence(level2)
    fig3.savefig(FIGURES_DIR / "fig3_lobbying_prominence.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)

    logger.info("Generating Figure 4: Party patterns...")
    fig4 = plot_party_patterns(level1)
    fig4.savefig(FIGURES_DIR / "fig4_party_patterns.png", dpi=150, bbox_inches='tight')
    plt.close(fig4)

    logger.info("Generating Figure 5: Policy heatmap...")
    fig5 = plot_policy_heatmap(level1)
    fig5.savefig(FIGURES_DIR / "fig5_policy_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close(fig5)

    logger.info(f"\nOutputs saved to:")
    logger.info(f"  Figures: {FIGURES_DIR}")
    logger.info(f"  Tables:  {TABLES_DIR}")


if __name__ == "__main__":
    main()
