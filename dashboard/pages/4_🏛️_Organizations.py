"""
🏛️ Organizations - Organization-Level Analysis
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils.config import configure_page, apply_custom_css, CATEGORY_COLORS
from dashboard.utils.data_loader import load_all_data, get_figure_path
from dashboard.utils.visualizations import create_scatter_plot, create_bar_chart

# Page configuration
configure_page(title="Organizations - Interest Group Prominence", icon="🏛️")
apply_custom_css()

st.title("🏛️ Organization-Level Analysis")
st.markdown("Explore patterns in how different organizations are mentioned in Congress")

# Load data
with st.spinner("Loading organization data..."):
    data = load_all_data()
    level2 = data['level2']
    level1 = data['level1']

if level2.empty:
    st.error("Organization-level data not available")
    st.stop()

# Summary stats
st.header("📊 Organization Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Organizations", f"{len(level2):,}")

with col2:
    if 'total_mentions' in level2.columns:
        avg_mentions = level2['total_mentions'].mean()
        st.metric("Avg Mentions per Org", f"{avg_mentions:.1f}")
    else:
        st.metric("Avg Mentions per Org", "N/A")

with col3:
    if 'prominence_rate' in level2.columns:
        avg_prom = level2['prominence_rate'].mean()
        st.metric("Avg Prominence Rate", f"{avg_prom:.1f}%")
    else:
        st.metric("Avg Prominence Rate", "N/A")

with col4:
    if 'category' in level2.columns:
        n_categories = level2['category'].nunique()
        st.metric("Organization Types", f"{n_categories}")
    else:
        st.metric("Organization Types", "N/A")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Rankings",
    "💰 Lobbying Analysis",
    "📊 Categories",
    "🔍 Detailed View"
])

with tab1:
    st.header("Organization Rankings")
    
    # Ranking selector
    ranking_metric = st.selectbox(
        "Rank by:",
        ['Total Mentions', 'Prominence Rate', 'Prominent Mentions Count']
    )
    
    top_n_rank = st.slider("Number of organizations to display", 10, 50, 20, key='rank_slider')
    
    # Prepare ranking data
    rank_df = level2.copy()
    
    if ranking_metric == 'Total Mentions' and 'total_mentions' in rank_df.columns:
        rank_df = rank_df.nlargest(top_n_rank, 'total_mentions')
        y_col = 'total_mentions'
        title = f'Top {top_n_rank} Most Mentioned Organizations'
    elif ranking_metric == 'Prominence Rate' and 'prominence_rate' in rank_df.columns:
        # Filter for orgs with minimum mentions for reliability
        if 'total_mentions' in rank_df.columns:
            rank_df = rank_df[rank_df['total_mentions'] >= 5]
        rank_df = rank_df.nlargest(top_n_rank, 'prominence_rate')
        y_col = 'prominence_rate'
        title = f'Top {top_n_rank} Organizations by Prominence Rate (min. 5 mentions)'
    else:  # Prominent Mentions Count
        if 'num_prominent' in rank_df.columns:
            rank_df = rank_df.nlargest(top_n_rank, 'num_prominent')
            y_col = 'num_prominent'
        elif 'total_mentions' in rank_df.columns and 'prominence_rate' in rank_df.columns:
            rank_df['num_prominent'] = rank_df['total_mentions'] * rank_df['prominence_rate'] / 100
            rank_df = rank_df.nlargest(top_n_rank, 'num_prominent')
            y_col = 'num_prominent'
        else:
            st.warning("Data for prominent mentions count not available")
            rank_df = pd.DataFrame()
            y_col = None
        title = f'Top {top_n_rank} Organizations by Prominent Mentions'
    
    if not rank_df.empty and y_col and 'org_name' in rank_df.columns:
        # Create bar chart
        color_col = 'category' if 'category' in rank_df.columns else None
        
        fig = create_bar_chart(
            rank_df,
            x_col='org_name',
            y_col=y_col,
            title=title,
            orientation='h',
            color_col=color_col
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        with st.expander("📋 View Data Table"):
            display_cols = ['org_name', 'total_mentions', 'prominence_rate', 'category']
            available_cols = [col for col in display_cols if col in rank_df.columns]
            st.dataframe(rank_df[available_cols].reset_index(drop=True), use_container_width=True)
    else:
        st.warning("Required data columns not available for this ranking")

with tab2:
    st.header("Lobbying and Prominence")
    
    st.markdown("""
    ### Key Finding: Lobbying Predicts Prominence
    
    Organizations that spend more on lobbying are significantly more likely to receive 
    prominent mentions in congressional speech (+7.1% per log unit increase, p < 0.001).
    """)
    
    # Check for lobbying data
    has_lobbying = 'lobbying' in level2.columns or 'total_lobbying' in level2.columns
    lobbying_col = 'lobbying' if 'lobbying' in level2.columns else 'total_lobbying'
    
    if has_lobbying and 'prominence_rate' in level2.columns:
        # Create scatter plot
        plot_df = level2[[lobbying_col, 'prominence_rate', 'org_name']].dropna()
        
        if not plot_df.empty:
            # Filter out zeros for log scale
            plot_df = plot_df[plot_df[lobbying_col] > 0]
            
            fig = px.scatter(
                plot_df,
                x=lobbying_col,
                y='prominence_rate',
                hover_data=['org_name'],
                log_x=True,
                labels={
                    lobbying_col: 'Lobbying Expenditure ($)',
                    'prominence_rate': 'Prominence Rate (%)',
                    'org_name': 'Organization'
                },
                title='Lobbying Expenditure vs. Prominence Rate',
                trendline='ols'
            )
            
            fig.update_layout(plot_bgcolor='white', hovermode='closest')
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation statistics
            corr = plot_df[lobbying_col].corr(plot_df['prominence_rate'])
            st.metric("Correlation Coefficient", f"{corr:.3f}")
            
            # Binned analysis
            st.subheader("Prominence by Lobbying Quartile")
            
            plot_df['lobbying_quartile'] = pd.qcut(
                plot_df[lobbying_col],
                q=4,
                labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']
            )
            
            quartile_stats = plot_df.groupby('lobbying_quartile').agg({
                'prominence_rate': ['mean', 'count'],
                lobbying_col: 'mean'
            }).reset_index()
            
            quartile_stats.columns = ['Quartile', 'Avg Prominence Rate', 'N Orgs', 'Avg Lobbying']
            
            fig = px.bar(
                quartile_stats,
                x='Quartile',
                y='Avg Prominence Rate',
                text='Avg Prominence Rate',
                labels={'Avg Prominence Rate': 'Average Prominence Rate (%)'},
                title='Average Prominence Rate by Lobbying Quartile'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(quartile_stats, use_container_width=True, hide_index=True)
    else:
        st.warning("Lobbying expenditure data not available in Level 2 dataset")
        
        # Try to show existing figure if available
        fig_path = get_figure_path("fig3_lobbying_prominence.png")
        if fig_path.exists():
            st.image(str(fig_path), caption="Lobbying vs. Prominence", use_container_width=True)

with tab3:
    st.header("Organization Categories")
    
    if 'category' in level2.columns:
        # Category distribution
        st.subheader("Distribution of Organizations by Type")
        
        category_counts = level2['category'].value_counts().reset_index()
        category_counts.columns = ['category', 'count']
        
        fig = px.pie(
            category_counts,
            names='category',
            values='count',
            title='Organization Types',
            color='category',
            color_discrete_map=CATEGORY_COLORS
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Prominence by category
        if 'prominence_rate' in level2.columns:
            st.subheader("Prominence Rate by Organization Type")
            
            cat_prom = level2.groupby('category').agg({
                'prominence_rate': 'mean',
                'org_name': 'count'
            }).reset_index()
            cat_prom.columns = ['category', 'avg_prominence', 'count']
            cat_prom = cat_prom.sort_values('avg_prominence', ascending=False)
            
            fig = px.bar(
                cat_prom,
                x='category',
                y='avg_prominence',
                text='avg_prominence',
                color='category',
                color_discrete_map=CATEGORY_COLORS,
                labels={
                    'category': 'Organization Type',
                    'avg_prominence': 'Average Prominence Rate (%)'
                },
                title='Average Prominence Rate by Organization Type'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(showlegend=False, plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics table
            st.dataframe(cat_prom, use_container_width=True, hide_index=True)
        
        # Mentions by category
        if 'total_mentions' in level2.columns:
            st.subheader("Total Mentions by Organization Type")
            
            cat_mentions = level2.groupby('category')['total_mentions'].sum().reset_index()
            cat_mentions = cat_mentions.sort_values('total_mentions', ascending=False)
            
            fig = px.bar(
                cat_mentions,
                x='category',
                y='total_mentions',
                color='category',
                color_discrete_map=CATEGORY_COLORS,
                labels={
                    'category': 'Organization Type',
                    'total_mentions': 'Total Mentions'
                },
                title='Total Mentions by Organization Type'
            )
            fig.update_layout(showlegend=False, plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Organization category data not available")

with tab4:
    st.header("Detailed Organization View")
    
    # Search functionality
    st.subheader("🔍 Search for an Organization")
    
    if 'org_name' in level2.columns:
        search_query = st.text_input("Enter organization name:")
        
        if search_query:
            matches = level2[level2['org_name'].str.contains(search_query, case=False, na=False)]
            
            if not matches.empty:
                st.markdown(f"**Found {len(matches)} organization(s)**")
                
                # Display matches
                for idx, row in matches.iterrows():
                    with st.expander(f"📊 {row['org_name']}"):
                        col_detail1, col_detail2 = st.columns(2)
                        
                        with col_detail1:
                            if 'total_mentions' in row:
                                st.metric("Total Mentions", f"{row['total_mentions']}")
                            if 'prominence_rate' in row:
                                st.metric("Prominence Rate", f"{row['prominence_rate']:.1f}%")
                            if 'category' in row:
                                st.markdown(f"**Category:** {row['category']}")
                        
                        with col_detail2:
                            if 'lobbying' in row or 'total_lobbying' in row:
                                lob_col = 'lobbying' if 'lobbying' in row else 'total_lobbying'
                                st.metric("Lobbying", f"${row[lob_col]:,.0f}")
                            if 'dem_mentions' in row and 'rep_mentions' in row:
                                st.metric("Dem/Rep Ratio", f"{row['dem_mentions']/max(row['rep_mentions'], 1):.2f}")
                        
                        # Show individual mentions if available
                        if not level1.empty and 'org_name' in level1.columns:
                            org_mentions = level1[level1['org_name'] == row['org_name']]
                            
                            if not org_mentions.empty:
                                st.markdown("---")
                                st.markdown(f"**Sample Mentions ({len(org_mentions)} total)**")
                                
                                sample_cols = ['date', 'speaker_name', 'party', 'prominence_prediction']
                                available_sample_cols = [col for col in sample_cols if col in org_mentions.columns]
                                
                                if available_sample_cols:
                                    st.dataframe(
                                        org_mentions[available_sample_cols].head(10),
                                        use_container_width=True
                                    )
            else:
                st.info("No organizations found matching your search")
    
    # Browse all organizations
    st.subheader("📑 Browse All Organizations")
    
    # Sorting options
    sort_by = st.selectbox(
        "Sort by:",
        ['Name (A-Z)', 'Total Mentions (High to Low)', 'Prominence Rate (High to Low)']
    )
    
    display_df = level2.copy()
    
    if sort_by == 'Name (A-Z)' and 'org_name' in display_df.columns:
        display_df = display_df.sort_values('org_name')
    elif sort_by == 'Total Mentions (High to Low)' and 'total_mentions' in display_df.columns:
        display_df = display_df.sort_values('total_mentions', ascending=False)
    elif sort_by == 'Prominence Rate (High to Low)' and 'prominence_rate' in display_df.columns:
        display_df = display_df.sort_values('prominence_rate', ascending=False)
    
    # Select columns to display
    all_cols = display_df.columns.tolist()
    default_cols = ['org_name', 'total_mentions', 'prominence_rate', 'category']
    available_default_cols = [col for col in default_cols if col in all_cols]
    
    selected_cols = st.multiselect(
        "Select columns to display:",
        options=all_cols,
        default=available_default_cols
    )
    
    if selected_cols:
        # Pagination
        page_size = st.selectbox("Rows per page:", [10, 25, 50, 100], index=1)
        total_pages = (len(display_df) - 1) // page_size + 1
        page = st.number_input("Page:", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        st.dataframe(
            display_df[selected_cols].iloc[start_idx:end_idx].reset_index(drop=True),
            use_container_width=True
        )
        
        st.markdown(f"Showing {start_idx + 1}-{min(end_idx, len(display_df))} of {len(display_df)} organizations")

# Export functionality
st.markdown("---")
st.header("📥 Export Data")

col_export1, col_export2 = st.columns(2)

with col_export1:
    if not level2.empty:
        csv_data = level2.to_csv(index=False)
        st.download_button(
            label="Download Organization Data (CSV)",
            data=csv_data,
            file_name="organization_level_data.csv",
            mime="text/csv"
        )

with col_export2:
    st.info(f"Dataset contains {len(level2):,} organizations")
