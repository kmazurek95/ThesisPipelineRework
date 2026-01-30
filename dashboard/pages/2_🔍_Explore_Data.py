"""
🔍 Explore Data - Interactive Filters and Visualizations
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

from dashboard.utils.config import configure_page, apply_custom_css, PARTY_COLORS
from dashboard.utils.data_loader import load_all_data
from dashboard.utils.visualizations import create_time_series_plot, create_bar_chart, create_scatter_plot

# Page configuration
configure_page(title="Explore Data - Interest Group Prominence", icon="🔍")
apply_custom_css()

st.title("🔍 Explore the Data")
st.markdown("Interactive exploration of interest group mentions in congressional speech")

# Load data
with st.spinner("Loading data..."):
    data = load_all_data()
    level1 = data['level1']
    level2 = data['level2']
    level3 = data['level3']

# Sidebar filters
st.sidebar.header("🎛️ Filters")

# Initialize filter variables
filtered_level1 = level1.copy()
filtered_level2 = level2.copy()
filtered_level3 = level3.copy()

# Party filter
if 'party' in level1.columns:
    parties = sorted(level1['party'].dropna().unique())
    selected_parties = st.sidebar.multiselect(
        "Political Party",
        options=parties,
        default=parties,
        help="Filter mentions by speaker's party"
    )
    if selected_parties:
        filtered_level1 = filtered_level1[filtered_level1['party'].isin(selected_parties)]

# Chamber filter
if 'chamber' in level1.columns:
    chambers = sorted(level1['chamber'].dropna().unique())
    selected_chambers = st.sidebar.multiselect(
        "Chamber",
        options=chambers,
        default=chambers,
        help="Filter by House or Senate"
    )
    if selected_chambers:
        filtered_level1 = filtered_level1[filtered_level1['chamber'].isin(selected_chambers)]

# Date range filter
if 'date' in level1.columns and not level1['date'].isna().all():
    min_date = pd.to_datetime(level1['date']).min()
    max_date = pd.to_datetime(level1['date']).max()
    
    if pd.notna(min_date) and pd.notna(max_date):
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        if len(date_range) == 2:
            filtered_level1 = filtered_level1[
                (pd.to_datetime(filtered_level1['date']) >= pd.Timestamp(date_range[0])) &
                (pd.to_datetime(filtered_level1['date']) <= pd.Timestamp(date_range[1]))
            ]

# Prominence filter
if 'prominence_prediction' in level1.columns:
    prominence_options = st.sidebar.radio(
        "Mention Type",
        options=['All', 'Prominent Only', 'Passing Only'],
        index=0
    )
    if prominence_options == 'Prominent Only':
        filtered_level1 = filtered_level1[filtered_level1['prominence_prediction'] == 1]
    elif prominence_options == 'Passing Only':
        filtered_level1 = filtered_level1[filtered_level1['prominence_prediction'] == 0]

st.sidebar.markdown(f"**{len(filtered_level1):,}** mentions selected")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📈 Time Trends",
    "🏛️ Organizations",
    "👥 Politicians"
])

with tab1:
    st.header("Dataset Overview")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Mentions", f"{len(filtered_level1):,}")
    
    with col2:
        if 'prominence_prediction' in filtered_level1.columns:
            prom_rate = (filtered_level1['prominence_prediction'] == 1).mean() * 100
            st.metric("Prominence Rate", f"{prom_rate:.1f}%")
        else:
            st.metric("Prominence Rate", "N/A")
    
    with col3:
        if 'org_name' in filtered_level1.columns:
            unique_orgs = filtered_level1['org_name'].nunique()
            st.metric("Unique Organizations", f"{unique_orgs:,}")
        else:
            st.metric("Unique Organizations", "N/A")
    
    with col4:
        if 'speaker_name' in filtered_level1.columns:
            unique_speakers = filtered_level1['speaker_name'].nunique()
            st.metric("Unique Speakers", f"{unique_speakers:,}")
        else:
            st.metric("Unique Speakers", "N/A")
    
    st.markdown("---")
    
    # Distribution visualizations
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        if 'party' in filtered_level1.columns:
            st.subheader("Mentions by Party")
            party_counts = filtered_level1['party'].value_counts().reset_index()
            party_counts.columns = ['party', 'count']
            
            fig = px.bar(
                party_counts,
                x='party',
                y='count',
                color='party',
                color_discrete_map=PARTY_COLORS,
                labels={'party': 'Party', 'count': 'Number of Mentions'}
            )
            fig.update_layout(showlegend=False, plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
    
    with col_viz2:
        if 'chamber' in filtered_level1.columns:
            st.subheader("Mentions by Chamber")
            chamber_counts = filtered_level1['chamber'].value_counts().reset_index()
            chamber_counts.columns = ['chamber', 'count']
            
            fig = px.bar(
                chamber_counts,
                x='chamber',
                y='count',
                color='chamber',
                labels={'chamber': 'Chamber', 'count': 'Number of Mentions'}
            )
            fig.update_layout(showlegend=False, plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
    
    # Prominence breakdown
    if 'prominence_prediction' in filtered_level1.columns and 'party' in filtered_level1.columns:
        st.markdown("---")
        st.subheader("Prominence by Party")
        
        prom_by_party = filtered_level1.groupby('party')['prominence_prediction'].agg([
            ('Total', 'count'),
            ('Prominent', lambda x: (x == 1).sum()),
            ('Prominence_Rate', lambda x: (x == 1).mean() * 100)
        ]).reset_index()
        
        fig = px.bar(
            prom_by_party,
            x='party',
            y='Prominence_Rate',
            color='party',
            color_discrete_map=PARTY_COLORS,
            text='Prominence_Rate',
            labels={'party': 'Party', 'Prominence_Rate': 'Prominence Rate (%)'}
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(showlegend=False, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Time Trends")
    
    if 'date' in filtered_level1.columns and not filtered_level1['date'].isna().all():
        # Mentions over time
        st.subheader("Mentions Over Time")
        
        # Aggregate by date
        time_df = filtered_level1.copy()
        time_df['date'] = pd.to_datetime(time_df['date'])
        
        # Group by week
        time_agg = time_df.groupby(pd.Grouper(key='date', freq='W')).size().reset_index()
        time_agg.columns = ['date', 'count']
        
        fig = px.line(
            time_agg,
            x='date',
            y='count',
            labels={'date': 'Date', 'count': 'Number of Mentions'},
            title='Weekly Mention Count'
        )
        fig.update_layout(hovermode='x unified', plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
        
        # By party over time
        if 'party' in filtered_level1.columns:
            st.subheader("Mentions Over Time by Party")
            
            time_party = time_df.groupby([pd.Grouper(key='date', freq='W'), 'party']).size().reset_index()
            time_party.columns = ['date', 'party', 'count']
            
            fig = px.line(
                time_party,
                x='date',
                y='count',
                color='party',
                color_discrete_map=PARTY_COLORS,
                labels={'date': 'Date', 'count': 'Number of Mentions', 'party': 'Party'}
            )
            fig.update_layout(hovermode='x unified', plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
        
        # Prominence rate over time
        if 'prominence_prediction' in filtered_level1.columns:
            st.subheader("Prominence Rate Over Time")
            
            time_prom = time_df.groupby(pd.Grouper(key='date', freq='W'))['prominence_prediction'].mean().reset_index()
            time_prom['prominence_prediction'] = time_prom['prominence_prediction'] * 100
            
            fig = px.line(
                time_prom,
                x='date',
                y='prominence_prediction',
                labels={'date': 'Date', 'prominence_prediction': 'Prominence Rate (%)'},
                title='Weekly Prominence Rate'
            )
            fig.update_layout(hovermode='x unified', plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Date information not available in the dataset")

with tab3:
    st.header("Organizations")
    
    if 'org_name' in filtered_level1.columns:
        # Top mentioned organizations
        st.subheader("Most Mentioned Organizations")
        
        top_n = st.slider("Number of organizations to display", 10, 50, 20)
        
        org_mentions = filtered_level1['org_name'].value_counts().head(top_n).reset_index()
        org_mentions.columns = ['org_name', 'count']
        
        fig = create_bar_chart(
            org_mentions,
            x_col='org_name',
            y_col='count',
            title=f'Top {top_n} Most Mentioned Organizations',
            orientation='h'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Organization search
        st.subheader("🔍 Search Organizations")
        search_term = st.text_input("Enter organization name:")
        
        if search_term:
            matching_orgs = filtered_level1[
                filtered_level1['org_name'].str.contains(search_term, case=False, na=False)
            ]
            
            if not matching_orgs.empty:
                st.markdown(f"**Found {len(matching_orgs):,} mentions**")
                
                # Show details
                org_details = matching_orgs.groupby('org_name').agg({
                    'org_name': 'size',
                    'prominence_prediction': lambda x: (x == 1).mean() * 100 if 'prominence_prediction' in x else 0
                }).reset_index()
                org_details.columns = ['Organization', 'Total Mentions', 'Prominence Rate (%)']
                
                st.dataframe(org_details, use_container_width=True)
            else:
                st.info("No matching organizations found")
        
        # Show sample mentions table
        st.subheader("Sample Mentions")
        display_cols = ['org_name', 'speaker_name', 'party', 'chamber', 'date']
        available_display_cols = [col for col in display_cols if col in filtered_level1.columns]
        
        if available_display_cols:
            st.dataframe(
                filtered_level1[available_display_cols].head(100),
                use_container_width=True
            )

with tab4:
    st.header("Politicians")
    
    if 'speaker_name' in filtered_level1.columns:
        # Top speakers
        st.subheader("Most Active Speakers")
        
        top_speakers_n = st.slider("Number of speakers to display", 10, 50, 20, key='speaker_slider')
        
        speaker_counts = filtered_level1['speaker_name'].value_counts().head(top_speakers_n).reset_index()
        speaker_counts.columns = ['speaker_name', 'count']
        
        # Add party if available
        if 'party' in filtered_level1.columns:
            speaker_party = filtered_level1.groupby('speaker_name')['party'].first()
            speaker_counts = speaker_counts.merge(
                speaker_party.reset_index(),
                on='speaker_name',
                how='left'
            )
            
            fig = create_bar_chart(
                speaker_counts,
                x_col='speaker_name',
                y_col='count',
                title=f'Top {top_speakers_n} Most Active Speakers',
                orientation='h',
                color_col='party'
            )
        else:
            fig = create_bar_chart(
                speaker_counts,
                x_col='speaker_name',
                y_col='count',
                title=f'Top {top_speakers_n} Most Active Speakers',
                orientation='h'
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Prominence rate by speaker
        if 'prominence_prediction' in filtered_level1.columns:
            st.subheader("Prominence Rate by Top Speakers")
            
            speaker_prom = filtered_level1.groupby('speaker_name').agg({
                'prominence_prediction': ['count', lambda x: (x == 1).mean() * 100]
            }).reset_index()
            speaker_prom.columns = ['speaker_name', 'total_mentions', 'prominence_rate']
            speaker_prom = speaker_prom[speaker_prom['total_mentions'] >= 5]  # Filter for reliability
            speaker_prom = speaker_prom.nlargest(20, 'prominence_rate')
            
            fig = px.scatter(
                speaker_prom,
                x='total_mentions',
                y='prominence_rate',
                hover_data=['speaker_name'],
                labels={
                    'total_mentions': 'Total Mentions',
                    'prominence_rate': 'Prominence Rate (%)',
                    'speaker_name': 'Speaker'
                },
                title='Speaker Activity vs. Prominence Rate (min. 5 mentions)'
            )
            fig.update_layout(plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

# Download filtered data
st.markdown("---")
st.header("📥 Download Data")

col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    if not filtered_level1.empty:
        csv = filtered_level1.to_csv(index=False)
        st.download_button(
            label="Download Filtered Mentions (CSV)",
            data=csv,
            file_name="filtered_mentions.csv",
            mime="text/csv"
        )

with col_dl2:
    st.info(f"Current selection: {len(filtered_level1):,} mentions")
