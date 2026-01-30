"""
Visualization utilities for Streamlit dashboard
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional
from dashboard.utils.config import PARTY_COLORS, CATEGORY_COLORS, PROMINENCE_COLORS


def create_time_series_plot(
    df: pd.DataFrame,
    date_col: str = 'date',
    value_col: str = 'count',
    group_col: Optional[str] = None,
    title: str = "Mentions Over Time"
) -> go.Figure:
    """Create an interactive time series plot"""
    
    if group_col and group_col in df.columns:
        fig = px.line(
            df,
            x=date_col,
            y=value_col,
            color=group_col,
            title=title,
            labels={date_col: 'Date', value_col: 'Count'},
            color_discrete_map=PARTY_COLORS if group_col == 'party' else None
        )
    else:
        fig = px.line(
            df,
            x=date_col,
            y=value_col,
            title=title,
            labels={date_col: 'Date', value_col: 'Count'}
        )
    
    fig.update_layout(
        hovermode='x unified',
        plot_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        title_font_size=16
    )
    
    return fig


def create_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    orientation: str = 'h',
    color_col: Optional[str] = None,
    top_n: Optional[int] = None
) -> go.Figure:
    """Create an interactive bar chart"""
    
    plot_df = df.copy()
    if top_n:
        plot_df = plot_df.nlargest(top_n, y_col)
    
    if orientation == 'h':
        fig = px.bar(
            plot_df,
            y=x_col,
            x=y_col,
            title=title,
            orientation='h',
            color=color_col,
            color_discrete_map=PARTY_COLORS if color_col == 'party' else None
        )
        fig.update_yaxes(categoryorder='total ascending')
    else:
        fig = px.bar(
            plot_df,
            x=x_col,
            y=y_col,
            title=title,
            color=color_col,
            color_discrete_map=PARTY_COLORS if color_col == 'party' else None
        )
    
    fig.update_layout(
        plot_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        title_font_size=16,
        showlegend=True if color_col else False
    )
    
    return fig


def create_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    hover_data: Optional[list] = None
) -> go.Figure:
    """Create an interactive scatter plot"""
    
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        title=title,
        hover_data=hover_data,
        color_discrete_map=PARTY_COLORS if color_col == 'party' else None
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        title_font_size=16
    )
    
    # Add trendline if no grouping
    if not color_col:
        # Calculate trendline
        z = pd.DataFrame({x_col: df[x_col], y_col: df[y_col]}).dropna()
        if len(z) > 1:
            coeffs = pd.Series(z[y_col]).corr(pd.Series(z[x_col]))
    
    return fig


def create_heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    title: str
) -> go.Figure:
    """Create an interactive heatmap"""
    
    # Pivot data for heatmap
    pivot_df = df.pivot_table(
        index=y_col,
        columns=x_col,
        values=value_col,
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale='RdYlBu_r',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        font=dict(family="Arial, sans-serif"),
        title_font_size=16
    )
    
    return fig


def create_distribution_plot(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    group_col: Optional[str] = None
) -> go.Figure:
    """Create a distribution plot (histogram or box plot)"""
    
    if group_col and group_col in df.columns:
        fig = px.box(
            df,
            y=value_col,
            x=group_col,
            title=title,
            color=group_col,
            color_discrete_map=PARTY_COLORS if group_col == 'party' else None
        )
    else:
        fig = px.histogram(
            df,
            x=value_col,
            title=title,
            nbins=30
        )
    
    fig.update_layout(
        plot_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        title_font_size=16
    )
    
    return fig


def create_pie_chart(
    df: pd.DataFrame,
    names_col: str,
    values_col: str,
    title: str
) -> go.Figure:
    """Create an interactive pie chart"""
    
    fig = px.pie(
        df,
        names=names_col,
        values=values_col,
        title=title
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    
    fig.update_layout(
        font=dict(family="Arial, sans-serif"),
        title_font_size=16
    )
    
    return fig
