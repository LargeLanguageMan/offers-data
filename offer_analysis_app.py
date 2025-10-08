import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
from datetime import datetime
import numpy as np
import os

# Configure page
st.set_page_config(
    page_title="Automotive Offer Effectiveness Analysis",
    page_icon="🚗",
    layout="wide"
)

# Configure Gemini AI
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY not found. Please configure it in Streamlit secrets or environment variables.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash-exp')

@st.cache_data
def load_comprehensive_data():
    """Load the comprehensive joined dataset"""
    df = pd.read_csv('data/generated/comprehensive_joined_dataset.csv')

    # Convert sales to numeric, handling empty strings
    df['Sales_Total'] = pd.to_numeric(df['Sales_Total'], errors='coerce').fillna(0)

    # Filter out rows where retail offer is empty/null for offer analysis
    df_offers = df[df['Retail_Offer'].notna() & (df['Retail_Offer'] != '')].copy()

    # Create month ordering for proper timeline sorting
    month_order = ['Aug-23', 'Sep-23', 'Oct-23', 'Nov-23', 'Dec-23',
                   'Jan-24', 'Feb-24', 'Mar-24', 'Apr-24', 'May-24', 'Jun-24', 'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24', 'Nov-24', 'Dec-24',
                   'Jan-25', 'Feb-25', 'Mar-25', 'Apr-25', 'May-25', 'Jun-25', 'Jul-25', 'Aug-25', 'Sep-25']

    df_offers['Month_Order'] = df_offers['Month'].map({month: i for i, month in enumerate(month_order)})
    df_offers = df_offers.sort_values('Month_Order')

    return df, df_offers

def parse_combined_offers(offer_str):
    """Parse combined offers separated by semicolons"""
    if pd.isna(offer_str) or offer_str == '':
        return []
    return [offer.strip() for offer in str(offer_str).split(';')]

def get_all_offer_types(df_offers):
    """Get all unique offer types including from combined offers"""
    all_offers = set()

    for offer_str in df_offers['Retail_Offer'].dropna():
        individual_offers = parse_combined_offers(offer_str)
        all_offers.update(individual_offers)

    return sorted(list(all_offers))

def create_market_offer_effectiveness_chart(df_offers, selected_make=None):
    """Create market-level offer effectiveness chart"""

    # Filter by make if selected
    if selected_make and selected_make != 'All Makes':
        df_filtered = df_offers[df_offers['Make'] == selected_make]
        title_suffix = f" - {selected_make}"
    else:
        df_filtered = df_offers
        title_suffix = " - All Makes"

    # Get all offer types
    offer_types = get_all_offer_types(df_filtered)

    # Create monthly aggregation for each offer type
    monthly_data = {}

    for month in df_filtered['Month'].unique():
        monthly_data[month] = {}
        month_data = df_filtered[df_filtered['Month'] == month]

        for offer_type in offer_types:
            total_sales = 0

            for _, row in month_data.iterrows():
                if offer_type in parse_combined_offers(row['Retail_Offer']):
                    total_sales += row['Sales_Total']

            monthly_data[month][offer_type] = total_sales

    # Create the chart
    fig = go.Figure()

    # Sort months chronologically
    month_order = ['Aug-23', 'Sep-23', 'Oct-23', 'Nov-23', 'Dec-23',
                   'Jan-24', 'Feb-24', 'Mar-24', 'Apr-24', 'May-24', 'Jun-24', 'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24', 'Nov-24', 'Dec-24',
                   'Jan-25', 'Feb-25', 'Mar-25', 'Apr-25', 'May-25', 'Jun-25', 'Jul-25', 'Aug-25', 'Sep-25']

    sorted_months = [month for month in month_order if month in monthly_data.keys()]

    # Color palette for different offers
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5']

    # Add traces for each offer type
    for i, offer_type in enumerate(offer_types[:15]):  # Limit to top 15 for readability
        sales_data = [monthly_data[month].get(offer_type, 0) for month in sorted_months]

        # Only add if there's meaningful data (avoid flat zero lines)
        if max(sales_data) > 0:
            fig.add_trace(go.Scatter(
                x=sorted_months,
                y=sales_data,
                mode='lines+markers',
                name=offer_type,
                line=dict(width=3, color=colors[i % len(colors)]),
                marker=dict(size=6),
                hovertemplate=f'<b>{offer_type}</b><br>Month: %{{x}}<br>Sales Volume: %{{y:,}}<extra></extra>'
            ))

    fig.update_layout(
        title=f'<b>Market Offer Effectiveness Over Time{title_suffix}</b><br><sub>Sales Volume Driven by Each Offer Type</sub>',
        xaxis_title='Month',
        yaxis_title='Sales Volume',
        font=dict(family="Arial", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig

def create_segment_offer_effectiveness_chart(df_offers, selected_segment=None):
    """Create segment-level offer effectiveness chart"""

    # Filter by segment if selected
    if selected_segment and selected_segment != 'All Segments':
        df_filtered = df_offers[df_offers['Segment'] == selected_segment]
        title_suffix = f" - {selected_segment}"
    else:
        df_filtered = df_offers
        title_suffix = " - All Segments"

    # Get all offer types
    offer_types = get_all_offer_types(df_filtered)

    # Create monthly aggregation for each offer type
    monthly_data = {}

    for month in df_filtered['Month'].unique():
        monthly_data[month] = {}
        month_data = df_filtered[df_filtered['Month'] == month]

        for offer_type in offer_types:
            total_sales = 0

            for _, row in month_data.iterrows():
                if offer_type in parse_combined_offers(row['Retail_Offer']):
                    total_sales += row['Sales_Total']

            monthly_data[month][offer_type] = total_sales

    # Create the chart
    fig = go.Figure()

    # Sort months chronologically
    month_order = ['Aug-23', 'Sep-23', 'Oct-23', 'Nov-23', 'Dec-23',
                   'Jan-24', 'Feb-24', 'Mar-24', 'Apr-24', 'May-24', 'Jun-24', 'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24', 'Nov-24', 'Dec-24',
                   'Jan-25', 'Feb-25', 'Mar-25', 'Apr-25', 'May-25', 'Jun-25', 'Jul-25', 'Aug-25', 'Sep-25']

    sorted_months = [month for month in month_order if month in monthly_data.keys()]

    # Color palette for different offers
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5']

    # Add traces for each offer type
    for i, offer_type in enumerate(offer_types[:15]):  # Limit to top 15 for readability
        sales_data = [monthly_data[month].get(offer_type, 0) for month in sorted_months]

        # Only add if there's meaningful data (avoid flat zero lines)
        if max(sales_data) > 0:
            fig.add_trace(go.Scatter(
                x=sorted_months,
                y=sales_data,
                mode='lines+markers',
                name=offer_type,
                line=dict(width=3, color=colors[i % len(colors)]),
                marker=dict(size=6),
                hovertemplate=f'<b>{offer_type}</b><br>Month: %{{x}}<br>Sales Volume: %{{y:,}}<extra></extra>'
            ))

    fig.update_layout(
        title=f'<b>Segment Offer Effectiveness Over Time{title_suffix}</b><br><sub>Sales Volume Driven by Each Offer Type</sub>',
        xaxis_title='Month',
        yaxis_title='Sales Volume',
        font=dict(family="Arial", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig

# CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .page-header {
        color: #2c3e50;
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #3498db;
    }
    .filter-container {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main app
def main():
    # Load data
    df, df_offers = load_comprehensive_data()

    # Sidebar navigation
    st.sidebar.title("🚗 Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis:",
        ["Market Offer Effectiveness", "Segment Offer Effectiveness"]
    )

    # Market Offer Effectiveness Page
    if page == "Market Offer Effectiveness":
        st.markdown('<div class="page-header">📈 Market Offer Effectiveness Analysis</div>', unsafe_allow_html=True)
        st.markdown("Analyze how different offer types drive sales volumes across the entire automotive market")

        # Filter container
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])

        with col1:
            # Make filter
            all_makes = ['All Makes'] + sorted(df_offers['Make'].unique())
            selected_make = st.selectbox("Filter by Make:", all_makes, key="market_make_filter")

        with col2:
            st.metric("Total Records", f"{len(df_offers):,}")

        st.markdown('</div>', unsafe_allow_html=True)

        # Create and display chart
        fig = create_market_offer_effectiveness_chart(df_offers, selected_make)
        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.subheader("📊 Market Summary")

        col1, col2, col3, col4 = st.columns(4)

        # Filter data for metrics
        if selected_make and selected_make != 'All Makes':
            filtered_data = df_offers[df_offers['Make'] == selected_make]
        else:
            filtered_data = df_offers

        with col1:
            total_sales = filtered_data['Sales_Total'].sum()
            st.metric("Total Sales Volume", f"{total_sales:,}")

        with col2:
            unique_offers = len(get_all_offer_types(filtered_data))
            st.metric("Unique Offer Types", unique_offers)

        with col3:
            unique_makes = filtered_data['Make'].nunique()
            st.metric("Makes Analyzed", unique_makes)

        with col4:
            date_range = f"{filtered_data['Month'].min()} to {filtered_data['Month'].max()}"
            st.metric("Date Range", date_range)

        st.markdown('</div>', unsafe_allow_html=True)

    # Segment Offer Effectiveness Page
    elif page == "Segment Offer Effectiveness":
        st.markdown('<div class="page-header">🎯 Segment Offer Effectiveness Analysis</div>', unsafe_allow_html=True)
        st.markdown("Analyze how different offer types drive sales volumes within specific vehicle segments")

        # Filter container
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])

        with col1:
            # Segment filter
            all_segments = ['All Segments'] + sorted(df_offers['Segment'].unique())
            selected_segment = st.selectbox("Filter by Segment:", all_segments, key="segment_filter")

        with col2:
            st.metric("Total Records", f"{len(df_offers):,}")

        st.markdown('</div>', unsafe_allow_html=True)

        # Create and display chart
        fig = create_segment_offer_effectiveness_chart(df_offers, selected_segment)
        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.subheader("📊 Segment Summary")

        col1, col2, col3, col4 = st.columns(4)

        # Filter data for metrics
        if selected_segment and selected_segment != 'All Segments':
            filtered_data = df_offers[df_offers['Segment'] == selected_segment]
        else:
            filtered_data = df_offers

        with col1:
            total_sales = filtered_data['Sales_Total'].sum()
            st.metric("Total Sales Volume", f"{total_sales:,}")

        with col2:
            unique_offers = len(get_all_offer_types(filtered_data))
            st.metric("Unique Offer Types", unique_offers)

        with col3:
            unique_segments = filtered_data['Segment'].nunique()
            st.metric("Segments Analyzed", unique_segments)

        with col4:
            unique_models = filtered_data['Model'].nunique()
            st.metric("Models in Analysis", unique_models)

        st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Analysis Notes")
    st.sidebar.markdown("""
    - **Line thickness** represents sales volume
    - **Multiple offers** (e.g., 'Finance;Servicing') are parsed separately
    - **Make/Segment filters** allow focused analysis
    - **Hover** over lines for detailed metrics
    """)

    st.sidebar.markdown("### 📊 Data Source")
    st.sidebar.markdown("Comprehensive joined dataset with RBA cash rates")

if __name__ == "__main__":
    main()