import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
import tempfile
import os

# Configure page
st.set_page_config(
    page_title="Automotive Offer Effectiveness Analysis",
    page_icon="🚗",
    layout="wide"
)

# Authentication function
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (st.session_state.get("username") == "offers" and
            st.session_state.get("password") == "innocean2025"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
            del st.session_state["username"]  # Don't store username
        else:
            st.session_state["password_correct"] = False

    # First run, show inputs for username + password
    if "password_correct" not in st.session_state:
        st.markdown("## 🔐 Login")
        st.markdown("Please enter your credentials to access the dashboard")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            if st.button("Login", type="primary", use_container_width=True):
                password_entered()
        return False
    # Password correct
    elif st.session_state["password_correct"]:
        return True
    # Password incorrect
    else:
        st.markdown("## 🔐 Login")
        st.markdown("Please enter your credentials to access the dashboard")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            if st.button("Login", type="primary", use_container_width=True):
                password_entered()
            st.error("😕 Username or password incorrect")
        return False

# Check authentication before proceeding
if not check_password():
    st.stop()

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

@st.cache_data
def load_website_data():
    """Load and aggregate website visit data"""
    # Load total website visits
    df_total_web = pd.read_csv('data/archive/website-visit - total-web-month.csv')
    # Load offers page views
    df_offers_web = pd.read_csv('data/archive/website-visit - offers-web-month.csv')

    # Aggregate by month (sum daily counts)
    total_web_monthly = df_total_web.groupby('Month')['Event count'].sum().reset_index()
    total_web_monthly.columns = ['Month', 'Total_Visits']

    offers_web_monthly = df_offers_web.groupby('Month')['Event count'].sum().reset_index()
    offers_web_monthly.columns = ['Month', 'Offers_Page_Views']

    # Merge the two datasets
    web_data = pd.merge(total_web_monthly, offers_web_monthly, on='Month', how='outer')

    return web_data

def get_hyundai_offers_by_month(df_offers):
    """Extract Hyundai offers aggregated by month"""
    # Filter for Hyundai
    hyundai_offers = df_offers[df_offers['Make'] == 'Hyundai'].copy()

    # Group by month and aggregate offers
    monthly_offers = {}

    for month in hyundai_offers['Month'].unique():
        month_data = hyundai_offers[hyundai_offers['Month'] == month]

        # Get all unique offers for this month across all Hyundai models
        offers_list = []
        for offer in month_data['Retail_Offer'].dropna():
            if str(offer).strip() and str(offer) != 'nan':
                # Split multiple offers separated by semicolons
                offers = [o.strip() for o in str(offer).split(';') if o.strip()]
                offers_list.extend(offers)

        # Get unique offers
        unique_offers = list(set(offers_list))

        if unique_offers:
            monthly_offers[month] = {
                'offers': unique_offers,
                'offer_display': ' + '.join(unique_offers) if len(unique_offers) > 1 else unique_offers[0],
                'has_offer': True
            }
        else:
            monthly_offers[month] = {
                'offers': [],
                'offer_display': None,
                'has_offer': False
            }

    return monthly_offers

@st.cache_data
def load_make_level_media_spend():
    """Load make-level media spend data from car-make-report.csv"""
    # Load the car-make-report with malformed header handling
    with open('data/source/car-make-report.csv', 'r') as f:
        lines = f.readlines()

    if '"Sales' in lines[0]:
        header_line = lines[0].strip().replace('"Sales\n', 'Sales_') + lines[1].strip()
        lines[1] = header_line + '\n'
        lines = lines[1:]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        tmp_file.writelines(lines)
        tmp_path = tmp_file.name

    df_make = pd.read_csv(tmp_path)
    os.unlink(tmp_path)

    # Filter for Total Media Spend rows
    df_media = df_make[df_make['Category'] == 'Total Media Spend'].copy()

    # Get monthly columns
    monthly_cols = [col for col in df_media.columns if '-' in col and len(col) == 6]

    # Create a dictionary: {Make: {Month: MediaSpend}}
    media_spend_dict = {}

    for _, row in df_media.iterrows():
        make = row['Make']
        media_spend_dict[make] = {}

        for month_col in monthly_cols:
            if month_col in row.index and pd.notna(row[month_col]):
                # Clean and convert media spend value
                media_val = str(row[month_col]).replace('$', '').replace(',', '').strip()
                try:
                    media_spend_dict[make][month_col] = float(media_val)
                except (ValueError, TypeError):
                    media_spend_dict[make][month_col] = 0
            else:
                media_spend_dict[make][month_col] = 0

    return media_spend_dict

@st.cache_data
def load_data():
    """Load and process the automotive data with enhanced offer information"""
    # Load original car model report
    with open('data/source/car-model-report.csv', 'r') as f:
        lines = f.readlines()

    if '"Sales' in lines[0]:
        header_line = lines[0].strip().replace('"Sales\n', 'Sales_') + lines[1].strip()
        lines[1] = header_line + '\n'
        lines = lines[1:]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        tmp_file.writelines(lines)
        tmp_path = tmp_file.name

    df_original = pd.read_csv(tmp_path)
    os.unlink(tmp_path)

    column_mapping = {
        'SalesRank': 'Sales_Rank',
        'Monthly Average': 'Monthly_Average',
        '2024 YTD': '2024_YTD',
        '2025 YTD': '2025_YTD'
    }

    df_original = df_original.rename(columns=column_mapping)

    # Load enhanced data with accurate offer information
    with open('data/source/enhanced-data.csv', 'r') as f:
        enhanced_lines = f.readlines()

    if '"Sales' in enhanced_lines[0]:
        enhanced_header = enhanced_lines[0].strip().replace('"Sales\n', 'Sales_') + enhanced_lines[1].strip()
        enhanced_lines[1] = enhanced_header + '\n'
        enhanced_lines = enhanced_lines[1:]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        tmp_file.writelines(enhanced_lines)
        tmp_path = tmp_file.name

    df_enhanced = pd.read_csv(tmp_path)
    os.unlink(tmp_path)

    enhanced_column_mapping = {
        'SalesRank': 'Sales_Rank',
        '12 Month Average': 'Monthly_Average'
    }

    df_enhanced = df_enhanced.rename(columns=enhanced_column_mapping)

    # Merge the enhanced offer data with original data
    # For Hyundai models, replace the retail offer data with enhanced data
    df_merged = df_original.copy()

    # Update retail offer data for Hyundai models using enhanced data
    hyundai_enhanced = df_enhanced[df_enhanced['Make'] == 'Hyundai']

    for idx, enhanced_row in hyundai_enhanced.iterrows():
        if enhanced_row['Category'] == 'Retail Offer':
            # Find matching row in original data
            mask = (
                (df_merged['Make'] == enhanced_row['Make']) &
                (df_merged['Model'] == enhanced_row['Model']) &
                (df_merged['Category'] == 'Retail Offer')
            )

            if mask.any():
                # Get monthly columns that exist in both datasets
                monthly_cols_original = [col for col in df_merged.columns if '-' in col and len(col) == 6]
                monthly_cols_enhanced = [col for col in df_enhanced.columns if '-' in col and len(col) == 6]

                # Update the retail offer data with enhanced data
                for month_col in monthly_cols_original:
                    if month_col in monthly_cols_enhanced:
                        df_merged.loc[mask, month_col] = enhanced_row[month_col]

    return df_merged

def get_model_data(df, model_name, make='Hyundai'):
    """Extract comprehensive timeline data for a specific model"""
    monthly_cols = [col for col in df.columns if '-' in col and len(col) == 6]
    model_data = df[(df['Make'] == make) & (df['Model'] == model_name)].copy()

    if model_data.empty:
        return None

    # Extract different data types
    sales_row = model_data[model_data['Category'] == 'Sales']
    offers_row = model_data[model_data['Category'] == 'Retail Offer']
    media_spend_row = model_data[model_data['Category'] == 'Total Media Spend']

    # Get segment info
    segment = sales_row.iloc[0]['Segment'] if not sales_row.empty else 'Unknown'

    # Prepare timeline data
    timeline_data = []

    for month_col in monthly_cols:
        try:
            # Sales data
            sales_val = sales_row.iloc[0][month_col] if not sales_row.empty else 0
            sales_clean = pd.to_numeric(str(sales_val).replace(',', ''), errors='coerce')

            # Offer data - handle multiple offers separated by semicolons
            offer_val = offers_row.iloc[0][month_col] if not offers_row.empty else None
            offer_clean = str(offer_val) if pd.notna(offer_val) and str(offer_val) != 'nan' else None

            # Clean up and format offers
            if offer_clean and offer_clean.strip():
                # Handle multiple offers separated by semicolons
                offers = [o.strip() for o in offer_clean.split(';') if o.strip()]
                offer_display = ' + '.join(offers) if len(offers) > 1 else offers[0] if offers else None
                has_offer = bool(offers)
            else:
                offer_display = None
                has_offer = False

            # Media spend
            media_val = media_spend_row.iloc[0][month_col] if not media_spend_row.empty else 0
            if pd.notna(media_val) and str(media_val) != 'nan':
                media_clean = pd.to_numeric(str(media_val).replace('$', '').replace(',', ''), errors='coerce')
            else:
                media_clean = 0

            date = pd.to_datetime(month_col, format='%b-%y')

            timeline_data.append({
                'Date': date,
                'Month': month_col,
                'Sales': sales_clean if pd.notna(sales_clean) else 0,
                'Media_Spend': media_clean if pd.notna(media_clean) else 0,
                'Offer': offer_display,
                'Has_Offer': has_offer
            })

        except Exception as e:
            continue

    timeline_df = pd.DataFrame(timeline_data)

    return {
        'data': timeline_df,
        'model': model_name,
        'make': make,
        'segment': segment
    }

def create_comprehensive_timeline(model_info):
    """Create the comprehensive timeline chart you requested"""
    data = model_info['data']
    model_name = model_info['model']
    segment = model_info['segment']

    if data.empty:
        return None

    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.15,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        # subplot_titles=[
        #     f'{model_name} - Sales & Media Spend Timeline',
        #     'Offer Strategy by Month'
        # ]
    )

    # 1. Sales line (primary y-axis) - More elegant styling
    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=data['Sales'],
            mode='lines+markers',
            name='Monthly Sales',
            line=dict(color='#2E86AB', width=4, shape='spline'),
            marker=dict(size=8, color='#2E86AB', opacity=0.8),
            fill='tonexty',
            fillcolor='rgba(46, 134, 171, 0.1)',
            hovertemplate='<b>%{x|%b %Y}</b><br>Sales: %{y:,} units<extra></extra>'
        ),
        row=1, col=1, secondary_y=False
    )

    # 2. Media spend line (secondary y-axis) - More subtle
    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=data['Media_Spend'],
            mode='lines',
            name='Media Investment',
            line=dict(color='#A23B72', width=2, dash='dot'),
            opacity=0.7,
            hovertemplate='<b>%{x|%b %Y}</b><br>Media: $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1, secondary_y=True
    )

    # 3. Offer indicators - Elegant markers instead of big stars
    offer_months = data[data['Has_Offer'] == True].copy()

    if not offer_months.empty:
        # Add subtle background highlighting for offer months
        for idx, row in offer_months.iterrows():
            fig.add_vline(
                x=row['Date'],
                line_width=1,
                line_dash="dash",
                line_color="rgba(241, 143, 1, 0.3)",
                row=1, col=1
            )

        # Add elegant offer markers
        fig.add_trace(
            go.Scatter(
                x=offer_months['Date'],
                y=offer_months['Sales'],
                mode='markers',
                name='Active Offers',
                marker=dict(
                    size=12,
                    color='#F18F01',
                    symbol='circle',
                    line=dict(width=2, color='#ffffff'),
                    opacity=0.9
                ),
                hovertemplate='<b>%{x|%b %Y}</b><br>Sales: %{y:,} units<br>Offer: %{text}<extra></extra>',
                text=offer_months['Offer']
            ),
            row=1, col=1, secondary_y=False
        )

    # 4. Bottom chart: Offer types as text annotations
    # Create a bar chart showing offer types
    offer_data = data[data['Has_Offer'] == True].copy()

    if not offer_data.empty:
        # Create categorical y-values for different offer types
        unique_offers = offer_data['Offer'].unique()
        offer_mapping = {offer: i+1 for i, offer in enumerate(unique_offers)}

        offer_data['Offer_Y'] = offer_data['Offer'].map(offer_mapping)

        for offer_type in unique_offers:
            offer_subset = offer_data[offer_data['Offer'] == offer_type]

            fig.add_trace(
                go.Scatter(
                    x=offer_subset['Date'],
                    y=offer_subset['Offer_Y'],
                    mode='markers+text',
                    name=offer_type,
                    marker=dict(size=12, symbol='square'),
                    text=offer_subset['Month'],
                    textposition="middle center",
                    textfont=dict(size=10),
                    hovertemplate=f'<b>%{{x|%b %Y}}</b><br>Offer: {offer_type}<extra></extra>'
                ),
                row=2, col=1
            )

    # Update layout with modern, clean styling matching offer charts
    fig.update_layout(
        title=dict(
            text=f'<b style="color:#1a1a1a; font-size:22px;">Hyundai {model_name}</b><br><span style="color:#4a4a4a; font-size:14px;">{segment} • Performance Timeline</span>',
            x=0.5,
            xanchor='center',
            font=dict(color='#1a1a1a')
        ),
        height=800,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=12, color='#1a1a1a'),
        hoverlabel=dict(
            bgcolor='white',
            bordercolor='#1a1a1a',
            font=dict(color='#1a1a1a')
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.10,
            xanchor="center",
            x=0.49,
            bgcolor='white',
            bordercolor='#1a1a1a',
            borderwidth=2,
            font=dict(color='#1a1a1a', size=10),
            itemsizing='constant',
            itemwidth=30
        ),
        margin=dict(l=80, r=80, t=220, b=80)
    )

    # Update axes with improved styling matching offer charts
    fig.update_yaxes(
        title=dict(text='<b style="color:#1a1a1a;">Sales (Units)</b>', font=dict(size=14, color='#1a1a1a')),
        row=1, col=1, secondary_y=False,
        tickfont=dict(color='#1a1a1a', size=11),
        gridcolor='#d0d0d0',
        showgrid=True,
        gridwidth=1,
        linecolor='#1a1a1a',
        linewidth=2,
        showline=True,
        mirror=True,
        zeroline=False,
        tickformat=',d'
    )
    fig.update_yaxes(
        title=dict(text='<b style="color:#1a1a1a;">Media Investment ($)</b>', font=dict(size=14, color='#1a1a1a')),
        row=1, col=1, secondary_y=True,
        tickfont=dict(color='#1a1a1a', size=11),
        gridcolor='#e8e8e8',
        showgrid=False,
        linecolor='#1a1a1a',
        linewidth=2,
        showline=True,
        mirror=True,
        zeroline=False,
        tickformat='$,.0f'
    )
    fig.update_yaxes(
        title=dict(text='<b style="color:#1a1a1a;">Offer Types</b>', font=dict(size=14, color='#1a1a1a')),
        row=2, col=1,
        tickfont=dict(color='#1a1a1a', size=11),
        tickmode='array',
        tickvals=list(range(1, len(offer_data['Offer'].unique())+1)) if not offer_data.empty else [],
        ticktext=list(offer_data['Offer'].unique()) if not offer_data.empty else [],
        gridcolor='#d0d0d0',
        showgrid=True,
        gridwidth=1,
        linecolor='#1a1a1a',
        linewidth=2,
        showline=True,
        mirror=True
    )

    fig.update_xaxes(
        title=dict(text='<b style="color:#1a1a1a;">Timeline</b>', font=dict(size=14, color='#1a1a1a')),
        row=2, col=1,
        tickfont=dict(color='#1a1a1a', size=11),
        showgrid=True,
        gridcolor='#d0d0d0',
        gridwidth=1,
        linecolor='#1a1a1a',
        linewidth=2,
        showline=True,
        mirror=True,
        zeroline=False
    )

    # Also update the main timeline x-axis
    fig.update_xaxes(
        title=dict(text='<b style="color:#1a1a1a;">Month</b>', font=dict(size=14, color='#1a1a1a')),
        row=1, col=1,
        tickfont=dict(color='#1a1a1a', size=11),
        showgrid=True,
        gridcolor='#d0d0d0',
        gridwidth=1,
        linecolor='#1a1a1a',
        linewidth=2,
        showline=True,
        mirror=True,
        zeroline=False
    )

    # Clean x-axis for main chart
    fig.update_xaxes(
        row=1, col=1,
        showgrid=True,
        gridcolor='rgba(128,128,128,0.1)',
        zeroline=False
    )

    return fig

def generate_ai_insights(model_info):
    """Generate AI insights for the specific model data"""
    data = model_info['data']
    model_name = model_info['model']
    segment = model_info['segment']

    if data.empty:
        return "No data available for analysis."

    try:
        # Prepare data summary for AI analysis
        total_sales = data['Sales'].sum()
        avg_sales = data['Sales'].mean()
        max_sales = data['Sales'].max()
        min_sales = data['Sales'].min()

        # Find outliers (using IQR method)
        Q1 = data['Sales'].quantile(0.25)
        Q3 = data['Sales'].quantile(0.75)
        IQR = Q3 - Q1
        outlier_threshold_low = Q1 - 1.5 * IQR
        outlier_threshold_high = Q3 + 1.5 * IQR

        outliers = data[(data['Sales'] < outlier_threshold_low) | (data['Sales'] > outlier_threshold_high)]

        # Peak and low performance months
        peak_month = data.loc[data['Sales'].idxmax()]
        low_month = data.loc[data['Sales'].idxmin()]

        # Media spend analysis
        total_media = data['Media_Spend'].sum()
        avg_media = data['Media_Spend'].mean()

        # Advanced Offer Analysis
        offer_months = data[data['Has_Offer'] == True]
        no_offer_months = data[data['Has_Offer'] == False]
        total_months = len(data)

        if len(offer_months) > 0 and len(no_offer_months) > 0:
            avg_sales_with_offers = offer_months['Sales'].mean()
            avg_sales_without_offers = no_offer_months['Sales'].mean()
            offer_impact = ((avg_sales_with_offers - avg_sales_without_offers) / avg_sales_without_offers * 100) if avg_sales_without_offers > 0 else 0
        else:
            avg_sales_with_offers = offer_months['Sales'].mean() if len(offer_months) > 0 else 0
            avg_sales_without_offers = no_offer_months['Sales'].mean() if len(no_offer_months) > 0 else 0
            offer_impact = 0

        # Detailed offer performance analysis
        offer_performance = {}
        if len(offer_months) > 0:
            # Split individual offers from combined offers (e.g., "Gift Card + Servicing")
            all_individual_offers = []
            for idx, row in offer_months.iterrows():
                if row['Offer']:
                    # Split by + and ; to get individual offer types
                    individual_offers = []
                    for offer_part in row['Offer'].replace(' + ', ';').split(';'):
                        individual_offers.append(offer_part.strip())

                    for individual_offer in individual_offers:
                        if individual_offer:
                            all_individual_offers.append({
                                'Offer_Type': individual_offer,
                                'Sales': row['Sales'],
                                'Month': row['Month'],
                                'Media_Spend': row['Media_Spend']
                            })

            # Analyze performance by offer type
            if all_individual_offers:
                offers_df = pd.DataFrame(all_individual_offers)
                offer_performance = offers_df.groupby('Offer_Type').agg({
                    'Sales': ['mean', 'max', 'count'],
                    'Media_Spend': 'mean'
                }).round(0)

                # Flatten column names
                offer_performance.columns = ['_'.join(col).strip() for col in offer_performance.columns]
                offer_performance = offer_performance.reset_index()

                # Sort by average sales performance
                offer_performance = offer_performance.sort_values('Sales_mean', ascending=False)

        # Create analysis prompt
        data_summary = f"""
        Hyundai {model_name} Performance Analysis ({segment})

        SALES PERFORMANCE:
        - Total Sales (24 months): {total_sales:,.0f} units
        - Average Monthly Sales: {avg_sales:.0f} units
        - Peak Month: {peak_month['Month']} with {peak_month['Sales']:,.0f} units
        - Lowest Month: {low_month['Month']} with {low_month['Sales']:,.0f} units
        - Sales Range: {min_sales:.0f} to {max_sales:.0f} units

        OUTLIERS & ANOMALIES:
        - Statistical Outliers Found: {len(outliers)} months
        {f"- Outlier Details: {outliers[['Month', 'Sales']].to_string(index=False)}" if len(outliers) > 0 else "- No significant statistical outliers"}

        MEDIA INVESTMENT:
        - Total Media Spend: ${total_media:,.0f}
        - Average Monthly Spend: ${avg_media:.0f}
        - Peak Media Month: {data.loc[data['Media_Spend'].idxmax(), 'Month']} (${data['Media_Spend'].max():,.0f})

        OFFER EFFECTIVENESS:
        - Months with Offers: {len(offer_months)} out of 24
        - Average Sales with Offers: {avg_sales_with_offers:.0f} units
        - Average Sales without Offers: {avg_sales_without_offers:.0f} units
        - Offer Impact: {offer_impact:+.1f}% sales difference

        OFFER TYPE PERFORMANCE RANKING:
        {offer_performance.to_string(index=False) if len(offer_performance) > 0 else "No detailed offer performance data available"}

        MONTHLY DATA SAMPLE:
        {data[['Month', 'Sales', 'Media_Spend', 'Offer']].head(8).to_string(index=False)}
        ...
        {data[['Month', 'Sales', 'Media_Spend', 'Offer']].tail(4).to_string(index=False)}
        """

        prompt = f"""
        As an automotive market analyst, provide a comprehensive analysis of this Hyundai {model_name} performance data.

        {data_summary}

        Please provide insights on:

        1. **PERFORMANCE PATTERNS**: Identify key trends, seasonal patterns, and performance cycles

        2. **OUTLIERS & ANOMALIES**: Explain any unusual spikes, drops, or irregular patterns in sales

        3. **OFFER PERFORMANCE ANALYSIS**:
           - Offer Penetration Rate: {len(offer_months)}/{total_months} months = {(len(offer_months)/total_months*100):.1f}% - assess if this is optimal or if there are missed opportunities
           - Analyze which specific offer types (Finance, Driveaway, Factory Bonus, Gift Card, Servicing, etc.) correlate with highest sales
           - Identify the best-performing individual offer types based on the performance ranking data
           - Explain why certain offers might be more effective than others for this model/segment
           - Comment on offer combination effectiveness (e.g., Gift Card + Servicing)

        4. **MEDIA SPEND EFFICIENCY**: Assess the relationship between marketing investment and sales results

        5. **COMPETITIVE POSITIONING**: Based on the segment ({segment}), comment on market position and performance

        6. **STRATEGIC RECOMMENDATIONS**:
           - Provide 2-3 actionable insights for improving {model_name} performance
           - Focus specifically on offer strategy optimization based on the performance data

        Focus on data-driven insights, specific examples from the data, and practical business implications.
        Keep the analysis concise but detailed, around 300-400 words.
        """

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"AI analysis temporarily unavailable. Error: {str(e)}"

def show_model_summary(model_info):
    """Show key metrics for the model"""
    data = model_info['data']
    model_name = model_info['model']

    if data.empty:
        st.error(f"No data available for {model_name}")
        return

    # Calculate metrics
    total_sales = data['Sales'].sum()
    avg_sales = data['Sales'].mean()
    total_media = data['Media_Spend'].sum()
    months_with_offers = data['Has_Offer'].sum()
    total_months = len(data)

    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Sales", f"{total_sales:,.0f}")
    with col2:
        st.metric("Avg Monthly Sales", f"{avg_sales:.0f}")
    with col3:
        st.metric("Total Media Spend", f"${total_media:,.0f}")
    with col4:
        st.metric("Months with Offers", f"{months_with_offers}/{total_months}")

# Main App
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

def generate_offer_ai_insights(df_offers, analysis_type="market", selected_filter=None, selected_makes=None):
    """Generate AI insights for offer effectiveness analysis"""
    try:
        # Prepare data summary for AI analysis
        if analysis_type == "market" and selected_makes:
            # Multi-make analysis
            filtered_data = df_offers[df_offers['Make'].isin(selected_makes)]
            filter_desc = f"Makes: {', '.join(selected_makes)}"
        elif analysis_type == "market" and selected_filter and selected_filter != 'All Makes':
            filtered_data = df_offers[df_offers['Make'] == selected_filter]
            filter_desc = f"Make: {selected_filter}"
        elif analysis_type == "segment" and selected_filter and selected_filter != 'All Segments':
            filtered_data = df_offers[df_offers['Segment'] == selected_filter]
            filter_desc = f"Segment: {selected_filter}"
        else:
            filtered_data = df_offers
            filter_desc = "All data"

        # Calculate key metrics
        total_sales = filtered_data['Sales_Total'].sum()
        unique_offers = filtered_data['Retail_Offer'].nunique()
        date_range = f"{filtered_data['Month'].min()} to {filtered_data['Month'].max()}"

        # Top offers by total sales
        offer_performance = {}
        for _, row in filtered_data.iterrows():
            offers = parse_combined_offers(row['Retail_Offer'])
            for offer in offers:
                if offer not in offer_performance:
                    offer_performance[offer] = {'total_sales': 0, 'months': set()}
                offer_performance[offer]['total_sales'] += row['Sales_Total']
                offer_performance[offer]['months'].add(row['Month'])

        # Sort offers by performance
        top_offers = sorted(offer_performance.items(), key=lambda x: x[1]['total_sales'], reverse=True)[:10]

        # Make/Segment performance if multi-selection
        entity_performance = ""
        if analysis_type == "market" and selected_makes:
            make_totals = filtered_data.groupby('Make')['Sales_Total'].sum().sort_values(ascending=False)
            entity_performance = f"\n\n**Make Performance Ranking:**\n"
            for i, (make, sales) in enumerate(make_totals.items(), 1):
                entity_performance += f"{i}. {make}: {sales:,} total sales\n"

        # Cash rate context
        cash_rates = filtered_data['Cash_Rate_Percent'].unique()
        cash_rate_range = f"{min(cash_rates):.1f}% to {max(cash_rates):.1f}%" if len(cash_rates) > 1 else f"{cash_rates[0]:.1f}%"

        # Create comprehensive data summary
        data_summary = f"""
        **{analysis_type.title()} Offer Effectiveness Analysis**

        **Filter Applied:** {filter_desc}
        **Analysis Period:** {date_range}
        **Total Sales Volume:** {total_sales:,} units
        **Unique Offer Types:** {unique_offers}
        **Cash Rate Range:** {cash_rate_range}

        **Top 10 Performing Offers by Sales Volume:**
        """

        for i, (offer, data) in enumerate(top_offers, 1):
            months_active = len(data['months'])
            avg_monthly = data['total_sales'] / months_active if months_active > 0 else 0
            data_summary += f"{i}. {offer}: {data['total_sales']:,} total sales ({months_active} months active, {avg_monthly:.0f} avg/month)\n"

        data_summary += entity_performance

        # Monthly trends
        monthly_totals = filtered_data.groupby('Month')['Sales_Total'].sum().sort_index()
        peak_month = monthly_totals.idxmax()
        lowest_month = monthly_totals.idxmin()
        data_summary += f"\n**Peak Performance:** {peak_month} ({monthly_totals[peak_month]:,} sales)"
        data_summary += f"\n**Lowest Performance:** {lowest_month} ({monthly_totals[lowest_month]:,} sales)"

        # AI analysis prompt
        prompt = f"""
        Analyze this automotive offer effectiveness data and provide strategic insights:

        {data_summary}

        Please provide a comprehensive analysis covering:
        1. **Offer Performance Insights** - Which offer types are most/least effective and why
        2. **Seasonal Trends** - Notable patterns in timing and performance
        3. **Market Dynamics** - How cash rate changes may have impacted offer effectiveness
        4. **Strategic Recommendations** - Specific actionable insights for improving offer strategy
        5. **Competitive Positioning** - Analysis of offer mix and market positioning

        Focus on practical insights that would help a CMO optimize their offer strategy.
        Use bullet points and clear sections. Be specific about numbers and trends.
        """

        # Generate AI response
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Analysis temporarily unavailable. Error: {str(e)}"

def create_market_offer_effectiveness_chart(df_offers, selected_make=None, selected_makes=None):
    """Create market-level offer effectiveness chart"""

    # Load make-level media spend data
    media_spend_dict = load_make_level_media_spend()

    # Filter by make(s) if selected
    if selected_makes:
        df_filtered = df_offers[df_offers['Make'].isin(selected_makes)]
        title_suffix = f" - {', '.join(selected_makes[:3])}{'...' if len(selected_makes) > 3 else ''}"
        makes_to_aggregate = selected_makes
    elif selected_make and selected_make != 'All Makes':
        df_filtered = df_offers[df_offers['Make'] == selected_make]
        title_suffix = f" - {selected_make}"
        makes_to_aggregate = [selected_make]
    else:
        df_filtered = df_offers
        title_suffix = " - All Makes"
        makes_to_aggregate = df_offers['Make'].unique().tolist()

    # Get all offer types
    offer_types = get_all_offer_types(df_filtered)

    # Create monthly aggregation for each offer type and media spend
    monthly_data = {}
    monthly_media_spend = {}

    for month in df_filtered['Month'].unique():
        monthly_data[month] = {}

        # Calculate total media spend for the selected makes from make-level data
        total_media = 0
        for make in makes_to_aggregate:
            if make in media_spend_dict and month in media_spend_dict[make]:
                total_media += media_spend_dict[make][month]

        monthly_media_spend[month] = total_media

        # Calculate sales by offer type
        month_data = df_filtered[df_filtered['Month'] == month]
        for offer_type in offer_types:
            total_sales = 0

            for _, row in month_data.iterrows():
                if offer_type in parse_combined_offers(row['Retail_Offer']):
                    total_sales += row['Sales_Total']

            monthly_data[month][offer_type] = total_sales

    # Create the chart with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Sort months chronologically
    month_order = ['Aug-23', 'Sep-23', 'Oct-23', 'Nov-23', 'Dec-23',
                   'Jan-24', 'Feb-24', 'Mar-24', 'Apr-24', 'May-24', 'Jun-24', 'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24', 'Nov-24', 'Dec-24',
                   'Jan-25', 'Feb-25', 'Mar-25', 'Apr-25', 'May-25', 'Jun-25', 'Jul-25', 'Aug-25', 'Sep-25']

    sorted_months = [month for month in month_order if month in monthly_data.keys()]

    # Enhanced color palette with better contrast and visibility
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7', '#F72585', '#4361EE', '#F77F00', '#FCBF49',
              '#06FFA5', '#FB8500', '#219EBC', '#8ECAE6', '#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#E0BBE4', '#957DAD']

    # Add traces for each offer type (primary y-axis)
    for i, offer_type in enumerate(offer_types[:15]):  # Limit to top 15 for readability
        sales_data = [monthly_data[month].get(offer_type, 0) for month in sorted_months]

        # Only add if there's meaningful data (avoid flat zero lines)
        if max(sales_data) > 0:
            fig.add_trace(go.Scatter(
                x=sorted_months,
                y=sales_data,
                mode='lines+markers',
                name=offer_type,
                line=dict(width=4, color=colors[i % len(colors)], shape='spline'),
                marker=dict(size=8, color=colors[i % len(colors)], line=dict(width=2, color='white')),
                hovertemplate=f'<b>{offer_type}</b><br>Month: %{{x}}<br>Sales Volume: <b>%{{y:,}}</b><extra></extra>'
            ), secondary_y=False)

    # Add media spend line (secondary y-axis)
    media_spend_data = [monthly_media_spend.get(month, 0) for month in sorted_months]

    if max(media_spend_data) > 0:
        fig.add_trace(go.Scatter(
            x=sorted_months,
            y=media_spend_data,
            mode='lines',
            name='Total Media Spend',
            line=dict(width=3, color='#000000', dash='dot'),
            opacity=0.8,
            hovertemplate='<b>Media Spend</b><br>Month: %{x}<br>Total: <b>$%{y:,.0f}</b><extra></extra>'
        ), secondary_y=True)

    fig.update_layout(
        title=dict(
            text=f'<b style="color:#1a1a1a; font-size:22px;">Market Offer Effectiveness Over Time{title_suffix}</b><br><span style="color:#4a4a4a; font-size:14px;">Sales Volume Driven by Each Offer Type</span>',
            x=0.5,
            xanchor='center',
            font=dict(color='#1a1a1a')
        ),
        xaxis=dict(
            title=dict(text='<b>Month</b>', font=dict(size=14, color='#1a1a1a')),
            tickfont=dict(color='#1a1a1a', size=11),
            showgrid=True,
            gridwidth=1,
            gridcolor='#d0d0d0',
            linecolor='#1a1a1a',
            linewidth=2,
            showline=True,
            mirror=True
        ),
        font=dict(family="Arial", size=12, color='#1a1a1a'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=650,
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='white',
            bordercolor='#1a1a1a',
            font=dict(color='#1a1a1a')
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor='white',
            bordercolor='#1a1a1a',
            borderwidth=2,
            font=dict(color='#1a1a1a', size=12, family='Arial')
        ),
        margin=dict(l=80, r=180, t=120, b=80)
    )

    # Set y-axes titles
    fig.update_yaxes(
        title=dict(text="<b>Sales Volume</b>", font=dict(size=14, color='#1a1a1a')),
        secondary_y=False,
        tickfont=dict(color='#1a1a1a', size=11),
        showgrid=True,
        gridwidth=1,
        gridcolor='#d0d0d0',
        linecolor='#1a1a1a',
        linewidth=2,
        showline=True,
        mirror=True
    )

    fig.update_yaxes(
        title=dict(text="<b>Media Spend ($)</b>", font=dict(size=14, color='#1a1a1a')),
        secondary_y=True,
        tickfont=dict(color='#1a1a1a', size=11),
        showgrid=False,
        linecolor='#1a1a1a',
        linewidth=2,
        showline=True,
        mirror=True,
        tickformat='$,.0f'
    )

    return fig

def create_website_visits_chart(web_data, media_spend_dict, monthly_offers):
    """Create chart showing total website visits vs Hyundai media spend with offers"""

    # Get Hyundai media spend
    hyundai_media = media_spend_dict.get('Hyundai', {})

    # Month order for proper sorting
    month_order = ['Aug-23', 'Sep-23', 'Oct-23', 'Nov-23', 'Dec-23',
                   'Jan-24', 'Feb-24', 'Mar-24', 'Apr-24', 'May-24', 'Jun-24', 'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24', 'Nov-24', 'Dec-24',
                   'Jan-25', 'Feb-25', 'Mar-25', 'Apr-25', 'May-25', 'Jun-25', 'Jul-25', 'Aug-25', 'Sep-25']

    # Sort web_data by month order
    web_data['Month_Order'] = web_data['Month'].map({month: i for i, month in enumerate(month_order)})
    web_data_sorted = web_data.sort_values('Month_Order').drop('Month_Order', axis=1)

    # Prepare data for chart
    months = []
    visits = []
    media_spend = []
    dates = []

    for _, row in web_data_sorted.iterrows():
        month = row['Month']
        months.append(month)
        visits.append(row['Total_Visits'])
        media_spend.append(hyundai_media.get(month, 0))
        # Convert month to date
        try:
            dates.append(pd.to_datetime(month, format='%b-%y'))
        except:
            dates.append(None)

    # Create chart with secondary y-axis and 2 rows
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.1,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # Add website visits trace (primary y-axis - left) - ROW 1
    fig.add_trace(go.Scatter(
        x=months,
        y=visits,
        mode='lines+markers',
        name='Total Website Visits',
        line=dict(color='#2E86AB', width=4, shape='spline'),
        marker=dict(size=8, color='#2E86AB', line=dict(width=2, color='white')),
        hovertemplate='<b>%{x}</b><br>Total Visits: <b>%{y:,}</b><extra></extra>'
    ), row=1, col=1, secondary_y=False)

    # Add media spend trace (secondary y-axis - right) - ROW 1
    fig.add_trace(go.Scatter(
        x=months,
        y=media_spend,
        mode='lines',
        name='Hyundai Media Spend',
        line=dict(width=3, color='#000000', dash='dot'),
        opacity=0.8,
        hovertemplate='<b>%{x}</b><br>Media Spend: <b>$%{y:,.0f}</b><extra></extra>'
    ), row=1, col=1, secondary_y=True)

    # Add vertical lines for months with offers
    offer_months_list = []
    offer_visits_list = []
    offer_text_list = []

    for i, month in enumerate(months):
        if month in monthly_offers and monthly_offers[month]['has_offer']:
            # Add vertical line
            date_val = dates[i] if dates[i] else month
            fig.add_vline(
                x=month,
                line_width=1,
                line_dash="dash",
                line_color="rgba(241, 143, 1, 0.3)",
                row=1, col=1
            )
            offer_months_list.append(month)
            offer_visits_list.append(visits[i])
            offer_text_list.append(monthly_offers[month]['offer_display'])

    # Add offer markers on the main chart
    if offer_months_list:
        fig.add_trace(go.Scatter(
            x=offer_months_list,
            y=offer_visits_list,
            mode='markers',
            name='Active Offers',
            marker=dict(
                size=12,
                color='#F18F01',
                symbol='circle',
                line=dict(width=2, color='#ffffff'),
                opacity=0.9
            ),
            hovertemplate='<b>%{x}</b><br>Visits: %{y:,}<br>Offer: %{text}<extra></extra>',
            text=offer_text_list,
            showlegend=False  # Hide from legend to reduce clutter
        ), row=1, col=1, secondary_y=False)

    # Add offer timeline in bottom subplot (ROW 2)
    offer_data = [(month, monthly_offers[month]) for month in months if month in monthly_offers and monthly_offers[month]['has_offer']]

    if offer_data:
        unique_offers = list(set([item[1]['offer_display'] for item in offer_data]))
        offer_mapping = {offer: i+1 for i, offer in enumerate(unique_offers)}

        for offer_type in unique_offers:
            offer_subset_months = [item[0] for item in offer_data if item[1]['offer_display'] == offer_type]
            offer_subset_y = [offer_mapping[offer_type]] * len(offer_subset_months)

            fig.add_trace(
                go.Scatter(
                    x=offer_subset_months,
                    y=offer_subset_y,
                    mode='markers+text',
                    name=offer_type,
                    marker=dict(size=12, symbol='square'),
                    text=offer_subset_months,
                    textposition="middle center",
                    textfont=dict(size=10),
                    hovertemplate=f'<b>%{{x}}</b><br>Offer: {offer_type}<extra></extra>',
                    showlegend=False  # Hide from legend to reduce clutter
                ),
                row=2, col=1
            )

    fig.update_layout(
        title=dict(
            text='<b style="color:#1a1a1a; font-size:22px;">Total Website Visits vs Hyundai Media Spend</b><br><span style="color:#4a4a4a; font-size:14px;">Monthly Trends with Offer Overlay</span>',
            x=0.5,
            xanchor='center',
            font=dict(color='#1a1a1a')
        ),
        font=dict(family="Arial", size=12, color='#1a1a1a'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=750,
        hovermode='x unified',
        showlegend=True,
        hoverlabel=dict(
            bgcolor='white',
            bordercolor='#1a1a1a',
            font=dict(color='#1a1a1a')
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='white',
            bordercolor='#1a1a1a',
            borderwidth=2,
            font=dict(color='#1a1a1a', size=11)
        ),
        margin=dict(l=80, r=120, t=120, b=80)
    )

    # Update main chart axes (Row 1)
    fig.update_xaxes(
        title=dict(text='<b>Month</b>', font=dict(size=14, color='#1a1a1a')),
        tickfont=dict(color='#1a1a1a', size=11),
        showgrid=True,
        gridwidth=1,
        gridcolor='#d0d0d0',
        linecolor='#1a1a1a',
        linewidth=2,
        showline=True,
        mirror=True,
        row=1, col=1
    )

    fig.update_yaxes(
        title=dict(text="<b>Website Visits</b>", font=dict(size=14, color='#1a1a1a')),
        secondary_y=False,
        tickfont=dict(color='#1a1a1a', size=11),
        showgrid=True,
        gridwidth=1,
        gridcolor='#d0d0d0',
        linecolor='#1a1a1a',
        linewidth=2,
        showline=True,
        mirror=True,
        tickformat=',d',
        row=1, col=1
    )

    fig.update_yaxes(
        title=dict(text="<b>Media Spend ($)</b>", font=dict(size=14, color='#1a1a1a')),
        secondary_y=True,
        tickfont=dict(color='#1a1a1a', size=11),
        showgrid=False,
        linecolor='#1a1a1a',
        linewidth=2,
        showline=True,
        mirror=True,
        tickformat='$,.0f',
        row=1, col=1
    )

    # Update offer timeline axes (Row 2)
    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        linecolor='#1a1a1a',
        linewidth=2,
        showline=True,
        row=2, col=1
    )

    fig.update_yaxes(
        title=dict(text="<b>Offers</b>", font=dict(size=12, color='#1a1a1a')),
        tickfont=dict(color='#1a1a1a', size=10),
        showgrid=False,
        showticklabels=False,
        linecolor='#1a1a1a',
        linewidth=2,
        showline=True,
        row=2, col=1
    )

    return fig

def create_offers_page_views_chart(web_data, media_spend_dict, monthly_offers):
    """Create chart showing offers page views vs Hyundai media spend with offers"""

    # Get Hyundai media spend
    hyundai_media = media_spend_dict.get('Hyundai', {})

    # Month order for proper sorting
    month_order = ['Aug-23', 'Sep-23', 'Oct-23', 'Nov-23', 'Dec-23',
                   'Jan-24', 'Feb-24', 'Mar-24', 'Apr-24', 'May-24', 'Jun-24', 'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24', 'Nov-24', 'Dec-24',
                   'Jan-25', 'Feb-25', 'Mar-25', 'Apr-25', 'May-25', 'Jun-25', 'Jul-25', 'Aug-25', 'Sep-25']

    # Sort web_data by month order
    web_data['Month_Order'] = web_data['Month'].map({month: i for i, month in enumerate(month_order)})
    web_data_sorted = web_data.sort_values('Month_Order').drop('Month_Order', axis=1)

    # Prepare data for chart
    months = []
    page_views = []
    media_spend = []
    dates = []

    for _, row in web_data_sorted.iterrows():
        month = row['Month']
        months.append(month)
        page_views.append(row['Offers_Page_Views'])
        media_spend.append(hyundai_media.get(month, 0))
        # Convert month to date
        try:
            dates.append(pd.to_datetime(month, format='%b-%y'))
        except:
            dates.append(None)

    # Create chart with secondary y-axis and 2 rows
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.1,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # Add offers page views trace (primary y-axis - left) - ROW 1
    fig.add_trace(go.Scatter(
        x=months,
        y=page_views,
        mode='lines+markers',
        name='Offers Page Views',
        line=dict(color='#F18F01', width=4, shape='spline'),
        marker=dict(size=8, color='#F18F01', line=dict(width=2, color='white')),
        hovertemplate='<b>%{x}</b><br>Offers Page Views: <b>%{y:,}</b><extra></extra>'
    ), row=1, col=1, secondary_y=False)

    # Add media spend trace (secondary y-axis - right) - ROW 1
    fig.add_trace(go.Scatter(
        x=months,
        y=media_spend,
        mode='lines',
        name='Hyundai Media Spend',
        line=dict(width=3, color='#000000', dash='dot'),
        opacity=0.8,
        hovertemplate='<b>%{x}</b><br>Media Spend: <b>$%{y:,.0f}</b><extra></extra>'
    ), row=1, col=1, secondary_y=True)

    # Add vertical lines for months with offers
    offer_months_list = []
    offer_page_views_list = []
    offer_text_list = []

    for i, month in enumerate(months):
        if month in monthly_offers and monthly_offers[month]['has_offer']:
            # Add vertical line
            fig.add_vline(
                x=month,
                line_width=1,
                line_dash="dash",
                line_color="rgba(241, 143, 1, 0.3)",
                row=1, col=1
            )
            offer_months_list.append(month)
            offer_page_views_list.append(page_views[i])
            offer_text_list.append(monthly_offers[month]['offer_display'])

    # Add offer markers on the main chart
    if offer_months_list:
        fig.add_trace(go.Scatter(
            x=offer_months_list,
            y=offer_page_views_list,
            mode='markers',
            name='Active Offers',
            marker=dict(
                size=12,
                color='#2E86AB',
                symbol='circle',
                line=dict(width=2, color='#ffffff'),
                opacity=0.9
            ),
            hovertemplate='<b>%{x}</b><br>Page Views: %{y:,}<br>Offer: %{text}<extra></extra>',
            text=offer_text_list,
            showlegend=False  # Hide from legend to reduce clutter
        ), row=1, col=1, secondary_y=False)

    # Add offer timeline in bottom subplot (ROW 2)
    offer_data = [(month, monthly_offers[month]) for month in months if month in monthly_offers and monthly_offers[month]['has_offer']]

    if offer_data:
        unique_offers = list(set([item[1]['offer_display'] for item in offer_data]))
        offer_mapping = {offer: i+1 for i, offer in enumerate(unique_offers)}

        for offer_type in unique_offers:
            offer_subset_months = [item[0] for item in offer_data if item[1]['offer_display'] == offer_type]
            offer_subset_y = [offer_mapping[offer_type]] * len(offer_subset_months)

            fig.add_trace(
                go.Scatter(
                    x=offer_subset_months,
                    y=offer_subset_y,
                    mode='markers+text',
                    name=offer_type,
                    marker=dict(size=12, symbol='square'),
                    text=offer_subset_months,
                    textposition="middle center",
                    textfont=dict(size=10),
                    hovertemplate=f'<b>%{{x}}</b><br>Offer: {offer_type}<extra></extra>',
                    showlegend=False  # Hide from legend to reduce clutter
                ),
                row=2, col=1
            )

    fig.update_layout(
        title=dict(
            text='<b style="color:#1a1a1a; font-size:22px;">Offers Page Views vs Hyundai Media Spend</b><br><span style="color:#4a4a4a; font-size:14px;">Monthly Trends with Offer Overlay</span>',
            x=0.5,
            xanchor='center',
            font=dict(color='#1a1a1a')
        ),
        font=dict(family="Arial", size=12, color='#1a1a1a'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=750,
        hovermode='x unified',
        showlegend=True,
        hoverlabel=dict(
            bgcolor='white',
            bordercolor='#1a1a1a',
            font=dict(color='#1a1a1a')
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='white',
            bordercolor='#1a1a1a',
            borderwidth=2,
            font=dict(color='#1a1a1a', size=11)
        ),
        margin=dict(l=80, r=120, t=120, b=80)
    )

    # Update main chart axes (Row 1)
    fig.update_xaxes(
        title=dict(text='<b>Month</b>', font=dict(size=14, color='#1a1a1a')),
        tickfont=dict(color='#1a1a1a', size=11),
        showgrid=True,
        gridwidth=1,
        gridcolor='#d0d0d0',
        linecolor='#1a1a1a',
        linewidth=2,
        showline=True,
        mirror=True,
        row=1, col=1
    )

    fig.update_yaxes(
        title=dict(text="<b>Offers Page Views</b>", font=dict(size=14, color='#1a1a1a')),
        secondary_y=False,
        tickfont=dict(color='#1a1a1a', size=11),
        showgrid=True,
        gridwidth=1,
        gridcolor='#d0d0d0',
        linecolor='#1a1a1a',
        linewidth=2,
        showline=True,
        mirror=True,
        tickformat=',d',
        row=1, col=1
    )

    fig.update_yaxes(
        title=dict(text="<b>Media Spend ($)</b>", font=dict(size=14, color='#1a1a1a')),
        secondary_y=True,
        tickfont=dict(color='#1a1a1a', size=11),
        showgrid=False,
        linecolor='#1a1a1a',
        linewidth=2,
        showline=True,
        mirror=True,
        tickformat='$,.0f',
        row=1, col=1
    )

    # Update offer timeline axes (Row 2)
    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        linecolor='#1a1a1a',
        linewidth=2,
        showline=True,
        row=2, col=1
    )

    fig.update_yaxes(
        title=dict(text="<b>Offers</b>", font=dict(size=12, color='#1a1a1a')),
        tickfont=dict(color='#1a1a1a', size=10),
        showgrid=False,
        showticklabels=False,
        linecolor='#1a1a1a',
        linewidth=2,
        showline=True,
        row=2, col=1
    )

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

    # Create the chart (no secondary y-axis for segments)
    fig = go.Figure()

    # Sort months chronologically
    month_order = ['Aug-23', 'Sep-23', 'Oct-23', 'Nov-23', 'Dec-23',
                   'Jan-24', 'Feb-24', 'Mar-24', 'Apr-24', 'May-24', 'Jun-24', 'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24', 'Nov-24', 'Dec-24',
                   'Jan-25', 'Feb-25', 'Mar-25', 'Apr-25', 'May-25', 'Jun-25', 'Jul-25', 'Aug-25', 'Sep-25']

    sorted_months = [month for month in month_order if month in monthly_data.keys()]

    # Enhanced color palette with better contrast and visibility
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7', '#F72585', '#4361EE', '#F77F00', '#FCBF49',
              '#06FFA5', '#FB8500', '#219EBC', '#8ECAE6', '#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#E0BBE4', '#957DAD']

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
                line=dict(width=4, color=colors[i % len(colors)], shape='spline'),
                marker=dict(size=8, color=colors[i % len(colors)], line=dict(width=2, color='white')),
                hovertemplate=f'<b>{offer_type}</b><br>Month: %{{x}}<br>Sales Volume: <b>%{{y:,}}</b><extra></extra>'
            ))

    fig.update_layout(
        title=dict(
            text=f'<b style="color:#1a1a1a; font-size:22px;">Segment Offer Effectiveness Over Time{title_suffix}</b><br><span style="color:#4a4a4a; font-size:14px;">Sales Volume Driven by Each Offer Type</span>',
            x=0.5,
            xanchor='center',
            font=dict(color='#1a1a1a')
        ),
        xaxis=dict(
            title=dict(text='<b>Month</b>', font=dict(size=14, color='#1a1a1a')),
            tickfont=dict(color='#1a1a1a', size=11),
            showgrid=True,
            gridwidth=1,
            gridcolor='#d0d0d0',
            linecolor='#1a1a1a',
            linewidth=2,
            showline=True,
            mirror=True
        ),
        yaxis=dict(
            title=dict(text='<b>Sales Volume</b>', font=dict(size=14, color='#1a1a1a')),
            tickfont=dict(color='#1a1a1a', size=11),
            showgrid=True,
            gridwidth=1,
            gridcolor='#d0d0d0',
            linecolor='#1a1a1a',
            linewidth=2,
            showline=True,
            mirror=True
        ),
        font=dict(family="Arial", size=12, color='#1a1a1a'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=650,
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='white',
            bordercolor='#1a1a1a',
            font=dict(color='#1a1a1a')
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor='white',
            bordercolor='#1a1a1a',
            borderwidth=2,
            font=dict(color='#1a1a1a', size=12, family='Arial')
        ),
        margin=dict(l=80, r=180, t=120, b=80)
    )

    return fig

# Enhanced CSS for better readability
st.markdown("""
<style>
    /* Main app styling */
    .main .block-container {
        padding-top: 2rem;
        color: #2c3e50;
    }

    /* Headers */
    h1, h2, h3 {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* Metrics styling */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e8e8e8;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    [data-testid="metric-container"] > div {
        color: #2c3e50 !important;
    }

    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: white;
        color: #2c3e50;
    }

    /* Button styling */
    .stButton > button {
        background-color: #3498db;
        color: white;
        border: none;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #2980b9;
    }

    /* Plot container */
    .js-plotly-plot {
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Load both datasets
    df = load_data()  # Original Hyundai data
    df_comprehensive, df_offers = load_comprehensive_data()  # Comprehensive market data

    # Enhanced navigation with individual buttons
    st.sidebar.title("🚗 Dashboard Navigation")
    st.sidebar.markdown("Choose your analysis type:")

    # Initialize session state for page selection if not exists
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Hyundai Model Analysis"

    # Individual navigation buttons
    if st.sidebar.button("🏠 **Hyundai Model Analysis**", key="hyundai_btn", use_container_width=True):
        st.session_state.current_page = "Hyundai Model Analysis"
    st.sidebar.caption("📈 Individual model performance timelines with AI insights")
    st.sidebar.markdown("")

    if st.sidebar.button("📊 **Market Offer Effectiveness**", key="market_btn", use_container_width=True):
        st.session_state.current_page = "Market Offer Effectiveness"
    st.sidebar.caption("🌍 Compare offer effectiveness across all automotive brands")
    st.sidebar.markdown("")

    if st.sidebar.button("🎯 **Segment Offer Effectiveness**", key="segment_btn", use_container_width=True):
        st.session_state.current_page = "Segment Offer Effectiveness"
    st.sidebar.caption("🚗 Analyze offer performance within vehicle segments")
    st.sidebar.markdown("")

    if st.sidebar.button("🌐 **Website Analytics**", key="website_btn", use_container_width=True):
        st.session_state.current_page = "Website Analytics"
    st.sidebar.caption("📊 Hyundai website traffic vs media spend correlation")

    # Get the selected page
    page = st.session_state.current_page

    # Market Offer Effectiveness Page
    if page == "Market Offer Effectiveness":
        st.title("📈 Market Offer Effectiveness Analysis")
        st.markdown("Analyze how different offer types drive sales volumes across the entire automotive market")

        # Filter container
        col1, col2 = st.columns([2, 1])

        with col1:
            # Make filter - allow multiple selection
            st.write("**Filter by Make:**")
            col1a, col1b = st.columns([1, 1])
            with col1a:
                all_makes = sorted(df_offers['Make'].unique())
                selected_makes = st.multiselect("Select Makes (leave empty for all):", all_makes, key="market_makes_filter")
            with col1b:
                if selected_makes:
                    # Show ranking option when multiple makes selected
                    show_ranking = st.checkbox("Show Make Rankings", value=True, key="show_ranking")
                else:
                    show_ranking = False

            # Fallback to single selection if no multi-select
            selected_make = None
            if not selected_makes:
                single_makes = ['All Makes'] + all_makes
                selected_make = st.selectbox("Or select single make:", single_makes, key="market_single_make_filter")

        with col2:
            st.metric("Total Records", f"{len(df_offers):,}")

        # Create and display chart
        fig = create_market_offer_effectiveness_chart(df_offers, selected_make, selected_makes)
        st.plotly_chart(fig, use_container_width=True)

        # Show make rankings if requested
        if selected_makes and show_ranking:
            st.subheader("📈 Make Performance Rankings")
            ranking_data = df_offers[df_offers['Make'].isin(selected_makes)].groupby('Make')['Sales_Total'].sum().sort_values(ascending=False)

            cols = st.columns(len(selected_makes))
            for i, (make, total_sales) in enumerate(ranking_data.items()):
                with cols[i % len(cols)]:
                    st.metric(f"#{i+1} {make}", f"{total_sales:,}", help="Total sales volume")

        # Summary metrics
        st.subheader("📊 Market Summary")

        col1, col2, col3, col4 = st.columns(4)

        # Filter data for metrics
        if selected_makes:
            filtered_data = df_offers[df_offers['Make'].isin(selected_makes)]
        elif selected_make and selected_make != 'All Makes':
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

        # AI Insights Section
        st.subheader("🤖 AI Market Analysis")

        col1, col2 = st.columns([1, 4])

        with col1:
            generate_market_insights = st.button(
                "Generate AI Analysis",
                type="primary",
                help="Click to generate AI insights for market offer effectiveness",
                key="market_ai_button"
            )

        with col2:
            if generate_market_insights:
                st.info("🔄 Analyzing market offer effectiveness... This may take a moment.")

        # Generate and display insights
        if generate_market_insights:
            with st.spinner("Generating AI insights for market analysis..."):
                insights = generate_offer_ai_insights(df_offers, "market", selected_make, selected_makes)

            # Display insights in an attractive container
            st.markdown("### 📊 Market Offer Analysis")

            with st.container():
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                        padding: 25px;
                        border-radius: 10px;
                        border-left: 5px solid #1f77b4;
                        border: 1px solid #e9ecef;
                        margin: 15px 0;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        color: #212529;
                        font-size: 16px;
                        line-height: 1.6;
                    ">
                    <div style="color: #212529; font-weight: 400;">
                    {insights}
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Add timestamp
            from datetime import datetime
            st.caption(f"Analysis generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")

    # Segment Offer Effectiveness Page
    elif page == "Segment Offer Effectiveness":
        st.title("🎯 Segment Offer Effectiveness Analysis")
        st.markdown("Analyze how different offer types drive sales volumes within specific vehicle segments")

        # Filter container
        col1, col2 = st.columns([2, 1])

        with col1:
            # Segment filter
            all_segments = ['All Segments'] + sorted(df_offers['Segment'].unique())
            selected_segment = st.selectbox("Filter by Segment:", all_segments, key="segment_filter")

        with col2:
            st.metric("Total Records", f"{len(df_offers):,}")

        # Create and display chart
        fig = create_segment_offer_effectiveness_chart(df_offers, selected_segment)
        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics
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

        # AI Insights Section for Segment Analysis
        st.subheader("🤖 AI Segment Analysis")

        col1, col2 = st.columns([1, 4])

        with col1:
            generate_segment_insights = st.button(
                "Generate AI Analysis",
                type="primary",
                help="Click to generate AI insights for segment offer effectiveness",
                key="segment_ai_button"
            )

        with col2:
            if generate_segment_insights:
                st.info("🔄 Analyzing segment offer effectiveness... This may take a moment.")

        # Generate and display insights
        if generate_segment_insights:
            with st.spinner("Generating AI insights for segment analysis..."):
                insights = generate_offer_ai_insights(df_offers, "segment", selected_segment)

            # Display insights in an attractive container
            st.markdown("### 📊 Segment Offer Analysis")

            with st.container():
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                        padding: 25px;
                        border-radius: 10px;
                        border-left: 5px solid #1f77b4;
                        border: 1px solid #e9ecef;
                        margin: 15px 0;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        color: #212529;
                        font-size: 16px;
                        line-height: 1.6;
                    ">
                    <div style="color: #212529; font-weight: 400;">
                    {insights}
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Add timestamp
            from datetime import datetime
            st.caption(f"Analysis generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")

    # Website Analytics Page
    elif page == "Website Analytics":
        st.title("🌐 Website Analytics Dashboard")
        st.markdown("Analyze Hyundai website traffic and offers page engagement in relation to media spend")

        # Load website and media spend data
        web_data = load_website_data()
        media_spend_dict = load_make_level_media_spend()
        monthly_offers = get_hyundai_offers_by_month(df_offers)

        # Summary metrics
        st.subheader("📊 Overall Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_visits = web_data['Total_Visits'].sum()
            st.metric("Total Website Visits", f"{total_visits:,}")

        with col2:
            total_offers_views = web_data['Offers_Page_Views'].sum()
            st.metric("Total Offers Page Views", f"{total_offers_views:,}")

        with col3:
            conversion_rate = (total_offers_views / total_visits * 100) if total_visits > 0 else 0
            st.metric("Offers Conversion Rate", f"{conversion_rate:.2f}%")

        with col4:
            hyundai_media = media_spend_dict.get('Hyundai', {})
            total_media_spend = sum(hyundai_media.values())
            st.metric("Total Hyundai Media Spend", f"${total_media_spend:,.0f}")

        st.divider()

        # Chart 1: Total Website Visits
        st.subheader("📈 Total Website Visits vs Media Spend")
        fig1 = create_website_visits_chart(web_data, media_spend_dict, monthly_offers)
        st.plotly_chart(fig1, use_container_width=True)

        st.divider()

        # Chart 2: Offers Page Views
        st.subheader("🎯 Offers Page Views vs Media Spend")
        fig2 = create_offers_page_views_chart(web_data, media_spend_dict, monthly_offers)
        st.plotly_chart(fig2, use_container_width=True)

        # Additional insights
        st.divider()
        st.subheader("💡 Key Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Engagement Metrics:**")
            avg_visits = web_data['Total_Visits'].mean()
            avg_offers_views = web_data['Offers_Page_Views'].mean()
            st.write(f"- Average monthly visits: **{avg_visits:,.0f}**")
            st.write(f"- Average monthly offers page views: **{avg_offers_views:,.0f}**")
            st.write(f"- Average conversion to offers page: **{(avg_offers_views/avg_visits*100):.2f}%**")

        with col2:
            st.markdown("**Media Investment:**")
            avg_media = sum(hyundai_media.values()) / len(hyundai_media) if hyundai_media else 0
            st.write(f"- Average monthly media spend: **${avg_media:,.0f}**")
            cost_per_visit = total_media_spend / total_visits if total_visits > 0 else 0
            cost_per_offer_view = total_media_spend / total_offers_views if total_offers_views > 0 else 0
            st.write(f"- Cost per website visit: **${cost_per_visit:.2f}**")
            st.write(f"- Cost per offers page view: **${cost_per_offer_view:.2f}**")

    # Original Hyundai Model Analysis Page
    else:
        st.title("🚗 Hyundai Model Analysis Dashboard")
        st.markdown("Select a model to view comprehensive timeline analysis")

        # Get Hyundai models
        hyundai_models = sorted(df[df['Make'] == 'Hyundai']['Model'].unique())

        # Check if we have a model selected in session state
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = None

        # Home page - model selection
        if st.session_state.selected_model is None:
            st.markdown("## 🚗 Select a Hyundai Model")
            st.markdown("Choose a model to view detailed performance analysis")

            # Custom CSS for better looking cards
            st.markdown("""
        <style>
        .model-card {
            background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
            border: 1px solid #e9ecef;
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            transition: all 0.3s ease;
            cursor: pointer;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }
        .model-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            border-color: #1f77b4;
        }
        .model-name {
            font-size: 1.2em;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 8px;
        }
        .model-segment {
            font-size: 0.9em;
            color: #6c757d;
            background: #e9ecef;
            padding: 4px 12px;
            border-radius: 20px;
            display: inline-block;
        }
        </style>
            """, unsafe_allow_html=True)

            # Group models by category and create enhanced cards
            model_data_map = {}
            for model in hyundai_models:
                model_data = df[(df['Make'] == 'Hyundai') & (df['Model'] == model)]
                if not model_data.empty:
                    sales_row = model_data[model_data['Category'] == 'Sales']
                    if not sales_row.empty:
                        segment = sales_row.iloc[0]['Segment']
                        model_data_map[model] = segment

            # Categorize models
            suvs = [model for model, segment in model_data_map.items() if 'SUV' in segment]
            sedans = [model for model, segment in model_data_map.items() if 'SUV' not in segment and segment != 'Unknown']
            others = [model for model, segment in model_data_map.items() if segment == 'Unknown']

            # Display SUVs section
            if suvs:
                st.subheader("🚙 SUVs")
                cols = st.columns(3)
                for i, model in enumerate(suvs):
                    col_idx = i % 3
                    with cols[col_idx]:
                        # Enhanced card with icon and styling
                        segment = model_data_map.get(model, 'Unknown')

                        # Model icon based on segment
                        if 'Light' in segment:
                            icon = "🚙"
                        elif 'Small' in segment:
                            icon = "🚛"
                        elif 'Medium' in segment:
                            icon = "🚚"
                        else:
                            icon = "🚗"

                        with st.container():
                            if st.button(f"{icon} **{model}**", key=f"btn_{model}", use_container_width=True):
                                st.session_state.selected_model = model
                                st.rerun()
                            st.caption(f"🏷️ {segment}")
                            st.markdown("")

            # Display Sedans section
            if sedans:
                st.markdown("---")
                st.subheader("🚗 Sedans & Others")
                cols = st.columns(3)
                for i, model in enumerate(sedans):
                    col_idx = i % 3
                    with cols[col_idx]:
                        segment = model_data_map.get(model, 'Unknown')
                        with st.container():
                            if st.button(f"🚗 **{model}**", key=f"btn_{model}", use_container_width=True):
                                st.session_state.selected_model = model
                                st.rerun()
                            st.caption(f"🏷️ {segment}")
                            st.markdown("")

            # Display others if any
            if others:
                st.markdown("---")
                st.subheader("🚜 Other Models")
                cols = st.columns(3)
                for i, model in enumerate(others):
                    col_idx = i % 3
                    with cols[col_idx]:
                        segment = model_data_map.get(model, 'Unknown')
                        with st.container():
                            if st.button(f"🚜 **{model}**", key=f"btn_{model}", use_container_width=True):
                                st.session_state.selected_model = model
                                st.rerun()
                            st.caption(f"🏷️ {segment}")
                            st.markdown("")

            # Add helpful information at the bottom
            st.markdown("---")
            st.info("💡 **Pro Tip:** Each model analysis includes sales timelines, media spend correlation, offer effectiveness tracking, and AI-powered strategic insights.")

        else:
            # Model detail page
            model_name = st.session_state.selected_model

            # Back button
            if st.button("← Back to Model Selection"):
                st.session_state.selected_model = None
                st.rerun()

            st.header(f"Hyundai {model_name} Analysis")

            # Get model data
            model_info = get_model_data(df, model_name)

            if model_info is None:
                st.error(f"No data found for {model_name}")
                return

            # Show summary metrics
            show_model_summary(model_info)

            st.divider()

            # Create and show comprehensive timeline
            fig = create_comprehensive_timeline(model_info)

            if fig:
                st.plotly_chart(fig, use_container_width=True)

                # AI Insights Section
                st.subheader("🤖 AI Performance Insights")

                # Add a button to generate insights
                col1, col2 = st.columns([1, 4])

                with col1:
                    generate_insights = st.button(
                        "Generate AI Analysis",
                        type="primary",
                        help="Click to generate in-depth AI analysis of sales patterns, outliers, and offer effectiveness"
                    )

                with col2:
                    if generate_insights:
                        st.info("🔄 Generating AI insights... This may take a moment.")

                # Generate and display insights
                if generate_insights:
                    with st.spinner("Analyzing data with Gemini AI..."):
                        insights = generate_ai_insights(model_info)

                    # Display insights in an attractive container
                    st.markdown("### 📊 Comprehensive Analysis")

                    # Create an expandable insights container
                    with st.container():
                        st.markdown(
                            f"""
                            <div style="
                                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                                padding: 25px;
                                border-radius: 10px;
                                border-left: 5px solid #1f77b4;
                                border: 1px solid #e9ecef;
                                margin: 15px 0;
                                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                                color: #212529;
                                font-size: 16px;
                                line-height: 1.6;
                            ">
                            <div style="color: #212529; font-weight: 400;">
                            {insights}

                            """,
                            unsafe_allow_html=True
                        )

                    # Add timestamp
                    from datetime import datetime
                    st.caption(f"Analysis generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")

            else:
                st.error("Could not create timeline chart")

            # Show data table
            if st.checkbox("Show raw data"):
                st.subheader("Timeline Data")
                display_data = model_info['data'][['Month', 'Sales', 'Media_Spend', 'Offer']].copy()
                display_data['Media_Spend'] = display_data['Media_Spend'].apply(lambda x: f"${x:,.0f}")
                display_data['Sales'] = display_data['Sales'].apply(lambda x: f"{x:,.0f}")
                st.dataframe(display_data, use_container_width=True)

if __name__ == "__main__":
    main()