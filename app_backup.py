import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
import tempfile
import os

# Configure page
st.set_page_config(
    page_title="Hyundai Model Analysis",
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
def load_data():
    """Load and process the automotive data with enhanced offer information"""
    # Load original car model report
    with open('car-model-report.csv', 'r') as f:
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
    with open('enhanced-data.csv', 'r') as f:
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
        vertical_spacing=0.1,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        subplot_titles=[
            f'{model_name} - Sales & Media Spend Timeline',
            'Offer Strategy by Month'
        ]
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

    # Update layout with modern, clean styling
    fig.update_layout(
        title=dict(
            text=f'<b style="color: #2c3e50;">Hyundai {model_name}</b><br><span style="color: #6c757d; font-size: 14px;">{segment} • Performance Timeline</span>',
            x=0.5,
            font=dict(size=24, family="Arial, sans-serif")
        ),
        height=800,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12, color="#2c3e50"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        )
    )

    # Update axes with improved styling
    fig.update_yaxes(
        title_text="<b>Sales (Units)</b>",
        row=1, col=1, secondary_y=False,
        gridcolor='rgba(128,128,128,0.2)',
        showgrid=True,
        zeroline=False,
        tickformat=',d'
    )
    fig.update_yaxes(
        title_text="<b>Media Investment ($)</b>",
        row=1, col=1, secondary_y=True,
        gridcolor='rgba(128,128,128,0.1)',
        showgrid=False,
        zeroline=False,
        tickformat='$,.0f'
    )
    fig.update_yaxes(
        title_text="<b>Offer Types</b>",
        row=2, col=1,
        tickmode='array',
        tickvals=list(range(1, len(offer_data['Offer'].unique())+1)) if not offer_data.empty else [],
        ticktext=list(offer_data['Offer'].unique()) if not offer_data.empty else [],
        gridcolor='rgba(128,128,128,0.1)',
        showgrid=True
    )

    fig.update_xaxes(
        title_text="<b>Timeline</b>",
        row=2, col=1,
        showgrid=True,
        gridcolor='rgba(128,128,128,0.2)',
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
def main():
    st.title("🚗 Hyundai Model Analysis Dashboard")
    st.markdown("Select a model to view comprehensive timeline analysis")

    # Load data
    df = load_data()

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

        # Create a grid of model cards
        cols = st.columns(3)

        for i, model in enumerate(hyundai_models):
            col_idx = i % 3

            with cols[col_idx]:
                # Get segment info for this model
                model_data = df[(df['Make'] == 'Hyundai') & (df['Model'] == model)]
                segment = 'Unknown'
                if not model_data.empty:
                    sales_row = model_data[model_data['Category'] == 'Sales']
                    if not sales_row.empty:
                        segment = sales_row.iloc[0]['Segment']

                # Create custom styled button
                if st.button(f"🚙 {model}", key=f"btn_{model}", use_container_width=True):
                    st.session_state.selected_model = model
                    st.rerun()

                st.markdown(f'<div style="text-align: center; margin-top: -10px;"><span class="model-segment">{segment}</span></div>', unsafe_allow_html=True)

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