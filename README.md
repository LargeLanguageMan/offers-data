# Automotive Offers Effectiveness Dashboard

This Streamlit dashboard analyzes automotive offer effectiveness across the Australian market, focusing on the key questions from the CMO brief.

## 🚀 Quick Start

1. **Launch the dashboard:**
   ```bash
   streamlit run dashboard.py
   ```

2. **Access the dashboard:**
   - Local URL: http://localhost:8502
   - The dashboard will open automatically in your browser

## 📊 Dashboard Features

### Main Visualizations
- **Sales Timeline**: Monthly sales performance with month-year on x-axis
- **Market Share Analysis**: Pie charts showing market share by model
- **Offer Type Comparison**: Bar charts comparing primary offer types
- **Hyundai Venue Case Study**: Detailed analysis of weekly payment offers

### Interactive Filters
- Model selection (Ranger, HiLux, RAV4, Kona, Venue, etc.)
- Vehicle segment filtering
- Date range selection (Sep 2023 - Aug 2025)

### AI-Powered Analysis
- **Market Trends**: Gemini AI analyzes seasonal patterns and trends
- **Strategic Recommendations**: AI-generated insights for offer optimization
- **Real-time Analysis**: Click buttons to generate fresh insights

## 🎯 Key Questions Answered

### 1. Offer Penetration Analysis
- **Finance vs Drive-away**: Compare penetration rates across models
- **Model-specific Performance**: Which offers work best for each vehicle
- **Geographic Variations**: Limited by available data

### 2. Venue Model Case Study
- Weekly payment offer effectiveness
- Timeline of offer types used
- Incentive spending patterns
- Market share performance in SUV Light segment

### 3. Time-based Trends
- 18+ months of historical data analysis
- Seasonal patterns in offer effectiveness
- Marketing spend impact on sales conversion

## 📈 Current Data Analysis

### Available Models
- **Ford Ranger** (PU/CC segment)
- **Toyota HiLux** (PU/CC segment)
- **Toyota RAV4** (SUV Medium < $60K)
- **Hyundai Kona** (SUV Small < $45K)
- **Hyundai Venue** (SUV Light) - *Primary case study*

### Offer Types Identified
1. **Finance-based Offers**
   - Standard financing packages
   - Finance + Roadside Assist bundles
   - Weekly payment plans

2. **Drive-away Pricing**
   - Complete "on the road" pricing
   - Reduced price promotions

3. **Value-added Services**
   - Servicing packages
   - Accessory bundles
   - Factory bonuses

## 🔍 Data Insights Summary

### Key Findings from analysis.txt:
- **Venue Strategy**: Consistently uses "Driveaway" pricing approach
- **Market Share**: Venue maintains 11-12% of SUV Light segment
- **Incentive Spending**: Ranges from $289K to $4.3M across periods
- **Seasonal Trends**: Visible fluctuations in sales performance

## 🌐 External Data Sources Needed

### Competitor Analysis
- **Recommended Sources:**
  - VFACTS (Vehicle sales data)
  - Glass's Guide (Automotive market intelligence)
  - AutoTrader market reports
  - Roy Morgan automotive research

### Market Context Data
- **Cost-of-living Impact:**
  - Australian Bureau of Statistics (ABS)
  - RBA interest rate data
  - Consumer confidence indices

### Website Analytics
- **Offer Page Visits:**
  - Google Analytics integration needed
  - Marketing spend correlation data
  - Conversion funnel analysis

### Additional Data Requirements
- **Hyundai Capital Data**: November 2024+ (mentioned in brief)
- **Previous Finance Supplier**: Pre-November 2024 data
- **Regional Breakdown**: State/territory performance data
- **Demographic Data**: Target audience segmentation

## 🔧 Technical Requirements

### Dependencies
```bash
streamlit==1.50.0
pandas>=2.3.2
plotly>=6.3.0
google-generativeai>=0.8.5
numpy>=2.3.3
```

### Configuration
- **Gemini API**: Integrated with provided key
- **Model**: Using gemini-2.0-flash-exp for analysis
- **Data Processing**: Automated CSV parsing and cleaning

## 📋 Usage Instructions

1. **Start Analysis**: Launch dashboard and select desired models/segments
2. **Timeline Analysis**: Use date slider to focus on specific periods
3. **AI Insights**: Click analysis buttons for AI-generated insights
4. **Venue Deep-dive**: Focus on Venue model for weekly payment analysis
5. **Export Data**: Use browser tools to export charts as needed

## 🚨 Data Limitations

- **Geographic**: Limited regional breakdown available
- **Demographics**: Customer demographic data not present
- **Competitor Data**: External sources required
- **Website Data**: Analytics integration needed
- **Weekly Payments**: Specific performance metrics need deeper analysis

## 🎯 Next Steps for Complete Analysis

1. **Integrate Competitor Data**: Add external market intelligence
2. **Website Analytics**: Connect Google Analytics for page visit data
3. **Regional Analysis**: Add state/territory breakdown
4. **Cost-of-living Impact**: Correlate with economic indicators
5. **Advanced AI Models**: Implement predictive analytics

## 📞 Support

For dashboard issues or enhancement requests:
- Review analysis.txt for detailed data insights
- Check debug section in dashboard for data quality issues
- Modify filters if visualizations appear empty

---
*Dashboard created for CMO offer effectiveness analysis | Data period: Sep 2023 - Aug 2025*