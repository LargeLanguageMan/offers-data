import pandas as pd
import tempfile
import os
from datetime import datetime

def create_comprehensive_joined_dataset():
    """Create a comprehensive joined dataset with all data sources including RBA cash rates"""

    # RBA Cash Rate data mapping (effective dates to monthly application)
    rba_rates = {
        # 2023 rates
        'Aug-23': 4.10, 'Sep-23': 4.10, 'Oct-23': 4.10, 'Nov-23': 4.35, 'Dec-23': 4.35,
        # 2024 rates
        'Jan-24': 4.35, 'Feb-24': 4.35, 'Mar-24': 4.35, 'Apr-24': 4.35, 'May-24': 4.35,
        'Jun-24': 4.35, 'Jul-24': 4.35, 'Aug-24': 4.35, 'Sep-24': 4.35, 'Oct-24': 4.35,
        'Nov-24': 4.35, 'Dec-24': 4.35,
        # 2025 rates
        'Jan-25': 4.35, 'Feb-25': 4.10, 'Mar-25': 4.10, 'Apr-25': 4.10, 'May-25': 3.85,
        'Jun-25': 3.85, 'Jul-25': 3.85, 'Aug-25': 3.60, 'Sep-25': 3.60
    }

    def load_csv_with_malformed_header(filepath):
        """Load CSV handling malformed headers"""
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Fix malformed header if present
        if '"Sales' in lines[0]:
            header_line = lines[0].strip().replace('"Sales\n', 'Sales_') + lines[1].strip()
            lines[1] = header_line + '\n'
            lines = lines[1:]

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.writelines(lines)
            tmp_path = tmp_file.name

        df = pd.read_csv(tmp_path)
        os.unlink(tmp_path)
        return df

    print("Loading enhanced-data-v2.csv...")
    df_enhanced_v2 = load_csv_with_malformed_header('data/source/enhanced-data-v2.csv')

    # Rename columns for consistency
    df_enhanced_v2 = df_enhanced_v2.rename(columns={
        'Sales Rank': 'Sales_Rank',
        '12 Month Average': 'Monthly_Average'
    })

    print("Loading original car-model-report.csv for media spend data...")
    df_original = load_csv_with_malformed_header('data/source/car-model-report.csv')
    df_original = df_original.rename(columns={
        'SalesRank': 'Sales_Rank',
        'Monthly Average': 'Monthly_Average'
    })

    # Get monthly columns
    monthly_cols = [col for col in df_enhanced_v2.columns if '-' in col and len(col) == 6]
    monthly_cols.sort()

    print(f"Processing {len(monthly_cols)} months: {monthly_cols}")

    # Create the comprehensive joined dataset
    joined_data = []

    # Process each model and category combination
    for idx, row in df_enhanced_v2.iterrows():
        make = row['Make']
        model = row['Model']
        segment = row['Segment']
        category = row['Category']

        # Skip non-essential categories for the joined dataset
        if category not in ['Sales', 'Segment Share', 'Retail Offer']:
            continue

        # Get media spend data from original dataset if available
        media_spend_row = None
        media_categories = ['Above the Line Weighted Incentive', 'Total Media Spend', 'CPV']

        for media_cat in media_categories:
            media_mask = (
                (df_original['Make'] == make) &
                (df_original['Model'] == model) &
                (df_original['Category'] == media_cat)
            )
            if media_mask.any():
                media_spend_row = df_original[media_mask].iloc[0]
                break

        # Create record for each month
        for month_col in monthly_cols:
            if month_col in row.index and pd.notna(row[month_col]) and str(row[month_col]).strip():

                record = {
                    'Month': month_col,
                    'Make': make,
                    'Model': model,
                    'Segment': segment,
                    'Sales_Rank': row.get('Sales_Rank', ''),
                    'Cash_Rate_Percent': rba_rates.get(month_col, ''),
                }

                # Add category-specific data
                if category == 'Sales':
                    record['Sales_Total'] = row[month_col]
                    record['Segment_Share_Percent'] = ''
                    record['Retail_Offer'] = ''

                elif category == 'Segment Share':
                    record['Sales_Total'] = ''
                    record['Segment_Share_Percent'] = row[month_col]
                    record['Retail_Offer'] = ''

                elif category == 'Retail Offer':
                    record['Sales_Total'] = ''
                    record['Segment_Share_Percent'] = ''
                    record['Retail_Offer'] = row[month_col]

                # Add media spend data if available
                if media_spend_row is not None and month_col in media_spend_row.index:
                    record['Media_Spend_Total'] = media_spend_row[month_col] if pd.notna(media_spend_row[month_col]) else ''
                else:
                    record['Media_Spend_Total'] = ''

                joined_data.append(record)

    # Convert to DataFrame
    df_joined = pd.DataFrame(joined_data)

    # Group by Month, Make, Model to combine Sales, Segment Share, and Offers into single rows
    print("Consolidating data by month/make/model...")

    consolidated_data = []

    for (month, make, model), group in df_joined.groupby(['Month', 'Make', 'Model']):
        consolidated_record = {
            'Month': month,
            'Make': make,
            'Model': model,
            'Segment': group['Segment'].iloc[0],
            'Sales_Rank': group['Sales_Rank'].iloc[0],
            'Cash_Rate_Percent': group['Cash_Rate_Percent'].iloc[0],
            'Sales_Total': '',
            'Segment_Share_Percent': '',
            'Retail_Offer': '',
            'Media_Spend_Total': group['Media_Spend_Total'].iloc[0] if not group['Media_Spend_Total'].iloc[0] == '' else ''
        }

        # Consolidate the different category values
        for _, row in group.iterrows():
            if row['Sales_Total']:
                consolidated_record['Sales_Total'] = row['Sales_Total']
            if row['Segment_Share_Percent']:
                consolidated_record['Segment_Share_Percent'] = row['Segment_Share_Percent']
            if row['Retail_Offer']:
                consolidated_record['Retail_Offer'] = row['Retail_Offer']

        consolidated_data.append(consolidated_record)

    df_final = pd.DataFrame(consolidated_data)

    # Sort by Month, Make, Model for better organization
    month_order = monthly_cols
    df_final['Month_Order'] = df_final['Month'].map({month: i for i, month in enumerate(month_order)})
    df_final = df_final.sort_values(['Month_Order', 'Make', 'Model']).drop('Month_Order', axis=1)

    # Save the comprehensive joined dataset
    output_file = 'data/generated/comprehensive_joined_dataset.csv'
    df_final.to_csv(output_file, index=False)

    print(f"\n✅ Comprehensive joined dataset created: {output_file}")
    print(f"📊 Total records: {len(df_final)}")
    print(f"📅 Months covered: {len(monthly_cols)} ({monthly_cols[0]} to {monthly_cols[-1]})")
    print(f"🚗 Makes covered: {df_final['Make'].nunique()}")
    print(f"🏷️  Models covered: {df_final['Model'].nunique()}")

    print(f"\n📋 Columns in final dataset:")
    for col in df_final.columns:
        print(f"   - {col}")

    # Show sample data
    print(f"\n🔍 Sample records:")
    print(df_final.head(10).to_string())

    return df_final

if __name__ == "__main__":
    df = create_comprehensive_joined_dataset()