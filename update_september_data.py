import pandas as pd
import numpy as np

# Read the September data
print("Reading September data...")
sept_df = pd.read_csv('data/source/Offers data - September.csv')

# Read the existing comprehensive dataset
print("Reading comprehensive dataset...")
comp_df = pd.read_csv('data/generated/comprehensive_joined_dataset.csv')

# Pivot the September data so that Category values become columns
print("Transforming September data...")
sept_pivot = sept_df.pivot_table(
    index=['Segment', 'Make', 'Model', 'Sales Rank'],
    columns='Category',
    values='Sep-25',
    aggfunc='first'
).reset_index()

# Rename columns to match comprehensive dataset
sept_pivot = sept_pivot.rename(columns={
    'Sales': 'Sales_Total',
    'Segment Share': 'Segment_Share_Percent',
    'Retail Offer': 'Retail_Offer',
    'Sales Rank': 'Sales_Rank'
})

# Add Month column
sept_pivot['Month'] = 'Sep-25'

# Convert Sales_Total to numeric (remove commas if present)
if 'Sales_Total' in sept_pivot.columns:
    sept_pivot['Sales_Total'] = pd.to_numeric(sept_pivot['Sales_Total'], errors='coerce')

# Convert Sales_Rank to numeric
if 'Sales_Rank' in sept_pivot.columns:
    sept_pivot['Sales_Rank'] = pd.to_numeric(sept_pivot['Sales_Rank'], errors='coerce')

# For now, set Media_Spend_Total to NaN (since media spend not ready yet)
sept_pivot['Media_Spend_Total'] = np.nan

# Set Cash_Rate_Percent to NaN for now (can be filled in later if needed)
sept_pivot['Cash_Rate_Percent'] = np.nan

# Select only the columns we need to match comprehensive dataset
columns_needed = ['Month', 'Make', 'Model', 'Segment', 'Sales_Rank', 'Cash_Rate_Percent',
                  'Sales_Total', 'Segment_Share_Percent', 'Retail_Offer', 'Media_Spend_Total']

# Make sure all required columns exist, add them if missing
for col in columns_needed:
    if col not in sept_pivot.columns:
        sept_pivot[col] = np.nan

# Aggregate by Make, Model, Segment, Month - sum sales and combine offers
print("Aggregating variants by Make/Model/Segment...")
sept_aggregated = sept_pivot.groupby(['Month', 'Make', 'Model', 'Segment']).agg({
    'Sales_Total': 'sum',  # Sum sales across variants
    'Sales_Rank': 'min',  # Take the best (lowest) sales rank
    'Cash_Rate_Percent': 'first',
    'Segment_Share_Percent': 'first',  # Take first since it should be same
    'Retail_Offer': lambda x: ';'.join(sorted(set([str(i) for i in x.dropna()]))),  # Combine unique offers
    'Media_Spend_Total': 'first'
}).reset_index()

# Clean up retail offer (remove 'nan' and empty strings)
sept_aggregated['Retail_Offer'] = sept_aggregated['Retail_Offer'].replace('', np.nan)
sept_aggregated['Retail_Offer'] = sept_aggregated['Retail_Offer'].apply(
    lambda x: np.nan if pd.isna(x) or x == '' or x == 'nan' else x
)

sept_final = sept_aggregated[columns_needed]

print(f"\nSeptember data shape: {sept_final.shape}")
print(f"September total sales: {sept_final['Sales_Total'].sum():,.0f}")
print(f"September records with offers: {sept_final['Retail_Offer'].notna().sum()}")

# Remove existing Sep-25 data from comprehensive dataset
print("\nRemoving old Sep-25 data...")
comp_df_filtered = comp_df[comp_df['Month'] != 'Sep-25']
print(f"Records before Sep-25 removal: {len(comp_df)}")
print(f"Records after Sep-25 removal: {len(comp_df_filtered)}")

# Append the new September data
print("\nAppending new September data...")
comp_df_updated = pd.concat([comp_df_filtered, sept_final], ignore_index=True)

print(f"Total records after update: {len(comp_df_updated)}")

# Save the updated dataset
print("\nSaving updated dataset...")
comp_df_updated.to_csv('data/generated/comprehensive_joined_dataset.csv', index=False)

print("\n✅ Successfully updated comprehensive dataset with September 2025 data!")
print(f"\nSummary:")
print(f"  - Total records: {len(comp_df_updated):,}")
print(f"  - Sep-25 records: {len(sept_final):,}")
print(f"  - Sep-25 total sales: {sept_final['Sales_Total'].sum():,.0f}")
print(f"  - Sep-25 unique makes: {sept_final['Make'].nunique()}")
print(f"  - Sep-25 unique models: {sept_final['Model'].nunique()}")
