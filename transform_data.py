import pandas as pd
from datetime import datetime, timedelta
import calendar

# Read the CSV file
df = pd.read_csv('DAC-foxtel-AU - sheet3.csv')

# Define region mapping based on Product Desc
def map_region(product_desc):
    product_desc_upper = str(product_desc).upper()

    # Southern regions
    if any(x in product_desc_upper for x in ['SOUTH', 'SOUTHERN']):
        return 'southern'
    # Eastern regions
    elif any(x in product_desc_upper for x in ['EAST', 'EASTERN', 'NSW', 'QLD', 'QUEENSLAND']):
        return 'eastern'
    # Northern regions
    elif any(x in product_desc_upper for x in ['NORTH', 'NORTHERN', 'NT']):
        return 'northern'
    # Western regions
    elif any(x in product_desc_upper for x in ['WEST', 'WESTERN', 'WA']):
        return 'western'
    # Central regions (including SA, Adelaide)
    elif any(x in product_desc_upper for x in ['CENTRAL', 'SA', 'ADELAIDE']):
        return 'central'
    else:
        return 'central'

# Apply region mapping
df['region'] = df['Product Desc'].apply(map_region)

# Create rows for each day of each month
results = []

for _, row in df.iterrows():
    platform = row['Media Type Desc']
    region = row['region']

    # Process July 2025 (31 days)
    if pd.notna(row['Jul 2025']) and row['Jul 2025'] > 0:
        days_in_july = 31
        daily_spend = row['Jul 2025'] / days_in_july
        for day in range(1, days_in_july + 1):
            date_str = f"{day:02d}/07/2025"
            results.append({
                'date': date_str,
                'platform': platform,
                'region': region,
                'spend': daily_spend
            })

    # Process August 2025 (31 days)
    if pd.notna(row['Aug 2025']) and row['Aug 2025'] > 0:
        days_in_august = 31
        daily_spend = row['Aug 2025'] / days_in_august
        for day in range(1, days_in_august + 1):
            date_str = f"{day:02d}/08/2025"
            results.append({
                'date': date_str,
                'platform': platform,
                'region': region,
                'spend': daily_spend
            })

    # Process September 2025 (30 days)
    if pd.notna(row['Sep 2025']) and row['Sep 2025'] > 0:
        days_in_september = 30
        daily_spend = row['Sep 2025'] / days_in_september
        for day in range(1, days_in_september + 1):
            date_str = f"{day:02d}/09/2025"
            results.append({
                'date': date_str,
                'platform': platform,
                'region': region,
                'spend': daily_spend
            })

# Create output dataframe
output_df = pd.DataFrame(results)

# Group by date, platform, region and sum the spend
output_df = output_df.groupby(['date', 'platform', 'region'], as_index=False)['spend'].sum()

# Save to CSV
output_df.to_csv('transformed_output.csv', index=False)

print(f"Transformation complete. Output saved to transformed_output.csv")
print(f"Total rows: {len(output_df)}")
print(f"\nSample data:")
print(output_df.head(10))
print(f"\nUnique regions: {sorted(output_df['region'].unique())}")
print(f"Unique platforms: {sorted(output_df['platform'].unique())}")
print(f"\nExpected days: 92 (31 July + 31 August + 30 September)")
