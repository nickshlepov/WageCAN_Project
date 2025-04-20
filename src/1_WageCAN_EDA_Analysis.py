# WageCAN Project
# Block 1: Exploratory Data Analysis (EDA)
# Author: Nick Shlepov
# Description:
# This script conducts an exploratory analysis of Canadian wage data (2016–2024),
# including national, provincial, and regional-level wage trends,
# wage fluctuation patterns within occupations, and initial interactive visualizations.
# Key outputs include boxplots, summary statistics, heatmaps, and interactive charts for further analysis.


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import altair as alt

from pathlib import Path

from scipy.stats import pearsonr, spearmanr

# Ensure all necessary output directories exist
Path("../output/figures").mkdir(parents=True, exist_ok=True)
Path("../output/logs").mkdir(parents=True, exist_ok=True)
Path("../output/csv").mkdir(parents=True, exist_ok=True)
Path("../data").mkdir(parents=True, exist_ok=True)

# Define paths
DATA_PATH = '../data'
FIGURE_PATH = '../output/figures'
LOG_PATH = '../output/logs'
CSV_PATH = '../output/csv'

# Read csv file created in Block 0
merged_wages = pd.read_csv(Path(CSV_PATH) / "merged_wages_2016_2024_mapped_to_NOC2021.csv")


##### ================== Exploratory data analysis (EDA) =================================

# Matplotlib boxplot style function (created to avoid a redundancy)
def style_boxplot(bp, colors=None):
    # Boxes
    if colors is not None:
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
    else:
        for patch in bp['boxes']:
            patch.set_facecolor('#cccccc')
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)

    # Medians
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)

    # Whiskers
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.0)

    # Fliers
    for flier in bp['fliers']:
        flier.set(marker='o', color='black', alpha=0.3)


# Province abbreviation dictionary (used in multiple sections)
province_abbr = {
    'Alberta': 'AB',
    'British Columbia': 'BC',
    'Manitoba': 'MB',
    'New Brunswick': 'NB',
    'Newfoundland and Labrador': 'NL',
    'Nova Scotia': 'NS',
    'Ontario': 'ON',
    'Prince Edward Island': 'PE',
    'Quebec': 'QC',
    'Saskatchewan': 'SK',
    'Northwest Territories': 'NT',
    'Nunavut': 'NU',
    'Yukon Territory': 'YT'
}

#### NATIONAL LEVEL =====
### IMPORTANT ====== Create national_df to reuse many times later

# Filter to national-level data only
national_df = merged_wages[
    (merged_wages['Province'] == 'National') & (merged_wages['Region'] == 'Canada')
].copy()

# Descriptive statistics saved to a file eda_summary_national
national_summary = national_df[['Median_Wage_2016', 'Median_Wage_2020', 'Median_Wage_2024']].describe()
with open(Path(LOG_PATH) / "1-1-eda_summary_national.txt", "w") as f:
    f.write("National-level Median Wage Summary Statistics:\n")
    f.write(national_summary.to_string())
    f.write("\n" + "=" * 80 + "\n\n")

#### ====== 1-1-1 National Median Wages Boxplots 2016-2020-2024 =====

# Prepare data for boxplot
box_data = pd.DataFrame({
    '2016': national_df['Median_Wage_2016'],
    '2020': national_df['Median_Wage_2020'],
    '2024': national_df['Median_Wage_2024']
})

# Create boxplot
labels = ['2016', '2020', '2024']
colors = ['#fde0dd', '#fa9fb5', '#c51b8a']

# Create plot
fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot(box_data, patch_artist=True, tick_labels=labels, notch=True, widths=0.5)

# Apply style_boxplot function
style_boxplot(bp, colors)

# Titles and labels
ax.set_title('National Median Wage Distribution (2016–2024)', fontsize=14)
ax.set_ylabel('Median Hourly Wage ($)', fontsize=12)
ax.grid(True, axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(Path(FIGURE_PATH) / "1-1-1-National-boxplot_median_wages_national_years.png")
plt.close(fig)

#### === 1-1-2 National Wages 2024 vs TEER ====

# Use existing national_df and ensure TEER is numeric
national_df['TEER_Code'] = pd.to_numeric(national_df['TEER_Code'], errors='coerce')

# Drop rows with missing TEER or wage
national_df = national_df.dropna(subset=['TEER_Code', 'Median_Wage_2024'])

# Compute correlations
pearson_corr, _ = pearsonr(national_df['TEER_Code'], national_df['Median_Wage_2024'])
spearman_corr, _ = spearmanr(national_df['TEER_Code'], national_df['Median_Wage_2024'])

# Group by TEER
grouped_teer = national_df.groupby('TEER_Code')['Median_Wage_2024'].agg(['count', 'mean', 'median']).round(2)

# Save to file
with open(Path(LOG_PATH) / "1-1-eda_summary_national.txt", "a") as f:
    f.write("Median Wage 2024 by TEER Level (National):\n")
    f.write(grouped_teer.to_string())
    f.write("\n\n")
    f.write(f"Pearson Correlation (TEER vs. Median Wage 2024): {pearson_corr:.3f}\n")
    f.write(f"Spearman Correlation (TEER vs. Median Wage 2024): {spearman_corr:.3f}\n")
    f.write("=" * 80 + "\n\n")


# Prepare boxplot data by TEER
teer_levels = sorted(national_df['TEER_Code'].unique())
box_data = [national_df[national_df['TEER_Code'] == teer]['Median_Wage_2024'] for teer in teer_levels]

# Colors from light to dark blue
colors = ['#deebf7', '#9ecae1', '#6baed6', '#3182bd', '#08519c', '#08306b'][:len(teer_levels)]

# Plot
fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot(box_data, patch_artist=True, tick_labels=[str(int(teer)) for teer in teer_levels], notch=True, widths=0.5)

# Apply style_boxplot function
style_boxplot(bp, colors)

# Titles and labels
ax.set_title('Median Wage by TEER Level (2024, National)', fontsize=14)
ax.set_xlabel('TEER Code')
ax.set_ylabel('Median Hourly Wage ($)', fontsize=12)
ax.grid(True, axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(Path(FIGURE_PATH) / "1-1-2-National-boxplot_TEER_vs_MedianWage2024.png")
plt.close(fig)

#### === 1-1-3 National Wages 2024 vs Broad Occupational Category ===

# Drop missing values
national_df['Broad_Category_Code'] = pd.to_numeric(national_df['Broad_Category_Code'], errors='coerce')
national_df = national_df.dropna(subset=['Broad_Category_Code', 'Median_Wage_2024'])

# Compute correlations
broad_pearson, _ = pearsonr(national_df['Broad_Category_Code'], national_df['Median_Wage_2024'])
broad_spearman, _ = spearmanr(national_df['Broad_Category_Code'], national_df['Median_Wage_2024'])

# Group stats
grouped_broad = national_df.groupby('Broad_Category_Code')['Median_Wage_2024'].agg(['count', 'mean', 'median']).round(2)

# Save to file
with open(Path(LOG_PATH) / "1-1-eda_summary_national.txt", "a") as f:
    f.write("Median Wage 2024 by Broad Occupational Category (National):\n")
    f.write(grouped_broad.to_string())
    f.write("\n\n")
    f.write(f"Pearson Correlation (Broad Category vs. Median Wage 2024): {broad_pearson:.3f}\n")
    f.write(f"Spearman Correlation (Broad Category vs. Median Wage 2024): {broad_spearman:.3f}\n")
    f.write("=" * 80 + "\n\n")

# Prepare boxplot data
category_order = sorted(national_df['Broad_Category_Code'].unique())
box_data = [national_df[national_df['Broad_Category_Code'] == cat]['Median_Wage_2024'] for cat in category_order]

# Get category names
category_names = national_df[['Broad_Category_Code', 'Broad_Category_Name']].drop_duplicates().set_index('Broad_Category_Code')
colors = plt.colormaps['tab10'](range(len(category_order)))

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
bp = ax.boxplot(box_data, patch_artist=True, tick_labels=[str(int(code)) for code in category_order], notch=True, widths=0.5)

# Apply style_boxplot function
style_boxplot(bp, colors)

# Add legend manually
handles = [plt.Line2D([0], [0], color='white', marker='s', markersize=10,
                      markerfacecolor=color, label=category_names.loc[cat, 'Broad_Category_Name'])
           for cat, color in zip(category_order, colors)]
ax.legend(handles=handles, title='Broad Occupational Category', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)

# Labels
ax.set_title('Median Wage by Broad Occupational Category (2024, National)', fontsize=14)
ax.set_xlabel('Broad Category Code')
ax.set_ylabel('Median Hourly Wage ($)', fontsize=12)
ax.grid(True, axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(Path(FIGURE_PATH) / "1-1-3-National-boxplot_BroadCategory_vs_MedianWage2024.png")
plt.close(fig)

#### ========== PROVINCIAL LEVEL ===============
# Filter to true provincial-level data: Province != 'National' and Region == Province
# Filter to provincial-level data only (no National, no Territories, Region == Province)

### ======= Provincial Median Wages 2016-2020-2024 =====

'''
#Source for Top3 and Bottom3 Provinces - 2022 Median Annual Wages Data by Statistics Canada'
#https://www150.statcan.gc.ca/n1/daily-quotidien/240412/t002a-eng.csv
#official_wages = pd.read_csv('t002a-eng.csv')
'''

bottom3 = ['Prince Edward Island', 'Newfoundland and Labrador', 'Nova Scotia']
top3 = ['Alberta', 'Ontario', 'British Columbia']

# Filter to provincial-level data only (excluding National and Territories)
prov_df = merged_wages[
    (merged_wages['Province'] != 'National') &
    (merged_wages['Province'] == merged_wages['Region']) &
    (~merged_wages['Province'].isin(['Yukon Territory', 'Northwest Territories', 'Nunavut']))
]

# Combine top and bottom 3
target_provinces = top3 + bottom3
filtered_df = prov_df[prov_df['Province'].isin(target_provinces)]


### ===== 1-2 Provincial Median Wages Boxplots =======
# Filter to provincial-level data (exclude territories and national)
excluded_territories = ["Yukon Territory", "Northwest Territories", "Nunavut"]
provincial_df = merged_wages[
    (merged_wages['Province'] != 'National') &
    (merged_wages['Region'] == merged_wages['Province']) &
    (~merged_wages['Province'].isin(excluded_territories))
]

# Keep ordered list and abbreviations
province_order = sorted(provincial_df['Province'].unique())
province_abbr_order = [province_abbr[p] for p in province_order]

# Generate a colormap with len(province_order) distinct colors
cmap = ListedColormap(plt.cm.tab20.colors[:len(province_order)])
province_colors = {prov: cmap(i) for i, prov in enumerate(province_order)}

# Set up the figure
fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
years = ['2016', '2020', '2024']
wage_columns = ['Median_Wage_2016', 'Median_Wage_2020', 'Median_Wage_2024']

for i, (ax, year, column) in enumerate(zip(axs, years, wage_columns)):
    box_data = [provincial_df[provincial_df['Province'] == prov][column].dropna()
                for prov in province_order]

    bp = ax.boxplot(box_data, patch_artist=True, tick_labels=province_abbr_order, notch=True, widths=0.5)

    # Apply consistent colors
    for patch, prov in zip(bp['boxes'], province_order):
        patch.set_facecolor(province_colors[prov])
        patch.set_alpha(0.7)

    ax.set_title(f'Median Wages ({year})', fontsize=13)
    if i == 0:
        ax.set_ylabel('Hourly Median Wage ($)', fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    ax.grid(True, linestyle='--', alpha=0.4)

# Add a common title and adjust layout
fig.suptitle('Median Wage Distribution by Province (2016, 2020, 2024)', fontsize=18, fontweight='bold')
plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))  # leave space at the top for the title

plt.savefig(Path(FIGURE_PATH) / "1-2-Provincial-boxplot_median_wages_province_years.png")

#### ========== REGIONAL OUTLIERS=============

# Filter regional data
regional_df = merged_wages[
    (merged_wages['Province'] != 'National') &
    (merged_wages['Region'] != 'Canada') &
    (merged_wages['Region'] != merged_wages['Province'])
]

# Keep only regions with >= min_records
min_records = 10
region_counts = regional_df['Region'].value_counts()
valid_regions = region_counts[region_counts >= min_records].index
filtered_regional = regional_df[regional_df['Region'].isin(valid_regions)].copy()

# Map province abbreviation and create new Region_Label
filtered_regional['Province_Abbr'] = filtered_regional['Province'].map(province_abbr)
filtered_regional['Region_Label'] = filtered_regional['Region'] + ', ' + filtered_regional['Province_Abbr']

# Compute means for 2024 median wage using Region_Label
region_means = (
    filtered_regional.groupby('Region_Label')['Median_Wage_2024']
    .mean()
    .sort_values()
)

# Combine bottom and top 10
bottom_10 = region_means.head(10)
top_10 = region_means.tail(10).sort_values(ascending=False)

combined = pd.concat([bottom_10, top_10])
assert isinstance(combined, pd.Series)
ordered_regions = combined.sort_values(ascending=True).index.tolist()


#### ========= 1-3 Regional Outliers Boxplots ===========

# Prepare data for boxplot
plot_data = []
for region in ordered_regions:
    region_data = filtered_regional[filtered_regional['Region_Label'] == region]['Median_Wage_2024']
    plot_data.append(region_data)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

bp = ax.boxplot(
    plot_data,
    patch_artist=True,
    vert=False,
    notch=True,
    widths=0.6
)

# Apply color styling
# Count how many are bottom regions in ordered_regions
num_bottom = sum(region in bottom_10.index for region in ordered_regions)
colors_ordered = ['#d62728'] * num_bottom + ['#2ca02c'] * (len(ordered_regions) - num_bottom)

for patch, color in zip(bp['boxes'], colors_ordered):
    patch.set_facecolor(color)

# Axis formatting
ax.set_yticklabels(ordered_regions, fontsize=9)
ax.set_xlabel('Median Wage 2024 ($)')
ax.set_title('Median Wage 2024 – Top & Bottom 10 Regions (≥10 Records)', fontsize=14, weight='bold')
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.savefig(Path(FIGURE_PATH) / "1-3-Regional-boxplot_top10_bottom10_regions.png")

# === Write summaries to file ===
with open(Path(LOG_PATH) / "1-2-eda_summary_provincial.txt", "w", encoding="utf-8") as f:
    f.write("=== PROVINCIAL WAGE SUMMARY (Top 3 and Bottom 3 Provinces) ===\n\n")
    for province in target_provinces:
        province_df = filtered_df[filtered_df['Province'] == province]
        f.write(f"--- Summary for {province} ---\n")
        summary = province_df[['Median_Wage_2016', 'Median_Wage_2020', 'Median_Wage_2024']].describe()
        f.write(summary.to_string())
        f.write("\n\n")

    f.write("=== REGIONAL WAGE SUMMARY (2024 Median Wages) ===\n\n")
    f.write("Bottom 10 Regions by Mean Median_Wage_2024 (only regions with ≥ 10 records):\n")
    f.write(region_means.head(10).round(2).to_string())
    f.write("\n\nTop 10 Regions by Mean Median_Wage_2024 (only regions with ≥ 10 records):\n")
    f.write(region_means.tail(10).sort_values(ascending=False).round(2).to_string())
    f.write("\n" + "=" * 80 + "\n")


#### ========= INTERACTIVE VISUALIZATION ==============

#### ======= Fluctuation of wages within NOCs  ==========
## Normalized difference between highest and lowest wages for each NOC

### Preparation and Analysis

# Helper function for classification
def classify_fluctuation(rel_diff):
    if rel_diff < 0.5:
        return "Low, under 50%"
    elif rel_diff < 0.75:
        return "Average, 50-75%"
    elif rel_diff < 1.0:
        return "High, 75-100%"
    else:
        return "Very High, over 100%"


# Years to analyze
years = ['2016', '2020', '2024']

# List to collect results
fluctuation_records = []

# Loop through each year and compute relative differences (use national_df)
for year in years:
    year_df = national_df.copy()
    low_col = f'Low_Wage_{year}'
    med_col = f'Median_Wage_{year}'
    high_col = f'High_Wage_{year}'

    # Filter out rows with missing wage data
    year_df = year_df.dropna(subset=[low_col, med_col, high_col])

    # Compute relative high-low difference
    rel_diff = (year_df[high_col] - year_df[low_col]) / year_df[med_col]

    # Classify into fluctuation categories
    cluster = rel_diff.apply(classify_fluctuation)

    # Store results
    temp_df = pd.DataFrame({
        'NOC_2021': year_df['NOC_2021'],  # or use 'Major_Group_Code' if available
        'NOC_Title': year_df['NOC_Title_2021'],
        'Year': int(year),
        'rel_diff_high_low': rel_diff,
        'Wage_Fluctuation_Cluster': cluster
    })

    fluctuation_records.append(temp_df)

# Concatenate into a long-form DataFrame
fluctuation_long_df = pd.concat(fluctuation_records, ignore_index=True)

#### ==== 1-4 National wages fluctuation groups visualization =====

# Group and count
cluster_counts = (
    fluctuation_long_df
    .groupby(['Wage_Fluctuation_Cluster', 'Year'])
    .size()
    .reset_index(name='NOC Count')
)

# Order clusters manually for consistent x-axis
cluster_order = ['Low, under 50%', 'Average, 50-75%', 'High, 75-100%', 'Very High, over 100%']
cluster_counts['Wage_Fluctuation_Cluster'] = pd.Categorical(
    cluster_counts['Wage_Fluctuation_Cluster'], categories=cluster_order, ordered=True
)

# Altair bar chart
fluctuation_chart = alt.Chart(cluster_counts).mark_bar().encode(
    x=alt.X('Wage_Fluctuation_Cluster:N', title='Wage Fluctuation Category', sort=cluster_order),
    y=alt.Y('NOC Count:Q', title='Number of NOC Codes in the Category'),
    color=alt.Color('Wage_Fluctuation_Cluster:N',
                    title='Fluctuation Category',
                    scale=alt.Scale(domain=cluster_order,
                                    range=['#c6dbef', '#6baed6', '#3182bd', '#08306b'])),
    column=alt.Column('Year:O', title=None, spacing=10),  # Optional: Facet by year instead
    tooltip=[
        alt.Tooltip('Year:O'),
        alt.Tooltip('Wage_Fluctuation_Cluster:N', title='Category'),
        alt.Tooltip('NOC Count:Q')
    ]
).properties(
    title={
        "text": ["Wage Fluctuation Category Distribution Over Time"],
      "subtitle": ["Based on ((High Wage - Low Wage) / Median Wage)*100%",
                   ""]
    },
    width=300,
    height=400
).configure_axis(
    labelAngle=0
)

fluctuation_chart = fluctuation_chart.configure_title(
    fontSize=18,
    anchor='start'
).configure_axisX(
    labelAngle=45
)

fluctuation_chart.save(Path(FIGURE_PATH) / "1-4-National-wages-fluctuation-groups.html")

# The increase in NOC codes falling into Low and Average fluctuation categories in 2024
# may reflect greater wage stability within occupations.
# This trend could be influenced by market forces —
# or equally by the structural improvement introduced with the shift to 5-digit NOC codes,
# which allows for more precise and meaningful classification.


#### ======= Outliers in 2024 (very low and very high fluctuation of wages within NOC) ========

fluctuation_long_df_2024 = fluctuation_long_df[fluctuation_long_df['Year'] == 2024]


# Sort by relative fluctuation
top_10_high = (
    fluctuation_long_df_2024
    .sort_values(by='rel_diff_high_low', ascending=False)
    .head(10)
)

bottom_10_low = (
    fluctuation_long_df_2024
    .sort_values(by='rel_diff_high_low', ascending=True)
    .head(10)
)

# Save summary to file
with open(Path(LOG_PATH) / "1-3-eda_summary_fluctuation.txt", "w", encoding='utf-8') as f:
    f.write("=== NATIONAL WAGE FLUCTUATION SUMMARY ===\n\n")
    f.write("Wage Fluctuation Category Distribution Over Time:\n")
    f.write(cluster_counts.to_string(index=False))
    f.write("\n\n")

    f.write("=== Top 10 NOCs with Highest Fluctuation (2024) ===\n")
    f.write(top_10_high.to_string(index=False))
    f.write("\n\n=== Bottom 10 NOCs with Lowest Fluctuation (2024) ===\n")
    f.write(bottom_10_low.to_string(index=False))
    f.write("\n" + "="*80 + "\n")


# No need to create a visualization
# Interesting finding - jobs related to fishing can be found in both
# Low fluctuation and Very High fluctuation groups


#### ========== 1-5-1 NATIONAL WAGES BY BOC FROM 2016 THROUGH 2020 TO 2024 - BARCHART ===========

# Compute average median wages by Broad Occupational Category
boc_growth_df = national_df.groupby('Broad_Category_Code')[
    ['Median_Wage_2016', 'Median_Wage_2020', 'Median_Wage_2024']
].mean().reset_index()

# Reshape to long format for Altair
boc_melted = boc_growth_df.melt(
    id_vars='Broad_Category_Code',
    value_vars=['Median_Wage_2016', 'Median_Wage_2020', 'Median_Wage_2024'],
    var_name='Year',
    value_name='Median_Wage'
)
boc_melted['Year'] = boc_melted['Year'].str.extract(r'(\d{4})').astype(int)

# === Short names for BOC ===
short_names = {
    "0 - Legislative and senior management occupations": "0 - Legislative & Top Management",
    "1 - Business, finance and administration occupations": "1 - Business & Finance",
    "2 - Natural and applied sciences and related occupations": "2 - Sciences",
    "3 - Health occupations": "3 - Health",
    "4 - Education, law, social and community services": "4 - Education, Law & Social Services",
    "5 - Art, culture, recreation and sport": "5 - Arts & Sports",
    "6 - Sales and service occupations": "6 - Sales & Service",
    "7 - Trades, transport and equipment operators": "7 - Trades & Transport",
    "8 - Natural resources, agriculture and production": "8 - Resources & Agriculture",
    "9 - Manufacturing and utilities": "9 - Manufacturing & Utilities"
}

lookup = national_df[['Broad_Category_Code', 'Broad_Category_Name']].drop_duplicates()
lookup["Broad_Category_Short"] = lookup["Broad_Category_Name"].map(short_names)

# Merge in short names
boc_melted = boc_melted.merge(lookup, on='Broad_Category_Code', how='left')

# Add number of NOCs per BOC
nocs_per_boc = national_df.groupby('Broad_Category_Code')['NOC_2021'].nunique().reset_index()
nocs_per_boc.rename(columns={'NOC_2021': 'NOC_Count'}, inplace=True)
boc_melted = boc_melted.merge(nocs_per_boc, on='Broad_Category_Code', how='left')

# Calculate % growth to 2016
base_2016 = boc_melted[boc_melted['Year'] == 2016][['Broad_Category_Code', 'Median_Wage']]
base_2016.rename(columns={'Median_Wage': 'Base_2016'}, inplace=True)
boc_melted = boc_melted.merge(base_2016, on='Broad_Category_Code', how='left')
boc_melted['Growth_to_2016'] = ((boc_melted['Median_Wage'] - boc_melted['Base_2016']) / boc_melted['Base_2016']) * 100

# Set readable category order
category_order = lookup.sort_values('Broad_Category_Code')['Broad_Category_Short'].tolist()
boc_melted['Broad_Category_Short'] = pd.Categorical(
    boc_melted['Broad_Category_Short'], categories=category_order, ordered=True
)

# Build Altair grouped bar chart
chart = alt.Chart(boc_melted).mark_bar().encode(
    x=alt.X('Broad_Category_Short:N', title='Broad Occupational Category', sort=category_order),
    xOffset='Year:O',
    y=alt.Y('Median_Wage:Q', title='Median Hourly Wage ($)'),
    color=alt.Color('Year:O', title='Year', scale=alt.Scale(
        domain=[2016, 2020, 2024],
        range=['#a6cee3', '#1f78b4', '#08306b']
    )),
    tooltip=[
        alt.Tooltip('Year:O', title='Year'),
        alt.Tooltip('Broad_Category_Short:N', title='Category'),
        alt.Tooltip('NOC_Count:Q', title='Number of NOCs'),
        alt.Tooltip('Median_Wage:Q', title='Median Wage ($)', format=',.2f'),
        alt.Tooltip('Growth_to_2016:Q', title='Growth Since 2016 (%)', format='.1f')
    ]
).properties(
    width=650,
    height=400,
    title='Median Wages by Broad Occupational Category (2016–2024)'
).configure_axisX(
    labelAngle=30
)

# Save chart
chart.save(Path(FIGURE_PATH) / "1-5-1-National-median_wages_by_boc_grouped_bar.html")


#### ========== 1-5-2 WAGES HEATMAP, BROAD CATEGORY VS TEER  - NATIONAL ===========
# Group by Broad Category and TEER for each year, and compute average median wage
avg_wages = national_df.groupby(['Broad_Category_Code', 'TEER_Code']).agg({
    'Median_Wage_2016': 'mean',
    'Median_Wage_2020': 'mean',
    'Median_Wage_2024': 'mean'
}).reset_index()

# Melt to long format for plotting
melted = avg_wages.melt(
    id_vars=['Broad_Category_Code', 'TEER_Code'],
    value_vars=['Median_Wage_2016', 'Median_Wage_2020', 'Median_Wage_2024'],
    var_name='Year',
    value_name='Median_Wage'
)

# Extract numeric year
melted['Year'] = melted['Year'].str.extract(r'(\d{4})')

# Prepare a lookup table for names (Broad Category + TEER)
lookup_boc_teer = national_df[['Broad_Category_Code', 'TEER_Code',
                       'Broad_Category_Name', 'TEER_Level_Name', 'Major_Group_Name']].drop_duplicates()

# You can customize these manually or use string shortening logic
short_names = {
    "0 - Legislative and senior management occupations": "0 - Legislative & Top Management",
    "1 - Business, finance and administration occupations": "1 - Business & Finance",
    "2 - Natural and applied sciences and related occupations": "2 - Sciences",
    "3 - Health occupations": "3 - Health",
    "4 - Education, law, social and community services": "4 - Education, Law & Social Services",
    "5 - Art, culture, recreation and sport": "5 - Arts & Sports",
    "6 - Sales and service occupations": "6 - Sales & Service",
    "7 - Trades, transport and equipment operators": "7 - Trades & Transport",
    "8 - Natural resources, agriculture and production": "8 - Resources & Agriculture",
    "9 - Manufacturing and utilities": "9 - Manufacturing & Utilities"}

lookup_boc_teer["Broad_Category_Short"] = lookup_boc_teer["Broad_Category_Name"].map(short_names)

# Merge readable names, use lookup
melted = melted.merge(lookup_boc_teer, on=['Broad_Category_Code', 'TEER_Code'], how='left')

# Build heatmap
heatmap = alt.Chart(melted).mark_rect().encode(
    x=alt.X('Broad_Category_Short:N', title='Broad Occupational Category'),
    y=alt.Y('TEER_Code:O', title='TEER Level'),
    color=alt.Color('Median_Wage:Q', scale=alt.Scale(scheme='reds'), title='Median Wage ($)'),
    tooltip=[
        alt.Tooltip('Year:O'),
        alt.Tooltip('Broad_Category_Name:N', title='Category'),
        alt.Tooltip('TEER_Level_Name:N', title='TEER Level'),
        alt.Tooltip('Median_Wage:Q', title='Median Wage ($)', format=',.0f')
    ]
).properties(width=400, height=400).facet(
    column=alt.Column('Year:O', title=None, header=alt.Header(labelFontSize=14))
)

heatmap.save(Path(FIGURE_PATH) / "1-5-2-National-heatmap_national_by_broad_teer.html")


### ===  1-5-3 KDE PLOT – NATIONAL WAGE DISTRIBUTION BY BROAD CATEGORY ====
kde_melted = national_df.melt(
    id_vars=['Broad_Category_Code', 'TEER_Code'],
    value_vars=['Median_Wage_2016', 'Median_Wage_2020', 'Median_Wage_2024'],
    var_name='Year', value_name='Median_Wage'
)

kde_melted = kde_melted[kde_melted['Broad_Category_Code'] != 0].copy()
kde_melted['Year'] = kde_melted['Year'].str.extract(r'(\d{4})')

kde_melted = kde_melted.merge(
    lookup_boc_teer[['Broad_Category_Code', 'TEER_Code', 'Broad_Category_Name', 'Broad_Category_Short']],
    on=['Broad_Category_Code', 'TEER_Code'], how='left'
)

selection = alt.selection_point(fields=['Broad_Category_Short'], bind='legend')

kde_chart = alt.Chart(kde_melted).transform_density(
    density='Median_Wage',
    groupby=['Broad_Category_Short', 'Year'],
    as_=['Median_Wage', 'Density'],
    extent=[0, 110]
).mark_line(strokeWidth=3).encode(
    x=alt.X('Median_Wage:Q', title='Median Hourly Wage ($)'),
    y=alt.Y('Density:Q'),
    color=alt.Color('Broad_Category_Short:N', scale=alt.Scale(scheme='dark2')),
    opacity=alt.condition(selection, alt.value(1), alt.value(0.15)),
    tooltip=[
        alt.Tooltip('Year:O'),
        alt.Tooltip('Broad_Category_Short:N'),
        alt.Tooltip('Median_Wage:Q', format=',.2f'),
        alt.Tooltip('Density:Q')
    ]
).properties(
    width=600, height=500,
    title='National Wage Distribution by Broad Category (Legend is Clickable)'
).facet(
    column=alt.Column('Year:O')
).add_params(selection)

kde_chart.save(Path(FIGURE_PATH) / "1-5-3-National-kde_wage_distribution_by_category.html")


### ==== 1-6 WAGES HEATMAP BROAD CATEGORY VS TEER  - TOP3 AND BOTTOM 3 PROVINCES =====

# Prepare data for heatmap
def prepare_heatmap_data(sub_df):
    avg = sub_df.groupby(
        ['Province', 'Broad_Category_Code', 'TEER_Code', 'Broad_Category_Name', 'TEER_Level_Name', 'Major_Group_Name']
    ).agg({
        'Median_Wage_2016': 'mean',
        'Median_Wage_2020': 'mean',
        'Median_Wage_2024': 'mean'
    }).reset_index()

    melted = avg.melt(
        id_vars=['Province', 'Broad_Category_Code', 'TEER_Code',
                 'Broad_Category_Name', 'TEER_Level_Name', 'Major_Group_Name'],
        value_vars=['Median_Wage_2016', 'Median_Wage_2020', 'Median_Wage_2024'],
        var_name='Year',
        value_name='Median_Wage'
    )
    melted['Year'] = melted['Year'].str.extract(r'(\d{4})')
    return melted

# Build chart function
def build_heatmap(data, title):
    ordered_provinces = [
        'Prince Edward Island', 'Newfoundland and Labrador', 'Nova Scotia',
        'Ontario', 'British Columbia', 'Alberta'
    ]
    data['Province'] = pd.Categorical(data['Province'], categories=ordered_provinces, ordered=True)

    return alt.Chart(data).mark_rect().encode(
        x=alt.X('Broad_Category_Code:O', title='Broad Occupational Category'),
        y=alt.Y('TEER_Code:O', title='TEER Level'),
        color=alt.Color('Median_Wage:Q', scale=alt.Scale(scheme='reds'), title='Median Wage ($)'),
        tooltip=[
            alt.Tooltip('Province:N', title='Province'),
            alt.Tooltip('Year:O', title='Year'),
            alt.Tooltip('Major_Group_Name:N', title='Major Group'),
            alt.Tooltip('Broad_Category_Name:N', title='Broad Category'),
            alt.Tooltip('TEER_Level_Name:N', title='TEER Level'),
            alt.Tooltip('Median_Wage:Q', title='Median Wage ($)', format=',.0f')
        ]
    ).properties(
        width=200,
        height=200,
        title=title
    ).facet(
        column=alt.Column('Province:N', title=None, sort=ordered_provinces),
        row=alt.Row('Year:O', title=None)
    )

# Keep only valid provincial-level rows (no National, no Region ≠ Province)
provincial_data = merged_wages[
    (merged_wages['Province'] != 'National') &
    (merged_wages['Province'] == merged_wages['Region'])
]

# Drop provinces with < 100 records
# We did not include 3 Territories in our analysis because of the lack of data
# Yukon Territory                22 records
# Northwest Territories          17 records
# Nunavut                         9 records
province_counts = provincial_data['Province'].value_counts()


valid_provinces = province_counts[province_counts >= 100].index
filtered_df = provincial_data[provincial_data['Province'].isin(valid_provinces)]

# Subset for each group
top3_df = filtered_df[filtered_df['Province'].isin(top3)]
bottom3_df = filtered_df[filtered_df['Province'].isin(bottom3)]

#print(list(bottom3), list(top3))

# Build and save
top3_melted = prepare_heatmap_data(top3_df)
bottom3_melted = prepare_heatmap_data(bottom3_df)

# Combine Top 3 and Bottom 3 provinces
combined_df = pd.concat([top3_melted, bottom3_melted])

top3_bottom3_chart = build_heatmap(combined_df, "Top3 and Bottom3 Provinces - Wage Heatmap")
top3_bottom3_chart.save(Path(FIGURE_PATH) / "1-6-Provincial-heatmap_top3_bottom3_provinces.html")


# Save national_df and provincial_df to use in the future analysis
national_df.to_csv(Path(CSV_PATH) / "national_df.csv", index=False)
provincial_df.to_csv(Path(CSV_PATH) / "provincial_df.csv", index=False)
print("Saved national_df and provincial_df for next analysis blocks.")