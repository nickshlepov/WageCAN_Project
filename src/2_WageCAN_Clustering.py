# WageCAN Project
# Block 2: Clustering Analysis
# Author: Nick Shlepov
# Description:
# This script performs clustering analyses on Canadian wage data (2016–2024),
# using K-Means and Hierarchical clustering techniques at occupational and provincial levels.
# Outputs include interactive cluster visualizations, cluster statistics, and identification of outlier occupations.

import pandas as pd
from pathlib import Path
import altair as alt

from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


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

# Read csv files created in Block 0
merged_wages = pd.read_csv(Path(CSV_PATH) / "merged_wages_2016_2024_mapped_to_NOC2021.csv")
national_df = pd.read_csv(Path(CSV_PATH) / "national_df.csv")


### =========== CLUSTERING =====================
### NOC-based analysis

# Filter and clean occupation data
occupation_df = merged_wages.drop_duplicates(subset='NOC_2021').copy()
occupation_df['TEER_Code'] = pd.to_numeric(occupation_df['TEER_Code'], errors='coerce')
occupation_df = occupation_df.dropna(subset=[
    'Median_Wage_2016', 'Median_Wage_2020', 'Median_Wage_2024', 'TEER_Code'
])

# Feature selection and scaling
features = occupation_df[['Median_Wage_2016', 'Median_Wage_2020', 'Median_Wage_2024', 'TEER_Code']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)


### ===== 2-1-1 K-Means clustering (k=4) ==========

kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=10)
occupation_df['Cluster_k4'] = kmeans_4.fit_predict(X_scaled)

# Compute cluster summary stats
summary_stats_4 = occupation_df.groupby('Cluster_k4')[[
    'Median_Wage_2016', 'Median_Wage_2020', 'Median_Wage_2024', 'TEER_Code'
]].mean().round(2)

# Assign cluster names based on wages + TEER
cluster_summary = summary_stats_4.copy()
sorted_clusters = cluster_summary.sort_values(
    by=['Median_Wage_2024', 'TEER_Code'], ascending=[False, True]
).index.tolist()

label_mapping = {
    sorted_clusters[1]: "3 - High Wage, High Skill",
    sorted_clusters[2]: "2 - Mid Wage, Mid Skill",
    sorted_clusters[3]: "1 - Entry-Level & Low Wage",
    sorted_clusters[0]: "4 - Specialized / Outlier Roles"
}

occupation_df['Cluster_Label_k4'] = occupation_df['Cluster_k4'].map(label_mapping)
summary_stats_4['Cluster_Label'] = summary_stats_4.index.map(label_mapping)
summary_stats_4.index = summary_stats_4['Cluster_Label']
summary_stats_4.drop(columns='Cluster_Label', inplace=True)

ordered_labels = [
    "1 - Entry-Level & Low Wage",
    "2 - Mid Wage, Mid Skill",
    "3 - High Wage, High Skill",
    "4 - Specialized / Outlier Roles",
]

summary_stats_4 = summary_stats_4.loc[ordered_labels]


# PCA for 2D projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
occupation_df['PCA1'] = X_pca[:, 0]
occupation_df['PCA2'] = X_pca[:, 1]

# Build Altair interactive scatter plot
chart = alt.Chart(occupation_df).mark_circle(size=120, opacity=0.8).encode(
    x=alt.X('PCA1:Q', title='Wage Level & TEER Blend'),
    y=alt.Y('PCA2:Q', title='Wage Trend / Skill Pattern'),
    color=alt.Color('Cluster_Label_k4:N', title='Occupation Cluster', scale=alt.Scale(scheme='dark2')),
    tooltip=[
        alt.Tooltip('NOC_2021:N'),
        alt.Tooltip('NOC_Title_2021:N', title='Occupation'),
        alt.Tooltip('Cluster_Label_k4:N', title='Cluster'),
        alt.Tooltip('Broad_Category_Name:N', title='Broad Category'),
        alt.Tooltip('TEER_Level_Name:N', title='TEER Level'),
        alt.Tooltip('Median_Wage_2016:Q', title='Wage 2016 ($)', format=',.2f'),
        alt.Tooltip('Median_Wage_2024:Q', title='Wage 2024 ($)', format=',.2f')
    ]
).properties(
    width=750,
    height=500,
    title='Occupation Clusters (k=4) – PCA Projection [Interactive Chart]'
).interactive()

# Save chart and print summary
chart.save(Path(FIGURE_PATH) / "2-1-1-Occupation-clusters_k4_Altair.html")

## === k=4 evaluation ====

# Get centroids
centroids_k4 = kmeans_4.cluster_centers_

# Assign each point its distance to the cluster centroid
distances_k4 = cdist(X_scaled, centroids_k4)
occupation_df['Distance_to_Centroid_k4'] = [distances_k4[i, cluster] for i, cluster in enumerate(occupation_df['Cluster_k4'])]

# Evaluate average distance per cluster
cluster_distance_stats_k4 = occupation_df.groupby('Cluster_k4')['Distance_to_Centroid_k4'].agg(['mean', 'std', 'max']).round(3)
cluster_distance_stats_k4['Label'] = cluster_distance_stats_k4.index.map(label_mapping)
cluster_distance_stats_k4.set_index('Label', inplace=True)

#### ========2-1-2 K-Means clustering (k=7)===========

kmeans_7 = KMeans(n_clusters=7, random_state=42, n_init=10)
occupation_df['Cluster_k7'] = kmeans_7.fit_predict(X_scaled)

# Compute cluster summary stats
summary_stats_7 = occupation_df.groupby('Cluster_k7')[[
    'Median_Wage_2016', 'Median_Wage_2020', 'Median_Wage_2024', 'TEER_Code'
]].mean().round(2)

# Assign descriptive labels based on sorted wage & TEER
cluster_summary = summary_stats_7.copy()
sorted_clusters = cluster_summary.sort_values(
    by=['Median_Wage_2024', 'TEER_Code'], ascending=[False, True]
).index.tolist()

label_mapping_7 = {
    sorted_clusters[0]: "7 - Specialized / Outlier Roles",
    sorted_clusters[1]: "6 - High Wage, High Skill",
    sorted_clusters[2]: "5 - High Wage, Mid Skill",
    sorted_clusters[3]: "4 - Mid Wage, Mid Skill",
    sorted_clusters[4]: "2 - Low Wage, Low Skill",
    sorted_clusters[5]: "3 - Low Wage, Mid Skill",
    sorted_clusters[6]: "1 - Very Low Wage, Minimal Skill"
}

occupation_df['Cluster_Label_k7'] = occupation_df['Cluster_k7'].map(label_mapping_7)
summary_stats_7['Cluster_Label'] = summary_stats_7.index.map(label_mapping_7)
summary_stats_7.index = summary_stats_7['Cluster_Label']
summary_stats_7.drop(columns='Cluster_Label', inplace=True)

ordered_labels = [
    "1 - Very Low Wage, Minimal Skill",
    "2 - Low Wage, Low Skill",
    "3 - Low Wage, Mid Skill",
    "4 - Mid Wage, Mid Skill",
    "5 - High Wage, Mid Skill",
    "6 - High Wage, High Skill",
    "7 - Specialized / Outlier Roles"
]

summary_stats_7 = summary_stats_7.loc[ordered_labels]

# PCA for 2D projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
occupation_df['PCA1'] = X_pca[:, 0]
occupation_df['PCA2'] = X_pca[:, 1]

# Build Altair interactive scatter plot
chart = alt.Chart(occupation_df).mark_circle(size=120, opacity=0.8).encode(
    x=alt.X('PCA1:Q', title='Wage Level & TEER Blend'),
    y=alt.Y('PCA2:Q', title='Wage Trend / Skill Pattern'),
    color=alt.Color('Cluster_Label_k7:N', title='Occupation Cluster', scale=alt.Scale(scheme='dark2')),
    tooltip=[
        alt.Tooltip('NOC_2021:N'),
        alt.Tooltip('NOC_Title_2021:N', title='Occupation'),
        alt.Tooltip('Cluster_Label_k7:N', title='Cluster'),
        alt.Tooltip('Broad_Category_Name:N', title='Broad Category'),
        alt.Tooltip('TEER_Level_Name:N', title='TEER Level'),
        alt.Tooltip('Median_Wage_2016:Q', title='Wage 2016 ($)', format=',.2f'),
        alt.Tooltip('Median_Wage_2024:Q', title='Wage 2024 ($)', format=',.2f')
    ]
).properties(
    width=750,
    height=500,
    title='Occupation Clusters (k=7) – PCA Projection [Interactive Chart]'
).interactive()

# Save chart and print summary
chart.save(Path(FIGURE_PATH) / "2-1-2-Occupation-clusters_k7_Altair.html")

## === k=7 evaluation ====
# Get centroids
centroids_k7 = kmeans_7.cluster_centers_

# Assign each point its distance to the cluster centroid
distances_k7 = cdist(X_scaled, centroids_k7)
occupation_df['Distance_to_Centroid_k7'] = [distances_k7[i, cluster] for i, cluster in enumerate(occupation_df['Cluster_k7'])]

# Evaluate average distance per cluster
cluster_distance_stats_k7 = occupation_df.groupby('Cluster_k7')['Distance_to_Centroid_k7'].agg(['mean', 'std', 'max']).round(3)
cluster_distance_stats_k7['Label'] = cluster_distance_stats_k7.index.map(label_mapping_7)
cluster_distance_stats_k7.set_index('Label', inplace=True)
cluster_distance_stats_k7 = cluster_distance_stats_k7.loc[ordered_labels]



### ===== PROVINCIAL BASED K-MEAN =======

# 2-2-1 Do wages for the same jobs vary significantly from Province to Province ?
# Filter to provincial-level data (no National or Territories)
prov_df = merged_wages[
    (merged_wages['Province'] != 'National') &
    (merged_wages['Region'] == merged_wages['Province']) &
    (~merged_wages['Province'].isin(['Yukon Territory', 'Northwest Territories', 'Nunavut']))
].copy()

# Pivot: NOC_2021 × Province → Median_Wage_2024
pivot = prov_df.pivot_table(
    index='NOC_2021',
    columns='Province',
    values='Median_Wage_2024',
    aggfunc='mean'
)

# Get national median wage per NOC_2021
national_avg = national_df[['NOC_2021', 'Median_Wage_2024']].groupby('NOC_2021').mean()
national_avg.columns = ['National_Median_Wage_2024']

# Merge to get fallback wage values
pivot_filled = pivot.copy()
pivot_filled = pivot_filled.merge(national_avg, on='NOC_2021', how='left')

# Fill missing province wages with national median
province_cols = pivot.columns.tolist()
for province in province_cols:
    pivot_filled[province] = pivot_filled[province].fillna(pivot_filled['National_Median_Wage_2024'])

# Drop the fallback column
pivot_filled.drop(columns='National_Median_Wage_2024', inplace=True)

# Drop any rows that still have NaN (i.e., no national wage data)
pivot_filled = pivot_filled.dropna()

# Merge back metadata
meta = prov_df.drop_duplicates(subset='NOC_2021')[[
    'NOC_2021', 'NOC_Title_2021', 'TEER_Code', 'Broad_Category_Name', 'TEER_Level_Name'
]]
meta['TEER_Code'] = pd.to_numeric(meta['TEER_Code'], errors='coerce')
df = pivot_filled.merge(meta, on='NOC_2021')

# Feature scaling
features = df[province_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# PCA projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# K-Means clustering (k=4)
kmeans4 = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster_k4'] = kmeans4.fit_predict(X_scaled)

# Cluster labeling (by variation across provinces)
cluster_variation = df.groupby('Cluster_k4')[province_cols].std().mean(axis=1).sort_values()
sorted_clusters = cluster_variation.index.tolist()

label_map_k4 = {
    sorted_clusters[0]: "1 - Very Low Wage",
    sorted_clusters[1]: "2 - Low Wage",
    sorted_clusters[2]: "3 - Mid Wage",
    sorted_clusters[3]: "4 - High Wage"
}

df['Cluster_Label_k4'] = df['Cluster_k4'].map(label_map_k4)

# Compute wage profile stats per row
df['Avg_Wage'] = df[province_cols].mean(axis=1).round(2)
df['Min_Wage'] = df[province_cols].min(axis=1).round(2)
df['Max_Wage'] = df[province_cols].max(axis=1).round(2)

# Identify province for min/max
df['Min_Wage_Province'] = df[province_cols].idxmin(axis=1)
df['Max_Wage_Province'] = df[province_cols].idxmax(axis=1)

# Altair chart (k=4)
chart = alt.Chart(df).mark_circle(size=120, opacity=0.8).encode(
    x=alt.X('PCA1:Q', title='Overall Wage Level of the NOC'),
    y=alt.Y('PCA2:Q', title='Wage Divergence Across Provinces'),
    color=alt.Color('Cluster_Label_k4:N', title='Wage Consistency Cluster',
                    scale=alt.Scale(scheme='dark2')),
    tooltip=[
        alt.Tooltip('NOC_Title_2021:N', title='Occupation'),
        alt.Tooltip('Cluster_Label_k4:N', title='Cluster'),
        alt.Tooltip('Avg_Wage:Q', title='Avg Median Wage ($)', format=',.2f'),
        alt.Tooltip('Min_Wage:Q', title='Lowest Median Wage ($)', format=',.2f'),
        alt.Tooltip('Min_Wage_Province:N', title='Lowest in Province'),
        alt.Tooltip('Max_Wage:Q', title='Highest Median Wage ($)', format=',.2f'),
        alt.Tooltip('Max_Wage_Province:N', title='Highest in Province')
    ]
).properties(
    width=750,
    height=500,
    title='Occupation Clusters by National vs Provincial Wage Profiles (k=4) [Interactive Chart]'
).interactive()

chart.save(Path(FIGURE_PATH) / "2-2-1-NOC-provincial-wage-variation_k4.html")

# ===== 2-2-1 Cluster Summary Stats (k=4) =====

# Compute summary statistics (average wage and TEER per cluster)
summary_stats_k4 = df.groupby('Cluster_Label_k4')[province_cols + ['TEER_Code']].mean().round(2)

# Ensure cluster order is consistent with labeling
ordered_labels_k4 = [
    "1 - Very Low Wage",
    "2 - Low Wage",
    "3 - Mid Wage",
    "4 - High Wage"
]
summary_stats_k4 = summary_stats_k4.loc[ordered_labels_k4]

# === Cluster Centroid Evaluation (k=4) ===

# Get centroids
centroids_k4 = kmeans4.cluster_centers_

# Compute distances to centroids
distances_k4 = cdist(X_scaled, centroids_k4)
df['Distance_to_Centroid_k4'] = [distances_k4[i, cluster] for i, cluster in enumerate(df['Cluster_k4'])]

# Evaluate per-cluster distance stats
cluster_distance_stats_k4 = df.groupby('Cluster_k4')['Distance_to_Centroid_k4'].agg(['mean', 'std', 'max']).round(3)

# Map labels and re-order
cluster_distance_stats_k4['Label'] = cluster_distance_stats_k4.index.map(label_map_k4)
cluster_distance_stats_k4.set_index('Label', inplace=True)
cluster_distance_stats_k4 = cluster_distance_stats_k4.loc[ordered_labels_k4]

# === Identify Top 10 Outliers from Cluster Centroids (k=4, Provincial Wage Clustering) ===
top_n = 10  # Adjust as needed

# Sort by distance from centroid
top_outliers_k4 = (
    df.sort_values(by='Distance_to_Centroid_k4', ascending=False)
      .head(top_n)
      .copy()
)

# Select key columns for review
columns_to_view = [
    'NOC_2021', 'NOC_Title_2021', 'Cluster_Label_k4',
    'Avg_Wage', 'Min_Wage', 'Max_Wage',
    'Min_Wage_Province', 'Max_Wage_Province',
    'Distance_to_Centroid_k4'
]

# ==== BIG QUESTION: WHERE DO HIGH-WAGE NOC CLUSTERS LIVE? =====

### ===== 2-2-2 Average wage in each province by cluster ====

# Make sure cluster labels exist
cluster_wage_df = df.copy()

# Unpivot provincial wage columns
melted = cluster_wage_df.melt(
    id_vars=['NOC_2021', 'Cluster_Label_k4'],
    value_vars=province_cols,
    var_name='Province',
    value_name='Median_Wage_2024'
)

# Group: Average wage per province per cluster
heatmap_data = melted.groupby(['Province', 'Cluster_Label_k4'])['Median_Wage_2024'].mean().reset_index()

# Build Altair heatmap
heatmap = alt.Chart(heatmap_data).mark_rect().encode(
    x=alt.X('Cluster_Label_k4:N', title='Cluster (k=4)', sort=sorted(heatmap_data['Cluster_Label_k4'].unique())),
    y=alt.Y('Province:N', title='Province'),
    color=alt.Color('Median_Wage_2024:Q', title='Avg Wage ($)', scale=alt.Scale(scheme='reds')),
    tooltip=[
        alt.Tooltip('Province:N'),
        alt.Tooltip('Cluster_Label_k4:N', title='Cluster'),
        alt.Tooltip('Median_Wage_2024:Q', title='Average Wage ($)', format=',.2f')
    ]
).properties(
    width=350,
    height=400,
    title='Average Wage by Province per Cluster (k=4)'
)

# Save chart
heatmap.save(Path(FIGURE_PATH) / "2-2-2-Province-heatmap_by_cluster_k4.html")


### ====== 2-2-3 NOCs per province/cluster=========

# Long-form (NOC, Province) wage records
long_df = prov_df[['NOC_2021', 'Province', 'Median_Wage_2024']].copy()
long_df = long_df.dropna()

# Add cluster label per NOC
noc_cluster_map = df[['NOC_2021', 'Cluster_Label_k4']].drop_duplicates()
long_df = long_df.merge(noc_cluster_map, on='NOC_2021', how='left')

# Count NOCs per (Province, Cluster)
cluster_counts = (
    long_df.groupby(['Province', 'Cluster_Label_k4'])['NOC_2021']
    .nunique()
    .reset_index()
    .rename(columns={'NOC_2021': 'NOC_Count'})
)

# Count total NOCs per province
total_per_province = (
    long_df.groupby('Province')['NOC_2021']
    .nunique()
    .reset_index()
    .rename(columns={'NOC_2021': 'Total_NOC_Province'})
)

# Merge totals, compute % per cluster
cluster_counts = cluster_counts.merge(total_per_province, on='Province', how='left')
cluster_counts['Percent'] = (cluster_counts['NOC_Count'] / cluster_counts['Total_NOC_Province'] * 100).round(1)

# Altair heatmap
normalized_heatmap = alt.Chart(cluster_counts).mark_rect().encode(
    x=alt.X('Cluster_Label_k4:N', title='Cluster (k=4)', sort=sorted(cluster_counts['Cluster_Label_k4'].unique())),
    y=alt.Y('Province:N', title='Province'),
    color=alt.Color('Percent:Q', title='% of Provincial NOCs', scale=alt.Scale(scheme='reds')),
    tooltip=[
        alt.Tooltip('Province:N'),
        alt.Tooltip('Cluster_Label_k4:N', title='Cluster'),
        alt.Tooltip('NOC_Count:Q', title='NOC Count'),
        alt.Tooltip('Percent:Q', title='% of Provincial NOCs')
    ]
).properties(
    width=350,
    height=400,
    title='Share of Occupations per Cluster by Province (k=4)'
)

normalized_heatmap.save(Path(FIGURE_PATH) / "2-2-3-Province-normalized_frequency_by_cluster_k4.html")


##### ======= 2-2-4 Hierarchical Clustering ==========

# Use the same wage_matrix (already cleaned, pivoted, province columns only)
wage_matrix = pivot_filled.dropna()
province_cols = wage_matrix.columns.tolist()

# Standardize provincial wages
scaler = StandardScaler()
X_scaled = scaler.fit_transform(wage_matrix[province_cols])

# Hierarchical clustering (Ward)
linked = linkage(X_scaled, method='ward')
cluster_labels = fcluster(linked, t=10, criterion='distance')

# Prepare dataframe
alt_df = wage_matrix.copy()
alt_df['Cluster_Hierarchical'] = cluster_labels

# Merge job metadata
alt_df = alt_df.merge(
    df[['NOC_2021', 'NOC_Title_2021', 'TEER_Level_Name', 'Broad_Category_Name']],
    left_index=True,
    right_on='NOC_2021',
    how='left'
)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
alt_df['PCA1'] = X_pca[:, 0]
alt_df['PCA2'] = X_pca[:, 1]


# Assign cluster logic here (e.g., based on avg wage or TEER)
# Compute average provincial wage per cluster
cluster_means = alt_df.groupby('Cluster_Hierarchical')[province_cols].mean()
cluster_avg_wage = cluster_means.mean(axis=1)

# Sort clusters by average wage (ascending)
sorted_clusters = cluster_avg_wage.sort_values().index.tolist()

# Assign descriptive names (customize as you wish)
label_map_hier = {
    sorted_clusters[0]: "1 - Very Low Wage",
    sorted_clusters[1]: "2 - Low Wage",
    sorted_clusters[2]: "3 - Mid Wage (Lower)",
    sorted_clusters[3]: "4 - Mid Wage (Higher)",
    sorted_clusters[4]: "5 - High Wage",
    sorted_clusters[5]: "6 - Very High Wage",
    sorted_clusters[6]: "7 - Specialized / Outlier Roles"
}

# Apply labels to main dataframe
alt_df['Cluster_Label_Hierarchical'] = alt_df['Cluster_Hierarchical'].map(label_map_hier)

# Compute wage profile stats per NOC
alt_df['Avg_Wage'] = alt_df[province_cols].mean(axis=1).round(2)
alt_df['Min_Wage'] = alt_df[province_cols].min(axis=1).round(2)
alt_df['Max_Wage'] = alt_df[province_cols].max(axis=1).round(2)

# Identify province for min/max wage
alt_df['Min_Wage_Province'] = alt_df[province_cols].idxmin(axis=1)
alt_df['Max_Wage_Province'] = alt_df[province_cols].idxmax(axis=1)

alt_df.to_csv(Path(CSV_PATH) / 'alt_df.csv', index=False)

chart = alt.Chart(alt_df).mark_circle(size=120, opacity=0.85).encode(
    x=alt.X('PCA1:Q', title='Provincial Wage Level Profile'),
    y=alt.Y('PCA2:Q', title='Wage Divergence Across Provinces'),
    color=alt.Color('Cluster_Label_Hierarchical:N', title='Wage Cluster (Hierarchical)',
                    scale=alt.Scale(scheme='dark2')),
    tooltip=[
        alt.Tooltip('NOC_Title_2021:N', title='Occupation'),
        alt.Tooltip('Cluster_Label_Hierarchical:N', title='Cluster'),
        alt.Tooltip('Avg_Wage:Q', title='Avg Median Wage ($)', format=',.2f'),
        alt.Tooltip('Min_Wage:Q', title='Lowest Median Wage ($)', format=',.2f'),
        alt.Tooltip('Min_Wage_Province:N', title='Lowest in Province'),
        alt.Tooltip('Max_Wage:Q', title='Highest Median Wage ($)', format=',.2f'),
        alt.Tooltip('Max_Wage_Province:N', title='Highest in Province')
    ]
).properties(
    width=750,
    height=500,
    title='Occupations Clustered by Provincial Wage Patterns (Hierarchical Clustering) [Interactive]'
).interactive()

chart.save(Path(FIGURE_PATH) / "2-2-4-Altair_Hierarchical_clusters_distance15.html")

# ===== 2-2-4 Cluster Summary Stats (Hierarchical) =====

# Compute summary statistics (average wage and TEER per cluster)
summary_stats_hier = alt_df.groupby('Cluster_Label_Hierarchical')[province_cols].mean().round(2)

# Ensure ordered labels for display
ordered_labels_hier = [
    "1 - Very Low Wage",
    "2 - Low Wage",
    "3 - Mid Wage (Lower)",
    "4 - Mid Wage (Higher)",
    "5 - High Wage",
    "6 - Very High Wage",
    "7 - Specialized / Outlier Roles"
]
summary_stats_hier = summary_stats_hier.loc[ordered_labels_hier]

# === 2-2-4 Cluster Centroid Evaluation (Hierarchical) ===

# Calculate cluster centroids in PCA space
centroids_hier = alt_df.groupby('Cluster_Hierarchical')[['PCA1', 'PCA2']].mean().values

# Compute distances to cluster centroids
distances_hier = cdist(alt_df[['PCA1', 'PCA2']], centroids_hier)
alt_df['Distance_to_Centroid_Hier'] = [
    distances_hier[i, label - 1] for i, label in enumerate(alt_df['Cluster_Hierarchical'])
]

# Evaluate per-cluster distance stats
distance_stats_hier = (
    alt_df.groupby('Cluster_Hierarchical')['Distance_to_Centroid_Hier']
    .agg(['mean', 'std', 'max'])
    .round(3)
)

# Map back to human-readable cluster labels
distance_stats_hier['Label'] = distance_stats_hier.index.map(label_map_hier)
distance_stats_hier.set_index('Label', inplace=True)
distance_stats_hier = distance_stats_hier.loc[ordered_labels_hier]

# === Compute Distances to Centroid for Cluster 7 ===
# Select only Cluster 7 rows
cluster_7_mask = alt_df['Cluster_Label_Hierarchical'] == '7 - Specialized / Outlier Roles'
cluster_7_data = alt_df.loc[cluster_7_mask, ['PCA1', 'PCA2']].values

# Compute centroid
centroid_7 = cluster_7_data.mean(axis=0)

# Compute distances and assign back
alt_df['Distance_to_Centroid_7'] = None
alt_df.loc[cluster_7_mask, 'Distance_to_Centroid_7'] = cdist(cluster_7_data, [centroid_7]).flatten()


# === Identify Top 5 Cluster 7 Outliers ===
# Get top 5 farthest occupations in Cluster 7 (regardless of threshold)
top_cluster7_outliers = (
    alt_df[alt_df['Cluster_Label_Hierarchical'] == '7 - Specialized / Outlier Roles']
    .sort_values(by='Distance_to_Centroid_7', ascending=False)
    .head(5)
)

# Display key fields
columns_to_show = [
    'NOC_2021', 'NOC_Title_2021', 'Avg_Wage',
    'Min_Wage', 'Min_Wage_Province', 'Max_Wage', 'Max_Wage_Province',
    'Distance_to_Centroid_7'
]

#### ========== Save clustering results ==========

# Save Occupation-based clustering k=4 and k=7
with open(Path(LOG_PATH) / "2-1-clustering_summary_k=n.txt", "w", encoding="utf-8") as f:
    f.write("Summary Stats for Clusters (k=4) with Descriptive Labels:\n")
    f.write(summary_stats_4.to_string())
    f.write("\n\n=== k=4 Cluster Centroid Evaluation ===\n")
    f.write(cluster_distance_stats_k4.to_string())
    f.write("\n" + "=" * 80 + "\n")

    f.write("Summary Stats for Clusters (k=7) with Descriptive Labels:\n")
    f.write(summary_stats_7.to_string())
    f.write("\n\n=== k=7 Cluster Centroid Evaluation ===\n")
    f.write(cluster_distance_stats_k7.to_string())
    f.write("\n" + "=" * 80 + "\n")


# Save Provincial k=4 and Hierarchical summaries and outliers
with open(Path(LOG_PATH) / "2-2-clustering_summary_hierarchical.txt", "w", encoding="utf-8") as f:
    f.write("Summary Stats for Provincial Wage Clusters (k=4):\n")
    f.write(summary_stats_k4.to_string())
    f.write("\n\n=== Provincial k=4 Cluster Centroid Evaluation ===\n")
    f.write(cluster_distance_stats_k4.to_string())
    f.write("\n" + "=" * 80 + "\n")

    f.write(f"\nTop {top_n} Distant NOCs from Cluster Centroids (Provincial Clustering k=4):\n")
    f.write(top_outliers_k4[columns_to_view].to_string(index=False))
    f.write("\n" + "=" * 80 + "\n")

    f.write("Summary Stats for Clusters (Hierarchical) with Descriptive Labels:\n")
    f.write(summary_stats_hier.to_string())
    f.write("\n\n=== Hierarchical Cluster Centroid Evaluation ===\n")
    f.write(distance_stats_hier.to_string())
    f.write("\nCluster 7 is by far the most scattered cluster, likely includes edge cases or rare occupations\n")
    f.write("=" * 80 + "\n")

    f.write("Top 5 Distant NOCs in Cluster 7 – Specialized / Outlier Roles:\n")
    f.write(top_cluster7_outliers[columns_to_show].to_string(index=False))
    f.write("\n" + "=" * 80 + "\n")

print("Clustering analysis completed. Results saved to /output/figures and /output/logs.")

