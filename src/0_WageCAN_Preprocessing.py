# WageCAN Project
# Block 0: Preprocessing
# Author: Nick Shlepov
# Description: This script handles the preprocessing of Canadian wage data (2016–2024),
# including cleaning, NOC code mapping, merging, and feature engineering.

import pandas as pd
from pathlib import Path

# Ensure all columns are displayed
pd.set_option('display.max_columns', None)

# Ensure all necessary output directories exist
Path("../output/figures").mkdir(parents=True, exist_ok=True)
Path("../output/logs").mkdir(parents=True, exist_ok=True)
Path("../output/csv").mkdir(parents=True, exist_ok=True)
Path("../data").mkdir(parents=True, exist_ok=True)

'''
Datasets for this project have been collected from the Government of Canada website
Link: https://open.canada.ca/data/en/dataset/adad580f-76b0-4502-bd05-20c125de9116
Due to the small size, all 3 datasets were downloaded and stored locally.
No changes were applied to the original datasets.
All pre-processing is included in the code below.
'''

# Define paths
DATA_PATH = '../data'
FIGURE_PATH = '../output/figures'
LOG_PATH = '../output/logs'
CSV_PATH = '../output/csv'



# Read datasets
wages2016 = pd.read_csv(Path(DATA_PATH) / 'Ca_wages_2016.csv')
wages2020 = pd.read_csv(Path(DATA_PATH) / 'Ca_wages_2020.csv')
wages2024 = pd.read_csv(Path(DATA_PATH) / 'Ca_wages_2024.csv')

# Initial examination of datasets
with open(Path(LOG_PATH) / "0-1-initial_inspection.txt", "w") as f:
    f.write(f"wages2016 shape: {wages2016.shape}\n")
    f.write(f"wages2020 shape: {wages2020.shape}\n")
    f.write(f"wages2024 shape: {wages2024.shape}\n\n")

    f.write(f"wages2016 columns:\n{list(wages2016.columns)}\n\n")
    f.write(f"wages2020 columns:\n{list(wages2020.columns)}\n\n")
    f.write(f"wages2024 columns:\n{list(wages2024.columns)}\n\n")


# Cleaning function (written as a result of the previous detailed examination and cleaning)
def clean_wage_data(df, year, is_noc2021=False):
    if year == 2024:
        rename_dict = {
            'NOC_CNP': 'NOC_2021',
            'NOC_Title_eng': 'NOC_Title_2021',
            'prov': 'Province',
            'ER_Code_Code_RE': 'ER_Code',
            'ER_Name': 'Region',
            'Low_Wage_Salaire_Minium': f'Low_Wage_{year}',
            'Median_Wage_Salaire_Median': f'Median_Wage_{year}',
            'High_Wage_Salaire_Maximal': f'High_Wage_{year}',
            'Reference_Period': 'Reference_Period',
            'Revision_Date_Date_revision': 'Revision_Date'
        }
    else:
        rename_dict = {
            'NOC_CNP': 'NOC_2016',
            'NOC_Title': 'NOC_Title_2016',
            'PROV': 'Province',
            'ER_Code_Code_RE': 'ER_Code',
            'Low_Wage_Salaire_Minium': f'Low_Wage_{year}',
            'Median_Wage_Salaire_Median': f'Median_Wage_{year}',
            'High_Wage_Salaire_Maximal': f'High_Wage_{year}',
            'Reference_Period': 'Reference_Period',
            'Revision_Date_Date_revision': 'Revision_Date'
        }

    df = df[list(rename_dict.keys())].rename(columns=rename_dict)

    # Remove old period rows
    drop_year = {2016: '2011', 2020: '2016', 2024: '2021'}[year]
    df = df[df['Reference_Period'] != drop_year]

    # Drop NA in main wage column
    df.dropna(subset=[f'Median_Wage_{year}'], inplace=True)

    if year == 2024:
        df.dropna(subset=[f'Low_Wage_{year}', f'High_Wage_{year}'], inplace=True)

    # Normalize title
    title_col = 'NOC_Title_2021' if is_noc2021 else 'NOC_Title_2016'
    df[title_col] = df[title_col].str.lower().str.strip()

    # Drop columns we no longer need
    df.drop(columns=['Reference_Period', 'Revision_Date'], inplace=True)

    return df

# Clean them
wages2016 = clean_wage_data(wages2016, 2016)
wages2020 = clean_wage_data(wages2020, 2020)
wages2024 = clean_wage_data(wages2024, 2024, is_noc2021=True)

# Standardize province names across datasets
province_mapping = {
    "QC": "Quebec", "ON": "Ontario", "AB": "Alberta", "BC": "British Columbia",
    "MB": "Manitoba", "NB": "New Brunswick", "NL": "Newfoundland and Labrador",
    "NS": "Nova Scotia", "PE": "Prince Edward Island", "SK": "Saskatchewan", "YK": "Yukon Territory",
    "NW": "Northwest Territories",    "NU": "Nunavut"
}

wages2016['Province'] = wages2016['Province'].replace(province_mapping)
wages2020['Province'] = wages2020['Province'].replace(province_mapping)
wages2024['Province'] = wages2024['Province'].replace(province_mapping)

# Replace NaN in Province with "National"
wages2020.loc[wages2020['Province'].isna(), 'Province'] = "National"

prov_2016 = set(wages2016['Province'].unique())
prov_2020 = set(wages2020['Province'].unique())


merged_wages = wages2016.merge(
    wages2020,
    on=['NOC_2016', 'ER_Code'],
    how='outer'
)

# If Province exists in both datasets, keep only one
merged_wages['Province'] = merged_wages['Province_y'].fillna(merged_wages['Province_x'])
merged_wages.drop(columns=['Province_x', 'Province_y'], inplace=True)

# If NOC_Title exists in both datasets, keep only one
merged_wages['NOC_Title_2016'] = merged_wages['NOC_Title_2016_y'].fillna(merged_wages['NOC_Title_2016_x'])
merged_wages.drop(columns=['NOC_Title_2016_y', 'NOC_Title_2016_x'], inplace=True)

# Drop rows with missing critical fields
merged_wages = merged_wages.dropna()

# Clean NOC_2016 code
merged_wages["NOC_2016"] = merged_wages["NOC_2016"].str.replace("NOC_", "", regex=False)

# Uncomment to export the intermediate merged dataset (2016 + 2020)
merged_wages.to_csv(Path(CSV_PATH) / "merged_wages.csv", index=False)

# Log results of merging 2016 and 2020 datasets
with open(Path(LOG_PATH) / "0-2-merge_check.txt", "w") as f:
    f.write("=== Merge Check ===\n\n")
    f.write(f"Unique provinces in 2016:\n{prov_2016}\n\n")
    f.write(f"Unique provinces in 2020:\n{prov_2020}\n\n")
    f.write(f"Missing values in Province (2016): {wages2016['Province'].isna().sum()}\n")
    f.write(f"Missing values in Province (2020): {wages2020['Province'].isna().sum()}\n")
    f.write(f"Missing values in Province (2024): {wages2024['Province'].isna().sum()}\n")


# === Handle Transition from NOC_2016 to NOC_2021 ===

# Load original mapping CSV
# https://www.statcan.gc.ca/en/statistical-programs/document/noc2016v1_3-noc2021v1_0
noc_mapping = pd.read_csv(Path(DATA_PATH) / 'noc2016v1_3-noc2021v1_0-eng.csv')

# Drop unnecessary columns
noc_mapping = noc_mapping.drop(columns=['Notes', 'Unnamed: 6'], errors='ignore')

# Rename columns for clarity
noc_mapping.rename(columns={
    'NOC 2016 V1.3 Code': 'NOC_2016',
    'NOC 2016 V1.3 Title': 'NOC_Title_2016',
    ' GSIM Type of Change': 'Change_Type',
    'NOC 2021 V1.0 Code': 'NOC_2021',
    'NOC 2021 V1.0 Title': 'NOC_Title_2021'
}, inplace=True)

# Convert NOC codes to string and ensure leading zeros
noc_mapping['NOC_2016'] = noc_mapping['NOC_2016'].astype(str).str.zfill(4)
noc_mapping['NOC_2021'] = noc_mapping['NOC_2021'].astype(str)

# Store original NOC_Title_2016 values before modifications
noc_title_dict = dict(zip(noc_mapping['NOC_2016'], noc_mapping['NOC_Title_2016']))

# Transitioning from NOC_2016 to NOC_2021 involved changes to both codes and titles.
# However, some older codes were split or transferred into multiple groups.
# This could create inconsistencies when comparing wages, as the criteria for these splits were not provided.
#
# To address this issue, I identified all splits and transfers and used a Large Language Model (LLM)
# to perform sentiment analysis on NOC_2016 and NOC_2021 titles. The goal was to determine the closest matches
# and establish a 1-to-1 mapping wherever possible.
#
# As a result, I created a refined mapping file called best_noc_matches.csv.
# I carefully reviewed it against the original mapping to ensure accuracy.
#
# The refined mappings have been applied, and based on my thorough checks,
# I confirm that they are accurate to the best of my knowledge and belief.


# Load refined 1-to-1 mappings from sentiment analysis
best_matches = pd.read_csv(Path(DATA_PATH) / 'best_noc_matches.csv')

# Convert best_matches to a dictionary for quick lookup
best_match_dict = dict(zip(best_matches['NOC_2016'].astype(str).str.zfill(4),
                           best_matches['NOC_2021'].astype(str)))

# Find NOC_2016 codes that require replacement
noc_to_replace = set(best_match_dict.keys())

# Remove those rows from noc_mapping before adding the refined ones
noc_mapping = noc_mapping[~noc_mapping['NOC_2016'].isin(noc_to_replace)]

# Append refined best_matches rows with restored titles
best_matches['NOC_2016'] = best_matches['NOC_2016'].astype(str).str.zfill(4)
best_matches['NOC_2021'] = best_matches['NOC_2021'].astype(str)
best_matches['NOC_Title_2016'] = best_matches['NOC_2016'].map(noc_title_dict)  # Restore correct titles
noc_mapping = pd.concat([noc_mapping, best_matches], ignore_index=True)

# Normalize NOC_Title column
noc_mapping['NOC_Title_2021'] = noc_mapping['NOC_Title_2021'].str.lower().str.strip()

# Remove duplicates
noc_mapping.drop_duplicates(subset=['NOC_2016', 'NOC_2021'], inplace=True)

# Ensure 1-to-1 mapping for all NOC_2016 → NOC_2021
noc_mapping = noc_mapping.drop_duplicates(subset=['NOC_2016'], keep='first')

# Log mapping integrity check
with open(Path(LOG_PATH) / "0-3-noc_mapping_check.txt", "w") as f:
    f.write(f"Duplicated NOC_2016 entries: {noc_mapping.duplicated(subset=['NOC_2016']).sum()}\n")
    multi_mapped = noc_mapping.groupby('NOC_2016')['NOC_2021'].nunique().loc[lambda x: x > 1]
    f.write(f"NOC_2016 mapped to multiple NOC_2021 (should be 0): {len(multi_mapped)}\n")

# Save the final updated mapping
noc_mapping.to_csv(Path(CSV_PATH) / 'updated_noc_mapping_final.csv', index=False)

# MAP NOC_2016 to NOC_2021 USING THE MAPPING DATASET
merged_wages = merged_wages.merge(noc_mapping[['NOC_2016', 'NOC_2021', 'NOC_Title_2021']],
                                  on='NOC_2016',
                                  how='left')
merged_wages['NOC_2021'] = merged_wages['NOC_2021'].astype(str).str.zfill(5)


# MERGE 2016-2020 MAPPED DATASET TO 2024 DATASET

# Remove "NOC_" prefix from NOC_2021 column
wages2024["NOC_2021"] = wages2024["NOC_2021"].str.replace("NOC_", "", regex=False)

merged_wages = merged_wages.merge(
    wages2024,
    on=['NOC_2021', 'ER_Code'],
    how='left'
)

# If Province exists in both datasets, keep only one
merged_wages['Province'] = merged_wages['Province_x'].fillna(merged_wages['Province_y'])
merged_wages.drop(columns=['Province_x', 'Province_y'], inplace=True)

# If NOC_Title_2021 exists in both datasets, keep only one
merged_wages['NOC_Title_2021'] = merged_wages['NOC_Title_2021_x'].fillna(merged_wages['NOC_Title_2021_y'])
merged_wages.drop(columns=['NOC_Title_2021_x', 'NOC_Title_2021_y'], inplace=True)

# Define the desired column order
cols = [
    "ER_Code", "NOC_2016", "NOC_2021",
    "Low_Wage_2016", "Median_Wage_2016", "High_Wage_2016",
    "Low_Wage_2020", "Median_Wage_2020", "High_Wage_2020",
    "Low_Wage_2024", "Median_Wage_2024", "High_Wage_2024",
    "Province", "Region", "NOC_Title_2021"
]

# Reorder the DataFrame
merged_wages = merged_wages[cols]

# Extract codes from NOC_2021
merged_wages['Broad_Category_Code'] = merged_wages['NOC_2021'].str[0]
merged_wages['TEER_Code'] = merged_wages['NOC_2021'].str[1]
merged_wages['Major_Group_Code'] = merged_wages['NOC_2021'].str[:2]

# Mapping dictionaries
broad_category_names = {
    '0': '0 - Legislative and senior management occupations',
    '1': '1 - Business, finance and administration occupations',
    '2': '2 - Natural and applied sciences and related occupations',
    '3': '3 - Health occupations',
    '4': '4 - Education, law, social and community services',
    '5': '5 - Art, culture, recreation and sport',
    '6': '6 - Sales and service occupations',
    '7': '7 - Trades, transport and equipment operators',
    '8': '8 - Natural resources, agriculture and production',
    '9': '9 - Manufacturing and utilities'
}

teer_level_names = {
    '0': '0 - Management occupations',
    '1': '1 - University degree usually required',
    '2': '2 - College diploma/apprenticeship (2+ yrs) or supervisory',
    '3': '3 - College diploma/apprenticeship (<2 yrs) or >6mo training',
    '4': '4 - Secondary diploma or several weeks training',
    '5': '5 - Short-term demo or no formal education'
}

major_group_names = {
    '00': 'Legislative and senior managers',
    '10': 'Mid-management: admin/finance/communications',
    '11': 'Finance and business professionals',
    '12': 'Supervisors/admin specialists',
    '13': 'Admin and logistics occupations',
    '14': 'Support and supply chain occupations',
    '20': 'Mid-management: engineering, IT, sciences',
    '21': 'Natural/applied science professionals',
    '22': 'Science/tech technicians',
    '30': 'Mid-management in healthcare',
    '31': 'Health professionals',
    '32': 'Health technicians',
    '33': 'Health support workers',
    '40': 'Managers: education/social/public services',
    '41': 'Education, law, social professionals',
    '42': 'Public protection and legal support',
    '43': 'Education/legal/public assistants',
    '44': 'Care/public protection support',
    '45': 'Student monitors, crossing guards, etc.',
    '50': 'Mid-management in culture/sport',
    '51': 'Art/culture professionals',
    '52': 'Culture/sport technicians',
    '53': 'Culture/sport general workers',
    '54': 'Support in sport',
    '55': 'Support in art and culture',
    '60': 'Mid-management: retail/services',
    '62': 'Sales/service supervisors',
    '63': 'General sales/service workers',
    '64': 'Customer/personal service reps',
    '65': 'Sales/service support',
    '70': 'Mid-management: trades and transport',
    '72': 'Technical trades and transport',
    '73': 'General trades',
    '74': 'Transport ops and maintenance',
    '75': 'Labourers in trades/transport',
    '80': 'Mid-management: production/agriculture',
    '82': 'Supervisors: natural resources/agriculture',
    '83': 'Occupations in natural resources/agriculture',
    '84': 'Workers: resources/agriculture',
    '85': 'Labourers: harvesting/landscaping',
    '90': 'Mid-management: manufacturing/utilities',
    '92': 'Supervisors: manufacturing/utilities',
    '93': 'Control ops/assemblers/inspectors',
    '94': 'Machine operators: manufacturing',
    '95': 'Labourers: manufacturing/utilities'
}

# Apply mappings
merged_wages['Broad_Category_Name'] = merged_wages['Broad_Category_Code'].map(broad_category_names)
merged_wages['TEER_Level_Name'] = merged_wages['TEER_Code'].map(teer_level_names)
merged_wages['Major_Group_Name'] = merged_wages['Major_Group_Code'].map(major_group_names)

# Drop rows with any NaNs in key columns needed for analysis
merged_wages = merged_wages.dropna(
    subset=[
        'Median_Wage_2016', 'Median_Wage_2020', 'Median_Wage_2024',
        'NOC_2021', 'NOC_Title_2021', 'Province', 'Region',
        'Broad_Category_Code', 'TEER_Code', 'Major_Group_Code',
        'Broad_Category_Name', 'TEER_Level_Name', 'Major_Group_Name'
    ]
)

# Log the number of records in the final merged dataset
with open(Path(LOG_PATH) / "0-3-noc_mapping_check.txt", "a") as f:
    f.write(f"Final dataset contains: {len(merged_wages)} records\n")

# Save the final 2016-2024 dataset mapped from NOC_2016 to NOC_2021
merged_wages.to_csv(Path(CSV_PATH) / "merged_wages_2016_2024_mapped_to_NOC2021.csv", index=False)
print("=" * 80)
print("Final merged dataset saved. Proceeding to EDA.")
print("=" * 80)

