# WageCAN Project
# Block 3: Modeling and Wage Prediction
# Author: Nick Shlepov
# Description:
# This script builds models to predict Canadian wages (2016–2024) using TEER, Broad Categories, Provinces, and NOC Title embeddings.
# Outputs include model evaluations, cross-validation scores, and wage stability analysis based on historical trends.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

import altair as alt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sentence_transformers import SentenceTransformer

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
provincial_df = pd.read_csv(Path(CSV_PATH) / "provincial_df.csv")
national_df = pd.read_csv(Path(CSV_PATH) / "national_df.csv")


##### ======= FINAL STEP BEFORE MODELING: Mapping Wage Patterns by TEER and BOC Influence =====

# Create a CustomIndex in the format "TEER + BOC Rank", where:
# - TEER is the TEER_Code (0–5)
# - BOC Rank is the ranking of Broad Occupational Categories by their average national median wage
# in 2024 (higher wage = lower rank)
# This index will be used on the x-axis to test the hypothesis that TEER and BOC together produce a strong wage pattern

# Compute average 2024 wage by BOC
boc_rank_df = national_df.groupby("Broad_Category_Code")["Median_Wage_2024"].mean().reset_index()

# Rank BOC codes by their average wage (higher wage = lower rank number)
boc_rank_df["BOC_Rank"] = boc_rank_df["Median_Wage_2024"].rank(method="min", ascending=False).astype(int) - 1

# Merge the rank into national_df
national_df = national_df.merge(
    boc_rank_df[["Broad_Category_Code", "BOC_Rank"]],
    on="Broad_Category_Code", how="left"
)

# Ensure TEER_Code and BOC_Rank are numeric (or at least no leading zeros)
national_df['TEER_Code'] = national_df['TEER_Code'].astype(str)
national_df['BOC_Rank'] = national_df['BOC_Rank'].astype(str)
national_df["CustomIndex"] = national_df.apply(
    lambda row: f"{row['TEER_Code']}-{int(row['BOC_Rank'])}", axis=1
)
#national_df['CustomIndex'] = national_df['TEER_Code'] + national_df['BOC_Rank']

# Create short name mapping again
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

# Create and merge lookup for short names
lookup = national_df[['Broad_Category_Code', 'Broad_Category_Name']].drop_duplicates()
lookup['Broad_Category_Short'] = lookup['Broad_Category_Name'].map(short_names)

# Merge short name into national_df
national_df = national_df.merge(lookup, on=['Broad_Category_Code', 'Broad_Category_Name'], how='left')

# Filter to 2024 records only
national_df_2024 = national_df[national_df["Median_Wage_2024"].notna()].copy()

# Ensure CustomIndex is treated as string for correct plotting
national_df_2024["CustomIndex"] = national_df_2024["CustomIndex"].astype(str)

#### ======= 3-1-1 Median Wage by Custom Index (TEER × BOC Rank) — National 2024 =====

selection = alt.selection_point(fields=['Broad_Category_Short'], bind='legend')

# Build Altair scatter plot
custom_scatter = alt.Chart(national_df_2024).mark_circle(size=60).encode(
    x=alt.X('CustomIndex:O', title='Custom Index (TEER - BOC Rank)'),
    y=alt.Y('Median_Wage_2024:Q', title='Median Wage ($)', scale=alt.Scale(zero=False)),
    color=alt.Color('Broad_Category_Short:N', title='Broad Occupational Category',
                    scale=alt.Scale(scheme='category10')),
    opacity=alt.condition(selection, alt.value(1), alt.value(0.15)),
    tooltip=[
        alt.Tooltip('NOC_2021:N', title='NOC Code'),
        alt.Tooltip('NOC_Title_2021:N', title='NOC Title'),
        alt.Tooltip('Broad_Category_Short:N', title='BOC'),
        alt.Tooltip('TEER_Level_Name:N', title='TEER'),
        alt.Tooltip('Median_Wage_2024:Q', title='Median Wage ($)', format=',.2f'),
        alt.Tooltip('CustomIndex:N', title='Custom Index')
    ]
).add_params(
    selection
).properties(
    width=1000,
    height=700,
    title='Median Wage by Custom Index (TEER × BOC Rank) — National 2024'
).configure_axisX(
    labelAngle=0
).interactive()

custom_scatter.save(Path(FIGURE_PATH) / "3-1-1-National-custom_index_scatter_2024.html")


### ==== BRIDGE INTO MODELLING: SIMPLE LINEAR REGRESSION ======
### === 3-1-2-National-custom_index_regression_line =====

# Convert TEER and BOC rank back to integers for modeling
national_df_2024["TEER_Code"] = national_df_2024["TEER_Code"].astype(int)
national_df_2024["BOC_Rank"] = national_df_2024["BOC_Rank"].astype(int)

# Train a simple linear regression model
X = national_df_2024[["TEER_Code", "BOC_Rank"]]
y = national_df_2024["Median_Wage_2024"]
reg = LinearRegression()
reg.fit(X, y)

# Predict values across all combinations of TEER (0–5) and BOC_Rank (0–9)
teer_vals = range(0, 6)
boc_vals = range(0, 10)
prediction_grid = pd.DataFrame([(t, b) for t in teer_vals for b in boc_vals], columns=["TEER_Code", "BOC_Rank"])
prediction_grid["Predicted_Wage"] = reg.predict(prediction_grid)

# Create numeric x position for prediction line
prediction_grid["Index_Position"] = prediction_grid["TEER_Code"] * 10 + prediction_grid["BOC_Rank"]

# Map same numeric position to national_df_2024 for consistency
national_df_2024["Index_Position"] = national_df_2024["TEER_Code"] * 10 + national_df_2024["BOC_Rank"]

# Base scatter plot (keep CustomIndex for user readability)
base_scatter = alt.Chart(national_df_2024).mark_circle(size=60).encode(
    x=alt.X('CustomIndex:O', title='Custom Index (TEER - BOC Rank)',
            sort=sorted(national_df_2024['CustomIndex'].unique(),
                        key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))),
    y=alt.Y('Median_Wage_2024:Q', title='Median Wage ($)', scale=alt.Scale(zero=False)),
    color=alt.Color('Broad_Category_Short:N', title='Broad Occupational Category',
                    scale=alt.Scale(scheme='category10')),
    opacity=alt.condition(selection, alt.value(1), alt.value(0.15)),
    tooltip=[
        alt.Tooltip('NOC_2021:N', title='NOC Code'),
        alt.Tooltip('NOC_Title_2021:N', title='NOC Title'),
        alt.Tooltip('Broad_Category_Short:N', title='BOC'),
        alt.Tooltip('TEER_Level_Name:N', title='TEER'),
        alt.Tooltip('Median_Wage_2024:Q', title='Median Wage ($)', format=',.2f'),
        alt.Tooltip('CustomIndex:N', title='Custom Index')
    ]
).add_params(selection)

# Regression line using Index_Position
regression_line = alt.Chart(prediction_grid).mark_line(
    strokeDash=[5, 3], strokeWidth=3, color="black"
).encode(
    x=alt.X('Index_Position:Q', title=None, axis=alt.Axis(labels=False, ticks=False)),
    y=alt.Y('Predicted_Wage:Q'),
    tooltip=[
        alt.Tooltip('Predicted_Wage:Q', title='Predicted Wage ($)', format=',.2f')
    ]
)

# Overlay
final_chart = alt.layer(base_scatter, regression_line).configure_axisX(
    labelAngle=0
).properties(
    width=1000,
    height=700,
    title='Median Wage by Custom Index (TEER × BOC Rank) — National 2024'
).interactive()

# Save
final_chart.save(Path(FIGURE_PATH) / "3-1-2-National-custom_index_regression_line.html")


#### QUESTION : If I study a lot and pick the right industry, will I make decent money in wages? =====

# Predict
y_pred = reg.predict(X)

# Evaluation
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
intercept = reg.intercept_
coefs = reg.coef_

# Cross-validate TEER + BOC Rank model
cv_scores_custom = cross_val_score(reg, X, y, cv=5, scoring='r2')

# Save linear regression summary for Custom Index (TEER × BOC Rank)
with open(Path(LOG_PATH) / "3-1-model_eval_custom_index.txt", "w", encoding="utf-8") as f:
    f.write("QUESTION: If I study a lot and pick the right industry, will I make decent money in wages?\n")
    f.write("\n=== Linear Regression Summary: Custom Index (TEER × BOC Rank) factor Median_Wage_2024 ===\n")
    f.write(f"R²: {r2:.3f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"Intercept: {intercept:.2f}\n")
    f.write(f"TEER coefficient: {coefs[0]:.2f}\n")
    f.write(f"BOC rank coefficient: {coefs[1]:.2f}\n\n")
    f.write(f"Cross-validated R² (5-fold): Mean = {cv_scores_custom.mean():.3f}, "
            f"Std = {cv_scores_custom.std():.3f}\n")
    f.write("=" * 80 + "\n")


# === 5-1-3 National: Actual vs Predicted Median Wages (Simple Linear Regression: TEER + BOC Rank) ===
# Prepare DataFrame for Altair
actual_vs_pred_df = national_df_2024.copy()
actual_vs_pred_df['Predicted_Median_Wage_2024'] = y_pred

# Create selection object (Broad_Category_Short)
selection = alt.selection_point(fields=['Broad_Category_Short'], bind='legend')

# Altair scatter plot
scatter_actual_pred = alt.Chart(actual_vs_pred_df).mark_circle(size=120).encode(
    x = alt.X('Predicted_Median_Wage_2024:Q', title='Predicted Median Wage 2024 ($)'),
    y = alt.Y('Median_Wage_2024:Q', title='Actual Median Wage 2024 ($)'),
    color=alt.Color('Broad_Category_Short:N', title='Broad Occupational Category',
                    scale=alt.Scale(scheme='category10')),
    opacity=alt.condition(selection, alt.value(1), alt.value(0.15)),
    tooltip=[
        alt.Tooltip('NOC_2021:N', title='NOC Code'),
        alt.Tooltip('NOC_Title_2021:N', title='NOC Title'),
        alt.Tooltip('Broad_Category_Short:N', title='BOC'),
        alt.Tooltip('TEER_Level_Name:N', title='TEER Level'),
        alt.Tooltip('Median_Wage_2024:Q', title='Actual Wage ($)', format=',.2f'),
        alt.Tooltip('Predicted_Median_Wage_2024:Q', title='Predicted Wage ($)', format=',.2f')
    ]
).add_params(
    selection
).properties(
    width=800,
    height=800,
    title='Simple Linear Regression (TEER + BOC Rank): Predicted vs Actual 2024 Wages'
).interactive()

# Create ideal perfect line (Actual = Predicted)
perfect_line_actual_pred = alt.Chart(pd.DataFrame({
    'x': [actual_vs_pred_df['Median_Wage_2024'].min(), actual_vs_pred_df['Median_Wage_2024'].max()],
    'y': [actual_vs_pred_df['Median_Wage_2024'].min(), actual_vs_pred_df['Median_Wage_2024'].max()]
})).mark_line(
    color='black',
    strokeDash=[5,5]
).encode(
    x='x:Q',
    y='y:Q'
)

# Combine scatter and perfect line
final_actual_pred_chart = (scatter_actual_pred + perfect_line_actual_pred).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16,
    anchor='start'
)

# Save chart
final_actual_pred_chart.save(Path(FIGURE_PATH) / "3-1-3-National-actual_vs_predicted_simple_linear.html")


# Save answer/interpretation for custom index regression
with open(Path(LOG_PATH) / "3-1-model_eval_custom_index.txt", "a", encoding="utf-8") as f:
    f.write("\nANSWER:\n")
    f.write("Pretty much yes. TEER and BOC rank together explain a good portion of wage variation.\n")
    f.write("Not perfect, but strong for just two factors.\n")
    f.write("=" * 80 + "\n")


##### ================= MODELLING ====================

#### ============== NATIONAL LEVEL ================

### === Baseline model Median Wage vs TEER (Linear Regression) ===

model_df = national_df.dropna(subset=['TEER_Code', 'Median_Wage_2024'])
X = model_df[['TEER_Code']]
y = model_df['Median_Wage_2024']

reg = LinearRegression()
reg.fit(X, y)
y_pred = reg.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
cv_scores_baseline = cross_val_score(reg, X, y, cv=5, scoring='r2')

# Save baseline linear regression summary (TEER → Wage)
with open(Path(LOG_PATH) / "3-2-model_national.txt", "w", encoding="utf-8") as f:
    f.write("=== Linear Regression Summary: TEER_Code factor Median_Wage_2024 ===\n")
    f.write(f"Intercept: {reg.intercept_:.2f}\n")
    f.write(f"Coefficient: {reg.coef_[0]:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R² Score: {r2:.3f}\n\n")
    f.write(f"Cross-validated R² (5-fold): Mean = {cv_scores_baseline.mean():.3f}, "
            f"Std = {cv_scores_baseline.std():.3f}\n")
    f.write("=" * 80 + "\n")



### === Random Forest: TEER + Broad Category ===

df = national_df[['TEER_Code', 'Broad_Category_Code', 'Median_Wage_2024']].dropna()
df['TEER_Code'] = pd.to_numeric(df['TEER_Code'], errors='coerce')
df.dropna(inplace=True)

encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(df[['Broad_Category_Code']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

X = pd.concat([df[['TEER_Code']].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
y = df['Median_Wage_2024'].values

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
y_pred = rf.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
cv_scores_rf = cross_val_score(rf, X, y, cv=5, scoring='r2')
importances = pd.Series(rf.feature_importances_, index=X.columns)

# Save Random Forest model summary (full fit + CV + feature importance)
with open(Path(LOG_PATH) / "3-2-model_national.txt", "a", encoding="utf-8") as f:
    f.write("\n=== Random Forest Regression ===\n")
    f.write(f"RMSE (on full set): {rmse:.2f}\n")
    f.write(f"R² Score (on full set): {r2:.3f}\n")
    f.write(f"Cross-validated R² (5-fold): Mean = {cv_scores_rf.mean():.3f}, "
            f"Std = {cv_scores_rf.std():.3f}\n\n")
    f.write("Top Features by Importance:\n")
    f.write(importances.sort_values(ascending=False).head(10).round(3).to_string())
    f.write("\n" + "=" * 80 + "\n")


### === Gradient Boosting: TEER + Broad Category ===

df = national_df[['TEER_Code', 'Broad_Category_Code', 'Median_Wage_2024']].dropna()
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(df[['Broad_Category_Code']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

X = pd.concat([df[['TEER_Code']].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
y = df['Median_Wage_2024'].values

gbr = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42)
gbr.fit(X, y)
y_pred = gbr.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
cv_scores_gb = cross_val_score(gbr, X, y, cv=5, scoring='r2')
importances = pd.Series(gbr.feature_importances_, index=X.columns)

# Save Gradient Boosting model summary
with open(Path(LOG_PATH) / "3-2-model_national.txt", "a", encoding="utf-8") as f:
    f.write("\n=== Gradient Boosting Regression ===\n")
    f.write(f"RMSE (on full set): {rmse:.2f}\n")
    f.write(f"R² Score (on full set): {r2:.3f}\n")
    f.write(f"Cross-validated R² (5-fold): Mean = {cv_scores_gb.mean():.3f}, "
            f"Std = {cv_scores_gb.std():.3f}\n\n")
    f.write("Top Features by Importance:\n")
    f.write(importances.sort_values(ascending=False).head(10).round(3).to_string())
    f.write("\n" + "=" * 80 + "\n")


## ====== Embedding NOC_Titles =========
# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare your data
df = national_df.copy()
df = df.dropna(subset=['NOC_Title_2021'])

# Generate embeddings for job titles
embeddings = model.encode(df['NOC_Title_2021'].tolist(), show_progress_bar=True)

# Convert to DataFrame
embedding_df = pd.DataFrame(embeddings, columns=[f'embedding_{i}' for i in range(embeddings.shape[1])])

# Merge with original data
df = df.reset_index(drop=True)
df_embed = pd.concat([df, embedding_df], axis=1)

### ==== Embedding-Only Random Forest Model with 5-Fold CV ====

# Prepare features and target
X = df_embed[[f'embedding_{i}' for i in range(embeddings.shape[1])]]
y = df_embed['Median_Wage_2024']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Predict and evaluate on test set
y_pred = model_rf.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

# 5-fold Cross-validation
cv_scores_emb = cross_val_score(model_rf, X, y, cv=5, scoring='r2')

# Save embedding-only model evaluation
with open(Path(LOG_PATH) / "3-2-model_national.txt", "a", encoding="utf-8") as f:
    f.write("\n=== Embedding-Only Random Forest Model ===\n")
    f.write(f"Test RMSE: {rmse:.2f}\n")
    f.write(f"Test R² Score: {r2:.3f}\n")
    f.write(f"Cross-validated R² (5-fold): Mean = {cv_scores_emb.mean():.3f}, "
            f"Std = {cv_scores_emb.std():.3f}\n")
    f.write("=" * 80 + "\n")



### ==== Training Model Using TEER, Broad Category and Embedding  =====

# Filter national data
df = national_df.dropna(subset=['TEER_Code', 'Broad_Category_Code', 'Median_Wage_2024', 'NOC_Title_2021']).copy()

# Shorter TEER labels
teer_short_names = {
    'Management occupations': '0 – Management',
    'University degree usually required': '1 – University',
    'College diploma/apprenticeship (2+ yrs) or supervisory': '2 – College 2+ years',
    'College diploma/apprenticeship (<2 yrs) or >6mo training': '3 – College < 2 yrs',
    'Secondary diploma or several weeks training': '4 – High School',
    'Short-term demo or no formal education': '5 – Short-term demo',
}
df['TEER_Label_Short'] = df['TEER_Level_Name'].map(teer_short_names)

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['NOC_Title_2021'].tolist(), show_progress_bar=True)
embedding_df = pd.DataFrame(embeddings, columns=[f'embedding_{i}' for i in range(embeddings.shape[1])])
embedding_df.index = df.index

# One-hot encode Broad Category
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(df[['Broad_Category_Code']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Broad_Category_Code']))
encoded_df.index = df.index

# Combine features
X = pd.concat([df[['TEER_Code']], encoded_df, embedding_df], axis=1)
y = df['Median_Wage_2024']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

with open(Path(LOG_PATH) / "3-2-model_national.txt", "a", encoding="utf-8") as f:
    f.write("\n=== Combined TEER + Broad Category + Embedding ===\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R² Score: {r2:.3f}\n")
    f.write("=" * 80 + "\n")

# 5-fold Cross-validation
rf = RandomForestRegressor(n_estimators=100, random_state=42)
cv_scores_final = cross_val_score(rf, X, y, cv=5, scoring='r2')

with open(Path(LOG_PATH) / "3-2-model_national.txt", "a", encoding="utf-8") as f:
    f.write("\n=== National Level – 5-Fold CV ===\n")
    f.write(f"Cross-validated R² (5-fold): Mean = {cv_scores_final.mean():.3f}, Std = {cv_scores_final.std():.3f}\n")
    f.write("=" * 80 + "\n")



# === 3-2 National: Actual vs Predicted Median Wages (Random Forest: TEER + Broad Category + Embedding) ===

# Prepare DataFrame for Altair
df_pred = df.loc[y_test.index].copy().reset_index(drop=True)
df_pred['Predicted_Wage_2024'] = y_pred
df_pred['Actual_Wage_2024'] = y_test.values

# Make sure Broad_Category_Short is present
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

# Map short Broad Category names if not already present
if 'Broad_Category_Short' not in df_pred.columns:
    df_pred['Broad_Category_Short'] = df_pred['Broad_Category_Name'].map(short_names)

# Selection for interactive legend (Broad_Category_Short!)
selection = alt.selection_point(fields=['Broad_Category_Short'], bind='legend')

# Define common min/max scale for x and y
wage_min = min(df_pred['Actual_Wage_2024'].min(), df_pred['Predicted_Wage_2024'].min())
wage_max = max(df_pred['Actual_Wage_2024'].max(), df_pred['Predicted_Wage_2024'].max())

# Altair scatter plot
scatter_actual_pred_3_2 = alt.Chart(df_pred).mark_circle(size=120).encode(
    x=alt.X('Predicted_Wage_2024:Q', title='Predicted Median Wage 2024 ($)', scale=alt.Scale(domain=[wage_min, wage_max])),
    y=alt.Y('Actual_Wage_2024:Q', title='Actual Median Wage 2024 ($)', scale=alt.Scale(domain=[wage_min, wage_max])),
    color=alt.Color('Broad_Category_Short:N', title='Broad Occupational Category',
                    scale=alt.Scale(scheme='category10')),
    opacity=alt.condition(selection, alt.value(1), alt.value(0.15)),
    tooltip=[
        alt.Tooltip('NOC_Title_2021:N', title='Occupation'),
        alt.Tooltip('Actual_Wage_2024:Q', title='Actual Wage ($)', format=',.2f'),
        alt.Tooltip('Predicted_Wage_2024:Q', title='Predicted Wage ($)', format=',.2f'),
        alt.Tooltip('Broad_Category_Short:N', title='Broad Category'),
        alt.Tooltip('TEER_Level_Name:N', title='TEER Level')
    ]
).add_params(
    selection
).properties(
    title={
        "text": ["Actual vs Predicted Median Wages (TEER + Broad + Embedding)"],
      "subtitle": ["Test Data Only, 20% of dataset",
                   ""]
    },
    width=800,
    height=800
).interactive()

# Perfect prediction line
perfect_line_3_2 = alt.Chart(pd.DataFrame({
    'x': [wage_min, wage_max],
    'y': [wage_min, wage_max]
})).mark_line(
    color='black',
    strokeDash=[5,5]
).encode(
    x='x:Q',
    y='y:Q'
)

# Combine scatter and perfect line
final_chart_3_2 = (scatter_actual_pred_3_2 + perfect_line_3_2).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16,
    anchor='start'
)

# Save
final_chart_3_2.save(Path(FIGURE_PATH) / "3-2-National-actual_vs_predicted_rf_embedding.html")


# Note:
# The plot shows a reasonable fit on the available data.
# However, cross-validation reveals instability and signs of overfitting,
# indicating that the model's ability to predict new data is limited.


#### ============== PROVINCIAL LEVEL ================

# Same logic as national but we also use Province as a predictor
# Reuse `provincial_df` prepared earlier for boxplots (10 provinces only)

### ====== Baseline Linear Regression (TEER_Code only) ======
baseline_df = provincial_df.dropna(subset=['TEER_Code', 'Median_Wage_2024'])
X = baseline_df[['TEER_Code']]
y = baseline_df['Median_Wage_2024']

reg = LinearRegression()
reg.fit(X, y)
y_pred = reg.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
cv_scores_prov_baseline = cross_val_score(reg, X, y, cv=5, scoring='r2')

with open(Path(LOG_PATH) / "3-3-model_eval_provincial.txt", "w", encoding="utf-8") as f:
    f.write("=== [Provincial] Linear Regression: TEER_Code → Median_Wage_2024 ===\n")
    f.write(f"Intercept: {reg.intercept_:.2f}\n")
    f.write(f"Coefficient: {reg.coef_[0]:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R² Score: {r2:.3f}\n")
    f.write(f"Cross-validated R² (5-fold): Mean = {cv_scores_prov_baseline.mean():.3f}, "
            f"Std = {cv_scores_prov_baseline.std():.3f}\n")
    f.write("=" * 80 + "\n")


### ====== Random Forest with TEER + Broad + Province ======
rf_df = provincial_df.dropna(subset=['TEER_Code', 'Broad_Category_Code', 'Province', 'Median_Wage_2024']).copy()

encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(rf_df[['Broad_Category_Code', 'Province']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Broad_Category_Code', 'Province']))
encoded_df.index = rf_df.index

X = pd.concat([rf_df[['TEER_Code']], encoded_df], axis=1)
y = rf_df['Median_Wage_2024'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
importances = pd.Series(model_rf.feature_importances_, index=X.columns)
cv_scores_prov_rf = cross_val_score(model_rf, X, y, cv=5, scoring='r2')

with open(Path(LOG_PATH) / "3-3-model_eval_provincial.txt", "a", encoding="utf-8") as f:
    f.write("\n=== [Provincial] Random Forest Regression ===\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R² Score: {r2:.3f}\n\n")
    f.write("Top Features by Importance:\n")
    f.write(importances.sort_values(ascending=False).head(10).round(3).to_string())
    f.write("\n\n")
    f.write(f"Cross-validated R² (5-fold): Mean = {cv_scores_prov_rf.mean():.3f}, "
            f"Std = {cv_scores_prov_rf.std():.3f}\n")
    f.write("=" * 80 + "\n")


### ====== Gradient Boosting (TEER + Broad + Province) ======
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(rf_df[['Broad_Category_Code', 'Province']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Broad_Category_Code', 'Province']))
encoded_df.index = rf_df.index

X = pd.concat([rf_df[['TEER_Code']], encoded_df], axis=1)
y = rf_df['Median_Wage_2024']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
gbr = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42)
gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)
importances = pd.Series(gbr.feature_importances_, index=X.columns).sort_values(ascending=False)
cv_scores_prov_gb = cross_val_score(gbr, X, y, cv=5, scoring='r2')

with open(Path(LOG_PATH) / "3-3-model_eval_provincial.txt", "a", encoding="utf-8") as f:
    f.write("\n=== [Provincial] Gradient Boosting Regression ===\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R² Score: {r2:.3f}\n\n")
    f.write("Top Features by Importance:\n")
    f.write(importances.head(10).round(3).to_string())
    f.write("\n\n")
    f.write(f"Cross-validated R² (5-fold): Mean = {cv_scores_prov_gb.mean():.3f}, "
            f"Std = {cv_scores_prov_gb.std():.3f}\n")
    f.write("=" * 80 + "\n")


### ====== Generate Embeddings of NOC Titles ======
embed_df = provincial_df.dropna(subset=['NOC_Title_2021', 'TEER_Code', 'Broad_Category_Code', 'Province', 'Median_Wage_2024']).copy()
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embed_model.encode(embed_df['NOC_Title_2021'].tolist(), show_progress_bar=True)
embedding_df = pd.DataFrame(embeddings, columns=[f'embedding_{i}' for i in range(embeddings.shape[1])])
embedding_df.index = embed_df.index

### ====== Embedding-Only Model with Regularization + CV ======
X = embedding_df.copy()
y = embed_df['Median_Wage_2024']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

rf_embed = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
)
rf_embed.fit(X_train, y_train)

# Evaluate on test set
y_pred = rf_embed.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

# Evaluate on training set
train_pred = rf_embed.predict(X_train)
train_rmse = mean_squared_error(y_train, train_pred) ** 0.5
train_r2 = r2_score(y_train, train_pred)

# Cross-validation on full embedding model
cv_scores_prov_emb = cross_val_score(rf_embed, X, y, cv=5, scoring='r2')

with open(Path(LOG_PATH) / "3-3-model_eval_provincial.txt", "a", encoding="utf-8") as f:
    f.write("\n=== [Provincial] Embedding-Only RF Model ===\n")
    f.write(f"Train RMSE: {train_rmse:.2f} | Train R²: {train_r2:.3f}\n")
    f.write(f"Test  RMSE: {rmse:.2f} | Test  R²: {r2:.3f}\n")
    f.write(f"Cross-validated R² (5-fold): Mean = {cv_scores_prov_emb.mean():.3f}, "
            f"Std = {cv_scores_prov_emb.std():.3f}\n")
    f.write("=" * 80 + "\n")


### ====== [Provincial] Final Combined Model with Regularization + CV ======
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(embed_df[['Broad_Category_Code', 'Province']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Broad_Category_Code', 'Province']))
encoded_df.index = embed_df.index

X = pd.concat([embed_df[['TEER_Code']], encoded_df, embedding_df], axis=1)
y = embed_df['Median_Wage_2024']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

rf_final = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
)
rf_final.fit(X_train, y_train)

# Evaluate on test set
y_pred = rf_final.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

# Evaluate on training set
train_pred = rf_final.predict(X_train)
train_rmse = mean_squared_error(y_train, train_pred) ** 0.5
train_r2 = r2_score(y_train, train_pred)

# Cross-validation on full dataset
cv_scores_prov_final = cross_val_score(rf_final, X, y, cv=5, scoring='r2')

with open(Path(LOG_PATH) / "3-3-model_eval_provincial.txt", "a", encoding="utf-8") as f:
    f.write("\n=== [Provincial] Final Combined Model: TEER + Broad + Province + Embedding ===\n")
    f.write(f"Train RMSE: {train_rmse:.2f} | Train R²: {train_r2:.3f}\n")
    f.write(f"Test  RMSE: {rmse:.2f} | Test  R²: {r2:.3f}\n")
    f.write(f"Cross-validated R² (5-fold): Mean = {cv_scores_prov_final.mean():.3f}, "
            f"Std = {cv_scores_prov_final.std():.3f}\n")
    f.write("=" * 80 + "\n")


###### ============== SUMMARY AND VISUALIZATION OF MODELLING ==============

# Collect CV score arrays into a dictionary with clear model names
cv_scores_dict = {
    'National – Linear (TEER + Ranked Broad)': cv_scores_custom,
    'National – Linear (TEER)': cv_scores_baseline,
    'National – RF (TEER + Broad)': cv_scores_rf,
    'National – GBR (TEER + Broad)': cv_scores_gb,
    'National – RF (Embeddings Only)': cv_scores_emb,
    'National – RF (TEER + Broad + Embedding)': cv_scores_final,
    'Provincial – Linear (TEER)': cv_scores_prov_baseline,
    'Provincial – RF (TEER + Broad + Prov)': cv_scores_prov_rf,
    'Provincial – GBR (TEER + Broad + Prov)': cv_scores_prov_gb,
    'Provincial – RF (Embeddings Only)': cv_scores_prov_emb,
    'Provincial – RF (TEER + Broad + Prov + Embedding)': cv_scores_prov_final,
}

# Build summary table
cv_summary_df = pd.DataFrame({
    "Model": list(cv_scores_dict.keys()),
    "CV R² Mean": [scores.mean() for scores in cv_scores_dict.values()],
    "CV R² Std": [scores.std() for scores in cv_scores_dict.values()],
}).sort_values(by="CV R² Mean", ascending=False).reset_index(drop=True)

# Save summary of cross-validated scores
with open(Path(LOG_PATH) / "3-4-model_eval_cv_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n=== Summary of Cross-Validated R² Scores ===\n")
    f.write(cv_summary_df.to_string(index=False))
    f.write("\nInterpretation:\n")
    f.write("TEER is the strongest wage predictor; "
            "adding other features gives minor gains, while embeddings lead to overfitting.:\n")
    f.write("\n" + "=" * 80 + "\n")


# === 3-4 Model Comparison: Cross-Validated R² Scores (Bar Chart) ===

# Colors from light to dark blue (like in TEER plot)
colors = [
    '#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6',
    '#4292c6', '#2171b5', '#08519c', '#08306b', '#041f4a', '#021027'
]
bar_colors = colors * (len(cv_summary_df) // len(colors) + 1)  # repeat if needed

# Sort to keep same order top to bottom
cv_summary_df_sorted = cv_summary_df.sort_values("CV R² Mean", ascending=True).reset_index(drop=True)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(
    cv_summary_df_sorted["Model"],
    cv_summary_df_sorted["CV R² Mean"],
    xerr=cv_summary_df_sorted["CV R² Std"],
    color=colors,
    edgecolor='black'
)

# Style
ax.set_title("Model Comparison – 5-Fold Cross-Validated R² Scores", fontsize=14)
ax.set_xlabel("Cross-Validated R² (Mean ± Std)", fontsize=12)
ax.set_ylabel("")

# Grid and axis styling
ax.invert_yaxis()
ax.grid(True, axis='x', linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

# Tight layout and save
plt.tight_layout()
plt.savefig(Path(FIGURE_PATH) / "3-4-CV_Comparison_BarChart.png")



###### ============== LAST FOR THIS PROJECT ==============
### ===== Did the wages provide a stable pattern of growth? =======
### ===== Can we predict 2024 wages based on 2016 and 2020 info? =====

# Prepare dataset
wage_stability_df = national_df[['NOC_2021', 'NOC_Title_2021', 'Broad_Category_Short', 'TEER_Code',
                                 'Median_Wage_2016', 'Median_Wage_2020', 'Median_Wage_2024']].dropna()

X = wage_stability_df[['Median_Wage_2016', 'Median_Wage_2020']]
y = wage_stability_df['Median_Wage_2024']

# Models
lin_reg = LinearRegression()
rf_reg = RandomForestRegressor(random_state=42)
gb_reg = GradientBoostingRegressor(random_state=42)

models = {
    "Linear Regression": lin_reg,
    "Random Forest": rf_reg,
    "Gradient Boosting": gb_reg
}

results = {}

# Train and evaluate
for name, model in models.items():
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    results[name] = {
        "R²": r2,
        "RMSE": rmse,
        "CV Mean": cv_scores.mean(),
        "CV Std": cv_scores.std(),
        "y_pred": y_pred
    }

# Save results
with open(Path(LOG_PATH) / "3-5-wage_stability_prediction.txt", "w", encoding="utf-8") as f:
    f.write("=== Wage Stability Prediction Results (2016+2020 ➔ 2024) ===\n")
    for name, metrics in results.items():
        f.write(f"{name}:\n")
        f.write(f"  R² = {metrics['R²']:.3f}\n")
        f.write(f"  RMSE = {metrics['RMSE']:.2f}\n")
        f.write(f"  Cross-Validated R² Mean = {metrics['CV Mean']:.3f}, Std = {metrics['CV Std']:.3f}\n")
        f.write("-" * 60 + "\n")
    f.write("=" * 80 + "\n")


'''
# Create DataFrame for visualization
y_pred_gb = results["Gradient Boosting"]["y_pred"]
gb_pred_vs_actual = wage_stability_df.copy()
gb_pred_vs_actual['Predicted_Wage_2024'] = y_pred_gb
'''

# Prepare data
X = wage_stability_df[['Median_Wage_2016', 'Median_Wage_2020']]
y = wage_stability_df['Median_Wage_2024']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Fit Gradient Boosting on train
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y_train)

# Predict on test
y_pred_test = gbr.predict(X_test)

# Create dataframe for visualization (only test set)
gb_pred_vs_actual = wage_stability_df.iloc[y_test.index].copy()
gb_pred_vs_actual['Predicted_Wage_2024'] = y_pred_test


# Create selection object (based on Broad_Category_Short)
selection = alt.selection_point(fields=['Broad_Category_Short'], bind='legend')

# Altair scatter plot
scatter_plot = alt.Chart(gb_pred_vs_actual).mark_circle(size=120).encode(
    x=alt.X('Predicted_Wage_2024:Q', title='Predicted Median Wage 2024 ($)'),
    y=alt.Y('Median_Wage_2024:Q', title='Actual Median Wage 2024 ($)'),
    color=alt.Color('Broad_Category_Short:N', title='Broad Occupational Category',
                    scale=alt.Scale(scheme='category10')),
    opacity=alt.condition(selection, alt.value(1), alt.value(0.15)),
    tooltip=[
        alt.Tooltip('NOC_2021:N', title='NOC Code'),
        alt.Tooltip('NOC_Title_2021:N', title='NOC Title'),
        alt.Tooltip('Broad_Category_Short:N', title='Broad Category'),
        alt.Tooltip('TEER_Code:Q', title='TEER Level'),
        alt.Tooltip('Median_Wage_2024:Q', title='Actual Wage ($)', format=',.2f'),
        alt.Tooltip('Predicted_Wage_2024:Q', title='Predicted Wage ($)', format=',.2f')
    ]
).add_params(
    selection
).properties(
    title={
        "text": ["Gradient Boosting: Predicted Based on 2016-2020 Data vs Actual 2024 Wages"],
      "subtitle": ["Test Data Only, 20% of dataset",
                   ""]
    },
    width=800,
    height=800
).interactive()

# Add ideal reference line (perfect prediction)
perfect_line = alt.Chart(pd.DataFrame({
    'x': [gb_pred_vs_actual['Median_Wage_2024'].min(), gb_pred_vs_actual['Median_Wage_2024'].max()],
    'y': [gb_pred_vs_actual['Median_Wage_2024'].min(), gb_pred_vs_actual['Median_Wage_2024'].max()]
})).mark_line(
    color='black',
    strokeDash=[5,5]
).encode(
    x='x:Q',
    y='y:Q'
)

# Combine scatter and perfect line
final_scatter = (scatter_plot + perfect_line).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16,
    anchor='start'
)

# Save
final_scatter.save(Path(FIGURE_PATH) / '3-5-Wage_Pattern_Stability_Predicted_vs_Actual_GB.html')

