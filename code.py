"""
Weather Trend Forecasting Analysis - Complete Python Code
PM Accelerator Mission: Empowering innovation through data-driven insights

Save this as: weather_analysis.py
Run with: python weather_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("WEATHER TREND FORECASTING ANALYSIS")
print("PM Accelerator Mission: Empowering innovation through data-driven insights")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading Data...")
df = pd.read_csv('archive/GlobalWeatherRepository.csv')
print(f"Dataset loaded: {df.shape[0]:,} records, {df.shape[1]} features")

# ============================================================================
# 2. DATA CLEANING
# ============================================================================
print("\n2. Data Cleaning...")
df_clean = df.copy()

# Convert datetime
df_clean['last_updated'] = pd.to_datetime(df_clean['last_updated'])
df_clean['date'] = df_clean['last_updated'].dt.date
df_clean['hour'] = df_clean['last_updated'].dt.hour
df_clean['day_of_week'] = df_clean['last_updated'].dt.dayofweek
df_clean['month'] = df_clean['last_updated'].dt.month

# Fill missing values
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)

categorical_cols = df_clean.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col].fillna('Unknown', inplace=True)

print(f"Data cleaned: {df_clean.shape}")

# ============================================================================
# 3. EDA
# ============================================================================
print("\n3. Exploratory Data Analysis...")

# Climate zones
df_clean['climate_zone'] = pd.cut(df_clean['temperature_celsius'], 
                                   bins=[-np.inf, 0, 10, 20, 30, np.inf],
                                   labels=['Polar', 'Cold', 'Temperate', 'Warm', 'Hot'])

print(f"Countries: {df_clean['country'].nunique()}")
print(f"Locations: {df_clean['location_name'].nunique()}")

# ============================================================================
# 4. VISUALIZATIONS
# ============================================================================
print("\n4. Creating Visualizations...")

# 4.1 Time Series
daily_data = df_clean.groupby('date').agg({
    'temperature_celsius': 'mean',
    'precip_mm': 'mean'
}).reset_index()

fig, axes = plt.subplots(2, 1, figsize=(15, 10))
axes[0].plot(daily_data['date'], daily_data['temperature_celsius'], color='red', linewidth=2)
axes[0].set_title('Average Temperature Over Time', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Temperature (°C)')
axes[0].grid(True, alpha=0.3)

axes[1].bar(range(len(daily_data)), daily_data['precip_mm'], color='blue', alpha=0.6)
axes[1].set_title('Average Precipitation Over Time', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Precipitation (mm)')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('1_timeseries.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 1_timeseries.png")

# 4.2 Correlation Heatmap
corr_cols = ['temperature_celsius', 'humidity', 'wind_kph', 'pressure_mb', 'precip_mm', 'uv_index']
corr_matrix = df_clean[corr_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('2_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 2_correlation.png")

# 4.3 Distributions
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
metrics = [
    ('temperature_celsius', 'Temperature (°C)', 'red'),
    ('humidity', 'Humidity (%)', 'blue'),
    ('wind_kph', 'Wind Speed (km/h)', 'green'),
    ('pressure_mb', 'Pressure (mb)', 'purple'),
    ('precip_mm', 'Precipitation (mm)', 'cyan'),
    ('uv_index', 'UV Index', 'orange')
]
for idx, (col, title, color) in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]
    ax.hist(df_clean[col].dropna(), bins=50, color=color, alpha=0.7, edgecolor='black')
    ax.set_title(f'{title} Distribution', fontweight='bold')
    ax.set_xlabel(title)
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('3_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 3_distributions.png")

# 4.4 Geographic Distribution
fig, ax = plt.subplots(figsize=(12, 8))
df_clean['country'].value_counts().head(15).plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Top 15 Countries', fontsize=14, fontweight='bold')
ax.set_xlabel('Records')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('4_geography.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 4_geography.png")

# 4.5 Climate Zones
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
climate_counts = df_clean['climate_zone'].value_counts()
axes[0].pie(climate_counts.values, labels=climate_counts.index, autopct='%1.1f%%', startangle=90)
axes[0].set_title('Climate Zone Distribution', fontsize=14, fontweight='bold')
df_clean.boxplot(column='temperature_celsius', by='climate_zone', ax=axes[1])
axes[1].set_title('Temperature by Climate Zone', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Temperature (°C)')
plt.suptitle('')
plt.tight_layout()
plt.savefig('5_climate.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 5_climate.png")

# ============================================================================
# 5. ANOMALY DETECTION
# ============================================================================
print("\n5. Anomaly Detection...")

df_clean['temp_zscore'] = zscore(df_clean['temperature_celsius'].dropna())
df_clean['humidity_zscore'] = zscore(df_clean['humidity'].dropna())
df_clean['wind_zscore'] = zscore(df_clean['wind_kph'].dropna())

temp_anomalies = df_clean[np.abs(df_clean['temp_zscore']) > 3]
print(f"Temperature anomalies: {len(temp_anomalies):,}")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].scatter(range(len(df_clean)), df_clean['temp_zscore'], alpha=0.5, s=1)
axes[0].axhline(y=3, color='r', linestyle='--')
axes[0].axhline(y=-3, color='r', linestyle='--')
axes[0].set_title('Temperature Z-Score', fontweight='bold')
axes[0].set_ylabel('Z-Score')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(range(len(df_clean)), df_clean['humidity_zscore'], alpha=0.5, s=1, color='blue')
axes[1].axhline(y=3, color='r', linestyle='--')
axes[1].axhline(y=-3, color='r', linestyle='--')
axes[1].set_title('Humidity Z-Score', fontweight='bold')
axes[1].grid(True, alpha=0.3)

axes[2].scatter(range(len(df_clean)), df_clean['wind_zscore'], alpha=0.5, s=1, color='green')
axes[2].axhline(y=3, color='r', linestyle='--')
axes[2].axhline(y=-3, color='r', linestyle='--')
axes[2].set_title('Wind Speed Z-Score', fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('6_anomalies.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 6_anomalies.png")

# ============================================================================
# 6. MODEL BUILDING
# ============================================================================
print("\n6. Building Forecasting Models...")

df_model = df_clean.sort_values('last_updated')
feature_cols = ['humidity', 'wind_kph', 'pressure_mb', 'cloud', 'visibility_km', 'hour', 'day_of_week', 'month']
df_model = df_model.dropna(subset=feature_cols + ['temperature_celsius'])

X = df_model[feature_cols]
y = df_model['temperature_celsius']

split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training: {len(X_train):,}, Test: {len(X_test):,}")

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)
print(f"Linear Regression - RMSE: {lr_rmse:.4f}, R²: {lr_r2:.4f}")

# Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)
print(f"Random Forest - RMSE: {rf_rmse:.4f}, R²: {rf_r2:.4f}")

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_r2 = r2_score(y_test, gb_pred)
print(f"Gradient Boosting - RMSE: {gb_rmse:.4f}, R²: {gb_r2:.4f}")

# Ensemble
ensemble_pred = (lr_pred * 0.2 + rf_pred * 0.4 + gb_pred * 0.4)
ens_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
ens_r2 = r2_score(y_test, ensemble_pred)
print(f"Ensemble - RMSE: {ens_rmse:.4f}, R²: {ens_r2:.4f}")

# Model Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
models = ['Linear Reg', 'Random Forest', 'Gradient Boost', 'Ensemble']
rmse_values = [lr_rmse, rf_rmse, gb_rmse, ens_rmse]
r2_values = [lr_r2, rf_r2, gb_r2, ens_r2]

axes[0, 0].bar(models, rmse_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
axes[0, 0].set_title('RMSE Comparison', fontweight='bold')
axes[0, 0].set_ylabel('RMSE')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].bar(models, r2_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
axes[0, 1].set_title('R² Comparison', fontweight='bold')
axes[0, 1].set_ylabel('R² Score')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].scatter(y_test, ensemble_pred, alpha=0.5, s=10)
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 0].set_title('Actual vs Predicted', fontweight='bold')
axes[1, 0].set_xlabel('Actual')
axes[1, 0].set_ylabel('Predicted')
axes[1, 0].grid(True, alpha=0.3)

residuals = y_test.values - ensemble_pred
axes[1, 1].hist(residuals, bins=50, color='green', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Residual Distribution', fontweight='bold')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('7_models.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 7_models.png")

# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================
print("\n7. Feature Importance...")

importances = rf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
ax.set_xlabel('Importance')
ax.set_title('Feature Importance', fontweight='bold', fontsize=14)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('8_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 8_importance.png")

# ============================================================================
# 8. SPATIAL ANALYSIS
# ============================================================================
print("\n8. Spatial Analysis...")

spatial = df_clean[['latitude', 'longitude', 'temperature_celsius']].dropna()
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(spatial[['latitude', 'longitude']])
spatial['cluster'] = clusters

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
scatter = axes[0].scatter(spatial['longitude'], spatial['latitude'], 
                         c=spatial['cluster'], cmap='viridis', alpha=0.6, s=20)
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[0].set_title('Geographic Clustering', fontweight='bold')
plt.colorbar(scatter, ax=axes[0], label='Cluster')
axes[0].grid(True, alpha=0.3)

spatial.boxplot(column='temperature_celsius', by='cluster', ax=axes[1])
axes[1].set_title('Temperature by Cluster', fontweight='bold')
axes[1].set_ylabel('Temperature (°C)')
plt.suptitle('')
plt.tight_layout()
plt.savefig('9_spatial.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 9_spatial.png")

# ============================================================================
# 9. AIR QUALITY ANALYSIS
# ============================================================================
print("\n9. Air Quality Analysis...")

if 'air_quality_PM2.5' in df_clean.columns:
    aq_data = df_clean[['air_quality_PM2.5', 'temperature_celsius', 'humidity', 'wind_kph']].dropna()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].hist(aq_data['air_quality_PM2.5'], bins=50, color='brown', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('PM2.5 Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('PM2.5 (μg/m³)')
    axes[0, 0].axvline(x=35, color='red', linestyle='--', label='Unhealthy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(aq_data['temperature_celsius'], aq_data['air_quality_PM2.5'], alpha=0.5, s=10)
    axes[0, 1].set_title('PM2.5 vs Temperature', fontweight='bold')
    axes[0, 1].set_xlabel('Temperature (°C)')
    axes[0, 1].set_ylabel('PM2.5')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].scatter(aq_data['wind_kph'], aq_data['air_quality_PM2.5'], alpha=0.5, s=10, color='green')
    axes[1, 0].set_title('PM2.5 vs Wind Speed', fontweight='bold')
    axes[1, 0].set_xlabel('Wind Speed (km/h)')
    axes[1, 0].set_ylabel('PM2.5')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(aq_data['humidity'], aq_data['air_quality_PM2.5'], alpha=0.5, s=10, color='blue')
    axes[1, 1].set_title('PM2.5 vs Humidity', fontweight='bold')
    axes[1, 1].set_xlabel('Humidity (%)')
    axes[1, 1].set_ylabel('PM2.5')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('10_air_quality.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 10_air_quality.png")

# ============================================================================
# 10. GENERATE REPORT
# ============================================================================
print("\n10. Generating Report...")

report = []
report.append("="*80)
report.append("WEATHER TREND FORECASTING - ANALYSIS REPORT")
report.append("PM Accelerator Mission: Empowering innovation through data-driven insights")
report.append("="*80)
report.append("")
report.append("EXECUTIVE SUMMARY")
report.append("-"*80)
report.append(f"Total Records: {len(df_clean):,}")
report.append(f"Countries: {df_clean['country'].nunique()}")
report.append(f"Locations: {df_clean['location_name'].nunique()}")
report.append("")
report.append("KEY FINDINGS")
report.append("-"*80)
report.append(f"Avg Temperature: {df_clean['temperature_celsius'].mean():.2f}°C")
report.append(f"Avg Humidity: {df_clean['humidity'].mean():.2f}%")
report.append(f"Avg Wind Speed: {df_clean['wind_kph'].mean():.2f} km/h")
report.append("")
report.append("MODEL PERFORMANCE")
report.append("-"*80)
report.append(f"{'Model':<20} {'RMSE':<12} {'R²':<12}")
report.append("-"*80)
report.append(f"{'Linear Regression':<20} {lr_rmse:<12.4f} {lr_r2:<12.4f}")
report.append(f"{'Random Forest':<20} {rf_rmse:<12.4f} {rf_r2:<12.4f}")
report.append(f"{'Gradient Boosting':<20} {gb_rmse:<12.4f} {gb_r2:<12.4f}")
report.append(f"{'Ensemble':<20} {ens_rmse:<12.4f} {ens_r2:<12.4f}")
report.append("")
report.append("FEATURE IMPORTANCE")
report.append("-"*80)
for _, row in importance_df.head().iterrows():
    report.append(f"{row['Feature']:<20} {row['Importance']:.4f}")
report.append("")
report.append("="*80)

report_text = "\n".join(report)
with open('ANALYSIS_REPORT.txt', 'w') as f:
    f.write(report_text)

print(report_text)
print("\nReport saved: ANALYSIS_REPORT.txt")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated Files:")
print("  1. 1_timeseries.png")
print("  2. 2_correlation.png")
print("  3. 3_distributions.png")
print("  4. 4_geography.png")
print("  5. 5_climate.png")
print("  6. 6_anomalies.png")
print("  7. 7_models.png")
print("  8. 8_importance.png")
print("  9. 9_spatial.png")
print(" 10. 10_air_quality.png")
print(" 11. ANALYSIS_REPORT.txt")
print("="*80)