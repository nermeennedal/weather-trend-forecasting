# Weather Trend Forecasting Analysis

**Developer**: Nermeen Nedal Dawoud
**Organization**: PM Accelerator  
**Mission**: Empowering the next generation of product leaders

---

## About PM Accelerator

The **Product Manager Accelerator Program** is designed to support PM professionals through every stage of their career. From students looking for entry-level jobs to Directors looking to take on a leadership role, our program has helped over hundreds of students fulfill their career aspirations.

Our Product Manager Accelerator community is ambitious and committed. Through our program, they have learned, honed, and developed new PM and leadership skills, giving them a strong foundation for their future endeavors.

**Learn more**: [PM Accelerator LinkedIn](https://www.linkedin.com/school/pmaccelerator/)

---

## Project Overview

This project performs comprehensive weather trend forecasting analysis using machine learning models. It includes:

- **Data Cleaning & Preprocessing**: Handling missing values, datetime conversion, feature engineering
- **Exploratory Data Analysis (EDA)**: Statistical summaries, distributions, correlations
- **Visualization**: 10+ professional charts including time series, geographic distributions, climate zones
- **Anomaly Detection**: Z-score based outlier identification
- **Machine Learning Models**: Linear Regression, Random Forest, Gradient Boosting, and Ensemble
- **Advanced Analytics**: Spatial clustering, air quality analysis, feature importance
- **Automated Reporting**: Comprehensive text report with key findings

---

## Dataset

**Source**: GlobalWeatherRepository.csv  
**Location**: `archive/GlobalWeatherRepository.csv`

The dataset contains weather observations including:
- Temperature, humidity, wind speed, pressure
- Precipitation, UV index, cloud cover, visibility
- Geographic coordinates (latitude, longitude)
- Air quality metrics (PM2.5)
- Timestamp information

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone [your-repo-url]
cd weather-forecasting-analysis
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Data
Place the `GlobalWeatherRepository.csv` file in the `archive/` folder:
```
weather-forecasting-analysis/
├── archive/
│   └── GlobalWeatherRepository.csv
├── code.py
├── requirements.txt
└── README.md
```

### Step 4: Run the Analysis
```bash
python weather_analysis.py
```

---

## Output Files

The script generates the following outputs:

### Visualizations (PNG files)
1. **1_timeseries.png** - Temperature and precipitation trends over time
2. **2_correlation.png** - Correlation heatmap of weather variables
3. **3_distributions.png** - Distribution plots for 6 key metrics
4. **4_geography.png** - Top 15 countries by record count
5. **5_climate.png** - Climate zone distribution and temperature ranges
6. **6_anomalies.png** - Z-score anomaly detection for temperature, humidity, wind
7. **7_models.png** - Model performance comparison and predictions
8. **8_importance.png** - Feature importance from Random Forest
9. **9_spatial.png** - Geographic clustering and temperature patterns
10. **10_air_quality.png** - PM2.5 air quality analysis

### Report
- **ANALYSIS_REPORT.txt** - Comprehensive text report with executive summary, key findings, model performance, and feature importance

---

## Methodology

### 1. Data Cleaning
- Convert datetime fields and extract temporal features (hour, day of week, month)
- Impute missing numeric values with median
- Impute missing categorical values with 'Unknown'
- Create climate zone categories based on temperature ranges

### 2. Exploratory Data Analysis
- Statistical summaries of all numeric variables
- Geographic distribution analysis (countries, locations)
- Temporal pattern identification
- Climate zone classification

### 3. Feature Engineering
- Temporal features: hour, day_of_week, month
- Climate zones: Polar, Cold, Temperate, Warm, Hot
- Z-scores for anomaly detection

### 4. Modeling Approach
We implemented and compared four approaches:

**Linear Regression** (Baseline)
- Simple linear model with scaled features
- Fast training, interpretable coefficients

**Random Forest Regressor**
- 100 trees, max depth 15
- Handles non-linear relationships
- Provides feature importance

**Gradient Boosting Regressor**
- 100 estimators, learning rate 0.1
- Sequential error correction
- High predictive accuracy

**Ensemble Model**
- Weighted combination: 20% Linear + 40% RF + 40% GB
- Reduces overfitting
- Best overall performance

### 5. Model Evaluation Metrics
- **RMSE** (Root Mean Squared Error): Measures prediction error magnitude
- **R² Score**: Proportion of variance explained by the model

### 6. Advanced Analytics
- **Anomaly Detection**: Identifies outliers using z-score > 3
- **Spatial Clustering**: K-means clustering on geographic coordinates
- **Air Quality Analysis**: PM2.5 relationships with weather variables
- **Feature Importance**: Identifies most predictive variables

---

## Key Findings

The analysis reveals:

1. **Best Performing Model**: The Ensemble model typically achieves the lowest RMSE and highest R² score
2. **Key Predictors**: Humidity, pressure, and temporal features are most important for temperature forecasting
3. **Anomalies**: Approximately 0.3% of observations are statistical outliers (|z-score| > 3)
4. **Geographic Patterns**: Clear clustering of weather patterns by geographic region
5. **Air Quality**: PM2.5 shows correlations with temperature, wind speed, and humidity

---

## Project Structure

```
weather-forecasting-analysis/
│
├── archive/
│   └── GlobalWeatherRepository.csv    # Input dataset
│
├── weather_analysis.py                # Main analysis script
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
│
├── Output files (generated after running):
│   ├── 1_timeseries.png
│   ├── 2_correlation.png
│   ├── 3_distributions.png
│   ├── 4_geography.png
│   ├── 5_climate.png
│   ├── 6_anomalies.png
│   ├── 7_models.png
│   ├── 8_importance.png
│   ├── 9_spatial.png
│   ├── 10_air_quality.png
│   └── ANALYSIS_REPORT.txt
```

---

## Technical Stack

- **Python**: 3.8+
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Statistical Analysis**: scipy

---

## Future Enhancements

Potential improvements for this project:
- Time series forecasting with ARIMA/LSTM models
- Interactive dashboard using Plotly/Dash
- Real-time weather API integration
- Climate change trend analysis
- Extreme weather event prediction
- Multi-step ahead forecasting

---

## Demo Video

[Link to demo video will be added here]

---

## Troubleshooting

**Issue**: `FileNotFoundError: archive/GlobalWeatherRepository.csv`  
**Solution**: Ensure the CSV file is in the `archive/` folder

**Issue**: `ModuleNotFoundError`  
**Solution**: Run `pip install -r requirements.txt`

**Issue**: Memory errors with large datasets  
**Solution**: Reduce dataset size or increase system RAM

---

## Contact

**Developer**: Nermeen Nedal Dawoud
**Email**: nermeennedal11@gmail.com
**LinkedIn**:(https://www.linkedin.com/in/nermeennedallinkdlin/)
**GitHub**: https://github.com/nermeennedal
---

## License

This project is created for the PM Accelerator Technical Assessment.

---

## Acknowledgments

- **PM Accelerator**: For providing this learning opportunity
- **Dataset Source**: GlobalWeatherRepository
- **Libraries**: scikit-learn, pandas, matplotlib, seaborn

---

*This project demonstrates data analysis, machine learning, and visualization skills for weather trend forecasting. Created as part of the PM Accelerator Program Technical Assessment.*
