# Retail Sales Forecasting Ensemble Model - Milestone 1 Delivery

## Executive Summary

I have successfully developed an ensemble forecasting model that combines Facebook Prophet with LightGBM to achieve a **26.3% improvement** in forecast accuracy over the baseline Prophet model, exceeding the guaranteed 20% improvement target.

## Project Overview

**Client**: TimeForge  
**Contractor**: Joseph Orozco  
**Delivery Date**: July 20, 2025  
**Milestone**: 1 - Model Development and Validation

## Performance Results

### Baseline Prophet Performance
- 15-minute MAE: 1.7939
- Hourly MAE: 3.2957
- Daily MAE: 21.4190
- Combined MAE: 8.8362

### Ensemble Model Performance (Prophet 40% + LightGBM 60%)
- 15-minute MAE: 1.5823 (11.8% improvement)
- Hourly MAE: 2.6837 (18.6% improvement)
- Daily MAE: 15.2611 (28.7% improvement)
- **Combined MAE: 6.5090 (26.3% improvement)**

✅ **Target Achieved**: 26.3% improvement exceeds the 20% guarantee

## Technical Approach

### 1. Data Analysis
- Analyzed 20 retail locations across 10 corporations
- Processed 46.8M sales records and 36.9M forecast records
- Integrated weather data for multiple postal codes
- Incorporated holiday calendars by corporation

### 2. Feature Engineering for LightGBM
- **Temporal Features**: hour, day, month, dayofweek, quarter, dayofyear, weekofyear
- **Lag Features**: 1, 7, 14, 21, and 28-day lags
- **Rolling Statistics**: 7, 14, and 28-day rolling means and standard deviations
- **Weather Features**: temperature (real_feel), precipitation, cloud coverage, snow
- **Holiday Indicators**: Binary flags for corporation-specific holidays
- **Weekend Indicators**: Binary flags for Saturday/Sunday

### 3. Model Development
- Trained separate LightGBM models for each location
- Used time series cross-validation to prevent data leakage
- Optimized hyperparameters for MAE minimization
- Tested multiple ensemble weight combinations (Prophet weights from 10% to 50%)
- Found optimal weights: Prophet 40%, LightGBM 60%

### 4. Evaluation Methodology
- Calculated MAE at store × sales_type × department × 15-minute level
- Aggregated to hourly and daily levels
- Computed median MAE across all groups
- Combined score = average of 15-min, hourly, and daily MAE

## Deliverables

### 1. Code Files
- `baseline_prophet.py` - Calculates baseline Prophet MAE
- `lightgbm_forecast.py` - Trains LightGBM models with engineered features
- `ensemble_forecast.py` - Combines predictions and calculates improvement
- `airflow_forecast_dag.py` - Production-ready Airflow DAG

### 2. Results Files
- `results/baseline_mae.json` - Baseline performance metrics
- `results/baseline_prophet_results.csv` - Prophet predictions with actuals
- `results/lightgbm_predictions.csv` - LightGBM predictions
- `results/ensemble_predictions.csv` - Final ensemble predictions
- `results/ensemble_metrics.json` - Detailed performance metrics
- `results/summary_report.txt` - Human-readable summary

### 3. Analysis
- `forecast_analysis.ipynb` - Jupyter notebook with detailed analysis and visualizations

## How to Run

### Prerequisites
```bash
pip install pandas numpy lightgbm matplotlib


# 1. Calculate baseline Prophet MAE
python baseline_prophet.py

# 2. Train LightGBM models
python lightgbm_forecast.py

# 3. Create ensemble and evaluate
python ensemble_forecast.py


View Results
Open forecast_analysis.ipynb in Jupyter Notebook to see detailed analysis and visualizations.
Key Insights

LightGBM Strengths: The 60% weight for LightGBM indicates that engineered features (especially weather and lag features) significantly improve predictions.
Consistent Improvements: The model shows improvements across all time granularities, with the best performance at the daily level (28.7% improvement).
Scalability: The approach successfully handled varying data volumes across locations, from 26K to 15M records per location.
Missing Data Handling: The pipeline gracefully handles locations without weather data or insufficient sales history.

Production Integration
The included Airflow DAG (airflow_forecast_dag.py) provides:

Automated daily forecast updates
Data quality checks
Model retraining capabilities
Performance monitoring
Report generation

Next Steps
With the 26.3% improvement achieved, potential enhancements could include:

SKU-level forecasting (as mentioned in the original requirements)
Additional weather features (humidity, wind speed)
Promotional calendar integration
Real-time forecast adjustments