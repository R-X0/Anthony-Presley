# analyze_15min_patterns.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_15min_patterns():
    """Deep dive into 15-minute prediction patterns"""
    
    # Load predictions
    prophet_df = pd.read_csv('results/baseline_prophet_results.csv')
    lgb_df = pd.read_csv('results/lightgbm_predictions.csv')
    
    # Convert datetime
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    lgb_df['ds'] = pd.to_datetime(lgb_df['ds'])
    
    # Merge
    merged_df = pd.merge(
        prophet_df[['ds', 'location_id', 'sales_type_id', 'department_id', 'y', 'yhat']],
        lgb_df[['ds', 'location_id', 'sales_type_id', 'department_id', 'yhat_lgb']],
        on=['ds', 'location_id', 'sales_type_id', 'department_id']
    )
    
    # Add features
    merged_df['hour'] = merged_df['ds'].dt.hour
    merged_df['minute'] = merged_df['ds'].dt.minute
    merged_df['dayofweek'] = merged_df['ds'].dt.dayofweek
    
    # Calculate errors
    merged_df['prophet_error'] = abs(merged_df['y'] - merged_df['yhat'])
    merged_df['lgb_error'] = abs(merged_df['y'] - merged_df['yhat_lgb'])
    
    # Which model is better for different scenarios?
    print("Model Performance Analysis")
    print("="*60)
    
    # By sales magnitude
    sales_bins = [0, 10, 50, 100, 500, 10000]
    merged_df['sales_bin'] = pd.cut(merged_df['y'], bins=sales_bins)
    
    bin_analysis = merged_df.groupby('sales_bin').agg({
        'prophet_error': 'mean',
        'lgb_error': 'mean',
        'y': 'count'
    }).round(2)
    
    bin_analysis['better_model'] = np.where(
        bin_analysis['prophet_error'] < bin_analysis['lgb_error'], 
        'Prophet', 'LightGBM'
    )
    
    print("\nPerformance by Sales Magnitude:")
    print(bin_analysis)
    
    # By time patterns
    print("\n\nPerformance by 15-minute interval within hour:")
    minute_analysis = merged_df.groupby('minute').agg({
        'prophet_error': 'mean',
        'lgb_error': 'mean'
    }).round(2)
    
    minute_analysis['prophet_better'] = (minute_analysis['prophet_error'] < minute_analysis['lgb_error'])
    print(minute_analysis)
    
    # Look at variance
    print("\n\nPrediction Variance Analysis:")
    print(f"Actual sales std: {merged_df['y'].std():.2f}")
    print(f"Prophet predictions std: {merged_df['yhat'].std():.2f}")
    print(f"LightGBM predictions std: {merged_df['yhat_lgb'].std():.2f}")
    
    # Prophet tends to smooth too much, LightGBM might overfit
    # Let's see if that's the issue
    merged_df['prophet_underpredict'] = merged_df['yhat'] < merged_df['y']
    merged_df['lgb_overpredict'] = merged_df['yhat_lgb'] > merged_df['y']
    
    print(f"\nProphet underpredicts: {merged_df['prophet_underpredict'].mean()*100:.1f}% of the time")
    print(f"LightGBM overpredicts: {merged_df['lgb_overpredict'].mean()*100:.1f}% of the time")
    
    return merged_df

if __name__ == '__main__':
    analyze_15min_patterns()