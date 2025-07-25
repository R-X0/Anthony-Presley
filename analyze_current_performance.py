# analyze_current_performance.py
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_performance():
    # Load current results
    baseline_df = pd.read_csv('results/baseline_prophet_results.csv')
    ensemble_df = pd.read_csv('results/ensemble_predictions.csv')
    
    # Convert datetime
    baseline_df['ds'] = pd.to_datetime(baseline_df['ds'])
    ensemble_df['ds'] = pd.to_datetime(ensemble_df['ds'])
    
    # Merge to compare
    comparison_df = pd.merge(
        baseline_df[['ds', 'location_id', 'sales_type_id', 'department_id', 'y', 'yhat']],
        ensemble_df[['ds', 'location_id', 'sales_type_id', 'department_id', 'yhat_ensemble']],
        on=['ds', 'location_id', 'sales_type_id', 'department_id']
    )
    
    # Calculate errors
    comparison_df['baseline_error'] = abs(comparison_df['y'] - comparison_df['yhat'])
    comparison_df['ensemble_error'] = abs(comparison_df['y'] - comparison_df['yhat_ensemble'])
    
    # Add time features for analysis
    comparison_df['hour'] = comparison_df['ds'].dt.hour
    comparison_df['dayofweek'] = comparison_df['ds'].dt.dayofweek
    comparison_df['is_weekend'] = comparison_df['dayofweek'].isin([5, 6])
    
    # Analyze by different dimensions
    print("Performance Analysis Report")
    print("="*60)
    
    # By hour of day
    hourly_performance = comparison_df.groupby('hour').agg({
        'baseline_error': 'mean',
        'ensemble_error': 'mean'
    }).round(4)
    hourly_performance['improvement'] = ((hourly_performance['baseline_error'] - hourly_performance['ensemble_error']) / 
                                        hourly_performance['baseline_error'] * 100).round(1)
    
    print("\nPerformance by Hour of Day:")
    print(hourly_performance)
    
    # By day of week
    dow_performance = comparison_df.groupby('dayofweek').agg({
        'baseline_error': 'mean',
        'ensemble_error': 'mean'
    }).round(4)
    dow_performance['improvement'] = ((dow_performance['baseline_error'] - dow_performance['ensemble_error']) / 
                                      dow_performance['baseline_error'] * 100).round(1)
    
    print("\nPerformance by Day of Week:")
    print(dow_performance)
    
    # By location
    location_performance = comparison_df.groupby('location_id').agg({
        'baseline_error': 'mean',
        'ensemble_error': 'mean'
    }).round(4)
    location_performance['improvement'] = ((location_performance['baseline_error'] - location_performance['ensemble_error']) / 
                                          location_performance['baseline_error'] * 100).round(1)
    
    print("\nPerformance by Location:")
    print(location_performance.sort_values('improvement'))
    
    # Identify worst performing periods
    comparison_df['improvement'] = (comparison_df['baseline_error'] - comparison_df['ensemble_error']) / comparison_df['baseline_error'] * 100
    worst_periods = comparison_df.nsmallest(100, 'improvement')[['ds', 'location_id', 'hour', 'dayofweek', 'improvement']]
    
    print("\nWorst Performing Periods (where ensemble is worse than baseline):")
    print(worst_periods.head(20))
    
    # Save detailed analysis
    comparison_df.to_csv('results/performance_analysis.csv', index=False)
    
    return comparison_df

if __name__ == '__main__':
    analyze_performance()