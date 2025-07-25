# fix_zero_predictions.py
import pandas as pd
import numpy as np
import json

def fix_zero_predictions():
    """Post-process ensemble to better handle zero sales periods"""
    
    # Load predictions and operating schedule
    ensemble_df = pd.read_csv('results/ensemble_predictions.csv')
    operating_schedule = pd.read_csv('results/operating_schedule.csv')
    
    # Convert datetime
    ensemble_df['ds'] = pd.to_datetime(ensemble_df['ds'])
    
    # Add time features
    ensemble_df['hour'] = ensemble_df['ds'].dt.hour
    ensemble_df['dayofweek'] = ensemble_df['ds'].dt.dayofweek
    
    # Merge with operating schedule
    ensemble_df = ensemble_df.merge(
        operating_schedule,
        on=['location_id', 'dayofweek', 'hour'],
        how='left'
    )
    
    # Step 1: Set predictions to 0 for closed hours
    closed_mask = ensemble_df['is_typically_open'] == 0
    ensemble_df.loc[closed_mask, 'yhat_ensemble'] = 0
    print(f"Set {closed_mask.sum()} predictions to 0 for closed hours")
    
    # Step 2: Identify likely zero-sales periods based on historical patterns
    # Load historical data to find zero-sales patterns
    sales_df = pd.read_csv('data/sales.csv')
    sales_df['ds'] = pd.to_datetime(sales_df['ds'])
    sales_df['hour'] = sales_df['ds'].dt.hour
    sales_df['dayofweek'] = sales_df['ds'].dt.dayofweek
    
    # Calculate zero-sales frequency by location/hour/dow
    zero_freq = sales_df.groupby(['location_id', 'sales_type_id', 'department_id', 
                                  'hour', 'dayofweek']).agg(
        total_periods=('y', 'count'),
        zero_periods=('y', lambda x: (x == 0).sum()),
        avg_when_nonzero=('y', lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0)
    ).reset_index()
    
    zero_freq['zero_probability'] = zero_freq['zero_periods'] / zero_freq['total_periods']
    
    # Merge zero probability with ensemble
    ensemble_df = ensemble_df.merge(
        zero_freq[['location_id', 'sales_type_id', 'department_id', 
                   'hour', 'dayofweek', 'zero_probability', 'avg_when_nonzero']],
        on=['location_id', 'sales_type_id', 'department_id', 'hour', 'dayofweek'],
        how='left'
    )
    
    # Step 3: Apply threshold-based zero prediction
    # If zero_probability > 0.8 and current prediction < 20% of typical non-zero value
    zero_threshold = 0.8
    value_threshold = 0.2
    
    likely_zero_mask = (
        (ensemble_df['zero_probability'] > zero_threshold) & 
        (ensemble_df['yhat_ensemble'] < ensemble_df['avg_when_nonzero'] * value_threshold) &
        (ensemble_df['is_typically_open'] == 1)  # Only during open hours
    )
    
    print(f"\nBefore fix: {(ensemble_df['yhat_ensemble'] == 0).sum()} zero predictions")
    ensemble_df.loc[likely_zero_mask, 'yhat_ensemble'] = 0
    print(f"After fix: {(ensemble_df['yhat_ensemble'] == 0).sum()} zero predictions")
    print(f"Added {likely_zero_mask.sum()} new zero predictions")
    
    # Step 4: Also clip very small predictions to zero
    small_threshold = 0.5  # Less than $0.50
    small_mask = (ensemble_df['yhat_ensemble'] < small_threshold) & (ensemble_df['yhat_ensemble'] > 0)
    ensemble_df.loc[small_mask, 'yhat_ensemble'] = 0
    print(f"Clipped {small_mask.sum()} small predictions to zero")
    
    # Save fixed predictions
    ensemble_df_save = ensemble_df[['ds', 'location_id', 'sales_type_id', 'department_id', 
                                    'y', 'yhat', 'yhat_lgb', 'yhat_ensemble']]
    ensemble_df_save.to_csv('results/ensemble_predictions_fixed.csv', index=False)
    
    # Recalculate metrics
    calculate_fixed_metrics(ensemble_df_save)

def calculate_fixed_metrics(ensemble_df):
    """Recalculate metrics after fixing zero predictions"""
    
    # Load baseline MAE
    with open('results/baseline_mae.json', 'r') as f:
        baseline_mae = json.load(f)
    
    # Calculate new MAE
    ensemble_df['ae_ensemble'] = abs(ensemble_df['y'] - ensemble_df['yhat_ensemble'])
    mae_15min = ensemble_df.groupby(['location_id', 'sales_type_id', 'department_id'])['ae_ensemble'].mean()
    
    # Hourly
    ensemble_df['hour'] = ensemble_df['ds'].dt.floor('h')
    hourly_df = ensemble_df.groupby(['location_id', 'sales_type_id', 'department_id', 'hour']).agg({
        'y': 'sum',
        'yhat_ensemble': 'sum'
    }).reset_index()
    hourly_df['ae_hourly'] = abs(hourly_df['y'] - hourly_df['yhat_ensemble'])
    mae_hourly = hourly_df.groupby(['location_id', 'sales_type_id', 'department_id'])['ae_hourly'].mean()
    
    # Daily
    ensemble_df['date'] = ensemble_df['ds'].dt.date
    daily_df = ensemble_df.groupby(['location_id', 'sales_type_id', 'department_id', 'date']).agg({
        'y': 'sum',
        'yhat_ensemble': 'sum'
    }).reset_index()
    daily_df['ae_daily'] = abs(daily_df['y'] - daily_df['yhat_ensemble'])
    mae_daily = daily_df.groupby(['location_id', 'sales_type_id', 'department_id'])['ae_daily'].mean()
    
    # Calculate fixed MAE
    fixed_mae = {
        '15min': mae_15min.median(),
        'hourly': mae_hourly.median(),
        'daily': mae_daily.median(),
        'combined': (mae_15min.median() + mae_hourly.median() + mae_daily.median()) / 3
    }
    
    # Calculate improvements
    improvements = {
        '15min': (baseline_mae['15min'] - fixed_mae['15min']) / baseline_mae['15min'] * 100,
        'hourly': (baseline_mae['hourly'] - fixed_mae['hourly']) / baseline_mae['hourly'] * 100,
        'daily': (baseline_mae['daily'] - fixed_mae['daily']) / baseline_mae['daily'] * 100,
        'combined': (baseline_mae['combined'] - fixed_mae['combined']) / baseline_mae['combined'] * 100
    }
    
    print(f"\n{'='*60}")
    print(f"RESULTS AFTER ZERO-PREDICTION FIX")
    print(f"{'='*60}")
    print(f"\nFixed Ensemble MAE:")
    print(f"  15-min: {fixed_mae['15min']:.4f} (improvement: {improvements['15min']:.1f}%)")
    print(f"  Hourly: {fixed_mae['hourly']:.4f} (improvement: {improvements['hourly']:.1f}%)")
    print(f"  Daily: {fixed_mae['daily']:.4f} (improvement: {improvements['daily']:.1f}%)")
    print(f"  Combined: {fixed_mae['combined']:.4f} (improvement: {improvements['combined']:.1f}%)")
    
    # Save metrics
    with open('results/fixed_ensemble_metrics.json', 'w') as f:
        json.dump({
            'baseline_mae': baseline_mae,
            'fixed_mae': fixed_mae,
            'improvements': improvements
        }, f, indent=2)

if __name__ == '__main__':
    fix_zero_predictions()