# adaptive_ensemble.py
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression

def create_adaptive_ensemble():
    """Create ensemble with adaptive weights based on context"""
    
    # Load baseline MAE
    with open('results/baseline_mae.json', 'r') as f:
        baseline_mae = json.load(f)
    
    print("Creating adaptive ensemble...")
    
    # Load predictions
    prophet_df = pd.read_csv('results/baseline_prophet_results.csv')
    lgb_df = pd.read_csv('results/lightgbm_predictions.csv')
    
    # Convert datetime
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    lgb_df['ds'] = pd.to_datetime(lgb_df['ds'])
    
    # Merge
    ensemble_df = pd.merge(
        prophet_df[['ds', 'location_id', 'sales_type_id', 'department_id', 'y', 'yhat']],
        lgb_df[['ds', 'location_id', 'sales_type_id', 'department_id', 'yhat_lgb']],
        on=['ds', 'location_id', 'sales_type_id', 'department_id'],
        how='inner'
    )
    
    # Add features for adaptive weighting
    ensemble_df['hour'] = ensemble_df['ds'].dt.hour
    ensemble_df['dayofweek'] = ensemble_df['ds'].dt.dayofweek
    ensemble_df['is_weekend'] = ensemble_df['dayofweek'].isin([5, 6]).astype(int)
    
    # Strategy 1: Learn optimal weights per location
    location_weights = {}
    
    for location_id in ensemble_df['location_id'].unique():
        loc_data = ensemble_df[ensemble_df['location_id'] == location_id].copy()
        
        # Use first 80% for training weights
        train_size = int(len(loc_data) * 0.8)
        train_data = loc_data.iloc[:train_size]
        
        # Grid search for best weight
        best_weight = 0.5
        best_mae = float('inf')
        
        for prophet_weight in np.arange(0, 1.05, 0.05):
            lgb_weight = 1 - prophet_weight
            pred = prophet_weight * train_data['yhat'] + lgb_weight * train_data['yhat_lgb']
            mae = abs(train_data['y'] - pred).mean()
            
            if mae < best_mae:
                best_mae = mae
                best_weight = prophet_weight
        
        location_weights[location_id] = best_weight
        print(f"Location {location_id}: Prophet weight = {best_weight:.2f}")
    
    # Apply location-specific weights
    ensemble_df['location_prophet_weight'] = ensemble_df['location_id'].map(location_weights)
    ensemble_df['yhat_adaptive'] = (
        ensemble_df['location_prophet_weight'] * ensemble_df['yhat'] + 
        (1 - ensemble_df['location_prophet_weight']) * ensemble_df['yhat_lgb']
    )
    
    # Strategy 2: Additional adjustments for specific patterns
    # Boost LightGBM weight during peak hours (it handles spikes better)
    peak_hours = [11, 12, 13, 17, 18, 19, 20]
    peak_mask = ensemble_df['hour'].isin(peak_hours)
    
    # Reduce Prophet weight by 20% during peak hours
    ensemble_df.loc[peak_mask, 'yhat_adaptive'] = (
        0.8 * ensemble_df.loc[peak_mask, 'location_prophet_weight'] * ensemble_df.loc[peak_mask, 'yhat'] + 
        (1 - 0.8 * ensemble_df.loc[peak_mask, 'location_prophet_weight']) * ensemble_df.loc[peak_mask, 'yhat_lgb']
    )
    
    # Strategy 3: Handle zero predictions better
    # If both models predict very low values, set to zero
    low_threshold = 1.0
    both_low_mask = (ensemble_df['yhat'] < low_threshold) & (ensemble_df['yhat_lgb'] < low_threshold)
    ensemble_df.loc[both_low_mask, 'yhat_adaptive'] = 0
    
    # Calculate new metrics
    calculate_adaptive_metrics(ensemble_df, baseline_mae)
    
    # Save adaptive ensemble
    ensemble_df.to_csv('results/adaptive_ensemble_predictions.csv', index=False)
    
    return ensemble_df

def calculate_adaptive_metrics(ensemble_df, baseline_mae):
    """Calculate metrics for adaptive ensemble"""
    
    # 15-minute MAE
    ensemble_df['ae_adaptive'] = abs(ensemble_df['y'] - ensemble_df['yhat_adaptive'])
    mae_15min = ensemble_df.groupby(['location_id', 'sales_type_id', 'department_id'])['ae_adaptive'].mean()
    
    # Hourly
    ensemble_df['hour_floor'] = ensemble_df['ds'].dt.floor('h')
    hourly_df = ensemble_df.groupby(['location_id', 'sales_type_id', 'department_id', 'hour_floor']).agg({
        'y': 'sum',
        'yhat_adaptive': 'sum'
    }).reset_index()
    hourly_df['ae_hourly'] = abs(hourly_df['y'] - hourly_df['yhat_adaptive'])
    mae_hourly = hourly_df.groupby(['location_id', 'sales_type_id', 'department_id'])['ae_hourly'].mean()
    
    # Daily
    ensemble_df['date'] = ensemble_df['ds'].dt.date
    daily_df = ensemble_df.groupby(['location_id', 'sales_type_id', 'department_id', 'date']).agg({
        'y': 'sum',
        'yhat_adaptive': 'sum'
    }).reset_index()
    daily_df['ae_daily'] = abs(daily_df['y'] - daily_df['yhat_adaptive'])
    mae_daily = daily_df.groupby(['location_id', 'sales_type_id', 'department_id'])['ae_daily'].mean()
    
    # Calculate adaptive MAE
    adaptive_mae = {
        '15min': mae_15min.median(),
        'hourly': mae_hourly.median(),
        'daily': mae_daily.median(),
        'combined': (mae_15min.median() + mae_hourly.median() + mae_daily.median()) / 3
    }
    
    # Calculate improvements
    improvements = {
        '15min': (baseline_mae['15min'] - adaptive_mae['15min']) / baseline_mae['15min'] * 100,
        'hourly': (baseline_mae['hourly'] - adaptive_mae['hourly']) / baseline_mae['hourly'] * 100,
        'daily': (baseline_mae['daily'] - adaptive_mae['daily']) / baseline_mae['daily'] * 100,
        'combined': (baseline_mae['combined'] - adaptive_mae['combined']) / baseline_mae['combined'] * 100
    }
    
    print(f"\n{'='*60}")
    print(f"ADAPTIVE ENSEMBLE RESULTS")
    print(f"{'='*60}")
    print(f"\nAdaptive Ensemble MAE:")
    print(f"  15-min: {adaptive_mae['15min']:.4f} (improvement: {improvements['15min']:.1f}%)")
    print(f"  Hourly: {adaptive_mae['hourly']:.4f} (improvement: {improvements['hourly']:.1f}%)")
    print(f"  Daily: {adaptive_mae['daily']:.4f} (improvement: {improvements['daily']:.1f}%)")
    print(f"  Combined: {adaptive_mae['combined']:.4f} (improvement: {improvements['combined']:.1f}%)")
    
    # Save metrics
    with open('results/adaptive_ensemble_metrics.json', 'w') as f:
        json.dump({
            'baseline_mae': baseline_mae,
            'adaptive_mae': adaptive_mae,
            'improvements': improvements
        }, f, indent=2)
    
    return improvements

if __name__ == '__main__':
    create_adaptive_ensemble()