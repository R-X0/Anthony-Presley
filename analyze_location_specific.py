# analyze_location_specific.py
import pandas as pd
import numpy as np

def analyze_location_issues():
    """Deep dive into why certain locations aren't improving"""
    
    # Load data
    baseline_df = pd.read_csv('results/baseline_prophet_results.csv')
    ensemble_df = pd.read_csv('results/ensemble_predictions.csv')
    
    baseline_df['ds'] = pd.to_datetime(baseline_df['ds'])
    ensemble_df['ds'] = pd.to_datetime(ensemble_df['ds'])
    
    # Focus on problematic locations
    problematic_locations = [144249, 143027, 143099, 121752]  # Added 121752
    
    for loc_id in problematic_locations:
        print(f"\nAnalyzing Location {loc_id}")
        print("="*50)
        
        # Get data for this location
        loc_baseline = baseline_df[baseline_df['location_id'] == loc_id]
        loc_ensemble = ensemble_df[ensemble_df['location_id'] == loc_id]
        
        # Merge
        loc_merged = pd.merge(
            loc_baseline[['ds', 'sales_type_id', 'department_id', 'y', 'yhat']],
            loc_ensemble[['ds', 'sales_type_id', 'department_id', 'yhat_ensemble']],
            on=['ds', 'sales_type_id', 'department_id']
        )
        
        # Calculate errors
        loc_merged['baseline_error'] = abs(loc_merged['y'] - loc_merged['yhat'])
        loc_merged['ensemble_error'] = abs(loc_merged['y'] - loc_merged['yhat_ensemble'])
        
        # Statistics
        print(f"Total periods: {len(loc_merged)}")
        print(f"Baseline MAE: {loc_merged['baseline_error'].mean():.4f}")
        print(f"Ensemble MAE: {loc_merged['ensemble_error'].mean():.4f}")
        print(f"Improvement: {((loc_merged['baseline_error'].mean() - loc_merged['ensemble_error'].mean()) / loc_merged['baseline_error'].mean() * 100):.1f}%")
        
        # Check for zero sales periods
        zero_sales = loc_merged[loc_merged['y'] == 0]
        print(f"\nZero sales periods: {len(zero_sales)} out of {len(loc_merged)} ({len(zero_sales)/len(loc_merged)*100:.1f}%)")
        
        if len(zero_sales) > 0:
            baseline_zero_correct = (zero_sales['yhat'] == 0).sum()
            ensemble_zero_correct = (zero_sales['yhat_ensemble'] == 0).sum()
            print(f"Baseline predicts zero correctly: {baseline_zero_correct}/{len(zero_sales)} ({baseline_zero_correct/len(zero_sales)*100:.1f}%)")
            print(f"Ensemble predicts zero correctly: {ensemble_zero_correct}/{len(zero_sales)} ({ensemble_zero_correct/len(zero_sales)*100:.1f}%)")
            
            # Show average prediction when actual is 0
            print(f"When actual=0, baseline predicts avg: {zero_sales['yhat'].mean():.2f}")
            print(f"When actual=0, ensemble predicts avg: {zero_sales['yhat_ensemble'].mean():.2f}")
        
        # Check variance
        print(f"\nSales statistics:")
        print(f"Mean: {loc_merged['y'].mean():.2f}")
        print(f"Std: {loc_merged['y'].std():.2f}")
        print(f"Min: {loc_merged['y'].min():.2f}")
        print(f"Max: {loc_merged['y'].max():.2f}")
        
        # Look at sales_type distribution
        print(f"\nSales types in this location:")
        sales_type_counts = loc_merged.groupby('sales_type_id').size()
        for st_id, count in sales_type_counts.items():
            print(f"  Sales type {st_id}: {count} periods")

if __name__ == '__main__':
    analyze_location_issues()