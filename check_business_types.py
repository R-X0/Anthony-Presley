# check_business_types.py
import pandas as pd

def check_business_types():
    """Check what types of businesses we're dealing with"""
    
    # Load data
    location_df = pd.read_csv('data/locations.csv')
    department_df = pd.read_csv('data/department.csv')
    sales_type_df = pd.read_csv('data/sales_type.csv')
    
    print("Corporation Analysis")
    print("="*60)
    
    # Map from the original data description
    corp_map = {
        46113: 'restaurant (Canadian)',
        78415: 'pizza chain',
        89766: 'grocery',
        90816: 'grocery',
        91133: 'restaurant',
        91906: 'grocery',
        91908: 'grocery',
        91995: 'grocery',
        92250: 'restaurant',
        92274: 'restaurant'
    }
    
    # Show locations by corporation
    for corp_id, corp_type in corp_map.items():
        locs = location_df[location_df['corporation_id'] == corp_id]
        if len(locs) > 0:
            print(f"\nCorporation {corp_id} ({corp_type}):")
            for _, loc in locs.iterrows():
                print(f"  Location {loc['id']}")
                
                # Show departments for this corporation
                depts = department_df[department_df['corporation_id'] == corp_id]
                if len(depts) > 0 and len(depts) < 10:  # Only show if reasonable number
                    print(f"    Departments: {', '.join(depts['name'].head(5).tolist())}")

if __name__ == '__main__':
    check_business_types()