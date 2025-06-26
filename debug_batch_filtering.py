#!/usr/bin/env python3
"""
Debug batch filtering to understand why STARTSKU_DATE appears to be filtered
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def simulate_batch_search():
    """Simulate the batch search with the user's exact filters"""
    
    # Load the actual data
    csv_path = "database_results/final_with_aws_shapes_20250625_155822.csv"
    
    print("Loading CSV data...")
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(csv_path, encoding='latin-1')
        except:
            print("Failed to load CSV")
            return
    
    print(f"Loaded {len(df)} rows")
    
    # Convert SKU_COD to string
    df['SKU_COD'] = df['SKU_COD'].astype(str).str.strip()
    
    # User's selected filters
    matching_cols = [
        'BRAND_DES',
        'USERGENDER_DES', 
        'PRODUCT_TYPE_COD',
        'CTM_FIRST_FRONT_MATERIAL_DES',
        'FITTING_DES',
        'FlatTop_FlatTop_1',
        'GRANULAR_SHAPE_AWS',
        'MACRO_SHAPE_AWS',
        'bridge_Bridge_1',
        'browline_browline_1'
    ]
    
    # Find SKUs with STARTSKU_DATE = 2026-01-01
    target_date = '2026-01-01'
    skus_with_target_date = df[df['STARTSKU_DATE'] == target_date]
    print(f"\nFound {len(skus_with_target_date)} SKUs with STARTSKU_DATE = {target_date}")
    
    if len(skus_with_target_date) == 0:
        print("No SKUs found with target date!")
        return
    
    # Take first 5 as examples
    example_skus = skus_with_target_date.head(5)
    
    print("\nExample SKUs with 2026-01-01:")
    for idx, row in example_skus.iterrows():
        print(f"\nSKU: {row['SKU_COD']}")
        print(f"  filename_root: {row['filename_root']}")
        for col in matching_cols:
            if col in df.columns:
                print(f"  {col}: {row.get(col, 'N/A')}")
    
    # Now let's check what happens when we filter
    print("\n" + "="*60)
    print("TESTING FILTER BEHAVIOR")
    print("="*60)
    
    # Take one example SKU
    test_row = skus_with_target_date.iloc[0]
    test_sku = test_row['SKU_COD']
    print(f"\nTest SKU: {test_sku}")
    print(f"Test SKU STARTSKU_DATE: {test_row['STARTSKU_DATE']}")
    
    # Build filters based on matching columns
    filters = {}
    for col in matching_cols:
        if col in test_row and pd.notna(test_row[col]):
            filters[col] = test_row[col]
            print(f"Filter: {col} = {test_row[col]}")
    
    # Apply filters to find similar products
    mask = pd.Series([True] * len(df))
    for col, value in filters.items():
        if col in df.columns:
            mask &= (df[col] == value)
            remaining = mask.sum()
            print(f"  After {col} filter: {remaining} products remain")
    
    filtered_df = df[mask]
    print(f"\nTotal products after all filters: {len(filtered_df)}")
    
    # Check STARTSKU_DATE distribution
    date_dist = filtered_df['STARTSKU_DATE'].value_counts()
    print(f"\nSTARTSKU_DATE distribution in filtered results:")
    for date, count in date_dist.head(10).items():
        print(f"  {date}: {count} products ({count/len(filtered_df)*100:.1f}%)")
    
    # Check if all have same date
    unique_dates = filtered_df['STARTSKU_DATE'].unique()
    if len(unique_dates) == 1:
        print(f"\n⚠️  WARNING: All filtered products have the same STARTSKU_DATE: {unique_dates[0]}")
    
    # Let's check which filter is causing this
    print("\n" + "="*60)
    print("FINDING THE CULPRIT FILTER")
    print("="*60)
    
    # Test each filter individually
    for test_col in matching_cols:
        if test_col in filters and test_col in df.columns:
            single_filter_mask = df[test_col] == filters[test_col]
            single_filtered = df[single_filter_mask]
            
            # Check date distribution for this filter alone
            dates = single_filtered['STARTSKU_DATE'].value_counts()
            unique_count = len(dates)
            
            print(f"\n{test_col} = '{filters[test_col]}':")
            print(f"  Matches: {len(single_filtered)} products")
            print(f"  Unique STARTSKU_DATE values: {unique_count}")
            
            if unique_count <= 3:
                print(f"  ⚠️  SUSPICIOUS: Only {unique_count} unique dates!")
                for date, count in dates.items():
                    print(f"    {date}: {count} products")

if __name__ == "__main__":
    simulate_batch_search() 