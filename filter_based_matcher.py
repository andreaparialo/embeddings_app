#!/usr/bin/env python3
"""
Simple filter-based matcher - matches SKUs based on column filters only
No image search, just pure database filtering
"""

import pandas as pd
import os
import sys
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def filter_based_match(input_excel_path, matching_columns, allowed_statuses=['IL'], exclude_same_model=True):
    """
    Match SKUs based on column filters only
    
    Args:
        input_excel_path: Path to Excel file with SKUs
        matching_columns: List of columns to match on
        allowed_statuses: List of allowed MD_SKU_STATUS_COD values
        exclude_same_model: Whether to exclude same MODEL_COD
    
    Returns:
        DataFrame with all matches
    """
    
    print("="*80)
    print("FILTER-BASED MATCHER (No Image Search)")
    print("="*80)
    
    # Load the database CSV
    csv_path = "database_results/final_with_aws_shapes_20250625_155822.csv"
    print(f"\n1. Loading database from {csv_path}...")
    df_db = pd.read_csv(csv_path)
    print(f"   ✅ Loaded {len(df_db):,} products from database")
    
    # Apply baseline filters (status and date)
    print("\n2. Applying baseline filters...")
    initial_count = len(df_db)
    
    # Status filter
    if allowed_statuses:
        df_db = df_db[df_db['MD_SKU_STATUS_COD'].isin(allowed_statuses)]
        print(f"   ✅ Status filter: {initial_count:,} → {len(df_db):,} products")
    
    # Apply the baseline date filters from config
    import config_filtering
    if config_filtering.ENABLE_BASELINE_DATE_FILTER:
        date_mask = ~df_db['STARTSKU_DATE'].apply(config_filtering.should_exclude_by_baseline_date)
        df_db = df_db[date_mask]
        print(f"   ✅ Date filter: {len(df_db):,} products remain")
    
    # Load input Excel
    print(f"\n3. Loading input Excel: {input_excel_path}")
    df_input = pd.read_excel(input_excel_path)
    input_skus = df_input.iloc[:, 0].astype(str).str.strip().tolist()
    print(f"   ✅ Found {len(input_skus):,} input SKUs")
    
    # Find matches for each input SKU
    print(f"\n4. Finding matches based on columns: {matching_columns}")
    all_results = []
    
    for i, input_sku in enumerate(input_skus):
        if i % 100 == 0:
            print(f"   Processing SKU {i+1}/{len(input_skus)}...")
        
        # Find the input SKU in database
        input_row = df_db[df_db['SKU_COD'].astype(str).str.strip() == input_sku]
        
        if input_row.empty:
            print(f"   ⚠️ SKU {input_sku} not found in database")
            continue
        
        # Get filter values from the input SKU
        input_data = input_row.iloc[0]
        
        # Build filter mask
        mask = pd.Series([True] * len(df_db))
        
        # Apply each matching column filter
        for col in matching_columns:
            if col in df_db.columns and col in input_data:
                input_value = input_data[col]
                if pd.notna(input_value):
                    mask &= (df_db[col] == input_value)
        
        # Apply exclude same model if requested
        if exclude_same_model and 'MODEL_COD' in input_data:
            input_model = input_data['MODEL_COD']
            if pd.notna(input_model):
                mask &= (df_db['MODEL_COD'] != input_model)
        
        # Get matching products
        matches = df_db[mask]
        
        # Create result rows
        for _, match_row in matches.iterrows():
            result = {
                'Input_SKU': input_sku,
                'Matched_SKU': match_row['SKU_COD'],
            }
            
            # Add the matching column values for comparison
            for col in matching_columns:
                result[f'Input_{col}'] = input_data.get(col, '')
                result[f'Matched_{col}'] = match_row.get(col, '')
            
            # Add some additional useful columns
            result['Matched_MODEL_COD'] = match_row.get('MODEL_COD', '')
            result['Matched_STARTSKU_DATE'] = match_row.get('STARTSKU_DATE', '')
            result['Matched_MD_SKU_STATUS_COD'] = match_row.get('MD_SKU_STATUS_COD', '')
            
            all_results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    print(f"\n5. Results Summary:")
    print(f"   ✅ Total matches found: {len(results_df):,}")
    print(f"   ✅ Input SKUs with matches: {results_df['Input_SKU'].nunique():,}")
    print(f"   ✅ Average matches per SKU: {len(results_df) / results_df['Input_SKU'].nunique():.1f}")
    
    return results_df


def main():
    """Example usage"""
    
    # Test with a sample Excel file
    test_file = "test_filter_match.xlsx"
    
    # Create a test file if it doesn't exist
    if not os.path.exists(test_file):
        print("Creating test Excel file...")
        test_skus = [
            '20872780S53HA',
            '1097429005220', 
            '208727FMP539O',
            '20813570L48IT',
            '209027C9A5417'
        ]
        pd.DataFrame({'SKU_COD': test_skus}).to_excel(test_file, index=False)
    
    # Define matching columns
    matching_columns = [
        'BRAND_DES',
        'USERGENDER_DES', 
        'PRODUCT_TYPE_COD',
        'ACT_SKU_PRICE_RANGE_DES',
        'FITTING_DES',
        'GRANULAR_SHAPE_AWS',
        'MACRO_SHAPE_AWS',
        'CTM_FIRST_FRONT_MATERIAL_DES',
        'FlatTop_FlatTop_1',
        'RIM_TYPE_DES',
        'bridge_Bridge_1',
        'browline_browline_1'
    ]
    
    # Run the matcher
    results_df = filter_based_match(
        test_file,
        matching_columns,
        allowed_statuses=['IL'],
        exclude_same_model=True
    )
    
    # Save results
    output_file = f"filter_match_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    results_df.to_excel(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")
    
    # Show sample results
    if len(results_df) > 0:
        print("\nSample results (first 5 matches):")
        print(results_df.head())


if __name__ == "__main__":
    main() 