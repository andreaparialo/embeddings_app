#!/usr/bin/env python3
"""
Check data quality and patterns in the new database
"""

import pandas as pd
import numpy as np

def check_data_quality():
    # Read the new CSV
    df = pd.read_csv('database_results/DB_FINAL_SIMILARIT_270615.csv')
    
    print("🔍 Data Quality Check")
    print("=" * 80)
    
    # Check for duplicates
    print("\n📊 Duplicate Analysis:")
    print(f"Duplicate SKU_COD: {df['SKU_COD'].duplicated().sum()}")
    print(f"Duplicate filename_root: {df['filename_root'].duplicated().sum()}")
    
    # Check filename_root patterns
    print("\n📁 Filename Root Patterns:")
    sample_roots = df['filename_root'].head(20).tolist()
    print("Sample filename_roots:")
    for root in sample_roots[:10]:
        print(f"  - {root}")
    
    # Check if derivation rules still apply
    print("\n🔧 SKU to Filename Root Derivation Check:")
    errors = 0
    for idx, row in df.head(100).iterrows():
        sku = str(row['SKU_COD'])
        filename_root = str(row['filename_root'])
        model_cod = str(row['MODEL_COD'])
        
        # Check if model_cod matches first 6 chars of SKU
        if sku[:6] != model_cod:
            print(f"  ⚠️ Model mismatch: SKU={sku}, MODEL_COD={model_cod}")
            errors += 1
    
    if errors == 0:
        print("  ✅ All checked SKUs follow expected pattern")
    
    # Check status distribution
    print("\n📊 Status Distribution:")
    status_counts = df['MD_SKU_STATUS_COD'].value_counts()
    for status, count in status_counts.items():
        print(f"  {status}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Check date range
    print("\n📅 Date Range:")
    df['STARTSKU_DATE'] = pd.to_datetime(df['STARTSKU_DATE'])
    print(f"  Earliest: {df['STARTSKU_DATE'].min()}")
    print(f"  Latest: {df['STARTSKU_DATE'].max()}")
    print(f"  Unique dates: {df['STARTSKU_DATE'].nunique()}")
    
    # Check numeric columns for outliers
    print("\n📊 Numeric Column Statistics:")
    numeric_cols = ['SIZE_COD', 'FRONT_LENGTH_VAL', 'BRIDGE_LENGTH_VAL', 
                   'FRONT_HEIGHT_VAL', 'ACT_SKU_PRICE_VAL']
    
    for col in numeric_cols:
        if col in df.columns:
            print(f"\n  {col}:")
            print(f"    Min: {df[col].min()}")
            print(f"    Max: {df[col].max()}")
            print(f"    Mean: {df[col].mean():.2f}")
            print(f"    Std: {df[col].std():.2f}")
    
    # Check new columns
    print("\n✨ New Columns Analysis:")
    
    print("\n  COLOR:")
    print(f"    Unique values: {df['COLOR'].nunique()}")
    print(f"    Top 10: {df['COLOR'].value_counts().head(10).to_dict()}")
    
    print("\n  CTM_FIRST_TEMPLE_MATERIAL_DES:")
    print(f"    Values: {df['CTM_FIRST_TEMPLE_MATERIAL_DES'].value_counts().to_dict()}")
    
    print("\n  SHAPE_SEMI_GROUPED:")
    print(f"    Unique values: {df['SHAPE_SEMI_GROUPED'].nunique()}")
    print(f"    Distribution: {df['SHAPE_SEMI_GROUPED'].value_counts().to_dict()}")
    
    # Check for potential issues
    print("\n⚠️  Potential Issues:")
    
    # Check for missing critical values
    critical_cols = ['SKU_COD', 'filename_root', 'MODEL_COD']
    for col in critical_cols:
        null_count = df[col].isna().sum()
        if null_count > 0:
            print(f"  ❌ {col} has {null_count} null values!")
    
    # Check if all products are IL status
    if len(status_counts) == 1 and 'IL' in status_counts:
        print("  ⚠️  All products have 'IL' status - is this intentional?")
    
    # Save detailed report
    with open('data_quality_report.txt', 'w') as f:
        f.write("DATA QUALITY REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total rows: {len(df):,}\n")
        f.write(f"Total columns: {len(df.columns)}\n")
        f.write(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
        f.write(f"\nColumn data types:\n")
        for col, dtype in df.dtypes.items():
            f.write(f"  {col}: {dtype}\n")
    
    print("\n✅ Data quality check complete! Report saved to data_quality_report.txt")

if __name__ == "__main__":
    check_data_quality() 