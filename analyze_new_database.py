#!/usr/bin/env python3
"""
Script to analyze the new database and convert it to CSV
"""

import pandas as pd
import numpy as np
import os

def analyze_database():
    # Read the Excel file
    excel_path = 'database_results/DB_FINAL_SIMILARIT_270615.xlsx'
    print(f"Reading Excel file: {excel_path}")
    
    try:
        df = pd.read_excel(excel_path)
        print(f"✅ Successfully read Excel file")
        print(f"Shape: {df.shape} ({df.shape[0]:,} rows, {df.shape[1]} columns)")
        
        # Convert to CSV
        csv_path = 'database_results/DB_FINAL_SIMILARIT_270615.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Converted to CSV: {csv_path}")
        
        # Analyze columns
        print("\n📊 Column Names and Data Types:")
        print("-" * 80)
        for col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].count()
            null_count = df[col].isna().sum()
            unique_count = df[col].nunique()
            print(f"{col:40} | {str(dtype):15} | Non-null: {non_null:,} | Null: {null_count:,} | Unique: {unique_count:,}")
        
        # Show first 50 rows
        print("\n📋 First 50 Rows Preview:")
        print("-" * 80)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        print(df.head(50))
        
        # Check for mixed data types
        print("\n🔍 Checking for Mixed Data Types:")
        print("-" * 80)
        for col in df.columns:
            # Sample values from the column
            sample_values = df[col].dropna().head(100)
            if len(sample_values) > 0:
                # Check if there are different types
                types = set()
                for val in sample_values:
                    types.add(type(val).__name__)
                
                if len(types) > 1:
                    print(f"⚠️  {col}: Mixed types found - {types}")
                    # Show examples
                    for t in types:
                        examples = [str(v) for v in sample_values if type(v).__name__ == t][:3]
                        print(f"    {t}: {examples}")
        
        # Compare with old database
        old_csv_path = 'database_results/final_with_aws_shapes_enriched.csv'
        if os.path.exists(old_csv_path):
            print("\n📊 Comparing with Old Database:")
            print("-" * 80)
            old_df = pd.read_csv(old_csv_path)
            
            old_cols = set(old_df.columns)
            new_cols = set(df.columns)
            
            removed_cols = old_cols - new_cols
            added_cols = new_cols - old_cols
            common_cols = old_cols & new_cols
            
            print(f"Total columns - Old: {len(old_cols)}, New: {len(new_cols)}")
            print(f"\n❌ Removed columns ({len(removed_cols)}):")
            for col in sorted(removed_cols):
                print(f"  - {col}")
            
            print(f"\n✅ Added columns ({len(added_cols)}):")
            for col in sorted(added_cols):
                print(f"  + {col}")
            
            print(f"\n📌 Common columns ({len(common_cols)}):")
            for col in sorted(common_cols):
                print(f"  = {col}")
        
        # Check critical columns
        print("\n🔑 Checking Critical Columns:")
        print("-" * 80)
        critical_cols = ['SKU_COD', 'filename_root', 'MODEL_COD', 'BRAND_DES', 
                        'USERGENDER_DES', 'PRODUCT_TYPE_COD', 'MD_SKU_STATUS_COD']
        
        for col in critical_cols:
            if col in df.columns:
                print(f"✅ {col}: Present")
            else:
                print(f"❌ {col}: MISSING!")
        
        # Save analysis report
        with open('database_analysis_report.txt', 'w') as f:
            f.write("DATABASE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"File: {excel_path}\n")
            f.write(f"Shape: {df.shape}\n")
            f.write(f"Columns: {list(df.columns)}\n")
            
        print("\n✅ Analysis complete! Report saved to database_analysis_report.txt")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_database() 