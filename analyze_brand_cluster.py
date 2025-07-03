#!/usr/bin/env python3
"""
Analyze DB_SIMILARITY_NF_BRAND_CLUSTER.xlsx and prepare for merging with existing database
"""

import pandas as pd
import os

def analyze_brand_cluster_file():
    """Analyze the brand cluster Excel file and existing database"""
    
    print("="*80)
    print("BRAND CLUSTER DATABASE ANALYSIS")
    print("="*80)
    
    # File paths
    excel_path = "database_results/DB_SIMILARITY_NF_BRAND_CLUSTER.xlsx"
    active_db_path = "database_results/DB_ACTIVE.csv"
    
    # Load Excel file
    print(f"\n1. Loading Excel file: {excel_path}")
    df_excel = pd.read_excel(excel_path)
    print(f"   ✅ Excel shape: {df_excel.shape}")
    print(f"   📋 Columns: {list(df_excel.columns)}")
    
    # Load existing database
    print(f"\n2. Loading existing database: {active_db_path}")
    df_active = pd.read_csv(active_db_path)
    print(f"   ✅ Active DB shape: {df_active.shape}")
    print(f"   📋 Columns: {list(df_active.columns)}")
    
    # Check for BRAND_CLUSTER column
    print(f"\n3. Checking BRAND_CLUSTER column...")
    if 'BRAND_CLUSTER' in df_excel.columns:
        print(f"   ✅ BRAND_CLUSTER column found in Excel file")
        unique_clusters = df_excel['BRAND_CLUSTER'].nunique()
        null_clusters = df_excel['BRAND_CLUSTER'].isnull().sum()
        print(f"   📊 Unique brand clusters: {unique_clusters}")
        print(f"   📊 Null values: {null_clusters} ({null_clusters/len(df_excel)*100:.1f}%)")
        
        # Show cluster distribution
        print(f"\n   🏷️  Brand cluster distribution:")
        cluster_counts = df_excel['BRAND_CLUSTER'].value_counts()
        for cluster, count in cluster_counts.head(10).items():
            print(f"      {cluster}: {count} products")
        if len(cluster_counts) > 10:
            print(f"      ... and {len(cluster_counts) - 10} more clusters")
    else:
        print(f"   ❌ BRAND_CLUSTER column NOT found in Excel file")
        return
    
    # Check for SKU_COD column for merging
    print(f"\n4. Checking merge key column...")
    if 'SKU_COD' in df_excel.columns:
        print(f"   ✅ SKU_COD column found - can merge on SKU codes")
        excel_skus = set(df_excel['SKU_COD'].astype(str).str.strip())
        active_skus = set(df_active['SKU_COD'].astype(str).str.strip())
        
        print(f"   📊 Excel SKUs: {len(excel_skus):,}")
        print(f"   📊 Active DB SKUs: {len(active_skus):,}")
        
        # Find matches and differences
        common_skus = excel_skus & active_skus
        excel_only = excel_skus - active_skus
        active_only = active_skus - excel_skus
        
        print(f"   🎯 Common SKUs: {len(common_skus):,} ({len(common_skus)/len(active_skus)*100:.1f}% of active DB)")
        print(f"   📤 Excel-only SKUs: {len(excel_only):,}")
        print(f"   📤 Active-only SKUs: {len(active_only):,}")
        
        if excel_only:
            print(f"\n   🔍 Sample Excel-only SKUs:")
            for sku in list(excel_only)[:5]:
                print(f"      {sku}")
        
        if active_only:
            print(f"\n   🔍 Sample Active-only SKUs:")
            for sku in list(active_only)[:5]:
                print(f"      {sku}")
    else:
        print(f"   ❌ SKU_COD column NOT found - need to find alternative merge key")
        print(f"   🔍 Available columns in Excel: {list(df_excel.columns)}")
        return
    
    # Check if BRAND_CLUSTER already exists in active DB
    print(f"\n5. Checking if BRAND_CLUSTER already exists in active database...")
    if 'BRAND_CLUSTER' in df_active.columns:
        print(f"   ⚠️  BRAND_CLUSTER already exists in active database")
        existing_clusters = df_active['BRAND_CLUSTER'].nunique()
        existing_nulls = df_active['BRAND_CLUSTER'].isnull().sum()
        print(f"   📊 Existing unique clusters: {existing_clusters}")
        print(f"   📊 Existing null values: {existing_nulls} ({existing_nulls/len(df_active)*100:.1f}%)")
    else:
        print(f"   ✅ BRAND_CLUSTER does not exist in active database - safe to add")
    
    # Sample data preview
    print(f"\n6. Sample data preview...")
    if 'BRAND_CLUSTER' in df_excel.columns and 'SKU_COD' in df_excel.columns:
        print(f"   📋 Excel file sample:")
        sample_cols = ['SKU_COD', 'BRAND_CLUSTER']
        if 'BRAND_DES' in df_excel.columns:
            sample_cols.insert(1, 'BRAND_DES')
        
        print(df_excel[sample_cols].head(10).to_string(index=False))
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    # Recommendations
    print(f"\n📝 RECOMMENDATIONS:")
    if len(common_skus) > 0:
        coverage = len(common_skus) / len(active_skus) * 100
        if coverage > 90:
            print(f"   ✅ Excellent coverage ({coverage:.1f}%) - proceed with merge")
        elif coverage > 75:
            print(f"   ⚠️  Good coverage ({coverage:.1f}%) - proceed with caution")
        else:
            print(f"   ❌ Low coverage ({coverage:.1f}%) - investigate missing SKUs")
        
        print(f"   🔧 Merge strategy: LEFT JOIN on SKU_COD")
        print(f"   📊 Expected result: {len(df_active):,} rows with BRAND_CLUSTER coverage for {len(common_skus):,} SKUs")
    
    return {
        'excel_shape': df_excel.shape,
        'active_shape': df_active.shape,
        'common_skus': len(common_skus) if 'common_skus' in locals() else 0,
        'coverage_percent': len(common_skus) / len(active_skus) * 100 if 'common_skus' in locals() else 0,
        'brand_clusters': unique_clusters if 'BRAND_CLUSTER' in df_excel.columns else 0
    }

if __name__ == "__main__":
    analyze_brand_cluster_file() 