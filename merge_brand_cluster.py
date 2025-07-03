#!/usr/bin/env python3
"""
Merge BRAND_CLUSTER column from Excel file into the active database
"""

import pandas as pd
import os
from datetime import datetime

def merge_brand_cluster():
    """Merge BRAND_CLUSTER column into active database"""
    
    print("="*80)
    print("MERGING BRAND_CLUSTER INTO ACTIVE DATABASE")
    print("="*80)
    
    # File paths
    excel_path = "database_results/DB_SIMILARITY_NF_BRAND_CLUSTER.xlsx"
    active_db_path = "database_results/DB_ACTIVE.csv"
    
    # Load files
    print(f"\n1. Loading files...")
    print(f"   📂 Excel file: {excel_path}")
    df_excel = pd.read_excel(excel_path)
    print(f"   ✅ Loaded Excel: {df_excel.shape[0]:,} rows, {df_excel.shape[1]} columns")
    
    print(f"   📂 Active database: {active_db_path}")
    df_active = pd.read_csv(active_db_path)
    print(f"   ✅ Loaded Active DB: {df_active.shape[0]:,} rows, {df_active.shape[1]} columns")
    
    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"database_results/DB_ACTIVE_backup_{timestamp}.csv"
    print(f"\n2. Creating backup...")
    print(f"   💾 Backup: {backup_path}")
    df_active.to_csv(backup_path, index=False)
    print(f"   ✅ Backup created successfully")
    
    # Prepare merge data (only SKU_COD and BRAND_CLUSTER)
    print(f"\n3. Preparing merge data...")
    df_merge = df_excel[['SKU_COD', 'BRAND_CLUSTER']].copy()
    
    # Clean SKU codes for consistent merging
    df_merge['SKU_COD'] = df_merge['SKU_COD'].astype(str).str.strip()
    df_active['SKU_COD'] = df_active['SKU_COD'].astype(str).str.strip()
    
    print(f"   📊 Merge data shape: {df_merge.shape}")
    print(f"   📊 Unique SKUs in merge data: {df_merge['SKU_COD'].nunique():,}")
    print(f"   📊 Unique brand clusters: {df_merge['BRAND_CLUSTER'].nunique()}")
    
    # Check for duplicates in merge data
    duplicates = df_merge[df_merge['SKU_COD'].duplicated()]
    if len(duplicates) > 0:
        print(f"   ⚠️  Warning: {len(duplicates)} duplicate SKUs in merge data")
        # Keep first occurrence
        df_merge = df_merge.drop_duplicates(subset=['SKU_COD'], keep='first')
        print(f"   🔧 Removed duplicates, merge data shape: {df_merge.shape}")
    
    # Perform LEFT JOIN
    print(f"\n4. Performing merge...")
    print(f"   🔧 Strategy: LEFT JOIN on SKU_COD")
    print(f"   📊 Before merge: {df_active.shape}")
    
    df_merged = df_active.merge(df_merge, on='SKU_COD', how='left')
    print(f"   📊 After merge: {df_merged.shape}")
    
    # Analyze merge results
    print(f"\n5. Analyzing merge results...")
    total_rows = len(df_merged)
    with_brand_cluster = df_merged['BRAND_CLUSTER'].notna().sum()
    without_brand_cluster = df_merged['BRAND_CLUSTER'].isna().sum()
    
    print(f"   📊 Total rows: {total_rows:,}")
    print(f"   ✅ With BRAND_CLUSTER: {with_brand_cluster:,} ({with_brand_cluster/total_rows*100:.1f}%)")
    print(f"   ⚪ Without BRAND_CLUSTER: {without_brand_cluster:,} ({without_brand_cluster/total_rows*100:.1f}%)")
    
    # Show brand cluster distribution
    if with_brand_cluster > 0:
        print(f"\n   🏷️  Brand cluster distribution:")
        cluster_counts = df_merged['BRAND_CLUSTER'].value_counts()
        for cluster, count in cluster_counts.items():
            percentage = count / with_brand_cluster * 100
            print(f"      {cluster}: {count:,} ({percentage:.1f}%)")
    
    # Sample of SKUs without brand cluster
    if without_brand_cluster > 0:
        missing_skus = df_merged[df_merged['BRAND_CLUSTER'].isna()]['SKU_COD'].head(10).tolist()
        print(f"\n   🔍 Sample SKUs without BRAND_CLUSTER:")
        for sku in missing_skus:
            print(f"      {sku}")
    
    # Verify no data loss
    print(f"\n6. Verification...")
    if len(df_merged) == len(df_active):
        print(f"   ✅ Row count preserved: {len(df_merged):,} rows")
    else:
        print(f"   ❌ Row count mismatch! Before: {len(df_active):,}, After: {len(df_merged):,}")
        return False
    
    # Check for unexpected duplicates
    if df_merged['SKU_COD'].nunique() == df_active['SKU_COD'].nunique():
        print(f"   ✅ Unique SKU count preserved: {df_merged['SKU_COD'].nunique():,}")
    else:
        print(f"   ❌ Unique SKU count changed! Before: {df_active['SKU_COD'].nunique():,}, After: {df_merged['SKU_COD'].nunique():,}")
        return False
    
    # Save merged database
    print(f"\n7. Saving merged database...")
    output_path = "database_results/DB_ACTIVE.csv"
    df_merged.to_csv(output_path, index=False)
    print(f"   💾 Saved to: {output_path}")
    print(f"   📊 Final shape: {df_merged.shape}")
    
    # Column summary
    print(f"\n8. Final column summary...")
    print(f"   📋 Total columns: {len(df_merged.columns)}")
    print(f"   ➕ Added: BRAND_CLUSTER")
    print(f"   📊 New column position: {list(df_merged.columns).index('BRAND_CLUSTER') + 1}/{len(df_merged.columns)}")
    
    print(f"\n" + "="*80)
    print("MERGE COMPLETE!")
    print("="*80)
    print(f"✅ BRAND_CLUSTER column successfully added to active database")
    print(f"📊 Coverage: {with_brand_cluster:,}/{total_rows:,} SKUs ({with_brand_cluster/total_rows*100:.1f}%)")
    print(f"💾 Backup available at: {backup_path}")
    print(f"🎯 Ready for use in search and filtering!")
    
    return True

if __name__ == "__main__":
    merge_brand_cluster() 