#!/usr/bin/env python3
"""
Merge old and new databases with proper handling of new columns
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("🔄 Database Merge Process")
print("=" * 60)

# Load databases
print("\n📖 Loading databases...")
old_db = pd.read_csv("database_results/DB_FINAL_SIMILARIT_270615.csv")
new_db = pd.read_csv("database_results/DB_SIMILARITY_NF.csv")

print(f"   Old DB: {old_db.shape}")
print(f"   New DB: {new_db.shape}")

# Analyze the merge strategy
print("\n📊 Merge Strategy:")
print("   1. Use new DB as the base (has more SKUs and new columns)")
print("   2. Map images using 'filename_root' column")
print("   3. Preserve data from old DB for SKUs that were removed")

# First, let's understand what columns we need to handle
old_cols = set(old_db.columns)
new_cols = set(new_db.columns)
common_cols = old_cols & new_cols

print(f"\n📋 Column Analysis:")
print(f"   Common columns: {len(common_cols)}")
print(f"   New columns: {sorted(new_cols - old_cols)}")
print(f"   Removed columns: {sorted(old_cols - new_cols)}")

# Create the merged database
print("\n🔨 Creating merged database...")

# Start with the new database as base
merged_db = new_db.copy()

# Add removed columns from old DB with default values
removed_cols = old_cols - new_cols
for col in removed_cols:
    if col not in ['BRIDGE_LENGTH_VAL']:  # Skip this as we have BRIDGE_LENGTH now
        merged_db[col] = np.nan

# For SKUs that exist in both, we might want to preserve some old data
# But since new DB should be authoritative, we'll keep new DB values

# Handle SKUs that were removed (exist in old but not in new)
old_skus = set(old_db['SKU_COD'].astype(str))
new_skus = set(new_db['SKU_COD'].astype(str))
removed_skus = old_skus - new_skus

print(f"\n📝 Handling removed SKUs: {len(removed_skus)}")

if len(removed_skus) > 0:
    # Get rows for removed SKUs from old DB
    removed_rows = old_db[old_db['SKU_COD'].astype(str).isin(removed_skus)].copy()
    
    # Add new columns with default values
    removed_rows['BRIDGE_LENGTH'] = np.nan
    removed_rows['LENS_BASE'] = np.nan
    
    # Ensure all columns match
    for col in merged_db.columns:
        if col not in removed_rows.columns:
            removed_rows[col] = np.nan
    
    # Reorder columns to match merged_db
    removed_rows = removed_rows[merged_db.columns]
    
    # Append removed rows
    merged_db = pd.concat([merged_db, removed_rows], ignore_index=True)

print(f"\n✅ Merged database created:")
print(f"   Total rows: {len(merged_db)}")
print(f"   Total columns: {len(merged_db.columns)}")

# Data quality checks
print("\n🔍 Data Quality Checks:")
print(f"   Duplicate SKUs: {merged_db['SKU_COD'].duplicated().sum()}")
print(f"   Null SKUs: {merged_db['SKU_COD'].isna().sum()}")
print(f"   Null filename_root: {merged_db['filename_root'].isna().sum()}")
print(f"   BRIDGE_LENGTH coverage: {merged_db['BRIDGE_LENGTH'].notna().sum()} ({merged_db['BRIDGE_LENGTH'].notna().sum()/len(merged_db)*100:.1f}%)")
print(f"   LENS_BASE coverage: {merged_db['LENS_BASE'].notna().sum()} ({merged_db['LENS_BASE'].notna().sum()/len(merged_db)*100:.1f}%)")

# Save the merged database
output_path = f"database_results/DB_MERGED_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
merged_db.to_csv(output_path, index=False)
print(f"\n💾 Merged database saved to: {output_path}")

# Also save as the new active database
active_path = "database_results/DB_ACTIVE.csv"
merged_db.to_csv(active_path, index=False)
print(f"💾 Also saved as active database: {active_path}")

# Create a summary report
print("\n📊 Summary Report:")
print("=" * 40)
print(f"Original old DB SKUs: {len(old_skus)}")
print(f"Original new DB SKUs: {len(new_skus)}")
print(f"Final merged DB SKUs: {merged_db['SKU_COD'].nunique()}")
print(f"SKUs with images (via filename_root): {merged_db['filename_root'].notna().sum()}")

# Column order recommendation
print("\n💡 Recommended column order for frontend:")
important_cols = [
    'SKU_COD', 'MODEL_COD', 'BRAND_DES', 'filename_root',
    'USERGENDER_DES', 'COLOR', 'COLOR_FAMILY_1_DES',
    'SHAPE_SEMI_GROUPED', 'GRANULAR_SHAPE_AWS',
    'SIZE_COD', 'LENSHEIGHTVAL', 'BRIDGE_LENGTH', 'LENS_BASE',
    'CTM_FIRST_FRONT_MATERIAL_DES', 'CTM_FIRST_TEMPLE_MATERIAL_DES',
    'RIM_TYPE_DES', 'PRODUCT_TYPE_COD', 'MD_SKU_STATUS_COD'
]

print("\nKey columns preserved:")
for col in important_cols:
    if col in merged_db.columns:
        print(f"   ✓ {col}") 