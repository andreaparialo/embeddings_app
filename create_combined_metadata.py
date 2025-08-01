#!/usr/bin/env python3
"""
Create a combined metadata CSV that includes products from both indexes
This will help dual engine search work properly with measurement-only products
"""

import json
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_combined_metadata():
    """Create a combined CSV with all products from both indexes"""
    
    # Load existing database
    logger.info("📊 Loading existing database...")
    df_main = pd.read_csv("database_results/DB_ACTIVE.csv")
    logger.info(f"   Main database: {len(df_main)} rows")
    
    # Load measurement metadata
    logger.info("📏 Loading measurement metadata...")
    with open("indexes/index_measurements/corrected_metadata.json", 'r') as f:
        measurement_data = json.load(f)
    
    # Extract all products from measurement index
    measurement_products = []
    for idx_str, normalized_path in measurement_data.get("product_mapping", {}).items():
        filename_root = normalized_path.split('/')[-1].replace('.jpg', '')
        measurement_products.append({
            'filename_root': filename_root,
            'measurement_idx': int(idx_str),
            'normalized_path': normalized_path,
            'source': 'measurement_only'
        })
    
    df_measurement = pd.DataFrame(measurement_products)
    logger.info(f"   Measurement products: {len(df_measurement)} rows")
    
    # Find products that are in measurement but not in main
    main_filename_roots = set(df_main['filename_root'].unique())
    measurement_filename_roots = set(df_measurement['filename_root'].unique())
    
    measurement_only = measurement_filename_roots - main_filename_roots
    both_indexes = main_filename_roots & measurement_filename_roots
    main_only = main_filename_roots - measurement_filename_roots
    
    logger.info(f"\n📊 Product distribution:")
    logger.info(f"   Both indexes: {len(both_indexes)}")
    logger.info(f"   Main only: {len(main_only)}")
    logger.info(f"   Measurement only: {len(measurement_only)}")
    
    # Create rows for measurement-only products
    measurement_only_rows = []
    for filename_root in measurement_only:
        # Get measurement info
        meas_info = df_measurement[df_measurement['filename_root'] == filename_root].iloc[0]
        
        # Create a minimal row with default values
        row = {
            'SKU_COD': f'MEAS_{filename_root}',  # Synthetic SKU
            'filename_root': filename_root,
            'MD_SKU_STATUS_COD': 'MEASUREMENT_ONLY',  # Special status
            'source_index': 'measurement_only',
            'measurement_idx': meas_info['measurement_idx']
        }
        
        # Add default values for other columns
        for col in df_main.columns:
            if col not in row:
                row[col] = None
        
        measurement_only_rows.append(row)
    
    # Create combined dataframe
    df_measurement_only = pd.DataFrame(measurement_only_rows)
    df_combined = pd.concat([df_main, df_measurement_only], ignore_index=True)
    
    # Add index information to all rows
    df_combined['in_main_index'] = df_combined['filename_root'].isin(main_filename_roots)
    df_combined['in_measurement_index'] = df_combined['filename_root'].isin(measurement_filename_roots)
    
    logger.info(f"\n✅ Combined database: {len(df_combined)} rows")
    logger.info(f"   Original rows: {len(df_main)}")
    logger.info(f"   Added measurement-only: {len(df_measurement_only)}")
    
    # Save the combined database
    output_path = "database_results/DB_COMBINED_DUAL_INDEX.csv"
    df_combined.to_csv(output_path, index=False)
    logger.info(f"\n📄 Saved combined database to: {output_path}")
    
    # Show some examples of measurement-only products
    logger.info("\n📋 Sample measurement-only products:")
    for i, row in df_measurement_only.head(5).iterrows():
        logger.info(f"   {row['filename_root']} (SKU: {row['SKU_COD']})")
    
    return df_combined

if __name__ == "__main__":
    create_combined_metadata()