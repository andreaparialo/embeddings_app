#!/usr/bin/env python3
"""
Fix to properly identify which products exist in which indexes
"""

import logging
import pandas as pd
from dual_index_data_loader import dual_index_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_index_membership():
    """Analyze actual index membership"""
    
    # Initialize
    csv_path = "database_results/DB_ACTIVE.csv"
    if not dual_index_loader.initialize(csv_path):
        return
    
    # Get all products from main index
    main_products = set()
    for path in dual_index_loader.main_metadata['image_paths']:
        filename_root = path.split('/')[-1].replace('.jpg', '')
        main_products.add(filename_root)
    
    # Get all products from measurement index
    measurement_products = set()
    for info in dual_index_loader.measurement_path_mapping.values():
        filename_root = info['normalized'].split('/')[-1].replace('.jpg', '')
        measurement_products.add(filename_root)
    
    # Calculate overlaps
    both_indexes = main_products & measurement_products
    main_only = main_products - measurement_products
    measurement_only = measurement_products - main_products
    
    logger.info(f"\n📊 ACTUAL INDEX MEMBERSHIP:")
    logger.info(f"   Products in both indexes: {len(both_indexes):,} ({len(both_indexes)/len(main_products)*100:.1f}% of main)")
    logger.info(f"   Products in main only: {len(main_only):,}")
    logger.info(f"   Products in measurement only: {len(measurement_only):,}")
    logger.info(f"   Total unique products: {len(main_products | measurement_products):,}")
    
    # Test with specific SKU
    test_sku_root = "20560805NCUC"
    logger.info(f"\n🔍 Testing {test_sku_root}:")
    logger.info(f"   In main index: {test_sku_root in main_products}")
    logger.info(f"   In measurement index: {test_sku_root in measurement_products}")
    logger.info(f"   In both: {test_sku_root in both_indexes}")
    
    # Sample some products in both indexes
    logger.info(f"\n📋 Sample products in BOTH indexes:")
    for i, product in enumerate(list(both_indexes)[:5]):
        logger.info(f"   {product}")
    
    # The key insight: most products SHOULD be in both indexes!
    # We need to update the dual index system to track this properly

if __name__ == "__main__":
    analyze_index_membership()