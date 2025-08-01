#!/usr/bin/env python3
"""
Test dual engine search with specific SKU and filters
"""

import numpy as np
import pandas as pd
import logging
from dual_index_data_loader import dual_index_loader
from batch_processor_optimized import OptimizedBatchProcessor
from data_loader import DataLoader
from gme_model import GMEModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_specific_sku():
    """Test with SKU 2056085NC52UC and specific filters"""
    
    # Initialize components
    logger.info("🚀 Initializing components...")
    data_loader = DataLoader()
    gme_model = GMEModel()
    
    # Load data
    csv_path = "database_results/DB_ACTIVE.csv"
    df = data_loader.load_csv(csv_path)
    if df is None:
        logger.error("Failed to load CSV")
        return
    
    # Initialize FAISS index
    if not data_loader.load_faiss_index("indexes", "1095", "1095"):
        logger.error("Failed to load FAISS index")
        return
    
    # Initialize dual index
    if not dual_index_loader.initialize(csv_path):
        logger.error("Failed to initialize dual index")
        return
    
    # Find the test SKU
    test_sku = "2056085NC52UC"
    logger.info(f"\n🔍 Looking for SKU: {test_sku}")
    
    sku_data = df[df['SKU_COD'] == test_sku]
    if sku_data.empty:
        logger.error(f"SKU {test_sku} not found in database")
        return
    
    # Get the product info
    product = sku_data.iloc[0]
    filename_root = product['filename_root']
    
    logger.info(f"\n📊 Product details:")
    logger.info(f"   SKU: {test_sku}")
    logger.info(f"   Filename root: {filename_root}")
    logger.info(f"   Status: {product.get('MD_SKU_STATUS_COD', 'N/A')}")
    logger.info(f"   Brand: {product.get('BRAND_DES', 'N/A')}")
    logger.info(f"   Product Type: {product.get('PRODUCT_TYPE_COD', 'N/A')}")
    logger.info(f"   Shape Semi Grouped: {product.get('SHAPE_SEMI_GROUPED', 'N/A')}")
    
    # Check if this product exists in both indexes
    logger.info(f"\n🔍 Checking index presence...")
    
    # Check main index
    if filename_root in data_loader.filename_to_idx:
        main_idx = data_loader.filename_to_idx[filename_root]
        logger.info(f"   ✅ Found in main index at position {main_idx}")
        main_embedding = data_loader.embeddings[main_idx]
    else:
        logger.error(f"   ❌ NOT found in main index")
        return
    
    # Check measurement index
    measurement_embedding = dual_index_loader.get_measurement_embedding_by_filename(filename_root)
    if measurement_embedding is not None:
        logger.info(f"   ✅ Found in measurement index")
    else:
        logger.info(f"   ❌ NOT found in measurement index")
    
    # Create filters
    filters = {
        'BRAND_DES': product['BRAND_DES'],
        'PRODUCT_TYPE_COD': product['PRODUCT_TYPE_COD'],
        'SHAPE_SEMI_GROUPED': product['SHAPE_SEMI_GROUPED']
    }
    
    logger.info(f"\n📋 Using filters: {filters}")
    
    # Test 1: Global mode search
    logger.info(f"\n🌍 Testing GLOBAL mode (search all, then filter)...")
    
    # Search main index without filters
    main_distances_global, main_indices_global = dual_index_loader.search_main_index(
        main_embedding, 30, filters=None
    )
    
    # Search measurement index without filters
    if measurement_embedding is not None:
        meas_distances_global, meas_indices_global = dual_index_loader.search_measurement_index(
            measurement_embedding, 30
        )
    else:
        meas_distances_global = np.array([])
        meas_indices_global = np.array([])
    
    # Combine results
    combined_global = dual_index_loader.combine_search_results(
        main_distances_global, main_indices_global,
        meas_distances_global, meas_indices_global,
        top_k=20
    )
    
    logger.info(f"\nGlobal mode results: {len(combined_global)} products")
    
    # Test 2: Filtered mode search
    logger.info(f"\n🔍 Testing FILTERED mode (filter first, then search)...")
    
    # Search main index with filters
    main_distances_filtered, main_indices_filtered = dual_index_loader.search_main_index(
        main_embedding, 30, filters=filters
    )
    
    # Search measurement index with filters
    if measurement_embedding is not None:
        meas_distances_filtered, meas_indices_filtered = dual_index_loader.search_measurement_index_with_filters(
            measurement_embedding, 30, filters
        )
    else:
        meas_distances_filtered = np.array([])
        meas_indices_filtered = np.array([])
    
    # Combine results
    combined_filtered = dual_index_loader.combine_search_results(
        main_distances_filtered, main_indices_filtered,
        meas_distances_filtered, meas_indices_filtered,
        top_k=20
    )
    
    logger.info(f"\nFiltered mode results: {len(combined_filtered)} products")
    
    # Now test batch processor
    logger.info(f"\n🚀 Testing Batch Processor...")
    
    # Create batch processor
    batch_processor = OptimizedBatchProcessor(None, data_loader, gme_model)
    
    # Prepare image groups
    image_groups = {
        filename_root: {
            'skus': [test_sku],
            'source_item': product.to_dict()
        }
    }
    
    # Test both modes
    for mode in ['global', 'filtered']:
        logger.info(f"\n📊 Batch processor in {mode.upper()} mode:")
        
        results = batch_processor.process_image_groups_with_prefilter(
            image_groups=image_groups,
            matching_cols=['BRAND_DES', 'PRODUCT_TYPE_COD', 'SHAPE_SEMI_GROUPED'],
            max_results_per_sku=20,
            dual_engine_enabled=True,
            main_weight=0.7,
            measurement_weight=0.3,
            search_mode=mode
        )
        
        logger.info(f"   Total results: {len(results)}")
        
        if results:
            # Analyze index coverage
            df_results = pd.DataFrame(results)
            if 'Index_Coverage' in df_results.columns:
                coverage_counts = df_results['Index_Coverage'].value_counts()
                logger.info("   Index Coverage:")
                for coverage, count in coverage_counts.items():
                    logger.info(f"      {coverage}: {count}")

if __name__ == "__main__":
    test_specific_sku()