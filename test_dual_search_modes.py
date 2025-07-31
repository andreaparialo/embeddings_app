#!/usr/bin/env python3
"""
Test dual index search with both search modes
"""

import numpy as np
from dual_index_data_loader import dual_index_loader
from batch_processor_optimized import OptimizedBatchProcessor
from data_loader import DataLoader
from search_engine import HybridSearchEngine
from gme_model import GMEModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_search_modes():
    """Test both global and filtered search modes"""
    
    # Initialize system
    logger.info("🚀 Initializing search system...")
    csv_path = "database_results/DB_ACTIVE.csv"
    
    # Initialize search engine (needed for batch processor)
    search_engine = HybridSearchEngine()
    if not search_engine.initialize(csv_path):
        logger.error("Failed to initialize search engine")
        return
    
    # Initialize dual index loader
    if not dual_index_loader.initialize(csv_path):
        logger.error("Failed to initialize dual index loader")
        return
    
    # Create batch processor
    batch_processor = OptimizedBatchProcessor(
        search_engine, 
        search_engine.data_loader, 
        search_engine.gme_model
    )
    
    # Test data - use a known filename
    test_filename = "10003502M200"
    test_groups = {
        test_filename: {
            'source_item': {
                'SKU_COD': '10003502M200',
                'filename_root': test_filename,
                'USERGENDER_DES': 'MAN',
                'FRONT_HEIGHT_VAL': 153.0,
                'LACE_WIDTH_VAL': 6.5
            },
            'skus': ['10003502M200']
        }
    }
    
    # Test filters
    matching_cols = ['USERGENDER_DES', 'FRONT_HEIGHT_VAL', 'LACE_WIDTH_VAL']
    
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST 1: GLOBAL SEARCH MODE")
    logger.info("="*60)
    
    # Test Mode 1: Global search (search all, then filter)
    results_global = batch_processor.process_image_groups_with_prefilter(
        test_groups,
        matching_cols,
        max_results_per_sku=10,
        dual_engine_enabled=True,
        main_weight=0.7,
        measurement_weight=0.3,
        search_mode="global"
    )
    
    logger.info(f"✅ Global mode found {len(results_global)} results")
    if results_global:
        # Show first few results
        for i, result in enumerate(results_global[:3]):
            logger.info(f"  Result {i+1}: SKU={result.get('Similar_SKU')}, Score={result.get('Final_Score', 0):.3f}, Coverage={result.get('Index_Coverage')}")
    
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST 2: FILTERED SEARCH MODE")
    logger.info("="*60)
    
    # Test Mode 2: Filtered search (filter first, then search)
    results_filtered = batch_processor.process_image_groups_with_prefilter(
        test_groups,
        matching_cols,
        max_results_per_sku=10,
        dual_engine_enabled=True,
        main_weight=0.7,
        measurement_weight=0.3,
        search_mode="filtered"
    )
    
    logger.info(f"✅ Filtered mode found {len(results_filtered)} results")
    if results_filtered:
        # Show first few results
        for i, result in enumerate(results_filtered[:3]):
            logger.info(f"  Result {i+1}: SKU={result.get('Similar_SKU')}, Score={result.get('Final_Score', 0):.3f}, Coverage={result.get('Index_Coverage')}")
    
    # Compare results
    logger.info("\n" + "="*60)
    logger.info("📊 COMPARISON")
    logger.info("="*60)
    logger.info(f"Global mode: {len(results_global)} results")
    logger.info(f"Filtered mode: {len(results_filtered)} results")
    
    # Check if top results are different
    if results_global and results_filtered:
        global_top_skus = [r['Similar_SKU'] for r in results_global[:5]]
        filtered_top_skus = [r['Similar_SKU'] for r in results_filtered[:5]]
        
        logger.info(f"\nTop 5 SKUs (Global): {global_top_skus}")
        logger.info(f"Top 5 SKUs (Filtered): {filtered_top_skus}")
        
        # Check overlap
        overlap = set(global_top_skus) & set(filtered_top_skus)
        logger.info(f"\nOverlap in top 5: {len(overlap)} SKUs")

if __name__ == "__main__":
    test_search_modes()