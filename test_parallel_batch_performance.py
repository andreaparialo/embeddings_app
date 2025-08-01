#!/usr/bin/env python3
"""
Test script to benchmark parallel batch processing performance
Compares standard vs parallel filtering approaches
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import required modules
from search_engine import HybridSearchEngine
from data_loader import DataLoader
from gme_model import GMEModel
from batch_processor_optimized import OptimizedBatchProcessor
from batch_processor_parallel import ParallelBatchProcessor

def create_test_data(num_skus: int = 100) -> pd.DataFrame:
    """Create test SKU data with various filter attributes"""
    # Create realistic test data based on the filters shown in user's log
    brands = ["PRIVE' REVAUX", "RAY-BAN", "OAKLEY", "VERSACE", "GUCCI"]
    colors = ["CR", "BK", "BR", "BL", "GR"]
    genders = ["WOMAN", "MAN", "UNISEX ADULT"]
    materials = ["INJECTED", "METAL", "ACETATE", "TITANIUM"]
    shapes = ["CAT_EYE", "AVIATOR", "SQUARE", "ROUND", "RECTANGLE"]
    
    data = []
    for i in range(num_skus):
        sku = f"TEST{i:06d}"
        data.append({
            'SKU_COD': sku,
            'filename_root': f"test_image_{i % 20}",  # 20 unique images
            'BRAND_DES': np.random.choice(brands),
            'PRODUCT_TYPE_COD': '1',
            'COLOR_FAMILY_1_DES': np.random.choice(colors),
            'USERGENDER_DES': np.random.choice(genders),
            'CTM_FIRST_FRONT_MATERIAL_DES': np.random.choice(materials),
            'CTM_FIRST_TEMPLE_MATERIAL_DES': np.random.choice(materials),
            'SHAPE_SEMI_GROUPED': np.random.choice(shapes),
            'RIM_TYPE_DES': 'FULL RIM',
            'MD_SKU_STATUS_COD': 'IL',
            'STARTSKU_DATE': '2024-01-01',
            'ACT_SKU_PRICE_VAL': np.random.uniform(50, 500),
            'FRONT_HEIGHT_VAL': np.random.uniform(40, 60)
        })
    
    return pd.DataFrame(data)

def benchmark_batch_processing(processor_type: str, processor, test_groups: Dict, 
                             matching_cols: List[str]) -> float:
    """Benchmark a batch processor"""
    logger.info(f"\n{'='*60}")
    logger.info(f"🧪 Testing {processor_type} Batch Processor")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        results = processor.process_image_groups_with_prefilter(
            image_groups=test_groups,
            matching_cols=matching_cols,
            max_results_per_sku=50,
            exclude_same_model=True,
            allowed_statuses=['IL'],
            group_unisex=False,
            dual_engine_enabled=True,
            batch_size=16,
            main_weight=0.7,
            measurement_weight=0.3,
            search_mode="filtered"
        )
        
        elapsed = time.time() - start_time
        
        logger.info(f"✅ {processor_type} completed in {elapsed:.2f} seconds")
        logger.info(f"📊 Results count: {len(results)}")
        logger.info(f"⚡ Performance: {len(test_groups)/elapsed:.1f} images/sec")
        
        return elapsed
        
    except Exception as e:
        logger.error(f"❌ Error in {processor_type}: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')

def main():
    """Main test function"""
    logger.info("🚀 Starting Parallel Batch Processing Performance Test")
    
    # Initialize components
    logger.info("📦 Loading models and data...")
    
    try:
        # Initialize search engine
        search_engine = HybridSearchEngine()
        if not search_engine.initialize("database_results/DB_ACTIVE.csv", "indexes", "1095"):
            logger.error("Failed to initialize search engine")
            return
            
        # Get references to data_loader and gme_model from search engine
        data_loader = search_engine.data_loader
        gme_model = search_engine.gme_model
        df = data_loader.df
        
        logger.info(f"✅ Loaded {len(df)} products")
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return
    
    # Create test data
    num_test_skus = 100
    test_df = create_test_data(num_test_skus)
    
    # Create image groups with strong filters (similar to user's example)
    test_groups = {}
    matching_cols = [
        'BRAND_DES', 'PRODUCT_TYPE_COD', 'COLOR_FAMILY_1_DES',
        'USERGENDER_DES', 'CTM_FIRST_FRONT_MATERIAL_DES',
        'CTM_FIRST_TEMPLE_MATERIAL_DES', 'SHAPE_SEMI_GROUPED',
        'RIM_TYPE_DES'
    ]
    
    # Group by filename_root and create test groups
    for filename_root, group in test_df.groupby('filename_root'):
        source_item = group.iloc[0].to_dict()
        test_groups[filename_root] = {
            'source_item': source_item,
            'skus': group['SKU_COD'].tolist()
        }
    
    logger.info(f"\n📊 Test Configuration:")
    logger.info(f"  - Total SKUs: {num_test_skus}")
    logger.info(f"  - Unique images: {len(test_groups)}")
    logger.info(f"  - Matching columns: {len(matching_cols)}")
    logger.info(f"  - Filter selectivity: Very High (similar to user's case)")
    
    # Test standard optimized processor
    optimized_proc = OptimizedBatchProcessor(search_engine, data_loader, gme_model)
    optimized_time = benchmark_batch_processing("OPTIMIZED", optimized_proc, test_groups, matching_cols)
    
    # Clear cache between tests
    optimized_proc.clear_filter_cache()
    time.sleep(2)  # Give system time to clean up
    
    # Test parallel processor
    parallel_proc = ParallelBatchProcessor(search_engine, data_loader, gme_model)
    parallel_time = benchmark_batch_processing("PARALLEL", parallel_proc, test_groups, matching_cols)
    
    # Calculate improvement
    if optimized_time > 0 and parallel_time > 0:
        speedup = optimized_time / parallel_time
        improvement_pct = ((optimized_time - parallel_time) / optimized_time) * 100
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 PERFORMANCE COMPARISON")
        logger.info(f"{'='*60}")
        logger.info(f"Standard Optimized: {optimized_time:.2f}s")
        logger.info(f"Parallel Processing: {parallel_time:.2f}s")
        logger.info(f"🚀 Speedup: {speedup:.2f}x")
        logger.info(f"✨ Improvement: {improvement_pct:.1f}%")
        logger.info(f"{'='*60}")
    
    logger.info("\n✅ Test completed!")

if __name__ == "__main__":
    main()