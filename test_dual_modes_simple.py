#!/usr/bin/env python3
"""
Simple test for dual index search modes
"""

import numpy as np
from dual_index_data_loader import dual_index_loader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_search_modes():
    """Test both global and filtered search modes"""
    
    # Initialize dual index loader
    logger.info("🚀 Initializing dual index system...")
    csv_path = "database_results/DB_ACTIVE.csv"
    
    if not dual_index_loader.initialize(csv_path):
        logger.error("Failed to initialize dual index loader")
        return
    
    # Get a test embedding from the first product
    test_embedding = dual_index_loader.main_embeddings[0]
    logger.info(f"✅ Using test embedding with shape: {test_embedding.shape}")
    
    # Test filters
    test_filters = {
        'USERGENDER_DES': 'MAN',
        'FRONT_HEIGHT_VAL': 45.0  # This will use range filtering
    }
    
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST 1: GLOBAL SEARCH (no pre-filtering)")
    logger.info("="*60)
    
    # Mode 1: Search without filters
    main_distances_global, main_indices_global = dual_index_loader.search_main_index(
        test_embedding, top_k=20, filters=None
    )
    
    logger.info(f"✅ Found {len(main_indices_global)} results")
    logger.info(f"Top 5 indices: {main_indices_global[:5]}")
    logger.info(f"Top 5 distances: {main_distances_global[:5]}")
    
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST 2: FILTERED SEARCH (with pre-filtering)")
    logger.info("="*60)
    
    # Mode 2: Search with filters
    main_distances_filtered, main_indices_filtered = dual_index_loader.search_main_index(
        test_embedding, top_k=20, filters=test_filters
    )
    
    logger.info(f"✅ Found {len(main_indices_filtered)} results")
    logger.info(f"Top 5 indices: {main_indices_filtered[:5]}")
    logger.info(f"Top 5 distances: {main_distances_filtered[:5]}")
    
    # Compare results
    logger.info("\n" + "="*60)
    logger.info("📊 COMPARISON")
    logger.info("="*60)
    
    # Check how many of the filtered results appear in global top 20
    filtered_set = set(main_indices_filtered)
    global_set = set(main_indices_global)
    overlap = filtered_set & global_set
    
    logger.info(f"Overlap between modes: {len(overlap)} products")
    logger.info(f"Unique to global mode: {len(global_set - filtered_set)} products")
    logger.info(f"Unique to filtered mode: {len(filtered_set - global_set)} products")
    
    # Test measurement index search too
    if dual_index_loader.measurement_embeddings is not None:
        logger.info("\n" + "="*60)
        logger.info("🧪 MEASUREMENT INDEX TEST")
        logger.info("="*60)
        
        # Get first measurement embedding
        meas_embedding = dual_index_loader.measurement_embeddings[0]
        meas_distances, meas_indices = dual_index_loader.search_measurement_index(
            meas_embedding, top_k=10
        )
        
        logger.info(f"✅ Measurement search found {len(meas_indices)} results")
        logger.info(f"Top 5 indices: {meas_indices[:5]}")

if __name__ == "__main__":
    test_search_modes()