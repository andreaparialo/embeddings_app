#!/usr/bin/env python3
"""
Debug why products aren't being classified as 'both_indexes'
"""

import numpy as np
import logging
from dual_index_data_loader import dual_index_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_both_indexes():
    # Initialize
    csv_path = "database_results/DB_ACTIVE.csv"
    if not dual_index_loader.initialize(csv_path):
        return
    
    # Test with the specific SKU
    test_filename = "20560805NCUC"  # The SKU from the test
    
    # Find in main index
    main_idx = None
    for i, path in enumerate(dual_index_loader.main_metadata['image_paths']):
        if test_filename in path:
            main_idx = i
            logger.info(f"✅ Found {test_filename} in main index at position {main_idx}")
            logger.info(f"   Path: {path}")
            break
    
    # Find in measurement index  
    meas_idx = None
    for idx, info in dual_index_loader.measurement_path_mapping.items():
        if test_filename in info['normalized']:
            meas_idx = idx
            logger.info(f"✅ Found {test_filename} in measurement index at position {meas_idx}")
            logger.info(f"   Path: {info['normalized']}")
            logger.info(f"   In main index: {info['in_main_index']}")
            break
    
    # Check the mapping
    if meas_idx in dual_index_loader.measurement_to_main_mapping:
        mapped_main_idx = dual_index_loader.measurement_to_main_mapping[meas_idx]
        logger.info(f"✅ Measurement index {meas_idx} maps to main index {mapped_main_idx}")
        logger.info(f"   Does it match our main_idx? {mapped_main_idx == main_idx}")
    
    # Now let's do a search and see what happens
    logger.info("\n🔍 Testing search...")
    
    main_embedding = dual_index_loader.main_embeddings[main_idx]
    measurement_embedding = dual_index_loader.get_measurement_embedding_by_filename(test_filename)
    
    # Search main
    main_distances, main_indices = dual_index_loader.search_main_index(main_embedding, 10)
    logger.info(f"\nMain search top 5: {main_indices[:5]}")
    
    # Search measurement
    meas_distances, meas_indices = dual_index_loader.search_measurement_index(measurement_embedding, 10)
    logger.info(f"Measurement search top 5: {meas_indices[:5]}")
    
    # The key question: when we search with these embeddings, do the results overlap?
    logger.info("\n🔄 Checking if search results overlap...")
    
    # Convert measurement indices to main indices
    main_from_meas = []
    for m_idx in meas_indices[:10]:
        if m_idx in dual_index_loader.measurement_to_main_mapping:
            main_from_meas.append(dual_index_loader.measurement_to_main_mapping[m_idx])
    
    logger.info(f"Measurement indices converted to main indices: {main_from_meas[:5]}")
    
    # Check overlap
    main_set = set(main_indices)
    meas_converted_set = set(main_from_meas)
    overlap = main_set & meas_converted_set
    
    logger.info(f"\n📊 Overlap analysis:")
    logger.info(f"   Main results: {len(main_set)}")
    logger.info(f"   Measurement results (that map to main): {len(meas_converted_set)}")
    logger.info(f"   Overlap: {len(overlap)} products")
    logger.info(f"   Overlapping indices: {overlap}")
    
    # Now test combine_search_results
    logger.info("\n🔄 Testing combine_search_results...")
    combined = dual_index_loader.combine_search_results(
        main_distances, main_indices,
        meas_distances, meas_indices,
        top_k=10
    )
    
    # Check the distribution
    sources = {}
    for result in combined:
        source = result.get('score_source', 'unknown')
        sources[source] = sources.get(source, 0) + 1
        if source == 'both_indexes':
            logger.info(f"   Both indexes product: {result.get('filename_root')}")
    
    logger.info(f"\n📊 Final distribution: {sources}")

if __name__ == "__main__":
    debug_both_indexes()