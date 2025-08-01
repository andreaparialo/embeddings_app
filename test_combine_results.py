#!/usr/bin/env python3
"""
Test the combine_search_results function directly
"""

import numpy as np
import logging
from dual_index_data_loader import dual_index_loader

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Set specific logger to DEBUG
logging.getLogger('dual_index_data_loader').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

def test_combine_results():
    """Test combine_search_results with known data"""
    
    # Initialize dual index
    csv_path = "database_results/DB_ACTIVE.csv"
    if not dual_index_loader.initialize(csv_path):
        logger.error("Failed to initialize dual index loader")
        return
    
    # Get a test product that exists in both indexes
    test_filename = "1034040KJ100"  # We know this exists from previous test
    
    # Get embeddings
    main_embedding = None
    measurement_embedding = dual_index_loader.get_measurement_embedding_by_filename(test_filename)
    
    if measurement_embedding is None:
        logger.error(f"No measurement embedding for {test_filename}")
        return
        
    # Find main embedding
    for idx, path in enumerate(dual_index_loader.main_metadata['image_paths']):
        if test_filename in path:
            main_embedding = dual_index_loader.main_embeddings[idx]
            logger.info(f"Found main embedding at index {idx}")
            break
    
    if main_embedding is None:
        logger.error(f"No main embedding for {test_filename}")
        return
    
    # Search both indexes
    logger.info("\n🔍 Searching main index...")
    main_distances, main_indices = dual_index_loader.search_main_index(main_embedding, 30)
    logger.info(f"Main results: {len(main_indices)} items")
    logger.info(f"First 5 indices: {main_indices[:5]}")
    logger.info(f"First 5 distances: {main_distances[:5]}")
    
    logger.info("\n🔍 Searching measurement index...")
    meas_distances, meas_indices = dual_index_loader.search_measurement_index(measurement_embedding, 30)
    logger.info(f"Measurement results: {len(meas_indices)} items")
    logger.info(f"First 5 indices: {meas_indices[:5]}")
    logger.info(f"First 5 distances: {meas_distances[:5]}")
    
    # Check if measurement indices are in path mapping
    logger.info("\n📊 Checking measurement indices in path mapping:")
    found_count = 0
    for i, idx in enumerate(meas_indices[:10]):
        if idx in dual_index_loader.measurement_path_mapping:
            info = dual_index_loader.measurement_path_mapping[idx]
            logger.info(f"   Index {idx}: ✅ Found - {info['normalized']}")
            found_count += 1
        else:
            logger.info(f"   Index {idx}: ❌ NOT FOUND in path mapping")
    
    logger.info(f"\n📊 Path mapping check: {found_count}/10 measurement indices found")
    
    # Now test combine_search_results
    logger.info("\n🔄 Testing combine_search_results...")
    combined_results = dual_index_loader.combine_search_results(
        main_distances, main_indices,
        meas_distances, meas_indices,
        top_k=20
    )
    
    logger.info(f"\n📊 Combined results: {len(combined_results)} items")
    
    # Analyze the results
    if combined_results:
        sources = {}
        for result in combined_results:
            source = result.get('score_source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        logger.info("\n📊 Result distribution:")
        for source, count in sources.items():
            logger.info(f"   {source}: {count}")
        
        # Show first few results
        logger.info("\n📋 First 3 combined results:")
        for i, result in enumerate(combined_results[:3]):
            logger.info(f"\n   Result {i+1}:")
            logger.info(f"     Image: {result.get('image_path', 'N/A')}")
            logger.info(f"     Source: {result.get('score_source', 'N/A')}")
            logger.info(f"     Main sim: {result.get('main_similarity', 0):.4f}")
            logger.info(f"     Meas sim: {result.get('measurement_similarity', 0):.4f}")
            logger.info(f"     Combined: {result.get('similarity_score', 0):.4f}")

if __name__ == "__main__":
    test_combine_results()