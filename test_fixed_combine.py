#!/usr/bin/env python3
"""
Test the fixed combine_search_results
"""

import numpy as np
import logging
from dual_index_data_loader import dual_index_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fixed_combine():
    # Initialize
    csv_path = "database_results/DB_ACTIVE.csv"
    if not dual_index_loader.initialize(csv_path):
        return
    
    # Test with a product we know exists in both indexes
    test_filename = "1034040KJ100"
    main_embedding = dual_index_loader.main_embeddings[126]
    measurement_embedding = dual_index_loader.get_measurement_embedding_by_filename(test_filename)
    
    # Search both indexes
    logger.info("🔍 Testing fixed combine_search_results...")
    main_distances, main_indices = dual_index_loader.search_main_index(main_embedding, 20)
    meas_distances, meas_indices = dual_index_loader.search_measurement_index(measurement_embedding, 20)
    
    logger.info(f"\nMain search results (first 5):")
    logger.info(f"  Indices: {main_indices[:5]}")
    logger.info(f"  Distances: {main_distances[:5]}")
    
    logger.info(f"\nMeasurement search results (first 5):")
    logger.info(f"  Indices: {meas_indices[:5]}")  
    logger.info(f"  Distances: {meas_distances[:5]}")
    
    # Use the actual combine_search_results
    combined_results = dual_index_loader.combine_search_results(
        main_distances, main_indices,
        meas_distances, meas_indices,
        top_k=10
    )
    
    logger.info(f"\n📊 Combined results: {len(combined_results)} items")
    
    # Analyze distribution
    sources = {}
    for result in combined_results:
        source = result.get('score_source', 'unknown')
        sources[source] = sources.get(source, 0) + 1
    
    logger.info(f"\n📊 Distribution:")
    for source, count in sources.items():
        logger.info(f"   {source}: {count}")
    
    # Show top 5 results
    logger.info(f"\n🏆 Top 5 results:")
    for i, result in enumerate(combined_results[:5]):
        logger.info(f"\n{i+1}. {result.get('filename_root', 'N/A')}")
        logger.info(f"   Source: {result.get('score_source', 'N/A')}")
        logger.info(f"   Combined score: {result.get('similarity_score', 0):.4f}")
        logger.info(f"   Main sim: {result.get('main_similarity', 0):.4f}")
        logger.info(f"   Meas sim: {result.get('measurement_similarity', 0):.4f}")

if __name__ == "__main__":
    test_fixed_combine()