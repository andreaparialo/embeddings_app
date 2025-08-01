#!/usr/bin/env python3
"""
Detailed test of combine_search_results to find where both_indexes products disappear
"""

import numpy as np
import logging
from dual_index_data_loader import dual_index_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_combine_detailed():
    # Initialize
    csv_path = "database_results/DB_ACTIVE.csv"
    if not dual_index_loader.initialize(csv_path):
        return
    
    # Use known test data
    test_filename = "1034040KJ100"
    
    # Get embeddings
    main_embedding = dual_index_loader.main_embeddings[126]  # We know this from previous test
    measurement_embedding = dual_index_loader.get_measurement_embedding_by_filename(test_filename)
    
    # Search with small top_k to debug
    logger.info("🔍 Searching with top_k=5...")
    main_distances, main_indices = dual_index_loader.search_main_index(main_embedding, 5)
    meas_distances, meas_indices = dual_index_loader.search_measurement_index(measurement_embedding, 5)
    
    logger.info(f"\nMain indices: {main_indices}")
    logger.info(f"Meas indices: {meas_indices}")
    
    # Now let's manually trace through combine_search_results
    logger.info("\n📊 Manual trace of combine_search_results logic:")
    
    # Normalize distances
    main_similarities = 1.0 / (1.0 + main_distances)
    measurement_similarities = 1.0 / (1.0 + meas_distances)
    
    # Normalize to 0-1 range
    if len(main_similarities) > 1:
        main_similarities = (main_similarities - np.min(main_similarities)) / (np.max(main_similarities) - np.min(main_similarities) + 1e-8)
    if len(measurement_similarities) > 1:
        measurement_similarities = (measurement_similarities - np.min(measurement_similarities)) / (np.max(measurement_similarities) - np.min(measurement_similarities) + 1e-8)
    
    logger.info(f"\nNormalized similarities:")
    logger.info(f"Main: {main_similarities}")
    logger.info(f"Meas: {measurement_similarities}")
    
    # Create combined results dict
    combined_results = {}
    
    # Add main results
    for i, (main_idx, main_sim) in enumerate(zip(main_indices, main_similarities)):
        if main_idx < len(dual_index_loader.main_metadata["image_paths"]):
            image_path = dual_index_loader.main_metadata["image_paths"][main_idx]
            combined_results[image_path] = {
                'main_similarity': float(main_sim),
                'main_rank': i + 1,
                'measurement_similarity': 0.0,
                'measurement_rank': None,
                'combined_score': float(main_sim * 0.7 + 0.5 * 0.3),  # Default meas sim = 0.5
                'source': 'main_only'
            }
            logger.info(f"\nAdded main: {image_path}")
            logger.info(f"  Combined score: {combined_results[image_path]['combined_score']:.4f}")
    
    # Add measurement results
    for i, (meas_idx, meas_sim) in enumerate(zip(meas_indices, measurement_similarities)):
        if meas_idx in dual_index_loader.measurement_path_mapping:
            normalized_path = dual_index_loader.measurement_path_mapping[meas_idx]['normalized']
            
            if normalized_path in combined_results:
                # Update existing entry
                logger.info(f"\n🎯 FOUND IN BOTH: {normalized_path}")
                logger.info(f"  Before: {combined_results[normalized_path]}")
                
                combined_results[normalized_path]['measurement_similarity'] = float(meas_sim)
                combined_results[normalized_path]['measurement_rank'] = i + 1
                combined_results[normalized_path]['combined_score'] = float(
                    combined_results[normalized_path]['main_similarity'] * 0.7 + meas_sim * 0.3
                )
                combined_results[normalized_path]['source'] = 'both_indexes'
                
                logger.info(f"  After: {combined_results[normalized_path]}")
    
    # Sort by combined score
    logger.info("\n📊 All results before sorting:")
    for path, scores in combined_results.items():
        logger.info(f"{path}: score={scores['combined_score']:.4f}, source={scores['source']}")
    
    sorted_results = sorted(combined_results.items(), 
                          key=lambda x: x[1]['combined_score'], 
                          reverse=True)[:5]
    
    logger.info("\n📊 Results after sorting (top 5):")
    for path, scores in sorted_results:
        logger.info(f"{path}: score={scores['combined_score']:.4f}, source={scores['source']}")
    
    # Now check the final conversion
    logger.info("\n🔍 Checking final conversion...")
    final_results = []
    for image_path, scores in sorted_results:
        filename_root = image_path.split('/')[-1].replace('.jpg', '')
        matching_products = dual_index_loader.df[dual_index_loader.df['filename_root'] == filename_root]
        
        logger.info(f"\nChecking {filename_root}:")
        logger.info(f"  Found {len(matching_products)} products in dataframe")
        logger.info(f"  Source: {scores['source']}")
        
        if not matching_products.empty:
            product_info = matching_products.iloc[0].to_dict()
            product_info.update({
                'score_source': scores['source'],
                'similarity_score': scores['combined_score']
            })
            final_results.append(product_info)
    
    # Summary
    sources = {}
    for result in final_results:
        source = result.get('score_source', 'unknown')
        sources[source] = sources.get(source, 0) + 1
    
    logger.info(f"\n📊 Final distribution: {sources}")

if __name__ == "__main__":
    test_combine_detailed()