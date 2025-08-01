#!/usr/bin/env python3
"""
Debug why measurement results don't match with main index paths
"""

import logging
from dual_index_data_loader import dual_index_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_path_mismatch():
    # Initialize
    csv_path = "database_results/DB_ACTIVE.csv"
    if not dual_index_loader.initialize(csv_path):
        return
    
    # Check the first measurement result from our test
    test_meas_idx = 2983  # This should be 1034040KJ100.jpg
    
    if test_meas_idx in dual_index_loader.measurement_path_mapping:
        meas_info = dual_index_loader.measurement_path_mapping[test_meas_idx]
        normalized_path = meas_info['normalized']
        logger.info(f"\n📊 Measurement index {test_meas_idx}:")
        logger.info(f"   Normalized path: {normalized_path}")
        logger.info(f"   In main index: {meas_info['in_main_index']}")
        
        # Check if this path exists in main_metadata
        if normalized_path in dual_index_loader.main_metadata['image_paths']:
            main_idx = dual_index_loader.main_metadata['image_paths'].index(normalized_path)
            logger.info(f"   ✅ Found in main metadata at index: {main_idx}")
        else:
            logger.info(f"   ❌ NOT found in main metadata image_paths")
            
            # Check for similar paths
            filename_root = normalized_path.split('/')[-1].replace('.jpg', '')
            logger.info(f"\n🔍 Searching for similar paths with root: {filename_root}")
            
            found_similar = False
            for i, path in enumerate(dual_index_loader.main_metadata['image_paths'][:100]):
                if filename_root in path:
                    logger.info(f"   Found similar: index {i} -> {path}")
                    found_similar = True
            
            if not found_similar:
                logger.info("   No similar paths found in first 100 entries")
        
        # Check the measurement_to_main_mapping
        if test_meas_idx in dual_index_loader.measurement_to_main_mapping:
            main_idx = dual_index_loader.measurement_to_main_mapping[test_meas_idx]
            logger.info(f"\n✅ Measurement index {test_meas_idx} maps to main index {main_idx}")
            logger.info(f"   Main path: {dual_index_loader.main_metadata['image_paths'][main_idx]}")
        else:
            logger.info(f"\n❌ Measurement index {test_meas_idx} NOT in measurement_to_main_mapping")
    
    # Let's check what's in combined_results dict during combine_search_results
    logger.info("\n📊 Checking combined_results logic...")
    
    # Simulate what happens in combine_search_results
    main_indices = [126, 16265, 4446]  # First few from test
    meas_indices = [2983, 2981, 13853]  # First few from test
    
    combined_results = {}
    
    # Process main results
    for i, main_idx in enumerate(main_indices):
        if main_idx < len(dual_index_loader.main_metadata["image_paths"]):
            image_path = dual_index_loader.main_metadata["image_paths"][main_idx]
            combined_results[image_path] = {
                'main_similarity': 1.0,
                'main_rank': i + 1,
                'measurement_similarity': 0.0,
                'measurement_rank': None,
                'source': 'main_only'
            }
            logger.info(f"\nAdded main result: {image_path}")
    
    # Process measurement results
    logger.info("\n🔄 Processing measurement results...")
    for i, meas_idx in enumerate(meas_indices):
        if meas_idx in dual_index_loader.measurement_path_mapping:
            normalized_path = dual_index_loader.measurement_path_mapping[meas_idx]['normalized']
            logger.info(f"\nMeasurement {meas_idx} -> {normalized_path}")
            logger.info(f"   In combined_results? {normalized_path in combined_results}")
            
            if normalized_path not in combined_results:
                # Check if any key in combined_results contains the filename
                filename_root = normalized_path.split('/')[-1].replace('.jpg', '')
                for key in combined_results:
                    if filename_root in key:
                        logger.info(f"   Found similar key: {key}")

if __name__ == "__main__":
    debug_path_mismatch()