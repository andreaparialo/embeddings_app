#!/usr/bin/env python3
"""
Debug the index structure differences between main and measurement indexes
"""

import numpy as np
import json
import logging
from dual_index_data_loader import dual_index_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_index_structure():
    # Initialize
    csv_path = "database_results/DB_ACTIVE.csv"
    if not dual_index_loader.initialize(csv_path):
        return
    
    logger.info("\n🔍 MAIN INDEX STRUCTURE:")
    logger.info(f"   Total vectors: {dual_index_loader.main_index.ntotal}")
    logger.info(f"   Embeddings shape: {dual_index_loader.main_embeddings.shape}")
    logger.info(f"   Metadata image_paths: {len(dual_index_loader.main_metadata['image_paths'])}")
    
    # Check how main index search returns indices
    test_embedding = dual_index_loader.main_embeddings[0]
    distances, indices = dual_index_loader.main_index.search(test_embedding.reshape(1, -1), 5)
    logger.info(f"\n   Test search indices: {indices[0]}")
    logger.info(f"   These map to paths:")
    for idx in indices[0][:3]:
        if idx < len(dual_index_loader.main_metadata['image_paths']):
            logger.info(f"      {idx} -> {dual_index_loader.main_metadata['image_paths'][idx]}")
    
    logger.info("\n📏 MEASUREMENT INDEX STRUCTURE:")
    logger.info(f"   Total vectors: {dual_index_loader.measurement_index.ntotal}")
    if dual_index_loader.measurement_embeddings is not None:
        logger.info(f"   Embeddings shape: {dual_index_loader.measurement_embeddings.shape}")
    logger.info(f"   Path mapping entries: {len(dual_index_loader.measurement_path_mapping)}")
    
    # Check measurement metadata structure
    with open("indexes/index_measurements/corrected_metadata.json", 'r') as f:
        meas_meta = json.load(f)
    
    logger.info(f"\n   Product mapping sample:")
    for i, (idx_str, path) in enumerate(list(meas_meta['product_mapping'].items())[:3]):
        logger.info(f"      '{idx_str}' -> {path}")
    
    # Check how measurement index search returns indices
    if dual_index_loader.measurement_embeddings is not None:
        test_meas_embedding = dual_index_loader.measurement_embeddings[0]
        meas_distances, meas_indices = dual_index_loader.measurement_index.search(test_meas_embedding.reshape(1, -1), 5)
        logger.info(f"\n   Test search indices: {meas_indices[0]}")
        logger.info(f"   These map to paths:")
        for idx in meas_indices[0][:3]:
            if idx in dual_index_loader.measurement_path_mapping:
                info = dual_index_loader.measurement_path_mapping[idx]
                logger.info(f"      {idx} -> {info['normalized']}")
    
    # Check overlap
    logger.info(f"\n🔄 OVERLAP ANALYSIS:")
    logger.info(f"   measurement_to_main_mapping entries: {len(dual_index_loader.measurement_to_main_mapping)}")
    
    # Sample some mappings
    sample_mappings = list(dual_index_loader.measurement_to_main_mapping.items())[:5]
    logger.info(f"\n   Sample mappings (measurement_idx -> main_idx):")
    for meas_idx, main_idx in sample_mappings:
        meas_path = dual_index_loader.measurement_path_mapping[meas_idx]['normalized']
        main_path = dual_index_loader.main_metadata['image_paths'][main_idx]
        logger.info(f"      Meas {meas_idx} ({meas_path}) -> Main {main_idx} ({main_path})")
        logger.info(f"         Match: {meas_path == main_path}")

if __name__ == "__main__":
    debug_index_structure()