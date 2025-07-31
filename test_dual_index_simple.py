#!/usr/bin/env python3
"""
Simple test to verify dual index system is working correctly
"""

import logging
import numpy as np
from dual_index_data_loader import dual_index_loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🧪 Testing Dual Index System")
    
    # Initialize dual index
    csv_path = "/lambda/nfs/SPEEDINGTHEPROCESS/old_app/database_results/DB_ACTIVE.csv"
    index_dir = "/lambda/nfs/SPEEDINGTHEPROCESS/old_app/indexes"
    
    if not dual_index_loader.initialize(csv_path, index_dir):
        logger.error("❌ Failed to initialize dual index")
        return
    
    logger.info("✅ Dual index initialized successfully!")
    
    # Check embeddings
    if dual_index_loader.measurement_embeddings is None:
        logger.error("❌ Measurement embeddings not loaded!")
        return
    
    logger.info(f"📊 Measurement embeddings loaded: {dual_index_loader.measurement_embeddings.shape}")
    logger.info(f"📊 Filename mappings: {len(dual_index_loader.measurement_filename_to_embedding)} products")
    
    # Test retrieval
    test_filenames = ["10003502M200", "1000350FG400", "1000350KY200", "NOTEXIST"]
    
    for filename in test_filenames:
        embedding = dual_index_loader.get_measurement_embedding_by_filename(filename)
        if embedding is not None:
            logger.info(f"✅ {filename}: Found embedding (norm={np.linalg.norm(embedding):.4f})")
        else:
            logger.info(f"❌ {filename}: No embedding found")
    
    # Test search with weights
    logger.info("\n🔍 Testing weighted search...")
    
    # Get a sample filename that exists in both indexes
    # Find a product that exists in both indexes
    sample_filename = None
    for filename in ["10003502M200", "1000350FG400", "1000350KY200"]:
        if dual_index_loader.get_measurement_embedding_by_filename(filename) is not None:
            sample_filename = filename
            break
    
    if not sample_filename:
        logger.error("Could not find a sample that exists in measurement index")
        return
    
    # Get embeddings from data loader
    if sample_filename not in dual_index_loader.main_data_loader.filename_to_idx:
        logger.error(f"Sample {sample_filename} not found in main index")
        return
        
    sample_idx = dual_index_loader.main_data_loader.filename_to_idx[sample_filename]
    
    # Generate a random query embedding for testing (since we can't access the actual embeddings)
    # In real usage, this would come from the GME model
    gme_embedding = np.random.randn(3584).astype(np.float32)
    gme_embedding = gme_embedding / np.linalg.norm(gme_embedding)
    
    logger.info(f"Using sample: {sample_filename}")
    
    # Get measurement embedding
    meas_embedding = dual_index_loader.get_measurement_embedding_by_filename(sample_filename)
    
    # Test different weights
    weight_tests = [
        (0.7, 0.3, "Default"),
        (1.0, 0.0, "GME Only"),
        (0.0, 1.0, "Technical Only"),
        (0.5, 0.5, "Equal")
    ]
    
    for main_w, meas_w, name in weight_tests:
        logger.info(f"\n📊 Testing {name} weights: GME={main_w}, Technical={meas_w}")
        dual_index_loader.set_scoring_weights(main_w, meas_w)
        
        # Search
        main_dist, main_idx = dual_index_loader.search_main_index(gme_embedding, 5)
        
        if meas_embedding is not None:
            meas_dist, meas_idx = dual_index_loader.search_measurement_index(meas_embedding, 5)
        else:
            meas_dist = np.array([])
            meas_idx = np.array([])
        
        # Combine
        results = dual_index_loader.combine_search_results(
            main_dist, main_idx, meas_dist, meas_idx, 3
        )
        
        logger.info(f"Top 3 results:")
        for i, r in enumerate(results):
            logger.info(f"  {i+1}. {r['filename_root']}: "
                       f"GME={r['main_similarity']:.3f}, "
                       f"Tech={r['measurement_similarity']:.3f}, "
                       f"Final={r['similarity_score']:.3f}, "
                       f"Source={r['score_source']}")

if __name__ == "__main__":
    main()