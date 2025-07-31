#!/usr/bin/env python3
"""
Test Dual Batch Search with Different Weight Combinations
Verifies that the true weighted scoring is working correctly
"""

import json
import logging
import numpy as np
from dual_index_data_loader import dual_index_loader
from batch_processor_optimized import OptimizedBatchProcessor
from data_loader import DataLoader
from gme_model import GMEModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_weight_combinations():
    """Test different weight combinations to verify implementation"""
    
    logger.info("🧪 Testing Dual Batch Search with Different Weight Combinations")
    
    # Initialize components
    logger.info("📊 Initializing components...")
    
    # Load data
    csv_path = "database_results/DB_ACTIVE.csv"
    
    # Initialize dual index loader
    if not dual_index_loader.initialize(csv_path):
        logger.error("❌ Failed to initialize dual index loader")
        return
    
    # Check if embeddings are loaded
    if dual_index_loader.measurement_embeddings is None:
        logger.error("❌ Measurement embeddings not loaded!")
        return
    
    logger.info(f"✅ Dual index initialized successfully")
    logger.info(f"   GME embeddings: {dual_index_loader.main_index.ntotal}")
    logger.info(f"   Measurement embeddings: {len(dual_index_loader.measurement_embeddings)}")
    
    # Test different weight combinations
    test_cases = [
        {"name": "Default (70/30)", "main": 0.7, "meas": 0.3},
        {"name": "GME Only (100/0)", "main": 1.0, "meas": 0.0},
        {"name": "Technical Only (0/100)", "main": 0.0, "meas": 1.0},
        {"name": "Equal Weight (50/50)", "main": 0.5, "meas": 0.5},
        {"name": "GME Heavy (90/10)", "main": 0.9, "meas": 0.1},
        {"name": "Technical Heavy (10/90)", "main": 0.1, "meas": 0.9},
    ]
    
    # Test with a sample filename that exists in both indexes
    test_filename = "10003502M200"  # Should exist in the dataset
    
    # Get embeddings for test
    logger.info(f"\n🔍 Testing with filename: {test_filename}")
    
    # Get GME embedding
    gme_embedding = None
    for idx, path in enumerate(dual_index_loader.main_metadata["image_paths"]):
        if test_filename in path:
            gme_embedding = dual_index_loader.main_embeddings[idx]
            logger.info(f"✅ Found GME embedding at index {idx}")
            break
    
    if gme_embedding is None:
        logger.error(f"❌ Could not find GME embedding for {test_filename}")
        return
    
    # Get measurement embedding
    meas_embedding = dual_index_loader.get_measurement_embedding_by_filename(test_filename)
    if meas_embedding is not None:
        logger.info(f"✅ Found measurement embedding")
    else:
        logger.info(f"⚠️ No measurement embedding found for {test_filename}")
    
    # Test each weight combination
    for test_case in test_cases:
        logger.info(f"\n📊 Testing: {test_case['name']}")
        
        # Set weights
        dual_index_loader.set_scoring_weights(test_case['main'], test_case['meas'])
        
        # Search in both indexes
        main_distances, main_indices = dual_index_loader.search_main_index(
            gme_embedding, top_k=10
        )
        
        if meas_embedding is not None:
            meas_distances, meas_indices = dual_index_loader.search_measurement_index(
                meas_embedding, top_k=10
            )
        else:
            meas_distances = np.array([])
            meas_indices = np.array([])
        
        # Combine results
        combined = dual_index_loader.combine_search_results(
            main_distances, main_indices,
            meas_distances, meas_indices,
            top_k=5
        )
        
        # Display results
        logger.info(f"Top 5 results:")
        for i, result in enumerate(combined[:5]):
            logger.info(f"  {i+1}. {result.get('filename_root', 'N/A')}")
            logger.info(f"     - GME Score: {result.get('main_similarity', 0):.3f}")
            logger.info(f"     - Technical Score: {result.get('measurement_similarity', 0):.3f}")
            logger.info(f"     - Combined Score: {result.get('similarity_score', 0):.3f}")
            logger.info(f"     - Source: {result.get('score_source', 'unknown')}")
            
            # Verify formula
            expected = (result.get('main_similarity', 0) * test_case['main'] + 
                       result.get('measurement_similarity', 0) * test_case['meas'])
            actual = result.get('similarity_score', 0)
            
            if abs(expected - actual) < 0.001:
                logger.info(f"     ✅ Formula verified: {expected:.3f} ≈ {actual:.3f}")
            else:
                logger.error(f"     ❌ Formula mismatch: expected {expected:.3f}, got {actual:.3f}")

def test_batch_processor_integration():
    """Test the batch processor with weights"""
    logger.info("\n\n🧪 Testing Batch Processor Integration")
    
    # Initialize components
    data_loader = DataLoader()
    data_loader.load_csv("database_results/DB_ACTIVE.csv")
    data_loader.load_faiss_index("indexes", "1095")
    
    gme_model = GMEModel()
    # Note: We don't need to load the model for filename-based search
    
    from search_engine import HybridSearchEngine
    search_engine = HybridSearchEngine(data_loader, gme_model)
    
    batch_processor = OptimizedBatchProcessor(search_engine, data_loader, gme_model)
    
    # Create test image groups
    test_groups = {
        "10003502M200": {
            "source_item": {
                "SKU_COD": "TEST123",
                "filename_root": "10003502M200",
                "MACROCATEGORY_DES": "SHOES",
                "USERGENDER_DES": "MAN"
            },
            "skus": ["TEST123"],
            "embedding": data_loader.embeddings[0]  # Use first embedding as test
        }
    }
    
    # Test with different weights
    logger.info("\n📊 Testing batch processor with GME=80%, Technical=20%")
    results = batch_processor.process_image_groups_with_prefilter(
        test_groups,
        matching_cols=["MACROCATEGORY_DES", "USERGENDER_DES"],
        max_results_per_sku=5,
        dual_engine_enabled=True,
        main_weight=0.8,
        measurement_weight=0.2
    )
    
    logger.info(f"Found {len(results)} results")
    if results:
        first_result = results[0]
        if 'GME_Score' in first_result:
            logger.info("✅ Scoring details found in results:")
            logger.info(f"   GME Score: {first_result.get('GME_Score', 'N/A')}")
            logger.info(f"   Technical Score: {first_result.get('Technical_Score', 'N/A')}")
            logger.info(f"   Final Score: {first_result.get('Final_Score', 'N/A')}")
            logger.info(f"   Formula: {first_result.get('Score_Formula', 'N/A')}")
            logger.info(f"   Coverage: {first_result.get('Index_Coverage', 'N/A')}")
        else:
            logger.warning("⚠️ No scoring details in results - dual engine may not be working")

def main():
    """Main test function"""
    try:
        test_weight_combinations()
        test_batch_processor_integration()
        logger.info("\n\n✅ All tests completed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()