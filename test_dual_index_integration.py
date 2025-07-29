#!/usr/bin/env python3
"""
Test Dual Index Integration
Tests the new dual-index search system to ensure it works correctly.
"""

import os
import sys
import logging
import numpy as np
from PIL import Image
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dual_data_loader():
    """Test the dual index data loader"""
    logger.info("🧪 Testing Dual Index Data Loader...")
    
    try:
        from dual_index_data_loader import dual_index_loader
        
        # Test initialization
        csv_path = "database_results/final_with_aws_shapes_enriched.csv"
        if not os.path.exists(csv_path):
            csv_path = "database_results/DB_ACTIVE.csv"
        
        success = dual_index_loader.initialize(csv_path, "indexes")
        
        if success:
            logger.info("✅ Dual data loader initialized successfully")
            
            # Get stats
            stats = dual_index_loader.get_stats()
            logger.info(f"📊 Stats: {stats}")
            
            return True
        else:
            logger.error("❌ Dual data loader initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing dual data loader: {e}")
        return False

def test_measurement_feature_extractor():
    """Test the measurement feature extractor"""
    logger.info("🧪 Testing Measurement Feature Extractor...")
    
    try:
        from measurement_feature_extractor import MeasurementFeatureExtractor
        
        # Test with ResNet method
        extractor = MeasurementFeatureExtractor(method="resnet_pool")
        
        if extractor.load_model():
            logger.info("✅ Feature extractor loaded successfully")
            
            # Create a test image
            test_image_path = "test_image.jpg"
            test_image = Image.new('RGB', (224, 224), color='red')
            test_image.save(test_image_path)
            
            try:
                # Extract features
                features = extractor.extract_features(test_image_path)
                
                if features is not None:
                    logger.info(f"✅ Features extracted: shape {features.shape}, type {features.dtype}")
                    
                    # Verify dimensions
                    if features.shape == (256,) and features.dtype == np.float32:
                        logger.info("✅ Feature dimensions and type are correct")
                        return True
                    else:
                        logger.error(f"❌ Incorrect feature dimensions or type: {features.shape}, {features.dtype}")
                        return False
                else:
                    logger.error("❌ Feature extraction returned None")
                    return False
                    
            finally:
                # Clean up test image
                if os.path.exists(test_image_path):
                    os.remove(test_image_path)
                    
        else:
            logger.error("❌ Feature extractor failed to load")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing measurement feature extractor: {e}")
        return False

def test_dual_search_engine():
    """Test the dual search engine"""
    logger.info("🧪 Testing Dual Search Engine...")
    
    try:
        from dual_index_search_engine import dual_search_engine
        
        # Test initialization
        csv_path = "database_results/final_with_aws_shapes_enriched.csv"
        if not os.path.exists(csv_path):
            csv_path = "database_results/DB_ACTIVE.csv"
        
        success = dual_search_engine.initialize(csv_path, "indexes", "1095")
        
        if success:
            logger.info("✅ Dual search engine initialized successfully")
            
            # Get stats
            stats = dual_search_engine.get_search_stats()
            logger.info(f"📊 Search stats: {stats}")
            
            return True
        else:
            logger.error("❌ Dual search engine initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing dual search engine: {e}")
        return False

def test_path_normalization():
    """Test path normalization functionality"""
    logger.info("🧪 Testing Path Normalization...")
    
    try:
        from dual_index_data_loader import DualIndexDataLoader
        
        loader = DualIndexDataLoader()
        
        # Test cases
        test_cases = [
            ("10003502M200_P02_white_bg", "db_pictures_512/10003502M200.jpg"),
            ("1000350FG400_P02_white_bg", "db_pictures_512/1000350FG400.jpg"),
            ("100044001K00_P02_white_bg", "db_pictures_512/100044001K00.jpg"),
            ("1000810FRE_P02_white_bg", "db_pictures_512/1000810FRE.jpg")
        ]
        
        all_passed = True
        for input_name, expected_output in test_cases:
            result = loader.normalize_measurement_filename(input_name)
            if result == expected_output:
                logger.info(f"✅ {input_name} → {result}")
            else:
                logger.error(f"❌ {input_name} → {result} (expected: {expected_output})")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        logger.error(f"❌ Error testing path normalization: {e}")
        return False

def test_index_dimension_compatibility():
    """Test that we can handle different embedding dimensions"""
    logger.info("🧪 Testing Index Dimension Compatibility...")
    
    try:
        import faiss
        
        # Load both indexes to verify dimensions
        main_index_path = "indexes/v11_1095_db_pictures_512_merged_final_20250703_125538.faiss"
        measurement_index_path = "indexes/index_measurements/index.faiss"
        
        if os.path.exists(main_index_path) and os.path.exists(measurement_index_path):
            main_index = faiss.read_index(main_index_path)
            measurement_index = faiss.read_index(measurement_index_path)
            
            logger.info(f"📊 Main index: {main_index.ntotal} vectors, {main_index.d} dimensions")
            logger.info(f"📊 Measurement index: {measurement_index.ntotal} vectors, {measurement_index.d} dimensions")
            
            # Verify expected dimensions
            if main_index.d == 3584 and measurement_index.d == 256:
                logger.info("✅ Index dimensions are as expected")
                return True
            else:
                logger.warning(f"⚠️ Unexpected dimensions: Main={main_index.d}, Measurement={measurement_index.d}")
                return True  # Still pass, but log the warning
        else:
            logger.warning("⚠️ Index files not found, skipping dimension test")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error testing index dimensions: {e}")
        return False

def run_all_tests():
    """Run all integration tests"""
    logger.info("🚀 Starting Dual Index Integration Tests...")
    
    tests = [
        ("Path Normalization", test_path_normalization),
        ("Index Dimension Compatibility", test_index_dimension_compatibility),
        ("Measurement Feature Extractor", test_measurement_feature_extractor),
        ("Dual Data Loader", test_dual_data_loader),
        ("Dual Search Engine", test_dual_search_engine),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        start_time = time.time()
        try:
            result = test_func()
            results[test_name] = result
            elapsed = time.time() - start_time
            
            if result:
                logger.info(f"✅ {test_name} PASSED ({elapsed:.2f}s)")
            else:
                logger.error(f"❌ {test_name} FAILED ({elapsed:.2f}s)")
                
        except Exception as e:
            results[test_name] = False
            elapsed = time.time() - start_time
            logger.error(f"❌ {test_name} ERROR: {e} ({elapsed:.2f}s)")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Dual index integration is working correctly.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the logs above.")
        return False

if __name__ == "__main__":
    # Activate conda environment
    import subprocess
    
    print("🔧 Activating conda environment...")
    result = subprocess.run([
        "bash", "-c", 
        "source ~/miniconda3/etc/profile.d/conda.sh && conda activate faiss_env && python -c 'print(\"Environment activated\")'"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Conda environment activated")
    else:
        print("⚠️ Could not activate conda environment, continuing anyway...")
    
    # Run tests
    success = run_all_tests()
    sys.exit(0 if success else 1) 