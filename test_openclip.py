#!/usr/bin/env python3
"""
Test script for OpenCLIP implementation
"""

import logging
from openclip_model import openclip_model
from openclip_data_loader import openclip_data_loader
from openclip_search_engine import openclip_search_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_openclip_model():
    """Test OpenCLIP model loading and basic functionality"""
    logger.info("=" * 80)
    logger.info("Testing OpenCLIP Model")
    logger.info("=" * 80)
    
    # Test loading model
    logger.info("1. Testing model loading...")
    if openclip_model.load_model("epoch_008_model.pth"):
        logger.info("✅ Model loaded successfully")
    else:
        logger.error("❌ Failed to load model")
        return False
    
    # Test encoding a single image
    logger.info("\n2. Testing single image encoding...")
    test_image = "pictures/10003502M200_O00.JPG"  # Use an example image
    embedding = openclip_model.encode_image(test_image)
    
    if embedding is not None:
        logger.info(f"✅ Image encoded successfully")
        logger.info(f"   Embedding shape: {embedding.shape}")
        logger.info(f"   Embedding dtype: {embedding.dtype}")
        logger.info(f"   Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
    else:
        logger.error("❌ Failed to encode image")
        return False
    
    # Test memory usage
    logger.info("\n3. Checking GPU memory usage...")
    memory_info = openclip_model.get_memory_usage()
    logger.info(f"   GPU Memory: {memory_info}")
    
    return True

def test_data_loader():
    """Test OpenCLIP data loader"""
    logger.info("\n" + "=" * 80)
    logger.info("Testing OpenCLIP Data Loader")
    logger.info("=" * 80)
    
    # Test loading CSV
    logger.info("1. Testing CSV loading...")
    csv_path = "database_results/final_comprehensive_data_20250624_229050.csv"
    df = openclip_data_loader.load_csv(csv_path)
    
    if df is not None:
        logger.info(f"✅ CSV loaded successfully")
        logger.info(f"   Shape: {df.shape}")
        logger.info(f"   Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
    else:
        logger.error("❌ Failed to load CSV")
        return False
    
    # Test loading index (if exists)
    logger.info("\n2. Testing index loading...")
    index_loaded = openclip_data_loader.load_faiss_index("indexes/openclip")
    
    if index_loaded:
        logger.info(f"✅ Index loaded successfully")
        logger.info(f"   Index size: {openclip_data_loader.index.ntotal}")
        logger.info(f"   Embeddings shape: {openclip_data_loader.embeddings.shape if openclip_data_loader.embeddings is not None else 'None'}")
    else:
        logger.warning("⚠️  No pre-computed index found (this is expected if you haven't created embeddings yet)")
    
    return True

def test_search_engine():
    """Test OpenCLIP search engine initialization"""
    logger.info("\n" + "=" * 80)
    logger.info("Testing OpenCLIP Search Engine")
    logger.info("=" * 80)
    
    # Test initialization
    logger.info("1. Testing search engine initialization...")
    csv_path = "database_results/final_comprehensive_data_20250624_229050.csv"
    
    if openclip_search_engine.initialize(csv_path):
        logger.info("✅ Search engine initialized successfully")
    else:
        logger.error("❌ Failed to initialize search engine")
        return False
    
    # Test SKU search
    logger.info("\n2. Testing SKU search...")
    test_sku = "10003502M200"  # Example SKU
    results = openclip_search_engine.search_by_sku(test_sku, top_k=5)
    
    if results:
        logger.info(f"✅ SKU search returned {len(results)} results")
        logger.info(f"   First result: {results[0].get('SKU_COD', 'N/A')}")
    else:
        logger.warning("⚠️  No results found for test SKU")
    
    # Test filter options
    logger.info("\n3. Testing filter options...")
    filter_options = openclip_search_engine.get_filter_options()
    
    if filter_options:
        logger.info(f"✅ Retrieved filter options for {len(filter_options)} columns")
        logger.info(f"   Sample filters: {list(filter_options.keys())[:5]}...")
    else:
        logger.warning("⚠️  No filter options retrieved")
    
    return True

def main():
    """Run all tests"""
    logger.info("Starting OpenCLIP system tests...")
    
    all_passed = True
    
    # Test each component
    if not test_openclip_model():
        all_passed = False
    
    if not test_data_loader():
        all_passed = False
    
    if not test_search_engine():
        all_passed = False
    
    # Summary
    logger.info("\n" + "=" * 80)
    if all_passed:
        logger.info("✅ All tests passed!")
        logger.info("\nNext steps:")
        logger.info("1. If no embeddings exist, run: python openclip_create_embeddings.py")
        logger.info("2. To start the OpenCLIP app, run: python app_openclip.py")
        logger.info("3. Access the app at: http://localhost:8001")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
    logger.info("=" * 80)

if __name__ == "__main__":
    main() 