#!/usr/bin/env python3
"""
Debug script for batch dual engine search issues
Tests why measurement index returns 0 results
"""

import os
os.environ['USE_FAISS_GPU'] = 'true'

import logging
import pandas as pd
import numpy as np
from dual_index_data_loader import dual_index_loader
from batch_processor_optimized import OptimizedBatchProcessor
from data_loader import DataLoader
from gme_model import GMEModel

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_measurement_index_mapping():
    """Test if measurement index mapping is working correctly"""
    logger.info("🔍 Testing Measurement Index Mapping...")
    
    # Initialize dual index loader
    csv_path = "database_results/DB_ACTIVE.csv"
    if not dual_index_loader.initialize(csv_path):
        logger.error("Failed to initialize dual index loader")
        return False
    
    # Check measurement path mapping
    logger.info(f"📊 Measurement path mapping entries: {len(dual_index_loader.measurement_path_mapping)}")
    logger.info(f"📊 Measurement to main mapping entries: {len(dual_index_loader.measurement_to_main_mapping)}")
    
    # Sample some mappings
    sample_count = 5
    logger.info(f"\n📋 Sample measurement path mappings (first {sample_count}):")
    for i, (idx, info) in enumerate(dual_index_loader.measurement_path_mapping.items()):
        if i >= sample_count:
            break
        logger.info(f"   Index {idx}: {info['normalized']} (in_main: {info['in_main_index']})")
    
    # Check filename to embedding mapping
    logger.info(f"\n📊 Measurement filename to embedding mapping: {len(dual_index_loader.measurement_filename_to_embedding)} entries")
    
    # Sample some filename mappings
    logger.info(f"\n📋 Sample filename to embedding mappings (first {sample_count}):")
    for i, (filename, embedding) in enumerate(dual_index_loader.measurement_filename_to_embedding.items()):
        if i >= sample_count:
            break
        logger.info(f"   {filename}: embedding shape {embedding.shape}, norm={np.linalg.norm(embedding):.4f}")
    
    return True

def test_specific_product_lookup():
    """Test looking up specific products in measurement index"""
    logger.info("\n🎯 Testing Specific Product Lookups...")
    
    # Test products from the batch logs
    test_products = [
        "1034040KJ100", "1043470RHL00", "1065600PJP00", "1070260OIT00",
        "1070780XYO00", "1078960PJP00", "1081000N9P00", "1084100OIT00"
    ]
    
    for filename_root in test_products[:4]:  # Test first 4
        logger.info(f"\n🔍 Testing {filename_root}:")
        
        # Check if it's in the measurement filename mapping
        embedding = dual_index_loader.get_measurement_embedding_by_filename(filename_root)
        if embedding is not None:
            logger.info(f"   ✅ Found measurement embedding: shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f}")
        else:
            logger.info(f"   ❌ No measurement embedding found")
            
            # Debug: Check variations
            variations = [
                filename_root,
                filename_root.upper(),
                filename_root.lower()
            ]
            logger.debug(f"   Tried variations: {variations}")
            
            # Check if it exists with a different format
            normalized_path = f"db_pictures_512/{filename_root}.jpg"
            found_in_mapping = False
            for idx, info in dual_index_loader.measurement_path_mapping.items():
                if info['normalized'] == normalized_path:
                    logger.info(f"   📍 Found in path mapping at index {idx}")
                    found_in_mapping = True
                    break
            
            if not found_in_mapping:
                logger.info(f"   ❌ Not found in measurement path mapping either")

def test_measurement_search():
    """Test actual measurement index search"""
    logger.info("\n🔍 Testing Measurement Index Search...")
    
    # Get a measurement embedding that we know exists
    test_filename = None
    test_embedding = None
    
    for filename, emb in list(dual_index_loader.measurement_filename_to_embedding.items())[:5]:
        test_filename = filename
        test_embedding = emb
        break
    
    if test_embedding is None:
        logger.error("No measurement embeddings available for testing")
        return
    
    logger.info(f"📊 Testing with filename: {test_filename}")
    
    # Search measurement index
    distances, indices = dual_index_loader.search_measurement_index(test_embedding, 10)
    
    logger.info(f"🎯 Search results:")
    logger.info(f"   Distances: {distances[:5]}")
    logger.info(f"   Indices: {indices[:5]}")
    
    # Check if indices are in path mapping
    found_count = 0
    for idx in indices[:5]:
        if idx in dual_index_loader.measurement_path_mapping:
            info = dual_index_loader.measurement_path_mapping[idx]
            logger.info(f"   ✅ Index {idx} found: {info['normalized']}")
            found_count += 1
        else:
            logger.info(f"   ❌ Index {idx} NOT in path mapping")
    
    logger.info(f"\n📊 Summary: {found_count}/{min(5, len(indices))} indices found in path mapping")

def test_batch_processor_dual_search():
    """Test the batch processor dual search with a small batch"""
    logger.info("\n🚀 Testing Batch Processor Dual Search...")
    
    # Initialize components
    data_loader = DataLoader()
    gme_model = GMEModel()
    
    # Load data
    csv_path = "database_results/DB_ACTIVE.csv"
    df = data_loader.load_csv(csv_path)
    if df is None:
        logger.error("Failed to load CSV")
        return
    
    # Initialize FAISS index
    if not data_loader.load_faiss_index("indexes", "1095", "1095"):
        logger.error("Failed to load FAISS index")
        return
    
    # Create batch processor
    search_engine = None  # Not needed for this test
    batch_processor = OptimizedBatchProcessor(search_engine, data_loader, gme_model)
    
    # Prepare a small test batch
    test_skus = ["1034040KJ100L", "1043470RHL00S"]  # Use actual SKUs from the database
    
    # Find these SKUs in the database
    image_groups = {}
    for sku in test_skus:
        matching_rows = df[df['SKU_COD'] == sku]
        if not matching_rows.empty:
            row = matching_rows.iloc[0]
            filename_root = row['filename_root']
            if filename_root not in image_groups:
                image_groups[filename_root] = {
                    'skus': [sku],
                    'source_item': row.to_dict()
                }
            else:
                image_groups[filename_root]['skus'].append(sku)
            logger.info(f"✅ Found SKU {sku} with filename_root: {filename_root}")
        else:
            logger.warning(f"❌ SKU {sku} not found in database")
    
    if not image_groups:
        logger.error("No valid SKUs found for testing")
        return
    
    logger.info(f"\n📋 Testing with {len(image_groups)} image groups")
    
    # Run dual engine batch search
    results = batch_processor.process_image_groups_with_prefilter(
        image_groups=image_groups,
        matching_cols=['GENDER_STD'],  # Simple filter
        max_results_per_sku=10,
        dual_engine_enabled=True,
        main_weight=0.7,
        measurement_weight=0.3,
        search_mode="global"
    )
    
    logger.info(f"\n📊 Results: {len(results)} total")
    
    # Analyze the results
    if results:
        df_results = pd.DataFrame(results)
        if 'Index_Coverage' in df_results.columns:
            coverage_counts = df_results['Index_Coverage'].value_counts()
            logger.info("\n📊 Index Coverage Distribution:")
            for coverage, count in coverage_counts.items():
                logger.info(f"   {coverage}: {count}")

def main():
    """Run all diagnostic tests"""
    logger.info("="*60)
    logger.info("🔍 DUAL ENGINE BATCH SEARCH DIAGNOSTIC")
    logger.info("="*60)
    
    # Test 1: Check measurement index mapping
    if not test_measurement_index_mapping():
        logger.error("❌ Measurement index mapping test failed")
        return
    
    # Test 2: Test specific product lookups
    test_specific_product_lookup()
    
    # Test 3: Test measurement index search
    test_measurement_search()
    
    # Test 4: Test batch processor
    test_batch_processor_dual_search()
    
    logger.info("\n" + "="*60)
    logger.info("🏁 Diagnostic tests complete")
    logger.info("="*60)

if __name__ == "__main__":
    main()