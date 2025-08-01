#!/usr/bin/env python3
"""
Check what metadata is available in the measurement index
"""

import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_measurement_metadata():
    """Check what metadata is stored in measurement index"""
    
    logger.info("📏 Loading measurement metadata...")
    with open("indexes/index_measurements/corrected_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"\n📊 Metadata keys: {list(metadata.keys())}")
    
    # Check if there's any product information beyond paths
    if "product_info" in metadata:
        logger.info("\n✅ Found product_info in metadata")
        sample = list(metadata["product_info"].items())[:3]
        for key, info in sample:
            logger.info(f"   {key}: {info}")
    else:
        logger.info("\n❌ No product_info found in metadata")
    
    # Check original mapping
    if "original_mapping" in metadata:
        logger.info(f"\n📋 Original mapping has {len(metadata['original_mapping'])} entries")
        sample = list(metadata["original_mapping"].items())[:5]
        for norm_path, orig_name in sample:
            logger.info(f"   {norm_path} -> {orig_name}")
    
    # Check product mapping
    if "product_mapping" in metadata:
        logger.info(f"\n📋 Product mapping has {len(metadata['product_mapping'])} entries")
        sample = list(metadata["product_mapping"].items())[:5]
        for idx, path in sample:
            logger.info(f"   Index {idx} -> {path}")

if __name__ == "__main__":
    check_measurement_metadata()