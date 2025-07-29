#!/usr/bin/env python3
"""
Create Corrected Measurement Metadata
Converts the measurement index metadata to use the correct path format.
"""

import json
import os
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_measurement_filename(measurement_name: str) -> str:
    """Convert measurement index naming to main index format"""
    base_name = measurement_name
    
    # Remove various suffix patterns
    suffixes_to_remove = ['_P02_white_bg', '_P02_white_bg.jpg', '_white_bg', '_P02']
    
    for suffix in suffixes_to_remove:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break
    
    return f"db_pictures_512/{base_name}.jpg"

def create_corrected_metadata():
    """Create the corrected metadata file"""
    try:
        # Input and output paths
        input_metadata_path = "indexes/index_measurements/metadata.json"
        output_metadata_path = "indexes/index_measurements/corrected_metadata.json"
        main_metadata_path = "indexes/v11_1095_db_pictures_512_merged_final_20250703_125538_metadata.json"
        
        logger.info("🔄 Loading original measurement metadata...")
        
        # Load original measurement metadata
        with open(input_metadata_path, 'r') as f:
            measurement_data = json.load(f)
        
        logger.info(f"📊 Found {len(measurement_data['product_mapping'])} products in measurement index")
        
        # Load main index metadata for comparison
        with open(main_metadata_path, 'r') as f:
            main_data = json.load(f)
        
        main_paths = set(main_data["image_paths"])
        logger.info(f"📊 Found {len(main_paths)} products in main index")
        
        # Create corrected metadata - PRESERVE original structure with normalized paths
        corrected_data = {
            "product_mapping": {},  # Keep the same structure FAISS expects!
            "original_mapping": {},
            "overlap_analysis": {
                "total_measurement": 0,
                "total_main": 0,
                "overlapping": 0,
                "measurement_only": 0,
                "main_only": 0
            }
        }
        
        overlapping_count = 0
        measurement_only_count = 0
        
        logger.info("🔄 Processing and normalizing paths...")
        logger.info("⚠️  PRESERVING original numerical indices for FAISS compatibility")
        
        # Process each measurement product - KEEP THE SAME KEYS!
        for idx_str, measurement_name in measurement_data["product_mapping"].items():
            # Normalize the path
            normalized_path = normalize_measurement_filename(measurement_name)
            
            # Keep the same numerical key that FAISS expects!
            corrected_data["product_mapping"][idx_str] = normalized_path
            corrected_data["original_mapping"][normalized_path] = measurement_name
            
            # Track overlap
            if normalized_path in main_paths:
                overlapping_count += 1
            else:
                measurement_only_count += 1
        
        # Update analysis
        corrected_data["overlap_analysis"] = {
            "total_measurement": len(measurement_data["product_mapping"]),
            "total_main": len(main_paths),
            "overlapping": overlapping_count,
            "measurement_only": measurement_only_count,
            "main_only": len(main_paths) - overlapping_count
        }
        
        # Save corrected metadata
        logger.info(f"💾 Saving corrected metadata to {output_metadata_path}...")
        with open(output_metadata_path, 'w') as f:
            json.dump(corrected_data, f, indent=2)
        
        # Create a sample comparison
        logger.info("📋 Conversion Examples:")
        sample_items = list(measurement_data["product_mapping"].items())[:5]
        for idx, original_name in sample_items:
            normalized = normalize_measurement_filename(original_name)
            in_main = "✅" if normalized in main_paths else "❌"
            logger.info(f"  {original_name} → {normalized} {in_main}")
        
        # Summary
        logger.info(f"✅ Corrected metadata created successfully!")
        logger.info(f"📊 Summary:")
        logger.info(f"   Total measurement products: {corrected_data['overlap_analysis']['total_measurement']}")
        logger.info(f"   Products also in main index: {corrected_data['overlap_analysis']['overlapping']}")
        logger.info(f"   Measurement-only products: {corrected_data['overlap_analysis']['measurement_only']}")
        logger.info(f"   Overlap percentage: {corrected_data['overlap_analysis']['overlapping']/corrected_data['overlap_analysis']['total_measurement']*100:.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating corrected metadata: {e}")
        return False

def verify_corrected_metadata():
    """Verify the corrected metadata is valid"""
    try:
        output_metadata_path = "indexes/index_measurements/corrected_metadata.json"
        
        if not os.path.exists(output_metadata_path):
            logger.error("❌ Corrected metadata file not found")
            return False
        
        with open(output_metadata_path, 'r') as f:
            corrected_data = json.load(f)
        
        # Verify structure - updated for new format
        required_keys = ["product_mapping", "original_mapping", "overlap_analysis"]
        for key in required_keys:
            if key not in corrected_data:
                logger.error(f"❌ Missing key in corrected metadata: {key}")
                return False
        
        # Verify all paths have correct format
        correct_format_count = 0
        for idx, path in corrected_data["product_mapping"].items():
            if path.startswith("db_pictures_512/") and path.endswith(".jpg"):
                correct_format_count += 1
            else:
                logger.warning(f"⚠️ Incorrect path format for index {idx}: {path}")
        
        logger.info(f"✅ Verified {correct_format_count}/{len(corrected_data['product_mapping'])} paths have correct format")
        
        return correct_format_count == len(corrected_data["product_mapping"])
        
    except Exception as e:
        logger.error(f"❌ Error verifying corrected metadata: {e}")
        return False

def main():
    """Main function"""
    logger.info("🚀 Creating Corrected Measurement Index Metadata...")
    
    # Step 1: Create corrected metadata
    if not create_corrected_metadata():
        logger.error("❌ Failed to create corrected metadata")
        return False
    
    # Step 2: Verify the corrected metadata
    if not verify_corrected_metadata():
        logger.error("❌ Corrected metadata verification failed")
        return False
    
    logger.info("🎉 Successfully created and verified corrected measurement metadata!")
    logger.info("📄 Files created:")
    logger.info("   - indexes/index_measurements/corrected_metadata.json")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 