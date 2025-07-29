#!/usr/bin/env python3
"""
Measurement Index Adapter
Converts measurement index metadata to match the main index path format.
"""

import json
import os
import logging
from typing import Dict, List, Tuple
import numpy as np
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MeasurementIndexAdapter:
    """Adapts measurement index to match main index path format"""
    
    def __init__(self):
        self.measurement_metadata_path = "indexes/index_measurements/metadata.json"
        self.measurement_index_path = "indexes/index_measurements/index.faiss" 
        self.main_metadata_path = "indexes/v11_1095_db_pictures_512_merged_final_20250703_125538_metadata.json"
        
        # Output paths for adapted measurement index
        self.adapted_metadata_path = "indexes/index_measurements/adapted_metadata.json"
        self.adapted_index_path = "indexes/index_measurements/adapted_index.faiss"
        
    def normalize_filename(self, measurement_name: str) -> str:
        """
        Convert measurement index naming to main index format
        Example: 10003502M200_P02_white_bg -> db_pictures_512/10003502M200.jpg
        """
        # Remove the _P02_white_bg suffix (or similar patterns)
        base_name = measurement_name
        
        # Handle various suffix patterns
        suffixes_to_remove = ['_P02_white_bg', '_P02_white_bg.jpg', '_white_bg', '_P02']
        
        for suffix in suffixes_to_remove:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        
        # Add standard path and extension
        normalized_path = f"db_pictures_512/{base_name}.jpg"
        return normalized_path
    
    def load_main_index_paths(self) -> set:
        """Load paths from main index for comparison"""
        try:
            with open(self.main_metadata_path, 'r') as f:
                main_data = json.load(f)
            
            main_paths = set(main_data["image_paths"])
            logger.info(f"Loaded {len(main_paths)} paths from main index")
            return main_paths
            
        except Exception as e:
            logger.error(f"Error loading main index paths: {e}")
            return set()
    
    def analyze_overlap(self) -> Tuple[Dict, Dict]:
        """Analyze overlap between measurement and main indexes"""
        try:
            # Load measurement metadata
            with open(self.measurement_metadata_path, 'r') as f:
                measurement_data = json.load(f)
            
            # Load main index paths
            main_paths = self.load_main_index_paths()
            
            # Create mapping and analyze overlap
            normalized_mapping = {}
            overlapping_indices = []
            measurement_only_indices = []
            main_only_paths = set(main_paths)
            
            logger.info("Analyzing measurement index naming patterns...")
            
            for idx_str, measurement_name in measurement_data["product_mapping"].items():
                normalized_path = self.normalize_filename(measurement_name)
                normalized_mapping[idx_str] = {
                    'original': measurement_name,
                    'normalized': normalized_path,
                    'in_main_index': normalized_path in main_paths
                }
                
                if normalized_path in main_paths:
                    overlapping_indices.append(int(idx_str))
                    main_only_paths.discard(normalized_path)  # Remove from main-only set
                else:
                    measurement_only_indices.append(int(idx_str))
            
            analysis = {
                'total_measurement': len(measurement_data["product_mapping"]),
                'total_main': len(main_paths),
                'overlapping': len(overlapping_indices),
                'measurement_only': len(measurement_only_indices),
                'main_only': len(main_only_paths),
                'overlapping_indices': overlapping_indices,
                'measurement_only_indices': measurement_only_indices[:10],  # Sample
                'main_only_paths': list(main_only_paths)[:10]  # Sample
            }
            
            logger.info(f"Analysis Results:")
            logger.info(f"  Total measurement products: {analysis['total_measurement']}")
            logger.info(f"  Total main index products: {analysis['total_main']}")
            logger.info(f"  Overlapping products: {analysis['overlapping']}")
            logger.info(f"  Measurement-only products: {analysis['measurement_only']}")
            logger.info(f"  Main-only products: {analysis['main_only']}")
            
            return normalized_mapping, analysis
            
        except Exception as e:
            logger.error(f"Error analyzing overlap: {e}")
            return {}, {}
    
    def create_adapted_metadata(self, normalized_mapping: Dict) -> bool:
        """Create adapted metadata file with normalized paths"""
        try:
            # Create adapted metadata in the same format as main index
            adapted_data = {
                "image_paths": [],
                "original_mapping": {},  # Keep track of original names
                "overlap_info": "Adapted from measurement index to match main index format"
            }
            
            # Add normalized paths and keep mapping
            for idx_str, mapping_info in normalized_mapping.items():
                adapted_data["image_paths"].append(mapping_info['normalized'])
                adapted_data["original_mapping"][mapping_info['normalized']] = mapping_info['original']
            
            # Save adapted metadata
            os.makedirs(os.path.dirname(self.adapted_metadata_path), exist_ok=True)
            with open(self.adapted_metadata_path, 'w') as f:
                json.dump(adapted_data, f, indent=2)
            
            logger.info(f"Created adapted metadata with {len(adapted_data['image_paths'])} paths")
            logger.info(f"Saved to: {self.adapted_metadata_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating adapted metadata: {e}")
            return False
    
    def copy_faiss_index(self) -> bool:
        """Copy the FAISS index file to the adapted location"""
        try:
            import shutil
            shutil.copy2(self.measurement_index_path, self.adapted_index_path)
            logger.info(f"Copied FAISS index to: {self.adapted_index_path}")
            return True
        except Exception as e:
            logger.error(f"Error copying FAISS index: {e}")
            return False
    
    def validate_index_dimensions(self) -> bool:
        """Validate that both indexes have compatible dimensions"""
        try:
            # Load measurement index
            measurement_index = faiss.read_index(self.measurement_index_path)
            
            # Load main index  
            main_index_path = "indexes/v11_1095_db_pictures_512_merged_final_20250703_125538.faiss"
            main_index = faiss.read_index(main_index_path)
            
            logger.info(f"Index Dimensions:")
            logger.info(f"  Measurement index: {measurement_index.ntotal} vectors, {measurement_index.d} dimensions")
            logger.info(f"  Main index: {main_index.ntotal} vectors, {main_index.d} dimensions")
            
            if measurement_index.d != main_index.d:
                logger.error(f"❌ Dimension mismatch! Measurement: {measurement_index.d}, Main: {main_index.d}")
                return False
            
            logger.info("✅ Index dimensions are compatible")
            return True
            
        except Exception as e:
            logger.error(f"Error validating index dimensions: {e}")
            return False
    
    def run_adaptation(self) -> bool:
        """Run the complete adaptation process"""
        logger.info("🚀 Starting measurement index adaptation...")
        
        # Step 1: Validate index dimensions
        if not self.validate_index_dimensions():
            return False
        
        # Step 2: Analyze overlap and create mapping
        normalized_mapping, analysis = self.analyze_overlap()
        if not normalized_mapping:
            return False
        
        # Step 3: Create adapted metadata
        if not self.create_adapted_metadata(normalized_mapping):
            return False
        
        # Step 4: Copy FAISS index
        if not self.copy_faiss_index():
            return False
        
        logger.info("✅ Measurement index adaptation completed successfully!")
        
        # Print summary
        logger.info(f"\n📊 Adaptation Summary:")
        logger.info(f"  Adapted {analysis['total_measurement']} measurement products")
        logger.info(f"  Found {analysis['overlapping']} overlapping products with main index")
        logger.info(f"  {analysis['measurement_only']} products are unique to measurement index")
        logger.info(f"  Files created:")
        logger.info(f"    - {self.adapted_metadata_path}")
        logger.info(f"    - {self.adapted_index_path}")
        
        return True

def main():
    """Main function to run the adaptation"""
    adapter = MeasurementIndexAdapter()
    success = adapter.run_adaptation()
    
    if success:
        print("\n🎉 Measurement index successfully adapted!")
        print("You can now integrate it with the main search system.")
    else:
        print("\n❌ Adaptation failed. Check the logs for details.")
    
    return success

if __name__ == "__main__":
    main() 