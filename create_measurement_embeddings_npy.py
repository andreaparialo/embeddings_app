#!/usr/bin/env python3
"""
Create Measurement Embeddings NPY File
Extracts embeddings for all products in the measurement index and saves them as NPY.
Since IndexIVFFlat doesn't support reconstruct(), we'll re-extract features from images.
"""

import numpy as np
import json
import os
import logging
from typing import Dict, Optional
import time
from measurement_feature_extractor import MeasurementFeatureExtractor
from PIL import Image
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MeasurementEmbeddingExtractor:
    def __init__(self):
        self.extractor = MeasurementFeatureExtractor(method="resnet_pool")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def extract_embeddings(self):
        """Extract embeddings for all products in measurement index"""
        
        # Load measurement metadata
        logger.info("📊 Loading measurement metadata...")
        with open('indexes/index_measurements/corrected_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        product_mapping = metadata['product_mapping']
        num_products = len(product_mapping)
        logger.info(f"Found {num_products} products in measurement index")
        
        # Load the feature extractor model
        logger.info("🔧 Loading feature extraction model...")
        if not self.extractor.load_model():
            raise RuntimeError("Failed to load feature extraction model")
        
        # Initialize embeddings array
        embeddings = np.zeros((num_products, 256), dtype=np.float32)
        
        # Process each product
        logger.info("🚀 Starting feature extraction...")
        start_time = time.time()
        processed = 0
        missing = 0
        
        for idx_str, normalized_path in product_mapping.items():
            idx = int(idx_str)
            
            # Construct full image path
            # The normalized path is like "db_pictures_512/ABC123.jpg"
            full_path = os.path.join("/lambda/nfs/SPEEDINGTHEPROCESS/old_app", normalized_path)
            
            if os.path.exists(full_path):
                try:
                    # Extract features
                    features = self.extractor.extract_features(full_path)
                    if features is not None:
                        embeddings[idx] = features
                        processed += 1
                    else:
                        logger.warning(f"Failed to extract features for {normalized_path}")
                        missing += 1
                except Exception as e:
                    logger.error(f"Error processing {normalized_path}: {e}")
                    missing += 1
            else:
                # Try alternative path without the db_pictures_512 prefix
                alt_path = os.path.join("/lambda/nfs/SPEEDINGTHEPROCESS/old_app/db_pictures_512", 
                                       os.path.basename(normalized_path))
                if os.path.exists(alt_path):
                    try:
                        features = self.extractor.extract_features(alt_path)
                        if features is not None:
                            embeddings[idx] = features
                            processed += 1
                        else:
                            logger.warning(f"Failed to extract features for {normalized_path}")
                            missing += 1
                    except Exception as e:
                        logger.error(f"Error processing {normalized_path}: {e}")
                        missing += 1
                else:
                    logger.warning(f"Image not found: {normalized_path}")
                    missing += 1
            
            # Progress update
            if (idx + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                eta = (num_products - processed) / rate
                logger.info(f"Progress: {idx + 1}/{num_products} ({processed} processed, {missing} missing) "
                          f"- Rate: {rate:.1f} imgs/sec - ETA: {eta/60:.1f} min")
        
        # Save embeddings
        output_path = 'indexes/index_measurements/embeddings.npy'
        np.save(output_path, embeddings)
        
        # Final statistics
        elapsed = time.time() - start_time
        logger.info(f"✅ Extraction completed in {elapsed/60:.1f} minutes")
        logger.info(f"📊 Results: {processed} extracted, {missing} missing")
        logger.info(f"💾 Saved embeddings to {output_path}")
        logger.info(f"📐 Embeddings shape: {embeddings.shape}")
        
        # Save extraction metadata
        extraction_metadata = {
            'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_products': num_products,
            'successfully_extracted': processed,
            'missing_or_failed': missing,
            'extraction_method': 'resnet_pool',
            'embedding_dimension': 256,
            'device': str(self.device)
        }
        
        metadata_path = 'indexes/index_measurements/embeddings_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(extraction_metadata, f, indent=2)
        logger.info(f"📄 Saved extraction metadata to {metadata_path}")
        
        return embeddings

def main():
    """Main function"""
    try:
        extractor = MeasurementEmbeddingExtractor()
        embeddings = extractor.extract_embeddings()
        
        # Verify the embeddings
        logger.info("\n🔍 Verifying embeddings...")
        non_zero = np.count_nonzero(embeddings.any(axis=1))
        logger.info(f"Non-zero embeddings: {non_zero}/{len(embeddings)}")
        
        # Check a few embeddings
        for i in range(min(5, len(embeddings))):
            norm = np.linalg.norm(embeddings[i])
            logger.info(f"Embedding {i} norm: {norm:.4f}")
            
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())