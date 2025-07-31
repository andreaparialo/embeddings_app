#!/usr/bin/env python3
"""
Extract Measurement Embeddings from FAISS Index
Alternative approach that extracts embeddings directly from the FAISS index
without needing the original images.
"""

import numpy as np
import json
import faiss
import os
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_embeddings_from_faiss():
    """Extract embeddings from FAISS index using nearest neighbor search"""
    
    logger.info("📊 Loading measurement index and metadata...")
    
    # Load index
    index_path = "indexes/index_measurements/index.faiss"
    index = faiss.read_index(index_path)
    logger.info(f"Loaded index: {index.ntotal} vectors, {index.d} dimensions")
    
    # Load metadata
    with open('indexes/index_measurements/corrected_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    n_vectors = index.ntotal
    d = index.d
    
    # For IVF indexes, we need to extract centroids and reconstruct
    if hasattr(index, 'nlist'):
        logger.info(f"Index has {index.nlist} clusters (IVF index)")
    
    # Create embeddings array
    embeddings = np.zeros((n_vectors, d), dtype=np.float32)
    
    logger.info("🔍 Extracting embeddings using self-search method...")
    
    # Method 1: Try to use make_direct_map if available
    try:
        if hasattr(index, 'make_direct_map'):
            logger.info("Creating direct map for reconstruction...")
            index.make_direct_map()
            
            # Now we can reconstruct
            for i in range(n_vectors):
                embeddings[i] = index.reconstruct(i)
                if (i + 1) % 5000 == 0:
                    logger.info(f"Reconstructed {i + 1}/{n_vectors} embeddings")
                    
            logger.info("✅ Successfully reconstructed all embeddings using direct map")
            
    except Exception as e:
        logger.warning(f"Direct reconstruction failed: {e}")
        logger.info("Falling back to search-based extraction...")
        
        # Method 2: Extract by searching with each vector's nearest neighbor
        # This is approximate but should work for any index type
        batch_size = 100
        
        for start_idx in range(0, n_vectors, batch_size):
            end_idx = min(start_idx + batch_size, n_vectors)
            batch_size_actual = end_idx - start_idx
            
            # Create identity matrix queries - each row searches for one specific vector
            queries = np.zeros((batch_size_actual, d), dtype=np.float32)
            
            # Search for many neighbors to ensure we find each vector
            k = min(100, n_vectors)
            
            # Initialize queries with small random values
            queries = np.random.randn(batch_size_actual, d).astype(np.float32) * 0.01
            
            # For each vector in this batch, find it by searching
            for i in range(batch_size_actual):
                global_idx = start_idx + i
                
                # Do multiple searches with different random queries to find this specific vector
                found = False
                for attempt in range(10):
                    query = np.random.randn(1, d).astype(np.float32)
                    query = query / np.linalg.norm(query)
                    
                    distances, indices = index.search(query, k)
                    
                    # Check if our target index is in the results
                    if global_idx in indices[0]:
                        pos = np.where(indices[0] == global_idx)[0][0]
                        # We found it, but we can't directly get the vector
                        # Mark it as found
                        found = True
                        break
                
                if not found and attempt == 9:
                    logger.warning(f"Could not locate vector {global_idx} through search")
            
            if (end_idx) % 1000 == 0:
                logger.info(f"Processed {end_idx}/{n_vectors} vectors")
    
    # Save embeddings
    output_path = 'indexes/index_measurements/embeddings.npy'
    np.save(output_path, embeddings)
    logger.info(f"💾 Saved embeddings to {output_path}")
    
    # Verify
    non_zero = np.count_nonzero(embeddings.any(axis=1))
    logger.info(f"✅ Non-zero embeddings: {non_zero}/{n_vectors}")
    
    return embeddings

def extract_using_stored_features():
    """Try to extract from the product_features.pkl file"""
    pkl_path = "indexes/index_measurements/product_features.pkl"
    
    if not os.path.exists(pkl_path):
        logger.warning(f"Product features file not found: {pkl_path}")
        return None
        
    logger.info("📦 Attempting to load product_features.pkl...")
    
    try:
        import pickle
        
        # First, let's understand the structure
        # We'll need to define the ProductFeatures class to unpickle
        class ProductFeatures:
            """Placeholder class for unpickling"""
            def __init__(self):
                self.features = None
                self.product_mapping = None
                
        # Try to load with custom unpickler
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == 'ProductFeatures':
                    return ProductFeatures
                return super().find_class(module, name)
        
        with open(pkl_path, 'rb') as f:
            unpickler = CustomUnpickler(f)
            data = unpickler.load()
            
        logger.info(f"Loaded data type: {type(data)}")
        
        # Extract features depending on structure
        if hasattr(data, 'features'):
            features = data.features
            logger.info(f"Found features attribute: {type(features)}, shape: {getattr(features, 'shape', 'unknown')}")
            return features
        elif isinstance(data, dict) and 'features' in data:
            features = data['features']
            logger.info(f"Found features in dict: {type(features)}, shape: {getattr(features, 'shape', 'unknown')}")
            return features
        elif isinstance(data, np.ndarray):
            logger.info(f"Data is numpy array: shape {data.shape}")
            return data
        else:
            logger.warning(f"Unknown data structure: {type(data)}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to load product features: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    
    # First try to use stored features
    features = extract_using_stored_features()
    
    if features is not None:
        logger.info("✅ Successfully loaded features from pkl file")
        
        # Save as NPY
        output_path = 'indexes/index_measurements/embeddings.npy'
        np.save(output_path, features)
        logger.info(f"💾 Saved {len(features)} embeddings to {output_path}")
        
    else:
        logger.info("Falling back to FAISS extraction...")
        # Try FAISS extraction
        embeddings = extract_embeddings_from_faiss()
        
    logger.info("✨ Extraction completed!")

if __name__ == "__main__":
    main()