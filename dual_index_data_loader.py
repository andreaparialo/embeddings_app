#!/usr/bin/env python3
"""
Dual Index Data Loader
Manages both main index (3584d GME embeddings) and measurement index (256d feature embeddings)
with proper path normalization and scoring combination.
"""

import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import faiss
import pandas as pd
from data_loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DualIndexDataLoader:
    """Manages both main and measurement indexes for combined search"""
    
    def __init__(self):
        # Main index (GME embeddings, 3584d)
        self.main_data_loader = DataLoader()
        self.main_index = None
        self.main_embeddings = None
        self.main_metadata = None
        
        # Measurement index (feature embeddings, 256d) 
        self.measurement_index = None
        self.measurement_embeddings = None
        self.measurement_metadata = None
        self.measurement_path_mapping = {}  # original -> normalized path mapping
        self.measurement_to_main_mapping = {}  # measurement index -> main index mapping
        
        # Combined product database
        self.df = None
        
        # Index file paths
        self.main_index_path = "indexes/v11_1095_db_pictures_512_merged_final_20250703_125538.faiss"
        self.main_metadata_path = "indexes/v11_1095_db_pictures_512_merged_final_20250703_125538_metadata.json"
        self.measurement_index_path = "indexes/index_measurements/index.faiss"
        self.measurement_metadata_path = "indexes/index_measurements/corrected_metadata.json"
        
        # Configuration
        self.main_weight = 0.7  # Weight for main index similarity
        self.measurement_weight = 0.3  # Weight for measurement index similarity
        
    def normalize_measurement_filename(self, measurement_name: str) -> str:
        """Convert measurement index naming to main index format"""
        base_name = measurement_name
        
        # Remove various suffix patterns
        suffixes_to_remove = ['_P02_white_bg', '_P02_white_bg.jpg', '_white_bg', '_P02']
        
        for suffix in suffixes_to_remove:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        
        return f"db_pictures_512/{base_name}.jpg"
    
    def load_main_index(self, csv_path: str, index_dir: str = "indexes", index_id: str = None) -> bool:
        """Load the main GME-based index"""
        try:
            logger.info("📊 Loading main index (GME embeddings, 3584d)...")
            
            # Use the existing DataLoader for main index - exactly like search_engine.py does
            self.df = self.main_data_loader.load_csv(csv_path)
            if self.df is None:
                logger.error("❌ Failed to load CSV data")
                return False
            
            # Load FAISS index using the proper method - exactly like search_engine.py does
            # Use index_id if provided, otherwise use "1095" as default
            if not self.main_data_loader.load_faiss_index(index_dir, "1095", index_id or "1095"):
                logger.error("❌ Failed to load main FAISS index")
                return False
            
            # Get references to main index components
            self.main_index = self.main_data_loader.index
            self.main_embeddings = self.main_data_loader.embeddings
            
            # Load main metadata - try the same fallback approach as FAISS loading
            metadata_paths = [
                "indexes/v11_1095_db_pictures_512_merged_final_20250703_125538_metadata.json",
                "indexes/v11_complete_merged_20250625_115302_metadata.json",
                "indexes/metadata.json"
            ]
            
            metadata_loaded = False
            for metadata_path in metadata_paths:
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.main_metadata = json.load(f)
                    logger.info(f"Loaded main metadata from: {metadata_path}")
                    metadata_loaded = True
                    break
            
            if not metadata_loaded:
                logger.error("❌ Could not find main index metadata file")
                return False
            
            logger.info(f"✅ Main index loaded: {self.main_index.ntotal} vectors, {self.main_index.d}d")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading main index: {e}")
            return False
    
    def load_measurement_index(self) -> bool:
        """Load the measurement-based index"""
        try:
            logger.info("📏 Loading measurement index (feature embeddings, 256d)...")
            
            # Try multiple paths for measurement index (following main index pattern)
            measurement_index_paths = [
                "indexes/index_measurements/index.faiss",
                "index_measurements/index.faiss",
                "indexes/measurement_index.faiss"
            ]
            
            index_loaded = False
            for index_path in measurement_index_paths:
                if os.path.exists(index_path):
                    self.measurement_index = faiss.read_index(index_path)
                    logger.info(f"Loaded measurement FAISS index from: {index_path}")
                    index_loaded = True
                    break
            
            if not index_loaded:
                logger.error("❌ Could not find measurement FAISS index file")
                return False
            
            # Try multiple paths for measurement metadata (following main index pattern)
            measurement_metadata_paths = [
                "indexes/index_measurements/corrected_metadata.json",
                "indexes/index_measurements/metadata.json",
                "index_measurements/corrected_metadata.json",
                "index_measurements/metadata.json"
            ]
            
            metadata_loaded = False
            for metadata_path in measurement_metadata_paths:
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        measurement_data = json.load(f)
                    logger.info(f"Loaded measurement metadata from: {metadata_path}")
                    metadata_loaded = True
                    break
            
            if not metadata_loaded:
                logger.error("❌ Could not find measurement metadata file")
                return False
            
            self.measurement_metadata = measurement_data
            
            # Load measurement embeddings if available
            self.measurement_embeddings_path = "indexes/index_measurements/embeddings.npy"
            if os.path.exists(self.measurement_embeddings_path):
                self.measurement_embeddings = np.load(self.measurement_embeddings_path)
                logger.info(f"✅ Loaded measurement embeddings: {self.measurement_embeddings.shape}")
                
                # Create filename to embedding mapping
                self.measurement_filename_to_idx = {}
                self.measurement_filename_to_embedding = {}
                
                for idx_str, normalized_path in self.measurement_metadata["product_mapping"].items():
                    idx = int(idx_str)
                    # Extract filename_root from normalized path
                    filename_root = os.path.basename(normalized_path).replace('.jpg', '')
                    self.measurement_filename_to_idx[filename_root] = idx
                    self.measurement_filename_to_embedding[filename_root] = self.measurement_embeddings[idx]
                
                logger.info(f"📊 Created filename mappings for {len(self.measurement_filename_to_embedding)} products")
            else:
                logger.warning("⚠️ Measurement embeddings file not found - batch dual search will be limited")
                self.measurement_embeddings = None
                self.measurement_filename_to_embedding = {}
            
            # Create path mapping using corrected metadata
            logger.info("🔄 Creating path mapping from corrected metadata...")
            main_paths = set(self.main_metadata["image_paths"])
            
            overlapping_count = 0
            
            # Handle the corrected format with preserved numerical indices
            if "product_mapping" in measurement_data:
                # Corrected format with preserved FAISS indices
                logger.info("Using corrected metadata format with preserved FAISS indices...")
                for idx_str, normalized_path in measurement_data["product_mapping"].items():
                    idx = int(idx_str)
                    
                    # Get original name from original_mapping if available
                    original_name = measurement_data.get("original_mapping", {}).get(normalized_path, f"unknown_{idx}")
                    
                    self.measurement_path_mapping[idx] = {
                        'original': original_name,
                        'normalized': normalized_path,
                        'in_main_index': normalized_path in main_paths
                    }
                    
                    # Create reverse mapping for overlapping products
                    if normalized_path in main_paths:
                        main_idx = self.main_metadata["image_paths"].index(normalized_path)
                        self.measurement_to_main_mapping[idx] = main_idx
                        overlapping_count += 1
            else:
                # Old format with image_paths array (this was the broken version)
                logger.warning("Using legacy image_paths format - this may cause index misalignment!")
                for idx, normalized_path in enumerate(measurement_data.get("image_paths", [])):
                    # Get original name from original_mapping
                    original_name = measurement_data.get("original_mapping", {}).get(normalized_path, f"unknown_{idx}")
                    
                    self.measurement_path_mapping[idx] = {
                        'original': original_name,
                        'normalized': normalized_path,
                        'in_main_index': normalized_path in main_paths
                    }
                    
                    # Create reverse mapping for overlapping products
                    if normalized_path in main_paths:
                        main_idx = self.main_metadata["image_paths"].index(normalized_path)
                        self.measurement_to_main_mapping[idx] = main_idx
                        overlapping_count += 1
            
            logger.info(f"✅ Measurement index loaded: {self.measurement_index.ntotal} vectors, {self.measurement_index.d}d")
            logger.info(f"📊 Path mapping: {overlapping_count} overlapping products found")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading measurement index: {e}")
            return False
    
    def initialize(self, csv_path: str, index_dir: str = "indexes", index_id: str = None) -> bool:
        """Initialize both indexes"""
        logger.info("🚀 Initializing Dual Index System...")
        
        # Load main index first
        if not self.load_main_index(csv_path, index_dir, index_id):
            return False
        
        # Load measurement index
        if not self.load_measurement_index():
            return False
        
        logger.info("✅ Dual Index System initialized successfully!")
        logger.info(f"📊 Summary:")
        logger.info(f"   Main index: {self.main_index.ntotal} products, {self.main_index.d}d")
        logger.info(f"   Measurement index: {self.measurement_index.ntotal} products, {self.measurement_index.d}d")
        logger.info(f"   Overlapping products: {len(self.measurement_to_main_mapping)}")
        logger.info(f"   Scoring weights: Main={self.main_weight}, Measurement={self.measurement_weight}")
        
        return True
    
    def search_main_index(self, query_embedding: np.ndarray, top_k: int = 50, filters: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """Search in main index with optional pre-filtering"""
        if self.main_index is None:
            raise ValueError("Main index not loaded")
        
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # If filters are provided, use the optimized search with pre-filtering
        if filters and self.df is not None and hasattr(self.main_data_loader, 'idx_to_filename_root') and self.main_embeddings is not None:
            try:
                logger.info(f"🔍 Using pre-filtered search with filters: {filters}")
                logger.info(f"📊 Filter validation: df={self.df is not None}, embeddings={self.main_embeddings is not None}, idx_mapping={hasattr(self.main_data_loader, 'idx_to_filename_root')}")
                # Use the same pre-filtering logic as regular batch search
                from optimized_faiss_search import OptimizedFAISSSearch
                
                # Create OptimizedFAISSSearch with correct parameters
                optimized_search = OptimizedFAISSSearch(
                    index=self.main_index,
                    embeddings=self.main_embeddings,  # Use the embeddings!
                    metadata_df=self.df,
                    filename_to_idx=getattr(self.main_data_loader, 'filename_to_idx', {}),
                    idx_to_filename=getattr(self.main_data_loader, 'idx_to_filename_root', {})
                )
                
                # Use search_with_prefilter which applies filters before FAISS search
                distances, indices = optimized_search.search_with_prefilter(
                    query_embedding, filters, top_k
                )
                return distances[0], indices[0]
            except Exception as e:
                logger.warning(f"⚠️ Pre-filtered search failed ({e}), falling back to raw search")
                import traceback
                traceback.print_exc()
        else:
            if not filters:
                logger.info("🔍 Using raw FAISS search (no filters provided)")
            elif self.df is None:
                logger.warning("⚠️ Cannot use filtered search: df is None")
            elif self.main_embeddings is None:
                logger.warning("⚠️ Cannot use filtered search: embeddings are None")
            elif not hasattr(self.main_data_loader, 'idx_to_filename_root'):
                logger.warning("⚠️ Cannot use filtered search: idx_to_filename_root not available")
        
        # Fallback to raw FAISS search without filtering
        logger.info("🔍 Using raw FAISS search (fallback)")
        distances, indices = self.main_index.search(query_embedding, top_k)
        return distances[0], indices[0]
    
    def search_measurement_index(self, query_embedding: np.ndarray, top_k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Search in measurement index"""
        if self.measurement_index is None:
            raise ValueError("Measurement index not loaded")
        
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.measurement_index.search(query_embedding, top_k)
        return distances[0], indices[0]
    
    def search_measurement_index_with_filters(self, query_embedding: np.ndarray, top_k: int = 50, filters: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """Search in measurement index with optional filtering"""
        if self.measurement_index is None:
            raise ValueError("Measurement index not loaded")
        
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # If no filters, use standard search
        if not filters or not self.df:
            logger.info("🔍 Using raw measurement search (no filters)")
            distances, indices = self.measurement_index.search(query_embedding, top_k * 3)  # Get more results for filtering
            return distances[0], indices[0]
        
        # Apply filters to measurement index
        logger.info(f"🔍 Using filtered measurement search with filters: {filters}")
        
        # First, get a larger set of results
        distances, indices = self.measurement_index.search(query_embedding, top_k * 5)
        distances = distances[0]
        indices = indices[0]
        
        # Filter the results based on product attributes
        filtered_distances = []
        filtered_indices = []
        
        for i, meas_idx in enumerate(indices):
            if meas_idx in self.measurement_path_mapping:
                normalized_path = self.measurement_path_mapping[meas_idx]['normalized']
                filename_root = os.path.basename(normalized_path).replace('.jpg', '')
                
                # Get products matching this filename_root
                matching_products = self.df[self.df['filename_root'] == filename_root]
                if matching_products.empty:
                    continue
                
                # Check if any product matches all filters
                match_found = False
                for _, product in matching_products.iterrows():
                    all_match = True
                    for filter_key, filter_value in filters.items():
                        if filter_key in product:
                            product_val = product[filter_key]
                            # Handle list values (like status codes)
                            if isinstance(filter_value, list):
                                if product_val not in filter_value:
                                    all_match = False
                                    break
                            else:
                                if product_val != filter_value:
                                    all_match = False
                                    break
                    
                    if all_match:
                        match_found = True
                        break
                
                if match_found:
                    filtered_distances.append(distances[i])
                    filtered_indices.append(meas_idx)
                    
                    if len(filtered_indices) >= top_k:
                        break
        
        logger.info(f"📊 Measurement filter results: {len(indices)} → {len(filtered_indices)} products")
        
        # If we have enough filtered results, return them
        if filtered_indices:
            return np.array(filtered_distances), np.array(filtered_indices)
        else:
            # No results after filtering, return empty arrays
            logger.warning("⚠️ No measurement results after filtering")
            return np.array([]), np.array([])
    
    def combine_search_results(self, main_distances: np.ndarray, main_indices: np.ndarray,
                              measurement_distances: np.ndarray, measurement_indices: np.ndarray,
                              top_k: int = 50) -> List[Dict]:
        """
        Combine results from both indexes with weighted scoring.
        Handles the dimension mismatch by normalizing scores independently.
        """
        try:
            # Normalize distances to similarity scores (0-1 range)
            # Convert L2 distances to similarities: similarity = 1 / (1 + distance)
            main_similarities = 1.0 / (1.0 + main_distances)
            measurement_similarities = 1.0 / (1.0 + measurement_distances)
            
            # Normalize to 0-1 range
            if len(main_similarities) > 1:
                main_similarities = (main_similarities - np.min(main_similarities)) / (np.max(main_similarities) - np.min(main_similarities) + 1e-8)
            if len(measurement_similarities) > 1:
                measurement_similarities = (measurement_similarities - np.min(measurement_similarities)) / (np.max(measurement_similarities) - np.min(measurement_similarities) + 1e-8)
            
            # Create combined results
            combined_results = {}
            
            # Add main index results
            for i, (main_idx, main_sim) in enumerate(zip(main_indices, main_similarities)):
                if main_idx < len(self.main_metadata["image_paths"]):
                    image_path = self.main_metadata["image_paths"][main_idx]
                    # For products only in main index, use a default measurement similarity
                    # based on the average of existing similarities to avoid unfair penalization
                    default_meas_sim = 0.5  # Middle of normalized range
                    combined_results[image_path] = {
                        'main_similarity': float(main_sim),
                        'main_rank': i + 1,
                        'measurement_similarity': 0.0,
                        'measurement_rank': None,
                        'combined_score': float(
                            main_sim * self.main_weight + 
                            default_meas_sim * self.measurement_weight
                        ),
                        'source': 'main_only'
                    }
            
            # Add measurement index results
            for i, (meas_idx, meas_sim) in enumerate(zip(measurement_indices, measurement_similarities)):
                if meas_idx in self.measurement_path_mapping:
                    normalized_path = self.measurement_path_mapping[meas_idx]['normalized']
                    
                    if normalized_path in combined_results:
                        # Product exists in both indexes - combine scores
                        combined_results[normalized_path]['measurement_similarity'] = float(meas_sim)
                        combined_results[normalized_path]['measurement_rank'] = i + 1
                        combined_results[normalized_path]['combined_score'] = float(
                            combined_results[normalized_path]['main_similarity'] * self.main_weight +
                            meas_sim * self.measurement_weight
                        )
                        combined_results[normalized_path]['source'] = 'both_indexes'
                    else:
                        # Product only in measurement index
                        # Use a default main similarity to avoid unfair penalization
                        default_main_sim = 0.5  # Middle of normalized range
                        combined_results[normalized_path] = {
                            'main_similarity': 0.0,
                            'main_rank': None,
                            'measurement_similarity': float(meas_sim),
                            'measurement_rank': i + 1,
                            'combined_score': float(
                                default_main_sim * self.main_weight +
                                meas_sim * self.measurement_weight
                            ),
                            'source': 'measurement_only'
                        }
            
            # Sort by combined score and limit to top_k
            sorted_results = sorted(combined_results.items(), 
                                  key=lambda x: x[1]['combined_score'], 
                                  reverse=True)[:top_k]
            
            # Convert to final format with product information
            final_results = []
            for image_path, scores in sorted_results:
                # Get product info from main dataframe
                filename_root = os.path.basename(image_path).replace('.jpg', '')
                matching_products = self.df[self.df['filename_root'] == filename_root]
                
                if not matching_products.empty:
                    product_info = matching_products.iloc[0].to_dict()
                    product_info.update({
                        'image_path': image_path,
                        'similarity_score': scores['combined_score'],
                        'main_similarity': scores['main_similarity'],
                        'measurement_similarity': scores['measurement_similarity'],
                        'main_rank': scores['main_rank'],
                        'measurement_rank': scores['measurement_rank'],
                        'score_source': scores['source']
                    })
                    final_results.append(product_info)
            
            logger.info(f"🎯 Combined search results: {len(final_results)} products")
            logger.info(f"   Main only: {sum(1 for _, s in sorted_results if s['source'] == 'main_only')}")
            logger.info(f"   Measurement only: {sum(1 for _, s in sorted_results if s['source'] == 'measurement_only')}")
            logger.info(f"   Both indexes: {sum(1 for _, s in sorted_results if s['source'] == 'both_indexes')}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Error combining search results: {e}")
            return []
    
    def get_measurement_embedding_by_filename(self, filename_root: str) -> Optional[np.ndarray]:
        """Get measurement embedding for a filename_root"""
        if self.measurement_embeddings is None:
            return None
        
        # Try direct match and case variations
        variations = [
            filename_root,
            filename_root.upper(),
            filename_root.lower()
        ]
        
        for variant in variations:
            if variant in self.measurement_filename_to_embedding:
                return self.measurement_filename_to_embedding[variant]
        
        # If not found, log for debugging
        logger.debug(f"Measurement embedding not found for: {filename_root}")
        return None
    
    def set_scoring_weights(self, main_weight: float, measurement_weight: float):
        """Update the scoring weights for combining results"""
        total = main_weight + measurement_weight
        if total <= 0:
            raise ValueError("Weights must be positive")
        
        self.main_weight = main_weight / total
        self.measurement_weight = measurement_weight / total
        
        logger.info(f"🎛️ Updated scoring weights: Main={self.main_weight:.2f}, Measurement={self.measurement_weight:.2f}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the dual index system"""
        return {
            'main_index': {
                'total_vectors': self.main_index.ntotal if self.main_index else 0,
                'dimensions': self.main_index.d if self.main_index else 0,
                'type': 'GME embeddings'
            },
            'measurement_index': {
                'total_vectors': self.measurement_index.ntotal if self.measurement_index else 0,
                'dimensions': self.measurement_index.d if self.measurement_index else 0,
                'type': 'Feature embeddings'
            },
            'overlapping_products': len(self.measurement_to_main_mapping),
            'scoring_weights': {
                'main_weight': self.main_weight,
                'measurement_weight': self.measurement_weight
            }
        }

# Global instance
dual_index_loader = DualIndexDataLoader() 