#!/usr/bin/env python3
"""
Dual Index Search Engine
Enhanced search engine that can search both GME embeddings (3584d) and measurement feature embeddings (256d)
with intelligent result combination.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import os
from dual_index_data_loader import dual_index_loader
from gme_model import gme_model
from measurement_feature_extractor import MeasurementFeatureExtractor

logger = logging.getLogger(__name__)

class DualIndexSearchEngine:
    """Enhanced search engine supporting both GME and measurement indexes"""
    
    def __init__(self):
        self.is_initialized = False
        self.gme_loaded = False
        self.measurement_extractor_loaded = False
        self.checkpoint = "1095"
        
        # Feature extractor for measurement index (256d)
        self.measurement_extractor = None
    
    def initialize(self, csv_path: str, index_dir: str = "indexes", checkpoint: str = "1095", index_id: str = None):
        """Initialize the dual search engine with both indexes"""
        try:
            logger.info("🚀 Initializing Dual Index Search Engine...")
            logger.info(f"📊 Loading product data from: {csv_path}")
            
            # Initialize dual index data loader
            if not dual_index_loader.initialize(csv_path, index_dir, index_id):
                return False
            
            # Store checkpoint for lazy loading GME model later
            self.checkpoint = checkpoint
            self.gme_loaded = False
            self.measurement_extractor_loaded = False
            
            self.is_initialized = True
            logger.info("✅ Dual Index Search Engine initialized successfully")
            logger.info("💡 Models will be loaded on first search (lazy loading)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing dual search engine: {e}")
            return False
    
    def _ensure_gme_loaded(self):
        """Lazy load GME model when needed"""
        if not self.gme_loaded:
            logger.info("🤖 Loading GME model for main index search...")
            if gme_model.load_model("gme-Qwen2-VL-7B-Instruct", self.checkpoint):
                self.gme_loaded = True
                logger.info("✅ GME model loaded successfully")
            else:
                logger.error("❌ Failed to load GME model")
                return False
        return True
    
    def _ensure_measurement_extractor_loaded(self):
        """Lazy load measurement feature extractor when needed"""
        if not self.measurement_extractor_loaded:
            logger.info("📏 Loading measurement feature extractor...")
            try:
                self.measurement_extractor = MeasurementFeatureExtractor()
                if self.measurement_extractor.load_model():
                    self.measurement_extractor_loaded = True
                    logger.info("✅ Measurement feature extractor loaded successfully")
                else:
                    logger.error("❌ Failed to load measurement feature extractor")
                    return False
            except Exception as e:
                logger.error(f"❌ Error loading measurement feature extractor: {e}")
                return False
        return True
    
    def search_by_image_similarity_dual(self, query_image_path: str, filters: Dict = None, 
                                       top_k: int = 50, main_weight: float = 0.7, 
                                       measurement_weight: float = 0.3) -> List[Dict]:
        """
        Search using both indexes with configurable weighting
        """
        try:
            if not self.is_initialized:
                logger.error("Dual search engine not initialized")
                return []
            
            logger.info(f"🔍 Starting dual-index image similarity search")
            logger.info(f"📁 Query image: {query_image_path}")
            logger.info(f"🎯 Target results: {top_k}")
            logger.info(f"⚖️ Weights: Main={main_weight:.2f}, Measurement={measurement_weight:.2f}")
            
            # Update scoring weights
            dual_index_loader.set_scoring_weights(main_weight, measurement_weight)
            
            # Generate GME embedding for main index
            gme_embedding = None
            if main_weight > 0:
                if not self._ensure_gme_loaded():
                    logger.error("❌ Cannot load GME model for main index search")
                    main_weight = 0
                else:
                    logger.info("🤖 Generating GME embedding for main index...")
                    gme_embedding = gme_model.encode_image(query_image_path)
                    if gme_embedding is None:
                        logger.error("❌ Could not generate GME embedding")
                        main_weight = 0
                    else:
                        logger.info(f"✅ GME embedding generated (dimension: {gme_embedding.shape})")
            
            # Generate measurement embedding for measurement index
            measurement_embedding = None
            if measurement_weight > 0:
                if not self._ensure_measurement_extractor_loaded():
                    logger.error("❌ Cannot load measurement feature extractor")
                    measurement_weight = 0
                else:
                    logger.info("📏 Generating measurement features for measurement index...")
                    measurement_embedding = self.measurement_extractor.extract_features(query_image_path)
                    if measurement_embedding is None:
                        logger.error("❌ Could not generate measurement features")
                        measurement_weight = 0
                    else:
                        logger.info(f"✅ Measurement features generated (dimension: {measurement_embedding.shape})")
            
            # Check if we have at least one valid embedding
            if main_weight == 0 and measurement_weight == 0:
                logger.error("❌ No valid embeddings generated")
                return []
            
            # Search in main index
            main_distances = np.array([])
            main_indices = np.array([])
            if main_weight > 0 and gme_embedding is not None:
                logger.info("🔍 Searching in main index (GME embeddings)...")
                search_k = min(top_k * 2, 1000)  # Get more results for better combination
                main_distances, main_indices = dual_index_loader.search_main_index(
                    gme_embedding, search_k
                )
                logger.info(f"📊 Main index found {len(main_distances)} results")
            
            # Search in measurement index
            measurement_distances = np.array([])
            measurement_indices = np.array([])
            if measurement_weight > 0 and measurement_embedding is not None:
                logger.info("🔍 Searching in measurement index (feature embeddings)...")
                search_k = min(top_k * 2, 1000)  # Get more results for better combination
                measurement_distances, measurement_indices = dual_index_loader.search_measurement_index(
                    measurement_embedding, search_k
                )
                logger.info(f"📊 Measurement index found {len(measurement_distances)} results")
            
            # Combine results
            logger.info("🎯 Combining results from both indexes...")
            combined_results = dual_index_loader.combine_search_results(
                main_distances, main_indices, 
                measurement_distances, measurement_indices, 
                top_k
            )
            
            # Apply filters if provided
            if filters and combined_results:
                logger.info(f"🔧 Applying filters: {list(filters.keys())}")
                filtered_results = self._apply_filters(combined_results, filters)
                logger.info(f"📊 After filtering: {len(filtered_results)} results")
                combined_results = filtered_results
            
            logger.info(f"✅ Dual-index search completed - returned {len(combined_results)} results")
            return combined_results
            
        except Exception as e:
            logger.error(f"❌ Error in dual-index search: {e}")
            return []
    
    def search_by_image_similarity_main_only(self, query_image_path: str, filters: Dict = None, top_k: int = 50) -> List[Dict]:
        """Search using only the main GME index (fallback method)"""
        return self.search_by_image_similarity_dual(
            query_image_path, filters, top_k, 
            main_weight=1.0, measurement_weight=0.0
        )
    
    def search_by_image_similarity_measurement_only(self, query_image_path: str, filters: Dict = None, top_k: int = 50) -> List[Dict]:
        """Search using only the measurement index"""
        return self.search_by_image_similarity_dual(
            query_image_path, filters, top_k, 
            main_weight=0.0, measurement_weight=1.0
        )
    
    def _apply_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """Apply filters to search results"""
        try:
            filtered_results = []
            
            for result in results:
                should_include = True
                
                for filter_col, filter_value in filters.items():
                    if filter_col in result:
                        result_value = result[filter_col]
                        
                        # Handle different filter types
                        if isinstance(filter_value, list):
                            # Multiple values filter
                            if result_value not in filter_value:
                                should_include = False
                                break
                        elif isinstance(filter_value, str):
                            # String filter
                            if str(result_value).upper() != str(filter_value).upper():
                                should_include = False
                                break
                        else:
                            # Exact match filter
                            if result_value != filter_value:
                                should_include = False
                                break
                
                if should_include:
                    filtered_results.append(result)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"❌ Error applying filters: {e}")
            return results
    
    def get_search_stats(self) -> Dict:
        """Get statistics about the dual index system"""
        stats = dual_index_loader.get_stats()
        stats.update({
            'models_loaded': {
                'gme_model': self.gme_loaded,
                'measurement_extractor': self.measurement_extractor_loaded
            },
            'search_capabilities': {
                'main_index_search': True,
                'measurement_index_search': True,
                'dual_index_search': True,
                'configurable_weights': True
            }
        })
        return stats
    
    def set_default_weights(self, main_weight: float, measurement_weight: float):
        """Set default weights for dual-index searches"""
        dual_index_loader.set_scoring_weights(main_weight, measurement_weight)

# Global instance
dual_search_engine = DualIndexSearchEngine() 