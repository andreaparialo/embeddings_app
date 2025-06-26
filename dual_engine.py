import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from old_app.data_loader import DataLoader
from old_app.gme_model import GMEModel
from old_app.search_engine import HybridSearchEngine

logger = logging.getLogger(__name__)

class DualEngineSearchEngine:
    """
    Dual Engine Search that combines results from two different model checkpoints.
    When a SKU appears in both result sets, it gets a priority boost.
    """
    
    def __init__(self):
        self.primary_engine = None
        self.secondary_engine = None 
        self.is_dual_initialized = False
        self.primary_checkpoint = None
        self.secondary_checkpoint = None
        
    def initialize_dual_engine(self, csv_path: str, primary_checkpoint: str = "1095", secondary_checkpoint: str = "680", index_dir: str = "indexes"):
        """Initialize both engines with different checkpoints"""
        try:
            logger.info(f"🚀 Initializing dual engine mode")
            logger.info(f"📊 Primary engine: checkpoint-{primary_checkpoint}")
            logger.info(f"📊 Secondary engine: checkpoint-{secondary_checkpoint}")
            
            # Initialize primary engine (main one)
            logger.info("🔧 Initializing primary engine...")
            self.primary_engine = HybridSearchEngine()
            if not self.primary_engine.initialize(csv_path, index_dir, primary_checkpoint):
                logger.error("❌ Failed to initialize primary engine")
                return False
            
            # Initialize secondary engine 
            logger.info("🔧 Initializing secondary engine...")
            self.secondary_engine = HybridSearchEngine()
            if not self.secondary_engine.initialize(csv_path, index_dir, secondary_checkpoint):
                logger.error("❌ Failed to initialize secondary engine")
                return False
            
            self.primary_checkpoint = primary_checkpoint
            self.secondary_checkpoint = secondary_checkpoint
            self.is_dual_initialized = True
            
            logger.info("✅ Dual engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing dual engine: {e}")
            return False
    
    def search_by_image_similarity_dual(self, query_image_path: str, filters: Dict = None, top_k: int = 50) -> List[Dict]:
        """
        Perform dual-engine image similarity search.
        Combines results from both engines with priority boost for intersection.
        """
        try:
            if not self.is_dual_initialized:
                logger.error("❌ Dual engine not initialized")
                return []
            
            logger.info(f"🚀 Starting dual-engine image similarity search")
            logger.info(f"📁 Query image: {query_image_path}")
            logger.info(f"🎯 Target results: {top_k}")
            
            # Get more results from each engine to ensure good combination
            search_k = top_k * 2
            
            # Search with primary engine (e.g., checkpoint-680)
            logger.info(f"🔍 Searching with primary engine (checkpoint-{self.primary_checkpoint})...")
            primary_results = self.primary_engine.search_by_image_similarity(
                query_image_path, filters, search_k
            )
            logger.info(f"📊 Primary engine found {len(primary_results)} results")
            
            # Search with secondary engine (e.g., checkpoint-1095)
            logger.info(f"🔍 Searching with secondary engine (checkpoint-{self.secondary_checkpoint})...")
            secondary_results = self.secondary_engine.search_by_image_similarity(
                query_image_path, filters, search_k
            )
            logger.info(f"📊 Secondary engine found {len(secondary_results)} results")
            
            # Combine results with intersection boost
            combined_results = self._combine_results(primary_results, secondary_results, top_k)
            
            logger.info(f"🎯 Combined dual-engine results: {len(combined_results)} items")
            return combined_results
            
        except Exception as e:
            logger.error(f"❌ Error in dual-engine search: {e}")
            return []
    
    def _combine_results(self, primary_results: List[Dict], secondary_results: List[Dict], top_k: int) -> List[Dict]:
        """
        Combine results from two engines with intersection priority boost.
        SKUs that appear in both get higher priority.
        """
        try:
            # Create mappings by SKU
            primary_by_sku = {item.get('SKU_COD', ''): item for item in primary_results}
            secondary_by_sku = {item.get('SKU_COD', ''): item for item in secondary_results}
            
            # Find intersection (SKUs in both results)
            intersection_skus = set(primary_by_sku.keys()) & set(secondary_by_sku.keys())
            logger.info(f"🔗 Found {len(intersection_skus)} SKUs in both engines")
            
            combined_results = []
            processed_skus = set()
            
            # 1. Process intersection SKUs first (highest priority)
            for sku in intersection_skus:
                if sku and sku not in processed_skus:
                    primary_item = primary_by_sku[sku]
                    secondary_item = secondary_by_sku[sku]
                    
                    # Combine the items with averaged similarity and boost
                    combined_item = primary_item.copy()  # Use primary as base
                    
                    # Average the similarity scores and apply intersection boost
                    primary_sim = primary_item.get('similarity_score', 1.0)
                    secondary_sim = secondary_item.get('similarity_score', 1.0)
                    avg_similarity = (primary_sim + secondary_sim) / 2
                    
                    # Apply intersection boost (reduce distance = higher similarity)
                    boost_factor = 0.9  # 10% boost for intersection
                    boosted_similarity = avg_similarity * boost_factor
                    
                    combined_item['similarity_score'] = boosted_similarity
                    combined_item['dual_engine_boost'] = True
                    combined_item['primary_similarity'] = primary_sim
                    combined_item['secondary_similarity'] = secondary_sim
                    
                    combined_results.append(combined_item)
                    processed_skus.add(sku)
            
            # 2. Add remaining primary results
            for item in primary_results:
                sku = item.get('SKU_COD', '')
                if sku and sku not in processed_skus:
                    item_copy = item.copy()
                    item_copy['dual_engine_boost'] = False
                    item_copy['source_engine'] = 'primary'
                    combined_results.append(item_copy)
                    processed_skus.add(sku)
            
            # 3. Add remaining secondary results
            for item in secondary_results:
                sku = item.get('SKU_COD', '')
                if sku and sku not in processed_skus:
                    item_copy = item.copy()
                    item_copy['dual_engine_boost'] = False
                    item_copy['source_engine'] = 'secondary'
                    combined_results.append(item_copy)
                    processed_skus.add(sku)
            
            # Sort by similarity score (lower is better)
            combined_results.sort(key=lambda x: x.get('similarity_score', 1.0))
            
            # Return top_k results
            final_results = combined_results[:top_k]
            
            # Log statistics
            boosted_count = sum(1 for item in final_results if item.get('dual_engine_boost', False))
            logger.info(f"🎯 Final results: {len(final_results)} items ({boosted_count} with intersection boost)")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Error combining results: {e}")
            return primary_results[:top_k]  # Fallback to primary results

# Global instance
dual_engine = DualEngineSearchEngine() 