import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import os
from old_app.data_loader import data_loader
from old_app.gme_model import gme_model

logger = logging.getLogger(__name__)

class HybridSearchEngine:
    def __init__(self):
        self.is_initialized = False
    
    def initialize(self, csv_path: str, index_dir: str = "indexes", checkpoint: str = "1095"):
        """Initialize the search engine with CSV data and FAISS index"""
        try:
            logger.info("🚀 Initializing Hybrid Search Engine...")
            logger.info(f"📊 Loading CSV data from: {csv_path}")
            logger.info(f"📁 Loading FAISS index from: {index_dir}")
            logger.info(f"🎯 Using checkpoint: {checkpoint}")
            
            # Load CSV data
            self.df = data_loader.load_csv(csv_path)
            
            # Load FAISS index
            if not data_loader.load_faiss_index(index_dir, checkpoint):
                return False
            
            # Load GME model with LoRA
            if not gme_model.load_model("gme-Qwen2-VL-7B-Instruct", checkpoint):
                logger.warning("Failed to load GME model, some features may not work")
            
            self.is_initialized = True
            logger.info("✅ Hybrid Search Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing search engine: {e}")
            return False
    
    def search_by_image_similarity(self, query_image_path: str, filters: Dict = None, top_k: int = 50) -> List[Dict]:
        """Search by image similarity with optional filters"""
        try:
            if not self.is_initialized:
                logger.error("Search engine not initialized")
                return []
            
            logger.info(f"🔍 Starting image similarity search")
            logger.info(f"📁 Query image: {query_image_path}")
            logger.info(f"🎯 Target results: {top_k}")
            if filters:
                logger.info(f"🔧 Filters applied: {list(filters.keys())}")
            
            # Encode query image
            logger.info("🤖 Encoding query image with GME model...")
            
            # Check if GME model is properly loaded
            if gme_model.model is None:
                logger.error("❌ GME model not loaded - cannot encode query image")
                return []
            
            query_embedding = gme_model.encode_image(query_image_path)
            if query_embedding is None:
                logger.error("❌ Could not encode query image")
                return []
            
            logger.info(f"✅ Image encoded successfully (dimension: {query_embedding.shape})")
            
            # Search in FAISS index
            logger.info("🔍 Searching in FAISS index...")
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            # Increase search pool significantly when filters are applied
            # With strict filters, we need a much larger initial pool
            search_k = min(top_k * 20, 1000) if filters else top_k  # Get many more if filtering
            logger.info(f"🔍 Searching for top {search_k} similar items (filters: {'yes' if filters else 'no'})")
            distances, indices = data_loader.index.search(query_embedding, search_k)
            
            logger.info(f"📊 Found {len(distances[0])} initial matches")
            
            # Get results
            logger.info("📋 Processing search results...")
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx in data_loader.idx_to_filename_root:
                    filename_root = data_loader.idx_to_filename_root[idx]
                    
                    # Get corresponding CSV row
                    csv_row = self.df[self.df['filename_root'] == filename_root]
                    if len(csv_row) > 0:
                        result = csv_row.iloc[0].to_dict()
                        result['similarity_score'] = float(distance)
                        result['image_path'] = self._get_image_path(filename_root)
                        results.append(result)
                        
                        if (i + 1) % 10 == 0:
                            logger.info(f"⚡ Processed {i + 1}/{len(distances[0])} matches...")
            
            logger.info(f"✅ Retrieved {len(results)} products from database")
            
            # Apply filters if provided
            if filters:
                logger.info("🔧 Applying filters...")
                before_filter = len(results)
                results = self._apply_filters(results, filters)
                logger.info(f"🎯 Filtering: {before_filter} → {len(results)} results")
            
            # Return top_k results
            final_results = results[:top_k]
            logger.info(f"🏆 Returning top {len(final_results)} results")
            
            if final_results:
                avg_similarity = sum(r.get('similarity_score', 0) for r in final_results) / len(final_results)
                logger.info(f"📈 Average similarity score: {avg_similarity:.4f}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Error in image similarity search: {e}")
            return []
    
    def search_by_filename_similarity(self, filename_root: str, filters: Dict = None, top_k: int = 50) -> List[Dict]:
        """Search by filename similarity using pre-computed embeddings"""
        try:
            if not self.is_initialized:
                logger.error("Search engine not initialized")
                return []
            
            # Find the index for this filename_root
            if filename_root not in data_loader.filename_to_idx:
                logger.error(f"Filename root {filename_root} not found in index")
                return []
            
            query_idx = data_loader.filename_to_idx[filename_root]
            query_embedding = data_loader.embeddings[query_idx].reshape(1, -1).astype(np.float32)
            
            # Search in FAISS index
            # Use larger search pool when filters might be applied downstream
            search_k = min(top_k * 20, 1000) if filters else top_k * 2
            distances, indices = data_loader.index.search(query_embedding, search_k)
            
            # Get results
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx in data_loader.idx_to_filename_root:
                    result_filename_root = data_loader.idx_to_filename_root[idx]
                    
                    # Skip self-match
                    if result_filename_root == filename_root:
                        continue
                    
                    # Get corresponding CSV row
                    csv_row = self.df[self.df['filename_root'] == result_filename_root]
                    if len(csv_row) > 0:
                        result = csv_row.iloc[0].to_dict()
                        result['similarity_score'] = float(distance)
                        result['image_path'] = self._get_image_path(result_filename_root)
                        results.append(result)
            
            # Apply filters if provided
            if filters:
                results = self._apply_filters(results, filters)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in filename similarity search: {e}")
            return []
    
    def search_by_sku(self, sku_query: str, top_k: int = 50) -> List[Dict]:
        """Search by SKU pattern matching"""
        try:
            if not self.is_initialized:
                logger.error("Search engine not initialized")
                return []
            
            logger.info(f"🔍 Starting SKU search")
            logger.info(f"🏷️  Query: '{sku_query}'")
            logger.info(f"🎯 Target results: {top_k}")
            
            # Clean and prepare search query
            sku_query = str(sku_query).strip().upper()
            logger.info(f"🧹 Cleaned query: '{sku_query}'")
            
            # Search strategies in order of priority
            search_results = []
            
            # 1. Exact SKU match
            logger.info("🎯 Strategy 1: Exact SKU match...")
            exact_matches = self.df[
                self.df['SKU_COD'].astype(str).str.upper() == sku_query
            ]
            if len(exact_matches) > 0:
                logger.info(f"✅ Found {len(exact_matches)} exact SKU matches")
                search_results.extend(exact_matches.to_dict('records'))
            else:
                logger.info("ℹ️  No exact SKU matches found")
            
            # 2. SKU contains pattern
            if len(search_results) < top_k:
                logger.info("🔍 Strategy 2: SKU contains pattern...")
                contains_matches = self.df[
                    self.df['SKU_COD'].astype(str).str.upper().str.contains(sku_query, na=False)
                ]
                # Remove exact matches to avoid duplicates
                if len(exact_matches) > 0:
                    contains_matches = contains_matches[~contains_matches.index.isin(exact_matches.index)]
                
                if len(contains_matches) > 0:
                    logger.info(f"✅ Found {len(contains_matches)} SKU contains matches")
                    search_results.extend(contains_matches.to_dict('records'))
                else:
                    logger.info("ℹ️  No SKU contains matches found")
            
            # 3. Model code match
            if len(search_results) < top_k:
                logger.info("🔍 Strategy 3: Model code match...")
                model_matches = self.df[
                    self.df['MODEL_COD'].astype(str).str.upper().str.contains(sku_query, na=False)
                ]
                # Remove previous matches
                existing_indices = [r.get('index', -1) for r in search_results if 'index' in r]
                if existing_indices:
                    model_matches = model_matches[~model_matches.index.isin(existing_indices)]
                
                if len(model_matches) > 0:
                    logger.info(f"✅ Found {len(model_matches)} model code matches")
                    search_results.extend(model_matches.to_dict('records'))
                else:
                    logger.info("ℹ️  No model code matches found")
            
            # 4. Filename root match
            if len(search_results) < top_k:
                logger.info("🔍 Strategy 4: Filename root match...")
                filename_matches = self.df[
                    self.df['filename_root'].astype(str).str.upper().str.contains(sku_query, na=False)
                ]
                # Remove previous matches
                existing_indices = [r.get('index', -1) for r in search_results if 'index' in r]
                if existing_indices:
                    filename_matches = filename_matches[~filename_matches.index.isin(existing_indices)]
                
                if len(filename_matches) > 0:
                    logger.info(f"✅ Found {len(filename_matches)} filename matches")
                    search_results.extend(filename_matches.to_dict('records'))
                else:
                    logger.info("ℹ️  No filename matches found")
            
            logger.info(f"📊 Total search results: {len(search_results)}")
            
            # Add image paths
            logger.info("🖼️  Adding image paths...")
            for result in search_results:
                if 'filename_root' in result:
                    result['image_path'] = self._get_image_path(result['filename_root'])
            
            # Return top results
            final_results = search_results[:top_k]
            logger.info(f"🏆 Returning top {len(final_results)} results")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Error in SKU search: {e}")
            return []
    
    def search_by_sku_list(self, sku_list: List[str]) -> List[Dict]:
        """Search by list of SKU codes (batch search)"""
        try:
            if not self.is_initialized:
                logger.error("Search engine not initialized")
                return []
            
            logger.info(f"🔍 Starting batch SKU search")
            logger.info(f"📦 Processing {len(sku_list)} SKUs")
            
            all_results = []
            found_count = 0
            not_found = []
            
            for i, sku in enumerate(sku_list):
                if (i + 1) % 10 == 0:
                    logger.info(f"⚡ Processing SKU {i + 1}/{len(sku_list)}...")
                
                results = self.search_by_sku(sku, top_k=5)  # Get top 5 for each SKU
                if results:
                    all_results.extend(results)
                    found_count += 1
                else:
                    not_found.append(sku)
            
            logger.info(f"✅ Batch search complete:")
            logger.info(f"   📊 Found: {found_count}/{len(sku_list)} SKUs")
            logger.info(f"   📋 Total results: {len(all_results)}")
            if not_found:
                logger.info(f"   ❌ Not found: {len(not_found)} SKUs")
                if len(not_found) <= 10:  # Show first few not found
                    logger.info(f"      Missing: {', '.join(not_found)}")
            
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Error in batch SKU search: {e}")
            return []
    
    def search_by_filters(self, filters: Dict, top_k: int = 50) -> List[Dict]:
        """Search by applying filters to the database"""
        try:
            if not self.is_initialized:
                logger.error("Search engine not initialized")
                return []
            
            logger.info(f"🔍 Starting filter search")
            logger.info(f"🔧 Filters: {list(filters.keys())}")
            logger.info(f"🎯 Target results: {top_k}")
            
            # Start with full dataset
            filtered_df = self.df.copy()
            logger.info(f"📊 Starting with {len(filtered_df)} products")
            
            # Apply each filter
            for filter_name, filter_value in filters.items():
                if filter_value and filter_name in filtered_df.columns:
                    before_count = len(filtered_df)
                    
                    if isinstance(filter_value, str):
                        # String filter - case insensitive contains
                        filtered_df = filtered_df[
                            filtered_df[filter_name].astype(str).str.upper().str.contains(
                                filter_value.upper(), na=False
                            )
                        ]
                    else:
                        # Exact match for non-string values
                        filtered_df = filtered_df[filtered_df[filter_name] == filter_value]
                    
                    after_count = len(filtered_df)
                    logger.info(f"   🔧 {filter_name} = '{filter_value}': {before_count} → {after_count}")
                    
                    if after_count == 0:
                        logger.warning(f"   ⚠️  No results after applying {filter_name} filter")
                        break
            
            logger.info(f"✅ Filtering complete: {len(filtered_df)} results")
            
            # Convert to list of dictionaries
            logger.info("📋 Converting to results format...")
            results = filtered_df.to_dict('records')
            
            # Add image paths
            logger.info("🖼️  Adding image paths...")
            for result in results:
                if 'filename_root' in result:
                    result['image_path'] = self._get_image_path(result['filename_root'])
            
            # Return top results
            final_results = results[:top_k]
            logger.info(f"🏆 Returning top {len(final_results)} results")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Error in filter search: {e}")
            return []
    
    def _apply_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """Apply filters to search results"""
        filtered_results = []
        
        for result in results:
            include = True
            for column, value in filters.items():
                if column in result and value is not None and value != '':
                    if isinstance(value, str):
                        if not str(result[column]).lower().find(value.lower()) >= 0:
                            include = False
                            break
                    else:
                        if result[column] != value:
                            include = False
                            break
            
            if include:
                filtered_results.append(result)
        
        return filtered_results
    
    def _get_image_path(self, filename_root: str) -> Optional[str]:
        """Get the actual image path for a filename_root"""
        if not filename_root:
            return None
        
        # Try both .jpg and .JPG extensions
        for ext in ['.jpg', '.JPG']:
            # Look for files starting with filename_root
            import glob
            pattern = f"pictures/{filename_root}_*{ext}"
            matches = glob.glob(pattern)
            if matches:
                logger.debug(f"🖼️  Found image for {filename_root}: {matches[0]}")
                return matches[0]  # Return first match
        
        # If no match found, try just the filename_root + extension
        for ext in ['.jpg', '.JPG']:
            import glob
            pattern = f"pictures/{filename_root}{ext}"
            matches = glob.glob(pattern)
            if matches:
                logger.debug(f"🖼️  Found direct image for {filename_root}: {matches[0]}")
                return matches[0]
        
        logger.warning(f"⚠️  No image found for filename_root: {filename_root}")
        return None
    
    def get_filter_options(self) -> Dict[str, List]:
        """Get unique values for each filterable column"""
        try:
            if not self.is_initialized or self.df is None:
                return {}
            
            filter_options = {}
            filter_columns = data_loader.get_filter_columns()
            
            for column in filter_columns:
                # Get unique non-null values
                unique_values = self.df[column].dropna().unique()
                # Convert to list and sort
                if len(unique_values) < 1000:  # Only for columns with reasonable number of options
                    filter_options[column] = sorted([str(val) for val in unique_values])
            
            return filter_options
            
        except Exception as e:
            logger.error(f"Error getting filter options: {e}")
            return {}
    
    def get_stats(self) -> Dict:
        """Get search engine statistics"""
        stats = {
            "initialized": self.is_initialized,
            "csv_rows": len(self.df) if self.df is not None else 0,
            "index_size": data_loader.index.ntotal if data_loader.index is not None else 0,
            "mapped_files": len(data_loader.filename_to_idx),
            "filter_columns": len(data_loader.get_filter_columns()) if self.df is not None else 0
        }
        
        # Add GPU memory info if available
        if gme_model.model is not None:
            stats["gpu_memory"] = gme_model.get_memory_usage()
        
        return stats

# Global instance
search_engine = HybridSearchEngine() 