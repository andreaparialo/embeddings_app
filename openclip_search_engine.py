import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import os
from openclip_data_loader import openclip_data_loader
from openclip_model import openclip_model

logger = logging.getLogger(__name__)

class OpenCLIPSearchEngine:
    def __init__(self):
        self.is_initialized = False
    
    def initialize(self, csv_path: str, index_dir: str = "indexes/openclip", checkpoint: str = "epoch_008_model.pth"):
        """Initialize the OpenCLIP search engine"""
        try:
            # Load CSV data
            logger.info("Loading CSV data...")
            openclip_data_loader.load_csv(csv_path)
            
            # Load FAISS index (if exists) or create new one
            logger.info("Loading FAISS index...")
            index_loaded = openclip_data_loader.load_faiss_index(index_dir)
            
            if not index_loaded:
                # Try loading from NPZ format
                npz_path = os.path.join(index_dir, "openclip_embeddings.npz")
                if os.path.exists(npz_path):
                    logger.info("Loading from NPZ format...")
                    index_loaded = openclip_data_loader.load_npz_embeddings(npz_path)
                
                if not index_loaded:
                    logger.warning("No pre-computed index found. You need to create embeddings first.")
                    # Don't fail initialization, just warn
            
            # Load OpenCLIP model
            logger.info("Loading OpenCLIP model...")
            if not openclip_model.load_model(checkpoint):
                logger.warning("Could not load OpenCLIP model - image search may not work")
            
            self.is_initialized = True
            logger.info("OpenCLIP search engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing OpenCLIP search engine: {e}")
            return False
    
    def search_by_image_similarity(self, query_image_path: str, filters: Dict = None, top_k: int = 50, search_pool_size: int = 1000) -> List[Dict]:
        """Search by image similarity with optional filters"""
        try:
            if not self.is_initialized:
                logger.error("Search engine not initialized")
                return []
            
            logger.info(f"🔍 Starting image similarity search with OpenCLIP")
            logger.info(f"📁 Query image: {query_image_path}")
            logger.info(f"🎯 Target results: {top_k}")
            logger.info(f"🏊 Search pool size: {search_pool_size}")
            if filters:
                logger.info(f"🔧 Filters applied: {list(filters.keys())}")
            
            # Check if index is loaded
            if openclip_data_loader.index is None:
                logger.error("❌ No embeddings index loaded - cannot perform similarity search")
                return []
            
            # Encode query image
            logger.info("🤖 Encoding query image with OpenCLIP model...")
            
            # Check if OpenCLIP model is properly loaded
            if openclip_model.model is None:
                logger.error("❌ OpenCLIP model not loaded - cannot encode query image")
                return []
            
            query_embedding = openclip_model.encode_image(query_image_path)
            if query_embedding is None:
                logger.error("❌ Could not encode query image")
                return []
            
            logger.info(f"✅ Image encoded successfully (dimension: {query_embedding.shape})")
            
            # Search in FAISS index using cosine similarity (dot product since embeddings are normalized)
            logger.info("🔍 Searching in FAISS index...")
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            
            # Use search_pool_size for initial search
            actual_search_k = min(search_pool_size, openclip_data_loader.index.ntotal)
            
            # For cosine similarity, higher scores are better
            similarities, indices = openclip_data_loader.index.search(query_embedding, actual_search_k)
            
            logger.info(f"📊 Found {len(similarities[0])} initial matches (searched top {actual_search_k})")
            
            # Get results
            logger.info("📋 Processing search results...")
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx in openclip_data_loader.idx_to_filename_root:
                    filename_root = openclip_data_loader.idx_to_filename_root[idx]
                    
                    # Get corresponding CSV row
                    csv_row = openclip_data_loader.df[openclip_data_loader.df['filename_root'] == filename_root]
                    if len(csv_row) > 0:
                        result = csv_row.iloc[0].to_dict()
                        # For OpenCLIP, similarity is already a score between -1 and 1
                        result['similarity_score'] = float(similarity)
                        result['image_path'] = self._get_image_path(filename_root)
                        results.append(result)
                        
                        if (i + 1) % 100 == 0:
                            logger.info(f"⚡ Processed {i + 1}/{len(similarities[0])} matches...")
            
            logger.info(f"✅ Retrieved {len(results)} products from database")
            
            # Apply filters if provided
            if filters:
                logger.info("🔧 Applying filters...")
                before_filter = len(results)
                results = self._apply_filters(results, filters)
                logger.info(f"🎯 Filtering: {before_filter} → {len(results)} results ({len(results)/before_filter*100:.1f}% pass rate)")
                
                # Log detailed filter effectiveness
                if len(results) < top_k and before_filter > 0:
                    logger.warning(f"⚠️ Only {len(results)} results passed filters (requested {top_k})")
                    logger.info(f"📊 Filter pass rate: {len(results)/before_filter*100:.1f}%")
                    logger.info(f"💡 Consider increasing search_pool_size (current: {search_pool_size}) or relaxing filters")
            
            # Return top_k results
            final_results = results[:top_k]
            logger.info(f"🏆 Returning top {len(final_results)} results")
            
            if final_results:
                avg_similarity = sum(r.get('similarity_score', 0) for r in final_results) / len(final_results)
                logger.info(f"📈 Average similarity score: {avg_similarity:.4f}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Error in image similarity search: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def search_by_filename_similarity(self, filename_root: str, filters: Dict = None, top_k: int = 50, search_pool_size: int = 1000) -> List[Dict]:
        """Search by filename similarity using pre-computed embeddings"""
        try:
            if not self.is_initialized:
                logger.error("Search engine not initialized")
                return []
            
            # Handle case variations
            filename_variations = [
                filename_root,
                filename_root.lower(),
                filename_root.upper()
            ]
            
            # Find the index for this filename_root
            query_idx = None
            actual_filename_root = None
            
            for variation in filename_variations:
                if variation in openclip_data_loader.filename_to_idx:
                    query_idx = openclip_data_loader.filename_to_idx[variation]
                    actual_filename_root = variation
                    break
                # Check if it's in the filename mappings
                elif variation in openclip_data_loader.filename_mappings:
                    mapped_root = openclip_data_loader.filename_mappings[variation]
                    if mapped_root in openclip_data_loader.filename_to_idx:
                        query_idx = openclip_data_loader.filename_to_idx[mapped_root]
                        actual_filename_root = mapped_root
                        break
            
            if query_idx is None:
                logger.error(f"Filename root {filename_root} (or variations) not found in index")
                return []
            
            query_embedding = openclip_data_loader.embeddings[query_idx].reshape(1, -1).astype(np.float32)
            
            # Use search_pool_size for initial search
            actual_search_k = min(search_pool_size, openclip_data_loader.index.ntotal)
            
            # Search in FAISS index
            similarities, indices = openclip_data_loader.index.search(query_embedding, actual_search_k)
            
            # Get results
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx in openclip_data_loader.idx_to_filename_root:
                    result_filename_root = openclip_data_loader.idx_to_filename_root[idx]
                    
                    # Skip self-match
                    if result_filename_root == actual_filename_root:
                        continue
                    
                    # Get corresponding CSV row
                    csv_row = openclip_data_loader.df[openclip_data_loader.df['filename_root'] == result_filename_root]
                    if len(csv_row) > 0:
                        result = csv_row.iloc[0].to_dict()
                        result['similarity_score'] = float(similarity)
                        result['image_path'] = self._get_image_path(result_filename_root)
                        results.append(result)
            
            # Apply filters if provided
            if filters:
                before_filter = len(results)
                results = self._apply_filters(results, filters)
                logger.info(f"🎯 Filtering: {before_filter} → {len(results)} results ({len(results)/before_filter*100:.1f}% pass rate)")
            
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
            exact_matches = openclip_data_loader.df[
                openclip_data_loader.df['SKU_COD'].astype(str).str.upper() == sku_query
            ]
            if len(exact_matches) > 0:
                logger.info(f"✅ Found {len(exact_matches)} exact SKU matches")
                search_results.extend(exact_matches.to_dict('records'))
            else:
                logger.info("ℹ️  No exact SKU matches found")
            
            # 2. SKU contains pattern
            if len(search_results) < top_k:
                logger.info("🔍 Strategy 2: SKU contains pattern...")
                contains_matches = openclip_data_loader.df[
                    openclip_data_loader.df['SKU_COD'].astype(str).str.upper().str.contains(sku_query, na=False)
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
                model_matches = openclip_data_loader.df[
                    openclip_data_loader.df['MODEL_COD'].astype(str).str.upper().str.contains(sku_query, na=False)
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
                filename_matches = openclip_data_loader.df[
                    openclip_data_loader.df['filename_root'].astype(str).str.upper().str.contains(sku_query, na=False)
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
        """Search by list of SKU codes (batch search) using bulk operations"""
        try:
            if not self.is_initialized:
                logger.error("Search engine not initialized")
                return []
            
            logger.info(f"🔍 Starting batch SKU search with bulk operations")
            logger.info(f"📦 Processing {len(sku_list)} SKUs")
            
            # Use bulk search from data loader
            results = openclip_data_loader.search_by_sku_list(sku_list)
            
            # Add image paths
            for result in results:
                if 'filename_root' in result:
                    result['image_path'] = self._get_image_path(result['filename_root'])
            
            logger.info(f"✅ Batch search complete:")
            logger.info(f"   📊 Found: {len(results)} results from {len(sku_list)} SKUs")
            
            return results
            
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
            filtered_df = openclip_data_loader.df.copy()
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
        
        # Get the parent directory's pictures folder
        pictures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pictures")
        
        # Try both .jpg and .JPG extensions
        for ext in ['.jpg', '.JPG']:
            # Look for files starting with filename_root
            import glob
            pattern = os.path.join(pictures_dir, f"{filename_root}_*{ext}")
            matches = glob.glob(pattern)
            if matches:
                # Return web path
                basename = os.path.basename(matches[0])
                logger.debug(f"🖼️  Found image for {filename_root}: {basename}")
                return f"/pictures/{basename}"  # Return web path
        
        # If no match found, try just the filename_root + extension
        for ext in ['.jpg', '.JPG']:
            import glob
            pattern = os.path.join(pictures_dir, f"{filename_root}{ext}")
            matches = glob.glob(pattern)
            if matches:
                basename = os.path.basename(matches[0])
                logger.debug(f"🖼️  Found direct image for {filename_root}: {basename}")
                return f"/pictures/{basename}"
        
        logger.warning(f"⚠️  No image found for filename_root: {filename_root}")
        return None
    
    def get_filter_options(self) -> Dict[str, List]:
        """Get unique values for each filterable column"""
        try:
            if not self.is_initialized or openclip_data_loader.df is None:
                return {}
            
            filter_options = {}
            filter_columns = openclip_data_loader.get_filter_columns()
            
            for column in filter_columns:
                # Get unique non-null values
                unique_values = openclip_data_loader.df[column].dropna().unique()
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
            "csv_rows": len(openclip_data_loader.df) if openclip_data_loader.df is not None else 0,
            "index_size": openclip_data_loader.index.ntotal if openclip_data_loader.index is not None else 0,
            "mapped_files": len(openclip_data_loader.filename_to_idx),
            "filter_columns": len(openclip_data_loader.get_filter_columns()) if openclip_data_loader.df is not None else 0
        }
        
        # Add GPU memory info if available
        if openclip_model.model is not None:
            stats["gpu_memory"] = openclip_model.get_memory_usage()
        
        return stats

# Global instance
openclip_search_engine = OpenCLIPSearchEngine() 