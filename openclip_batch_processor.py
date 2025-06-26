import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import json
from openclip_search_engine import openclip_search_engine
from openclip_data_loader import openclip_data_loader

logger = logging.getLogger(__name__)

class OpenCLIPBatchProcessor:
    """Batch processor for OpenCLIP with parallel GPU processing"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers  # Use all 4 GPUs
        self.thread_local = threading.local()
        
        # Initialize optimized search for pre-filtering
        from optimized_faiss_search import OptimizedFAISSSearch
        self.optimized_search = OptimizedFAISSSearch(
            index=openclip_data_loader.index,
            embeddings=openclip_data_loader.embeddings,
            metadata_df=openclip_data_loader.df,
            filename_to_idx=openclip_data_loader.filename_to_idx,
            idx_to_filename=openclip_data_loader.idx_to_filename_root
        )
        
    def process_batch_search(
        self,
        sku_list: List[str],
        matching_columns: List[str],
        max_results_per_sku: int = 50,
        exclude_same_model: bool = False,
        allowed_status_codes: List[str] = None,
        group_unisex: bool = False,
        search_pool_size: int = 1000
    ) -> List[Dict]:
        """Process batch search with parallel GPU processing"""
        
        start_time = time.time()
        logger.info(f"🚀 Starting batch processing with {self.max_workers} workers")
        logger.info(f"📦 Processing {len(sku_list)} SKUs")
        
        # Step 1: Bulk SKU lookup - Get all source items at once
        logger.info("📋 Step 1: Bulk SKU lookup...")
        all_source_items = openclip_data_loader.search_by_sku_list(sku_list)
        
        # Create mappings
        sku_to_source = {}
        unique_filename_roots = set()
        sku_to_filename_root = {}
        
        for item in all_source_items:
            sku = item.get('SKU_COD')
            if sku:
                sku_to_source[sku] = item
                filename_root = item.get('filename_root')
                if filename_root:
                    unique_filename_roots.add(filename_root)
                    sku_to_filename_root[sku] = filename_root
        
        logger.info(f"✅ Found {len(sku_to_source)} source items")
        logger.info(f"🖼️  Found {len(unique_filename_roots)} unique images to search")
        
        # Step 2: Batch encode all unique images in parallel
        logger.info("🎨 Step 2: Batch similarity search with GPU parallelization...")
        
        # Get pre-computed embeddings for all unique filename_roots
        filename_roots_list = list(unique_filename_roots)
        embeddings_dict = self._get_embeddings_batch(filename_roots_list)
        
        # Step 3: Parallel similarity search using multiple GPUs
        all_similarities = {}
        
        # Split work across GPUs
        chunks = [filename_roots_list[i:i+len(filename_roots_list)//self.max_workers+1] 
                 for i in range(0, len(filename_roots_list), len(filename_roots_list)//self.max_workers+1)]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {}
            
            for gpu_id, chunk in enumerate(chunks[:self.max_workers]):
                if chunk:
                    future = executor.submit(
                        self._process_similarity_chunk,
                        chunk,
                        embeddings_dict,
                        search_pool_size,
                        max_results_per_sku * 5,  # Get more for filtering
                        gpu_id % 4  # Assign to specific GPU
                    )
                    future_to_chunk[future] = chunk
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_chunk):
                chunk_results = future.result()
                all_similarities.update(chunk_results)
                completed += len(chunk_results)
                if completed % 100 == 0:
                    logger.info(f"⚡ Processed {completed}/{len(unique_filename_roots)} similarity searches")
        
        logger.info(f"✅ Completed all similarity searches")
        
        # Step 4: Apply filters and compile results
        logger.info("🔧 Step 4: Applying filters and compiling results...")
        all_results = []
        processed_count = 0
        
        for idx, sku in enumerate(sku_list):
            if sku not in sku_to_source:
                continue
                
            source_item = sku_to_source[sku]
            filename_root = sku_to_filename_root.get(sku)
            
            if not filename_root or filename_root not in all_similarities:
                continue
            
            similar_results = all_similarities[filename_root]
            
            # Apply filters
            filtered_results = self._apply_filters(
                similar_results, 
                source_item,
                matching_columns,
                exclude_same_model,
                allowed_status_codes
            )
            
            # Add results
            for result in filtered_results[:max_results_per_sku]:
                all_results.append({
                    'input_sku': sku,
                    'similar_sku': result.get('SKU_COD'),
                    'similarity_score': result.get('similarity_score', 0),
                    'model_cod': result.get('MODEL_COD'),
                    'status': result.get('MD_SKU_STATUS_COD'),
                    **{col: result.get(col) for col in matching_columns}
                })
            
            processed_count += 1
            
            if (idx + 1) % 500 == 0:
                logger.info(f"📊 Processed results for {idx + 1}/{len(sku_list)} SKUs")
        
        # Log performance stats
        total_time = time.time() - start_time
        logger.info(f"✅ Batch processing completed in {total_time:.2f}s")
        logger.info(f"⚡ Processing speed: {len(sku_list)/total_time:.1f} SKUs/second")
        logger.info(f"📊 Results summary:")
        logger.info(f"   - Processed: {processed_count}/{len(sku_list)} SKUs")
        logger.info(f"   - Unique images: {len(unique_filename_roots)}")
        logger.info(f"   - Total results: {len(all_results)}")
        
        return all_results
    
    def _get_embeddings_batch(self, filename_roots: List[str]) -> Dict[str, np.ndarray]:
        """Get pre-computed embeddings for a batch of filename_roots"""
        embeddings_dict = {}
        
        for filename_root in filename_roots:
            # Try different variations to find the embedding
            variations = [filename_root, filename_root.lower(), filename_root.upper()]
            
            for variation in variations:
                if variation in openclip_data_loader.filename_to_idx:
                    idx = openclip_data_loader.filename_to_idx[variation]
                    embedding = openclip_data_loader.embeddings[idx]
                    embeddings_dict[filename_root] = embedding
                    break
                elif variation in openclip_data_loader.filename_mappings:
                    mapped_root = openclip_data_loader.filename_mappings[variation]
                    if mapped_root in openclip_data_loader.filename_to_idx:
                        idx = openclip_data_loader.filename_to_idx[mapped_root]
                        embedding = openclip_data_loader.embeddings[idx]
                        embeddings_dict[filename_root] = embedding
                        break
        
        return embeddings_dict
    
    def _process_similarity_chunk(
        self, 
        filename_roots: List[str],
        embeddings_dict: Dict[str, np.ndarray],
        search_pool_size: int,
        top_k: int,
        gpu_id: int
    ) -> Dict[str, List[Dict]]:
        """Process a chunk of similarity searches on a specific GPU"""
        results = {}
        
        # Note: In a real multi-GPU setup, we'd set the GPU here
        # For now, FAISS will use the GPU it was initialized with
        
        for filename_root in filename_roots:
            if filename_root in embeddings_dict:
                # Use the pre-computed embedding for fast search
                query_embedding = embeddings_dict[filename_root].reshape(1, -1).astype(np.float32)
                
                # Search in FAISS index
                actual_search_k = min(search_pool_size, openclip_data_loader.index.ntotal)
                similarities, indices = openclip_data_loader.index.search(query_embedding, actual_search_k)
                
                # Convert to results format
                similar_items = []
                for similarity, idx in zip(similarities[0], indices[0]):
                    if idx in openclip_data_loader.idx_to_filename_root:
                        result_filename_root = openclip_data_loader.idx_to_filename_root[idx]
                        
                        # Skip self-match
                        if result_filename_root == filename_root:
                            continue
                        
                        # Get corresponding CSV row
                        csv_row = openclip_data_loader.df[
                            openclip_data_loader.df['filename_root'] == result_filename_root
                        ]
                        if len(csv_row) > 0:
                            result = csv_row.iloc[0].to_dict()
                            result['similarity_score'] = float(similarity)
                            similar_items.append(result)
                            
                            if len(similar_items) >= top_k:
                                break
                
                results[filename_root] = similar_items
        
        return results
    
    def _apply_filters(
        self,
        similar_results: List[Dict],
        source_item: Dict,
        matching_columns: List[str],
        exclude_same_model: bool,
        allowed_status_codes: List[str]
    ) -> List[Dict]:
        """Apply filters to search results"""
        filtered_results = []
        
        for similar_item in similar_results:
            # Skip same model code if requested
            if exclude_same_model:
                if similar_item.get('MODEL_COD') == source_item.get('MODEL_COD'):
                    continue
            
            # Check status code
            if allowed_status_codes:
                if similar_item.get('MD_SKU_STATUS_COD') not in allowed_status_codes:
                    continue
            
            # Apply matching columns
            match = True
            for col in matching_columns:
                if col in source_item and col in similar_item:
                    if source_item[col] != similar_item[col]:
                        match = False
                        break
            
            if match:
                filtered_results.append(similar_item)
        
        return filtered_results

    def process_batch_search_with_prefilter(
        self,
        sku_list: List[str],
        matching_columns: List[str],
        max_results_per_sku: int = 50,
        exclude_same_model: bool = False,
        allowed_status_codes: List[str] = None,
        group_unisex: bool = False,
        search_pool_size: int = 1000
    ) -> List[Dict]:
        """Process batch search with pre-filtering for optimal performance"""
        
        start_time = time.time()
        logger.info(f"🚀 Starting OpenCLIP batch processing with PRE-FILTERING")
        logger.info(f"📦 Processing {len(sku_list)} SKUs")
        
        # Step 1: Bulk SKU lookup - Get all source items at once
        logger.info("📋 Step 1: Bulk SKU lookup...")
        all_source_items = openclip_data_loader.search_by_sku_list(sku_list)
        
        # Create mappings and prepare queries
        sku_to_source = {}
        queries = []
        query_metadata = {}
        
        for item in all_source_items:
            sku = item.get('SKU_COD')
            if sku:
                sku_to_source[sku] = item
                filename_root = item.get('filename_root')
                
                if filename_root:
                    # Build filters for this query
                    filters = {}
                    for col in matching_columns:
                        if col in item and item[col] is not None:
                            filters[col] = item[col]
                    
                    # Add status filter
                    if allowed_status_codes:
                        filters['MD_SKU_STATUS_COD'] = allowed_status_codes
                    
                    # Handle gender filtering
                    if group_unisex and 'USERGENDER_DES' in filters:
                        source_gender = filters['USERGENDER_DES']
                        if source_gender in ['MAN', 'WOMAN']:
                            filters['USERGENDER_DES'] = [source_gender, 'UNISEX ADULT']
                    
                    # Get embedding
                    embedding = self._get_embeddings_batch([filename_root]).get(filename_root)
                    
                    if embedding is not None:
                        query_id = f"query_{sku}"
                        queries.append((query_id, embedding, filters))
                        query_metadata[query_id] = {
                            'sku': sku,
                            'source_item': item,
                            'exclude_model': item.get('MODEL_COD') if exclude_same_model else None
                        }
        
        logger.info(f"✅ Found {len(sku_to_source)} source items")
        logger.info(f"📋 Prepared {len(queries)} queries for batch search")
        
        # Log filter effectiveness
        unique_filters = {}
        for _, _, filters in queries:
            filter_key = json.dumps(filters, sort_keys=True)
            unique_filters[filter_key] = filters
        
        logger.info(f"🔍 Found {len(unique_filters)} unique filter combinations")
        
        # Show effectiveness of first few filter groups
        for i, (filter_key, filters) in enumerate(unique_filters.items()):
            if i < 3:
                filtered_indices = self.optimized_search.get_filtered_indices(filters)
                logger.info(f"   Filter group {i+1}: {len(filtered_indices)}/{self.optimized_search.total_items} items pass filters ({len(filtered_indices)/self.optimized_search.total_items*100:.1f}%)")
        
        # Perform batch search with pre-filtering
        logger.info("🔥 Starting batch FAISS search with PRE-FILTERING...")
        search_results = self.optimized_search.batch_search_with_prefilter(
            query_embeddings=queries,
            top_k=max_results_per_sku * 3,  # Get extra for post-filtering
            max_workers=self.max_workers
        )
        
        # Process results
        all_results = []
        
        for query_id, result_indices in search_results.items():
            metadata = query_metadata[query_id]
            sku = metadata['sku']
            source_item = metadata['source_item']
            exclude_model = metadata['exclude_model']
            
            # Convert embedding indices to full results
            for distance, embedding_idx in result_indices:
                if embedding_idx >= 0:
                    # Convert embedding index back to filename_root
                    if embedding_idx in openclip_data_loader.idx_to_filename_root:
                        similar_filename_root = openclip_data_loader.idx_to_filename_root[embedding_idx]
                        
                        # Find all rows in DataFrame with this filename_root
                        matching_rows = openclip_data_loader.df[
                            openclip_data_loader.df['filename_root'] == similar_filename_root
                        ]
                        
                        # Process each matching row (multiple SKUs can have same filename_root)
                        for _, row in matching_rows.iterrows():
                            similar_item = row.to_dict()
                            
                            # Apply model exclusion
                            if exclude_model and similar_item.get('MODEL_COD') == exclude_model:
                                continue
                            
                            all_results.append({
                                'input_sku': sku,
                                'similar_sku': similar_item.get('SKU_COD'),
                                'similarity_score': float(distance),
                                'model_cod': similar_item.get('MODEL_COD'),
                                'status': similar_item.get('MD_SKU_STATUS_COD'),
                                **{col: similar_item.get(col) for col in matching_columns}
                            })
                            
                            if len([r for r in all_results if r['input_sku'] == sku]) >= max_results_per_sku:
                                break
                    else:
                        logger.warning(f"Embedding index {embedding_idx} not found in idx_to_filename_root mapping")
                    
                    if len([r for r in all_results if r['input_sku'] == sku]) >= max_results_per_sku:
                        break
        
        # Sort results by input_sku and then by similarity_score (descending)
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df = results_df.sort_values(
                by=['input_sku', 'similarity_score'], 
                ascending=[True, False]
            )
            all_results = results_df.to_dict('records')
        
        # Performance summary
        total_time = time.time() - start_time
        processed_count = len(set([r['input_sku'] for r in all_results]))
        
        logger.info(f"✅ OpenCLIP batch processing with pre-filtering completed in {total_time:.2f}s")
        logger.info(f"⚡ Processing speed: {len(sku_list)/total_time:.1f} SKUs/second")
        logger.info(f"📊 Results summary:")
        logger.info(f"   - Processed: {processed_count}/{len(sku_list)} SKUs")
        logger.info(f"   - Total results: {len(all_results)}")
        
        # Show cache stats
        cache_stats = self.optimized_search.get_cache_stats()
        logger.info(f"💾 Filter cache: {cache_stats['cache_size']} entries cached")
        
        return all_results

# Global instance
openclip_batch_processor = OpenCLIPBatchProcessor(max_workers=4) 