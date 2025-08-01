import os
# Fix MKL threading conflicts in multiprocessing
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
import logging
import concurrent.futures
from threading import Lock
import torch
from collections import defaultdict
import time
import faiss
from optimized_faiss_search import OptimizedFAISSSearch
from parallel_filter_processor import ParallelFilterProcessor
import config_filtering

logger = logging.getLogger(__name__)

class ParallelBatchProcessor:
    """Batch processing with parallel filtering for faster searches"""
    
    def __init__(self, search_engine, data_loader, gme_model):
        self.search_engine = search_engine
        self.data_loader = data_loader
        self.gme_model = gme_model
        self.gpu_lock = Lock()
        
        # Initialize optimized search with proper filename mappings
        if not hasattr(data_loader, 'filename_to_idx') or not data_loader.filename_to_idx:
            logger.warning("⚠️ Data loader missing filename_to_idx mappings!")
            logger.warning("  Pre-filtering may not work correctly for SKU-based data")
        
        self.optimized_search = OptimizedFAISSSearch(
            index=data_loader.index,
            embeddings=data_loader.embeddings,
            metadata_df=data_loader.df,
            filename_to_idx=getattr(data_loader, 'filename_to_idx', {}),
            idx_to_filename=getattr(data_loader, 'idx_to_filename_root', {})
        )
        
        # Initialize parallel filter processor
        self.parallel_filter = ParallelFilterProcessor(
            df=data_loader.df,
            filename_to_idx=getattr(data_loader, 'filename_to_idx', {}),
            idx_to_filename=getattr(data_loader, 'idx_to_filename_root', {}),
            embeddings_count=len(data_loader.embeddings)
        )
        
    def process_image_groups_with_prefilter(self, image_groups: Dict[str, Dict], 
                                           matching_cols: List[str],
                                           max_results_per_sku: int = 50,
                                           exclude_same_model: bool = False,
                                           allowed_statuses: List[str] = None,
                                           group_unisex: bool = False,
                                           dual_engine_enabled: bool = False,
                                           batch_size: int = 8,
                                           main_weight: float = 0.7,
                                           measurement_weight: float = 0.3,
                                           search_mode: str = "global") -> List[Dict]:
        """
        Process multiple image groups with parallel pre-filtering for optimal performance
        """
        start_time = time.time()
        logger.info(f"🚀 Starting PARALLEL batch processing with PRE-FILTERING")
        logger.info(f"📊 Processing {len(image_groups)} unique images")
        logger.info(f"🔧 Matching columns: {matching_cols}")
        if dual_engine_enabled:
            logger.info(f"🔍 Search mode: {search_mode}")
            logger.info(f"⚖️ Weights: GME={main_weight:.1%}, Technical={measurement_weight:.1%}")
        
        # Define which columns to use for pre-filtering vs post-filtering
        prefilter_columns = config_filtering.get_prefilter_columns()
        prefilter_columns = [col for col in prefilter_columns if col in matching_cols]
        
        logger.info(f"🔍 Pre-filter columns (applied before similarity search): {prefilter_columns}")
        logger.info(f"📋 Post-filter columns (applied after similarity search): {[col for col in matching_cols if col not in prefilter_columns]}")
        
        # Prepare queries with their filters
        queries = []
        query_metadata = {}
        query_filters = []  # For parallel filtering
        
        for filename_root, group_data in image_groups.items():
            source_item = group_data['source_item']
            
            # Only use essential filters for pre-filtering
            prefilters = {}
            for col in prefilter_columns:
                if col in matching_cols and col in source_item and source_item[col] is not None:
                    prefilters[col] = source_item[col]
            
            # Handle gender filtering for unisex grouping
            if group_unisex and 'USERGENDER_DES' in prefilters:
                source_gender = prefilters['USERGENDER_DES']
                if source_gender in ['MAN', 'WOMAN']:
                    prefilters['USERGENDER_DES'] = [source_gender, 'UNISEX ADULT']
            
            # Get embedding
            embedding = self._get_embedding_for_filename(filename_root)
            if embedding is not None:
                query_id = f"query_{len(queries)}"
                queries.append((query_id, embedding, prefilters))
                query_filters.append((query_id, prefilters))
                query_metadata[query_id] = {
                    'filename_root': filename_root,
                    'group_data': group_data,
                    'source_item': source_item,
                    'exclude_model': source_item.get('MODEL_COD') if exclude_same_model else None,
                    'postfilter_columns': [col for col in matching_cols if col not in prefilter_columns],
                    'all_filters': {col: source_item.get(col) for col in matching_cols if col in source_item}
                }
        
        logger.info(f"📋 Prepared {len(queries)} queries for batch search")
        
        # PARALLEL FILTERING - This is the key optimization
        filter_start = time.time()
        filtered_indices_map = self.parallel_filter.parallel_filter_queries(query_filters)
        filter_time = time.time() - filter_start
        logger.info(f"⚡ Parallel filtering completed in {filter_time:.2f}s")
        
        # Log filter effectiveness
        total_filtered_products = sum(len(indices) for indices in filtered_indices_map.values())
        avg_products_per_query = total_filtered_products / len(queries) if queries else 0
        logger.info(f"📊 Average products per query after filtering: {avg_products_per_query:.1f}")
        
        # Perform batch search with pre-computed filtered indices
        logger.info("🔥 Starting batch FAISS search with PRE-COMPUTED filters...")
        
        if dual_engine_enabled:
            logger.info("🎭 Using DUAL INDEX search with parallel filtering!")
            # Use dual index search with pre-filtered indices
            search_results = self._batch_dual_index_search_parallel(
                queries, filtered_indices_map, max_results_per_sku * 3, query_metadata,
                main_weight=main_weight, measurement_weight=measurement_weight,
                search_mode=search_mode
            )
        else:
            # Use standard GME-only search with pre-computed filters
            search_results = self._batch_search_with_precomputed_filters(
                queries, filtered_indices_map, max_results_per_sku * 3
            )
        
        # Process results (same as original)
        all_results = []
        
        for query_id, result_indices in search_results.items():
            metadata = query_metadata[query_id]
            filename_root = metadata['filename_root']
            group_data = metadata['group_data']
            exclude_model = metadata['exclude_model']
            postfilter_columns = metadata['postfilter_columns']
            all_filters = metadata['all_filters']
            
            # Convert embedding indices to full results
            similar_items = []
            items_before_postfilter = 0
            items_filtered_out = 0
            
            # Process results (same logic as original)
            for i, item in enumerate(result_indices):
                if len(item) == 3:  # New format with scoring details
                    distance, embedding_idx, scoring_info = item
                else:  # Legacy format
                    distance, embedding_idx = item
                    scoring_info = None
                    
                if embedding_idx == -1 and scoring_info and 'filename_root' in scoring_info:
                    # Handle measurement-only products
                    similar_filename_root = scoring_info['filename_root']
                    
                    matching_rows = self.data_loader.df[
                        self.data_loader.df['filename_root'] == similar_filename_root
                    ]
                    
                    if matching_rows.empty:
                        # Create synthetic product info
                        item = {
                            'filename_root': similar_filename_root,
                            'SKU_COD': f'MEAS_{similar_filename_root}',
                            'similarity_score': distance
                        }
                        if scoring_info:
                            item['gme_score'] = scoring_info.get('main_similarity', 0.0)
                            item['technical_score'] = scoring_info.get('measurement_similarity', 0.0)
                            item['final_score'] = scoring_info.get('combined_score', 0.0)
                            item['search_source'] = scoring_info.get('source', 'unknown')
                            item['index_membership'] = scoring_info.get('index_membership', 'unknown')
                        similar_items.append(item)
                    else:
                        # Process normally if found in dataframe
                        for _, row in matching_rows.iterrows():
                            item = row.to_dict()
                            item['similarity_score'] = distance
                            if scoring_info:
                                item['gme_score'] = scoring_info.get('main_similarity', 0.0)
                                item['technical_score'] = scoring_info.get('measurement_similarity', 0.0)
                                item['final_score'] = scoring_info.get('combined_score', 0.0)
                                item['search_source'] = scoring_info.get('source', 'unknown')
                                item['index_membership'] = scoring_info.get('index_membership', 'unknown')
                            similar_items.append(item)
                elif embedding_idx >= 0:
                    # Process standard results
                    if scoring_info and 'filename_root' in scoring_info:
                        similar_filename_root = scoring_info['filename_root']
                    elif embedding_idx in self.data_loader.idx_to_filename_root:
                        similar_filename_root = self.data_loader.idx_to_filename_root[embedding_idx]
                    else:
                        continue
                    
                    matching_rows = self.data_loader.df[
                        self.data_loader.df['filename_root'] == similar_filename_root
                    ]
                    
                    if matching_rows.empty:
                        continue
                    
                    # Process each matching row
                    for _, row in matching_rows.iterrows():
                        item = row.to_dict()
                        item['similarity_score'] = distance
                        
                        if scoring_info:
                            item['gme_score'] = scoring_info.get('main_similarity', 0.0)
                            item['technical_score'] = scoring_info.get('measurement_similarity', 0.0)
                            item['final_score'] = scoring_info.get('combined_score', 0.0)
                            item['search_source'] = scoring_info.get('source', 'unknown')
                            item['index_membership'] = scoring_info.get('index_membership', 'unknown')
                        
                        # Apply model exclusion if needed
                        if exclude_model and item.get('MODEL_COD') == exclude_model:
                            continue
                        
                        items_before_postfilter += 1
                        
                        # Apply post-filters
                        skip_item = False
                        for col in postfilter_columns:
                            if col in all_filters and all_filters[col] is not None:
                                item_value = item.get(col)
                                filter_value = all_filters[col]
                                
                                if config_filtering.is_range_filter_column(col):
                                    # Range-based filtering
                                    min_val, max_val = config_filtering.get_range_bounds(filter_value, col)
                                    if min_val is not None and max_val is not None:
                                        try:
                                            if isinstance(item_value, str):
                                                item_value = item_value.replace(',', '.')
                                            item_numeric = float(item_value)
                                            
                                            if not (min_val <= item_numeric <= max_val):
                                                skip_item = True
                                                break
                                        except (ValueError, TypeError):
                                            skip_item = True
                                            break
                                else:
                                    # Text comparison
                                    if pd.isna(item_value) and pd.isna(filter_value):
                                        continue
                                    elif pd.isna(item_value) or pd.isna(filter_value):
                                        skip_item = True
                                        break
                                    else:
                                        item_str = str(item_value).strip().upper()
                                        filter_str = str(filter_value).strip().upper()
                                        if item_str != filter_str:
                                            skip_item = True
                                            break
                        
                        if not skip_item:
                            similar_items.append(item)
                        else:
                            items_filtered_out += 1
                        
                        if len(similar_items) >= max_results_per_sku:
                            break
                
                if len(similar_items) >= max_results_per_sku:
                    break
            
            # Format results for all SKUs in this group
            for input_sku in group_data['skus']:
                for similar_item in similar_items:
                    result_row = {
                        'Input_SKU': input_sku,
                        'Similar_SKU': similar_item.get('SKU_COD', ''),
                        'Similarity_Score': round(1 - similar_item.get('similarity_score', 0), 3)
                    }
                    
                    # Add scoring details if dual engine was used
                    if dual_engine_enabled and 'gme_score' in similar_item:
                        result_row.update({
                            'GME_Score': round(similar_item.get('gme_score', 0), 3),
                            'Technical_Score': round(similar_item.get('technical_score', 0), 3),
                            'Final_Score': round(similar_item.get('final_score', 0), 3),
                            'Score_Formula': f"{main_weight:.0%}×GME + {measurement_weight:.0%}×Technical",
                            'Search_Source': similar_item.get('search_source', 'unknown'),
                            'Index_Membership': similar_item.get('index_membership', 'unknown')
                        })
                    
                    # Add ALL columns from source and similar items
                    for col, value in group_data['source_item'].items():
                        if col not in ['Input_SKU', 'Similar_SKU', 'Similarity_Score']:
                            result_row[f'Source_{col}'] = value
                    
                    for col, value in similar_item.items():
                        if col not in ['similarity_score', 'SKU_COD']:
                            result_row[f'Similar_{col}'] = value
                    
                    all_results.append(result_row)
        
        # Sort results
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df = results_df.sort_values(
                by=['Input_SKU', 'Similarity_Score'], 
                ascending=[True, False]
            )
            all_results = results_df.to_dict('records')
        
        # Performance summary
        total_time = time.time() - start_time
        logger.info(f"✅ Parallel batch processing complete in {total_time:.1f} seconds")
        logger.info(f"⚡ Performance: {len(image_groups)/total_time:.1f} images/sec")
        logger.info(f"📊 Total results: {len(all_results)}")
        logger.info(f"🔧 Breakdown: Filtering={filter_time:.1f}s, Search={(total_time-filter_time):.1f}s")
        
        return all_results
    
    def _batch_search_with_precomputed_filters(self, queries: List[Tuple], 
                                              filtered_indices_map: Dict[str, np.ndarray],
                                              top_k: int) -> Dict[str, List[Tuple]]:
        """Perform batch search using pre-computed filtered indices"""
        results = {}
        
        for query_id, embedding, _ in queries:
            filtered_indices = filtered_indices_map.get(query_id, np.array([]))
            
            if len(filtered_indices) == 0:
                results[query_id] = []
                continue
            
            # Perform search on filtered subset
            if len(filtered_indices) < top_k:
                # If we have fewer items than requested, search them all
                subset_embeddings = self.data_loader.embeddings[filtered_indices]
                distances = np.linalg.norm(subset_embeddings - embedding.reshape(1, -1), axis=1)
                sorted_idx = np.argsort(distances)
                
                results[query_id] = [
                    (distances[i], filtered_indices[i]) 
                    for i in sorted_idx
                ]
            else:
                # Create temporary index for large filtered sets
                subset_embeddings = self.data_loader.embeddings[filtered_indices].astype(np.float32)
                temp_index = faiss.IndexFlatL2(subset_embeddings.shape[1])
                temp_index.add(subset_embeddings)
                
                query_emb = embedding.reshape(1, -1).astype(np.float32)
                distances, indices = temp_index.search(query_emb, min(top_k, len(filtered_indices)))
                
                results[query_id] = [
                    (distances[0][i], filtered_indices[indices[0][i]]) 
                    for i in range(len(indices[0]))
                ]
        
        return results
    
    def _batch_dual_index_search_parallel(self, queries: List[Tuple], 
                                        filtered_indices_map: Dict[str, np.ndarray],
                                        top_k: int, query_metadata: Dict, 
                                        main_weight: float = 0.7, 
                                        measurement_weight: float = 0.3,
                                        search_mode: str = "global") -> Dict[str, List[Tuple[float, int, Dict]]]:
        """
        Perform dual index search with pre-computed filtered indices
        """
        from dual_index_data_loader import dual_index_loader
        from dual_index_search_all import search_all_filtered_products
        
        logger.info("🔍 Performing PARALLEL dual index search...")
        
        results = {}
        
        for query_id, gme_embedding, filters in queries:
            try:
                filename_root = query_metadata[query_id]['filename_root']
                measurement_embedding = dual_index_loader.get_measurement_embedding_by_filename(filename_root)
                
                # In filtered mode, use pre-computed indices for more efficient search
                if search_mode == "filtered" and query_id in filtered_indices_map:
                    filtered_indices = filtered_indices_map[query_id]
                    logger.debug(f"Using {len(filtered_indices)} pre-filtered indices for {filename_root}")
                
                # Use the standard dual search function
                all_results = search_all_filtered_products(
                    gme_embedding, 
                    measurement_embedding,
                    filters,
                    search_mode
                )
                
                # Filter and process results
                exclude_model = query_metadata[query_id].get('exclude_model', None)
                if exclude_model:
                    all_results = [r for r in all_results if not (
                        r['filename_root'] in self.data_loader.filename_to_idx and
                        self.data_loader.df[self.data_loader.df['filename_root'] == r['filename_root']]['MODEL_COD'].iloc[0] == exclude_model
                    )]
                
                # Take only top_k results
                combined_results = all_results[:top_k]
                
                # Convert to batch processor format
                query_results = []
                for result in combined_results:
                    result_filename = result.get('filename_root', '')
                    
                    if result_filename and result_filename in self.data_loader.filename_to_idx:
                        embedding_idx = self.data_loader.filename_to_idx[result_filename]
                    else:
                        embedding_idx = -1
                    
                    query_results.append((
                        1.0 - result.get('combined_score', 0.0),
                        embedding_idx,
                        {
                            'main_similarity': result.get('main_similarity', 0.0),
                            'measurement_similarity': result.get('measurement_similarity', 0.0),
                            'combined_score': result.get('combined_score', 0.0),
                            'source': result.get('score_source', 'unknown'),
                            'index_membership': result.get('index_membership', 'unknown'),
                            'filename_root': result_filename
                        }
                    ))
                
                results[query_id] = query_results
                
            except Exception as e:
                logger.warning(f"Error in dual index search for {query_id}: {e}")
                results[query_id] = []
        
        return results
    
    def _get_embedding_for_filename(self, filename_root: str) -> Optional[np.ndarray]:
        """Get pre-computed embedding for filename"""
        variations = [
            filename_root,
            filename_root.lower(),
            filename_root.upper(),
        ]
        
        if filename_root and filename_root[-1].isalpha():
            variations.extend([
                filename_root[:-1],
                filename_root[:-1].lower()
            ])
        
        for variant in variations:
            if variant in self.data_loader.filename_to_idx:
                idx = self.data_loader.filename_to_idx[variant]
                return self.data_loader.embeddings[idx]
        
        logger.warning(f"⚠️ No embedding found for {filename_root}")
        return None
    
    def clear_filter_cache(self):
        """Clear the filter cache to free memory"""
        self.optimized_search.clear_cache()
        logger.info("🧹 Filter cache cleared")