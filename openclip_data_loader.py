import pandas as pd
import numpy as np
import faiss
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from faiss_gpu_utils import FaissGPUManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenCLIPDataLoader:
    def __init__(self):
        self.df = None
        self.index = None
        self.embeddings = None
        self.metadata = None
        self.filename_to_idx = {}
        self.idx_to_filename_root = {}
        
        # Additional mappings for case-insensitive search
        self.filename_to_actual = {}  # Map lowercase filename to actual filename
        self.filename_mappings = {}  # Map all filename variations to filename_root
        
        # Initialize GPU utilities
        self.gpu_utils = FaissGPUManager()
        
    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """Load CSV with proper encoding handling"""
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    self.df = pd.read_csv(csv_path, encoding=encoding)
                    logger.info(f"Successfully loaded CSV with {encoding} encoding")
                    logger.info(f"CSV shape: {self.df.shape}")
                    logger.info(f"Columns: {list(self.df.columns)}")
                    
                    # Create filename mappings after loading CSV
                    self._create_filename_mappings()
                    return self.df
                except UnicodeDecodeError:
                    continue
            raise Exception("Could not decode CSV with any common encoding")
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def load_faiss_index(self, index_dir: str) -> bool:
        """Load FAISS index and metadata for OpenCLIP embeddings"""
        try:
            # Load FAISS index
            index_path = os.path.join(index_dir, "openclip_index.faiss")
            if os.path.exists(index_path):
                # Load CPU index first
                cpu_index = faiss.read_index(index_path)
                logger.info(f"Loaded CPU FAISS index: {cpu_index.ntotal} vectors")
                
                # Convert to GPU
                self.index = self.gpu_utils.transfer_index_to_gpu(cpu_index)
                logger.info(f"Converted to GPU index")
            else:
                logger.warning(f"FAISS index not found at {index_path}")
                return False
            
            # Load embeddings
            embeddings_path = os.path.join(index_dir, "openclip_embeddings.npy")
            if os.path.exists(embeddings_path):
                self.embeddings = np.load(embeddings_path)
                logger.info(f"Loaded embeddings: {self.embeddings.shape}")
            else:
                logger.warning(f"Embeddings not found at {embeddings_path}")
                return False
            
            # Load metadata
            metadata_path = os.path.join(index_dir, "openclip_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.metadata.get('image_paths', []))} images")
            else:
                logger.warning(f"Metadata not found at {metadata_path}")
                return False
            
            # Create filename mappings
            self._create_filename_mappings()
            
            # Pre-warm the GPU
            logger.info("Pre-warming GPU...")
            self.gpu_utils.warm_up_gpu()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            return False
    
    def load_npz_embeddings(self, npz_path: str) -> bool:
        """Load embeddings from NPZ file format as specified by user"""
        try:
            # Load NPZ file
            data = np.load(npz_path)
            
            if 'embeddings' not in data or 'image_paths' not in data:
                logger.error("NPZ file missing required keys: 'embeddings' and 'image_paths'")
                return False
            
            self.embeddings = data['embeddings']
            image_paths = data['image_paths']
            
            logger.info(f"Loaded NPZ embeddings: {self.embeddings.shape}")
            logger.info(f"Number of image paths: {len(image_paths)}")
            
            # Create metadata format compatible with existing structure
            self.metadata = {'image_paths': image_paths.tolist() if isinstance(image_paths, np.ndarray) else image_paths}
            
            # Create FAISS index on GPU
            dimension = self.embeddings.shape[1]  # Should be 768 for OpenCLIP
            cpu_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            cpu_index.add(self.embeddings.astype(np.float32))
            logger.info(f"Created CPU FAISS index with {cpu_index.ntotal} vectors")
            
            # Convert to GPU
            self.index = self.gpu_utils.transfer_index_to_gpu(cpu_index)
            logger.info(f"Converted to GPU index")
            
            # Create filename mappings
            self._create_filename_mappings()
            
            # Pre-warm the GPU
            logger.info("Pre-warming GPU...")
            self.gpu_utils.warm_up_gpu()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading NPZ embeddings: {e}")
            return False
    
    def save_index(self, index_dir: str):
        """Save FAISS index and metadata"""
        try:
            os.makedirs(index_dir, exist_ok=True)
            
            # Convert GPU index back to CPU for saving
            if hasattr(self.index, '__class__') and 'Gpu' in self.index.__class__.__name__:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
            else:
                cpu_index = self.index
            
            # Save FAISS index
            index_path = os.path.join(index_dir, "openclip_index.faiss")
            faiss.write_index(cpu_index, index_path)
            logger.info(f"Saved FAISS index to {index_path}")
            
            # Save embeddings
            embeddings_path = os.path.join(index_dir, "openclip_embeddings.npy")
            np.save(embeddings_path, self.embeddings)
            logger.info(f"Saved embeddings to {embeddings_path}")
            
            # Save metadata
            metadata_path = os.path.join(index_dir, "openclip_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f)
            logger.info(f"Saved metadata to {metadata_path}")
            
            # Also save in NPZ format as specified by user
            npz_path = os.path.join(index_dir, "openclip_embeddings.npz")
            np.savez(npz_path, 
                     embeddings=self.embeddings,
                     image_paths=np.array(self.metadata['image_paths']))
            logger.info(f"Saved NPZ format to {npz_path}")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise
    
    def _create_filename_mappings(self):
        """Create mappings between filenames and indices - Enhanced version"""
        # Clear existing mappings
        self.filename_to_idx.clear()
        self.idx_to_filename_root.clear()
        self.filename_to_actual.clear()
        self.filename_mappings.clear()
        
        # First, create mappings from CSV if loaded
        if self.df is not None and 'filename_root' in self.df.columns:
            # Create mappings for ALL files in the metadata (not just local files)
            unique_filename_roots = self.df['filename_root'].dropna().unique()
            
            for filename_root in unique_filename_roots:
                # Store variations for flexible matching
                self.filename_mappings[filename_root.lower()] = filename_root
                self.filename_mappings[filename_root.upper()] = filename_root
                self.filename_mappings[filename_root] = filename_root
                
                # Also store without extension if present
                base_name = filename_root.split('.')[0]
                if base_name != filename_root:
                    self.filename_mappings[base_name.lower()] = filename_root
                    self.filename_mappings[base_name.upper()] = filename_root
                    self.filename_mappings[base_name] = filename_root
            
            logger.info(f"Created {len(self.filename_mappings)} filename mappings from CSV")
        
        # Then, if we have metadata from embeddings, create index mappings
        if self.metadata and 'image_paths' in self.metadata:
            for idx, image_path in enumerate(self.metadata['image_paths']):
                # Extract filename from path
                filename = os.path.basename(image_path)
                filename_root = filename.split('_')[0]
                
                self.filename_to_idx[filename_root] = idx
                self.idx_to_filename_root[idx] = filename_root
                
                # Also store lowercase mapping
                self.filename_to_actual[filename_root.lower()] = filename_root
            
            logger.info(f"Created index mappings for {len(self.filename_to_idx)} embedded files")
    
    def get_filter_columns(self) -> List[str]:
        """Get all available filter columns from CSV"""
        if self.df is None:
            return []
        
        # Exclude non-filter columns
        exclude_cols = ['filename_root']  # Add other non-filter columns as needed
        filter_cols = [col for col in self.df.columns if col not in exclude_cols]
        return filter_cols
    
    def search_by_sku(self, sku_cod: str) -> Optional[Dict]:
        """Search by SKU code"""
        if self.df is None:
            return None
        
        result = self.df[self.df['SKU_COD'] == sku_cod]
        if len(result) > 0:
            return result.iloc[0].to_dict()
        return None
    
    def search_by_sku_list(self, sku_list: List[str]) -> List[Dict]:
        """Search by list of SKU codes using bulk operations"""
        if self.df is None:
            return []
        
        # Convert to uppercase for consistent matching
        sku_list_upper = [str(sku).strip().upper() for sku in sku_list]
        
        # Bulk exact match
        exact_matches = self.df[self.df['SKU_COD'].astype(str).str.upper().isin(sku_list_upper)]
        results = exact_matches.to_dict('records')
        
        # Find missing SKUs
        found_skus = set(exact_matches['SKU_COD'].astype(str).str.upper())
        missing_skus = [sku for sku in sku_list_upper if sku not in found_skus]
        
        if missing_skus:
            logger.info(f"Found {len(results)} exact matches, searching for {len(missing_skus)} missing SKUs")
            
            # Try filename_root derived from SKU
            derived_roots = []
            for sku in missing_skus:
                parts = sku.split('.')
                if len(parts) >= 3:
                    derived_root = f"{parts[0]}.{parts[1]}.{parts[2]}"
                    derived_roots.append(derived_root)
            
            if derived_roots:
                # Bulk search by filename_root
                root_matches = self.df[self.df['filename_root'].astype(str).str.upper().isin(derived_roots)]
                results.extend(root_matches.to_dict('records'))
                
                # Update found SKUs
                if 'filename_root' in root_matches.columns:
                    found_roots = set(root_matches['filename_root'].astype(str).str.upper())
                    still_missing = [sku for sku, root in zip(missing_skus, derived_roots) 
                                   if root not in found_roots]
                else:
                    still_missing = missing_skus
            else:
                still_missing = missing_skus
            
            # For remaining missing, try truncated search
            if still_missing and len(still_missing) < 100:  # Limit to avoid too many regex operations
                for sku in still_missing[:50]:  # Further limit
                    truncated = sku[:min(10, len(sku))]
                    pattern_matches = self.df[
                        self.df['SKU_COD'].astype(str).str.upper().str.startswith(truncated)
                    ]
                    if len(pattern_matches) > 0:
                        results.extend(pattern_matches.head(1).to_dict('records'))
        
        return results
    
    def filter_dataframe(self, filters: Dict[str, any]) -> pd.DataFrame:
        """Apply filters to dataframe"""
        if self.df is None:
            return pd.DataFrame()
        
        filtered_df = self.df.copy()
        
        for column, value in filters.items():
            if column in filtered_df.columns and value is not None and value != '':
                if isinstance(value, str):
                    # Case-insensitive string matching
                    filtered_df = filtered_df[
                        filtered_df[column].astype(str).str.contains(value, case=False, na=False)
                    ]
                else:
                    # Exact match for non-string values
                    filtered_df = filtered_df[filtered_df[column] == value]
        
        return filtered_df
    
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get GPU memory usage statistics"""
        # Return basic memory info since FaissGPUManager doesn't have get_memory_usage
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                    "reserved": torch.cuda.memory_reserved() / 1024**3,      # GB
                    "total": torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                }
        except:
            pass
        return {"message": "GPU memory info not available"}

# Global instance
openclip_data_loader = OpenCLIPDataLoader() 