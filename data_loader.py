import pandas as pd
import numpy as np
import faiss
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from old_app.faiss_gpu_utils import faiss_gpu_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.df = None
        self.index = None
        self.cpu_index = None  # Keep CPU index for fallback
        self.embeddings = None
        self.metadata = None
        self.filename_to_idx = {}
        self.idx_to_filename_root = {}
        self.use_gpu = True  # Flag to control GPU usage
        
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
                    return self.df
                except UnicodeDecodeError:
                    continue
            raise Exception("Could not decode CSV with any common encoding")
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def load_faiss_index(self, index_dir: str, checkpoint: str = "1095") -> bool:
        """Load FAISS index and metadata"""
        try:
            # Pre-warm GPU if enabled
            if self.use_gpu and faiss_gpu_manager.is_gpu_available:
                faiss_gpu_manager.warm_up_gpu()
            
            # Load FAISS index - try new merged index first
            index_path = os.path.join(index_dir, f"v11_complete_merged_20250625_115302.faiss")
            if os.path.exists(index_path):
                self.cpu_index = faiss.read_index(index_path)
                logger.info(f"Loaded merged FAISS index: {self.cpu_index.ntotal} vectors")
            else:
                # Fallback to checkpoint-specific index
                index_path = os.path.join(index_dir, f"v11_o00_index_{checkpoint}.faiss")
                if os.path.exists(index_path):
                    self.cpu_index = faiss.read_index(index_path)
                    logger.info(f"Loaded FAISS index: {self.cpu_index.ntotal} vectors")
                else:
                    # Fallback to non-checkpoint version
                    index_path = os.path.join(index_dir, "v11_o00_index.faiss")
                    self.cpu_index = faiss.read_index(index_path)
                    logger.info(f"Loaded FAISS index (fallback): {self.cpu_index.ntotal} vectors")
            
            # Transfer to GPU if enabled
            if self.use_gpu and faiss_gpu_manager.is_gpu_available:
                # Use all 4 A100 GPUs for maximum performance
                if faiss.get_num_gpus() >= 4:
                    self.index = faiss_gpu_manager.transfer_index_to_all_gpus(self.cpu_index)
                else:
                    self.index = faiss_gpu_manager.transfer_index_to_gpu(self.cpu_index)
            else:
                self.index = self.cpu_index
                logger.info("🖥️  Using CPU mode for FAISS index")
            
            # Load embeddings - try merged embeddings first
            embeddings_path = os.path.join(index_dir, f"v11_complete_merged_20250625_115302_embeddings.npy")
            if os.path.exists(embeddings_path):
                self.embeddings = np.load(embeddings_path)
            else:
                # Fallback to checkpoint-specific embeddings
                embeddings_path = os.path.join(index_dir, f"v11_o00_index_{checkpoint}_embeddings.npy")
                if os.path.exists(embeddings_path):
                    self.embeddings = np.load(embeddings_path)
                else:
                    embeddings_path = os.path.join(index_dir, "v11_o00_index_embeddings.npy")
                    self.embeddings = np.load(embeddings_path)
            logger.info(f"Loaded embeddings: {self.embeddings.shape}")
            
            # Load metadata - try merged metadata first
            metadata_path = os.path.join(index_dir, f"v11_complete_merged_20250625_115302_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            else:
                # Fallback to checkpoint-specific metadata
                metadata_path = os.path.join(index_dir, f"v11_o00_index_{checkpoint}_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                else:
                    metadata_path = os.path.join(index_dir, "v11_o00_index_metadata.json")
                    with open(metadata_path, 'r') as f:
                        self.metadata = json.load(f)
            
            # Create filename mappings
            self._create_filename_mappings()
            return True
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            return False
    
    def _create_filename_mappings(self):
        """Create mappings between filenames and indices"""
        if not self.metadata or 'image_paths' not in self.metadata:
            return
        
        # Create mappings for ALL files in the metadata, not just ones that exist locally
        for idx, image_path in enumerate(self.metadata['image_paths']):
            # Extract filename from path
            filename = os.path.basename(image_path)
            # Extract filename_root (everything before _O00)
            if '_O00' in filename:
                filename_root = filename.split('_O00')[0]
            else:
                # Fallback to splitting by first underscore
                filename_root = filename.split('_')[0]
            
            # Store the mapping
            self.filename_to_idx[filename_root] = idx
            self.idx_to_filename_root[idx] = filename_root
            
            # Also store lowercase version for case-insensitive matching
            if filename_root.lower() != filename_root:
                self.filename_to_idx[filename_root.lower()] = idx
            
            # Also store uppercase version
            if filename_root.upper() != filename_root:
                self.filename_to_idx[filename_root.upper()] = idx
            
            # Also store without any trailing letters (for partial matches)
            # e.g., "20862708079O" -> also store "20862708079"
            if filename_root and filename_root[-1].isalpha():
                base_root = filename_root[:-1]
                if base_root not in self.filename_to_idx:
                    self.filename_to_idx[base_root] = idx
        
        logger.info(f"Created mappings for {len(self.metadata['image_paths'])} files")
        logger.info(f"Total filename mappings: {len(self.filename_to_idx)}")
    
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
        """Search by list of SKU codes"""
        if self.df is None:
            return []
        
        results = []
        for sku in sku_list:
            result = self.search_by_sku(sku)
            if result:
                results.append(result)
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
    
    def get_available_checkpoints(self, loras_dir: str = "loras") -> List[str]:
        """Get available LoRA checkpoints"""
        checkpoints = []
        v11_path = os.path.join(loras_dir, "v11-20250620-105815")
        if os.path.exists(v11_path):
            for item in os.listdir(v11_path):
                if item.startswith("checkpoint-") and os.path.isdir(os.path.join(v11_path, item)):
                    checkpoint_num = item.replace("checkpoint-", "")
                    checkpoints.append(checkpoint_num)
        
        # Sort and prioritize 1095 as default
        checkpoints = sorted(checkpoints, reverse=True)  # Descending order
        if "1095" in checkpoints:
            checkpoints.remove("1095")
            checkpoints.insert(0, "1095")  # Put 1095 first
        
        return checkpoints

# Global instance
data_loader = DataLoader() 