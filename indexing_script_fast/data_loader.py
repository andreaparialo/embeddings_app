import pandas as pd
import numpy as np
import faiss
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.df = None
        self.index = None
        self.gpu_index = None  # GPU accelerated index
        self.embeddings = None
        self.metadata = None
        self.filename_to_idx = {}
        self.idx_to_filename_root = {}
        # Check for force CPU mode
        force_cpu = os.environ.get('FORCE_CPU_FAISS', '').lower() in ['true', '1', 'yes']
        
        self.use_gpu = torch.cuda.is_available() and not force_cpu
        self.num_gpus = torch.cuda.device_count() if self.use_gpu else 0
        
        if force_cpu:
            logger.info("🖥️  FORCE_CPU_FAISS enabled - using CPU for FAISS index")
        elif self.use_gpu:
            logger.info(f"🚀 GPU support enabled with {self.num_gpus} GPUs available")
        else:
            logger.warning("⚠️  GPU support not available, falling back to CPU")
        
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
        """Load FAISS index and metadata with GPU acceleration"""
        try:
            # Load FAISS index - try different naming conventions
            index_paths = [
                os.path.join(index_dir, "v11_complete_merged_20250625_115302.faiss"),  # New merged index
                os.path.join(index_dir, f"v11_o00_index_{checkpoint}.faiss"),         # Checkpoint-specific
                os.path.join(index_dir, "v11_o00_index.faiss")                        # Fallback
            ]
            
            cpu_index = None
            used_path = None
            
            for index_path in index_paths:
                if os.path.exists(index_path):
                    cpu_index = faiss.read_index(index_path)
                    used_path = index_path
                    logger.info(f"Loaded FAISS index from {index_path}: {cpu_index.ntotal} vectors")
                    break
            
            if cpu_index is None:
                raise FileNotFoundError("No suitable FAISS index found")
            
            # Check GPU memory before attempting GPU initialization
            if self.use_gpu:
                logger.info(f"Checking GPU resources (detected {self.num_gpus} GPUs)")
                
                # Check GPU memory availability
                gpu_memory_ok = True
                try:
                    for i in range(min(self.num_gpus, 2)):  # Check first 2 GPUs
                        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                        allocated_memory = torch.cuda.memory_allocated(i) / 1024**3  # GB
                        free_memory = total_memory - allocated_memory
                        logger.info(f"GPU {i}: {free_memory:.1f}GB free / {total_memory:.1f}GB total")
                        
                        # Need at least 4GB free for FAISS index
                        if free_memory < 4.0:
                            logger.warning(f"GPU {i} has insufficient memory ({free_memory:.1f}GB free, need 4GB+)")
                            gpu_memory_ok = False
                except Exception as e:
                    logger.warning(f"Could not check GPU memory: {e}")
                    gpu_memory_ok = False
                
                # Skip GPU if memory issues detected
                if not gpu_memory_ok:
                    logger.warning("⚠️  Insufficient GPU memory detected, using CPU FAISS index")
                    self.index = cpu_index
                    self.use_gpu = False
                else:
                    # Try GPU initialization with aggressive timeout
                    gpu_success = False
                    
                    # Use threading with timeout for GPU initialization
                    import threading
                    import time
                    
                    def gpu_init_worker():
                        nonlocal gpu_success
                        try:
                            logger.info("🔧 Step 1: Clearing GPU cache...")
                            torch.cuda.empty_cache()
                            
                            logger.info("🔧 Step 2: Creating GPU resources with larger limits...")
                            res = faiss.StandardGpuResources()
                            # Try larger temp memory allocation
                            res.setTempMemory(2 * 1024 * 1024 * 1024)  # 2GB temp memory
                            # Set larger allocation limits
                            res.setDefaultNullStreamAllDevices()
                            
                            logger.info("🔧 Step 3: Attempting GPU transfer with batch processing...")
                            # Try cloning to GPU instead of direct transfer (sometimes more stable)
                            config = faiss.GpuClonerOptions()
                            config.useFloat16 = False  # Use full precision first
                            config.usePrecomputed = True
                            
                            self.gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index, config)
                            logger.info("🔧 Step 4: Verifying GPU index...")
                            
                            # Quick verification
                            if self.gpu_index.ntotal == cpu_index.ntotal:
                                self.index = self.gpu_index
                                gpu_success = True
                                logger.info("✅ FAISS index successfully moved to GPU 0")
                            else:
                                logger.warning("❌ GPU index verification failed")
                                
                        except Exception as e:
                            logger.warning(f"GPU initialization failed: {e}")
                            import traceback
                            logger.warning(f"Traceback: {traceback.format_exc()}")
                    
                    # Start GPU initialization in separate thread
                    gpu_thread = threading.Thread(target=gpu_init_worker)
                    gpu_thread.daemon = True
                    gpu_thread.start()
                    
                    # Wait for GPU initialization with timeout
                    gpu_thread.join(timeout=45)  # Increased to 45 second timeout
                    
                    if gpu_thread.is_alive():
                        logger.warning("⏰ GPU 0 initialization timed out after 45 seconds")
                        gpu_success = False
                        
                        # Try GPU 1 if available
                        if self.num_gpus > 1 and not gpu_success:
                            logger.info("🔄 Trying GPU 1 as fallback...")
                            gpu_success_gpu1 = False
                            
                            def gpu1_init_worker():
                                nonlocal gpu_success_gpu1
                                try:
                                    torch.cuda.empty_cache()
                                    res1 = faiss.StandardGpuResources()
                                    res1.setTempMemory(1 * 1024 * 1024 * 1024)  # 1GB
                                    
                                    config = faiss.GpuClonerOptions()
                                    config.useFloat16 = True  # Try half precision on GPU 1
                                    
                                    self.gpu_index = faiss.index_cpu_to_gpu(res1, 1, cpu_index, config)
                                    if self.gpu_index.ntotal == cpu_index.ntotal:
                                        self.index = self.gpu_index
                                        gpu_success_gpu1 = True
                                        logger.info("✅ FAISS index moved to GPU 1 (half precision)")
                                except Exception as e:
                                    logger.warning(f"GPU 1 failed: {e}")
                            
                            gpu1_thread = threading.Thread(target=gpu1_init_worker)
                            gpu1_thread.daemon = True
                            gpu1_thread.start()
                            gpu1_thread.join(timeout=30)
                            
                            gpu_success = gpu_success_gpu1
                    
                    # Fallback to CPU if all GPU attempts failed
                    if not gpu_success:
                        logger.warning("⚠️  All GPU initialization attempts failed/timed out, using CPU FAISS index")
                        logger.info("💡 GPU diagnostics: CUDA contexts might be conflicting or FAISS GPU version incompatible")
                        self.index = cpu_index
                        self.use_gpu = False
                        
            else:
                self.index = cpu_index
                logger.info("Using CPU FAISS index")
            
            # Load embeddings - try different naming conventions based on which index was loaded
            logger.info("Loading embeddings...")
            embeddings_paths = [
                os.path.join(index_dir, "v11_complete_merged_20250625_115302_embeddings.npy"),  # New merged embeddings
                os.path.join(index_dir, f"v11_o00_index_{checkpoint}_embeddings.npy"),          # Checkpoint-specific
                os.path.join(index_dir, "v11_o00_index_embeddings.npy")                         # Fallback
            ]
            
            embeddings_loaded = False
            for embeddings_path in embeddings_paths:
                if os.path.exists(embeddings_path):
                    try:
                        logger.info(f"Attempting to load embeddings from {embeddings_path}")
                        self.embeddings = np.load(embeddings_path)
                        logger.info(f"✅ Loaded embeddings from {embeddings_path}: {self.embeddings.shape}")
                        embeddings_loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load embeddings from {embeddings_path}: {e}")
                        continue
            
            if not embeddings_loaded:
                logger.warning("⚠️  No embeddings file found - some features may not work")
            
            # Load metadata - try different naming conventions based on which index was loaded
            logger.info("Loading metadata...")
            metadata_paths = [
                os.path.join(index_dir, "v11_complete_merged_20250625_115302_metadata_fixed.json"),  # Fixed metadata (priority)
                os.path.join(index_dir, "v11_complete_merged_20250625_115302_metadata.json"),       # Original merged metadata
                os.path.join(index_dir, f"v11_o00_index_{checkpoint}_metadata.json"),              # Checkpoint-specific
                os.path.join(index_dir, "v11_o00_index_metadata.json")                             # Fallback
            ]
            
            metadata_loaded = False
            for metadata_path in metadata_paths:
                if os.path.exists(metadata_path):
                    try:
                        logger.info(f"Attempting to load metadata from {metadata_path}")
                        with open(metadata_path, 'r') as f:
                            self.metadata = json.load(f)
                        logger.info(f"✅ Loaded metadata from {metadata_path}")
                        metadata_loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
                        continue
            
            if not metadata_loaded:
                logger.warning("⚠️  No metadata file found - filename mappings may not work properly")
            
            # Create filename mappings
            logger.info("Creating filename mappings...")
            try:
                self._create_filename_mappings()
                logger.info("✅ Filename mappings created successfully")
            except Exception as e:
                logger.warning(f"⚠️  Error creating filename mappings: {e}")
                # Continue anyway as this is not critical for basic functionality
            
            logger.info("🎉 FAISS index loading completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            return False
    
    def _create_filename_mappings(self):
        """Create mappings between filenames and indices"""
        if not self.metadata or 'image_paths' not in self.metadata:
            return
        
        pictures_dir = "pictures"
        
        # Get all actual picture files
        jpg_files = glob.glob(os.path.join(pictures_dir, "*.jpg"))
        JPG_files = glob.glob(os.path.join(pictures_dir, "*.JPG"))
        all_picture_files = jpg_files + JPG_files
        
        # Create a mapping of filename_root to actual file path
        filename_root_to_path = {}
        for file_path in all_picture_files:
            filename = os.path.basename(file_path)
            # Extract filename_root (everything before first underscore)
            filename_root = filename.split('_')[0]
            filename_root_to_path[filename_root] = file_path
        
        # Map index positions to filename_roots
        for idx, image_path in enumerate(self.metadata['image_paths']):
            # Extract filename from path
            filename = os.path.basename(image_path)
            filename_root = filename.split('_')[0]
            
            # Check if we have this file in our pictures directory
            if filename_root in filename_root_to_path:
                self.filename_to_idx[filename_root] = idx
                self.idx_to_filename_root[idx] = filename_root
        
        logger.info(f"Created mappings for {len(self.filename_to_idx)} files")
    
    def get_filter_columns(self) -> List[str]:
        """Get all available filter columns from CSV"""
        if self.df is None:
            return []
        
        # Priority filter columns - these will appear first in the UI
        priority_cols = [
            'MACRO_SHAPE_AWS',
            'GRANULAR_SHAPE_AWS', 
            'FlatTop_FlatTop_1',
            'browline_browline_1',
            'bridge_Bridge_1',
            'LENS_BASE_DES',
            'RIM_TYPE_DES'
        ]
        
        # Exclude non-filter columns
        exclude_cols = ['filename_root']  # Add other non-filter columns as needed
        
        # Get all available columns excluding the non-filter ones
        all_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        # Start with priority columns that exist in the dataframe
        filter_cols = [col for col in priority_cols if col in all_cols]
        
        # Add remaining columns
        remaining_cols = [col for col in all_cols if col not in priority_cols]
        filter_cols.extend(remaining_cols)
        
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
        return sorted(checkpoints)
    
    def get_gpu_memory_info(self) -> Dict[str, any]:
        """Get GPU memory information"""
        if not self.use_gpu:
            return {"message": "GPU not available"}
        
        info = {"gpus": []}
        for i in range(self.num_gpus):
            gpu_info = {
                "id": i,
                "allocated": torch.cuda.memory_allocated(i) / 1024**3,  # GB
                "cached": torch.cuda.memory_reserved(i) / 1024**3,      # GB
                "total": torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            }
            info["gpus"].append(gpu_info)
        
        return info

# Global instance
data_loader = DataLoader() 