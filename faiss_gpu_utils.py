import faiss
import numpy as np
import os
import time
import logging
from typing import Optional, Tuple
import torch

logger = logging.getLogger(__name__)

class FaissGPUManager:
    """Manages FAISS GPU operations with optimizations for fast transfer"""
    
    def __init__(self):
        self.gpu_index = None
        self.gpu_resources = None
        self.is_gpu_available = False
        self.gpu_id = 0  # Default to first GPU
        
        # Configure CUDA cache for faster subsequent loads
        os.environ['CUDA_CACHE_MAXSIZE'] = '2147483647'
        os.environ['CUDA_CACHE_DISABLE'] = '0'
        cache_dir = os.path.expanduser("~/.nv/ComputeCache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ['CUDA_CACHE_PATH'] = cache_dir
        
        self._check_gpu_availability()
    
    def _check_gpu_availability(self):
        """Check if GPU is available and FAISS GPU is properly installed"""
        try:
            # Check PyTorch CUDA availability
            if torch.cuda.is_available():
                self.is_gpu_available = True
                gpu_count = torch.cuda.device_count()
                logger.info(f"🎮 Found {gpu_count} GPU(s) available")
                
                # Get GPU info
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    logger.info(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                
                # Test FAISS GPU availability
                try:
                    test_res = faiss.StandardGpuResources()
                    logger.info("✅ FAISS GPU support confirmed")
                    self.is_gpu_available = True
                except Exception as e:
                    logger.warning(f"⚠️ FAISS GPU not available: {e}")
                    logger.warning("   Falling back to CPU mode")
                    self.is_gpu_available = False
            else:
                logger.warning("⚠️ No CUDA-capable GPU detected")
                self.is_gpu_available = False
                
        except Exception as e:
            logger.error(f"❌ Error checking GPU availability: {e}")
            self.is_gpu_available = False
    
    def warm_up_gpu(self):
        """Pre-warm GPU with dummy operations to trigger CUDA initialization"""
        if not self.is_gpu_available:
            return
        
        try:
            logger.info("🔥 Pre-warming GPU...")
            start_time = time.time()
            
            # Create dummy data
            dummy_data = np.random.random((100, 384)).astype('float32')
            dummy_index = faiss.IndexFlatL2(384)
            dummy_index.add(dummy_data)
            
            # Transfer dummy index to GPU
            res = faiss.StandardGpuResources()
            gpu_dummy = faiss.index_cpu_to_gpu(res, self.gpu_id, dummy_index)
            
            # Perform dummy search
            query = np.random.random((1, 384)).astype('float32')
            gpu_dummy.search(query, 10)
            
            warm_up_time = time.time() - start_time
            logger.info(f"✅ GPU warm-up completed in {warm_up_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ GPU warm-up failed: {e}")
    
    def create_gpu_resources(self, temp_memory_mb: int = 0, pinned_memory_mb: int = 256):
        """Create optimized GPU resources"""
        if not self.is_gpu_available:
            return None
        
        try:
            logger.info("🔧 Creating optimized GPU resources...")
            
            # Create GPU resources
            self.gpu_resources = faiss.StandardGpuResources()
            
            # Configure memory settings
            if temp_memory_mb == 0:
                # Disable temp memory for faster startup
                self.gpu_resources.setTempMemory(0)
                logger.info("   ⚡ Temporary memory disabled for fast startup")
            else:
                # Set custom temp memory
                self.gpu_resources.setTempMemory(temp_memory_mb * 1024 * 1024)
                logger.info(f"   📊 Temporary memory: {temp_memory_mb} MB")
            
            # Configure pinned memory for async transfers
            self.gpu_resources.setPinnedMemory(pinned_memory_mb * 1024 * 1024)
            logger.info(f"   📌 Pinned memory: {pinned_memory_mb} MB")
            
            return self.gpu_resources
            
        except Exception as e:
            logger.error(f"❌ Failed to create GPU resources: {e}")
            return None
    
    def transfer_index_to_gpu(self, cpu_index: faiss.Index, 
                            use_float16: bool = True,
                            use_precomputed: bool = False) -> Optional[faiss.Index]:
        """Transfer FAISS index from CPU to GPU with optimizations"""
        if not self.is_gpu_available:
            logger.warning("⚠️ GPU not available, returning CPU index")
            return cpu_index
        
        try:
            logger.info("🚀 Starting optimized GPU transfer...")
            logger.info(f"   📊 Index size: {cpu_index.ntotal:,} vectors")
            logger.info(f"   📏 Dimension: {cpu_index.d}")
            
            start_time = time.time()
            
            # Create GPU resources if not already created
            if self.gpu_resources is None:
                self.create_gpu_resources()
            
            # Configure cloner options for optimization
            co = faiss.GpuClonerOptions()
            co.useFloat16 = use_float16  # Use 16-bit precision for memory efficiency
            co.usePrecomputed = use_precomputed
            
            if use_float16:
                logger.info("   ⚡ Using Float16 for 2x memory efficiency")
            
            # Transfer to GPU
            logger.info(f"   🔄 Transferring to GPU {self.gpu_id}...")
            self.gpu_index = faiss.index_cpu_to_gpu(
                self.gpu_resources, 
                self.gpu_id, 
                cpu_index, 
                co
            )
            
            transfer_time = time.time() - start_time
            logger.info(f"✅ GPU transfer completed in {transfer_time:.2f}s")
            
            # Get GPU memory usage
            self._log_gpu_memory_usage()
            
            return self.gpu_index
            
        except Exception as e:
            logger.error(f"❌ GPU transfer failed: {e}")
            logger.warning("   Falling back to CPU index")
            return cpu_index
    
    def transfer_index_to_all_gpus(self, cpu_index: faiss.Index) -> Optional[faiss.Index]:
        """Transfer index to all available GPUs for maximum performance"""
        if not self.is_gpu_available:
            logger.warning("⚠️ GPU not available, returning CPU index")
            return cpu_index
        
        try:
            ngpus = faiss.get_num_gpus()
            logger.info(f"🚀 Distributing index across {ngpus} GPUs...")
            
            start_time = time.time()
            
            # Configure for multi-GPU
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True  # Shard the index across GPUs
            co.useFloat16 = True
            
            # Transfer to all GPUs
            self.gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co=co, ngpu=ngpus)
            
            transfer_time = time.time() - start_time
            logger.info(f"✅ Multi-GPU transfer completed in {transfer_time:.2f}s")
            
            return self.gpu_index
            
        except Exception as e:
            logger.error(f"❌ Multi-GPU transfer failed: {e}")
            return self.transfer_index_to_gpu(cpu_index)  # Fallback to single GPU
    
    def search(self, query_vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform GPU-accelerated search"""
        if self.gpu_index is None:
            raise ValueError("GPU index not initialized")
        
        # Ensure query vectors are contiguous and float32
        if not query_vectors.flags['C_CONTIGUOUS']:
            query_vectors = np.ascontiguousarray(query_vectors)
        query_vectors = query_vectors.astype('float32')
        
        # Perform search
        distances, indices = self.gpu_index.search(query_vectors, k)
        
        return distances, indices
    
    def _log_gpu_memory_usage(self):
        """Log current GPU memory usage"""
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    logger.info(f"   GPU {i} Memory: {allocated:.1f}GB allocated, "
                              f"{reserved:.1f}GB reserved, {total:.1f}GB total")
        except Exception as e:
            logger.debug(f"Could not log GPU memory: {e}")
    
    def cleanup(self):
        """Clean up GPU resources"""
        if self.gpu_index is not None:
            self.gpu_index = None
        if self.gpu_resources is not None:
            self.gpu_resources = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("🧹 GPU resources cleaned up")

# Global instance
faiss_gpu_manager = FaissGPUManager() 