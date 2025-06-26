#!/usr/bin/env python3
"""
Test script to benchmark FAISS GPU vs CPU performance
"""

import faiss
import numpy as np
import time
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from old_app.faiss_gpu_utils import faiss_gpu_manager
from old_app.data_loader import data_loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def benchmark_faiss_performance():
    """Benchmark FAISS index performance on CPU vs GPU"""
    
    logger.info("="*60)
    logger.info("🚀 FAISS GPU vs CPU Performance Benchmark")
    logger.info("="*60)
    
    # Load the index
    logger.info("\n📊 Loading FAISS index...")
    index_dir = "indexes"
    checkpoint = "1095"
    
    # Try to load index
    index_path = os.path.join(index_dir, "v11_complete_merged_20250625_115302.faiss")
    if not os.path.exists(index_path):
        index_path = os.path.join(index_dir, f"v11_o00_index_{checkpoint}.faiss")
    if not os.path.exists(index_path):
        index_path = os.path.join(index_dir, "v11_o00_index.faiss")
    
    if not os.path.exists(index_path):
        logger.error("❌ No FAISS index found!")
        return
    
    # Load CPU index
    logger.info(f"📁 Loading index from: {index_path}")
    cpu_index = faiss.read_index(index_path)
    logger.info(f"✅ Loaded index with {cpu_index.ntotal:,} vectors, dimension {cpu_index.d}")
    
    # Create test queries
    num_queries = 100
    query_vectors = np.random.random((num_queries, cpu_index.d)).astype('float32')
    k = 50
    
    logger.info(f"\n🧪 Test configuration:")
    logger.info(f"   - Number of queries: {num_queries}")
    logger.info(f"   - Top K results: {k}")
    logger.info(f"   - Vector dimension: {cpu_index.d}")
    
    # Benchmark CPU
    logger.info("\n" + "="*40)
    logger.info("🖥️  CPU Performance Test")
    logger.info("="*40)
    
    # Warm up
    cpu_index.search(query_vectors[:10], k)
    
    # Actual benchmark
    cpu_start = time.time()
    cpu_distances, cpu_indices = cpu_index.search(query_vectors, k)
    cpu_time = time.time() - cpu_start
    
    logger.info(f"✅ CPU search completed")
    logger.info(f"   ⏱️  Total time: {cpu_time:.3f} seconds")
    logger.info(f"   ⚡ Queries per second: {num_queries/cpu_time:.1f}")
    logger.info(f"   📊 Time per query: {cpu_time/num_queries*1000:.2f} ms")
    
    # Benchmark GPU if available
    if faiss_gpu_manager.is_gpu_available:
        logger.info("\n" + "="*40)
        logger.info("🎮 GPU Performance Test")
        logger.info("="*40)
        
        # Transfer to GPU
        logger.info("\n🔄 Transferring index to GPU...")
        transfer_start = time.time()
        gpu_index = faiss_gpu_manager.transfer_index_to_gpu(cpu_index, use_float16=True)
        transfer_time = time.time() - transfer_start
        
        if gpu_index == cpu_index:
            logger.error("❌ GPU transfer failed, index is still on CPU")
            return
        
        logger.info(f"✅ GPU transfer completed in {transfer_time:.2f} seconds")
        
        # Warm up GPU
        gpu_index.search(query_vectors[:10], k)
        
        # Actual GPU benchmark
        gpu_start = time.time()
        gpu_distances, gpu_indices = gpu_index.search(query_vectors, k)
        gpu_time = time.time() - gpu_start
        
        logger.info(f"\n✅ GPU search completed")
        logger.info(f"   ⏱️  Total time: {gpu_time:.3f} seconds")
        logger.info(f"   ⚡ Queries per second: {num_queries/gpu_time:.1f}")
        logger.info(f"   📊 Time per query: {gpu_time/num_queries*1000:.2f} ms")
        
        # Compare results
        logger.info("\n" + "="*40)
        logger.info("📊 Performance Comparison")
        logger.info("="*40)
        speedup = cpu_time / gpu_time
        logger.info(f"🚀 GPU Speedup: {speedup:.2f}x faster than CPU")
        logger.info(f"   - CPU time: {cpu_time:.3f}s")
        logger.info(f"   - GPU time: {gpu_time:.3f}s")
        logger.info(f"   - Time saved: {cpu_time - gpu_time:.3f}s")
        
        # Verify results are similar
        matches = np.sum(cpu_indices[:10] == gpu_indices[:10]) / (10 * k) * 100
        logger.info(f"\n✅ Result verification: {matches:.1f}% match in top 10 queries")
        
        # Test with larger batch
        logger.info("\n" + "="*40)
        logger.info("🔥 Large Batch Test (1000 queries)")
        logger.info("="*40)
        
        large_queries = np.random.random((1000, cpu_index.d)).astype('float32')
        
        # CPU large batch
        cpu_large_start = time.time()
        cpu_index.search(large_queries, k)
        cpu_large_time = time.time() - cpu_large_start
        
        # GPU large batch
        gpu_large_start = time.time()
        gpu_index.search(large_queries, k)
        gpu_large_time = time.time() - gpu_large_start
        
        large_speedup = cpu_large_time / gpu_large_time
        logger.info(f"🚀 Large batch GPU speedup: {large_speedup:.2f}x")
        logger.info(f"   - CPU: {cpu_large_time:.3f}s ({1000/cpu_large_time:.1f} queries/sec)")
        logger.info(f"   - GPU: {gpu_large_time:.3f}s ({1000/gpu_large_time:.1f} queries/sec)")
        
        # Memory usage
        faiss_gpu_manager._log_gpu_memory_usage()
        
        # Cleanup
        faiss_gpu_manager.cleanup()
        
    else:
        logger.warning("\n⚠️  GPU not available for testing")
        logger.warning("   Check if you have:")
        logger.warning("   1. CUDA-capable GPU")
        logger.warning("   2. Proper CUDA drivers installed")
        logger.warning("   3. faiss-gpu installed (preferably via conda)")
    
    logger.info("\n" + "="*60)
    logger.info("✅ Benchmark completed!")
    logger.info("="*60)

if __name__ == "__main__":
    benchmark_faiss_performance() 