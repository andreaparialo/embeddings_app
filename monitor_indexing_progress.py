#!/usr/bin/env python3
"""
Monitor the progress of delta indexing
"""

import os
import time
import json
from datetime import datetime

def monitor_indexing():
    print("📊 Monitoring Delta Indexing Progress")
    print("=" * 80)
    
    # Check for delta index files
    delta_files = {
        'index': 'indexes/delta_gme_v11.faiss',
        'embeddings': 'indexes/delta_gme_v11_embeddings.npy',
        'metadata': 'indexes/delta_gme_v11_metadata.json'
    }
    
    # Check GPU usage
    print("\n🖥️  GPU Status:")
    os.system("nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader")
    
    # Check for index files
    print("\n📁 Index Files:")
    for name, path in delta_files.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024**2)  # MB
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            print(f"  ✅ {name}: {path} ({size:.1f} MB, modified {mtime})")
        else:
            print(f"  ⏳ {name}: Not found yet")
    
    # Check for log files or temporary files
    print("\n📄 Log Files:")
    logs = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if "delta" in file and ("log" in file or "tmp" in file):
                logs.append(os.path.join(root, file))
    
    if logs:
        for log in logs[-5:]:  # Show last 5 logs
            print(f"  - {log}")
    else:
        print("  No log files found")
    
    # Check if indexing completed
    all_exist = all(os.path.exists(path) for path in delta_files.values())
    
    if all_exist:
        print("\n✅ Indexing appears to be complete!")
        
        # Load metadata to show stats
        with open(delta_files['metadata'], 'r') as f:
            metadata = json.load(f)
        
        print(f"\n📊 Index Statistics:")
        print(f"  Total embeddings: {metadata.get('total_embeddings', 'N/A')}")
        print(f"  Embedding dimension: {metadata.get('embedding_dimension', 'N/A')}")
        print(f"  Model: {metadata.get('model', 'N/A')}")
        
        return True
    else:
        print("\n⏳ Indexing still in progress...")
        return False

def continuous_monitor(interval=30, max_duration=7200):
    """Monitor continuously until completion or timeout"""
    start_time = time.time()
    
    while True:
        os.system("clear")  # Clear screen
        
        completed = monitor_indexing()
        
        if completed:
            print("\n🎉 Indexing complete! You can now run merge_final_indexes.py")
            break
        
        elapsed = time.time() - start_time
        if elapsed > max_duration:
            print(f"\n⏰ Timeout reached ({max_duration/3600:.1f} hours)")
            break
        
        print(f"\n⏱️  Elapsed time: {elapsed/60:.1f} minutes")
        print(f"💤 Checking again in {interval} seconds...")
        
        time.sleep(interval)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        continuous_monitor()
    else:
        monitor_indexing() 