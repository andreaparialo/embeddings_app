#!/usr/bin/env python3
"""
Simple startup script for the Hybrid Product Search Engine
Supports GPU selection to avoid overloading GPUs in use
"""

import uvicorn
import os
import sys
import argparse

def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Start the Hybrid Product Search Engine')
    parser.add_argument('--gpu', type=int, default=None, 
                       help='GPU ID to use (0-7). If not specified, uses CUDA_VISIBLE_DEVICES env var or default GPU')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint to use (e.g., 680, 1020, 1095). If not specified, uses default from config')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Force CPU mode for FAISS (no GPU acceleration)')
    args = parser.parse_args()
    
    # Set GPU configuration
    if args.no_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['FORCE_CPU_FAISS'] = 'true'
        print("🖥️  Running in CPU mode (GPU disabled)")
    elif args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"🎮 Using GPU {args.gpu}")
    elif 'CUDA_VISIBLE_DEVICES' not in os.environ:
        # Default to GPU 1 if not specified (to avoid GPU 0 which might be in use)
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        print("🎮 Using GPU 1 (default - to avoid GPU 0)")
    else:
        print(f"🎮 Using GPU(s): {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # Set checkpoint if specified
    if args.checkpoint:
        os.environ['DEFAULT_CHECKPOINT'] = str(args.checkpoint)
        print(f"📌 Using checkpoint: {args.checkpoint}")
    
    # Get the absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if required directories exist in current directory
    required_dirs = ["database_results", "indexes", "loras", "pictures"]
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = os.path.join(script_dir, dir_name)
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print("❌ Missing required directories:")
        for dir_name in missing_dirs:
            print(f"   - {dir_name}")
        print("\nPlease ensure all required directories are present in the current directory.")
        sys.exit(1)
    
    # Check if CSV file exists
    csv_path = os.path.join(script_dir, "database_results/DB_ACTIVE.csv")
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Check if FAISS index exists
    indexes_dir = os.path.join(script_dir, "indexes")
    faiss_files = [f for f in os.listdir(indexes_dir) if f.endswith('.faiss')]
    
    if not faiss_files:
        print("❌ No FAISS index files found in indexes/ directory")
        print("indexes/ exist")
        sys.exit(1)
    else:
        print(f"✅ Found {len(faiss_files)} FAISS index file(s):")
        for f in faiss_files:
            print(f"   - {f}")
    
    print("✅ All required files and directories found")
    print("🚀 Starting Hybrid Product Search Engine...")
    print("📊 This may take a few minutes to load the model and data...")
    print("🌐 Web interface will be available at: http://127.0.0.1:8080")
    
    # Show GPU status
    gpu_mode = "disabled" if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" else f"GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}"
    print(f"⚡ GPU acceleration: {gpu_mode}")
    
    # We're already in the correct directory
    os.chdir(script_dir)
    
    # Start the server
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8080,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main() 