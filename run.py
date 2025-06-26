#!/usr/bin/env python3
"""
Simple startup script for the Hybrid Product Search Engine
"""

import uvicorn
import os
import sys

def main():
    """Main entry point"""
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
    csv_path = os.path.join(script_dir, "database_results/final_with_aws_shapes_20250625_155822.csv")
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Check if FAISS index exists
    index_files = [
        "indexes/v11_complete_merged_20250625_115302.faiss",
        "indexes/v11_o00_index_1095.faiss",
        "indexes/v11_o00_index_680.faiss",
        "indexes/v11_o00_index.faiss"
    ]
    
    if not any(os.path.exists(os.path.join(script_dir, f)) for f in index_files):
        print("❌ No FAISS index files found in indexes/ directory")
        sys.exit(1)
    
    print("✅ All required files and directories found")
    print("🚀 Starting Hybrid Product Search Engine...")
    print("📊 This may take a few minutes to load the model and data...")
    print("🌐 Web interface will be available at: http://127.0.0.1:8080")
    print("⚡ GPU acceleration:", "enabled" if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" else "check your CUDA setup")
    
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