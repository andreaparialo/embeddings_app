#!/usr/bin/env python3
"""
Index db_pictures_512 folder using GME model with LoRA checkpoint 1095
"""

import os
import sys

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add indexing script path
sys.path.append('indexing_script_fast')

from datetime import datetime

def run_indexing():
    print("🚀 Indexing DB Pictures 512x512")
    print("=" * 80)
    
    # Configuration
    image_dir = "db_pictures_512"
    checkpoint_path = "loras/v11-20250620-105815/checkpoint-1095"
    index_name = "v11_1095_db_pictures_512"
    output_dir = "indexes"
    
    # Check if image directory exists
    if not os.path.exists(image_dir):
        print(f"❌ Image directory '{image_dir}' not found!")
        print("Please run resize_db_pictures_512.py first.")
        return False
    
    # Count images
    image_count = len([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"\n📊 Images to index: {image_count:,}")
    print(f"📁 Image directory: {image_dir}")
    print(f"🔧 LoRA checkpoint: {checkpoint_path}")
    print(f"🏷️  Index name: {index_name}")
    print(f"💾 Output directory: {output_dir}")
    
    # Run the indexing using the optimized script
    cmd = (
        f"python3 indexing_script_fast/lora_max_performance_indexing_custom.py "
        f"{checkpoint_path} {image_dir} {index_name} {output_dir}"
    )
    
    print(f"\n🖥️  Running command:")
    print(f"  {cmd}")
    
    # Execute the indexing
    start_time = datetime.now()
    result = os.system(cmd)
    end_time = datetime.now()
    
    if result == 0:
        elapsed = (end_time - start_time).total_seconds()
        print(f"\n✅ Indexing completed successfully!")
        print(f"⏱️  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"🚀 Speed: {image_count/elapsed:.1f} images/second")
        
        # Check output files
        index_files = [
            f"{output_dir}/{index_name}.faiss",
            f"{output_dir}/{index_name}_embeddings.npy",
            f"{output_dir}/{index_name}_metadata.json"
        ]
        
        print(f"\n📁 Output files:")
        for file_path in index_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path) / (1024**2)  # MB
                print(f"  ✅ {file_path} ({size:.1f} MB)")
            else:
                print(f"  ❌ {file_path} (not found)")
        
        return True
    else:
        print(f"\n❌ Indexing failed with code: {result}")
        return False

def create_indexing_script():
    """Create a shell script for easy re-running"""
    script_content = f"""#!/bin/bash
# Index DB Pictures 512x512 with GME + LoRA

echo "🚀 Starting DB Pictures 512x512 Indexing"
echo "========================================"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate faiss_env

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run indexing
python3 indexing_script_fast/lora_max_performance_indexing_custom.py \\
    loras/v11-20250620-105815/checkpoint-1095 \\
    db_pictures_512 \\
    v11_1095_db_pictures_512 \\
    indexes

echo ""
echo "✅ Indexing complete!"
"""
    
    script_path = "index_db_pictures_512.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"\n📝 Created shell script: {script_path}")
    print("   You can run it directly: ./index_db_pictures_512.sh")

if __name__ == "__main__":
    # Create shell script for convenience
    create_indexing_script()
    
    # Ask user if they want to run indexing now
    print("\n" + "="*60)
    response = input("Do you want to start indexing now? (y/n): ").lower().strip()
    
    if response == 'y':
        run_indexing()
    else:
        print("\n📝 You can run indexing later using:")
        print("   python3 index_db_pictures_512.py")
        print("   or")
        print("   ./index_db_pictures_512.sh") 