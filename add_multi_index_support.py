#!/usr/bin/env python3
"""
Add support for multiple image folders and indexes to the app
"""

import json
import os
from datetime import datetime

def create_index_config():
    """Create configuration for available indexes and image folders"""
    
    config = {
        "indexes": [
            {
                "id": "v11_merged_latest",
                "name": "Full Database (Mixed Sizes)",
                "description": "All products from new database with original image sizes",
                "index_path": "indexes/v11_merged_latest.faiss",
                "embeddings_path": "indexes/v11_merged_latest_embeddings.npy",
                "metadata_path": "indexes/v11_merged_latest_metadata.json",
                "image_folder": "pictures",
                "image_size": "variable",
                "embedding_dim": 3584,
                "model": "gme-Qwen2-VL-7B + LoRA v11-1095",
                "default": False
            },
            {
                "id": "v11_1095_db_pictures_512",
                "name": "DB Pictures 512x512",
                "description": "Standardized 512x512 images with white padding",
                "index_path": "indexes/v11_1095_db_pictures_512.faiss",
                "embeddings_path": "indexes/v11_1095_db_pictures_512_embeddings.npy",
                "metadata_path": "indexes/v11_1095_db_pictures_512_metadata.json",
                "image_folder": "db_pictures_512",
                "image_size": "512x512",
                "embedding_dim": 3584,
                "model": "gme-Qwen2-VL-7B + LoRA v11-1095",
                "default": True  # Make this the default
            }
        ],
        "database": {
            "path": "database_results/DB_FINAL_SIMILARIT_270615.csv",
            "encoding": "utf-8"
        },
        "created": datetime.now().isoformat(),
        "version": "2.0"
    }
    
    # Save configuration
    config_path = "index_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created index configuration: {config_path}")
    return config

def main():
    print("🔧 Setting Up Multi-Index Support")
    print("=" * 80)
    
    # Create index configuration
    print("\n1️⃣ Creating index configuration...")
    create_index_config()
    
    print("\n✅ Configuration created successfully!")
    print("\n📋 Next Steps:")
    print("1. Run: python3 create_db_pictures_folder.py")
    print("2. Run: python3 resize_db_pictures_512.py")
    print("3. Run: ./index_db_pictures_512.sh")
    print("4. Update app.py and data_loader.py to support multiple indexes")

if __name__ == "__main__":
    main() 