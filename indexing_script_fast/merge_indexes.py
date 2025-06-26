#!/usr/bin/env python3
"""
Index Merging Script - Merge existing v11_o00_index_1095 with delta index
Creates a complete unified index with all images
"""

import json
import sys
import time
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import os

def find_latest_delta_index() -> Tuple[str, str, str]:
    """Find the most recent delta index files"""
    
    indexes_dir = Path("indexes")
    if not indexes_dir.exists():
        return None, None, None
    
    # Find delta index files
    delta_faiss_files = list(indexes_dir.glob("v11_delta_*.faiss"))
    
    if not delta_faiss_files:
        return None, None, None
    
    # Get the most recent one
    latest_delta = max(delta_faiss_files, key=lambda p: p.stat().st_mtime)
    
    # Derive other filenames
    base_name = latest_delta.stem
    delta_faiss = str(latest_delta)
    delta_embeddings = str(indexes_dir / f"{base_name}_embeddings.npy")
    delta_metadata = str(indexes_dir / f"{base_name}_metadata.json")
    
    # Check if all files exist
    if (Path(delta_faiss).exists() and 
        Path(delta_embeddings).exists() and 
        Path(delta_metadata).exists()):
        return delta_faiss, delta_embeddings, delta_metadata
    
    return None, None, None


def load_existing_index(base_path: str = "indexes/v11_o00_index_1095") -> Dict[str, Any]:
    """Load the existing v11 index"""
    
    files = {
        'faiss': f"{base_path}.faiss",
        'embeddings': f"{base_path}_embeddings.npy", 
        'metadata': f"{base_path}_metadata.json"
    }
    
    # Check if all files exist
    for file_type, file_path in files.items():
        if not Path(file_path).exists():
            print(f"❌ Existing index file not found: {file_path}")
            return {}
    
    print(f"📖 Loading existing index: {base_path}")
    
    try:
        # Load FAISS index
        existing_index = faiss.read_index(files['faiss'])
        print(f"   ✅ FAISS index loaded: {existing_index.ntotal} vectors")
        
        # Load embeddings
        existing_embeddings = np.load(files['embeddings'])
        print(f"   ✅ Embeddings loaded: {existing_embeddings.shape}")
        
        # Load metadata
        with open(files['metadata'], 'r') as f:
            existing_metadata = json.load(f)
        
        image_paths = existing_metadata.get('image_paths', [])
        print(f"   ✅ Metadata loaded: {len(image_paths)} image paths")
        
        return {
            'index': existing_index,
            'embeddings': existing_embeddings,
            'image_paths': image_paths,
            'metadata': existing_metadata,
            'files': files
        }
        
    except Exception as e:
        print(f"❌ Error loading existing index: {e}")
        return {}


def load_delta_index(delta_faiss: str, delta_embeddings: str, delta_metadata: str) -> Dict[str, Any]:
    """Load the delta index"""
    
    print(f"📖 Loading delta index...")
    print(f"   📁 FAISS: {delta_faiss}")
    print(f"   📁 Embeddings: {delta_embeddings}")
    print(f"   📁 Metadata: {delta_metadata}")
    
    try:
        # Load FAISS index
        delta_index = faiss.read_index(delta_faiss)
        print(f"   ✅ FAISS index loaded: {delta_index.ntotal} vectors")
        
        # Load embeddings
        delta_emb = np.load(delta_embeddings)
        print(f"   ✅ Embeddings loaded: {delta_emb.shape}")
        
        # Load metadata
        with open(delta_metadata, 'r') as f:
            delta_meta = json.load(f)
        
        image_paths = delta_meta.get('image_paths', [])
        print(f"   ✅ Metadata loaded: {len(image_paths)} image paths")
        
        return {
            'index': delta_index,
            'embeddings': delta_emb,
            'image_paths': image_paths,
            'metadata': delta_meta
        }
        
    except Exception as e:
        print(f"❌ Error loading delta index: {e}")
        return {}


def merge_indexes(existing_data: Dict[str, Any], delta_data: Dict[str, Any]) -> Dict[str, Any]:
    """Merge existing and delta indexes"""
    
    print(f"\n🔄 MERGING INDEXES")
    print("=" * 40)
    
    start_time = time.time()
    
    # Get dimensions and verify compatibility
    existing_embeddings = existing_data['embeddings']
    delta_embeddings = delta_data['embeddings']
    
    if existing_embeddings.shape[1] != delta_embeddings.shape[1]:
        print(f"❌ Embedding dimension mismatch!")
        print(f"   Existing: {existing_embeddings.shape[1]}")
        print(f"   Delta: {delta_embeddings.shape[1]}")
        return {}
    
    dimension = existing_embeddings.shape[1]
    existing_count = len(existing_embeddings)
    delta_count = len(delta_embeddings)
    total_count = existing_count + delta_count
    
    print(f"📊 Merge Statistics:")
    print(f"   📦 Existing embeddings: {existing_count:,}")
    print(f"   🆕 Delta embeddings: {delta_count:,}")
    print(f"   📈 Total after merge: {total_count:,}")
    print(f"   🔢 Embedding dimension: {dimension}")
    print()
    
    print(f"🔧 Creating merged index...")
    
    # Create new FAISS index
    merged_index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
    
    # Combine embeddings
    print(f"   📦 Combining embeddings...")
    merged_embeddings = np.vstack([existing_embeddings, delta_embeddings])
    
    # Convert to float32 for FAISS compatibility
    print(f"   🔄 Converting to float32...")
    merged_embeddings = merged_embeddings.astype('float32')
    
    # Normalize for cosine similarity
    print(f"   🔧 Normalizing embeddings...")
    faiss.normalize_L2(merged_embeddings)
    
    # Add to FAISS index
    print(f"   ➕ Adding embeddings to FAISS index...")
    merged_index.add(merged_embeddings)
    
    # Combine metadata
    print(f"   📝 Combining metadata...")
    merged_image_paths = existing_data['image_paths'] + delta_data['image_paths']
    
    # Create merged metadata
    merged_metadata = {
        'image_paths': merged_image_paths,
        'created_at': datetime.now().isoformat(),
        'merge_info': {
            'existing_index': 'v11_o00_index_1095',
            'existing_count': existing_count,
            'delta_count': delta_count,
            'total_count': total_count,
            'merge_timestamp': datetime.now().isoformat()
        },
        'model_info': {
            'base_model': 'gme-Qwen2-VL-7B-Instruct',
            'lora_checkpoint': 'loras/v11-20250620-105815/checkpoint-1095'
        },
        'index_type': 'merged',
        'embedding_dimension': dimension,
        'total_embeddings': total_count
    }
    
    merge_time = time.time() - start_time
    
    print(f"✅ Merge completed in {merge_time:.1f} seconds")
    
    return {
        'index': merged_index,
        'embeddings': merged_embeddings,
        'metadata': merged_metadata,
        'statistics': {
            'existing_count': existing_count,
            'delta_count': delta_count,
            'total_count': total_count,
            'merge_time': merge_time
        }
    }


def save_merged_index(merged_data: Dict[str, Any], output_prefix: str = "v11_complete_merged") -> Dict[str, str]:
    """Save the merged index"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output filenames
    files = {
        'faiss': f"indexes/{output_prefix}_{timestamp}.faiss",
        'embeddings': f"indexes/{output_prefix}_{timestamp}_embeddings.npy",
        'metadata': f"indexes/{output_prefix}_{timestamp}_metadata.json"
    }
    
    print(f"\n💾 SAVING MERGED INDEX")
    print("=" * 40)
    
    # Ensure indexes directory exists
    Path("indexes").mkdir(exist_ok=True)
    
    try:
        # Save FAISS index
        print(f"   💾 Saving FAISS index...")
        faiss.write_index(merged_data['index'], files['faiss'])
        faiss_size = Path(files['faiss']).stat().st_size / 1024 / 1024
        print(f"   ✅ FAISS index saved: {files['faiss']} ({faiss_size:.1f} MB)")
        
        # Save embeddings
        print(f"   💾 Saving embeddings...")
        np.save(files['embeddings'], merged_data['embeddings'])
        emb_size = Path(files['embeddings']).stat().st_size / 1024 / 1024
        print(f"   ✅ Embeddings saved: {files['embeddings']} ({emb_size:.1f} MB)")
        
        # Save metadata
        print(f"   💾 Saving metadata...")
        with open(files['metadata'], 'w') as f:
            json.dump(merged_data['metadata'], f, indent=2)
        meta_size = Path(files['metadata']).stat().st_size / 1024
        print(f"   ✅ Metadata saved: {files['metadata']} ({meta_size:.1f} KB)")
        
        total_size = faiss_size + emb_size + (meta_size / 1024)
        print(f"   📊 Total index size: {total_size:.1f} MB")
        
        return files
        
    except Exception as e:
        print(f"❌ Error saving merged index: {e}")
        return {}


def validate_merged_index(merged_files: Dict[str, str], expected_count: int) -> bool:
    """Validate the merged index"""
    
    print(f"\n🔍 VALIDATING MERGED INDEX")
    print("=" * 40)
    
    try:
        # Load and check FAISS index
        index = faiss.read_index(merged_files['faiss'])
        faiss_count = index.ntotal
        print(f"   📊 FAISS index vectors: {faiss_count:,}")
        
        # Load and check embeddings
        embeddings = np.load(merged_files['embeddings'])
        emb_count = len(embeddings)
        print(f"   📊 Embeddings count: {emb_count:,}")
        
        # Load and check metadata
        with open(merged_files['metadata'], 'r') as f:
            metadata = json.load(f)
        
        meta_count = len(metadata.get('image_paths', []))
        print(f"   📊 Metadata paths: {meta_count:,}")
        
        # Validate counts match
        if faiss_count == emb_count == meta_count == expected_count:
            print(f"   ✅ All counts match expected: {expected_count:,}")
            return True
        else:
            print(f"   ❌ Count mismatch!")
            print(f"      Expected: {expected_count:,}")
            print(f"      FAISS: {faiss_count:,}")
            print(f"      Embeddings: {emb_count:,}")
            print(f"      Metadata: {meta_count:,}")
            return False
            
    except Exception as e:
        print(f"   ❌ Validation error: {e}")
        return False


def main():
    """Main merge function"""
    
    print("🔗 INDEX MERGING - Delta + Existing")
    print("=" * 60)
    print("Merging v11_o00_index_1095 with delta index")
    print()
    
    # Find delta index
    delta_faiss, delta_embeddings, delta_metadata = find_latest_delta_index()
    
    if not delta_faiss:
        print("❌ No delta index found")
        print("   Run 'python index_delta_only.py' first")
        sys.exit(1)
    
    print(f"✅ Found delta index: {Path(delta_faiss).name}")
    
    # Load existing index
    existing_data = load_existing_index()
    if not existing_data:
        print("❌ Could not load existing index")
        sys.exit(1)
    
    # Load delta index
    delta_data = load_delta_index(delta_faiss, delta_embeddings, delta_metadata)
    if not delta_data:
        print("❌ Could not load delta index")
        sys.exit(1)
    
    # Check if we have data to merge
    if len(delta_data['embeddings']) == 0:
        print("⚠️  Delta index is empty - nothing to merge")
        return
    
    # Merge indexes
    merged_data = merge_indexes(existing_data, delta_data)
    if not merged_data:
        print("❌ Index merging failed")
        sys.exit(1)
    
    # Save merged index
    merged_files = save_merged_index(merged_data)
    if not merged_files:
        print("❌ Failed to save merged index")
        sys.exit(1)
    
    # Validate merged index
    expected_count = merged_data['statistics']['total_count']
    if validate_merged_index(merged_files, expected_count):
        print(f"\n🎉 MERGE SUCCESSFUL!")
        print("=" * 40)
        print(f"📊 Final Statistics:")
        print(f"   📦 Total images indexed: {expected_count:,}")
        print(f"   🔄 Existing images: {merged_data['statistics']['existing_count']:,}")
        print(f"   🆕 New images added: {merged_data['statistics']['delta_count']:,}")
        print(f"   ⏱️  Merge time: {merged_data['statistics']['merge_time']:.1f} seconds")
        print()
        print(f"📁 Merged index files:")
        for file_type, file_path in merged_files.items():
            print(f"   📄 {file_type.upper()}: {file_path}")
        print()
        print(f"💡 Next steps:")
        print(f"   1. Test the merged index with search queries")
        print(f"   2. Update your applications to use the new merged index")
        print(f"   3. Consider backing up the original indexes")
        print(f"   4. Clean up intermediate delta files if desired")
    else:
        print(f"\n❌ MERGE VALIDATION FAILED")
        print("   Check the merged index files for issues")


if __name__ == "__main__":
    main() 