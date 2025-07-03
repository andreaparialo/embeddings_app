#!/usr/bin/env python3
"""
Verify the current state of copied files
"""

from pathlib import Path

def verify_state():
    # Check db_pictures_512
    db_pictures = Path("db_pictures_512")
    db_files = list(db_pictures.glob("*.jpg")) + list(db_pictures.glob("*.JPG"))
    print(f"📁 Files in db_pictures_512: {len(db_files)}")
    
    # Check IAN/resized
    ian_path = Path("IAN/resized")
    ian_files = list(ian_path.glob("*"))
    print(f"📁 Files in IAN/resized: {len(ian_files)}")
    
    # Check how many have _P02 pattern
    p02_files = [f for f in ian_files if '_P' in f.stem]
    print(f"📊 Files with _P suffix in IAN: {len(p02_files)}")
    
    # Sample some recent files in db_pictures_512
    recent_files = sorted(db_files, key=lambda x: x.stat().st_mtime)[-10:]
    print("\n🕐 10 most recently added files in db_pictures_512:")
    for f in recent_files:
        print(f"   - {f.name}")
    
    # Check if there are corresponding files
    print("\n🔍 Checking correspondence:")
    sample_ian = list(ian_path.glob("*"))[:5]
    for ian_file in sample_ian:
        base_name = ian_file.stem.split('_P')[0] if '_P' in ian_file.stem else ian_file.stem
        expected_db_name = base_name + '.jpg'
        exists = (db_pictures / expected_db_name).exists()
        print(f"   IAN: {ian_file.name} → DB: {expected_db_name} {'✅' if exists else '❌'}")

if __name__ == "__main__":
    verify_state() 