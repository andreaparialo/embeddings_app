#!/usr/bin/env python3
"""
Create db_pictures folder containing only pictures referenced in the database
"""

import pandas as pd
import os
import shutil
from tqdm import tqdm
from datetime import datetime

def create_db_pictures_folder():
    print("📁 Creating DB Pictures Folder")
    print("=" * 80)
    
    # Load the new database
    db_path = "database_results/DB_FINAL_SIMILARIT_270615.csv"
    print(f"\n📊 Loading database: {db_path}")
    df = pd.read_csv(db_path)
    
    # Get unique filename_roots
    unique_roots = df['filename_root'].unique()
    print(f"✅ Found {len(unique_roots):,} unique filename_roots in database")
    
    # Create output directory
    output_dir = "db_pictures"
    if os.path.exists(output_dir):
        print(f"\n⚠️  {output_dir} already exists. Backing up...")
        backup_dir = f"{output_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move(output_dir, backup_dir)
        print(f"✅ Backed up to: {backup_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Created directory: {output_dir}")
    
    # Source directory
    source_dir = "pictures"
    
    # Copy matching pictures
    copied_count = 0
    missing_count = 0
    missing_roots = []
    
    print(f"\n🔍 Searching for pictures in {source_dir}...")
    
    for root in tqdm(unique_roots, desc="Processing images"):
        found = False
        
        # Try different variations
        for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            for suffix in ['_O00', '']:
                source_filename = f"{root}{suffix}{ext}"
                source_path = os.path.join(source_dir, source_filename)
                
                if os.path.exists(source_path):
                    # Copy to db_pictures with consistent naming
                    dest_filename = f"{root}.jpg"  # Standardize to .jpg
                    dest_path = os.path.join(output_dir, dest_filename)
                    
                    try:
                        shutil.copy2(source_path, dest_path)
                        copied_count += 1
                        found = True
                        break
                    except Exception as e:
                        print(f"\n❌ Error copying {source_filename}: {e}")
                
            if found:
                break
        
        if not found:
            missing_count += 1
            missing_roots.append(root)
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"  ✅ Copied: {copied_count:,} pictures")
    print(f"  ❌ Missing: {missing_count:,} pictures")
    print(f"  📁 Output directory: {output_dir}")
    print(f"  💾 Total size: {sum(os.path.getsize(os.path.join(output_dir, f)) for f in os.listdir(output_dir)) / (1024**3):.2f} GB")
    
    # Save missing roots report
    if missing_roots:
        missing_report_path = f"{output_dir}_missing_report.txt"
        with open(missing_report_path, 'w') as f:
            f.write(f"Missing Pictures Report\n")
            f.write(f"======================\n")
            f.write(f"Total missing: {len(missing_roots)}\n\n")
            for root in missing_roots:
                f.write(f"{root}\n")
        print(f"\n📝 Missing pictures report saved to: {missing_report_path}")
    
    return output_dir, copied_count, missing_count

if __name__ == "__main__":
    create_db_pictures_folder() 