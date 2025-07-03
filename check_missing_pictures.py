#!/usr/bin/env python3
"""
Check if pictures exist for missing filename_roots
"""

import pandas as pd
import json
import os
import glob

def check_missing_pictures():
    print("🔍 Checking Pictures for Missing Filename Roots")
    print("=" * 80)
    
    # Load reduction plan to get missing roots
    with open('faiss_reduction_plan.json', 'r') as f:
        reduction_plan = json.load(f)
    
    missing_roots = reduction_plan['missing_roots']
    print(f"\n📊 Total missing embeddings: {len(missing_roots):,}")
    
    # Check pictures directory
    pictures_dir = "pictures"
    print(f"\n📁 Checking {pictures_dir} directory...")
    
    # Get all image files
    jpg_files = glob.glob(os.path.join(pictures_dir, "*.jpg"))
    JPG_files = glob.glob(os.path.join(pictures_dir, "*.JPG"))
    all_pictures = jpg_files + JPG_files
    print(f"  Total picture files: {len(all_pictures):,}")
    
    # Extract filename_roots from pictures
    picture_roots = set()
    for pic_path in all_pictures:
        filename = os.path.basename(pic_path)
        # Get filename_root (everything before first underscore or extension)
        if '_' in filename:
            root = filename.split('_')[0]
        else:
            root = filename.split('.')[0]
        picture_roots.add(root)
    
    print(f"  Unique filename_roots in pictures: {len(picture_roots):,}")
    
    # Check which missing roots have pictures
    missing_with_pictures = []
    missing_without_pictures = []
    
    for root in missing_roots:
        if root in picture_roots:
            missing_with_pictures.append(root)
        else:
            missing_without_pictures.append(root)
    
    print(f"\n📊 Picture Availability for Missing Embeddings:")
    print(f"  ✅ Have pictures: {len(missing_with_pictures):,} ({len(missing_with_pictures)/len(missing_roots)*100:.1f}%)")
    print(f"  ❌ No pictures: {len(missing_without_pictures):,} ({len(missing_without_pictures)/len(missing_roots)*100:.1f}%)")
    
    # Load new database for more details
    new_df = pd.read_csv('database_results/DB_FINAL_SIMILARIT_270615.csv')
    
    # Analyze missing with pictures by year
    if missing_with_pictures:
        print(f"\n🖼️  Missing Embeddings WITH Pictures:")
        with_pics_df = new_df[new_df['filename_root'].isin(missing_with_pictures)]
        with_pics_df['STARTSKU_DATE'] = pd.to_datetime(with_pics_df['STARTSKU_DATE'])
        
        year_counts = with_pics_df.groupby(with_pics_df['STARTSKU_DATE'].dt.year)['filename_root'].nunique().sort_index()
        print("\n  Distribution by year:")
        for year, count in year_counts.items():
            print(f"    {year}: {count} products")
        
        # Sample products with pictures
        print("\n  Sample products (first 10):")
        sample = with_pics_df.groupby('filename_root').first().head(10)
        for idx, row in sample.iterrows():
            # Check actual file
            found_files = []
            for ext in ['.jpg', '.JPG']:
                for suffix in ['_O00', '']:
                    test_path = os.path.join(pictures_dir, f"{idx}{suffix}{ext}")
                    if os.path.exists(test_path):
                        found_files.append(os.path.basename(test_path))
            
            print(f"    - {idx} | {row['BRAND_DES']} | Year: {row['STARTSKU_DATE'].year} | Files: {found_files}")
    
    # Analyze missing without pictures by year
    if missing_without_pictures:
        print(f"\n❌ Missing Embeddings WITHOUT Pictures:")
        without_pics_df = new_df[new_df['filename_root'].isin(missing_without_pictures)]
        without_pics_df['STARTSKU_DATE'] = pd.to_datetime(without_pics_df['STARTSKU_DATE'])
        
        year_counts = without_pics_df.groupby(without_pics_df['STARTSKU_DATE'].dt.year)['filename_root'].nunique().sort_index()
        print("\n  Distribution by year:")
        for year, count in year_counts.items():
            print(f"    {year}: {count} products")
        
        # Sample products without pictures
        print("\n  Sample products (first 10):")
        sample = without_pics_df.groupby('filename_root').first().head(10)
        for idx, row in sample.iterrows():
            print(f"    - {idx} | {row['BRAND_DES']} | Year: {row['STARTSKU_DATE'].year}")
    
    # Summary recommendations
    print(f"\n💡 Summary and Recommendations:")
    print(f"\n1. Products we CAN index (have pictures): {len(missing_with_pictures):,}")
    if missing_with_pictures:
        current_year_count = len([r for r in missing_with_pictures 
                                 if new_df[new_df['filename_root'] == r]['STARTSKU_DATE'].min().year <= 2024])
        future_year_count = len(missing_with_pictures) - current_year_count
        print(f"   - Current/past products (≤2024): {current_year_count}")
        print(f"   - Future products (2025-2026): {future_year_count}")
    
    print(f"\n2. Products we CANNOT index (no pictures): {len(missing_without_pictures):,}")
    print("   - These are likely future products not yet photographed")
    
    # Save detailed report
    report = {
        'total_missing_embeddings': len(missing_roots),
        'have_pictures': len(missing_with_pictures),
        'no_pictures': len(missing_without_pictures),
        'percentage_with_pictures': round(len(missing_with_pictures)/len(missing_roots)*100, 1),
        'missing_with_pictures': missing_with_pictures,
        'missing_without_pictures': missing_without_pictures
    }
    
    with open('missing_pictures_analysis.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Analysis complete! Report saved to missing_pictures_analysis.json")
    
    return missing_with_pictures, missing_without_pictures

if __name__ == "__main__":
    check_missing_pictures() 