#!/usr/bin/env python3
"""
Create a precise report of missing indexes with picture availability
"""

import pandas as pd
import json
import os
import glob

def create_missing_indexes_report():
    print("📊 Creating Missing Indexes Report")
    print("=" * 80)
    
    # Load reduction plan to get missing roots
    with open('faiss_reduction_plan.json', 'r') as f:
        reduction_plan = json.load(f)
    
    missing_roots = reduction_plan['missing_roots']
    print(f"\nTotal missing embeddings: {len(missing_roots):,}")
    
    # Check pictures directory
    pictures_dir = "pictures"
    
    # Get all image files
    jpg_files = glob.glob(os.path.join(pictures_dir, "*.jpg"))
    JPG_files = glob.glob(os.path.join(pictures_dir, "*.JPG"))
    all_pictures = jpg_files + JPG_files
    
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
    
    # Create report data
    report_data = []
    
    for filename_root in sorted(missing_roots):
        # Extract MODEL_COD (first 6 digits)
        model_cod = filename_root[:6]
        
        # Check if picture exists
        has_picture = filename_root in picture_roots
        present_status = "YES" if has_picture else "NO"
        
        report_data.append({
            'MODEL_COD': model_cod,
            'filename_root': filename_root,
            'PRESENT': present_status
        })
    
    # Create DataFrame
    report_df = pd.DataFrame(report_data)
    
    # Sort by PRESENT status (NO first) and then by MODEL_COD
    report_df = report_df.sort_values(['PRESENT', 'MODEL_COD', 'filename_root'])
    
    # Save to CSV
    output_file = 'missing_indexes_picture_status.csv'
    report_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved report to: {output_file}")
    
    # Print summary statistics
    print("\n📊 Summary:")
    print(f"  Total missing indexes: {len(report_df):,}")
    print(f"  With pictures (YES): {len(report_df[report_df['PRESENT'] == 'YES']):,}")
    print(f"  Without pictures (NO): {len(report_df[report_df['PRESENT'] == 'NO']):,}")
    
    # Show sample of each category
    print("\n📋 Sample entries WITHOUT pictures:")
    no_pics = report_df[report_df['PRESENT'] == 'NO'].head(10)
    for idx, row in no_pics.iterrows():
        print(f"  {row['MODEL_COD']} | {row['filename_root']} | {row['PRESENT']}")
    
    print("\n📋 Sample entries WITH pictures:")
    yes_pics = report_df[report_df['PRESENT'] == 'YES'].head(10)
    for idx, row in yes_pics.iterrows():
        print(f"  {row['MODEL_COD']} | {row['filename_root']} | {row['PRESENT']}")
    
    # Also create separate files for easier processing
    # File for missing pictures
    missing_pics_df = report_df[report_df['PRESENT'] == 'NO']
    missing_pics_df.to_csv('missing_indexes_need_pictures.csv', index=False)
    print(f"\n💾 Created file for indexes needing pictures: missing_indexes_need_pictures.csv")
    print(f"   Contains {len(missing_pics_df):,} entries")
    
    # File for available pictures
    available_pics_df = report_df[report_df['PRESENT'] == 'YES']
    available_pics_df.to_csv('missing_indexes_have_pictures.csv', index=False)
    print(f"\n💾 Created file for indexes with pictures: missing_indexes_have_pictures.csv")
    print(f"   Contains {len(available_pics_df):,} entries")
    
    print("\n✅ Report generation complete!")
    print("\n📁 Generated files:")
    print("  1. missing_indexes_picture_status.csv - Complete report with all missing indexes")
    print("  2. missing_indexes_need_pictures.csv - Only indexes that need pictures")
    print("  3. missing_indexes_have_pictures.csv - Only indexes that have pictures")
    
    return report_df

if __name__ == "__main__":
    create_missing_indexes_report() 