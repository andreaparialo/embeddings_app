#!/usr/bin/env python3
"""
Resize all images in db_pictures to 512x512 with white padding (maintaining aspect ratio)
"""

import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def resize_image_with_padding(image_path, output_path, target_size=512):
    """
    Resize image to target_size x target_size with white padding
    """
    try:
        # Open image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get current dimensions
        width, height = img.size
        
        # Calculate scaling factor (maintain aspect ratio)
        scale = min(target_size / width, target_size / height)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create white background
        background = Image.new('RGB', (target_size, target_size), (255, 255, 255))
        
        # Calculate position to center the image
        x_offset = (target_size - new_width) // 2
        y_offset = (target_size - new_height) // 2
        
        # Paste resized image onto white background
        background.paste(img_resized, (x_offset, y_offset))
        
        # Save the result
        background.save(output_path, 'JPEG', quality=95)
        
        return True, None
    except Exception as e:
        return False, str(e)

def process_batch(batch_info):
    """Process a batch of images"""
    successes = 0
    errors = []
    
    for image_path, output_path in batch_info:
        success, error = resize_image_with_padding(image_path, output_path)
        if success:
            successes += 1
        else:
            errors.append((image_path, error))
    
    return successes, errors

def resize_all_db_pictures():
    print("🖼️  Resizing DB Pictures to 512x512")
    print("=" * 80)
    
    # Directories
    input_dir = "db_pictures"
    output_dir = "db_pictures_512"
    
    if not os.path.exists(input_dir):
        print(f"❌ Input directory '{input_dir}' not found!")
        print("Please run create_db_pictures_folder.py first.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"📊 Found {len(image_files):,} images to resize")
    
    # Prepare batch data
    batch_data = []
    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        batch_data.append((input_path, output_path))
    
    # Process in parallel
    num_workers = min(multiprocessing.cpu_count(), 8)
    print(f"\n🚀 Processing with {num_workers} workers...")
    
    # Split data into batches
    batch_size = max(1, len(batch_data) // (num_workers * 10))
    batches = [batch_data[i:i + batch_size] for i in range(0, len(batch_data), batch_size)]
    
    total_processed = 0
    total_errors = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all batches
        futures = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}
        
        # Process results as they complete
        with tqdm(total=len(image_files), desc="Resizing images") as pbar:
            for future in as_completed(futures):
                successes, errors = future.result()
                total_processed += successes
                total_errors.extend(errors)
                pbar.update(successes + len(errors))
    
    # Summary
    print(f"\n📊 Resize Summary:")
    print(f"  ✅ Successfully resized: {total_processed:,} images")
    print(f"  ❌ Errors: {len(total_errors)} images")
    print(f"  📁 Output directory: {output_dir}")
    print(f"  🎯 Target size: 512x512 pixels")
    
    # Calculate total size
    total_size = sum(os.path.getsize(os.path.join(output_dir, f)) 
                    for f in os.listdir(output_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    print(f"  💾 Total size: {total_size / (1024**3):.2f} GB")
    
    # Save error report if any
    if total_errors:
        error_report_path = f"{output_dir}_errors.txt"
        with open(error_report_path, 'w') as f:
            f.write("Resize Errors Report\n")
            f.write("===================\n")
            f.write(f"Total errors: {len(total_errors)}\n\n")
            for img_path, error in total_errors:
                f.write(f"{img_path}: {error}\n")
        print(f"\n📝 Error report saved to: {error_report_path}")
    
    # Show sample info
    if total_processed > 0:
        sample_path = os.path.join(output_dir, os.listdir(output_dir)[0])
        sample_img = Image.open(sample_path)
        print(f"\n🔍 Sample image info:")
        print(f"  Size: {sample_img.size}")
        print(f"  Mode: {sample_img.mode}")
    
    return output_dir, total_processed

if __name__ == "__main__":
    resize_all_db_pictures() 