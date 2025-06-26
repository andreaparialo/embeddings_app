#!/usr/bin/env python3
"""
Image Resizer with White Padding
Resizes images to a target size using white padding to maintain aspect ratio,
then moves them to a destination folder.
"""

import os
import shutil
from PIL import Image, ImageOps
from pathlib import Path
import argparse

# ============================================================================
# CONFIGURATION - Edit these variables as needed
# ============================================================================

# Source folder containing images to resize
SOURCE_FOLDER = "rel20260101/pictures_O00_20260101"

# Destination folder where resized images will be moved
DESTINATION_FOLDER = "pictures"

# Target dimensions (width, height)
TARGET_WIDTH = 1456
TARGET_HEIGHT = 819

# Background color for padding (white)
PADDING_COLOR = (255, 255, 255)

# Supported image extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# ============================================================================


def resize_image_with_padding(image_path, target_width, target_height, padding_color=(255, 255, 255)):
    """
    Resize an image to target dimensions using white padding to maintain aspect ratio.
    
    Args:
        image_path (str): Path to the input image
        target_width (int): Target width
        target_height (int): Target height
        padding_color (tuple): RGB color for padding (default: white)
    
    Returns:
        PIL.Image: Resized image with padding
    """
    with Image.open(image_path) as img:
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Calculate scaling factor to fit within target dimensions
        width_ratio = target_width / img.width
        height_ratio = target_height / img.height
        scale_factor = min(width_ratio, height_ratio)
        
        # Calculate new dimensions
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * scale_factor)
        
        # Resize the image
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create a new image with target dimensions and padding color
        result = Image.new('RGB', (target_width, target_height), padding_color)
        
        # Calculate position to center the resized image
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Paste the resized image onto the padded background
        result.paste(img_resized, (x_offset, y_offset))
        
        return result


def get_image_files(folder_path):
    """
    Get list of supported image files in a folder.
    
    Args:
        folder_path (str): Path to the folder
    
    Returns:
        list: List of image file paths
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Source folder '{folder_path}' does not exist")
    
    image_files = []
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_files.append(file)
    
    return sorted(image_files)


def process_images(source_folder, destination_folder, target_width, target_height, padding_color, dry_run=False):
    """
    Process all images in the source folder.
    
    Args:
        source_folder (str): Source folder path
        destination_folder (str): Destination folder path
        target_width (int): Target width
        target_height (int): Target height
        padding_color (tuple): RGB color for padding
        dry_run (bool): If True, only show what would be done without actually processing
    
    Returns:
        tuple: (processed_count, skipped_count, error_count)
    """
    # Create destination folder if it doesn't exist
    dest_path = Path(destination_folder)
    if not dry_run:
        dest_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    try:
        image_files = get_image_files(source_folder)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 0, 0, 1
    
    if not image_files:
        print(f"No supported image files found in '{source_folder}'")
        return 0, 0, 0
    
    print(f"Found {len(image_files)} image files to process")
    if dry_run:
        print("DRY RUN MODE - No files will be modified")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, image_file in enumerate(image_files, 1):
        try:
            print(f"Processing {i}/{len(image_files)}: {image_file.name}")
            
            # Handle P02 to O00 renaming
            dest_filename = image_file.name
            if '_P02.' in dest_filename:
                dest_filename = dest_filename.replace('_P02.', '_O00.')
                print(f"  Renaming: {image_file.name} → {dest_filename}")
            
            # Check if destination file already exists
            dest_file = dest_path / dest_filename
            if dest_file.exists() and not dry_run:
                print(f"  Warning: '{dest_file}' already exists, skipping...")
                skipped_count += 1
                continue
            
            if not dry_run:
                # Resize the image
                resized_img = resize_image_with_padding(
                    image_file, target_width, target_height, padding_color
                )
                
                # Save to destination
                resized_img.save(dest_file, 'JPEG', quality=95)
                
                # Remove original file (move operation)
                image_file.unlink()
                
                print(f"  ✓ Resized and moved to '{dest_file}'")
            else:
                print(f"  Would resize and move to '{dest_file}'")
            
            processed_count += 1
            
        except Exception as e:
            print(f"  ✗ Error processing '{image_file.name}': {e}")
            error_count += 1
    
    return processed_count, skipped_count, error_count


def main():
    parser = argparse.ArgumentParser(description='Resize images with white padding and move to destination folder')
    parser.add_argument('--source', '-s', default=SOURCE_FOLDER,
                        help=f'Source folder (default: {SOURCE_FOLDER})')
    parser.add_argument('--destination', '-d', default=DESTINATION_FOLDER,
                        help=f'Destination folder (default: {DESTINATION_FOLDER})')
    parser.add_argument('--width', '-w', type=int, default=TARGET_WIDTH,
                        help=f'Target width (default: {TARGET_WIDTH})')
    parser.add_argument('--height', type=int, default=TARGET_HEIGHT,
                        help=f'Target height (default: {TARGET_HEIGHT})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without actually processing files')
    
    args = parser.parse_args()
    
    print("Image Resizer with White Padding")
    print("=" * 40)
    print(f"Source folder: {args.source}")
    print(f"Destination folder: {args.destination}")
    print(f"Target dimensions: {args.width}x{args.height}")
    print(f"Padding color: RGB{PADDING_COLOR}")
    print()
    
    # Confirm before processing (unless dry run)
    if not args.dry_run:
        response = input("Proceed with processing? (y/n): ").lower().strip()
        if response != 'y':
            print("Operation cancelled.")
            return
    
    # Process images
    processed, skipped, errors = process_images(
        args.source, args.destination, args.width, args.height, 
        PADDING_COLOR, args.dry_run
    )
    
    # Summary
    print()
    print("Summary:")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    
    if not args.dry_run and processed > 0:
        print(f"\n✓ Successfully processed {processed} images!")


if __name__ == "__main__":
    main() 