#!/usr/bin/env python3
"""
Image preparation script - Resize images to 512px max dimension while maintaining aspect ratio
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImagePreparer:
    """Efficient image preparation with multi-threading"""
    
    def __init__(self, max_size: int = 512, quality: int = 95, num_workers: int = 8):
        self.max_size = max_size
        self.quality = quality
        self.num_workers = num_workers
        self.processed_count = 0
        self.error_count = 0
        self.skipped_count = 0
        
    def resize_image(self, input_path: Path, output_path: Path) -> bool:
        """Resize a single image maintaining aspect ratio"""
        try:
            # Check if output already exists and is newer
            if output_path.exists():
                if output_path.stat().st_mtime >= input_path.stat().st_mtime:
                    self.skipped_count += 1
                    return True  # Skip if already processed and newer
            
            # Open and process image
            with Image.open(input_path) as img:
                # Convert to RGB if necessary (handles RGBA, etc.)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Get current dimensions
                width, height = img.size
                
                # Skip if already small enough
                if max(width, height) <= self.max_size:
                    # Just copy the file if it's already small enough
                    img.save(output_path, 'JPEG', quality=self.quality, optimize=True)
                    self.processed_count += 1
                    return True
                
                # Calculate new dimensions maintaining aspect ratio
                if width > height:
                    new_width = self.max_size
                    new_height = int((height * self.max_size) / width)
                else:
                    new_height = self.max_size
                    new_width = int((width * self.max_size) / height)
                
                # Resize image using high-quality resampling
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Save with optimization
                resized_img.save(output_path, 'JPEG', quality=self.quality, optimize=True)
                
                self.processed_count += 1
                return True
                
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            self.error_count += 1
            return False
    
    def prepare_images(self, source_dir: str, target_dir: str):
        """Prepare all images in source directory"""
        
        print("🖼️" * 60)
        print("🚀 IMAGE PREPARATION - RESIZE TO 512px")
        print("🖼️" * 60)
        
        source_path = Path(source_dir)
        target_path = Path(target_dir)
        
        if not source_path.exists():
            raise ValueError(f"Source directory {source_dir} does not exist")
        
        # Create target directory
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP', '.TIFF', '.WEBP']
        image_files = []
        for ext in supported_formats:
            image_files.extend(source_path.rglob(f"*{ext}"))
        
        if not image_files:
            raise ValueError(f"No image files found in {source_dir}")
        
        print(f"📁 Found {len(image_files)} images to prepare")
        print(f"📏 Target size: {self.max_size}px (max dimension)")
        print(f"🎨 Quality: {self.quality}%")
        print(f"👥 Workers: {self.num_workers}")
        print(f"📂 Source: {source_dir}")
        print(f"📂 Target: {target_dir}")
        print()
        
        # Prepare tasks for threading
        tasks = []
        for img_file in image_files:
            # Maintain directory structure
            relative_path = img_file.relative_to(source_path)
            output_file = target_path / relative_path.with_suffix('.jpg')  # Convert all to JPEG
            
            # Create output directory if needed
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            tasks.append((img_file, output_file))
        
        # Process images with thread pool
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.resize_image, input_path, output_path): (input_path, output_path)
                for input_path, output_path in tasks
            }
            
            # Process results with progress bar
            with tqdm(total=len(tasks), desc="🖼️ Preparing Images") as pbar:
                for future in as_completed(future_to_task):
                    input_path, output_path = future_to_task[future]
                    try:
                        success = future.result()
                        pbar.update(1)
                        
                        # Update progress info
                        pbar.set_postfix({
                            'processed': self.processed_count,
                            'skipped': self.skipped_count,
                            'errors': self.error_count
                        })
                        
                    except Exception as e:
                        logger.error(f"Task failed for {input_path}: {e}")
                        self.error_count += 1
                        pbar.update(1)
        
        # Summary
        total_handled = self.processed_count + self.skipped_count
        print(f"\n🎯 IMAGE PREPARATION RESULTS:")
        print(f"   📊 Total images: {len(image_files)}")
        print(f"   ✅ Processed: {self.processed_count}")
        print(f"   ⏭️  Skipped (already done): {self.skipped_count}")
        print(f"   ❌ Errors: {self.error_count}")
        print(f"   💡 Success rate: {total_handled/len(image_files)*100:.1f}%")
        print(f"   📂 Output directory: {target_dir}")
        
        if self.error_count > 0:
            print(f"   ⚠️  {self.error_count} images had errors - check logs above")
        
        return total_handled


def main():
    """Main function for image preparation"""
    
    # Configuration
    source_directory = "pictures"
    target_directory = "pictures_prepared"
    max_size = 512
    quality = 95  # JPEG quality (95 is very high quality)
    num_workers = 8  # Thread workers for I/O operations
    
    # Check source exists
    if not Path(source_directory).exists():
        print(f"❌ Source directory not found: {source_directory}")
        sys.exit(1)
    
    # Count source images
    source_path = Path(source_directory)
    supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP', '.TIFF', '.WEBP']
    source_images = []
    for ext in supported_formats:
        source_images.extend(source_path.rglob(f"*{ext}"))
    
    print(f"🎯 Image Preparation Configuration:")
    print(f"   Source directory: {source_directory}")
    print(f"   Target directory: {target_directory}")
    print(f"   Max dimension: {max_size}px")
    print(f"   JPEG quality: {quality}%")
    print(f"   Thread workers: {num_workers}")
    print(f"   Source images found: {len(source_images)}")
    print()
    
    if len(source_images) == 0:
        print("❌ No images found in source directory")
        sys.exit(1)
    
    # Estimate space savings
    print("💡 Benefits of preparation:")
    print("   • Faster indexing (smaller images)")
    print("   • Lower memory usage during processing")
    print("   • Consistent image sizes")
    print("   • Optimized JPEG compression")
    print()
    
    # Run preparation
    try:
        preparer = ImagePreparer(
            max_size=max_size,
            quality=quality,
            num_workers=num_workers
        )
        
        result = preparer.prepare_images(source_directory, target_directory)
        
        print(f"\n✅ Image preparation completed!")
        print(f"   📂 Prepared images ready in: {target_directory}")
        print(f"   🚀 Now you can run indexing on the prepared images for faster processing!")
        
    except Exception as e:
        print(f"❌ Image preparation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 