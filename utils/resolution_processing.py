#!/usr/bin/env python3
"""Resolution processing for the Object Reasoning dataset images."""

import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from PIL import Image
except ImportError:
    print("Error: Please install the Pillow library first: pip install Pillow")
    exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm is not installed, will use simple progress display. Install tqdm for more detailed progress: pip install tqdm")


def process_image(args):
    """Process a single image (crop+resize if it has a geo/conf field, else copy). For multiprocessing."""
    src_path, dst_path, has_geo = args

    try:
        if not has_geo:
            shutil.copy2(src_path, dst_path)
            return True, src_path.name, "copied"
        else:
            with Image.open(src_path) as img:
                width, height = img.size

                # Center-crop 1280x720 -> 1080x720 (100px off each side)
                if width == 1280 and height == 720:
                    left = (width - 1080) // 2
                    top = 0
                    right = left + 1080
                    bottom = 720
                    img_cropped = img.crop((left, top, right, bottom))
                else:
                    # Non-standard size: center crop, clamped to bounds
                    target_width = 1080
                    target_height = 720
                    left = (width - target_width) // 2
                    top = (height - target_height) // 2
                    right = left + target_width
                    bottom = top + target_height
                    left = max(0, left)
                    top = max(0, top)
                    right = min(width, right)
                    bottom = min(height, bottom)
                    img_cropped = img.crop((left, top, right, bottom))

                img_resized = img_cropped.resize((480, 320), Image.Resampling.LANCZOS)
                img_resized.save(dst_path, optimize=True)
                return True, src_path.name, "processed"
    except Exception as e:
        return False, src_path.name, str(e)


def main():
    script_dir = Path(__file__).parent
    images_dir = script_dir / "images"
    images_ver2_dir = script_dir / "images_ver2"

    images_ver2_dir.mkdir(exist_ok=True)

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'}
    all_images = [f for f in images_dir.iterdir()
                  if f.is_file() and f.suffix.lower() in image_extensions]

    print(f"Found {len(all_images)} image files")

    # Split by presence of geo/conf marker in the filename
    images_with_geo = []
    images_without_geo = []

    for img_path in all_images:
        if 'conf' in img_path.name.lower():
            images_with_geo.append(img_path)
        else:
            images_without_geo.append(img_path)

    print(f"Images with geo field: {len(images_with_geo)}")
    print(f"Images without geo field: {len(images_without_geo)}")

    tasks = []
    for img_path in images_with_geo:
        dst_path = images_ver2_dir / img_path.name
        tasks.append((img_path, dst_path, True))

    for img_path in images_without_geo:
        dst_path = images_ver2_dir / img_path.name
        tasks.append((img_path, dst_path, False))

    num_workers = min(os.cpu_count() or 4, 8)
    print(f"Using {num_workers} processes for parallel processing...")

    success_count = 0
    error_count = 0
    errors = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_image, task): task for task in tasks}

        if HAS_TQDM:
            pbar = tqdm(total=len(tasks), desc="Processing images")
        else:
            print("Start processing images...")
            processed = 0

        for future in as_completed(futures):
            success, filename, status = future.result()
            if success:
                success_count += 1
            else:
                error_count += 1
                errors.append((filename, status))

            if HAS_TQDM:
                pbar.update(1)
            else:
                processed += 1
                if processed % 100 == 0:
                    print(f"Processed: {processed}/{len(tasks)}")

        if HAS_TQDM:
            pbar.close()
        else:
            print(f"Processing completed: {processed}/{len(tasks)}")

    print(f"\nProcessing completed!")
    print(f"Success: {success_count}")
    if error_count > 0:
        print(f"Failed: {error_count}")
        print("\nError details:")
        for filename, error in errors[:10]:
            print(f"  {filename}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    # Verify images_ver2 mirrors images exactly
    print("\nVerifying file completeness...")
    ver2_files = {f.name for f in images_ver2_dir.iterdir() if f.is_file()}
    original_files = {f.name for f in all_images}

    missing_files = original_files - ver2_files
    extra_files = ver2_files - original_files

    if not missing_files and not extra_files:
        print("Verification passed: images_ver2 matches images with no extra files")
    else:
        if missing_files:
            print(f"Missing {len(missing_files)} files:")
            for f in list(missing_files)[:10]:
                print(f"  - {f}")
            if len(missing_files) > 10:
                print(f"  ... and {len(missing_files) - 10} more files")
        if extra_files:
            print(f"Extra {len(extra_files)} files:")
            for f in list(extra_files)[:10]:
                print(f"  - {f}")
            if len(extra_files) > 10:
                print(f"  ... and {len(extra_files) - 10} more files")

    print(f"\nAll processed images saved to: {images_ver2_dir.absolute()}")


if __name__ == "__main__":
    main()
