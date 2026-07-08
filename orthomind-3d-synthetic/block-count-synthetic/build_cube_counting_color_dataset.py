#!/usr/bin/env python3
"""
Build a cube counting dataset script (color version)

From the rendered data in OUTPUT_DIR, extract data and build a standard VLM dataset format.
For each color, generate a question asking for the number of blocks of that color.
Support splitting the dataset into eval and train two parts.
"""

import json
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from utils import generate_voxel_matrix

# Mapping from color ID to color name
COLOR_ID_TO_NAME = {
    1: "black",
    2: "white",
    3: "red",
    4: "yellow",
    5: "blue",
    6: "green",
}


def generate_model_generation(color_name: str, count: int) -> str:
    """Generate the model_generation field, following the example format"""
    # Generate the reasoning process template
    reasoning = f"The user now needs to count the number of {color_name} cubes in the image. Looking at the picture, there are {count} {color_name} cube{'s' if count != 1 else ''}, so the count is {count}."
    
    model_generation = f"<think>{reasoning}</think>\n\\boxed{{{count}}}"
    return model_generation


def count_visible_blocks_by_color(color_matrix, voxel_matrix):
    """
    Count the number of visible blocks for each color
    
    Args:
        color_matrix: color matrix [y][x][z], value is color ID (1-6), 0 represents empty
        voxel_matrix: visibility matrix [x][y][z], 1 represents visible, 0 represents empty, -1 represents blocked
    
    Returns:
        dict: {color_id: count} the number of visible blocks for each color
    """
    color_counts = {color_id: 0 for color_id in range(1, 7)}
    
    # Traverse the visibility matrix
    R = len(voxel_matrix)
    if R == 0:
        return color_counts
    
    C = len(voxel_matrix[0]) if R > 0 else 0
    H = len(voxel_matrix[0][0]) if C > 0 else 0
    
    for x in range(R):
        for y in range(C):
            for z in range(H):
                # Only count the visible blocks (voxel_matrix[x][y][z] == 1)
                if voxel_matrix[x][y][z] == 1:
                    # Get the corresponding color (note the index order: color_matrix[y][x][z])
                    if y < len(color_matrix) and x < len(color_matrix[y]) and z < len(color_matrix[y][x]):
                        color_id = color_matrix[y][x][z]
                        if color_id > 0 and color_id <= 6:
                            color_counts[color_id] += 1
    
    return color_counts


def process_level(level_dir: Path) -> List[Tuple[Dict[str, Any], Path]]:
    """
    Process all cases in a level directory
    
    Args:
        level_dir: level directory path (e.g. OUTPUT_COLOR/level333)
    
    Returns:
        the list of data records for this level, each element is a tuple of (record, source_image_path)
    """
    level_name = level_dir.name  # e.g. "level333"
    records = []
    
    # Get all numeric directories (0, 1, 2, ...)
    case_dirs = sorted(
        [d for d in level_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x.name)
    )
    
    # For each case inside the level, use tqdm to show progress
    for case_dir in tqdm(case_dirs, desc=f"Processing cases in {level_name}"):
        case_id = case_dir.name  # e.g. "0", "1", "2"
        
        # Check if the necessary files exist
        color_file = case_dir / "color.json"
        matrix_file = case_dir / "matrix.json"
        image_file = case_dir / "left-view-45.png"
        view_desc_file = case_dir / "view_desc.txt"
        
        # If there is no color.json, skip this case
        if not color_file.exists():
            continue
        
        if not matrix_file.exists():
            print(f"Warning: {case_dir} is missing matrix.json, skip")
            continue
        
        if not image_file.exists():
            print(f"Warning: {case_dir} is missing left-view-45.png, skip")
            continue
        
        # Read the color matrix and height matrix
        try:
            with open(color_file, 'r', encoding='utf-8') as f:
                color_matrix = json.load(f)
            with open(matrix_file, 'r', encoding='utf-8') as f:
                height_matrix = json.load(f)
        except Exception as e:
            print(f"Error: failed to read {case_dir}: {e}, skip")
            continue

        # Read the view description (abstract_caption)
        abstract_caption = ""
        if view_desc_file.exists():
            try:
                with open(view_desc_file, 'r', encoding='utf-8') as f:
                    abstract_caption = f.read().strip()
            except Exception as e:
                print(f"Warning: failed to read {view_desc_file}: {e}, using empty string as abstract_caption")
        else:
            print(f"Warning: {case_dir} is missing view_desc.txt, using empty string as abstract_caption")
        
        # Generate the visibility matrix
        try:
            voxel_matrix = generate_voxel_matrix(height_matrix)
        except Exception as e:
            print(f"Error: failed to generate visibility matrix {case_dir}: {e}, skip")
            continue
        
        # Count the number of visible blocks for each color
        color_counts = count_visible_blocks_by_color(color_matrix, voxel_matrix)
        
        # Generate a new image file name
        new_image_name = f"{level_name}_{case_id}_left-view-45.png"
        
        # Build the images list (only contains left-view-45.png)
        images = [f"./images/{new_image_name}"]
        
        # For each color with visible blocks, generate a question
        for color_id, count in color_counts.items():
            if count > 0:  # Only generate questions for colors with blocks
                color_name = COLOR_ID_TO_NAME[color_id]
                
                # Build the problem field
                problem = f"<image>\nHow many {color_name} blocks are there in the picture? Return your final response within \\boxed{{}}."
                
                # Build the data record
                record = {
                    "id": f"{level_name}_{case_id}",
                    "problem": problem,
                    "answer": str(count),
                    "images": images,
                    "model_generation": generate_model_generation(color_name, count),
                    "abstract_caption": abstract_caption,
                }
                
                records.append((record, image_file))
    
    return records


def parse_eval_per_level(config_str: str) -> Dict[str, int]:
    """
    Parse the eval number configuration specified by level.
    
    Supported format examples:
        "222:10,333:20"
        "level222:10, level333:20"
    
    Returns:
        dict: { "level222": 10, "level333": 20, ... }
    """
    eval_per_level: Dict[str, int] = {}
    if not config_str:
        return eval_per_level
    
    for part in config_str.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            level_str, num_str = part.split(":", 1)
            level_str = level_str.strip()
            num_str = num_str.strip()
            
            # Compatible with both "222" and "level222"
            if not level_str.startswith("level"):
                level_name = f"level{level_str}"
            else:
                level_name = level_str
            
            eval_num = int(num_str)
            if eval_num < 0:
                continue
            eval_per_level[level_name] = eval_num
        except Exception:
            print(f"Warning: failed to parse eval-per-level configuration fragment '{part}', please use the format like '222:10' or 'level222:10'")
            continue
    
    return eval_per_level


def main():
    """Main function"""
    # Parse the command line arguments
    parser = argparse.ArgumentParser(
        description='Build a cube counting dataset (color version, no longer responsible for splitting, only responsible for building the dataset from the specified train / eval source directories)'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='OUTPUT_COLOR',
        help='Input directory name (e.g. OUTPUT_COLOR_train or OUTPUT_COLOR_eval), default: OUTPUT_COLOR',
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'eval'],
        required=True,
        help='The type of dataset being built: train or eval, used to determine output to data/train or data/eval',
    )
    args = parser.parse_args()
    
    # Path configuration
    project_root = Path(__file__).parent
    output_dir = project_root / args.input_dir
    data_root = project_root.parent.parent / "data"
    
    eval_dataset_dir = data_root / "eval" / "cube_counting_color"
    eval_images_dir = eval_dataset_dir / "images"
    eval_jsonl_file = eval_dataset_dir / "cube_counting.jsonl"

    train_dataset_dir = data_root / "train" / "cube_counting_color"
    train_images_dir = train_dataset_dir / "images"
    train_jsonl_file = train_dataset_dir / "cube_counting.jsonl"
    
    # Get all level directories
    level_dirs = sorted([d for d in output_dir.iterdir() 
                        if d.is_dir() and d.name.startswith("level")])
    
    if not level_dirs:
        print(f"Error: no level directories found in {output_dir}")
        return
    
    print(f"Found {len(level_dirs)} level directories: {[d.name for d in level_dirs]}")
    
    # Process all levels, collect all records and source image paths
    all_data = []  # Store (record, source_image_path) tuples
    for level_dir in tqdm(level_dirs, desc="Processing levels"):
        print(f"\nProcessing {level_dir.name}...")
        data = process_level(level_dir)
        all_data.extend(data)
        print(f"  {level_dir.name}: processed {len(data)} cases")
    total_count = len(all_data)
    print(f"\nTotal collected {total_count} cases")

    # Determine output to train or eval based on split
    if args.split == "eval":
        eval_images_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing eval dataset...")
        eval_records = []
        eval_image_set = set()  # Record images that have been copied to avoid duplicate copying
        for record, source_image_path in tqdm(all_data, desc="Processing eval cases"):
            # Get the image file name
            image_name = record["images"][0].split("/")[-1]
            target_image_path = eval_images_dir / image_name
            
            # Copy the image (if not copied yet)
            if image_name not in eval_image_set:
                try:
                    shutil.copy2(source_image_path, target_image_path)
                    eval_image_set.add(image_name)
                except Exception as e:
                    print(f"Error: failed to copy image {source_image_path} to {target_image_path}: {e}, skip")
                    continue
            
            eval_records.append(record)
        
        # Write to eval jsonl file
        print(f"Writing eval dataset file: {eval_jsonl_file}")
        with open(eval_jsonl_file, 'w', encoding='utf-8') as f:
            for record in eval_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Output summary
        print(f"\nDone!")
        print(f"  Total cases: {total_count}")
        print(f"  Eval cases: {len(eval_records)}")
        print(f"  Eval dataset file: {eval_jsonl_file}")
        print(f"  Eval image directory: {eval_images_dir}")
    else:
        train_images_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing train dataset...")
        train_records = []
        train_image_set = set()  # Record images that have been copied to avoid duplicate copying
        for record, source_image_path in tqdm(all_data, desc="Processing train cases"):
            # Get the image file name
            image_name = record["images"][0].split("/")[-1]
            target_image_path = train_images_dir / image_name
            
            # Copy the image (if not copied yet)
            if image_name not in train_image_set:
                try:
                    shutil.copy2(source_image_path, target_image_path)
                    train_image_set.add(image_name)
                except Exception as e:
                    print(f"Error: failed to copy image {source_image_path} to {target_image_path}: {e}, skip")
                    continue
            
            train_records.append(record)
        
        # Write to train jsonl file
        print(f"Writing train dataset file: {train_jsonl_file}")
        with open(train_jsonl_file, 'w', encoding='utf-8') as f:
            for record in train_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Output summary
        print(f"\nDone!")
        print(f"  Total cases: {total_count}")
        print(f"  Train cases: {len(train_records)}")
        print(f"  Train dataset file: {train_jsonl_file}")
        print(f"  Train image directory: {train_images_dir}")


if __name__ == "__main__":
    main()
