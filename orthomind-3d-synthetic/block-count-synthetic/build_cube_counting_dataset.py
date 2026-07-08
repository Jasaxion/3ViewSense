#!/usr/bin/env python3
"""
Build a cube counting dataset script

Extract data from the rendered data under OUTPUT_DIR, and build a standard VLM dataset format.
Support splitting the dataset into eval and train two parts.
"""

import json
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm


def generate_model_generation(answer: str) -> str:
    """Generate the model_generation field, imitate the example format"""
    answer_int = int(answer.strip())
    
    # Generate the reasoning process template (even if there is no real thinking process)
    reasoning = f"The user now needs to count the number of cubes in the image. Looking at the picture, there are {answer_int} cube{'s' if answer_int != 1 else ''}, so the count is {answer_int}."
    
    model_generation = f"<think>{reasoning}</think>\n\\boxed{{{answer_int}}}"
    return model_generation


def process_level(level_dir: Path) -> List[Tuple[Dict[str, Any], Path]]:
    """
    Process all cases in a level directory
    
    Args:
        level_dir: level directory path (e.g. OUTPUT_DIR/level333)
    
    Returns:
        The list of all data records in the level, each element is a tuple of (record, source_image_path)
    """
    level_name = level_dir.name  # e.g. "level333"
    records = []
    
    # Get all numeric directories (0, 1, 2, ...)
    case_dirs = sorted(
        [d for d in level_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x.name)
    )
    
    # Use tqdm to show progress for each case inside the level
    for case_dir in tqdm(case_dirs, desc=f"Processing {level_name} cases"):
        case_id = case_dir.name  # e.g. "0", "1", "2"
        
        # Check if the necessary files exist
        answer_file = case_dir / "answer.txt"
        image_file = case_dir / "left-view-45.png"
        view_desc_file = case_dir / "view_desc.txt"
        
        if not answer_file.exists():
            print(f"Warning: answer.txt is missing in {case_dir}, skip")
            continue
        
        if not image_file.exists():
            print(f"Warning: left-view-45.png is missing in {case_dir}, skip")
            continue
        
        # Read the answer
        try:
            with open(answer_file, 'r', encoding='utf-8') as f:
                answer = f.read().strip()
        except Exception as e:
            print(f"Error: failed to read {answer_file}: {e}, skip")
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
            print(f"Warning: view_desc.txt is missing in {case_dir}, using empty string as abstract_caption")
        
        # Generate a new image file name
        new_image_name = f"{level_name}_{case_id}_left-view-45.png"
        
        # Build the problem field (uniform format)
        problem = "<image>\nHow many blocks are there in the picture? Return your final response within \\boxed{}."
        
        # Build the data record (temporarily not setting images, will be set later based on the output directory)
        record = {
            "id": f"{level_name}_{case_id}",
            "problem": problem,
            "answer": answer,
            "images": [f"./images/{new_image_name}"],  # Relative path, will be used later when copying images
            "model_generation": generate_model_generation(answer),
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
        description='Build a cube counting dataset (no longer responsible for splitting, only responsible for building the dataset from the specified train / eval source directories)'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='OUTPUT_DIR',
        help='Input directory name (e.g. OUTPUT_FULL_train or OUTPUT_FULL_eval), default: OUTPUT_DIR',
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
    
    eval_dataset_dir = data_root / "eval" / "cube_counting"
    eval_images_dir = eval_dataset_dir / "images"
    eval_jsonl_file = eval_dataset_dir / "cube_counting.jsonl"

    train_dataset_dir = data_root / "train" / "cube_counting"
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
        # Create output directory
        eval_images_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing eval dataset...")
        eval_records = []
        for record, source_image_path in tqdm(all_data, desc="Processing eval cases"):
            # Get the image file name
            image_name = record["images"][0].split("/")[-1]
            target_image_path = eval_images_dir / image_name
            
            # Copy the image
            try:
                shutil.copy2(source_image_path, target_image_path)
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
        # train
        train_images_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing train dataset...")
        train_records = []
        for record, source_image_path in tqdm(all_data, desc="Processing train cases"):
            # Get the image file name
            image_name = record["images"][0].split("/")[-1]
            target_image_path = train_images_dir / image_name
            
            # Copy the image
            try:
                shutil.copy2(source_image_path, target_image_path)
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
