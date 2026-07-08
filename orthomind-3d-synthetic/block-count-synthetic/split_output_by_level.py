#!/usr/bin/env python3
"""
Split the rendered output by level and extract some cases as eval:

- Move the first N cases in each levelXXX directory to a separate eval directory
- The remaining cases in input-dir are kept, directly used as train (not renamed)

Usage example:

    python split_output_by_level.py \\
        --input-dir OUTPUT_FULL \\
        --eval-per-level "222:15,333:245,444:120,555:120"

Will generate in the current directory:

    OUTPUT_FULL_eval/

Each levelXXX directory contains several numerical subdirectories (cases), the script will move the first N cases to *_eval according to the --eval-per-level configuration; the remaining cases in input-dir are kept as train without renaming.
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


def parse_eval_per_level(config_str: str) -> Dict[str, int]:
    """
    Parse the eval number configuration specified by level.

    Supported format examples:
        "222:10,333:20"
        "level222:10, level333:20"

    Return:
        A dictionary: { "level222": 10, "level333": 20, ... }
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

            # Accept both "222" and "level222" forms
            if not level_str.startswith("level"):
                level_name = f"level{level_str}"
            else:
                level_name = level_str

            eval_num = int(num_str)
            if eval_num < 0:
                continue
            eval_per_level[level_name] = eval_num
        except Exception:
            print(
                f"Warning: Cannot parse the eval-per-level configuration fragment '{part}',"
                "Please use the format like '222:10' or 'level222:10'"
            )
            continue

    return eval_per_level


def move_eval_cases_from_level(
    level_dir: Path,
    eval_root: Path,
    eval_num: int,
) -> None:
    """
    Split a single levelXXX directory according to eval_num:

    - Move the first eval_num cases from level_dir to eval_root/levelXXX
    - The remaining cases in level_dir are kept as train
    """
    level_name = level_dir.name

    # Get all numerical directories (0, 1, 2, ...)
    case_dirs: List[Path] = sorted(
        [d for d in level_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x.name),
    )
    total_level = len(case_dirs)

    if total_level == 0:
        print(f"  {level_name}: Does not contain any case directories, skip")
        return

    # Normalize eval_num
    if eval_num <= 0:
        eval_num_real = 0
    elif eval_num >= total_level:
        eval_num_real = total_level
    else:
        eval_num_real = eval_num

    print(
        f"  {level_name}: total_cases={total_level}, "
        f"eval={eval_num_real}, train={total_level - eval_num_real}"
    )

    # Target eval level directory
    eval_level_dir = eval_root / level_name
    eval_level_dir.mkdir(parents=True, exist_ok=True)

    if eval_num_real == 0:
        return

    # Randomly select eval_num_real cases as eval
    eval_cases = random.sample(case_dirs, eval_num_real)

    # Execute the move: only move the selected eval part, the remaining remain in the original place (as train)
    for case_dir in eval_cases:
        target_dir = eval_level_dir / case_dir.name
        if target_dir.exists():
            # If it already exists, delete it first to avoid stale leftovers
            shutil.rmtree(target_dir)
        # Use move, avoid duplicate copying; usually a renaming operation on the same file system, faster
        shutil.move(str(case_dir), str(target_dir))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract some cases as eval from input-dir by level:"
            "Move the specified number of case directories to a separate eval directory,"
            "The remaining cases in input-dir are kept, directly used as train."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="OUTPUT_FULL",
        help="The original rendered output directory (containing levelXXX subdirectories), default: OUTPUT_FULL",
    )
    parser.add_argument(
        "--output-eval-dir",
        type=str,
        default="",
        help="The eval directory name, default: '<input-dir>_eval' in the same level as input-dir",
    )
    parser.add_argument(
        "--eval-per-level",
        type=str,
        required=True,
        help=(
            'Specify the eval number by level, e.g. "222:15,333:245,444:120,555:120" '
            'or "level222:15,level333:245,level444:120,level555:120"'
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed, used to randomly select eval cases in each level, default: 42",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: The input directory does not exist or is not a directory: {input_dir}")
        return

    # Parse the eval output directory
    if args.output_eval_dir:
        eval_root = Path(args.output_eval_dir).resolve()
    else:
        eval_root = input_dir.with_name(input_dir.name + "_eval")

    eval_root.mkdir(parents=True, exist_ok=True)

    # Set the random seed, ensure the reproducibility of the division result
    if args.seed is not None:
        random.seed(args.seed)

    # Get all level directories
    level_dirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("level")]
    )
    if not level_dirs:
        print(f"Error: No level directories found in {input_dir}")
        return

    print(f"Input directory (remaining as train): {input_dir}")
    print(f"Eval output directory: {eval_root}")
    print(f"Found {len(level_dirs)} level directories: {[d.name for d in level_dirs]}")

    # Parse the eval-per-level configuration
    eval_per_level = parse_eval_per_level(args.eval_per_level)
    print(f"Using the eval-per-level configuration: {eval_per_level}")

    # Split each level (move eval cases)
    for level_dir in tqdm(level_dirs, desc="Splitting level directories"):
        level_name = level_dir.name
        eval_num = eval_per_level.get(level_name, 0)
        if eval_num <= 0:
            print(
                f"  {level_name}: No eval number specified or 0, all cases will be used as train"
            )
        move_eval_cases_from_level(
            level_dir=level_dir,
            eval_root=eval_root,
            eval_num=eval_num,
        )

    print("\nDone!")
    print(f"  train directory (i.e. modified input-dir): {input_dir}")
    print(f"  eval directory: {eval_root}")


if __name__ == "__main__":
    main()


