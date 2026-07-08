#!/usr/bin/env python3
"""
Convert the JSON description of three views (view_desc.json) in the block-count task into a rule-based English natural language description.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple


def ordinal(n: int) -> str:
    """Map 1,2,3,... to 'first','second',...; fall back to 'N-th' beyond the table."""
    mapping = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eighth",
        9: "ninth",
        10: "tenth",
    }
    if n in mapping:
        return mapping[n]
    return f"{n}-th"


def cube_phrase(has_color: bool, block: Dict[str, Any]) -> str:
    """Short phrase for a single cube (e.g. "a cube", "a red cube", "a hidden cube")."""
    # A cube invisible in isometric view is described as a hidden cube (color withheld)
    if not block.get("visible", True):
        return "a hidden cube"

    if has_color and "color_name" in block:
        return f"a {block['color_name']} cube"
    return "a cube"


def join_sequence(items: List[str]) -> str:
    """Join phrases with ", then " to express bottom-to-top / front-to-back order."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", then ".join(items)


def describe_front_view(record: Dict[str, Any]) -> str:
    """Generate English description from the front view in view_desc.json."""
    views = record.get("views", {})
    front = views.get("front", {})
    columns: List[Dict[str, Any]] = front.get("columns", [])
    has_color: bool = bool(record.get("has_color", False))

    parts: List[str] = []

    parts.append(
        "First, from the front side perspective, we describe the visible cubes in each column."
    )

    # Order columns left-to-right as seen in the image. The front view maps the
    # horizontal axis as (C-1-x), so we traverse x descending.
    columns_sorted = sorted(columns, key=lambda c: c.get("x", 0), reverse=True)

    for col_idx, col in enumerate(columns_sorted):
        layers = col.get("layers", [])
        if not layers:
            continue

        # Sort by z ascending (bottom to top)
        layers_sorted = sorted(layers, key=lambda l: l.get("z", 0))
        blocks_sequence: List[str] = []
        for layer in layers_sorted:
            blocks = layer.get("blocks", [])
            if not blocks:
                continue
            block = blocks[0]
            blocks_sequence.append(cube_phrase(has_color, block))

        if not blocks_sequence:
            continue

        col_ordinal = ordinal(col_idx + 1)
        seq_text = join_sequence(blocks_sequence)
        sentence = (
            f"In the front view, in the {col_ordinal} column from the left, "
            f"from bottom to top we can see {seq_text}."
        )
        parts.append(sentence)

    return "\n".join(parts)


def describe_left_view(record: Dict[str, Any]) -> str:
    """Generate English description from the left view in view_desc.json."""
    views = record.get("views", {})
    left = views.get("left", {})
    columns: List[Dict[str, Any]] = left.get("columns", [])
    has_color: bool = bool(record.get("has_color", False))

    parts: List[str] = []

    parts.append(
        "Second, from the left side perspective, we describe the visible cubes in each column."
    )

    # Sort columns by y ascending (left to right)
    columns_sorted = sorted(columns, key=lambda c: c.get("y", 0))

    for col_idx, col in enumerate(columns_sorted):
        layers = col.get("layers", [])
        if not layers:
            continue

        layers_sorted = sorted(layers, key=lambda l: l.get("z", 0))
        blocks_sequence: List[str] = []
        for layer in layers_sorted:
            blocks = layer.get("blocks", [])
            if not blocks:
                continue
            block = blocks[0]
            blocks_sequence.append(cube_phrase(has_color, block))

        if not blocks_sequence:
            continue

        col_ordinal = ordinal(col_idx + 1)
        seq_text = join_sequence(blocks_sequence)
        sentence = (
            f"In the left view, in the {col_ordinal} column from the left, "
            f"from bottom to top we can see {seq_text}."
        )
        parts.append(sentence)

    return "\n".join(parts)


def describe_top_view(record: Dict[str, Any]) -> str:
    """
    Generate English description from the top view in view_desc.json.

    In the image, the top view maps the horizontal axis as (C-1-x) and the
    vertical axis as (R-1-y). We therefore traverse x descending (columns left
    to right) and y descending (front to back), emitting the top cube per cell
    ("a red cube" / "a cube" / "a hidden cube") or "no cube" when empty.
    """
    views = record.get("views", {})
    top = views.get("top", {})
    cells: List[Dict[str, Any]] = top.get("cells", [])
    has_color: bool = bool(record.get("has_color", False))

    shape = record.get("shape", {})
    C = shape.get("C", 0)  # x-direction length
    R = shape.get("R", 0)  # y-direction length

    parts: List[str] = []

    parts.append(
        "Finally, from the top side perspective, we look at each column from left to right and describe what is on top of each stack from front to back."
    )

    # Build a quick (x, y) -> top block index
    cell_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for cell in cells:
        x = cell.get("x", 0)
        y = cell.get("y", 0)
        layers = cell.get("layers", [])
        if not layers:
            continue
        # layers holds only the highest visible cube by design
        cell_map[(x, y)] = layers[0]

    # x descending -> columns left to right in the image
    for col_idx, x in enumerate(range(C - 1, -1, -1)):
        row_desc: List[str] = []
        # y descending -> front to back in the image
        for y in range(R - 1, -1, -1):
            block = cell_map.get((x, y))
            if block is None:
                row_desc.append("no cube")
            else:
                row_desc.append(cube_phrase(has_color, block))

        # Skip entirely empty columns to avoid verbosity
        if all(desc == "no cube" for desc in row_desc):
            continue

        # Number columns by traversal position rather than raw x index
        col_ordinal = ordinal(col_idx + 1)
        seq_text = join_sequence(row_desc)
        sentence = (
            f"In the top view, in the {col_ordinal} column from the left, "
            f"from front to back we have {seq_text}."
        )
        parts.append(sentence)

    return "\n".join(parts)


def view_desc_to_abstract(record: Dict[str, Any]) -> str:
    """Convert a parsed view_desc.json dict into the English three-view abstract."""
    front_text = describe_front_view(record)
    left_text = describe_left_view(record)
    top_text = describe_top_view(record)

    # Separate the three view descriptions with blank lines; drop empty paragraphs
    parts = [front_text, left_text, top_text]
    parts = [p for p in parts if p.strip()]
    return "\n\n".join(parts)


def process_single_file(path: Path) -> str:
    """Read a single view_desc.json file and return the natural language description."""
    with path.open("r", encoding="utf-8") as f:
        record = json.load(f)
    return view_desc_to_abstract(record)


def batch_process_directory(input_dir: Path, filename: str = "view_desc.json") -> None:
    """
    Batch process all JSON files named filename in the directory,
    generate a .txt description file with the same name.
    """
    json_paths = list(input_dir.rglob(filename))
    if not json_paths:
        print(f"No '{filename}' files found under {input_dir}")
        return

    print(f"Found {len(json_paths)} '{filename}' files under {input_dir}")

    for json_path in json_paths:
        try:
            text = process_single_file(json_path)
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            continue

        txt_path = json_path.with_suffix(".txt")
        try:
            with txt_path.open("w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"Error writing {txt_path}: {e}")
            continue

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert block-count view_desc.json into abstract English descriptions."
    )
    parser.add_argument(
        "--file",
        type=str,
        default="",
        help="Path to a single view_desc.json file. If provided, print the description to stdout.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="",
        help="Root directory containing multiple view_desc.json files for batch processing.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="view_desc.json",
        help="Name of the JSON description files to search for in batch mode. Default: view_desc.json",
    )

    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"Error: file {path} does not exist.")
            return
        text = process_single_file(path)
        print(text)
        return

    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            print(f"Error: input directory {input_dir} does not exist or is not a directory.")
            return
        batch_process_directory(input_dir, filename=args.filename)
        return

    print("Please specify either --file or --input-dir.")


if __name__ == "__main__":
    main()


