"""
result collect script
python results_collect.py --input-dir {input directory} --output-file {output file name}

CSV columns:
model name, task name, correct_count, total_count, accuracy
accuracy = correct_count / total_count (3 decimal places)
"""

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_nested_model_name(data: Dict[str, Any]) -> Optional[str]:
    results = data.get("results", [])
    if not results:
        return None
    first = results[0]
    if not isinstance(first, dict):
        return None
    model_response = first.get("model_response")
    if isinstance(model_response, dict):
        model_name = model_response.get("model")
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()
    return None


def _get_model_name(data: Dict[str, Any], file_name: str) -> str:
    for key in ("model_name", "model_path", "model"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = _get_nested_model_name(data)
    if nested:
        return nested
    return os.path.splitext(file_name)[0]


def _get_task_name(data: Dict[str, Any], file_name: str) -> str:
    value = data.get("task_name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return os.path.splitext(file_name)[0]


def _extract_stats(data: Dict[str, Any], file_name: str) -> Tuple[str, int, int]:
    task_name = _get_task_name(data, file_name)
    correct_count = _safe_int(data.get("correct_count"), 0)
    total_count = _safe_int(data.get("total_count"), 0)
    return task_name, correct_count, total_count


def collect_results(input_dir: str) -> List[Dict[str, Any]]:
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"input dir not found: {input_dir}")

    rows: List[Dict[str, Any]] = []
    for file_name in sorted(os.listdir(input_dir)):
        if not file_name.endswith(".json"):
            continue
        file_path = os.path.join(input_dir, file_name)
        if not os.path.isfile(file_path):
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            continue
        model_name = _get_model_name(data, file_name)
        task_name, correct_count, total_count = _extract_stats(data, file_name)
        accuracy = (correct_count / total_count) if total_count else 0.0
        rows.append(
            {
                "model name": model_name,
                "task name": task_name,
                "correct_count": correct_count,
                "total_count": total_count,
                "accuracy": accuracy,
            }
        )
    return rows


def write_csv(rows: List[Dict[str, Any]], output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: (r["model name"], r["task name"]))
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model name", "task name", "correct_count", "total_count", "accuracy"])
        for row in rows_sorted:
            writer.writerow(
                [
                    row["model name"],
                    row["task name"],
                    row["correct_count"],
                    row["total_count"],
                    f"{row['accuracy']:.3f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect evaluation results into CSV")
    parser.add_argument("--input-dir", required=True, help="Input directory containing JSON results")
    parser.add_argument("--output-file", required=True, help="Output CSV file path")
    args = parser.parse_args()

    rows = collect_results(args.input_dir)
    write_csv(rows, args.output_file)
    print(f"Saved {len(rows)} rows to {args.output_file}")


if __name__ == "__main__":
    main()