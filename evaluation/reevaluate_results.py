"""
Re-evaluate all existing result JSON files using the fixed extraction logic.

This script:
1. Iterates over all JSON result files in the specified directory
2. For each result item, re-extracts the answer from model_response content
3. Re-evaluates correctness using the fixed logic
4. Overwrites the original JSON files with corrected results
5. Prints a before/after comparison report
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.eval_utils import (
    evaluate_counting,
    evaluate_position,
)


def get_model_response_content(result_item: Dict[str, Any]) -> Optional[str]:
    """Extract the raw model response content from a result item."""
    model_resp = result_item.get("model_response")
    if not model_resp:
        return None

    if isinstance(model_resp, dict):
        choices = model_resp.get("choices", [])
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                return message.get("content", "")
    elif isinstance(model_resp, str):
        return model_resp
    elif isinstance(model_resp, list):
        return model_resp[0] if model_resp else None

    return None


def reevaluate_file(filepath: str) -> Dict[str, Any]:
    """Re-evaluate a single result file and return change statistics."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    task_type = data.get("task_type", "")
    results = data.get("results", [])

    old_correct = data.get("correct_count", 0)
    changes = []

    for item in results:
        # Get raw model response content
        content = get_model_response_content(item)

        if content is None:
            # No model_response stored — check if predicted_text is the raw output
            # (for older eval_vlm.py results where model_response might be a plain string)
            content = item.get("predicted_text", "")
            if not content:
                # Cannot re-evaluate without model output
                item["is_correct"] = False
                continue

        old_correct_flag = item.get("is_correct", False)

        # Re-evaluate based on task type
        if task_type == "counting":
            eval_result = evaluate_counting(content, item.get("ground_truth", ""), extract_from_boxed=True)
        elif task_type == "position":
            eval_result = evaluate_position(content, item.get("ground_truth", []), extract_from_boxed=False)
        else:
            continue

        new_correct_flag = eval_result["is_correct"]

        # Update item fields
        item["is_correct"] = new_correct_flag
        item["predicted_text"] = content
        item["extracted_answer"] = eval_result.get("extracted_answer")

        if task_type == "counting":
            item["predicted_value"] = eval_result.get("predicted_value")
            item["ground_truth_value"] = eval_result.get("ground_truth_value")
            item["deviation"] = eval_result.get("deviation")
            item["deviation_percentage"] = eval_result.get("deviation_percentage")
        elif task_type == "position":
            item["predicted_position"] = eval_result.get("predicted_position", "")
            item["ground_truth_positions"] = eval_result.get("ground_truth_positions", [])
            item["deviation"] = eval_result.get("deviation")

        if old_correct_flag != new_correct_flag:
            changes.append({
                "id": item.get("id", "?"),
                "old": old_correct_flag,
                "new": new_correct_flag,
            })

    # Recalculate statistics
    new_correct = sum(1 for r in results if r.get("is_correct", False))
    total = len(results)
    accuracy = new_correct / total if total > 0 else 0.0

    data["correct_count"] = new_correct
    data["accuracy"] = accuracy

    # Write back
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "file": os.path.basename(filepath),
        "task_type": task_type,
        "total": total,
        "old_correct": old_correct,
        "new_correct": new_correct,
        "changes": changes,
    }


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate existing result JSON files with fixed extraction logic")
    parser.add_argument("--input-dir", type=str, default="./results/full", help="Directory containing result JSON files")
    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"Error: directory not found: {input_dir}")
        sys.exit(1)

    json_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".json"))
    print(f"Found {len(json_files)} result files in {input_dir}")
    print()

    total_old_correct = 0
    total_new_correct = 0
    total_samples = 0
    total_flipped_to_wrong = 0
    total_flipped_to_correct = 0

    report_lines = []

    for fname in json_files:
        filepath = os.path.join(input_dir, fname)
        stats = reevaluate_file(filepath)

        total_old_correct += stats["old_correct"]
        total_new_correct += stats["new_correct"]
        total_samples += stats["total"]

        flipped_wrong = sum(1 for c in stats["changes"] if c["old"] and not c["new"])
        flipped_correct = sum(1 for c in stats["changes"] if not c["old"] and c["new"])
        total_flipped_to_wrong += flipped_wrong
        total_flipped_to_correct += flipped_correct

        if stats["old_correct"] != stats["new_correct"]:
            short_name = fname.replace("_results.json", "")
            old_acc = stats["old_correct"] / stats["total"] * 100 if stats["total"] else 0
            new_acc = stats["new_correct"] / stats["total"] * 100 if stats["total"] else 0
            diff = stats["new_correct"] - stats["old_correct"]
            sign = "+" if diff > 0 else ""
            line = (
                f"  {short_name:<70} "
                f"{stats['old_correct']:>4} -> {stats['new_correct']:>4} "
                f"({old_acc:5.1f}% -> {new_acc:5.1f}%, {sign}{diff})"
            )
            report_lines.append(line)

    # Print report
    print("=" * 100)
    print("RE-EVALUATION REPORT")
    print("=" * 100)
    print()
    print(f"Total files processed: {len(json_files)}")
    print(f"Total samples: {total_samples}")
    print(f"Total correct (old): {total_old_correct} ({total_old_correct/total_samples*100:.2f}%)")
    print(f"Total correct (new): {total_new_correct} ({total_new_correct/total_samples*100:.2f}%)")
    print(f"Flipped correct -> wrong: {total_flipped_to_wrong}")
    print(f"Flipped wrong -> correct: {total_flipped_to_correct}")
    print()

    if report_lines:
        print("Files with accuracy changes:")
        print()
        for line in sorted(report_lines):
            print(line)
    else:
        print("No accuracy changes detected.")


if __name__ == "__main__":
    main()
