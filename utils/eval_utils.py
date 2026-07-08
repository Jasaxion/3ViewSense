import re
import os
import json
from typing import Union, List, Dict, Any, Optional
import csv

# 8 directions and their order for Position task
POSITION_DIRECTIONS = [
    "front", "back", "right", "left",
    "front left", "front right", "back left", "back right"
]

# Maximum text length thresholds for fallback extraction
_POSITION_MAX_TEXT_LENGTH = 100
_COUNTING_MAX_TEXT_LENGTH = 20


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> content from model output.
    If <think> is present but no closing </think>, the model was truncated
    during thinking and produced no answer — return empty string.
    """
    if not text:
        return ""
    if "<think>" in text:
        think_end = text.find("</think>")
        if think_end >= 0:
            return text[think_end + 8:].strip()
        else:
            return ""
    return text

# Angle mapping for directions (in degrees)
POSITION_ANGLES = {
    "front": 0,
    "front right": 45,
    "right": 90,
    "back right": 135,
    "back": 180,
    "back left": 225,
    "left": 270,
    "front left": 315,
}


def extract_boxed_answer(text: str) -> Optional[str]:
    if not text:
        return None
    
    patterns = [
        r'\\boxed\{([^}]+)\}',
        r'\\boxed\s*\{([^}]+)\}',
        r'\\box\{([^}]+)\}',
        r'\\box\s*\{([^}]+)\}',
        r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>',
        r'<\|begin_of_box\|>\s*(.*?)\s*<\|end_of_box\|>',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    return None


def parse_number(value: Union[str, int, float]) -> Optional[float]:
    if value is None:
        return None
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        value = value.strip()
        try:
            return float(value)
        except ValueError:
            match = re.search(r'-?\d+\.?\d*', value)
            if match:
                return float(match.group())
    
    return None


def evaluate_counting(
    predicted_text: str,
    ground_truth: Union[str, int, float],
    extract_from_boxed: bool = True
) -> Dict[str, Any]:
    """Evaluate a counting answer against ground truth.

    Returns a dict with is_correct, predicted_value, ground_truth_value,
    deviation (pred - gt), deviation_percentage, and extracted_answer.
    """
    clean_text = strip_thinking(predicted_text)

    predicted_str = None
    if extract_from_boxed:
        predicted_str = extract_boxed_answer(clean_text)
        if predicted_str is None:
            stripped = clean_text.strip() if clean_text else ""
            if stripped and len(stripped) <= _COUNTING_MAX_TEXT_LENGTH:
                predicted_str = stripped
            elif stripped:
                matches = re.findall(r'-?\d+\.?\d*', stripped)
                if matches:
                    predicted_str = matches[-1]
    else:
        stripped = clean_text.strip() if clean_text else ""
        if stripped and len(stripped) <= _COUNTING_MAX_TEXT_LENGTH:
            predicted_str = stripped
        elif stripped:
            matches = re.findall(r'-?\d+\.?\d*', stripped)
            if matches:
                predicted_str = matches[-1]

    predicted_value = parse_number(predicted_str)
    ground_truth_value = parse_number(ground_truth)

    result = {
        "is_correct": False,
        "predicted_value": predicted_value,
        "ground_truth_value": ground_truth_value,
        "deviation": None,
        "deviation_percentage": None,
        "extracted_answer": predicted_str,
    }

    # If either value cannot be parsed, return early
    if predicted_value is None or ground_truth_value is None:
        return result

    deviation = predicted_value - ground_truth_value
    result["deviation"] = deviation

    if ground_truth_value == 0:
        if deviation == 0:
            result["deviation_percentage"] = 0.0
        else:
            result["deviation_percentage"] = float('inf') if deviation > 0 else float('-inf')
    else:
        result["deviation_percentage"] = (deviation / ground_truth_value) * 100

    result["is_correct"] = (deviation == 0)

    return result


def normalize_position(position: str) -> str:
    """Normalize a position string: strip LaTeX, lowercase, collapse whitespace."""
    if not position:
        return ""

    # Remove LaTeX \text{...}
    position = re.sub(r'\\text\s*\{([^}]+)\}', r'\1', position, flags=re.IGNORECASE)

    # "back-left" -> "back left"
    position = position.replace('-', ' ')

    position = re.sub(r'[^\w\s]', '', position)
    position = position.strip().lower()
    position = re.sub(r'\s+', ' ', position)

    return position


def get_position_index(position: str) -> Optional[int]:
    normalized = normalize_position(position)
    normalized_directions = [normalize_position(d) for d in POSITION_DIRECTIONS]

    try:
        return normalized_directions.index(normalized)
    except ValueError:
        return None


def get_position_angle(position: str) -> Optional[int]:
    normalized = normalize_position(position)

    for direction, angle in POSITION_ANGLES.items():
        if normalize_position(direction) == normalized:
            return angle

    return None


def calculate_position_deviation(predicted: str, ground_truth: str) -> Optional[int]:
    pred_angle = get_position_angle(predicted)
    truth_angle = get_position_angle(ground_truth)

    if pred_angle is None or truth_angle is None:
        return None

    # Smaller angle on the circle, in units of 45 degrees per direction
    angle_diff = abs(pred_angle - truth_angle)
    angle_diff = min(angle_diff, 360 - angle_diff)
    deviation = int(round(angle_diff / 45))

    return deviation


def _extract_last_sentence(text: str) -> str:
    """Extract the last meaningful sentence from text for direction matching."""
    text = text.strip()
    # Split on sentence-ending punctuation or newlines
    parts = re.split(r'[.\n!?]+', text)
    # Find the last non-empty part
    for part in reversed(parts):
        part = part.strip()
        if part:
            return part
    return text


def _match_direction(text: str) -> Optional[str]:
    """Match a direction from the 8 standard directions in text.
    Returns the matched direction string, or None if no match."""
    sorted_directions = sorted(POSITION_DIRECTIONS, key=len, reverse=True)
    normalized = normalize_position(text)

    for direction in sorted_directions:
        normalized_dir = normalize_position(direction)
        if normalized == normalized_dir:
            return direction
        pattern = r'\b' + re.escape(normalized_dir) + r'\b'
        if re.search(pattern, normalized, re.IGNORECASE):
            return direction
    return None


def evaluate_position(
    predicted_text: str,
    ground_truth: Union[str, List[str]],
    extract_from_boxed: bool = True
) -> Dict[str, Any]:
    """Evaluate a position answer against ground truth (str or list of str).

    Returns a dict with is_correct, predicted_position, ground_truth_positions,
    deviation, and extracted_answer.
    """
    clean_text = strip_thinking(predicted_text)

    # Always try boxed format first
    predicted_str = extract_boxed_answer(clean_text)

    if predicted_str is None:
        if extract_from_boxed:
            predicted_str = None
        else:
            stripped = clean_text.strip() if clean_text else ""
            if not stripped:
                predicted_str = None
            elif len(stripped) <= _POSITION_MAX_TEXT_LENGTH:
                # Short output: likely a direct answer, search the whole text
                predicted_str = _match_direction(stripped)
            else:
                # Long output: only search the last sentence
                last_sentence = _extract_last_sentence(stripped)
                predicted_str = _match_direction(last_sentence)

    # Normalize ground truth to a list
    if isinstance(ground_truth, str):
        ground_truth_list = [ground_truth]
    elif isinstance(ground_truth, list):
        ground_truth_list = ground_truth
    else:
        ground_truth_list = []

    normalized_predicted = normalize_position(predicted_str) if predicted_str else ""
    normalized_ground_truth = [normalize_position(gt) for gt in ground_truth_list]

    result = {
        "is_correct": False,
        "predicted_position": normalized_predicted,
        "ground_truth_positions": normalized_ground_truth,
        "deviation": None,
        "extracted_answer": predicted_str,
    }

    if normalized_predicted and normalized_predicted in normalized_ground_truth:
        result["is_correct"] = True
        result["deviation"] = 0
    else:
        # Deviation relative to the first ground-truth direction
        if normalized_ground_truth and normalized_predicted:
            result["deviation"] = calculate_position_deviation(
                normalized_predicted,
                normalized_ground_truth[0]
            )

    return result


def sanitize_model_name(model_path: str) -> str:
    safe_name = model_path.replace("/", "_").replace("\\", "_")
    safe_name = re.sub(r'_+', '_', safe_name)
    safe_name = safe_name.strip('_')
    return safe_name


def get_task_result_file_path(
    model_path: str,
    task_name: str,
    output_dir: str = "./result"
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    safe_model_name = sanitize_model_name(model_path)
    filename = f"{safe_model_name}_{task_name}_results.json"
    return os.path.join(output_dir, filename)


def load_task_result(
    model_path: str,
    task_name: str,
    output_dir: str = "./result"
) -> Optional[Dict[str, Any]]:
    file_path = get_task_result_file_path(model_path, task_name, output_dir)
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading result file {file_path}: {e}")
            return None
    return None


def save_task_result(
    task_result: Dict[str, Any],
    model_path: str,
    task_name: str,
    output_dir: str = "./result"
) -> str:
    file_path = get_task_result_file_path(model_path, task_name, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(task_result, f, ensure_ascii=False, indent=2)
    
    return file_path


def get_completed_sample_ids(task_result: Dict[str, Any]) -> set:
    if "results" not in task_result:
        return set()
    
    completed_ids = set()
    for result_item in task_result["results"]:
        if "id" in result_item:
            completed_ids.add(result_item["id"])
    return completed_ids

def save_full_results(
    all_results: Dict[str, Dict[str, Any]],
    model_name: str,
    output_dir: str = "./result"
):
    os.makedirs(output_dir, exist_ok=True)
    
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    
    json_path = os.path.join(output_dir, f"{safe_model_name}_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    csv_path = os.path.join(output_dir, f"{safe_model_name}_results.csv")
    
    csv_rows = []
    for task_name, task_result in all_results.items():
        task_display_name = task_result["task_name"]
        task_type = task_result["task_type"]
        
        for result_item in task_result["results"]:
            row = {
                "model": safe_model_name,
                "task": task_display_name,
                "task_type": task_type,
                "id": result_item.get("id", ""),
                "is_correct": result_item.get("is_correct", False),
                "ground_truth": str(result_item.get("ground_truth", "")),
                "predicted_text": result_item.get("predicted_text", ""),
                "inference_time": result_item.get("inference_time", 0),
            }
            
            # Add task-specific fields
            if task_type == "counting":
                row["predicted_value"] = result_item.get("predicted_value")
                row["ground_truth_value"] = result_item.get("ground_truth_value")
                row["deviation"] = result_item.get("deviation")
                row["deviation_percentage"] = result_item.get("deviation_percentage")
            elif task_type == "position":
                row["predicted_position"] = result_item.get("predicted_position", "")
                row["ground_truth_positions"] = str(result_item.get("ground_truth_positions", []))
                row["deviation"] = result_item.get("deviation")
            
            csv_rows.append(row)
    
    # Write to CSV
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)