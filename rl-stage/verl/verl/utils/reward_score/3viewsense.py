import re
from typing import Any, Iterable, Optional


POSITION_DIRECTIONS = [
    "front",
    "back",
    "right",
    "left",
    "front left",
    "front right",
    "back left",
    "back right",
]


def extract_boxed_answer(text: str) -> Optional[str]:
    if not text:
        return None

    patterns = [
        r"\\boxed\{([^}]+)\}",
        r"\\boxed\s*\{([^}]+)\}",
        r"\\box\{([^}]+)\}",
        r"\\box\s*\{([^}]+)\}",
        r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>",
        r"<\|begin_of_box\|>\s*(.*?)\s*<\|end_of_box\|>",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    return None


def parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        value = value.strip()
        try:
            return float(value)
        except ValueError:
            match = re.search(r"-?\d+\.?\d*", value)
            if match:
                return float(match.group())

    return None


def normalize_position(position: str) -> str:
    if not position:
        return ""

    position = re.sub(r"\\text\s*\{([^}]+)\}", r"\1", position, flags=re.IGNORECASE)
    position = re.sub(r"[^\w\s-]", "", position)
    position = position.strip().lower()
    position = re.sub(r"\s+", " ", position)
    return position


def _extract_position(predicted_text: str) -> Optional[str]:
    predicted_str = extract_boxed_answer(predicted_text)
    if predicted_str is None:
        predicted_str = predicted_text.strip() if predicted_text else None

    if not predicted_str:
        return None

    normalized_pred = normalize_position(predicted_str)
    sorted_directions = sorted(POSITION_DIRECTIONS, key=len, reverse=True)
    for direction in sorted_directions:
        normalized_dir = normalize_position(direction)
        if normalized_pred == normalized_dir:
            return normalized_dir
        pattern = r"\b" + re.escape(normalized_dir) + r"\b"
        if re.search(pattern, normalized_pred, re.IGNORECASE):
            return normalized_dir

    return normalized_pred


def _score_counting(predicted_text: str, ground_truth: Any, mode: str) -> float:
    if isinstance(ground_truth, list):
        if not ground_truth:
            return 0.0
        ground_truth = ground_truth[0]

    predicted_value = parse_number(extract_boxed_answer(predicted_text) or predicted_text)
    ground_truth_value = parse_number(ground_truth)
    if predicted_value is None or ground_truth_value is None:
        return 0.0

    if mode == "strict":
        return 1.0 if predicted_value == ground_truth_value else 0.0

    diff_value = abs(predicted_value - ground_truth_value)
    diff = int(round(diff_value))
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.8
    if diff == 2:
        return 0.6
    if diff == 3:
        return 0.4
    if diff == 4:
        return 0.2
    return 0.0


def _score_position(predicted_text: str, ground_truth: Any, mode: str) -> float:
    predicted = _extract_position(predicted_text)
    if not predicted:
        return 0.0

    if isinstance(ground_truth, str):
        ground_truth_list = [ground_truth]
    elif isinstance(ground_truth, list):
        ground_truth_list = [str(item) for item in ground_truth]
    else:
        ground_truth_list = []

    normalized_pred = normalize_position(predicted)
    normalized_gt = [normalize_position(item) for item in ground_truth_list]

    if mode == "strict":
        return 1.0 if normalized_pred in normalized_gt else 0.0

    if normalized_pred in normalized_gt:
        return 1.0

    for gt in normalized_gt:
        gt_tokens = gt.split()
        pred_tokens = normalized_pred.split()
        if set(pred_tokens) & set(gt_tokens):
            return 0.5

    return 0.0


def _score_single(predicted_text: str, ground_truth: Any, task_type: str, mode: str) -> float:
    task_type = (task_type or "").strip().lower()
    if "counting" in task_type:
        return _score_counting(predicted_text, ground_truth, mode)
    if "position" in task_type:
        return _score_position(predicted_text, ground_truth, mode)
    return 0.0


def _ensure_list(value: Any, length: int) -> list:
    if isinstance(value, list):
        return value
    return [value for _ in range(length)]


def compute_score(
    data_sources: Iterable[str],
    solution_strs: Iterable[str],
    ground_truths: Iterable[Any],
    extra_infos: Iterable[dict] | None = None,
    mode: str = "strict",
    **kwargs,
):
    solution_list = list(solution_strs)
    ground_truth_list = list(ground_truths)
    extra_list = list(extra_infos) if extra_infos is not None else [{} for _ in solution_list]

    if len(extra_list) != len(solution_list):
        extra_list = _ensure_list(extra_infos, len(solution_list))

    mode = (mode or "strict").strip().lower()
    if mode not in {"strict", "slack"}:
        mode = "strict"

    scores = []
    for predicted_text, ground_truth, extra_info in zip(solution_list, ground_truth_list, extra_list):
        task_type = ""
        if isinstance(extra_info, dict):
            task_type = extra_info.get("type", "")
        scores.append(_score_single(predicted_text, ground_truth, task_type, mode))
    return scores

