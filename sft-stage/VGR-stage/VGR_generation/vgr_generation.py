import argparse
import base64
import copy
import json
import logging
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable

from prompt_template import SYSTEM_PROMPT, REASONING_GEN_PROMPT

DEFAULT_INPUT_DIR = Path("./OrthoMind-3D/vgr/input")
DEFAULT_OUTPUT_DIR = Path("./OrthoMind-3D/vgr/output")
DEFAULT_BATCH_SIZE = 64
DEFAULT_LOG_DIR = Path("./logs")
# Teacher-model endpoint; override via the API_URL environment variable.
API_URL = os.environ.get("API_URL", "")


def encode_image(image_path: Path) -> str:
    with image_path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def build_prompt(example: Dict[str, Any]) -> Tuple[str, Path]:
    messages = example.get("messages", [])
    problem = messages[0].get("value", "") if messages else ""
    exta_info = example.get("exta_info", {})
    answer = exta_info.get("answer", "")
    if isinstance(answer, list) and answer:
        answer = answer[0]
    json_format = exta_info.get("json_format", "")
    user_content = (
        REASONING_GEN_PROMPT.replace("{PROBLEM}", str(problem))
        .replace("{ANSWER}", str(answer))
        .replace("{THREE_VIEW_JSON}", str(json_format))
    )

    images = example.get("images", [])
    image_rel = Path(images[0]) if images else Path()
    return user_content, image_rel


def resolve_image_path(input_dir: Path, image_rel: Path) -> Path:
    if image_rel.is_absolute():
        return image_rel
    return (input_dir / image_rel).resolve()


def call_api(model_name: str, api_key: str, user_content: str, image_path: Path) -> str:
    trace_id = str(uuid.uuid4())
    data = {
        "model": model_name,
        "contents": {
            "role": "user",
            "parts": [
                {
                    "inline_data": {
                        "mimeType": "image/png",
                        "data": encode_image(image_path),
                    }
                },
                {"text": user_content},
            ],
        },
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Trace-Id": trace_id,
    }
    response = requests.post(API_URL, headers=headers, json=data, timeout=120)
    response.raise_for_status()
    payload = response.json()
    return payload["candidates"][0]["content"]["parts"][0]["text"]


def call_api_with_retry(model_name: str, api_key: str, user_content: str, image_path: Path) -> str:
    attempt = 0
    while True:
        attempt += 1
        try:
            return call_api(model_name, api_key, user_content, image_path)
        except Exception as exc:
            logging.exception("[Retry] attempt=%s error=%s", attempt, exc)
            time.sleep(3)


def find_gpt_message_index(messages: List[Dict[str, Any]]) -> Optional[int]:
    for idx, message in enumerate(messages):
        if message.get("from") == "gpt":
            return idx
    return None


def write_json_atomic(path: Path, data: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def ensure_output_images(input_dir: Path, output_dir: Path) -> None:
    src_images = input_dir / "images"
    dst_images = output_dir / "images"
    if dst_images.exists():
        return
    if not src_images.exists():
        raise FileNotFoundError(f"images dir not found: {src_images}")
    shutil.copytree(src_images, dst_images)


def load_output_data(
    input_data: List[Dict[str, Any]],
    output_train: Path,
) -> List[Optional[Dict[str, Any]]]:
    if not output_train.exists():
        return [None] * len(input_data)
    try:
        with output_train.open("r", encoding="utf-8") as f:
            existing = json.load(f)
    except Exception as exc:
        logging.warning("Failed to read output, reinitializing: %s", exc)
        return [None] * len(input_data)
    if not isinstance(existing, list):
        return [None] * len(input_data)
    if len(existing) != len(input_data):
        normalized: List[Optional[Dict[str, Any]]] = [None] * len(input_data)
        for idx in range(min(len(existing), len(input_data))):
            if isinstance(existing[idx], dict):
                normalized[idx] = existing[idx]
        return normalized
    return [item if isinstance(item, dict) else None for item in existing]


def pending_indices(data: List[Optional[Dict[str, Any]]]) -> List[int]:
    pending = []
    for idx, example in enumerate(data):
        if not isinstance(example, dict):
            pending.append(idx)
            continue
        messages = example.get("messages", [])
        gpt_idx = find_gpt_message_index(messages)
        if gpt_idx is None:
            pending.append(idx)
            continue
        if not messages[gpt_idx].get("value"):
            pending.append(idx)
    return pending


def process_single_example(
    idx: int,
    example: Dict[str, Any],
    input_dir: Path,
    model_name: str,
    api_key: str,
) -> Tuple[int, str]:
    user_content, image_rel = build_prompt(example)
    image_path = resolve_image_path(input_dir, image_rel)
    result = call_api_with_retry(model_name, api_key, user_content, image_path)
    return idx, result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage2 API generation script")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    return parser.parse_args()


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "stage2_generation.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_dir)
    input_train = input_dir / "train.json"
    output_train = output_dir / "train.json"

    model_name = os.environ.get("MODEL_NAME", "Gemini-3-Flash-Preview")
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY must be set in the environment.")
    if not API_URL:
        raise RuntimeError("API_URL must be set in the environment.")

    if args.batch_size != DEFAULT_BATCH_SIZE:
        logging.warning("batch_size=%s, expected %s", args.batch_size, DEFAULT_BATCH_SIZE)

    if not input_train.exists():
        raise FileNotFoundError(f"train.json not found: {input_train}")

    with input_train.open("r", encoding="utf-8") as f:
        input_data = json.load(f)

    ensure_output_images(input_dir, output_dir)

    output_data = load_output_data(input_data, output_train)
    pending = pending_indices(output_data)
    if not pending:
        logging.info("All samples completed, exiting.")
        return

    executor = ThreadPoolExecutor(max_workers=args.batch_size)
    futures = set()
    pending_iter = iter(pending)

    def submit_next() -> None:
        while len(futures) < args.batch_size:
            try:
                next_idx = next(pending_iter)
            except StopIteration:
                break
            future = executor.submit(
                process_single_example,
                next_idx,
                input_data[next_idx],
                input_dir,
                model_name,
                api_key,
            )
            futures.add(future)

    submit_next()
    completed = 0
    total = len(pending)
    pbar = tqdm(total=total, desc="Generating", unit="sample")

    while futures:
        done, futures = wait(futures, return_when=FIRST_COMPLETED)
        for future in done:
            idx, result = future.result()
            base_example = output_data[idx]
            if not isinstance(base_example, dict):
                base_example = copy.deepcopy(input_data[idx])
            messages = base_example.get("messages", [])
            gpt_idx = find_gpt_message_index(messages)
            if gpt_idx is None:
                messages.append({"from": "gpt", "value": result})
            else:
                messages[gpt_idx]["value"] = result
            base_example["messages"] = messages
            output_data[idx] = base_example
            write_json_atomic(output_train, output_data)
            completed += 1
            pbar.update(1)
            logging.info("Progress %s/%s done", completed, total)
            submit_next()

    pbar.close()
    logging.info("Generation completed.")


if __name__ == "__main__":
    main()
