"""Evaluate a local HuggingFace VLM on OrthoMind-3D (sequential inference)."""
import os
import time
import argparse
from typing import Dict, List, Any

import utils.logging
from utils.chat_vlm import ChatVLMBot
from utils.data_loader import CountBlockEvalDataset, ObjectReasoningEvalDataset
from utils.eval_utils import (
    evaluate_counting,
    evaluate_position,
    load_task_result,
    save_task_result,
    get_completed_sample_ids,
)

logger = utils.logging.get_logger(__name__)

# Hyperparameter: Save full model response in results
SAVE_MODEL_RESPONSE = True

# Evaluation Tasks
EVAL_TASKS = {
    "block_conf1": {
        "dataset": None,
        "task_type": "counting",
        "name": "Block Counting (conf_1)"
    },
    "block_conf2": {
        "dataset": None,
        "task_type": "counting",
        "name": "Block Counting (conf_2)"
    },
    "geometry_conf1_counting": {
        "dataset": None,
        "task_type": "counting",
        "name": "Geometry Counting (conf_1)"
    },
    "geometry_conf1_position": {
        "dataset": None,
        "task_type": "position",
        "name": "Geometry Position (conf_1)"
    },
    "geometry_conf2_counting": {
        "dataset": None,
        "task_type": "counting",
        "name": "Geometry Counting (conf_2)"
    },
    "geometry_conf2_position": {
        "dataset": None,
        "task_type": "position",
        "name": "Geometry Position (conf_2)"
    }
}

def initialize_datasets(split: str = "full"):
    """Initialize all datasets"""
    if split == "full":
        base_path = "../data/eval/full"
    elif split == "ood":
        base_path = "../data/eval/ood"
    else:
        raise ValueError(f"Unknown split: {split}. Expected one of: full, ood.")

    EVAL_TASKS["block_conf1"]["dataset"] = CountBlockEvalDataset(
        config="conf_1",
        base_path=f"{base_path}/cube_counting"
    )
    EVAL_TASKS["block_conf2"]["dataset"] = CountBlockEvalDataset(
        config="conf_2",
        base_path=f"{base_path}/cube_counting"
    )
    EVAL_TASKS["geometry_conf1_counting"]["dataset"] = ObjectReasoningEvalDataset(
        config="conf_1",
        qa_type="counting",
        base_path=f"{base_path}/geo_problem"
    )
    EVAL_TASKS["geometry_conf1_position"]["dataset"] = ObjectReasoningEvalDataset(
        config="conf_1",
        qa_type="positioning",
        base_path=f"{base_path}/geo_problem"
    )
    EVAL_TASKS["geometry_conf2_counting"]["dataset"] = ObjectReasoningEvalDataset(
        config="conf_2",
        qa_type="counting",
        base_path=f"{base_path}/geo_problem"
    )
    EVAL_TASKS["geometry_conf2_position"]["dataset"] = ObjectReasoningEvalDataset(
        config="conf_2",
        qa_type="positioning",
        base_path=f"{base_path}/geo_problem"
    )

def create_messages(problem: str, image_paths: List[str]) -> List[Dict[str, Any]]:
    content = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            content.append({"type": "image", "image": img_path})
        else:
            logger.warning(f"Image file not found: {img_path}")
    content.append({"type": "text", "text": problem})
    return [{"role": "user", "content": content}]


def evaluate_model_on_task(
    model: ChatVLMBot,
    model_path: str,
    task_name: str,
    task_config: Dict[str, Any],
    output_dir: str = "./result"
) -> Dict[str, Any]:
    dataset = task_config["dataset"]
    task_type = task_config["task_type"]
    task_display_name = task_config["name"]

    # Try to load existing results
    existing_result = load_task_result(model_path, task_name, output_dir)
    completed_ids = get_completed_sample_ids(existing_result) if existing_result else set()

    if existing_result:
        logger.info(f"Found existing results for task: {task_display_name} ({len(completed_ids)} samples completed)")
        results = existing_result.get("results", [])
        correct_count = existing_result.get("correct_count", 0)
    else:
        results = []
        correct_count = 0

    total_count = len(dataset)
    remaining_count = total_count - len(completed_ids)

    if remaining_count == 0:
        logger.info(f"Task {task_display_name} already completed, skipping")
        return existing_result

    logger.info(f"Evaluating task: {task_display_name} (total {total_count} samples, {remaining_count} remaining)")

    for idx, item in enumerate(dataset):
        item_id = item.get("id", f"item_{idx}")

        # Skip completed samples
        if item_id in completed_ids:
            continue

        problem = item.get("problem", "")
        answer = item.get("answer", "")
        images = item.get("images", [])

        if not images:
            logger.warning(f"Sample {item_id} has no images, skipping")
            result_item = {
                "id": item_id,
                "problem": problem,
                "ground_truth": answer,
                "predicted_text": "",
                "is_correct": False,
                "error": "No images found"
            }
            results.append(result_item)
            continue

        try:
            messages = create_messages(problem, images)
            start_time = time.time()
            response = model.chat(messages)
            inference_time = time.time() - start_time

            # Handle response (maybe a list)
            if isinstance(response, list):
                response_text = response[0] if response else ""
            else:
                response_text = str(response)

            # Evaluate based on task type
            if task_type == "counting":
                eval_result = evaluate_counting(response_text, answer, extract_from_boxed=True)
                is_correct = eval_result["is_correct"]
            elif task_type == "position":
                eval_result = evaluate_position(response_text, answer, extract_from_boxed=False)
                is_correct = eval_result["is_correct"]
            else:
                logger.error(f"Unknown task type: {task_type}")
                eval_result = {}
                is_correct = False

            if is_correct:
                correct_count += 1

            # Save results
            result_item = {
                "id": item_id,
                "problem": problem,
                "ground_truth": answer,
                "predicted_text": response_text,
                "is_correct": is_correct,
                "inference_time": inference_time,
                **eval_result
            }
            # Save full model response if enabled
            if SAVE_MODEL_RESPONSE:
                if isinstance(response, (str, int, float, bool, type(None))):
                    result_item["model_response"] = response
                elif isinstance(response, list):
                    result_item["model_response"] = response
                else:
                    result_item["model_response"] = str(response)
            results.append(result_item)

            # Save results periodically (every 10 samples)
            if len(results) % 10 == 0 and len(results) > 0:
                current_correct = sum(1 for r in results if r.get("is_correct", False))
                current_total = len(results)
                current_accuracy = current_correct / current_total if current_total > 0 else 0.0
                logger.info(f"Completed {current_total}/{total_count} samples, current accuracy: {current_accuracy*100:.2f}%")

                temp_task_result = {
                    "task_name": task_display_name,
                    "task_type": task_type,
                    "total_count": total_count,
                    "correct_count": current_correct,
                    "accuracy": current_accuracy,
                    "results": results
                }
                save_task_result(temp_task_result, model_path, task_name, output_dir)

        except Exception as e:
            logger.error(f"Error processing sample {item_id}: {e}")
            result_item = {
                "id": item_id,
                "problem": problem,
                "ground_truth": answer,
                "predicted_text": "",
                "is_correct": False,
                "error": str(e)
            }
            results.append(result_item)

    # Recalculate final statistics
    final_correct_count = sum(1 for r in results if r.get("is_correct", False))
    final_total_count = len(results)
    accuracy = final_correct_count / final_total_count if final_total_count > 0 else 0.0

    task_result = {
        "task_name": task_display_name,
        "task_type": task_type,
        "total_count": total_count,
        "correct_count": final_correct_count,
        "accuracy": accuracy,
        "results": results
    }

    # Immediately save task results
    result_file = save_task_result(task_result, model_path, task_name, output_dir)
    logger.info(f"Task {task_display_name} evaluation completed: accuracy = {accuracy*100:.2f}% ({final_correct_count}/{final_total_count})")
    logger.info(f"Results saved to: {result_file}")

    return task_result

def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM models using HF/flash_attn")
    parser.add_argument("--model_path", type=str, required=True, help="Model path (local or HuggingFace)")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Maximum tokens to generate")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results (default: ./results/{split})")
    parser.add_argument("--split", type=str, default="full", choices=["full", "ood"], help="Evaluation split")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else f"./results/{args.split}"
    model_path = args.model_path

    initialize_datasets(split=args.split)

    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating model: {model_path}")
    logger.info(f"Split: {args.split}, Output: {output_dir}")
    logger.info(f"{'='*80}\n")

    try:
        model = ChatVLMBot(
            model_path=model_path,
            max_new_tokens=args.max_tokens,
        )

        # Evaluate each task
        all_results = {}
        for task_name, task_config in EVAL_TASKS.items():
            if args.split == "ood":
                if task_name in ("geometry_conf2_counting", "geometry_conf2_position"):
                    continue
                if task_name == "geometry_conf1_counting":
                    task_name = "geometry_counting"
                elif task_name == "geometry_conf1_position":
                    task_name = "geometry_position"
            task_result = evaluate_model_on_task(
                model=model,
                model_path=model_path,
                task_name=task_name,
                task_config=task_config,
                output_dir=output_dir
            )
            all_results[task_name] = task_result

        # Print summary
        logger.info(f"\nModel {model_path} evaluation summary:")
        for task_name, task_result in all_results.items():
            logger.info(
                f"  {task_result['task_name']}: "
                f"Accuracy = {task_result['accuracy']*100:.2f}% "
                f"({task_result['correct_count']}/{task_result['total_count']})"
            )

    except Exception as e:
        logger.error(f"Error evaluating model {model_path}: {e}", exc_info=True)

if __name__ == "__main__":
    main()
