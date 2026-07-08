"""Evaluate proprietary VLM APIs on OrthoMind-3D (concurrent requests)."""
import os
import time
import argparse
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

import utils.logging
from utils.chat_api import ChatAPIBot
from utils.data_loader import CountBlockEvalDataset, ObjectReasoningEvalDataset
from utils.eval_utils import (
    evaluate_counting,
    evaluate_position,
    load_task_result,
    save_task_result,
    get_completed_sample_ids,
)

logger = utils.logging.get_logger(__name__)

# Save full model response in results
SAVE_MODEL_RESPONSE = True
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


def create_api_input(problem: str, image_paths: List[str]) -> Dict[str, Any]:
    images = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            try:
                images.append(Image.open(img_path).convert("RGB"))
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
        else:
            logger.warning(f"Image file not found: {img_path}")
    return {"texts": [problem], "images": images}


def process_single_sample(
    model: ChatAPIBot,
    item: Dict[str, Any],
    item_id: str,
    task_type: str
) -> Tuple[str, Dict[str, Any], float]:
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
        return item_id, result_item, 0.0
    
    try:
        api_input = create_api_input(problem, images)
        start_time = time.time()
        response = model.chat(api_input)
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
            # Convert response to serializable format
            if isinstance(response, (str, int, float, bool, type(None))):
                result_item["model_response"] = response
            elif isinstance(response, list):
                result_item["model_response"] = response
            else:
                result_item["model_response"] = str(response)
        return item_id, result_item, inference_time
        
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
        return item_id, result_item, 0.0


def evaluate_model_on_task(
    model: ChatAPIBot,
    model_name: str,
    task_name: str,
    task_config: Dict[str, Any],
    output_dir: str = "./result",
    batch_size: int = 10
) -> Dict[str, Any]:
    dataset = task_config["dataset"]
    task_type = task_config["task_type"]
    task_display_name = task_config["name"]
    
    # Add ChatAPI_ prefix to model name for result saving
    model_path_for_saving = f"ChatAPI_{model_name}"
    
    # Try to load existing results
    existing_result = load_task_result(model_path_for_saving, task_name, output_dir)
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
    
    logger.info(f"Evaluating task: {task_display_name} (total {total_count} samples, {remaining_count} remaining, batch_size={batch_size})")
    
    # Collect all samples that need to be processed
    samples_to_process = []
    for idx, item in enumerate(dataset):
        item_id = item.get("id", f"item_{idx}")
        if item_id not in completed_ids:
            samples_to_process.append((idx, item, item_id))
    
    # Process samples in batches
    total_batches = (len(samples_to_process) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(samples_to_process))
        batch_samples = samples_to_process[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_samples)} samples)")
        batch_start_time = time.time()
        
        # Process batch using ThreadPoolExecutor for concurrent requests
        batch_results = []
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Submit all tasks in the batch
            future_to_item = {
                executor.submit(
                    process_single_sample,
                    model,
                    item,
                    item_id,
                    task_type
                ): (idx, item, item_id)
                for idx, item, item_id in batch_samples
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_item):
                try:
                    item_id, result_item, inference_time = future.result()
                    batch_results.append(result_item)
                    results.append(result_item)
                except Exception as e:
                    idx, item, item_id = future_to_item[future]
                    logger.error(f"Error processing sample {item_id} in batch: {e}")
                    result_item = {
                        "id": item_id,
                        "problem": item.get("problem", ""),
                        "ground_truth": item.get("answer", ""),
                        "predicted_text": "",
                        "is_correct": False,
                        "error": str(e)
                    }
                    batch_results.append(result_item)
                    results.append(result_item)
        
        batch_time = time.time() - batch_start_time
        batch_avg_time = batch_time / len(batch_samples) if batch_samples else 0
        
        # Recalculate accuracy after batch
        current_correct = sum(1 for r in results if r.get("is_correct", False))
        current_total = len(results)
        current_accuracy = current_correct / current_total if current_total > 0 else 0.0
        
        logger.info(
            f"Batch {batch_idx + 1}/{total_batches} completed: "
            f"{len(batch_samples)} samples in {batch_time:.2f}s "
            f"(avg {batch_avg_time:.2f}s/sample), "
            f"total progress: {current_total}/{total_count}, "
            f"accuracy: {current_accuracy*100:.2f}%"
        )
        
        # Save results after each batch
        temp_task_result = {
            "task_name": task_display_name,
            "task_type": task_type,
            "total_count": total_count,
            "correct_count": current_correct,
            "accuracy": current_accuracy,
            "results": results
        }
        save_task_result(temp_task_result, model_path_for_saving, task_name, output_dir)
        logger.info(f"Results saved after batch {batch_idx + 1}/{total_batches}")
    
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
    result_file = save_task_result(task_result, model_path_for_saving, task_name, output_dir)
    logger.info(f"Task {task_display_name} evaluation completed: accuracy = {accuracy*100:.2f}% ({final_correct_count}/{final_total_count})")
    logger.info(f"Results saved to: {result_file}")
    
    return task_result


def main():
    parser = argparse.ArgumentParser(description="Evaluate proprietary VLM APIs on OrthoMind-3D")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="API model name(s), e.g. gpt-4o gemini-3-pro")
    parser.add_argument("--split", type=str, default="full", choices=["full", "ood"], help="Evaluation split")
    parser.add_argument("--batch_size", type=int, default=10, help="Concurrent requests per batch")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: ./results/{split})")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else f"./results/{args.split}"
    initialize_datasets(split=args.split)

    for model_name in args.models:
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating model: {model_name}")
        logger.info(f"{'='*80}\n")

        try:
            model = ChatAPIBot(model_name=model_name)

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
                    model_name=model_name,
                    task_name=task_name,
                    task_config=task_config,
                    output_dir=output_dir,
                    batch_size=args.batch_size
                )
                if task_result:
                    all_results[task_name] = task_result

            logger.info(f"\nModel {model_name} evaluation summary:")
            for task_name, task_result in all_results.items():
                logger.info(
                    f"  {task_result['task_name']}: "
                    f"Accuracy = {task_result['accuracy']*100:.2f}% "
                    f"({task_result['correct_count']}/{task_result['total_count']})"
                )

        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}", exc_info=True)
            continue

if __name__ == "__main__":
    main()
