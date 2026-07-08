import os
import time
import base64
import argparse
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

import utils.logging
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

def encode_image(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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

def create_messages_for_vllm(problem: str, image_paths: List[str]) -> List[Dict[str, Any]]:
    """Create OpenAI-format messages with base64-encoded images."""
    content = []

    for img_path in image_paths:
        if os.path.exists(img_path):
            try:
                base64_image = encode_image(img_path)
                # Determine MIME type based on file extension
                ext = os.path.splitext(img_path)[1].lower()
                mime_type = "image/png" if ext == ".png" else "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                })
            except Exception as e:
                logger.warning(f"Failed to encode image {img_path}: {e}")
        else:
            logger.warning(f"Image file not found: {img_path}")
    
    # Add text question
    content.append({
        "type": "text",
        "text": problem
    })
    
    return [{
        "role": "user",
        "content": content
    }]

def check_vllm_service(client: OpenAI) -> bool:
    try:
        client.models.list()
        return True
    except Exception as e:
        logger.warning(f"vllm service not available: {e}")
        return False

def process_single_sample(
    client: OpenAI,
    model_path: str,
    max_tokens: int,
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
        messages = create_messages_for_vllm(problem, images)
        start_time = time.time()
        
        response = client.chat.completions.create(
            model=model_path,
            messages=messages,
            max_tokens=max_tokens
        )
        
        inference_time = time.time() - start_time
        response_text = response.choices[0].message.content if response.choices else ""
        
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
            # Convert OpenAI response object to serializable dict
            try:
                # Convert the response object to a dict
                response_dict = {
                    "id": response.id if hasattr(response, 'id') else None,
                    "object": response.object if hasattr(response, 'object') else None,
                    "created": response.created if hasattr(response, 'created') else None,
                    "model": response.model if hasattr(response, 'model') else None,
                    "choices": []
                }
                if hasattr(response, 'choices') and response.choices:
                    for choice in response.choices:
                        choice_dict = {
                            "index": choice.index if hasattr(choice, 'index') else None,
                            "message": {
                                "role": choice.message.role if hasattr(choice.message, 'role') else None,
                                "content": choice.message.content if hasattr(choice.message, 'content') else None
                            },
                            "finish_reason": choice.finish_reason if hasattr(choice, 'finish_reason') else None
                        }
                        response_dict["choices"].append(choice_dict)
                if hasattr(response, 'usage'):
                    response_dict["usage"] = {
                        "prompt_tokens": response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else None,
                        "completion_tokens": response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else None,
                        "total_tokens": response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else None
                    }
                result_item["model_response"] = response_dict
            except Exception as e:
                # Fallback to string representation if conversion fails
                logger.warning(f"Failed to serialize model response for {item_id}: {e}")
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
    client: OpenAI,
    model_path: str,
    max_tokens: int,
    task_name: str,
    task_config: Dict[str, Any],
    output_dir: str = "./result",
    batch_size: int = 10
) -> Dict[str, Any]:
    dataset = task_config["dataset"]
    task_type = task_config["task_type"]
    task_display_name = task_config["name"]

    model_path_for_saving = f"VLLM_{model_path}"
    
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
                    client,
                    model_path,
                    max_tokens,
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

def eval_vlm_with_vllm(model_path: str, max_tokens: int = 16384, output_dir: str = "./result", batch_size: int = 10, split: str = "full", port: int = 8000, host: str = "localhost"):
    # Initialize datasets
    initialize_datasets(split=split)

    # Create OpenAI client to connect to vllm serve
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://{host}:{port}/v1",
        timeout=36000
    )
    
    # Check if service is available (if not, do not print any content, let bash script detect empty output and retry)
    if not check_vllm_service(client):
        return
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating model with vllm: {model_path}")
    logger.info(f"{'='*80}\n")
    
    try:
        # Evaluate each task
        all_results = {}
        for task_name, task_config in EVAL_TASKS.items():
            if split == "ood":
                if task_name in ("geometry_conf2_counting", "geometry_conf2_position"):
                    continue
                if task_name == "geometry_conf1_counting":
                    task_name = "geometry_counting"
                elif task_name == "geometry_conf1_position":
                    task_name = "geometry_position"
            task_result = evaluate_model_on_task(
                client=client,
                model_path=model_path,
                max_tokens=max_tokens,
                task_name=task_name,
                task_config=task_config,
                output_dir=output_dir,
                batch_size=batch_size
            )
            if task_result:
                all_results[task_name] = task_result
        
        # Print summary
        logger.info(f"\nModel {model_path} evaluation summary:")
        for task_name, task_result in all_results.items():
            logger.info(
                f"  {task_result['task_name']}: "
                f"Accuracy = {task_result['accuracy']*100:.2f}% "
                f"({task_result['correct_count']}/{task_result['total_count']})"
            )
        
        # Return success identifier (for bash script check)
        # Use print to stdout, so bash script can detect non-empty output
        print("EVALUATION_SUCCESS")
        
    except Exception as e:
        logger.error(f"Error evaluating model {model_path}: {e}", exc_info=True)
        print(f"EVALUATION_ERROR: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLM models using vllm")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Maximum tokens to generate")
    parser.add_argument("--output_dir", type=str, default="./result", help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for concurrent processing")
    parser.add_argument("--split", type=str, default="full", help="Split to evaluate")
    parser.add_argument("--port", type=int, default=8000, help="VLLM port")
    parser.add_argument("--host", type=str, default="localhost", help="VLLM host")
    args = parser.parse_args()

    eval_vlm_with_vllm(args.model_path, args.max_tokens, args.output_dir, args.batch_size, args.split, args.port, args.host)
