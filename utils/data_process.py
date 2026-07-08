import logging
import datetime
from chat_api import ChatAPIBot
from data_loader import Geometry3kDataset, CountBlockDataset
import os
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

MODEL_NAME = "doubao-1-5-thinking-vision-pro-250428"
MAX_WORKERS = 32
OUTPUT_DIR = "./output"

# Logging setup
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"chat_api_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

def construct_input(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    data_collect = []
    for da in data:
        data_collect.append(
            {
                "texts": [da["problem"]],
                "images": da.get("images", [None]),
            }
        )
    return data_collect

def save_images_and_get_paths(
    images_list: List[Any], 
    base_image_dir: str, 
    record_index: int
) -> List[str]:
    os.makedirs(base_image_dir, exist_ok=True)
    image_paths = []
    for img_idx, img_object in enumerate(images_list):
        if isinstance(img_object, Image.Image):
            filename = f"record_{record_index}_image_{img_idx}.png"
            file_path = os.path.join(base_image_dir, filename)
            
            img_object.save(file_path, 'PNG')
        
            image_paths.append(file_path)
        elif isinstance(img_object, str):
            image_paths.append(img_object)
            
    return image_paths

def process_data(data: List[Dict[str, Any]], client: ChatAPIBot) -> str:
    output_path = os.path.join(OUTPUT_DIR, f"Generation_{datetime.datetime.now().strftime('%Y%m%d')}.jsonl")
    base_image_dir = os.path.join(OUTPUT_DIR, "images")
    input = construct_input(data)
    total = len(input)
    logging.info(f"Number of samples: {total}")
    if total == 0:
        logging.info("Input is empty, skip.")
        return

    written = 0
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as rf:
            for _ in rf:
                written += 1
        logging.info(f"Detected existing result file: {written}/{total} lines written, resuming from line {written}.")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        with open(output_path, "a", encoding="utf-8") as wf:
            future_to_idx = {executor.submit(client.chat, input[i]): i for i in range(written, total)}

            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    response = future.result()
                except Exception as e:
                    logging.error(f"Line {i} processing failed: {e}")
                    response = "Processing exception, unable to generate answer."
                rec = data[i]
                rec["model_generation"] = response
                logging.info(f"Response for item {i}: {response[0:50]}...")
                if("images" in rec):
                    images_path = save_images_and_get_paths(rec.get("images", []), base_image_dir, i)
                    rec["images"] = images_path
                try:
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    wf.flush()
                except Exception as e:
                    logging.error(f"Line {i} writing file failed: {e}")
                logging.info(f"Line {i+1} saved")

    logging.info(f"==== Completed processing, output: {output_path} ====\n")
    return output_path

if __name__ == "__main__":
    client = ChatAPIBot(model_name=MODEL_NAME)
    logging.info(f"Using model: {MODEL_NAME}")

    # data = Geometry3kDataset("./data_source/geometer3k", split="test")
    data = CountBlockDataset(
        dataset_path="./data_source/block_count",
        split="test_3view"
    )
    logging.info(f"Loaded dataset with {len(data)} samples.")

    process_data(data, client)
