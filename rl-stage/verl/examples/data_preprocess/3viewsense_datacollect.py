"""
Preprocess the OrthoMind-3D RL dataset to parquet format.
python 3viewsense_datacollect.py --local_dataset_path xxx --local_save_dir xxx --test_ratio 0.2 --seed 42
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


def _resolve_data_files(local_dataset_path):
    if local_dataset_path is None:
        return None
    if os.path.isdir(local_dataset_path):
        return {"train": os.path.join(local_dataset_path, "train.jsonl")}
    return {"train": local_dataset_path}


def _normalize_list_field(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        return [stripped]
    return [str(value)]


def _load_local_jsonl(data_path):
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            example["answer"] = _normalize_list_field(example.get("answer"))
            example["images"] = _normalize_list_field(example.get("images"))
            example["images_left"] = _normalize_list_field(example.get("images_left"))
            example["images_right"] = _normalize_list_field(example.get("images_right"))
            records.append(example)
    return datasets.Dataset.from_list(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_dataset_path",
        default="./rl-stage/RL-data-30k",
        help="The local path to the raw dataset, if it exists.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.0,
        help="Randomly sample this ratio of data as test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/orthomind-3d-rl",
        help="The save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "jasaxion/orthomind-3d-rl"
    data_files = _resolve_data_files(local_dataset_path)

    if data_files is not None:
        train_path = data_files["train"]
        train_dataset = _load_local_jsonl(train_path)
    else:
        dataset = datasets.load_dataset(data_source)
        train_dataset = dataset["train"]
        train_dataset = train_dataset.map(lambda ex: {"answer": _normalize_list_field(ex.get("answer"))})
    test_dataset = None
    if args.test_ratio and args.test_ratio > 0:
        split = train_dataset.train_test_split(test_size=args.test_ratio, seed=args.seed, shuffle=True)
        train_dataset = split["train"]
        test_dataset = split["test"]

    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.get("problem", "")
            answer = example.get("answer", "")
            images = example.get("images", [])
            if isinstance(images, str):
                images = [images]

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": problem,
                    }
                ],
                "images": images,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                    "id": example.get("id"),
                    "image_id": example.get("image_id"),
                    "type": example.get("type"),
                    "source": example.get("source"),
                    "abstract_caption": example.get("abstract_caption"),
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    if test_dataset is not None:
        test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_save_dir = os.path.expanduser(local_save_dir)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    if test_dataset is not None:
        test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)