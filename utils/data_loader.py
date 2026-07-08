from datasets import load_dataset, Dataset, Features, Value, Sequence, Image
from typing import Optional
import os
import json

class GeneralMathDataset:
    """General math dataset wrapper class"""
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class Geometry3kDataset(GeneralMathDataset):
    """Geometry3k dataset wrapper"""
    def __init__(self, dataset_path: str, split: str = "train"):
        dataset = load_dataset(dataset_path, split=split)
        super().__init__(dataset)

class CountBlockDataset(GeneralMathDataset):
    """CountBlock dataset wrapper"""
    def __init__(self, dataset_path: str, split: str = "test", features_dict: Optional[Features] = None):
        features = Features({
            'id': Value('string'),
            'problem': Value('string'),
            'answer': Value('string'),
            'images': Sequence(Image())
        })
        final_features = features_dict if features_dict is not None else features.get('features')
        file_name = f"{split}.parquet"
        full_file_path = os.path.join(dataset_path, file_name)
        if not os.path.exists(full_file_path):
            raise FileNotFoundError(
                f"Data file not found: {full_file_path}\n"
                f"Please ensure the file named '{file_name}' exists in the directory '{dataset_path}'."
            )
        data_files_dict = {split: full_file_path}
        dataset_dict = load_dataset(
            "parquet",
            data_files=data_files_dict,
            features=final_features
        )
        dataset = dataset_dict[split]
        super().__init__(dataset)

class CountBlockEvalDataset(GeneralMathDataset):
    def __init__(self, config: str = "conf_1", base_path: Optional[str] = None):
        if base_path is None:
            base_path = "./data/eval/small/cube_counting"
        
        if config not in ["conf_1", "conf_2"]:
            raise ValueError(f"config must be 'conf_1' or 'conf_2', current is: {config}")
        
        config_dir = os.path.join(base_path, config)
        jsonl_path = os.path.join(config_dir, "test.jsonl")
        
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"Data file not found: {jsonl_path}\n"
                f"Please ensure the file named 'test.jsonl' exists in the directory '{config}'."
            )
        
        data_list = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if 'images' in item and isinstance(item['images'], list):
                        absolute_images = []
                        for img_path in item['images']:
                            if img_path.startswith('./'):
                                absolute_path = os.path.abspath(os.path.join(config_dir, img_path[2:]))
                            elif not os.path.isabs(img_path):
                                absolute_path = os.path.abspath(os.path.join(config_dir, img_path))
                            else:
                                absolute_path = img_path
                            absolute_images.append(absolute_path)
                        item['images'] = absolute_images
                    data_list.append(item)
        
        has_abstract_caption = any("abstract_caption" in item for item in data_list)
        features_dict = {
            "id": Value("string"),
            "problem": Value("string"),
            "answer": Value("string"),
            "images": Sequence(Value("string")),
        }
        if has_abstract_caption:
            features_dict["abstract_caption"] = Value("string")
        features = Features(features_dict)

        # Keep only fields defined in Features
        filtered_data_list = []
        for item in data_list:
            filtered_item = {key: item[key] for key in features.keys() if key in item}
            filtered_data_list.append(filtered_item)

        dataset = Dataset.from_list(filtered_data_list, features=features)

        super().__init__(dataset)

class ObjectReasoningEvalDataset(GeneralMathDataset):
    def __init__(self, config: str = "conf_1", qa_type: str = "counting", base_path: Optional[str] = None):
        """Object reasoning eval set. config: conf_1|conf_2; qa_type: counting|positioning."""
        if base_path is None:
            base_path = "./data/eval/small/geo_problem"
        
        if config not in ["conf_1", "conf_2"]:
            raise ValueError(f"config must be 'conf_1' or 'conf_2', current is: {config}")
        
        if qa_type not in ["counting", "positioning"]:
            raise ValueError(f"qa_type must be 'counting' or 'positioning', current is: {qa_type}")
        
        qa_type_dir = os.path.join(base_path, config, qa_type)
        jsonl_path = os.path.join(qa_type_dir, "test.jsonl")
        if not os.path.exists(qa_type_dir):
            # Fallback to layout without config dir, e.g. base_path/{qa_type}/
            qa_type_dir = os.path.join(base_path, qa_type)
            jsonl_path = os.path.join(qa_type_dir, "test.jsonl")
        if not os.path.exists(jsonl_path):
            # Fallback to {qa_type}.jsonl in the same dir (used by some ood splits)
            jsonl_path = os.path.join(qa_type_dir, f"{qa_type}.jsonl")
        
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"Data file not found: {jsonl_path}\n"
                f"Please ensure the file named 'test.jsonl' or '{qa_type}.jsonl' exists in the directory '{config}/{qa_type}' or '{qa_type}'."
            )
        
        data_list = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if 'images' in item and isinstance(item['images'], list):
                        absolute_images = []
                        for img_path in item['images']:
                            if img_path.startswith('./'):
                                absolute_path = os.path.abspath(os.path.join(qa_type_dir, img_path[2:]))
                            elif not os.path.isabs(img_path):
                                absolute_path = os.path.abspath(os.path.join(qa_type_dir, img_path))
                            else:
                                absolute_path = img_path
                            absolute_images.append(absolute_path)
                        item['images'] = absolute_images
                    data_list.append(item)
        
        has_abstract_caption = any("abstract_caption" in item for item in data_list)
        features_dict = {
            "id": Value("string"),
            "problem": Value("string"),
            "answer": Value("string") if qa_type == "counting" else Sequence(Value("string")),
            "images": Sequence(Value("string")),
        }
        if has_abstract_caption:
            features_dict["abstract_caption"] = Value("string")
        features = Features(features_dict)

        # Keep only fields defined in Features
        filtered_data_list = []
        for item in data_list:
            filtered_item = {key: item[key] for key in features.keys() if key in item}
            filtered_data_list.append(filtered_item)

        dataset = Dataset.from_list(filtered_data_list, features=features)

        super().__init__(dataset)