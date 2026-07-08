from typing import Any, Dict, List

import torch
from transformers import AutoProcessor

from utils.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = "Please reason step by step and put the answer in the \\box{}."


def _read_model_config(model_path: str) -> dict:
    import json
    import os

    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


class ChatVLMBot:
    """Local HuggingFace loader/inference wrapper for Qwen3-VL / Qwen2.5-VL."""

    def __init__(self, model_path="Qwen/Qwen3-VL-4B-Instruct", max_new_tokens=16384):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens

        config = _read_model_config(model_path)
        self._model_type = config.get("model_type", "")
        model_name_lower = self.model_path.lower()

        if "qwen3" in model_name_lower or self._model_type == "qwen3_vl":
            self._load_qwen3_vl()
        elif "qwen2" in model_name_lower or self._model_type == "qwen2_5_vl":
            self._load_qwen2_5_vl()
        else:
            self._load_auto_model()

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

    def _load_qwen3_vl(self):
        from transformers import Qwen3VLForConditionalGeneration

        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        except Exception as e:
            logger.warning(f"flash_attention_2 unavailable ({e}), falling back to sdpa")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path, dtype=torch.bfloat16,
                attn_implementation="sdpa", device_map="auto",
            )

    def _load_qwen2_5_vl(self):
        from transformers import Qwen2_5_VLForConditionalGeneration

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(f"flash_attention_2 unavailable ({e}), falling back to sdpa")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path, dtype=torch.bfloat16,
                attn_implementation="sdpa", device_map="auto", trust_remote_code=True,
            )

    def _load_auto_model(self):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )

    def chat(self, messages: List[Dict[str, Any]]):
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
