# Evaluation

Minimal framework to evaluate any VLM on the **OrthoMind-3D** benchmark (block counting + object reasoning) through three backends.

## Expected data layout

```
data/eval/{full,ood}/cube_counting/{conf_1,conf_2}/test.jsonl + images/
data/eval/{full,ood}/geo_problem/{conf_1,conf_2}/{counting,positioning}/test.jsonl + images/
```

Each JSONL line: `{"id", "problem", "answer", "images": [...]}` (answer is a string for counting, a list for positioning).

## Backends

| Script | Backend | Notes |
|--------|---------|-------|
| `eval_vlm.py` | Local HuggingFace (Qwen3-VL / Qwen2.5-VL) | Sequential; loads weights locally |
| `eval_vlm_with_vllm.py` | OpenAI-compatible vLLM server | Concurrent, base64 images |
| `eval_vlm_with_api.py` | Proprietary APIs (GPT / Gemini / Doubao …) | Concurrent, keys via env vars |

## Usage

```bash
# vLLM: launch a server per model and evaluate on full + ood
bash run_spatial_eval.sh /path/to/model

# against an already-running vLLM server
SPLIT=full bash eval_vlm_with_vllm.sh /path/to/model 8000

# local HuggingFace
PYTHONPATH=.. python eval_vlm.py --model_path /path/to/model --split full

# proprietary API
export OPENAI_API_KEY=...
PYTHONPATH=.. python eval_vlm_with_api.py --models gpt-4o --split full
```

## Helpers

- `reevaluate_results.py` — re-run answer extraction / scoring on saved result JSONs.
- `results_collect.py` — aggregate per-task results into a CSV summary.
