#!/bin/bash
# Run VLM evaluation against an already-running vllm OpenAI-compatible server.
#
# Prerequisite: start your own vllm serve, e.g.
#     vllm serve <model> --port 8000 ...
#
# Usage:
#     bash eval_vlm_with_vllm.sh [model_path] [port] [host]
# Examples:
#     bash eval_vlm_with_vllm.sh
#     bash eval_vlm_with_vllm.sh Qwen/Qwen3-VL-4B-Instruct 8000
#     bash eval_vlm_with_vllm.sh Qwen/Qwen3-VL-4B-Instruct 8000 192.168.1.10

# ---- args (override defaults) ----
MODEL_PATH="${1:-Qwen/Qwen3-VL-4B-Instruct}"
PORT="${2:-8000}"
HOST="${3:-localhost}"

# ---- eval config ----
SPLIT="${SPLIT:-full}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
OUTPUT_DIR="./results/${SPLIT}"
BATCH_SIZE="${BATCH_SIZE:-256}"

VLLM_BASE_URL="http://${HOST}:${PORT}/v1"

# ---- sanity check: vllm reachable ----
if ! curl -s "${VLLM_BASE_URL}/models" > /dev/null 2>&1; then
    echo "error: cannot reach vllm at ${VLLM_BASE_URL}/models"
    echo "make sure 'vllm serve' is running at ${HOST}:${PORT}"
    exit 1
fi

echo "=========================================="
echo "evaluating model: ${MODEL_PATH}"
echo "vllm server:     ${VLLM_BASE_URL}"
echo "split:           ${SPLIT}"
echo "max_tokens:      ${MAX_TOKENS}"
echo "output_dir:      ${OUTPUT_DIR}"
echo "batch_size:      ${BATCH_SIZE}"
echo "=========================================="

# stream eval output to terminal AND capture for success-marker check
TMP_LOG=$(mktemp)
trap 'rm -f "$TMP_LOG"' EXIT

PYTHONPATH=.. python eval_vlm_with_vllm.py \
    --model_path "${MODEL_PATH}" \
    --max_tokens ${MAX_TOKENS} \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --split ${SPLIT} \
    --host "${HOST}" \
    --port ${PORT} 2>&1 | tee "$TMP_LOG"
eval_exit_code=${PIPESTATUS[0]}

echo ""
if [[ $eval_exit_code -eq 0 ]] && grep -q "EVALUATION_SUCCESS" "$TMP_LOG"; then
    echo "✓ model ${MODEL_PATH} eval completed successfully"
    exit 0
else
    echo "✗ model ${MODEL_PATH} eval failed"
    if grep -q "EVALUATION_ERROR" "$TMP_LOG"; then
        echo "error info: $(grep "EVALUATION_ERROR" "$TMP_LOG")"
    fi
    exit 1
fi
