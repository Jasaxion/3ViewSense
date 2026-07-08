#!/bin/bash
# Evaluate one or more VLMs on the OrthoMind-3D benchmark (full + ood splits) via a vLLM server.
# Usage: bash run_spatial_eval.sh /path/to/model1 [/path/to/model2 ...]
set -uo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

VLLM_PORT=${VLLM_PORT:-8000}
MAX_TOKENS=${MAX_TOKENS:-4096}
BATCH_SIZE=${BATCH_SIZE:-256}
SPLITS=("full" "ood")

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Usage: bash run_spatial_eval.sh /path/to/model1 [/path/to/model2 ...]"
    exit 1
fi

kill_vllm() {
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vllm server (PID=$VLLM_PID)..."
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
        sleep 5
    fi
}

wait_for_vllm() {
    local url="http://localhost:${VLLM_PORT}/v1/models"
    local max_wait=300 waited=0
    echo "Waiting for vllm server at $url ..."
    while ! curl -s "$url" > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [ $waited -ge $max_wait ]; then
            echo "ERROR: vllm server did not start within ${max_wait}s"
            return 1
        fi
    done
    echo "vllm server is ready (took ${waited}s)"
}

trap kill_vllm EXIT

for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL")
    echo "=== Starting vllm server for: $MODEL_NAME ==="
    vllm serve "$MODEL" \
        --port "$VLLM_PORT" \
        --dtype bfloat16 \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code \
        > "/tmp/vllm_${MODEL_NAME}.log" 2>&1 &
    VLLM_PID=$!

    if ! wait_for_vllm; then
        echo "FAILED to start vllm for $MODEL_NAME, skipping"
        kill_vllm
        continue
    fi

    for SPLIT in "${SPLITS[@]}"; do
        echo "[$MODEL_NAME | $SPLIT] $(date '+%Y-%m-%d %H:%M:%S')"
        PYTHONPATH=.. python eval_vlm_with_vllm.py \
            --model_path "$MODEL" \
            --split "$SPLIT" \
            --output_dir "./results/$SPLIT" \
            --batch_size "$BATCH_SIZE" \
            --max_tokens "$MAX_TOKENS" \
            --port "$VLLM_PORT"
    done

    kill_vllm
done

echo "=== ALL EVALUATIONS COMPLETE: $(date '+%Y-%m-%d %H:%M:%S') ==="
