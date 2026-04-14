#!/bin/bash
# Speed probe: Test all models with clean state protocol
# Usage: ./scripts/run_speed_probe.sh

OUTPUT_DIR="results/speed_probe"
mkdir -p "$OUTPUT_DIR"

MODELS=(
    "google/gemma-4-26b-a4b"
    "huihui-qwen3.5-35b-a3b-claude-4.6-opus-abliterated-i1"
    "qwen/qwen3.5-9b"
    "qwen/qwen3.5-35b-a3b"
    "glm-4.7-flash"
)

echo "========================================"
echo "Speed Probe - $(date)"
echo "Models: ${#MODELS[@]}"
echo "Output: $OUTPUT_DIR"
echo "========================================"

for model in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "LOADING: $model"
    echo "========================================"

    # Step 1: Clean state
    lms unload --all 2>/dev/null
    sleep 3

    # Step 2: Load model
    lms load "$model" --gpu max --context-length 16384 2>&1 | tail -2

    # Step 3: Wait for model to be ready
    sleep 2

    # Step 4: Verify model responds
    HEALTH=$(curl -s http://127.0.0.1:1234/v1/models 2>/dev/null)
    if echo "$HEALTH" | grep -q "data"; then
        echo "Model loaded and API ready"
    else
        echo "ERROR: Model not responding, skipping"
        continue
    fi

    # Step 5: Run speed probe with monitoring
    python3 scripts/speed_probe.py "$model" "$OUTPUT_DIR"

    # Step 6: Unload
    lms unload --all 2>&1 | tail -1
    sleep 3
done

echo ""
echo "========================================"
echo "ALL PROBES COMPLETE - $(date)"
echo "========================================"
echo ""
echo "Results in: $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR/"*_results.json 2>/dev/null
