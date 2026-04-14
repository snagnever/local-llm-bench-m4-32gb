#!/usr/bin/env bash
# Overnight tool-calling benchmark runner for 10 models × 2 suites.
# Unloads previous model before loading the next. Thermal gate ≤60°C per question
# is enforced by tool_call_bench.py. Auto-skip reuses completed prior entries.
#
# Model order: fastest/smallest first so we bank wins early if the run has to abort.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source .venv-mlx/bin/activate

LOG_DIR="results/runs"
mkdir -p "$LOG_DIR"

COOLDOWN_TARGET=60
COOLDOWN_TIMEOUT=600

get_gpu_temp() {
    macmon pipe 2>/dev/null | head -1 | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    t = d.get('temp', {}).get('gpu_temp_avg', 0)
    print(round(t) if 5 <= t <= 120 else 50)
except:
    print(50)
" 2>/dev/null || echo 50
}

check_state() {
    local label=$1
    local ts
    ts=$(date +%H:%M:%S)
    echo ""
    echo "=========================================="
    echo "  STATE: $label @ $ts"
    echo "=========================================="
    local SWAP TEMP FREE
    SWAP=$(sysctl vm.swapusage | grep -oE 'used\s*=\s*[0-9.]+M' | grep -oE '[0-9.]+')
    TEMP=$(get_gpu_temp)
    FREE=$(vm_stat | grep "Pages free" | awk '{print $3}')
    echo "  Swap used: ${SWAP}M"
    echo "  GPU temp:  ${TEMP}°C"
    echo "  Free pages: $FREE"
    echo ""
}

cooldown() {
    local label=$1
    echo ""
    echo "=========================================="
    echo "  COOLDOWN after $label — waiting for GPU < ${COOLDOWN_TARGET}°C"
    echo "=========================================="
    local START NOW TEMP ELAPSED
    START=$(date +%s)
    while true; do
        TEMP=$(get_gpu_temp)
        NOW=$(date +%s)
        ELAPSED=$((NOW - START))
        if [ "$TEMP" -lt "$COOLDOWN_TARGET" ] 2>/dev/null; then
            echo "  ✓ GPU at ${TEMP}°C after ${ELAPSED}s — proceeding"
            return 0
        fi
        if [ "$ELAPSED" -gt "$COOLDOWN_TIMEOUT" ]; then
            echo "  ⚠ Cooldown timeout, GPU still ${TEMP}°C — proceeding anyway"
            return 0
        fi
        echo "  Waiting... GPU ${TEMP}°C → ${COOLDOWN_TARGET}°C (elapsed ${ELAPSED}s)"
        sleep 15
    done
}

run_model() {
    local lms_key=$1
    local display=$2
    local ctx=${3:-8192}

    echo ""
    echo "=============================================="
    echo "  MODEL: $display ($lms_key) ctx=$ctx"
    echo "  Started: $(date)"
    echo "=============================================="

    # Always unload everything first, wait a beat for memory to settle
    lms unload --all 2>&1 | head -3 || true
    sleep 2

    check_state "BEFORE LOAD ($display)"
    cooldown "BEFORE LOAD ($display)"

    echo "  Loading $lms_key ..."
    if ! lms load "$lms_key" --context-length "$ctx" 2>&1 | tail -5; then
        echo "  ⚠ Load failed for $lms_key — skipping"
        return 0
    fi
    sleep 2

    check_state "AFTER LOAD ($display)"

    # jdhodges suite
    echo ""
    echo "  ---- $display :: jdhodges ----"
    python3 scripts/tool_call_bench.py \
        --model "$lms_key" --suite jdhodges 2>&1 \
        | tee "$LOG_DIR/toolcall_jdhodges_${lms_key//\//_}_console_$(date +%Y%m%d_%H%M%S).log"

    check_state "AFTER jdhodges ($display)"

    # veerman suite
    echo ""
    echo "  ---- $display :: veerman ----"
    python3 scripts/tool_call_bench.py \
        --model "$lms_key" --suite veerman 2>&1 \
        | tee "$LOG_DIR/toolcall_veerman_${lms_key//\//_}_console_$(date +%Y%m%d_%H%M%S).log"

    check_state "AFTER veerman ($display)"

    # Unload to release memory for next model
    lms unload --all 2>&1 | head -3 || true
    sleep 2

    echo "  FINISHED $display at $(date +%H:%M:%S)"
}

echo "=========================================="
echo "  TOOL-CALLING BENCHMARK SUITE"
echo "  10 models × (jdhodges 40 + veerman 12)"
echo "  Started: $(date)"
echo "=========================================="

check_state "PRE-FLIGHT"
cooldown "PRE-FLIGHT"

# Order: small/fast → large. Existing baselines interleaved so user gets
# cross-comparison early if the run has to abort.

# ---- NEW candidates ordered by size ----
run_model "qwen_qwen3.5-4b" "Qwen3.5-4B (GGUF Q4)" 8192
run_model "nvidia-nemotron-3-nano-4b" "NVIDIA Nemotron 3 Nano 4B (GGUF Q4)" 8192
run_model "deephermes-toolcalling-specialist-atropos" "DeepHermes-ToolCalling-Atropos (GGUF Q4)" 8192
run_model "mistral-nemo-instruct-2407" "Mistral Nemo 12B (MLX 4bit)" 8192
run_model "hermes-4-14b" "Hermes-4-14B (MLX 4bit)" 8192

# ---- EXISTING baselines from previous benches ----
run_model "gemma-4-21b-a4b-it-reap" "Gemma 4 21B REAP (GGUF Q4)" 8192
run_model "zai-org/glm-4.7-flash" "GLM-4.7-Flash (GGUF Q4)" 8192
run_model "google/gemma-4-26b-a4b" "Gemma 4 26B A4B (GGUF Q4)" 8192
run_model "qwen3-coder-30b-a3b-instruct" "Qwen3 Coder 30B A3B (GGUF Q4)" 8192
run_model "huihui-qwen3.5-35b-a3b-claude-4.6-opus-abliterated-i1" "huihui Claude-4.6-Opus abliterated 35B (GGUF Q3)" 8192

check_state "FINAL"

echo ""
echo "=========================================="
echo "  ALL MODELS COMPLETE: $(date)"
echo "=========================================="
