#!/usr/bin/env bash
# Run all 4 benchmarks sequentially with thermal cooldown between
# Skip GPQA (mlx_vlm bug — actually works now after temperature fix, see WORKLOG_NIGHT.md)
# Order: fastest first → slowest last (DROP, MMLU, HumanEval, MATH)

set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source .venv-mlx/bin/activate

MODEL="deadbydawn101/gemma-4-21b-REAP-Tool-Calling-mlx-4bit"
LOG_DIR="results/runs"
COOLDOWN_TARGET=60   # Wait until GPU drops below this °C
COOLDOWN_TIMEOUT=600 # Max 10 min wait
mkdir -p "$LOG_DIR"

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
    echo ""
    echo "=========================================="
    echo "  STATE CHECK: $label @ $(date +%H:%M:%S)"
    echo "=========================================="
    local SWAP=$(sysctl vm.swapusage | grep -oE 'used\s*=\s*[0-9.]+M' | grep -oE '[0-9.]+')
    local TEMP=$(get_gpu_temp)
    echo "  Swap used: ${SWAP}M"
    echo "  GPU temp:  ${TEMP}°C"
    echo "  Free pages: $(vm_stat | grep "Pages free" | awk '{print $3}')"
    echo ""
}

cooldown() {
    local label=$1
    echo ""
    echo "=========================================="
    echo "  COOLDOWN after $label — waiting for GPU < ${COOLDOWN_TARGET}°C"
    echo "=========================================="
    local START=$(date +%s)
    while true; do
        local TEMP=$(get_gpu_temp)
        local NOW=$(date +%s)
        local ELAPSED=$((NOW - START))
        if [ "$TEMP" -lt "$COOLDOWN_TARGET" ] 2>/dev/null; then
            echo "  ✓ GPU at ${TEMP}°C after ${ELAPSED}s — proceeding"
            return 0
        fi
        if [ "$ELAPSED" -gt "$COOLDOWN_TIMEOUT" ]; then
            echo "  ⚠ Cooldown timeout (${COOLDOWN_TIMEOUT}s) reached, GPU still at ${TEMP}°C — proceeding anyway"
            return 0
        fi
        echo "  Waiting... GPU ${TEMP}°C → target ${COOLDOWN_TARGET}°C (elapsed ${ELAPSED}s)"
        sleep 15
    done
}

run_bench() {
    local BENCH=$1
    local START_TIME=$(date +%H:%M:%S)
    echo ""
    echo "=========================================="
    echo "  STARTING: $BENCH @ $START_TIME"
    echo "=========================================="
    python3 scripts/mlx_bench.py "$BENCH" \
        --examples 100 \
        --model "$MODEL" 2>&1 | tee "$LOG_DIR/console_mlx_${BENCH}_$(date +%Y%m%d_%H%M%S).log"
    local END_TIME=$(date +%H:%M:%S)
    echo ""
    echo "  FINISHED $BENCH at $END_TIME"
}

# Pre-flight
echo "==========================================="
echo "  MLX BENCHMARK SUITE — full 100q, deadbydawn REAP"
echo "  Started: $(date)"
echo "  Thermal safety: cooldown to ${COOLDOWN_TARGET}°C between benches"
echo "==========================================="
check_state "PRE-FLIGHT"

# Make sure LM Studio is unloaded
python3 scripts/lms.py unload 2>&1 || true

# Cooldown BEFORE starting (in case we're coming off a previous run)
cooldown "PRE-FLIGHT"

# Run benchmarks fast-to-slow with cooldown between
run_bench "drop"
check_state "AFTER DROP"
cooldown "DROP"

run_bench "mmlu"
check_state "AFTER MMLU"
cooldown "MMLU"

run_bench "humaneval"
check_state "AFTER HUMANEVAL"
cooldown "HUMANEVAL"

run_bench "math"
check_state "AFTER MATH"

echo ""
echo "==========================================="
echo "  ALL DONE: $(date)"
echo "==========================================="
