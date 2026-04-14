> **ARCHIVED — April 8, 2026 original plan**
> This document describes the plan made before any 100q runs were executed.
> It is preserved as historical record of the methodology design.
>
> **Execution status:** Plan executed April 8-12, 2026. Models tested:
> Qwen3-Coder 30B-A3B, Gemma 4 26B-A4B, huihui-qwen3.5-35b-a3b-claude-opus-abliterated-i1
>
> **Canonical results:** `results/FINAL_100Q_RESULTS.md`
> **Current harness:** `scripts/bench2.py` (not bench.py described below)
> **Bug history:** `results/AUDIT_REPORT.md`

---

# Action Plan: Local LLM Benchmarking on 32GB M4 Mac
**Date:** 2026-04-08 | **Based on:** 11 research reports from 11 agents

---

## What We Got Wrong Before

| Issue | What We Did | What We Should Do |
|-------|-------------|-------------------|
| **Framework** | simple-evals (deprecated July 2025) | lm-evaluation-harness (the standard) |
| **Sample size** | 20 questions (CI = +/-20pp) | 100+ minimum, 200+ ideal |
| **Benchmarks** | MMLU (contaminated, saturated) | MMLU-Pro, IFEval, GSM8K |
| **Temperature** | 0.5 (high variance at n=20) | 0 (reproducible) |
| **max_tokens** | 2048 (truncates thinking) | 4096+ |
| **Context** | 8192 | 16384 (for thinking models) |
| **Quantization** | Q3_K_S (15.3GB) | APEX Quality (21.3GB) or Q4_K_M+ |
| **Engine** | LM Studio llama.cpp | MLX-native (Rapid-MLX or mlx_lm.server) |
| **Model** | Abliterated+distilled (GPQA 60%) | Test pure base first |

---

## Phase 1: Setup New Benchmark Infrastructure

### Engine Choice: Start with LM Studio (we have it), plan MLX migration

**LM Studio** is our current setup and still works for GGUF models via llama.cpp backend. It has bugs with newer models (Gemma 4, Qwen3.5 MoE detection) but works fine for models already loaded.

**For models LM Studio can't load**, use:
- `mlx_lm.server` for MLX models (pip install mlx-lm)
- `llama-server` from llama.cpp for any GGUF

All expose OpenAI-compatible APIs, so our benchmark harness works with any of them.

### Benchmark Tool: lm-evaluation-harness

```bash
pip install "lm_eval[api]"
pip install langdetect immutabledict  # for IFEval

# Run against LM Studio (or any OpenAI-compatible API)
lm_eval --model local-chat-completions \
  --model_args model=local,base_url=http://localhost:1234/v1/chat/completions,num_concurrent=1,timeout=600 \
  --tasks mmlu_pro,ifeval,gsm8k_cot \
  --apply_chat_template \
  --limit 100 \
  --output_path ./results
```

### Alternative: Keep simple-evals for quick probes
Our modified simple-evals still works for fast sanity checks. Just fix:
- temperature: 0.5 → 0
- max_tokens: 2048 → 4096
- n_examples: 20 → 100+

---

## Phase 2: Model Candidates (16 models, 3 tiers)

### Quick-Probe Round (5 min per model - 2 questions)
Goal: Filter out broken/unusably-slow models

| # | Model | GGUF Size | Engine |
|---|-------|-----------|--------|
| 1 | Gemma-4-26B-A4B-it | 13.4GB IQ4 | mlx_lm or llama-server |
| 2 | GLM-4.7-Flash | 14.6GB Q3_K_M | LM Studio |
| 3 | Qwen3.5-35B-A3B base (official) | 15.3GB Q3_K_S | LM Studio |
| 4 | HauhauCS Uncensored-Aggressive | 15.3GB Q3_K_S | LM Studio |
| 5 | Qwen3.5-9B | 5.7GB Q4_K_M | LM Studio |
| 6 | Nemotron-Cascade-2-30B-A3B | ~20GB Q3_K_M | LM Studio |
| 7 | Huihui plain abliterated (no Claude) | 15.3GB Q3_K_S | LM Studio |
| 8 | Qwen3-Coder-30B-A3B | 13.3GB Q3_K_S | LM Studio |

### Mini-Bench Round (survivors from probe - 10 questions each)
Run simple-evals with 10 questions from MMLU, MATH, HumanEval, DROP

### Full Benchmark (top 3-5 models - 100+ questions)
Switch to lm-eval-harness with:
- MMLU-Pro (100 questions, stratified)
- IFEval (full 500, rule-based, fast)
- GSM8K (100 questions)
- GPQA Diamond (full 198)

---

## Phase 3: Quantization Optimization (for the winner)

Once we identify the best model architecture, test quantization levels:

| Quant | Size | Expected Quality |
|-------|------|-----------------|
| Q3_K_S | 15.3GB | ~88% of FP16 |
| APEX Compact | 16.1GB | ~95% (MoE-aware) |
| Q4_K_M | 22GB | ~92% |
| APEX Quality | 21.3GB | ~100% (beats FP16!) |
| Q5_K_M | 26GB | ~95-97% |

**Key finding: APEX Quality (21.3GB) actually BEATS F16 perplexity** for Qwen3.5-35B-A3B through implicit regularization of MoE shared experts. This is the optimal quant if it fits.

---

## Phase 4: Engine Optimization -- COMPLETED

**Result:** LM Studio (llama.cpp GGUF) is the fastest engine on M4 base 32GB.
Tested 5 engines on 2 models (see results/engine_comparison.md):
- LM Studio: 1.5x faster than all alternatives
- MLX engines: no speed advantage at 120 GB/s bandwidth
- Ollama: consistently slowest (Go wrapper overhead)

---

## Monitoring & Measurement Tools

### Installed:
- **macmon** (`brew install macmon`) - real-time CPU/GPU/memory/power monitoring
  - `macmon` for TUI, `macmon pipe` for JSON output, `macmon serve` for HTTP
  - No sudo required. Use during benchmarks to detect GPU spill or thermal throttling.
- **local-llm-bench** (tools/local-llm-bench/) - effective throughput measurement
  - Measures wall-clock time including prefill (not just decode tok/s)
  - `python3 tools/local-llm-bench/bench.py --backend lmstudio --model <model>`

### How to use during benchmarks:
```bash
# Terminal 1: Start macmon logging to file
macmon pipe > results/macmon_log.jsonl &

# Terminal 2: Run benchmark
python3 scripts/engine_benchmark.py ...

# Terminal 3 (optional): Watch real-time
macmon
```

### Key metrics to watch:
- `gpu_usage` > 90%: good, model is on GPU
- `swap_usage` increasing: bad, model is spilling to disk
- `gpu_power` near 0 during inference: model fell back to CPU

---

## Critical Parameters for All Tests

```
temperature: 0
max_tokens: 4096
context: 16384
parallel_slots: 1
flash_attention: on
kv_cache_type: q8_0
system_prompt: "You are a helpful assistant."
```

---

## Expected Timeline

| Phase | Duration | What |
|-------|----------|------|
| Setup | 30 min | Install lm-eval-harness, download 2-3 models |
| Quick-probe | 40 min | 8 models x 5 min each |
| Mini-bench | 2 hrs | 5 survivors x 10 questions x 4 benchmarks |
| Full bench | 4-8 hrs | Top 3 models x 100+ questions x 4 benchmarks |
| Quant test | 2-4 hrs | Winner at 3-4 quant levels |
| Engine test | 1-2 hrs | Winner on 3-4 engines |

---

## Key Hypotheses to Test

1. **"Pure Qwen3.5-35B-A3B base > our abliterated+distilled version"** - especially on GPQA (84.2 vs 60%)
2. **"Gemma-4-26B-A4B has highest Arena ELO (1441) but lower raw benchmarks"** - is it actually better in practice?
3. **"Abliteration + quantization compounds quality loss"** - compare abliterated vs base at same quant
4. **"APEX Quality quant > standard Q4_K_M"** - test MoE-aware quantization
5. **"MLX is 3x faster than llama.cpp for MoE"** - verify on our M4 hardware
