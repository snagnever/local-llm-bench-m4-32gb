# Final 100-Question Benchmark Results (CORRECTED)
**Date:** 2026-04-10 to 2026-04-12
**Hardware:** MacBook Air M4, 32GB, 120 GB/s bandwidth (fanless)
**Engine:** LM Studio (llama.cpp GGUF)
**Harness:** `scripts/bench2.py` (v2 with full logging)

**NOTE:** This report supersedes the previous version. MATH scores were corrupted by two bugs
(max_tokens=4096 cap + broken `\boxed{}` regex). All MATH runs were rerun with max_tokens=32768
and fixed LaTeX-normalized comparison. See `AUDIT_REPORT.md` for details.

**For inference engine comparison, REAP models, MLX vs LM Studio, and long-context performance**, see `research/NIGHT_FINDINGS.md` — that's where we tested alternative inference stacks (MLX, mlx_vlm, turboquant) and found that `deadbydawn101/gemma-4-21b-REAP-Tool-Calling-mlx-4bit` runs 3-5x faster than the LM Studio Gemma 4 baseline this document uses.

**For tool-calling benchmarks**, see `results/tool_calling_results.md` — 11-run suite (April 14). **Key finding:** this knowledge-bench ranking has zero predictive power for tool calling. **Gemma 4 21B REAP** (same REAP-pruned weights tested in NIGHT_FINDINGS, via `barozp/gemma-4-21b-a4b-it-REAP-GGUF` in LM Studio) scored the **worst** of any baseline on knowledge (77.3% avg) and the **best** of all 10 models on tool calling (96.2% combined). The same weights also run 1.9× faster as GGUF vs MLX on tool-calling workloads because short outputs are prefill-dominated. The tool-calling winner is not a future experiment — it's a baseline from this report re-tested on a different workload.

---

## Quality Results (100 questions each)

| Benchmark | Qwen3-Coder 30B-A3B | Gemma 4 26B-A4B | huihui-claude-i1 |
|-----------|---------------------|-----------------|------------------|
| **MMLU** | 67% | **84%** | 78% |
| **HumanEval** | 94% | **99%** | 91% |
| **MATH** | 78% | **82%** | 73% |
| **DROP** | 80% | 89% | 89% |
| **GPQA** | 42% | **64%** | 54% |
| **Average (5 bench)** | 72.2% | **83.6%** | 77.0% |

**Gemma 4 is best on every benchmark** (tied with huihui on DROP). The previous claim that Gemma was worst at MATH or untestable on GPQA was wrong — both were artifacts of bugs in the original runs.

For context on the GPQA score: GPT-4 ≈ 40%, Claude 3.5 Sonnet ≈ 60%, Claude 4 Opus ≈ 85%. A 16GB local MoE model reaching 64% is striking.

---

## Speed Results (100 questions each, minutes)

| Benchmark | Qwen3-Coder | Gemma 4 | huihui-ref |
|-----------|-------------|---------|------------|
| **MMLU** | **9** | 73 | 70 |
| **HumanEval** | **15** | 185 | 84 |
| **MATH** | **66** | 338 | 647 |
| **DROP** | **11** | 96 | 27 |
| **GPQA** | **38** | 687 | 102 |
| **Total** | **139** | **1440** (24h) | **930** (15.5h) |

Gemma 4 total is ~10x longer than Qwen3-Coder. The bulk comes from MATH (338 min) and GPQA (687 min) where thinking models legitimately need the time.

---

## Bug Impact: MATH Before/After Correction

| Model | Original (Bugged) | Corrected | Change | Truncations at 4096 | Truncations at 32768 |
|-------|------------------|-----------|--------|---------------------|----------------------|
| Qwen3-Coder | 74% | **78%** | +4pp | 1 | 0 |
| Gemma 4 | 59% | **82%** | **+23pp** | 25 | 0 |
| huihui-ref | 51% | **73%** | **+22pp** | 32 | 11 |

**The two bugs:**
1. `max_tokens=4096` hardcoded in MATH function while other benchmarks used 16384 — thinking models burned entire budget on reasoning, never reached answer
2. Broken `\boxed{}` regex couldn't handle nested braces like `\boxed{\frac{1}{2}}` — undercounted correct answers for all models

After fixing both + LaTeX normalization (`\frac83` == `\frac{8}{3}`), Gemma 4 went from worst at MATH to **best at MATH**. huihui-ref still has 11 questions that spiral past 32,768 tokens.

---

## Model Profiles (Fact-Based)

### Qwen3-Coder-30B-A3B (17GB Q4_K_M)
- **Scores:** MMLU 67%, HumanEval 94%, MATH 78%, DROP 80%, GPQA 42% — avg 72.2%
- **Speed:** Total 139 min for 5 benchmarks (2.3 hours) — fastest by 5-7x
- **Thinking:** Zero — 0 reasoning tokens per question
- **Token efficiency:** Average 970 completion tokens on MATH (all visible, no thinking)
- **Truncations:** 0 across all benchmarks (no thinking = fits in any budget)
- **Strengths:** Speed, code quality (94% HE), no thinking overhead
- **Weaknesses:** Weakest MMLU, weakest GPQA — knowledge/reasoning without thinking has limits

### Gemma 4 26B-A4B (16GB Q4_K_M)
- **Scores:** MMLU 84%, HumanEval 99%, MATH 82%, DROP 89%, **GPQA 64%** — avg 83.6%
- **Speed:** Total 1440 min for 5 benchmarks (24 hours) — slowest overall
- **Thinking:** Heavy — on GPQA, ~4700 reasoning tokens per question on average
- **Token efficiency:** 2362 avg tokens on MATH (1946 think + 416 vis), 4966 avg on GPQA
- **Truncations:** 0 at 32768 for MATH; 7 timeouts at 30min on GPQA (counted as wrong)
- **Strengths:** Highest scores on every benchmark including hard reasoning (GPQA 64%)
- **Weaknesses:** Slow (24h total), memory-intensive (18-19GB RAM, up to 11GB swap)
- **Capabilities:** Vision support, reasoning on/off toggle (untested)

### huihui-claude-i1 (14GB Q3_K_S)
- **Scores:** MMLU 78%, HumanEval 91%, MATH 73%, DROP 89%, GPQA 54% — avg 77.0%
- **Speed:** Total 930 min for 5 benchmarks (15.5 hours) — slowest overall due to MATH
- **Thinking:** Moderate-heavy — but 11 MATH questions spiraled past 32,768 tokens
- **Token efficiency:** 2038 avg on MATH but with 11 wasted full-budget spirals
- **Truncations:** 11 MATH questions hit 32k limit (these need 64k+ to answer)
- **Strengths:** Balanced scores across most benchmarks, best DROP tied with Gemma
- **Weaknesses:** Thinking spirals on hard MATH, 17 hours for a single benchmark

---

## Rankings by Benchmark

| Benchmark | #1 | #2 | #3 |
|-----------|-----|-----|-----|
| **MMLU** | Gemma 4 (84%) | huihui (78%) | Qwen3-Coder (67%) |
| **HumanEval** | Gemma 4 (99%) | Qwen3-Coder (94%) | huihui (91%) |
| **MATH** | **Gemma 4 (82%)** | Qwen3-Coder (78%) | huihui (73%) |
| **DROP** | Gemma 4 = huihui (89%) | — | Qwen3-Coder (80%) |
| **GPQA** | **Gemma 4 (64%)** | huihui (54%) | Qwen3-Coder (42%) |
| **Speed** | **Qwen3-Coder (139 min)** | huihui (930 min) | Gemma 4 (1440 min) |

---

## Use Case Recommendations

| Use Case | Best Model | Reason |
|----------|-----------|--------|
| **Speed-critical tasks** | Qwen3-Coder | 10x faster than Gemma 4, still scores 72% average |
| **Best overall quality** | Gemma 4 | Best on all 5 benchmarks, 83.6% average |
| **Coding** | Qwen3-Coder or Gemma 4 | Qwen fast (94%, 15 min), Gemma best (99%, 185 min) |
| **Math / reasoning** | Gemma 4 | 82% MATH, 64% GPQA (between GPT-4 and Claude 3.5 Sonnet) |
| **Knowledge questions** | Gemma 4 | 84% MMLU, 64% GPQA |
| **Reading comprehension** | huihui or Gemma | Tied at 89% DROP |
| **Long batch jobs** | Qwen3-Coder | Its speed compounds over many queries |

**Single-model pick:** Gemma 4 26B-A4B for quality when you can wait, Qwen3-Coder for speed. The time/quality tradeoff is significant — Gemma 4 is **10x slower** for 11.4 percentage points better average. Efficiency per quality point: Qwen3-Coder 1.9 min/pp vs Gemma 4 12.2 min/pp (Qwen3-Coder is 6.4x more efficient per quality gained).

---

## Methodology Notes

- All models tested on identical questions (seed=42 sample of 100 from each dataset)
- Parameters: temp=0, max_tokens=32768 (MATH/GPQA reruns) or 16384 (original runs)
- Context length: 65536 for MATH/GPQA reruns, 16384 for Qwen3-Coder (memory constraint)
- MATH answer grading uses `extract_boxed()` with brace-depth counting + LaTeX normalization
- HumanEval uses real code execution to verify correctness
- MMLU/GPQA use letter matching (A/B/C/D) from model output
- DROP uses substring matching against expected answers
- Per-question logging: prompt_tokens, completion_tokens, reasoning_tokens, finish_reason, GPU temp, RAM, swap, raw response text
- GPU temperature stable at 60-73°C throughout all runs — no thermal throttling observed
- Gemma 4 GPQA had 7 per-question timeouts at 1800s (30 min); these counted as wrong

---

## Open Questions

1. **Gemma 4 reasoning=off mode** — Gemma 4 exposes a `reasoning: on/off` toggle. Untested. Could potentially run at Qwen3-Coder-like speed without thinking. Would be valuable as a "fourth model" in comparisons.
2. **huihui-ref MATH spirals** — 11 questions hit 32,768 token limit. Would need 64k+ max_tokens to know if model can answer them. Low priority (huihui is not the MATH leader).
3. **MMLU-Pro and Humanity's Last Exam** — logical next benchmarks. MMLU-Pro for better differentiation at the top end. HLE probably too hard but worth a probe.
4. **Vision benchmarks** — Gemma 4 has vision capability. Untested on any vision tasks.
5. **Thermal behavior at 20h+ sustained loads** — longest observed run was 11.4h (Gemma GPQA). No throttling so far, but untested at longer durations.

---

## Data Files

- `results/extracted/master_summary.json` — aggregated results (this report)
- `results/extracted/master_results.csv` — CSV for spreadsheets
- `results/extracted/per_question_data_v2.jsonl` — per-question detail from new runs
- `results/runs/*.jsonl` — raw per-question logs with full response text
- `results/runs/*_summary.json` — per-run summaries with hardware/config
- `results/benchmark_analysis_executed.ipynb` — charts and analysis
