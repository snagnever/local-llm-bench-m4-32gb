> **ARCHIVED — April 8, 2026 research synthesis**
> This document synthesizes web/Reddit/Twitter research done BEFORE any actual
> 100q runs. It represents our best guesses about which models to test and why.
>
> **The "model ranking" and "test plan" sections were SUPERSEDED** by actual
> 100q benchmark results. See `results/FINAL_100Q_RESULTS.md`.
>
> **Still valid:** the "Confirmed Findings" about models, engines, methodology.
> These informed our benchmark design and remain accurate.

---

# Final Synthesis: Local LLM Benchmarking on 32GB M4 Mac
**Date:** 2026-04-08 | **Based on:** 16 research reports across 3 passes

---

## Confirmed Findings (High Confidence)

### Models
| Finding | Status | Sources |
|---------|--------|---------|
| Gemma 4 26B-A4B Arena ELO 1441 (highest MoE that fits) | **Confirmed** | Arena.ai, multiple X posts |
| Gemma 4 31B Dense Arena ELO ~1452 also fits at Q4 (~18-20GB) | **New - adds a contender** | Arena.ai |
| APEX quantization beats F16 perplexity (6.527 vs 6.537) | **Strongly confirmed** | mudler on X, HF collection of 23 models |
| Nemotron-Cascade-2 gold medals (IMO/IOI/ICPC) | **Confirmed** | NVIDIA paper, VentureBeat |
| HauhauCS most popular uncensored (815K dl, proprietary method) | **Confirmed** | HF stats, NOT standard abliteration |
| Qwen3.5-35B-A3B base: MMLU-Pro 85.3, GPQA 84.2 | **Confirmed** | Official Qwen benchmarks |
| Huihui3.5-67B-A3B exists (512 experts, 3B active) | **Confirmed** | HF, 1 day old |
| Huihui3.5-97B-A3B does NOT exist | **Corrected** | Not found on HF |

### Engines
| Finding | Status | Sources |
|---------|--------|---------|
| Ollama 0.20 is latest (April 2), has Gemma 4 support | **Updated from 0.19** | Ollama blog |
| LM Studio GGUF works for Gemma 4 (after runtime update) | **Fixed** | GitHub #1728 closed |
| LM Studio MLX broken for Gemma 4 AND Qwen3.5 MoE | **Still broken** | GitHub #1741, #1723 |
| MLX is 1.5-2x faster than llama.cpp (not 3x) | **Corrected** | Academic study, famstack |
| MLX tool-calling degrades after 5-10 rounds (GGUF doesn't) | **New critical finding** | @TeksEdge |
| Rapid-MLX claims 2-4x faster than Ollama | **Single source, unverified** | @Raullen only |
| MLX caching is broken for Qwen3.5 DeltaNet (10-100x slowdown) | **Confirmed** | mlx-lm GitHub PRs |

### Methodology
| Finding | Status | Sources |
|---------|--------|---------|
| simple-evals "not actively maintained" (not deprecated) | **Corrected** | GitHub README |
| lm-eval-harness is standard for leaderboards | **Confirmed** | HF, torchtune |
| lm-eval-harness CAN'T do MCQ via chat API (needs logprobs) | **Critical new finding** | GitHub #2934 |
| 20 questions = CI of +/-20pp, statistically meaningless | **Confirmed** | Cameron Wolfe paper |
| MMLU contaminated/saturated, MMLU-Pro better | **Confirmed** | HF community |
| Abliteration hurts MoE more than dense models | **Confirmed** | arxiv 2512.13655 |
| Abliteration + Q3 compounds less than feared for dense | **Nuanced** | Same paper |

### Quantization
| Finding | Status | Sources |
|---------|--------|---------|
| APEX is real, available, 23 models on HF | **Confirmed** | mudler HF collection |
| MoE models MORE robust to quant than dense | **Confirmed** | Benjamin Marie, research |
| Shared expert must stay Q8+ | **Confirmed** | APEX design, kurtosis data |
| KV cache q8_0 has no meaningful quality loss | **Confirmed** | Benjamin Marie |
| Activation rotation (April 1) dramatically helps Q4 KV | **New** | ggerganov PR #21038 |

---

## Corrections to Our Action Plan

### 1. Benchmark Framework: Hybrid Approach (not pure lm-eval-harness)

**Problem:** lm-eval-harness can't evaluate MCQ tasks (MMLU-Pro, GPQA, ARC) via local API - needs logprobs from completion endpoint.

**Solution:**
- **MCQ tasks (MMLU-Pro, GPQA):** Keep using simple-evals or custom script that generates text answers (A/B/C/D) via chat API. Or use lm-eval-harness with `local-completions` (not chat) if llama-server exposes logprobs.
- **Generative tasks (GSM8K, IFEval, coding):** Use lm-eval-harness via `local-chat-completions`
- **Speed benchmarks:** Use local-llm-bench for effective throughput

### 2. Engine Strategy: LM Studio GGUF for now, test MLX carefully

**Problem:** MLX has broken caching for Qwen3.5 DeltaNet AND tool-calling degradation after 5-10 rounds. Rapid-MLX claims are unverified.

**Solution:**
- **Primary:** LM Studio (llama.cpp GGUF backend) - most reliable, works for all models
- **Test:** Ollama 0.20 MLX for Gemma 4 specifically
- **Test carefully:** mlx_lm.server for speed comparison, but verify quality matches GGUF
- **Skip for now:** Rapid-MLX (unverified single-source claims)

### 3. Model List: Add Gemma 4 31B Dense and Devstral Small

**New additions from confirmation pass:**
- **Gemma 4 31B Dense** (ELO 1452, fits at Q4 ~18-20GB) - higher quality than MoE
- **Devstral Small 2 24B** - "severely underrated" for coding per Reddit
- **Holo3-35B-A3B** - computer-use MoE, 77.8% OSWorld
- **Qwen3.5-27B Dense** - community says outperforms MoE for quality

**Removed/demoted:**
- **Nemotron-Cascade-2:** May be "benchmaxxed" - great HumanEval but poor agentic. Still worth testing but temper expectations.
- **Rapid-MLX as engine:** Demoted until independently verified

### 4. Quantization: APEX is the clear winner for MoE

APEX models are available on HuggingFace from mudler for all our target models:
- `mudler/Qwen3.5-35B-A3B-APEX-GGUF` (21.3GB Quality, 16.1GB Compact)
- `mudler/gemma-4-26B-A4B-it-APEX-GGUF` (18.1GB Balanced)
- `mudler/GLM-4.7-Flash-APEX-GGUF`
- `mudler/Nemotron-Cascade-2-30B-A3B-APEX-GGUF`

---

## Updated Model Ranking (Post-Confirmation)

| # | Model | GGUF Size | Arena ELO | Key Strength | Confidence |
|---|-------|-----------|-----------|-------------|------------|
| 1 | **Gemma-4-26B-A4B** APEX | 18.1GB | 1441 | Best Arena ELO MoE, smallest | High |
| 2 | **Qwen3.5-35B-A3B base** APEX | 21.3GB | 1400 | Best raw benchmarks MoE | High |
| 3 | **Gemma-4-31B Dense** Q4 | ~18-20GB | ~1452 | Highest Arena ELO overall | High |
| 4 | **GLM-4.7-Flash** APEX | ~14-16GB | - | Fastest, best math (AIME 91.6) | Medium |
| 5 | **HauhauCS Uncensored** | 15.3GB Q3 | - | Most popular uncensored, 262K ctx | Medium |
| 6 | **Qwen3.5-27B Dense** Q4 | ~16-18GB | - | Community says best quality | Medium |
| 7 | **★ Our huihui-claude-opus-i1** | 15.3GB Q3 | - | TESTED: MMLU 90%, GPQA 60% | Baseline |
| 8 | **Nemotron-Cascade-2** APEX | ~20GB | - | Gold medals, maybe benchmaxxed | Medium |
| 9 | **Qwen3.5-9B** Q4 | 5.7GB | - | Beats 120B at tiny size | High |
| 10 | **Devstral Small 2 24B** Q4 | ~13GB | - | Reddit says underrated for code | Low |
| 11 | **Holo3-35B-A3B** APEX | ~21GB | - | Computer-use specialist | Low |
| 12 | **Qwen3-Coder-30B-A3B** Q3 | 13.3GB | - | Coding specialist | Medium |

---

## Updated Test Plan

### Phase 1: Quick Probe (use simple-evals, 5 questions each)
```
temperature: 0
max_tokens: 4096
context: 16384
engine: LM Studio llama.cpp (GGUF)
```

Download and test in this order:
1. Gemma-4-26B-A4B APEX Balanced (18.1GB) - top pick
2. Qwen3.5-35B-A3B APEX Quality (21.3GB) - best benchmarks
3. GLM-4.7-Flash APEX (14-16GB) - fastest
4. Qwen3.5-9B Q4_K_M (5.7GB) - tiny giant
5. Gemma-4-31B Dense Q4 (~18-20GB) - highest ELO

### Phase 2: Mini-Bench (survivors, 20-50 questions)
Mix of MMLU-Pro, MATH, HumanEval, GPQA via simple-evals

### Phase 3: Full Bench (top 3, 100+ questions)
- simple-evals for MCQ tasks (MMLU-Pro 100q, GPQA full 198q)
- lm-eval-harness for generative tasks (GSM8K 100q, IFEval full 500q)
- local-llm-bench for speed measurement

---

## Key Warnings

1. **KV cache varies wildly between models.** Gemma 4 uses 2-3x more KV cache than Qwen3.5 at same context (Qwen3.5 has hybrid RNN, only 10/40 layers need full attention).

2. **MoE speed is deceptive for agents.** Faster generation but more retry cycles means dense models sometimes win on time-to-completion.

3. **Gemma 4 26B-A4B tool calling produces malformed JSON** per Reddit. Needs custom sanitizer for agentic use.

4. **MLX caching is broken for Qwen3.5** (DeltaNet cache can't be trimmed). Stick with GGUF/llama.cpp for now.

5. **Temperature must match model recommendations.** Qwen3.5 recommends 0.6-1.0 for thinking mode. Our temp=0 may suppress thinking quality.

---

## Resources and Communities Discovered

Saved to: `notes/raw/communities_and_resources.md`

Key ones:
- **r/LocalLLaMA** (266K+ members) - primary community
- **SiliconBench** - Apple Silicon speed leaderboard
- **MLXBench** - TTFT/throughput comparison tool
- **local-llm-bench** - effective throughput measurement
- **APEX collection** - mudler's HF collection of 23 MoE-optimized GGUFs
- **Benjamin Marie / The Kaitchup** - best quantization analysis source
