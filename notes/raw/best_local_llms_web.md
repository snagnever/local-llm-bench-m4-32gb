# Best Local LLMs for 32GB Apple Silicon Mac (M4) - Web Research

**Date:** 2026-04-08
**Constraint:** ~14-16GB max GGUF file size at Q3-Q4 quantization
**Hardware:** 32GB Apple Silicon M4 Mac

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Current Baseline: huihui-qwen3.5-35b-a3b-claude-4.6-opus-abliterated](#current-baseline)
3. [Top MoE Contenders (3B Active Parameter Class)](#top-moe-contenders)
4. [GGUF File Size Comparison Table](#gguf-file-size-comparison)
5. [Benchmark Comparison Table](#benchmark-comparison)
6. [Detailed Model Profiles](#detailed-model-profiles)
7. [Dense Models Worth Considering](#dense-models)
8. [Abliterated / Uncensored Models](#abliterated-models)
9. [Inference Frameworks](#inference-frameworks)
10. [Recommendations](#recommendations)
11. [Sources](#sources)

---

## Executive Summary

The 3B-active-parameter MoE class is the sweet spot for 32GB Apple Silicon Macs. These models load all experts into memory (requiring 15-22GB at Q3-Q4) but only activate ~3B params per token, giving near-dense-27B quality at 3-4x the speed.

**Key finding:** The Qwen3.5-35B-A3B architecture (and its distilled/abliterated variants) currently leads this class on most benchmarks. However, several strong competitors exist, and the choice depends on use case.

**Top 5 models that fit the 14-16GB constraint at Q3-Q4:**
1. **Qwen3.5-35B-A3B** (Q3_K_S: 15.3GB) - Best overall intelligence
2. **GLM-4.7-Flash** (Q3_K_S: 13.3GB) - Best for coding/agents, fastest inference
3. **Gemma-4-26B-A4B** (UD-Q3_K_XL: 12.9GB) - Strong multimodal, smallest footprint
4. **Qwen3-Coder-30B-A3B** (Q3_K_S: 13.3GB) - Best for pure coding tasks
5. **Nemotron-3-Nano-30B-A3B** (Q3_K_S: 18.1GB) - Best long-context (1M tokens), but too large

---

## Current Baseline

### huihui-qwen3.5-35b-a3b-claude-4.6-opus-abliterated

**Architecture:**
- Base: Qwen3.5-35B-A3B (35B total, 3B active per token)
- Distillation source: Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled
- Abliteration: huihui-ai's refusal-removal via SVD on residual stream
- License: Apache 2.0
- Training: SFT + LoRA (1.31% trainable params) on Claude 4.6 Opus CoT reasoning data
- Format: `<think>{reasoning}</think>\n{answer}` structured output

**Your local benchmark scores (Q3_K_S):**
| Benchmark | Score |
|-----------|-------|
| MMLU | 90% |
| DROP | 95% / F1: 90.6 |
| HumanEval | 90% |
| MATH | 75% |
| GPQA | 60% |

**GGUF File Sizes (mradermacher quantizations):**
| Quant | Size |
|-------|------|
| Q2_K | 13.0 GB |
| Q3_K_S | 15.3 GB |
| Q3_K_M | 16.9 GB |
| Q4_K_S | 20.0 GB |
| Q4_K_M | 21.3 GB |
| Q5_K_M | 24.8 GB |
| Q6_K | 28.6 GB |
| Q8_0 | 37.0 GB |

**Key insight:** Q3_K_S at 15.3GB fits the constraint. Q3_K_M at 16.9GB is borderline. Q4+ variants exceed 16GB and may cause memory pressure on 32GB systems with OS overhead.

**HuggingFace stats:** 136,281 downloads/month, 36 likes
**Ollama:** `ollama run huihui_ai/qwen3.5-abliterated:35b-Claude` (requires Ollama v0.18.0+)

---

## Top MoE Contenders (3B Active Parameter Class)

### 1. Qwen3.5-35B-A3B (Base Model)

**Architecture:**
- 35B total params, 3B active per token
- 256 routed experts, 8 active + 1 shared per token
- Gated DeltaNet + standard attention in 3:1 ratio (linear attention dominant)
- Native context: 262K tokens (extendable to 1M)
- Native multimodal: text, image, video
- 201 language support

**Official Benchmark Scores:**
| Benchmark | Score |
|-----------|-------|
| MMLU-Pro | 85.3 |
| GPQA Diamond | 84.2 |
| LiveCodeBench v6 | 74.6% |
| SWE-bench Verified | 69.2% |
| TAU2-Bench | 81.2% |
| MMMU-Pro (Vision) | 75.1 |

**GGUF Sizes (unsloth):**
| Quant | Size |
|-------|------|
| UD-IQ2_XXS | 10.7 GB |
| UD-IQ2_M | 11.4 GB |
| UD-Q2_K_XL | 12.2 GB |
| UD-IQ3_XXS | 13.1 GB |
| Q3_K_S | 15.3 GB |
| UD-IQ3_S | 13.6 GB |
| Q3_K_M | 16.4 GB |
| UD-Q3_K_XL | 16.6 GB |
| UD-IQ4_XS | 17.5 GB |
| Q4_K_S | 20.7 GB |
| Q4_K_M | 22.0 GB |
| Q8_0 | 36.9 GB |
| BF16 | 69.4 GB |

**Performance:** ~111 tok/s on RTX 3090 (Q4_K_M), ~35 tok/s on RTX 4090 (Q4), estimated 30-50 tok/s on M4 Max via MLX

**Verdict:** Surpasses the previous Qwen3-235B-A22B (which had 22B active) on key benchmarks. The 35B-A3B is essentially the successor that achieves cross-generational parity with far fewer active parameters.

---

### 2. GLM-4.7-Flash (Zhipu AI)

**Architecture:**
- 30B total params, ~3B active per token
- 47 layers, 2048 hidden dimension
- 64 routed experts, 4 active + 1 shared per token
- Standard Transformer MoE (no experimental attention)
- Context: 128K tokens (max 202K)
- License: MIT (more permissive than Apache 2.0)

**Benchmark Scores:**
| Benchmark | Score |
|-----------|-------|
| MMLU-Pro | ~60.0 |
| GPQA Diamond | 75.2 |
| AIME 2025 | 91.6 |
| SWE-bench Verified | 59.2% |
| TAU2-Bench | 79.5% |
| LiveCodeBench v6 | 64.0% |

**GGUF Sizes (unsloth):**
| Quant | Size |
|-------|------|
| UD-TQ1_0 | 8.33 GB |
| UD-IQ1_M | 9.81 GB |
| UD-IQ2_XXS | 10.5 GB |
| Q2_K | 11.3 GB |
| UD-IQ3_XXS | 12.9 GB |
| Q3_K_S | 13.3 GB |
| Q3_K_M | 14.6 GB |
| UD-Q3_K_XL | 13.8 GB |
| IQ4_XS | 16.3 GB |
| Q4_K_S | 17.3 GB |
| Q4_K_M | 18.3 GB |
| Q8_0 | 31.8 GB |
| BF16 | 59.9 GB |

**Performance:** ~60-80+ tok/s at Q4 quantization. Fastest inference in the 3B-active class due to simpler architecture (no experimental DeltaNet layers).

**Key strengths:**
- Best math competition score (AIME 91.6 vs Qwen's 89.0)
- Near-identical agentic performance to Qwen (TAU2: 79.5 vs 81.2)
- MIT license - maximum freedom
- Free Z.AI API with no daily caps
- Fits comfortably at Q3_K_M (14.6GB)

**Key weaknesses:**
- Much lower general knowledge (MMLU-Pro ~60 vs Qwen's 85.3)
- No native vision/multimodal
- Shorter context window (128K vs 262K)

**Abliterated variants available:**
- GLM-4.7-Flash-Grande-Heretic-UNCENSORED (42B total/3B active MoE, 16GB VRAM at Q4)
- GLM-4.7-Flash-Uncensored-Heretic-NEO-CODE-Imatrix-MAX (30B total/3B active, coding-optimized)

---

### 3. Gemma-4-26B-A4B (Google DeepMind)

**Architecture:**
- 25.2B total params, 3.8B active per token
- MoE blocks added alongside standard MLP blocks (outputs summed)
- Context: 256K tokens
- Released April 3, 2026
- License: Apache 2.0 (Gemma terms)

**Benchmark Scores:**
| Benchmark | Score |
|-----------|-------|
| MMLU-Pro | 82.6 |
| GPQA Diamond | 82.3 |
| LiveCodeBench v6 | 77.1 |
| MMMLU | 86.3 |
| MMMU-Pro | 73.8 |
| TAU2-Bench | 68.2 |

**GGUF Sizes (unsloth):**
| Quant | Size |
|-------|------|
| UD-IQ1_M | 9.88 GB |
| UD-IQ2_XXS | 9.88 GB |
| UD-Q2_K_XL | 10.5 GB |
| UD-IQ3_XXS | 11.2 GB |
| UD-Q3_K_S | 12.5 GB |
| UD-Q3_K_M | 12.5 GB |
| UD-Q3_K_XL | 12.9 GB |
| UD-IQ4_XS | 13.4 GB |
| UD-Q4_K_S | 16.4 GB |
| UD-Q4_K_M | 16.9 GB |
| Q8_0 | 26.9 GB |
| BF16 | 50.5 GB |

**Key strengths:**
- Smallest GGUF footprint of the top MoE models (Q3_K_XL at just 12.9GB)
- Best coding benchmark in MoE class (LiveCodeBench 77.1)
- Strong multilingual (MMMLU 86.3 - best in class)
- Very new (April 2026) - latest architecture
- The 31B dense variant is #3 open model on Arena AI (Elo 1452)

**Key weaknesses:**
- Lowest TAU2-Bench (68.2) - weaker for agentic tasks
- Slightly lower GPQA than Qwen (82.3 vs 84.2)
- KV cache concern: at max context, ~22GB just for KV cache on top of model weights

**Note:** Gemma-4 uses Unsloth's "Dynamic" (UD-) quantization format, which may differ slightly from standard GGUF quants in llama.cpp.

---

### 4. Qwen3-Coder-30B-A3B

**Architecture:**
- 30.5B total params, 3.3B active per token
- Same MoE routing as Qwen3 base
- Context: 256K tokens (extendable to 1M via YaRN)
- Optimized for repository-scale code understanding
- License: Apache 2.0

**GGUF Sizes (unsloth):**
| Quant | Size |
|-------|------|
| UD-TQ1_0 | 8.01 GB |
| UD-IQ2_XXS | 10.3 GB |
| Q2_K | 11.3 GB |
| UD-IQ3_XXS | 12.8 GB |
| Q3_K_S | 13.3 GB |
| Q3_K_M | 14.7 GB |
| UD-Q3_K_XL | 13.8 GB |
| IQ4_XS | 16.4 GB |
| Q4_K_S | 17.5 GB |
| Q4_K_M | 18.6 GB |
| Q8_0 | 32.5 GB |
| BF16 | 61.1 GB |

**Strengths:** Purpose-built for coding with 256K context, strong multi-language code support
**Note:** This is based on the older Qwen3 architecture (not Qwen3.5's DeltaNet hybrid attention). Consider it for pure coding workloads where the newer Qwen3.5-35B-A3B's broader improvements aren't needed.

---

### 5. NVIDIA Nemotron-3-Nano-30B-A3B

**Architecture:**
- 31.6B total params, 3.2B active (3.5B with embeddings)
- Hybrid Mamba-2/Transformer MoE (unique architecture)
- 128 experts, 6 routed + 1 shared per token
- 52 total layers: 23 Mamba-2 + 23 MoE + 6 grouped query attention
- Context: 1M tokens native
- License: NVIDIA Open Model License

**Benchmark Scores:**
| Benchmark | Score |
|-----------|-------|
| MMLU-Pro | 78.3 |
| GPQA Diamond | 73.0 |
| SWE-bench Verified | 38.8 |
| LiveCodeBench v6 | 68.3 |
| TAU2-Bench | 49.0 |
| RULER @ 256K | 92.9% |
| RULER @ 1M | 86.3% |

**GGUF Sizes (unsloth):**
| Quant | Size |
|-------|------|
| UD-IQ2_XXS | 18.1 GB |
| Q2_K_L | 18.1 GB |
| Q3_K_S | 18.1 GB |
| Q3_K_M | 20.0 GB |
| Q4_K_S | 22.0 GB |
| Q4_K_M | 24.6 GB |
| Q8_0 | 33.6 GB |
| BF16 | 63.2 GB |

**Key strengths:**
- Best long-context model (RULER 86.3% at 1M tokens)
- 3.3x higher throughput than Qwen3-30B-A3B on identical hardware
- Configurable reasoning with adjustable thinking budgets
- ~25 tok/s streaming from SSD even on a 12GB GPU

**Key weaknesses:**
- DOES NOT FIT 16GB constraint (Q3_K_S already 18.1GB)
- Significantly weaker on coding (SWE-bench 38.8 vs Qwen's 69.2)
- Weakest agentic performance (TAU2 49.0)
- NVIDIA-restricted license

---

### 6. Qwen3-Next-80B-A3B

**Architecture:**
- 80B total params, 3B active per token
- Gated DeltaNet + Gated Attention (hybrid)
- 512 experts, 11 active (10 routed + 1 shared)
- Multi-Token Prediction (MTP) for faster inference
- Pre-trained on 15T tokens

**GGUF Sizes (unsloth):**
| Quant | Size |
|-------|------|
| UD-TQ1_0 | 20.5 GB |
| UD-IQ1_S | 22.9 GB |
| UD-IQ2_XXS | 26.2 GB |
| Q2_K | 29.2 GB |
| Q3_K_S | 34.6 GB |
| Q4_K_S | 45.5 GB |
| BF16 | 159 GB |

**Verdict:** DOES NOT FIT - even Q2_K is 29.2GB. This is a 80B model that needs 64GB+ RAM minimum. Included for reference only.

---

### 7. Llama 4 Scout (17B-16E) (Meta)

**Architecture:**
- 108B total params, 17B model with 16 experts
- MoE following DeepSeek V3 architecture
- 10M token context (abliterated versions)

**GGUF Sizes (mradermacher abliterated):**
| Quant | Size |
|-------|------|
| Q2_K | 39.7 GB |
| Q3_K_S | 46.8 GB |
| Q4_K_S | 61.6 GB |

**Verdict:** DOES NOT FIT - 108B total params is far too large. Q2_K alone is 39.7GB.

---

## GGUF File Size Comparison Table

### Models that fit 14-16GB at Q3-Q4 quantization:

| Model | Q3_K_S | Q3_K_M | UD-IQ4_XS | Q4_K_S | Active Params |
|-------|--------|--------|-----------|--------|---------------|
| **Gemma-4-26B-A4B** | 12.5 GB | 12.5 GB | 13.4 GB | 16.4 GB | 3.8B |
| **GLM-4.7-Flash** | 13.3 GB | 14.6 GB | - | 17.3 GB | ~3B |
| **Qwen3-Coder-30B-A3B** | 13.3 GB | 14.7 GB | 16.4 GB | 17.5 GB | 3.3B |
| **Qwen3.5-35B-A3B** | 15.3 GB | 16.4 GB | 17.5 GB | 20.7 GB | 3B |
| Nemotron-3-Nano-30B-A3B | 18.1 GB | 20.0 GB | 18.2 GB | 22.0 GB | 3.2B |

**Best fits for 14-16GB budget:**
- Gemma-4-26B-A4B at UD-Q4_K_S (16.4GB) or UD-Q3_K_XL (12.9GB)
- GLM-4.7-Flash at Q3_K_M (14.6GB)
- Qwen3-Coder-30B-A3B at Q3_K_M (14.7GB)
- Qwen3.5-35B-A3B at Q3_K_S (15.3GB)

---

## Benchmark Comparison Table

### MoE Models Head-to-Head (Official Benchmarks)

| Benchmark | Qwen3.5-35B-A3B | Gemma-4-26B-A4B | GLM-4.7-Flash | Qwen3-Coder-30B-A3B | Nemotron-3-Nano |
|-----------|-----------------|-----------------|---------------|---------------------|-----------------|
| MMLU-Pro | **85.3** | 82.6 | ~60.0 | - | 78.3 |
| GPQA Diamond | **84.2** | 82.3 | 75.2 | - | 73.0 |
| LiveCodeBench v6 | 74.6 | **77.1** | 64.0 | - | 68.3 |
| SWE-bench Verified | **69.2%** | - | 59.2% | - | 38.8 |
| TAU2-Bench | **81.2%** | 68.2 | 79.5% | - | 49.0 |
| MMMLU | 85.2 | **86.3** | - | - | - |
| MMMU-Pro | **75.1** | 73.8 | - | - | - |
| AIME 2025 | 89.0 | - | **91.6** | - | - |
| RULER @ 1M | - | - | - | - | **86.3%** |

### Your Local Scores vs Official (Qwen3.5-35B-A3B Claude Opus Distilled + Abliterated)

| Benchmark | Your Score (Q3_K_S) | Base Model Official |
|-----------|-------------------|-------------------|
| MMLU | 90% | 85.3 (MMLU-Pro) |
| DROP | 95% / F1:90.6 | - |
| HumanEval | 90% | - |
| MATH | 75% | - |
| GPQA | 60% | 84.2 (Diamond) |

**Note:** Your GPQA score (60%) is significantly lower than the base model's official GPQA Diamond (84.2). This could be due to: (1) different GPQA subset, (2) quantization degradation at Q3_K_S, (3) abliteration impact on specialized reasoning, or (4) different evaluation methodology.

---

## Detailed Model Profiles

### Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled (Base for abliterated version)

**Training Pipeline:**
```
Qwen3.5-35B-A3B (base)
  -> SFT + LoRA (465M trainable / 35.6B total = 1.31%)
  -> Claude 4.6 Opus CoT reasoning distillation
  -> Structured <think> reasoning format
```

**Training Datasets:**
1. nohurry/Opus-4.6-Reasoning-3000x-filtered (comprehensive reasoning trajectories)
2. TeichAI/claude-4.5-opus-high-reasoning-250x (high-intensity structured reasoning)
3. Jackrong/Qwen3.5-reasoning-700x (curated step-by-step solving)

**Training Details:**
- Tool: Unsloth for memory/compute optimization
- Strategy: train_on_responses_only
- Loss: computed only over `<think>` sequences and solutions
- Final training loss: ~0.384
- Context window: 8,192 tokens (reduced from base model's 262K)
- Hardware: 80GB VRAM GPUs required for training

**Key note:** The Opus reasoning distillation reportedly addresses Qwen3.5's tendency toward excessive repetitive reasoning, replacing it with more efficient structured thinking patterns.

---

## Dense Models Worth Considering

For reference, these dense models are alternatives if MoE architecture causes issues:

### Qwen3.5-27B (Dense)
- 28B params, all active
- VRAM (Q4): ~17GB
- Speed: ~34 tok/s (RTX 3090) - 3x slower than 35B-A3B MoE
- Better for simple tasks on tight VRAM budgets
- Q3_K_S likely ~13-14GB

### Gemma-4-31B (Dense)
- 31B params, all active
- #3 open model on Arena AI (Elo 1452)
- AIME 2026: 89.2%, GPQA Diamond: 85.7%
- Q4_K_M: ~19GB - too large for 16GB constraint
- Q3 variants likely 14-16GB range

### Qwen 2.5 Coder 14B
- 14B params dense
- Top-rated local coding model, ~85% HumanEval
- ~8-10GB at Q4 - very comfortable fit
- Fast inference but less capable than MoE models overall

---

## Abliterated / Uncensored Models

### Overview of Abliteration

Abliteration identifies and neutralizes the "refusal direction" in a model's latent space by:
1. Contrasting activations from harmful vs harmless prompts
2. Isolating the refusal direction via SVD
3. Projecting it out of the weight matrices
4. Result: removes compulsion to refuse while leaving core reasoning intact

### Top Abliterated Models (March 2026 Landscape)

**Best for your setup (fitting 14-16GB):**

1. **huihui-ai/Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated** - YOUR CURRENT MODEL
   - Q3_K_S: 15.3GB
   - 136K downloads/month
   - Claude Opus reasoning distillation + abliteration

2. **huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated** (without Opus distillation)
   - Same base architecture, pure abliteration without reasoning fine-tune
   - Likely similar file sizes

3. **Qwen3-30B-A3B-Claude-4.5-Opus-ABLITERATED-UNCENSORED-V2**
   - Older Qwen3 base, MLX 6-bit optimized for Apple Silicon
   - Predecessor to your current model

4. **GLM-4.7-Flash-Uncensored-Heretic-NEO-CODE-Imatrix-MAX**
   - 30B/3B active, coding-optimized
   - ~14GB at Q4

5. **GLM-4.7-Flash-Grande-Heretic-UNCENSORED**
   - 42B total/3B active MoE
   - 16GB VRAM at Q4

**Larger abliterated models (for reference):**
- Dolphin 3.0 (various bases: Llama 3.1 8B, Mistral 24B) - uncensored fine-tune
- Nous Hermes 3 - creative writing/roleplay focused
- Llama 4 Scout abliterated - too large for 32GB (39.7GB at Q2_K)

### Abliteration Methods:
- **huihui-ai method:** Uses remove-refusals-with-transformers (SVD-based, crude but effective)
- **Heretic framework:** Fully automatic censorship removal (github.com/p-e-w/heretic)
- **Standard SFT uncensored:** Trained on unfiltered datasets (Dolphin, etc.)

---

## Inference Frameworks

### For Apple Silicon Mac (ranked by recommendation):

1. **MLX** (Apple's native ML framework)
   - 20-30% faster than llama.cpp across model sizes
   - Gap widens on larger models
   - Built for unified memory architecture
   - mlx-lm for text, mlx-vlm for vision+text
   - Recommended for Qwen3.5 on Apple Silicon

2. **Ollama**
   - Easiest setup (one command)
   - Metal GPU acceleration
   - Good ecosystem, model library
   - `ollama run huihui_ai/qwen3.5-abliterated:35b-Claude`

3. **LM Studio**
   - GUI with model browser
   - Good for experimentation
   - GGUF native support

4. **llama.cpp**
   - Most flexible, bleeding-edge quantization support
   - Note: Qwen3.5-35B-A3B runs ~35% slower than Qwen3-30B-A3B on CUDA due to DeltaNet layer implementation issues (being fixed)
   - On Apple Silicon via Metal: functional but MLX is typically faster

### Performance Estimates (32GB M4 Mac):

| Model | Quant | Estimated tok/s | Notes |
|-------|-------|----------------|-------|
| Qwen3.5-35B-A3B | Q3_K_S | ~30-40 | Via MLX |
| GLM-4.7-Flash | Q3_K_M | ~40-60 | Simpler architecture = faster |
| Gemma-4-26B-A4B | UD-Q4_K_S | ~35-50 | Smallest model, good speed |
| Qwen3-Coder-30B-A3B | Q3_K_S | ~35-50 | Similar to GLM |

**Important MoE memory note:** MoE models are sparse in compute but dense in VRAM - you must load ALL experts into GPU memory even though only ~3B activate per token. Size them by total parameter count, not active count.

---

## Recommendations

### For your 32GB M4 Mac with 14-16GB GGUF budget:

**Best Overall Intelligence: Qwen3.5-35B-A3B (Q3_K_S, 15.3GB)**
- Highest MMLU-Pro (85.3), GPQA (84.2), TAU2 (81.2)
- Your current abliterated+opus-distilled variant is the right choice
- Native multimodal (vision, video, text)
- 262K context

**Best for Coding: GLM-4.7-Flash (Q3_K_M, 14.6GB) or Qwen3-Coder-30B-A3B (Q3_K_S, 13.3GB)**
- GLM: Best SWE-bench in class after Qwen, excellent agentic tool use, fastest inference
- Qwen3-Coder: Purpose-built for code, 256K context with repo-level understanding

**Smallest Footprint: Gemma-4-26B-A4B (UD-Q3_K_XL, 12.9GB)**
- Leaves most headroom for OS + KV cache
- Strong coding (LiveCodeBench 77.1 - best in class)
- Newest model (April 2026)
- Good multilingual support (MMMLU 86.3)

**Best Value for Abliterated Use: Your current model**
- The Claude Opus reasoning distillation adds structured thinking that standard abliteration doesn't have
- Your benchmark scores confirm strong performance at Q3_K_S
- High download count (136K/month) suggests community validation

### Models to Test Next:

1. **GLM-4.7-Flash** at Q3_K_M (14.6GB) - Compare coding/agent performance vs your current model
2. **Gemma-4-26B-A4B** at UD-Q4_K_S (16.4GB) - Test if the newer April 2026 architecture improves on Qwen3.5
3. **GLM-4.7-Flash-Uncensored-Heretic-NEO-CODE-Imatrix-MAX** - Direct coding-optimized uncensored competitor
4. **Qwen3-Coder-30B-A3B** at Q3_K_S (13.3GB) - If coding is primary use case

### What NOT to bother with:
- Nemotron-3-Nano-30B-A3B: Doesn't fit 16GB constraint, weak benchmarks except long-context
- Qwen3-Next-80B-A3B: Too large (29GB at Q2_K minimum)
- Llama 4 Scout: Far too large (108B total)
- Dense 70B models: Don't fit and slower than MoE at similar quality

---

## Sources

### Articles and Guides
- [Best Local LLMs for Apple Silicon Mac - APXML](https://apxml.com/posts/best-local-llms-apple-silicon-mac)
- [Qwen 3.5 vs Gemma 4 Benchmarks by Size - Maniac.ai](https://www.maniac.ai/blog/qwen-3-5-vs-gemma-4-benchmarks-by-size)
- [Gemma 4 Complete Guide - DEV Community](https://dev.to/linnn_charm_2e397112f3b51/gemma-4-complete-guide-architecture-models-and-deployment-in-2026-3m5b)
- [Qwen3.5-35B-A3B vs GLM-4.7-Flash Comparison - Awesome Agents](https://awesomeagents.ai/tools/qwen-3-5-35b-a3b-vs-glm-4-7-flash/)
- [Qwen3.5-35B-A3B vs Nemotron 3 Nano - Awesome Agents](https://awesomeagents.ai/tools/qwen-3-5-35b-a3b-vs-nemotron-3-nano/)
- [Qwen 3.5 Local Guide - InsiderLLM](https://insiderllm.com/guides/qwen35-local-guide-which-model-fits-your-gpu/)
- [20 Uncensored LLMs March 2026 - DecodesFuture](https://www.decodesfuture.com/articles/latest-uncensored-local-llm-releases-march-2026-update)
- [Best Uncensored Local LLMs 2026 - DecodesFuture](https://www.decodesfuture.com/articles/best-uncensored-local-llms-2026-privacy-policy-guide)
- [Best Uncensored AI Models March 2026 - MangoMind](https://www.mangomindbd.com/blog/best-uncensored-ai-models-march-2026/)
- [Running LLMs Locally on macOS 2026 - DEV Community](https://dev.to/bspann/running-llms-locally-on-macos-the-complete-2026-comparison-48fc)
- [Best Local LLM Models 2026 - SitePoint](https://www.sitepoint.com/best-local-llm-models-2026/)
- [Qwen3-Next-80B-A3B Analysis - Kaitchup](https://kaitchup.substack.com/p/qwen3-next-80b-a3b-a-fast-hybrid)
- [Qwen 3.5 Complete Guide - Techie007](https://techie007.substack.com/p/qwen-35-the-complete-guide-benchmarks)
- [GGUF vs GPTQ vs AWQ Compared - LocalAIMaster](https://localaimaster.com/blog/quantization-explained)
- [Abliteration Guide - HuggingFace Blog](https://huggingface.co/blog/mlabonne/abliteration)

### HuggingFace Model Pages
- [huihui-ai/Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated)
- [Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled](https://huggingface.co/Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled)
- [mradermacher/Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated-GGUF](https://huggingface.co/mradermacher/Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated-GGUF)
- [unsloth/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF)
- [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)
- [unsloth/GLM-4.7-Flash-GGUF](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF)
- [unsloth/Nemotron-3-Nano-30B-A3B-GGUF](https://huggingface.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF)
- [unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF)
- [unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF)
- [mradermacher/Llama-4-Scout-17B-16E-Instruct-abliterated-GGUF](https://huggingface.co/mradermacher/Llama-4-Scout-17B-16E-Instruct-abliterated-GGUF)
- [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [Qwen3.5-abliterated Collection - huihui-ai](https://huggingface.co/collections/huihui-ai/qwen35-abliterated)

### Leaderboards
- [Open LLM Leaderboard - HuggingFace](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- [LLM Performance Leaderboard - Artificial Analysis](https://huggingface.co/spaces/ArtificialAnalysis/LLM-Performance-Leaderboard)
- [Gemma 4 26B A4B vs Qwen3.5-35B-A3B - Artificial Analysis](https://artificialanalysis.ai/models/comparisons/gemma-4-26b-a4b-vs-qwen3-5-35b-a3b)
