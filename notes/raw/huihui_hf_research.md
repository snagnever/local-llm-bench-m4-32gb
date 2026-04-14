# HuggingFace Research: Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated

**Research Date:** 2026-04-08

---

## 1. Main Model: huihui-ai/Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated

### Key Metrics
| Metric | Value |
|--------|-------|
| Downloads (last month) | 136,281 |
| Likes/Stars | 36 |
| License | Apache 2.0 |
| Parameters | 36B (35B active architecture, MoE with ~3B active per forward pass) |
| Tensor Types | BF16, F32 |
| Format | Safetensors |
| Task | Image-Text-to-Text |
| Library | Transformers |

### Model Lineage
```
Qwen/Qwen3.5-35B-A3B (official base)
    |
    v
Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled (LoRA SFT on Claude 4.6 Opus reasoning data)
    |
    v
huihui-ai/Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated (abliteration to remove refusals)
```

### Description
This is an **uncensored/abliterated** version of Jackrong's Claude 4.6 Opus reasoning-distilled model. Created using the [remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers) tool -- a crude, proof-of-concept implementation that removes safety refusals from an LLM without TransformerLens.

### Model Card Content
- **No benchmark results** provided
- Safety warnings about sensitive/controversial outputs
- Research/experimental use only
- Available via Ollama (v0.18.0+): `ollama run huihui_ai/qwen3.5-abliterated:35b-Claude`
- Tags: qwen3_5_moe, abliterated, uncensored, Claude, reasoning, chain-of-thought, Dense, conversational

---

## 2. huihui-ai Organization Overview

### Organization Stats
- **Total Models:** 225
- **Total Collections:** 55
- **Total Datasets:** 11

### Most Recently Updated Models (as of 2026-04-08)

| Model | Size | Downloads | Likes | Updated |
|-------|------|-----------|-------|---------|
| Huihui3.5-67B-A3B-abliterated | 68B | 32 | 4 | 1 day ago |
| Huihui3.5-67B-A3B | 68B | 221 | 7 | 2 days ago |
| Huihui-Qwopus3.5-27B-v3-abliterated | 27B | 121 | 7 | 2 days ago |
| Huihui-Qwopus3.5-4B-v3-abliterated | 5B | 209 | 5 | 2 days ago |
| Huihui-Qwopus3.5-9B-v3-abliterated | 10B | 1,130 | 19 | 3 days ago |
| Huihui-OmniCoder-9B-abliterated | - | 64 | 5 | 8 days ago |
| Huihui-MiMo-V2-Flash-BF16-abliterated-GGUF | 309B | 321 | 5 | 8 days ago |
| Huihui-Mistral-Small-4-119B-2603-BF16-abliterated-v2-GGUF | 119B | 1,180 | 2 | 9 days ago |
| Huihui-Mistral-Small-4-119B-2603-BF16-abliterated-GGUF | 119B | 2,140 | 4 | 9 days ago |
| Huihui-Qwen3.5-35B-A3B-abliterated-NVFP4 | - | 16,400 | 5 | 13 days ago |

### Key Observations about huihui-ai
- **Prolific abliterator:** 225 models, virtually all are "abliterated" (uncensored) versions of popular open-source models
- **Covers many architectures:** Qwen, Mistral, Gemma, Llama, DeepSeek, etc.
- **Newer versions exist:** The "Qwopus" line (Huihui-Qwopus3.5-27B-v3-abliterated, etc.) appears to be a newer evolution
- **The 67B-A3B variants** (Huihui3.5-67B-A3B) are brand new (1-2 days old) - this suggests larger Qwen3.5 MoE models are now available
- **Most popular huihui-ai model in this family:** Huihui-Qwen3.5-35B-A3B-abliterated (plain, no Claude distill) with 62,700 downloads and 294 likes -- significantly more liked than the Claude-Opus variant (36 likes)

---

## 3. Competitive Landscape: Most Downloaded/Starred in 27B-35B Range

### Qwen3.5-35B-A3B Family (sorted by downloads)

| Model | Downloads | Likes |
|-------|-----------|-------|
| **Qwen/Qwen3.5-35B-A3B** (official) | 3,340,000 | 1,340 |
| **Qwen/Qwen3.5-35B-A3B-FP8** | 2,060,000 | 135 |
| **unsloth/Qwen3.5-35B-A3B-GGUF** | 1,520,000 | 780 |
| **HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive** | 815,000 | 1,230 |
| **Qwen/Qwen3.5-35B-A3B-GPTQ-Int4** | 660,000 | 68 |
| **cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit** | 529,000 | 35 |
| **lmstudio-community/Qwen3.5-35B-A3B-GGUF** | 399,000 | 22 |
| **bartowski/Qwen_Qwen3.5-35B-A3B-GGUF** | 285,000 | 59 |
| **QuantTrio/Qwen3.5-35B-A3B-AWQ** | 166,000 | 16 |
| **huihui-ai/...-Claude-4.6-Opus-abliterated** | 136,000 | 36 |
| **codgician/...-Claude-4.6-Opus-...-GPTQ-int4** | 121,000 | 5 |
| **huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated** | 62,700 | 294 |

**The huihui Claude-Opus abliterated model ranks ~10th by downloads** in the Qwen3.5-35B-A3B family. The HauhauCS Aggressive uncensored variant is dramatically more popular (815K downloads, 1.23K likes).

### Broader 27B-35B Competitive Landscape

| Model | Downloads | Likes |
|-------|-----------|-------|
| Qwen/Qwen2.5-32B-Instruct | 4,240,000 | 343 |
| Qwen/Qwen3.5-35B-A3B | 3,340,000 | 1,340 |
| Qwen/Qwen3-32B | 3,320,000 | 678 |
| Qwen/Qwen3-VL-32B-Instruct | 1,770,000 | 194 |
| Qwen/Qwen3-30B-A3B | 1,578,000 | 874 |
| Qwen/Qwen2.5-32B-Instruct-AWQ | 1,480,000 | 98 |
| Qwen/Qwen2.5-Coder-32B-Instruct | 1,050,000 | 2,000 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | 988,000 | 1,530 |
| google/gemma-3-27b-it | 845,000 | 1,950 |
| pytorch/gemma-3-27b-it-AWQ-INT4 | 870,000 | 7 |

**Key takeaway:** The official Qwen models dominate downloads. For community/likes engagement, DeepSeek-R1-Distill-Qwen-32B (1.53K likes), Gemma-3-27b-it (1.95K likes), and Qwen2.5-Coder-32B-Instruct (2K likes) lead.

---

## 4. Benchmark Results

### Official Qwen3.5-35B-A3B Benchmarks (from Qwen model card)

#### Language Benchmarks

| Benchmark | Qwen3.5-35B-A3B | GPT-5-mini 2025 | Claude-Sonnet-4.5 |
|-----------|-----------------|-----------------|-------------------|
| MMLU-Pro | 85.3 | 83.7 | N/A |
| C-Eval | 90.2 | 82.2 | N/A |
| IFEval | 91.9 | 93.9 | N/A |
| GPQA Diamond | 84.2 | 82.8 | 80.1 |
| SWE-bench Verified | 69.2 | 72.0 | 62.0 |
| CodeForces | 2028 | 2160 | 2157 |

#### Vision-Language Benchmarks

| Benchmark | Qwen3.5-35B-A3B | GPT-5-mini 2025 | Claude-Sonnet-4.5 |
|-----------|-----------------|-----------------|-------------------|
| MMMU | 81.4 | 79.0 | 79.6 |
| MathVision | 83.9 | 71.9 | 71.1 |
| MMBench EN | 91.5 | 86.8 | 88.3 |
| VideoMME (w/ sub.) | 86.6 | 83.5 | 81.1 |
| OCRBench | 91.0 | 82.1 | 76.6 |

### Derivative Model Benchmarks
- **huihui-ai/...-Claude-4.6-Opus-abliterated:** No benchmarks provided
- **Jackrong/...-Claude-4.6-Opus-Reasoning-Distilled:** No benchmarks provided (qualitative description only)
- **HauhauCS/...-Uncensored-Aggressive:** No benchmarks provided
- **huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated:** No benchmarks provided

**None of the derivative/abliterated models provide any benchmark results.**

---

## 5. MLX Community Version: mlx-community/Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated-4bit

| Metric | Value |
|--------|-------|
| Downloads (last month) | 11,384 |
| Likes/Stars | 9 |
| License | Apache 2.0 |
| Quantization | 4-bit |
| File Size | 20.4 GB |
| Format | Safetensors, MLX |
| Tensor Types | BF16, U32, F32 |
| Conversion Tool | mlx-vlm v0.4.0 |
| Base Model | Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled |

Usage:
```bash
pip install -U mlx-vlm
python -m mlx_vlm.generate --model mlx-community/Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated-4bit \
  --max-tokens 100 --temperature 0.0 --prompt "Describe this image." --image <path>
```

---

## 6. mradermacher i1-GGUF Version: mradermacher/Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated-i1-GGUF

| Metric | Value |
|--------|-------|
| Downloads (last month) | 24,805 |
| Likes/Stars | 5 |
| License | Apache 2.0 |
| Type | imatrix (importance-weighted) GGUF quants |
| Architecture | qwen35moe |

### Available Quantizations

| Quant | Size | Notes |
|-------|------|-------|
| IQ1_S | 7.6 GB | For the desperate |
| IQ1_M | 8.3 GB | Mostly desperate |
| IQ2_XXS | 9.6 GB | |
| IQ2_XS | 10.6 GB | |
| IQ2_S | 10.8 GB | |
| IQ2_M | 11.8 GB | |
| Q2_K_S | 12.3 GB | Very low quality |
| Q2_K | 13.0 GB | |
| IQ3_XXS | 13.7 GB | |
| IQ3_XS | 14.6 GB | |
| IQ3_S | 15.4 GB | Beats Q3_K* |
| IQ3_M | 15.5 GB | |
| Q3_K_S | 15.3 GB | |
| Q3_K_M | 16.9 GB | |
| Q3_K_L | 18.2 GB | |
| IQ4_XS | 18.8 GB | |
| Q4_0 | 19.9 GB | Fast, low quality |
| **Q4_K_S** | **20.0 GB** | **Optimal size/speed/quality** |
| **Q4_K_M** | **21.3 GB** | **Fast, recommended** |
| Q4_1 | 21.9 GB | |
| Q5_K_S | 24.1 GB | |
| Q5_K_M | 24.8 GB | |
| Q6_K | 28.6 GB | Near-lossless |

**Recommendations:** Q4_K_S (20 GB) for best balance, Q4_K_M (21.3 GB) for speed, Q6_K (28.6 GB) for quality.

There is also a **static (non-imatrix) GGUF** version at `mradermacher/Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated-GGUF` with 19,000 downloads and 7 likes.

**Combined GGUF downloads:** ~43,800 (i1: 24,805 + static: 19,000)

---

## 7. Base Model Analysis: Qwen3.5-35B-A3B and Qwen3-30B-A3B

### Qwen3.5-35B-A3B (newer, multimodal)

| Metric | Value |
|--------|-------|
| Downloads (last month) | 3,344,183 |
| Likes/Stars | 1,340 |
| Release | February 2026 |
| Architecture | Hybrid: Gated DeltaNet + Gated Attention + MoE |
| Total Parameters | 35B (3B active per token) |
| Experts | 256 total, 8 routed + 1 shared |
| Layers | 40 (pattern: 10 x (3x DeltaNet-MoE + 1x Attention-MoE)) |
| Native Context | 262,144 tokens (extendable to 1,010,000) |
| Modalities | Text + Image + Video (native multimodal) |
| Languages | 201 languages |
| Vocabulary | 248,320 tokens |

This is the **successor** to Qwen3-30B-A3B, featuring:
- Native multimodal (vision-language) capabilities
- Hybrid attention architecture (Gated DeltaNet linear attention for efficiency)
- Larger context window (262K vs 32K native)
- Multi-token prediction support

### Qwen3-30B-A3B (older, text-only)

| Metric | Value |
|--------|-------|
| Downloads (last month) | 1,578,213 |
| Likes/Stars | 874 |
| Release | ~May 2025 |
| Architecture | Standard Transformer + MoE |
| Total Parameters | 30.5B (3.3B active per token) |
| Experts | 128 total, 8 activated |
| Layers | 48 |
| Native Context | 32,768 tokens (extendable to 131,072 via YaRN) |
| Modalities | Text only |
| Languages | 100+ languages |

### Comparison

| Feature | Qwen3-30B-A3B | Qwen3.5-35B-A3B |
|---------|---------------|-----------------|
| Generation | 3rd gen | 3.5 gen |
| Vision | No | Yes (native) |
| Active params | 3.3B | ~3B |
| Total experts | 128 | 256 |
| Context (native) | 32K | 262K |
| Context (extended) | 131K | 1M |
| Attention | Standard | Hybrid DeltaNet |
| Downloads | 1.58M | 3.34M |

**Qwen3.5-35B-A3B is the clear successor** with native multimodal, 8x longer context, and more downloads.

---

## 8. Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled (Intermediate Model)

| Metric | Value |
|--------|-------|
| Downloads (last month) | 30,507 |
| Likes/Stars | 111 |
| License | Apache 2.0 |
| Context Window | 8,192 tokens |
| Training Method | LoRA-based SFT |
| Trainable Parameters | 465M / 35.5B total (~1.31%) |
| Final Training Loss | ~0.384 |

### Training Details
- Trained on Claude 4.6 Opus reasoning trajectories using 3 datasets:
  - nohurry/Opus-4.6-Reasoning-3000x-filtered
  - TeichAI/claude-4.5-opus-high-reasoning-250x
  - Jackrong/Qwen3.5-reasoning-700x
- Uses `train_on_responses_only` -- loss only on `<think>` sequences and solutions
- Format: `<think>{reasoning}</think>\n{answer}`
- Built with Unsloth for memory optimization
- Self-funded training

### Key Limitation
- **Context reduced to 8,192 tokens** (from 262K in the base model) -- a significant regression likely due to LoRA training constraints

---

## 9. Summary & Key Findings

### Model Popularity Ranking (Qwen3.5-35B-A3B uncensored variants)

| Rank | Model | Downloads | Likes |
|------|-------|-----------|-------|
| 1 | HauhauCS/...-Uncensored-Aggressive | 815,000 | 1,230 |
| 2 | huihui-ai/...-Claude-4.6-Opus-abliterated | 136,000 | 36 |
| 3 | huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated | 62,700 | 294 |

### Strengths of the Huihui Claude-Opus model
- Combines Claude 4.6 Opus reasoning patterns with uncensoring
- Good download numbers (136K) indicating real usage
- Multiple quantization formats available (MLX 4-bit, GGUF imatrix, static GGUF)
- Available via Ollama for easy local deployment

### Weaknesses / Concerns
1. **No benchmarks** -- neither the Jackrong distilled base nor the huihui abliterated version provide any quantitative evaluation
2. **Reduced context** -- the Jackrong LoRA training reduced context to 8,192 tokens from the base model's 262K
3. **Much less popular than competitors** -- HauhauCS Aggressive has 6x more downloads and 34x more likes
4. **The plain abliterated version (no Claude distill) is more liked** -- 294 likes vs 36, suggesting the community may prefer the pure Qwen3.5 abliteration over the Claude-distilled variant
5. **Abliteration is described as "crude, proof-of-concept"** by the creator themselves
6. **No vision capability verification** -- tagged as Image-Text-to-Text but unclear if multimodal still works after LoRA + abliteration pipeline

### Newer/Alternative Options from huihui-ai
- **Huihui-Qwopus3.5-27B-v3-abliterated** (2 days old) -- appears to be a newer "v3" iteration
- **Huihui-Qwopus3.5-9B-v3-abliterated** (3 days old) -- smaller, already has 1.13K downloads
- **Huihui3.5-67B-A3B-abliterated** (1 day old) -- larger 67B variant for those with more VRAM

### Recommended Quantizations for Local Use
- **Apple Silicon (MLX):** mlx-community/...-4bit at 20.4 GB
- **llama.cpp/Ollama:** mradermacher/...-i1-GGUF Q4_K_M at 21.3 GB (recommended) or Q4_K_S at 20.0 GB (best balance)
- **24GB VRAM GPU:** Q4_K_M fits comfortably; Q5_K_M at 24.8 GB is tight but possible
