# Reddit r/LocalLLaMA Research: Best Models for 32GB Apple Silicon Macs

**Research Date:** 2026-04-08
**Primary Source Community:** r/LocalLLaMA (266,500+ members as of 2026)
**Note:** Reddit blocks Anthropic's web crawler, so direct Reddit page fetches were unavailable. Data was gathered from search result summaries, community aggregator sites, blog posts citing Reddit discussions, and technical review sites that reference r/LocalLLaMA threads.

---

## Table of Contents

1. [Top Model Recommendations for 32GB Mac](#1-top-model-recommendations-for-32gb-mac)
2. [Qwen Model Family - The Community Favorite](#2-qwen-model-family---the-community-favorite)
3. [Mixture of Experts (MoE) Models](#3-mixture-of-experts-moe-models)
4. [Coding-Specific Models](#4-coding-specific-models)
5. [Abliterated/Uncensored Models](#5-abliterateduncensored-models)
6. [Claude Reasoning Distillation Models](#6-claude-reasoning-distillation-models)
7. [Apple Silicon Performance Benchmarks](#7-apple-silicon-performance-benchmarks)
8. [Inference Frameworks Comparison (MLX vs Ollama vs llama.cpp)](#8-inference-frameworks-comparison)
9. [GGUF Quantization Guide](#9-gguf-quantization-guide)
10. [Hardware-Specific Recommendations by Memory Tier](#10-hardware-specific-recommendations-by-memory-tier)
11. [Sources](#11-sources)

---

## 1. Top Model Recommendations for 32GB Mac

### Community Consensus (r/LocalLLaMA, early 2026)

The r/LocalLLaMA community has shifted decisively toward **Qwen models** as the default recommendation, marking "the first time they've resoundingly dethroned Llama" (per community aggregation sites citing subreddit sentiment).

**For 32GB unified memory, the top recommendations are:**

| Model | Quantization | VRAM Usage | Use Case | Speed (est.) |
|-------|-------------|-----------|----------|-------------|
| **Qwen 3 32B** | Q4_K_M | ~20GB | General / Reasoning | 15-22 tok/s |
| **Qwen3-Coder-32B** | Q4_K_M | ~20GB + 4-6GB KV cache | Coding | 15-20 tok/s |
| **Qwen3.5-35B-A3B** (MoE) | Q4_K_M | ~20-22GB | Speed-optimized general use | Much faster (MoE) |
| **Qwen3.5-27B** | Q4_K_M | ~16-18GB | Quality-focused tasks | 15-25 tok/s |
| **Llama 3.3 70B** | Q4_K_M | ~40GB (tight on 32GB) | Best quality when it fits | 8-15 tok/s |

**Key insight from the community:** "The model file should be no more than 60-70% of your total memory" to avoid performance degradation and crashes from memory pressure. For a 32GB Mac, this means targeting models that are ~19-22GB on disk.

**Critical note on 70B models:** While some sources mention running Llama 3.3 70B quantized on 32GB Macs, this is at the absolute limit. The community reports Mac users can sometimes run 70B models that would require 40GB+ dedicated VRAM on a PC, thanks to Apple's unified memory architecture, but expect significant slowdowns and memory pressure on a 32GB system.

### The Question Asked Weekly on r/LocalLLaMA

"What is the best local LLM to run?" The answer in 2026 is both simpler and more complex than ever -- simpler because the tools have matured dramatically, and more complex because the model landscape has exploded with viable options. The consensus: the "best" model depends on your specific use case, hardware, and priorities.

---

## 2. Qwen Model Family - The Community Favorite

### Qwen Has Taken Over r/LocalLLaMA

Per community analysis: "Qwen has taken over the LocalLLaMA subreddit, with the default choice now being Qwen, marking the first time they've resoundingly dethroned Llama."

Simon Willison (prominent developer and blogger) specifically endorses **Qwen3-8B** as "surprisingly capable" and notes that Chinese AI labs have "positively smoked Western alternatives throughout 2025."

### Key Qwen Models for Local Use

**Qwen 3 32B (Q4_K_M)**
- The standout for 32GB+ Macs
- Described as "expert-level quality" that "makes Mac local AI worthwhile"
- Near-frontier performance while fitting on mid-range hardware
- Problem: reasoning process can be very long (thinking times up to 47 minutes on complex coding questions)

**Qwen 3 14B (Q4_K_M)**
- Sweet spot for 24GB Macs
- "Significant improvement from 8B to 14B with better reasoning and fewer hallucinations"

**Qwen 3 8B (Q4_K_M)**
- Best all-rounder for 16GB Macs
- Simon Willison runs MLX 4-bit version consuming just 4-5GB RAM
- Strong instruction following, good at code, solid reasoning

**Qwen 3 72B**
- Scores 83.1 MMLU and 84.2 HumanEval
- Requires 48GB+ for comfortable operation
- Not practical on 32GB

### Qwen 3.5 Series (Latest)

**Qwen3.5-27B (Dense)**
- 27B parameters, all active per token
- ~16-18GB at Q4 quantization
- Superior for complex coding, creative writing, advanced reasoning
- M5 Max benchmarks: 23.6 tok/s at 4K context (6-bit quant)

**Qwen3.5-35B-A3B (MoE)**
- 35B total parameters, ~3B active per token
- ~20-22GB at Q4 quantization
- Up to 5x higher throughput than dense 27B
- Best for speed-sensitive applications, chatbots, general knowledge
- "The release of the Qwen 3.5 series has sparked a heated debate within the local LLM community, specifically on subreddits like r/LocalLLaMA"

**Qwen3.5-122B-A10B (Flagship MoE)**
- 122B total, ~10B active per token
- Requires 64GB+ unified memory
- M5 Max (128GB): 65.9 tok/s generation at 4K context

---

## 3. Mixture of Experts (MoE) Models

### MoE Dominance in 2026

"The overwhelming theme of 2025-2026 is the dominance of Mixture-of-Experts architectures, with seven of the ten top models using MoE (or hybrid MoE). By activating only a fraction of total parameters per token, MoE models achieve frontier-class quality at a fraction of the inference cost of dense models."

### Key MoE Models Discussed on r/LocalLLaMA

| Model | Total Params | Active Params | Notes |
|-------|-------------|--------------|-------|
| Qwen3.5-35B-A3B | 35B | ~3B | Best MoE for 32GB Mac |
| Qwen3-30B-A3B | 30B | ~3B | Predecessor, "outcompetes QwQ-32B with 10x fewer active params" |
| Qwen3.5-122B-A10B | 122B | ~10B | Requires 64GB+ |
| Qwen3.5-397B-A17B (Flagship) | 397B | ~17B | Requires 128GB+ |
| Llama 4 Scout | 109B | ~17B | Meta's MoE entry |
| Mixtral 8x7B | ~47B | ~13B | Older but proven |

### MoE Trade-offs (Community Consensus)

**Choose Dense (e.g., Qwen3.5-27B) for:**
- Complex coding tasks (fewer syntax errors, better logic)
- Creative writing and roleplay (character consistency)
- Advanced reasoning requiring deep parameter activation

**Choose MoE (e.g., Qwen3.5-35B-A3B) for:**
- Fast chatbots and real-time applications
- General knowledge queries
- Background tasks (logs, data extraction)
- Speed: 60-100+ tok/s vs 15-25 tok/s on RTX 3090

---

## 4. Coding-Specific Models

### Top Coding Models for Local Use (2026)

**#1: Qwen3-Coder-Next (MoE, 80B total, 3B active)**
- Ranked #1 best free local AI coding model in 2026
- "Performance comparable to models 10-20x larger"
- "If you want something that runs correctly the first time, Qwen3 Coder Next is the better bet"
- On M5 Max (128GB): 79.3 tok/s generation at 4K context (8-bit quant)

**#2: Qwen3-Coder-32B**
- Community consensus pick for coding with "extremely stable tool calling"
- ~20GB at Q4 plus 4-6GB for KV cache
- Requires 32GB+ hardware

**#3: DeepSeek-Coder-V2 (16B/33B)**
- "Beats GPT-4 on several coding benchmarks"
- 8GB+ for 16B; 16GB+ for 33B
- "Quietly built the best open-source coding model available"

**#4: Qwen2.5-Coder-14B**
- Top-rated local model for coding, scoring ~85% on HumanEval
- Compared to 68% for Llama 3.3 8B
- Good fit for 32GB systems

**#5: Qwen3.5 4B (for coding)**
- "Stands out as the optimal choice for most coding tasks"
- "Offering stability without performance drops and operating faster than the 9B variant"

### Simon Willison's Coding Workflow
- Uses Qwen3-8B via `llm-mlx` plugin
- MLX 4-bit quantized, consuming just 4-5GB RAM
- Handles SQL query generation, code writing, web application analysis
- "Extraordinary that a few GBs of floating point numbers can usefully achieve these various tasks"

---

## 5. Abliterated/Uncensored Models

### What is Abliteration?

The term was coined by Reddit user /u/FailSpai in early 2024 -- "ablate refusal features to the point of obliteration." The technique identifies "refusal vectors" in model weights and mathematically removes them, preserving the original model's reasoning capabilities while removing content filtering.

**Key difference from fine-tuning:** "Fine-tuning teaches a model what to say, while abliteration tells it what not to say (specifically, not to refuse). An abliterated model might be closer to the original base model in terms of its core knowledge & personality, just without the safety brakes."

### huihui-ai Models (Key Creator)

huihui-ai is a prominent creator of abliterated models, available on HuggingFace and Ollama:

- **huihui-ai/QwQ-32B-Abliterated-GGUF** - Quantized uncensored version optimized for local deployment on consumer hardware
- **huihui-ai/Qwen2.5-32B-Instruct-abliterated** - Abliterated version of Qwen2.5-32B
- **huihui-ai/Qwen3-8B-Abliterated-V2** - Uses advanced abliteration techniques on Qwen3-8B
- **huihui-ai/Qwen3-4B-Abliterated-V2** - Smaller variant for constrained hardware
- **huihui-ai/GLM-4.6-Abliterated** - Text-based LLM with reduced safety filters

### Top Uncensored Models (2026 Ranked)

**Efficiency Tier (7B-12B):**
- Qwen 2.5 7B Uncensored - "Current efficiency king for coding and logic" (6-8GB VRAM)
- DeepSeek R1 Distill 7B abliterated - Reasoning model (8GB VRAM)

**Mid-Range Tier (20B-30B):**
- GPT-OSS 20B Heretic - "Community favorite for creative writing" (14-16GB VRAM)
- GLM-4.7 Flash Uncensored - "Blazing fast throughput" (16GB VRAM)

**Enterprise Tier (70B+):**
- Llama 4 70B Uncensored (Abliterated) - "Gold standard for local intelligence" (40GB+ VRAM)
- Loki 70B Heretic V2.0 - Specialized for narrative work (48GB VRAM)

### Known Limitations of Abliteration

The Reddit community frequently reports:
- **Intelligence loss** -- reduced reasoning capability, increased hallucinations, degraded instruction-following
- **Context degradation** -- "Abliterated models losing their mind after 7-10 messages"
- These effects are inherent to the technique and vary by base model quality

---

## 6. Claude Reasoning Distillation Models

### Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled

The most prominent distillation project, by HuggingFace user Jackrong:

**How it was made:**
- Supervised fine-tuning with LoRA (rank 64) on ~3,950 reasoning trace samples from Claude 4.6 Opus
- v2 expanded to ~14,000 samples
- Outputs structured reasoning in `<think>` tags before final answers
- Mimics Claude's extended thinking behavior

**Performance:**
- 57,000+ downloads across variants within days of release
- Agentic coding: "Improved autonomy and stability -- capable of running continuously for over 9 minutes autonomously with zero human intervention" on RTX 3090
- Tool-calling: "Only the Claude-distilled 27B version demonstrates stable performance" among Qwen3.5 variants tested
- Standard benchmarks (MMLU, GPQA): "Mixed or absent" -- no clear improvement over base model

**Hardware Requirements:**
- Full BF16: 55.6 GB (multi-GPU)
- 4-bit GGUF: ~16 GB (fits on 32GB Mac)
- MLX 4-bit (Apple Silicon): ~14 GB
- On AMD Strix Halo (120 GB/s bandwidth): 10.3 tok/s at Q4_K_M

**Critical Limitations:**
- Context window reduced from 262K to 8K tokens (severe constraint)
- Multimodal capabilities removed
- Cannot recognize when its own output becomes incoherent
- "Captures the format of reasoning -- structured thinking patterns -- but lacks the implicit self-monitoring native to frontier models"
- Degenerated into "semantic gibberish" on wolf-goat-cabbage puzzle, filling 8K context window

**Community Reception:**
- Downloads: Very high (57K+)
- Critical voices: Training methodology (SFT alone) may be insufficient without reinforcement learning
- Best suited for: Local agent workflows on consumer GPUs
- Not suited for: Long-document processing or production use

### Other Distilled Models

- **Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2** - Smaller variant by same creator
- **GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-GGUF** - By TeichAI

### Controversy: Illicit Distillation

Anthropic reported that AI labs including DeepSeek, Moonshot AI, and MiniMax "created thousands of fraudulent accounts and generated millions of interactions with Claude" -- over 16 million exchanges through ~24,000 fraudulent accounts, violating terms of service.

---

## 7. Apple Silicon Performance Benchmarks

### M5 Max Benchmarks (128GB, 614 GB/s bandwidth, MLX framework)

| Model | Quant | Size | 4K ctx gen | 16K ctx gen | 32K ctx gen |
|-------|-------|------|-----------|------------|------------|
| Qwen3.5-122B-A10B | 4-bit | 69.6GB | 65.9 t/s | 60.6 t/s | 54.9 t/s |
| Qwen3-Coder-Next | 8-bit | 84.7GB | 79.3 t/s | 74.3 t/s | 68.6 t/s |
| GPT-OSS-120B | 8-bit | 64GB | 87.9 t/s | 76.0 t/s | 64.5 t/s |
| Qwen3.5-27B | 6-bit | 26GB | 23.6 t/s | 20.3 t/s | 14.9 t/s |

### M4 Pro Benchmarks (64GB, 273 GB/s bandwidth)

| Framework | Single-User | 32-User Throughput | TTFT |
|-----------|------------|-------------------|------|
| vllm-mlx | 42 t/s | 1,150 t/s | ~120ms |
| Ollama v0.8+ | 58 t/s | 720 t/s | ~45ms |
| llama.cpp (Metal) | 52 t/s | 890 t/s | ~85ms |

(Tested with DeepSeek V3 Q4_K_M)

### Estimated Performance for 32GB M4 Mac (120 GB/s bandwidth)

Based on bandwidth scaling from benchmarks:
- **Qwen 3 32B Q4_K_M:** ~15-22 tok/s generation
- **Qwen3.5-35B-A3B Q4:** Significantly faster due to MoE (only 3B active)
- **Llama 3.1 8B Q4:** ~55 tok/s (community-reported Ollama numbers)
- **Qwen 3 8B Q4:** ~75-85 tok/s via Ollama, ~95-110 tok/s via MLX

### Key Performance Insight

"Memory bandwidth is what determines your speed -- an M3 Max generates tokens faster than an M4 Pro because it has more bandwidth, even though the M4 Pro is newer."

- M4 base: 120 GB/s
- M4 Pro: 273 GB/s
- M4 Max: 546 GB/s
- M5 Max: 614 GB/s

### M5 Max vs RTX Pro 6000 Comparison

The RTX Pro 6000 delivers 2.5-3x higher generation throughput. However, the M5 Max costs ~$5,099 (MacBook Pro 128GB) vs $8,800+ for RTX Pro 6000 workstation GPU alone.

---

## 8. Inference Frameworks Comparison

### MLX vs Ollama vs llama.cpp on Apple Silicon (2026)

#### Performance Summary

**MLX (Apple's Native Framework):**
- 20-30% faster than llama.cpp for raw generation speed
- Leads by 20-87% for models under 14B parameters
- Gap closes to near-zero at 27B+ parameters (memory bandwidth becomes bottleneck)
- Best tok/s numbers in benchmarks

**Ollama (Most Popular):**
- v0.19 replaced Metal backend with MLX on Apple Silicon -- massive speed improvement
- 1.6-2x faster prompt processing and token generation with int4 quantization
- Lowest time-to-first-token (~45ms)
- "Fastest response, zero-config deployment"
- OpenAI-compatible API

**llama.cpp (Most Compatible):**
- Most ecosystem support (LM Studio, Jan, Ollama all built on it)
- Best for oversized model support and precise quantization control
- "Best control over static resources" via direct Metal API calls

#### Critical Finding: Synthetic vs Real-World Performance

A detailed analysis revealed a major discrepancy:
- MLX reported generation speed: 57 tok/s
- MLX effective throughput at 8.5K context: **3 tok/s**
- "At 8.5K tokens of context, prefill accounts for 94% of MLX's total time"

**The UI displays only generation speed, obscuring prefill dominance in real usage.**

In document classification tasks, GGUF/llama.cpp actually outperformed MLX:
- MLX range: 7.3-10.6 seconds per document
- GGUF range: 4.7-8.3 seconds per document

#### Framework Selection Guide

| Use Case | Recommended Framework |
|----------|----------------------|
| Quick prototyping, chat | Ollama (zero-config) |
| Python-first development, small models (<14B) | MLX |
| Short output, long input tasks | GGUF/llama.cpp |
| Streaming chat with long outputs | MLX |
| Agent workflows, tool calling | Ollama |
| Maximum compatibility | llama.cpp |
| High-throughput production on Mac | vllm-mlx |

#### Ollama 0.19 MLX Update (Significant)

Test with Qwen3.5-35B-A3B on M5 Max:
- Prefill: 1,851 tok/s (up from 1,100 with Metal) -- 1.7x improvement
- Decode: 134 tok/s (up from 58 with Metal) -- 2.3x improvement

For 32GB Macs: "Functional (20GB model + 12GB cache)" but tight. 48GB+ recommended for comfortable operation.

---

## 9. GGUF Quantization Guide

### Recommended Quantization Formats

| Format | Size Reduction | Quality Retention | Best For |
|--------|---------------|-------------------|----------|
| Q4_K_S | ~75% smaller | ~90% | 8GB RAM laptops, minimal footprint |
| **Q4_K_M** | ~75% smaller | **~92-98%** | **Community standard, Apple Silicon** |
| Q5_K_M | ~65% smaller | ~95% | High-end consumer GPUs (RTX 4070+) |
| Q6_K | ~50% smaller | ~97% | When VRAM permits maximum quality |
| Q8_0 | ~50% smaller | ~99% | Superior output when VRAM allows |

**Community consensus:** Q4_K_M is the most downloaded GGUF format because it has a low memory footprint and is "often as accurate as the original model." At 4-bit (Q4_K_M), quantized models retain 92-98% of original quality.

### Apple Silicon Specific Advice

- Q4_K_M recommended for Apple Silicon -- "Leverages Metal backend with CPU fallback support"
- For 32GB Mac: Target Q4_K_M for 30B-35B models
- The 60-70% memory rule: model should use no more than 60-70% of total memory

### Key GGUF Curators

- **Unsloth** -- Released 25 GGUF versions of Qwen3 8B and 26 versions for DeepSeek-V3.1-Terminus
- **Bartowski** -- Prominent GGUF creator (e.g., bartowski/huihui-ai_QwQ-32B-abliterated-GGUF)
- 5,000+ GGUF models updated daily, compatible with llama.cpp, LM Studio, Ollama, KoboldCpp

---

## 10. Hardware-Specific Recommendations by Memory Tier

### Quick Reference Table

| Memory | Best Model | Quantization | Speed | Notes |
|--------|-----------|-------------|-------|-------|
| 8GB | Llama 3.2 3B or Phi-4 Mini | Q4_K_M | 25-40 t/s | Basic tasks only |
| 16GB | Qwen 3 8B | Q4_K_M | 20-40 t/s | Best all-rounder |
| 24GB | Qwen 3 14B | Q4_K_M | 15-30 t/s | Significant quality jump |
| **32GB** | **Qwen 3 32B** | **Q4_K_M** | **15-22 t/s** | **"Makes Mac local AI worthwhile"** |
| 48GB | Qwen 3 32B (Q6/Q8) | Q6_K/Q8_0 | 15-22 t/s | Higher quality quant |
| 64GB | Llama 3.3 70B | Q4_K_M | 8-15 t/s | "Matches GPT-3.5 quality" |
| 128GB | Qwen 235B-A22B | Q4_K_M | 5-10 t/s | Frontier-class local |

### 32GB Mac Specific Recommendations

**Best Overall:** Qwen 3 32B Q4_K_M (~20GB model)
**Best Coding:** Qwen3-Coder-32B Q4_K_M (~20GB + KV cache)
**Best Speed:** Qwen3.5-35B-A3B Q4_K_M (MoE, only 3B active params)
**Best Reasoning Distilled:** Qwen3.5-27B-Claude-Opus-Distilled Q4 (~14GB MLX 4-bit)
**Best Uncensored:** huihui-ai/QwQ-32B-Abliterated Q4_K_M
**Best for Beginners:** Qwen 3 8B Q4_K_M via LM Studio or Ollama

### Critical Apple Silicon Considerations

1. **Unified memory is your ceiling** -- "If you buy too little memory at checkout, you are not adding more later. You are replacing the box."
2. **Bandwidth > chip generation** -- M3 Max (400 GB/s) generates tokens faster than M4 Pro (273 GB/s)
3. **The 60-70% rule** -- Keep model size under 60-70% of total memory
4. **Framework matters** -- Ollama 0.19+ with MLX backend gives best Apple Silicon experience
5. **MoE models are a game-changer** -- Qwen3.5-35B-A3B gives 35B knowledge with 3B inference cost

---

## 11. Sources

### Search-Derived Sources (Community Aggregations & Guides)
- [Best Local LLM Models 2026 | SitePoint](https://www.sitepoint.com/best-local-llm-models-2026/)
- [The Best Mac mini for local LLMs in 2026: M4 vs M4 Pro](https://www.popularai.org/p/the-best-mac-mini-for-local-llms)
- [Run AI Locally: Best LLMs for 8GB, 16GB, 32GB Memory | Micro Center](https://www.microcenter.com/site/mc-news/article/best-local-llms-8gb-16gb-32gb-memory-guide.aspx)
- [The Best Local LLMs To Run On Every Mac (Apple Silicon)](https://apxml.com/posts/best-local-llm-apple-silicon-mac)
- [Best Local LLMs for Mac in 2026 | InsiderLLM](https://insiderllm.com/guides/best-local-llms-mac-2026/)
- [Meta Llama Reddit: What r/LocalLLaMA Really Thinks (2026)](https://www.aitooldiscovery.com/guides/llama-reddit)
- [Local LLM Reddit: What the Privacy-First AI Community Thinks (2026)](https://www.aitooldiscovery.com/guides/local-llm-reddit)
- [Best Local LLM to Run 2026: Complete Hardware + Model Guide](https://neural-digest.com/best-local-llm-to-run-2026-complete-guide/)
- [Best Local LLMs for 24GB VRAM 2026 | LocalLLM.in](https://localllm.in/blog/best-local-llms-24gb-vram)
- [Best Mac for AI in 2026: Run Local LLMs on a Budget](https://www.refurb.me/blog/best-mac-for-ai)

### Model-Specific Sources
- [Qwen3-30B-A3B | Hugging Face](https://huggingface.co/Qwen/Qwen3-30B-A3B)
- [Qwen3.5-35B-A3B | Hugging Face](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [Qwen 3.5 27B vs 35B-A3B Benchmarks](https://vertu.com/ai-tools/qwen-3-5-27b-vs-qwen-3-5-35b-a3b-which-local-llm-reigns-supreme/)
- [Qwen3-Coder-Next: Complete 2026 Guide | DEV Community](https://dev.to/sienna/qwen3-coder-next-the-complete-2026-guide-to-running-powerful-ai-coding-agents-locally-1k95)
- [Qwen3-Coder-Next vs 4 other local AI coding models | XDA](https://www.xda-developers.com/tested-qwen3-coder-next-four-local-ai-coding-models-gap-embarassing/)
- [Simon Willison on Qwen](https://simonwillison.net/tags/qwen/)
- [Qwen2.5-Coder-32B runs on my Mac | Simon Willison](https://simonw.substack.com/p/qwen25-coder-32b-is-an-llm-that-can)

### Abliterated/Uncensored Model Sources
- [huihui_ai on Ollama](https://ollama.com/huihui_ai)
- [bartowski/huihui-ai_QwQ-32B-abliterated-GGUF | Hugging Face](https://huggingface.co/bartowski/huihui-ai_QwQ-32B-abliterated-GGUF)
- [Top 30 Uncensored Open-Source AI Models (2026)](https://www.decodesfuture.com/articles/top-uncensored-open-source-ai-models-2026-list)
- [Top 10 LLMs with No Restrictions in 2026](https://apidog.com/blog/llms-no-restrictions/)
- [Best Uncensored LLM on Ollama](https://www.arsturn.com/blog/finding-the-best-uncensored-llm-on-ollama-a-deep-dive-guide)

### Claude Distillation Sources
- [Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled | Hugging Face](https://huggingface.co/Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled)
- [Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2 | Hugging Face](https://huggingface.co/Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2)
- [Qwen3.5-27B Claude Opus Reasoning Distilled Review](https://renovateqr.com/blog/qwen35-27b-claude-opus-reasoning-distilled-review)
- [Distilled Reasoning on Strix Halo | TinyComputers.io](https://tinycomputers.io/posts/distilled-reasoning-on-strix-halo-qwen35-claude-thinking.html)
- [Illicit Distillation on LLMs: What happened with Claude | Medium](https://ashutoshkumars1ngh.medium.com/illicit-distillation-what-it-is-how-it-works-and-what-happened-with-claude-a74909930111)
- [Claude Opus Reasoning Distilled Into Open 27B Model | Awesome Agents](https://awesomeagents.ai/news/qwen-27b-claude-opus-reasoning-distilled/)

### Benchmark & Framework Sources
- [Apple M5 Max Local LLM Benchmarks vs RTX Pro 6000 and RTX 5090](https://www.hardware-corner.net/m5-max-local-llm-benchmarks-20261233/)
- [SiliconBench -- Apple Silicon LLM Benchmarks](https://siliconbench.radicchio.page/)
- [Silicon Score Bench](https://siliconscore.com/bench/)
- [2026 Mac Inference Framework Selection: vllm-mlx vs Ollama vs llama.cpp](https://macgpu.com/en/blog/2026-mac-inference-framework-vllm-mlx-ollama-llamacpp-benchmark.html)
- [Ollama 0.19 MLX Review: 2x Faster on Apple Silicon](https://andrew.ooo/posts/ollama-mlx-apple-silicon-review/)
- [57 tok/s on Screen, 3 tok/s in Practice: MLX vs llama.cpp](https://famstack.dev/guides/mlx-vs-gguf-apple-silicon/)
- [MLX vs llama.cpp on Apple Silicon | Groundy](https://groundy.com/articles/mlx-vs-llamacpp-on-apple-silicon-which-runtime-to-use-for-local-llm-inference/)
- [GGUF vs GPTQ vs AWQ Compared 2026 | Local AI Master](https://localaimaster.com/blog/quantization-explained)
- [The Ultimate Local LLM Guide: M4 Mac or RTX 50-Series](https://www.nullzen.dev/blog/local-llm-guide-m4-rtx50/)
- [Performance of llama.cpp on Apple Silicon M-series | GitHub](https://github.com/ggml-org/llama.cpp/discussions/4167)

### r/LocalLLaMA Community References
- [r/LocalLLaMA Subreddit Stats & Analysis](https://gummysearch.com/r/LocalLLaMA/)
- [r/LocalLLaMA Year in Review | GitHub Gist](https://gist.github.com/av/5e4820a48210600a458deee0f3385d4f)
- [Ask HN: Best LLM for consumer grade hardware](https://news.ycombinator.com/item?id=44134896)
- [Ask HN: Anyone Using a Mac Studio for Local AI/LLM?](https://news.ycombinator.com/item?id=46907001)
