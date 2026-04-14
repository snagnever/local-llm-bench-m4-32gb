# Inference Engines for LLMs on Apple Silicon M4 (32GB, 120 GB/s)

**Research Date:** April 8, 2026
**Hardware Target:** Apple M4 base chip, 32GB unified memory, 120 GB/s memory bandwidth

---

## Table of Contents

1. [Executive Summary & Recommendation](#executive-summary)
2. [Critical Hardware Context](#hardware-context)
3. [LM Studio](#1-lm-studio)
4. [MLX / mlx-lm](#2-mlx--mlx-lm)
5. [Ollama](#3-ollama)
6. [llama.cpp (raw)](#4-llamacpp-raw)
7. [vLLM-MLX & vllm-metal](#5-vllm-mlx--vllm-metal)
8. [Rapid-MLX](#6-rapid-mlx)
9. [Other Engines](#7-other-engines)
10. [Quantization Format Comparison](#8-quantization-format-comparison)
11. [Flash Attention & Speculative Decoding](#9-flash-attention--speculative-decoding)
12. [MoE-Specific Analysis](#10-moe-specific-analysis)
13. [Benchmark Summary Table](#11-benchmark-summary-table)
14. [Final Verdict for Your Use Case](#12-final-verdict)

---

## Executive Summary

**The landscape HAS changed significantly since you started with LM Studio + GGUF.** The biggest shift happened March 30, 2026: Ollama 0.19 switched to MLX backend on Apple Silicon, nearly doubling decode speed. MLX has matured into the consensus default for Apple Silicon inference in 2026.

**Bottom line for automated benchmarking on M4 32GB/120GB/s:**

| Priority | Engine | Why |
|----------|--------|-----|
| **Primary** | **mlx_lm.server** or **Rapid-MLX** | Fastest raw tok/s on Apple Silicon; OpenAI-compatible API; best for MoE models |
| **Runner-up** | **LM Studio 0.4.7+ (MLX engine)** | Polished UI, both MLX & GGUF backends, OpenAI API on :1234, continuous batching |
| **Fallback** | **llama.cpp server** | Maximum control, widest model support, better long-context prefill |
| **Watch** | **Ollama 0.19 MLX** | Easy setup but limited model support (Qwen3.5 only as of April 2026), Go wrapper adds ~30% overhead vs raw MLX |

**Critical caveat for base M4 (120 GB/s):** Your bandwidth is the bottleneck. An M3 Max (400 GB/s) will outperform your M4 despite being older. For MoE models like Qwen3.5-35B-A3B (22GB at Q4) and Gemma-4-26B-A4B (16-18GB at Q4), you can run them but expect roughly HALF the tok/s reported for M4 Pro/Max configurations. Expect ~25-40 tok/s generation for these MoE models via MLX on base M4 32GB.

---

## Hardware Context

### Why Memory Bandwidth Matters More Than Chip Generation

LLM inference is **memory-bandwidth bound**, not compute-bound. Every generated token requires reading the entire model weights from memory once.

| Chip | Bandwidth | Relative Speed |
|------|-----------|----------------|
| M4 base | 120 GB/s | **1.0x (your hardware)** |
| M4 Pro | 273 GB/s | ~2.3x |
| M3 Max | 400 GB/s | ~3.3x |
| M4 Max | 546 GB/s | ~4.5x |
| M4 Ultra | 819 GB/s | ~6.8x |

**Key implication:** When you see benchmarks reporting 130 tok/s for Qwen3.5-35B-A3B on M4 Max, divide by roughly 4-4.5x for your M4 base. Expect ~30-35 tok/s.

### What Fits in 32GB

After OS overhead (~4-6GB) and KV cache allocation, you have ~24-26GB available for model weights:

| Model | Q4 Size | Q8 Size | Fits in 32GB? |
|-------|---------|---------|---------------|
| Qwen3.5-35B-A3B | ~22 GB | ~35 GB | Q4 yes (tight), Q8 no |
| Gemma-4-26B-A4B | ~16-18 GB | ~28-30 GB | Q4 comfortable, Q8 tight |
| Qwen3-14B | ~8 GB | ~15 GB | Both easily |
| Qwen3-8B | ~5 GB | ~9 GB | Both easily |

---

## 1. LM Studio

### Current Version: 0.4.7 (as of April 2026)

### Backend Architecture

LM Studio supports **BOTH** backends and lets you select per-model:
- **llama.cpp engine** -- for GGUF models, Metal GPU acceleration
- **MLX engine** -- for MLX SafeTensors models, native Apple Silicon

LM Studio 0.3.4 (2025) first added MLX support. Version 0.4.0 (January 28, 2026) was a major release adding continuous batching and the headless `llmster` daemon. Version 0.4.2 (February 2026) added continuous batching for the MLX engine specifically.

### Performance on M4

Based on available benchmarks (scaled for M4 base 120GB/s):

| Model | Engine | Estimated tok/s (M4 base 32GB) |
|-------|--------|-------------------------------|
| 8B Q4 | MLX | ~35-45 tok/s |
| 8B Q4 | llama.cpp | ~25-35 tok/s |
| 14B Q4 | MLX | ~18-25 tok/s |
| Qwen3.5-35B-A3B Q4 | MLX | ~25-35 tok/s (MoE advantage) |
| Gemma-4-26B-A4B Q4 | MLX | ~25-35 tok/s (MoE advantage) |

Note: MoE models are faster than their total parameter count suggests because only 3-4B parameters are active per token.

### OpenAI-Compatible API

- **Port:** localhost:1234 (configurable)
- **Endpoints:** `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`
- **Also supports:** Anthropic-compatible `/v1/messages` endpoint
- **SDK compatible:** Works with OpenAI Python SDK by setting `base_url="http://localhost:1234/v1"`
- **v0.4.0+:** Added stateful REST API, parallel request processing with continuous batching
- **v0.4.7:** Fixed bugs in `/v1/responses` and Anthropic `/v1/messages` API

### MoE Model Issues (CRITICAL)

**Active bugs as of April 2026:**

1. **Gemma 4 GGUF fails to load** (Issue #1728): Both gemma-4-31B and gemma-4-26B-A4B GGUF models fail with "Failed to load model" -- llama.cpp backend in 0.4.8+ doesn't yet recognize Gemma 4 architecture
2. **Gemma 4 MLX fails to load** (Issue #1741): MLX backend throws `ValueError` -- `gemma4` architecture not yet defined in bundled MLX engine
3. **Qwen3.5 MoE not detected** (Issue #1723): Models with `qwen3_5_moe` type not recognized by model scanner, don't appear in model list despite the bundled mlx_lm supporting the architecture
4. **MoE offloading regression** (Issue #1419): v0.3.39 had targeted expert-only CPU offloading; v0.4.0 replaced this with full-layer offloading, much less efficient for MoE

### Pros for Automated Benchmarking
- Polished, reliable OpenAI-compatible API
- Can switch between MLX and llama.cpp per model
- Continuous batching (v0.4.2+)
- Wide model format support
- Good UI for manual debugging

### Cons for Automated Benchmarking
- Desktop app dependency (though `llmster` daemon exists for headless)
- Current MoE bugs with newest models (Gemma 4, Qwen3.5)
- Proprietary/closed-source
- MLX engine sometimes lags behind upstream mlx-lm releases

---

## 2. MLX / mlx-lm

### Current State (April 2026): Mature

MLX is Apple's open-source ML framework, purpose-built for Apple Silicon unified memory. It has become the **consensus recommendation** for native Apple Silicon inference in 2026.

Key milestones:
- Apple WWDC 2025: Official MLX session for LLM inference
- Ollama adopted MLX (March 2026)
- Active research paper accepted at EuroMLSys '26 (vllm-mlx)
- MLX Community on Hugging Face has hundreds of converted models

### mlx-lm vs mlx-vlm

| Package | Purpose | Use Case |
|---------|---------|----------|
| **mlx-lm** | Text-only LLM inference & fine-tuning | Chat, code generation, benchmarking |
| **mlx-vlm** | Vision-Language Model inference | Image/video/audio understanding |

The architectures are unified: mlx-lm handles text, mlx-vlm adds vision "add-ons" for image embeddings. Text-only chats with VLMs benefit from prompt caching. Both share the same MLX framework underneath.

### Performance vs llama.cpp -- Actual Numbers

**Generation speed (tok/s) benchmarks from published sources:**

| Hardware | Model | MLX | llama.cpp | MLX Advantage |
|----------|-------|-----|-----------|---------------|
| M4 Max 128GB | Qwen3-0.6B 4-bit | 525.5 | 281.5 | +87% |
| M4 Max 128GB | Qwen3-4B 4-bit | 159.0 | 118.2 | +35% |
| M4 Max 128GB | Qwen3-8B 4-bit | 93.3 | 76.9 | +21% |
| M4 Max 128GB | Llama-3.2-1B 4-bit | 461.9 | 331.3 | +39% |
| M1 Max 64GB | Qwen2.5-7B 4-bit | 63.7 | 40.75 | +56% |
| M1 Max 64GB | Qwen2.5-27B 4-bit | ~14 | ~14 | **Tied** |
| M4 Pro 64GB | Qwen3.5-35B-A3B | ~130 | ~43 (Ollama) | +3x (MoE) |
| M4 Max 128GB | Qwen3.5-35B-A3B | ~130 | ~43.5 (Ollama) | +3x (MoE) |
| M3 Ultra 192GB | Qwen3.5-35B-A3B 4-bit | 95.2 | -- | -- |

**Key patterns:**
- MLX leads by **20-87%** for models under 14B (compute-bound regime)
- Gap **collapses to near-zero at 27B+ dense** models (bandwidth-bound)
- MoE models show the **greatest MLX advantage (up to 3x)** over llama.cpp/Ollama
- For **base M4 (120 GB/s)**, expect roughly 40-50% of the M4 Pro figures

### CRITICAL: The Prefill Problem

**MLX has a significant weakness in prompt processing (prefill):**

| Context Length | MLX Prefill Time | GGUF Prefill Time |
|----------------|-------------------|-------------------|
| 655 tokens | 6.9s | 2.9s |
| 1.5K tokens | 9.2s | 4.1s |
| 3K tokens | 15.0s | 8.7s |
| 8.5K tokens | 49.4s | 37.8s |

(Measured on M1 Max with Qwen3.5-35B-A3B via LM Studio)

**The "57 tok/s on screen, 3 tok/s in practice" problem:** MLX can report 57 tok/s generation but at 8.5K context, prefill accounts for 94% of total time, making effective throughput just ~3 tok/s. This is because:
1. Broken prompt caching for Qwen3.5 multimodal in LM Studio's MLX runtime
2. Unoptimized hybrid attention (gated delta-net, sliding window) in MLX
3. bf16 dtype on M1/M2 (lacks native support; M4 has native bf16)

**Mitigation:** The oMLX runtime with tiered KV cache achieves 5x faster prefill, dropping 8K context prefill from 49s to ~1.7s. Rapid-MLX also has optimized caching.

### OpenAI-Compatible API Server

**Yes, via `mlx_lm.server`:**

```bash
mlx_lm.server --model mlx-community/Qwen3.5-35B-A3B-4bit
```

- Runs on `localhost:8080` by default
- Endpoints: `/v1/chat/completions`, `/v1/models`
- Supports streaming, temperature, top_p/top_k, speculative decoding via `draft_model`
- **NOT production-hardened** -- basic security only, single-threaded
- Auto-downloads models from Hugging Face

**Third-party alternatives:**
- **mlx-openai-server** (FastAPI, more endpoints)
- **Rapid-MLX** (fastest, 17 tool parsers, drop-in OpenAI replacement)
- **vllm-mlx** (continuous batching, highest throughput)
- **oMLX** (SSD-backed KV cache, best for agents)

### MoE Model Support

MLX has robust MoE support with parsers for `qwen3_moe`, `qwen3_5_moe`, `glm4_moe`, and auto-detected converters. The MoE advantage in MLX is significant -- up to 3x faster than llama.cpp for MoE models.

**No known MoE-specific bugs in raw mlx-lm** (bugs are in LM Studio's bundled version).

### Quantization Support

| Format | Supported | Notes |
|--------|-----------|-------|
| MLX SafeTensors | **Native** | 4-bit, 8-bit, bf16, fp16; configurable group size (default 64) |
| GGUF | **No** | Cannot load GGUF directly |
| Hugging Face SafeTensors | Via conversion | mlx-community has hundreds of pre-converted models |

**Memory efficiency:** MLX models are ~7-13% smaller than equivalent GGUF:
- Qwen3-Coder-30B-A3B: 34.7 GB (MLX) vs 40 GB (GGUF) = 13% savings

---

## 3. Ollama

### Current Version: 0.19 (preview, released March 30, 2026)

### The MLX Switch

**When:** Ollama 0.19, released March 30, 2026, as a preview.
**What changed:** On Apple Silicon Macs with 32GB+ unified memory, Ollama now uses MLX instead of llama.cpp/Metal.
**How to enable:** `OLLAMA_MLX=1 ollama serve` (or just update to 0.19+)

### Performance Benchmarks

Official benchmarks (tested on M5 Max with Qwen3.5-35B-A3B):

| Metric | Ollama 0.18 (llama.cpp) | Ollama 0.19 (MLX) | Improvement |
|--------|------------------------|-------------------|-------------|
| Prefill | 1,154 tok/s | 1,810 tok/s | +57% (1.6x) |
| Decode | 58 tok/s | 112 tok/s | +93% (1.9x) |
| Int4 Decode | -- | 134 tok/s | -- |

**For your M4 base (120 GB/s):** Expect roughly 30-40% of these numbers due to bandwidth. Estimated ~40-55 tok/s decode for Qwen3.5-35B-A3B Q4.

### Model Format Support

- **Primary:** NVFP4 (new, higher quality than INT4), int4, Q4_K_M
- **GGUF:** Supported (falls back to llama.cpp Metal backend)
- **MLX SafeTensors:** Via the new MLX backend
- **Custom models:** Modelfile support for importing custom models

### Limitations (CRITICAL for your use case)

1. **Model architecture support is extremely limited:** As of 0.19, only **Qwen3.5 models** are confirmed MLX-compatible. Unsupported models **silently fall back** to llama.cpp Metal backend
2. **Gemma 4 not yet supported** in MLX mode (planned for Ollama 0.20)
3. **Go wrapper overhead:** ~30% latency penalty compared to raw MLX
4. **No continuous batching in MLX mode** -- processes requests sequentially
5. **32GB minimum** required for MLX mode

### OpenAI-Compatible API

- **Port:** localhost:11434
- **Endpoints:** `/api/chat`, `/api/generate`, plus OpenAI-compatible `/v1/chat/completions`
- Works with OpenAI SDK by setting `base_url="http://localhost:11434/v1"`

### MoE Support

MoE models work through both backends. The MLX backend shows ~3x speedup for MoE models specifically. However, the limited architecture support means not all MoE models can use MLX yet.

---

## 4. llama.cpp (Raw)

### Metal Backend Performance on M4

llama.cpp is the foundational C/C++ inference engine. On Apple Silicon, it uses the Metal API for GPU acceleration.

**Estimated performance on M4 base 32GB (120 GB/s):**

| Model | Q4 tok/s | Notes |
|-------|----------|-------|
| 7-8B models | 25-40 | Comfortable |
| 14B | 15-22 | Good |
| Qwen3.5-35B-A3B | 20-30 | MoE helps |
| Gemma-4-26B-A4B | 20-30 | MoE helps |
| 70B | Not feasible | Won't fit in 32GB at Q4 |

### llama-server for API Access

```bash
./llama-server -m model.gguf --port 8080 --parallel 4 --cache-type-k q8_0
```

- Full OpenAI-compatible API on specified port
- **Endpoints:** `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`
- Supports `--parallel N` for concurrent sequences
- KV cache quantization via `--cache-type-k q8_0` (halves KV cache memory)
- Flash Attention enabled for Metal backend

### Advantages Over LM Studio's Wrapper

1. **Every knob exposed:** Context length, KV quantization types, batch sizes, parallel sequences, GPU layer count
2. **No wrapper overhead:** Direct Metal API calls, ~50% faster than Ollama's Go wrapper
3. **Layer offloading:** `-ngl` flag for partial GPU/CPU split when model doesn't fully fit
4. **Build with M4-specific flags:** Cmake with Metal and AMX support
5. **Widest model support:** Hundreds of architectures, including Gemma 4 from day one (April 2, 2026)
6. **Speculative decoding:** `--draft-model` flag for 2-3x throughput improvement

### Latest Apple Silicon Optimizations

- **AMX (Apple Matrix)** instruction set support for M4
- **Flash Attention** for Metal backend (critical for KV cache quantization)
- **KV cache quantization** (q4_0, q8_0) to extend effective context
- **TurboQuant** KV cache compression with rotation for better quality
- **FP8 Hybrid Inference** support

### Speculative Decoding for MoE

Supported via `--draft-model` flag. Use a smaller model (e.g., 0.6B) as draft for the larger MoE model. Expected 2-3x throughput improvement for MoE targets where draft acceptance rate is high.

---

## 5. vLLM-MLX & vllm-metal

### Two Paths to vLLM on Apple Silicon

| Project | vllm-mlx | vllm-metal |
|---------|----------|------------|
| Backend | MLX framework | Metal API |
| Org | Independent (waybarrios) | Official vllm-project |
| Paper | EuroMLSys '26 | -- |
| Multimodal | Text + Vision + Audio + Embeddings | Text only |
| Latest | v0.2.6 (Feb 2026) | Docker-based |

### vllm-mlx Performance (M4 Max 128GB)

| Model | Speed | Memory |
|-------|-------|--------|
| Qwen3-0.6B 8-bit | 402 tok/s | 0.7 GB |
| Llama-3.2-1B 4-bit | 464 tok/s | 0.7 GB |
| Llama-3.2-3B 4-bit | 200 tok/s | 1.8 GB |

**Continuous batching (5 concurrent requests):**
- Qwen3-0.6B: 3.4x speedup (328 -> 1,112 tok/s total)
- Llama-3.2-1B: 2.0x speedup (299 -> 613 tok/s total)

### High-Concurrency Performance (M4 Pro, 32 concurrent requests)

| Framework | Single-User | 32-User Total | TTFT |
|-----------|-------------|---------------|------|
| vllm-mlx | 42 tok/s | **1,150 tok/s** | ~120ms |
| Ollama | **58 tok/s** | 720 tok/s | ~45ms |
| llama.cpp | 52 tok/s | 890 tok/s | ~85ms |

### API Compatibility

- **OpenAI:** `/v1/chat/completions` (drop-in replacement)
- **Anthropic:** Native `/v1/messages` endpoint
- Works with Claude Code and OpenCode
- MCP tool calling support

### Relevance for Your Setup

vllm-mlx is **overkill for single-user benchmarking** but valuable if you need to run multiple benchmark tasks concurrently. For sequential benchmarking, raw mlx-lm or Rapid-MLX will be faster per-request.

---

## 6. Rapid-MLX

### Overview

A newcomer (released March 23, 2026) that's rapidly gaining traction. Claims to be the **fastest local AI engine for Apple Silicon**.

**Key claims:**
- 4.2x faster than Ollama on some models
- 0.08s cached TTFT
- 100% tool calling success rate
- Drop-in OpenAI replacement

### Performance Benchmarks

| Hardware | Model | Rapid-MLX | Ollama | Speedup |
|----------|-------|-----------|--------|---------|
| M3 Ultra 256GB | 122B model | 57 tok/s | -- | -- |
| M3 Ultra 256GB | Coder-Next 80B | 74 tok/s | -- | 0.10s TTFT |
| M3 Ultra 256GB | 35B model | 83 tok/s | -- | -- |
| M3 Ultra 256GB | 9B model | 108 tok/s | 47 tok/s | 2.3x |
| M3 Ultra 256GB | Gemma 4 26B-A4B 4-bit | 85 tok/s | 75 tok/s | 1.13x |
| Same Mac | Qwen3.5-9B | 79 tok/s | 33 tok/s | 2.4x |

### Key Features

- **OpenAI-compatible API** (drop-in replacement, works with Claude Code, Cursor, Aider)
- **17 tool parsers** with automatic recovery for quantized model failures
- **Persistent KV cache** across requests (only new tokens prefilled per turn)
- **RNN state snapshots** for hybrid models like Qwen3.5 DeltaNet
- **Reasoning model support** (Qwen3, DeepSeek-R1) with separate `reasoning_content` field
- **Benchmark script included:** `python scripts/benchmark_engines.py --engine rapid-mlx ollama --runs 3`

### For Your Use Case

**Strong candidate for automated benchmarking.** OpenAI-compatible, fast, handles MoE well, and has the caching optimizations that matter for multi-turn evaluation.

---

## 7. Other Engines

### llama-swap

**Purpose:** Multi-model proxy that hot-swaps between models without restarting.

- Sits in front of llama.cpp, vLLM, tabbyAPI, or any OpenAI-compatible endpoint
- Lightweight single binary
- Auto-starts/stops model servers based on incoming requests
- **Use case:** If your benchmark harness tests multiple models sequentially, llama-swap handles model loading/unloading automatically

### Jan.ai

- Uses llama.cpp for all inference (no MLX support)
- Hybrid local/cloud switching
- GPU acceleration via Metal/llama.cpp only
- Performance: ~12-14 tok/s (base configurations)
- **Not recommended** for Apple Silicon when MLX alternatives exist

### GPT4All

- No MLX support
- Metal/llama.cpp backend only
- 2026 Reasoner adds on-device reasoning with tool calling
- Performance lags LM Studio on Apple Silicon
- **Not recommended** for performance-sensitive benchmarking

### KoboldCpp

- High-performance llama.cpp wrapper
- GGUF models, Metal backend
- Good for creative/RP workloads
- No MLX support
- **Not suitable** for automated benchmarking (focused on UI/story generation)

### LocalAI

- Open-source alternative to OpenAI API
- Docker-based deployment
- Apple Silicon Docker support has been problematic (Issue #1659)
- **Not recommended** for native Apple Silicon performance

### Exo (Distributed Inference)

- Pools multiple Apple Silicon devices into a single inference cluster
- Uses MLX (`exo.inference.mlx.MLXShardedInferenceEngine`)
- Successfully ran DeepSeek V3 (671B) across 8x M4 Pro 64GB Mac Minis at ~5 tok/s
- Thunderbolt 5 RDMA for ultra-low latency
- macOS Tahoe 26.2+ required
- **Use case:** Only relevant if you have multiple Macs to cluster

### oMLX

**Emerging favorite for agent workloads:**
- Native macOS inference server built on MLX
- **Two-tier KV cache:** RAM hot cache + SSD cold cache (persisted as safetensors)
- Drops agent TTFT from 30-90s to under 5s
- OpenAI `/v1/chat/completions` AND Anthropic `/v1/messages` endpoints
- Multi-model support with LRU eviction, model pinning, per-model TTL
- **Strong candidate** for long-running benchmark sessions where context accumulates

---

## 8. Quantization Format Comparison

### GGUF vs MLX 4-bit vs Unsloth UD vs MXFP4_MOE

| Format | Quality | Speed (MLX) | Speed (llama.cpp) | Size | Best For |
|--------|---------|-------------|-------------------|------|----------|
| **MLX 4-bit** | Good | **Fastest on Apple Silicon** | N/A | Smallest (~7-13% smaller than GGUF) | Native MLX engines |
| **GGUF Q4_K_M** | Good ("sweet spot") | Via LM Studio MLX | Good | Standard | Cross-platform, widest compatibility |
| **Unsloth UD-Q4_K_XL** | **Best at 4-bit** (-0.8% vs FP16) | N/A (GGUF only) | Good | ~Same as Q4_K_M | Maximum quality at 4-bit, llama.cpp |
| **Unsloth UD-Q3_K_XL** | Very good (-0.6% vs FP16) | N/A (GGUF only) | Good | Smaller | Tight memory situations |
| **MXFP4_MOE (noctrex)** | Excellent (~98.5% of FP16) | Via GGUF | Good | Standard | **MoE models specifically** |
| **NVFP4 (Ollama)** | Higher than INT4 | Via Ollama MLX | N/A | Standard | Ollama 0.19+ |
| **IQ3_XXS** | SOTA Pareto frontier | N/A | Good | Very small | Maximum compression |

### Recommendations for M4 32GB

1. **For Qwen3.5-35B-A3B:**
   - **MLX route:** Use `mlx-community/Qwen3.5-35B-A3B-4bit` (~20GB) -- leaves room for KV cache
   - **GGUF route:** Use `unsloth/Qwen3.5-35B-A3B-UD-Q4_K_XL` (~22GB) -- best quality GGUF
   - **Note:** Unsloth warns "Currently no Qwen3.5 GGUF works in Ollama due to separate mmproj vision files. Use llama.cpp compatible backends."

2. **For Gemma-4-26B-A4B:**
   - **MLX route:** `unsloth/gemma-4-26B-A4B-it-UD-MLX-4bit` (~16-18GB) -- very comfortable fit
   - **GGUF route:** `unsloth/gemma-4-26B-A4B-it-UD-Q4_K_XL` (~16-18GB)
   - **MXFP4_MOE:** Available from noctrex, best MoE expert precision

3. **General rule:** Run the **largest model that fits at Q4** rather than squeezing a larger model at lower quantization. A model at Q8 consistently outperforms a larger model at Q4.

### Unsloth Dynamic (UD) Quantization Explained

Unsloth Dynamic 2.0 selectively upcasts important layers to 8 or 16-bit while keeping others at 4-bit. Calibrated on long-context chat, coding, and tool-calling data (not Wikipedia). Results:
- UD-Q4_K_XL: 80.5% accuracy vs 81.3% original (-0.8 points)
- UD-Q3_K_XL: 80.7% accuracy (-0.6 points)
- Relative error increase under 4.3%

**MXFP4 layers retired** from UD-Q2/Q3/Q4_K_XL variants for Qwen3.5 due to sensitivity issues.

### Unsloth UD-MLX

Unsloth also produces **UD-MLX** models that bring Dynamic 2.0 per-tensor quantization to Apple Silicon MLX format. Available for both Gemma 4 and Qwen 3.5 families.

---

## 9. Flash Attention & Speculative Decoding

### Flash Attention on Apple Silicon

| Engine | Flash Attention Support | Notes |
|--------|----------------------|-------|
| llama.cpp | **Yes** (Metal backend) | Enabled by default since v0.3.32-era; critical for KV cache quantization |
| MLX | **Partial** | Not as mature as llama.cpp; weaker at long-context prefill |
| LM Studio | **Yes** (both backends) | Enabled by default for Metal |
| Ollama | Via backend | Depends on which backend is active |

**Why it matters:** Without Flash Attention, KV cache quantization (TurboQuant) actually makes things **slower** because the cache must be dequantized for every attention computation.

### Speculative Decoding

| Engine | Support | How |
|--------|---------|-----|
| llama.cpp | **Yes** | `--draft-model small.gguf --num-draft-tokens 3` |
| mlx_lm.server | **Yes** | `draft_model` parameter in API requests |
| LM Studio | **Yes** (since v0.3.10) | Works with both MLX and llama.cpp models |
| Rapid-MLX | **Yes** | Built-in support |
| Ollama | Limited | Not directly exposed |

**Apple's ReDrafter:** Apple Research's Recurrent Drafter achieved up to **2.3x speedup** on Metal GPUs. Practical gains with speculative decoding: **20-50% speed improvement** depending on draft model quality and acceptance rate.

**For MoE models:** Speculative decoding pairs well with MoE targets -- use a tiny dense model (0.6B-1B) as draft for the larger MoE model. The draft model's speed is very high, and MoE target evaluation is relatively fast due to sparse activation.

---

## 10. MoE-Specific Analysis

### Why MoE Models Excel on Your Hardware

MoE models are **uniquely advantaged** on memory-constrained hardware:
- Qwen3.5-35B-A3B: 35B total, only **3B active** per token
- Gemma-4-26B-A4B: 26B total, only **3.8B active** per token (128 experts, top-8 routing)

This means inference speed is closer to a 3-4B dense model while having the knowledge capacity of the full model. On your M4 32GB with 120 GB/s bandwidth, you get MoE model performance that's ~10x better than a 35B dense model would be.

### Engine-Specific MoE Performance

| Engine | MoE Handling | Notes |
|--------|-------------|-------|
| **MLX/mlx-lm** | **Excellent** | Up to 3x faster than llama.cpp for MoE; native sparse routing |
| **Rapid-MLX** | **Excellent** | Includes RNN state snapshots for hybrid attention (DeltaNet) |
| **llama.cpp** | Good | Supports MoE offloading, expert CPU pinning |
| **LM Studio** | Buggy | Model detection issues for Qwen3.5 MoE; Gemma 4 not loading |
| **Ollama 0.19** | Limited | Only Qwen3.5 MoE confirmed; others fall back to llama.cpp |
| **vllm-mlx** | Good | Continuous batching works with MoE |

### Expert Offloading (When Model Doesn't Fully Fit)

On 32GB with a 22GB MoE model, you have limited KV cache space. Options:
- **llama.cpp:** Force expert weights to CPU, keep routing/attention on GPU
- **LM Studio (old v0.3.39):** Had targeted expert-only offloading (v0.4.0 broke this)
- **oMLX:** SSD-backed KV cache tier frees RAM for model weights
- **Experimental:** Rust-based engines that load experts on-demand from SSD via memory-mapped I/O

### Hybrid Attention (Qwen3.5 DeltaNet + Sliding Window)

Qwen3.5-35B-A3B uses hybrid attention (gated delta-net, sliding window). **This is currently better handled by llama.cpp than MLX.** The famstack.dev analysis found that MLX's effective throughput at longer contexts suffered specifically due to unoptimized hybrid attention. Rapid-MLX addresses this with DeltaNet state snapshots.

---

## 11. Benchmark Summary Table

### Estimated Performance on M4 Base 32GB (120 GB/s)

All figures are estimates scaled from published M4 Pro/Max benchmarks, accounting for the ~2.3-4.5x bandwidth disadvantage.

| Engine | Qwen3.5-35B-A3B Q4 (tok/s) | Gemma-4-26B-A4B Q4 (tok/s) | OpenAI API | Setup Complexity |
|--------|----------------------------|----------------------------|------------|-----------------|
| **Rapid-MLX** | ~30-40 | ~30-40 | Yes (drop-in) | Low (pip install) |
| **mlx_lm.server** | ~30-40 | ~30-40* | Yes | Low (pip install) |
| **LM Studio (MLX)** | ~25-35** | Broken*** | Yes (:1234) | Very Low (GUI) |
| **LM Studio (llama.cpp)** | ~15-25 | ~15-25 | Yes (:1234) | Very Low (GUI) |
| **Ollama 0.19 (MLX)** | ~30-40 | Not supported yet | Yes (:11434) | Very Low |
| **Ollama (llama.cpp)** | ~15-25 | ~15-25 | Yes (:11434) | Very Low |
| **llama.cpp server** | ~15-25 | ~15-25 | Yes | Medium (build) |
| **vllm-mlx** | ~25-35 | ~25-35 | Yes | Medium (pip) |
| **oMLX** | ~30-40 | ~30-40 | Yes | Low (macOS app) |

\* MLX support for Gemma 4 requires latest mlx-lm (check upstream)
\** LM Studio model scanner may not detect qwen3_5_moe architecture
\*** Gemma 4 fails to load on LM Studio as of 0.4.8 (both engines)

### Memory Overhead Beyond Model Weights

| Engine | Typical Overhead | Notes |
|--------|-----------------|-------|
| mlx_lm | ~1-2 GB | Lean Python process |
| Rapid-MLX | ~1-2 GB | Similar to mlx_lm |
| LM Studio | ~2-4 GB | Electron app + engine |
| Ollama | ~1-3 GB | Go runtime + engine |
| llama.cpp | ~0.5-1 GB | Minimal C++ process |
| vllm-mlx | ~2-3 GB | Python + batching buffers |
| oMLX | ~1-2 GB + SSD cache | Varies with cache tier |

### Context Window & KV Cache

| Engine | Max Context | KV Cache Quantization | Cache Persistence |
|--------|-------------|----------------------|-------------------|
| llama.cpp | Model limit | q4_0, q8_0 | Manual save/restore |
| mlx_lm | Model limit | Not natively | No |
| LM Studio | Configurable | Via llama.cpp engine | In-session |
| Ollama | Model limit | Via backend | In-memory prefix caching |
| oMLX | Model limit | q4/q8 | **SSD-backed persistent** |
| Rapid-MLX | Model limit | KV trimming + state snapshots | **Persistent across requests** |
| vllm-mlx | Model limit | Paged memory | Warm-start only |

---

## 12. Final Verdict for Your Use Case

### Your Requirements Mapped

| Requirement | Best Option |
|-------------|-------------|
| OpenAI-compatible API | All options support this |
| Automated benchmarking harness | Rapid-MLX, mlx_lm.server, or llama-server |
| Qwen3.5-35B-A3B | **Rapid-MLX** or **mlx_lm.server** (MLX 3x faster for MoE) |
| Gemma-4-26B-A4B | **llama.cpp server** (most reliable) or **mlx_lm.server** (if latest mlx-lm) |
| MoE model reliability | **llama.cpp** (widest support) or **mlx-lm** (fastest) |
| Maximum tok/s | **MLX-based engine** |
| Long context (8K+) | **llama.cpp** (better prefill) or **oMLX** (SSD KV cache) |
| Multi-model testing | **llama-swap** as proxy in front of any backend |

### Recommended Setup

**Option A: Maximum Performance (Recommended)**

```
Rapid-MLX (primary) --> OpenAI API on localhost:8080
  + llama-swap (for model switching)
  + llama.cpp server (fallback for unsupported models)
```

- Use Rapid-MLX for MLX-format models (Qwen3.5, etc.)
- Fall back to llama.cpp for models without good MLX conversions
- llama-swap handles routing transparently

**Option B: Simplicity**

```
LM Studio 0.4.7+ --> OpenAI API on localhost:1234
  - Use MLX engine for MoE models (when bugs are fixed)
  - Use llama.cpp engine as fallback
  - Single UI for everything
```

Wait for LM Studio to fix Gemma 4 and Qwen3.5 MoE detection bugs.

**Option C: Maximum Reliability**

```
llama.cpp server --> OpenAI API on localhost:8080
  - Widest model support
  - Most control
  - Slightly slower than MLX but rock-solid
  - All models work from day one
```

### Is There a Consensus?

**Yes, with nuance.** The 2026 consensus is:

> **MLX is the default recommendation for Apple Silicon.** It's 20-87% faster for generation, significantly faster for MoE models, and has matured into a production-ready framework. However, llama.cpp remains essential as a fallback for maximum model compatibility, long-context workloads, and cross-platform needs.

The direction is clear: Ollama switched to MLX, LM Studio supports MLX, vllm-mlx exists, Rapid-MLX builds on MLX. The ecosystem is converging on MLX as the Apple Silicon standard.

**For your specific case (automated benchmarking):** Start with Rapid-MLX or mlx_lm.server for speed, keep llama.cpp server as fallback for model compatibility, and consider llama-swap to manage multiple models.

---

## Sources

### Primary Benchmark Sources
- [SiliconBench -- Apple Silicon LLM Benchmarks](https://siliconbench.radicchio.page/)
- [Silicon Score Bench](https://siliconscore.com/bench/)
- [MLX vs llama.cpp Benchmark (famstack.dev)](https://famstack.dev/guides/mlx-vs-gguf-apple-silicon/)
- [MLX vs llama.cpp Benchmark (Groundy)](https://groundy.com/articles/mlx-vs-llamacpp-on-apple-silicon-which-runtime-to-use-for-local-llm-inference/)
- [2026 Mac Framework Selection (MACGPU)](https://macgpu.com/en/blog/2026-mac-inference-framework-vllm-mlx-ollama-llamacpp-benchmark.html)

### Engine-Specific Sources
- [Ollama MLX Blog Post](https://ollama.com/blog/mlx)
- [Ollama 93% Faster (DEV Community)](https://dev.to/alanwest/ollama-just-got-93-faster-on-mac-heres-how-to-enable-it-3gce)
- [Ollama MLX Review (andrew.ooo)](https://andrew.ooo/posts/ollama-mlx-apple-silicon-review/)
- [LM Studio Changelog](https://lmstudio.ai/changelog)
- [LM Studio 0.4.0 Blog](https://lmstudio.ai/blog/0.4.0)
- [LM Studio 0.3.4 MLX Blog](https://lmstudio.ai/blog/lmstudio-v0.3.4)
- [mlx_lm Server Documentation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md)
- [vllm-mlx GitHub](https://github.com/waybarrios/vllm-mlx)
- [Rapid-MLX GitHub](https://github.com/raullenchai/Rapid-MLX)
- [oMLX GitHub](https://github.com/jundot/omlx)
- [llama.cpp Discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167)

### Bug Reports
- [LM Studio: Gemma 4 GGUF fails to load (#1728)](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1728)
- [LM Studio: Gemma 4 MLX fails (#1741)](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1741)
- [LM Studio: Qwen3.5 MoE not detected (#1723)](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1723)
- [LM Studio: MoE offloading regression (#1419)](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1419)

### Quantization & Model Sources
- [Unsloth Dynamic 2.0 Documentation](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)
- [Unsloth Qwen3.5 Guide](https://unsloth.ai/docs/models/qwen3.5)
- [Unsloth Gemma 4 Guide](https://unsloth.ai/docs/models/gemma-4)
- [noctrex MXFP4_MOE Models](https://huggingface.co/noctrex/Qwen3-Coder-Next-MXFP4_MOE-GGUF)

### Overview & Guides
- [MLX: The Next Inference Engine for Apple Silicon](https://yage.ai/share/mlx-apple-silicon-en-20260331.html)
- [Local LLM Inference in 2026 Complete Guide (DEV Community)](https://dev.to/starmorph/local-llm-inference-in-2026-the-complete-guide-to-tools-hardware-open-weight-models-2iho)
- [Apple Silicon Local LLM Servers for Agents](https://stochasticsandbox.com/posts/the-stack-apple-silicon-local-agents-2026-03-28/)
- [Best Local LLMs for Mac 2026 (InsiderLLM)](https://insiderllm.com/guides/best-local-llms-mac-2026/)
- [Best Way to Run Qwen 3.5 on Mac (InsiderLLM)](https://insiderllm.com/guides/qwen35-mac-mlx-vs-ollama/)
- [Ollama Faster on Macs (MacRumors)](https://www.macrumors.com/2026/03/31/ollama-now-runs-faster-apple-silicon-macs/)
- [Ollama MLX (9to5Mac)](https://9to5mac.com/2026/03/31/ollama-adopts-mlx-for-faster-ai-performance-on-apple-silicon-macs/)
- [GGUF vs MLX Deep Dive (MinerAle)](https://www.mineraleyt.com/posts/gguf-vs-mlx/)
- [llama-swap Setup Guide](https://modelslab.com/blog/api/hot-swap-local-llms-instantly-llama-swap-setup-guide-2026)
- [LM Studio vs Jan vs GPT4All (ToolHalla)](https://toolhalla.ai/blog/lm-studio-vs-jan-vs-gpt4all-2026)
- [Gemma 4 on Apple Silicon 85 tok/s (DEV Community)](https://dev.to/raullen_chai_76e18e9705b0/gemma-4-on-apple-silicon-85-toks-with-a-pip-install-299a)
- [Production-Grade LLM Inference Comparative Study (arXiv)](https://arxiv.org/abs/2511.05502)
