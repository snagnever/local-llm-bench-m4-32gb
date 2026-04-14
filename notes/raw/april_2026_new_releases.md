# April 2026 New Releases Research (Third Pass)
**Date: April 8, 2026**
**Focus: Brand new models, engines, and tools for local inference on 32GB Mac**

---

## 1. BRAND NEW MODEL RELEASES (Late March - April 8, 2026)

### Google Gemma 4 (Released April 2, 2026) -- TOP CONTENDER
- **Four sizes**: E2B (2.3B), E4B (4.5B), 26B-A4B MoE, 31B Dense
- **License**: Apache 2.0
- **Key features**: Multimodal (text, images, video; E2B/E4B also audio), 256K context, 140+ languages
- **26B-A4B MoE**: 25.2B total params, only 3.8B active per token. ~10-14GB GGUF. EXCELLENT for 32GB Mac.
- **31B Dense**: 20-24GB memory needed. Fits 32GB Mac with ~8GB headroom. Strongest quality.
- **Already supported**: Ollama 0.20, LM Studio 0.4.9, llama.cpp, MLX native
- **APEX GGUF available**: mudler/gemma-4-26B-A4B-it-APEX-GGUF (18.1GB balanced, beats Q8_0 quality)
- **Unsloth GGUF available**: unsloth/gemma-4-26B-A4B-it-GGUF, unsloth/gemma-4-31B-it-GGUF

### Qwen 3.6-Plus (Released April 2, 2026)
- **API-only so far** -- available via OpenRouter free preview
- 1M context window, 65K output tokens, always-on CoT reasoning, native function calling
- Hybrid architecture: linear attention + sparse MoE
- Designed for agentic AI workflows
- **NOT available for local inference yet** -- no weights released

### Qwen3.5-Omni (Released March 30, 2026) -- INTERESTING
- Native multimodal: text, images, audio, video
- Thinker-Talker architecture with Hybrid-Attention MoE
- 256K context, 10+ hours audio, 113 languages for speech recognition
- Three sizes: Plus, Flash, Light
- **Local viability**: Depends on model size; details on weight availability TBD

### Huihui3.5-67B-A3B (Released ~April 6, 2026) -- NEW CONTENDER
- Built on Qwen3.5-35B-A3B base, replaces MLP with MoE layers (512 experts per layer, 8 active)
- 67B total params, ~3B active -- very efficient for inference
- Abliterated version also available (Huihui3.5-67B-A3B-abliterated)
- **32GB Mac**: Should fit comfortably with GGUF quantization given only 3B active params
- **Note**: No "97B-A3B" variant found -- may not exist or may be upcoming

### Huihui-Qwopus3.5-*-v3-abliterated (Very recent, ~April 5-6, 2026)
- Abliterated versions of Jackrong's Qwopus3.5 models
- 9B version: 1.13k likes, 19 downloads -- very popular
- 27B and 4B versions also available

### Huihui-gemma-4-E2B-it-abliterated (Released April 8, 2026 -- TODAY)
- Abliterated Gemma 4 E2B variant from huihui-ai
- Updated "2 minutes ago" as of search time

### Jackrong Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled -- VERY INTERESTING
- Qwen3.5-35B-A3B fine-tuned with Claude 4.6 Opus CoT distillation
- 25.3% shorter reasoning, 24% lower token cost per correct answer
- MLX versions available: 4-bit, 6-bit, 8-bit
- **32GB Mac**: 35B-A3B with only 3B active params -- fits easily
- Qwopus3.5-9B-v3: 87.80% pass@1, outperforms base Qwen3.5-9B (82.93%)
- Training notebook and codebase released April 5, 2026

### Nemotron Cascade 2 30B-A3B (Released March 19, 2026)
- 31.6B total params, 3B active (MoE + Mamba-2 hybrid)
- **Gold medal** performance on IMO, IOI, ICPC with only 3B active params
- Q4_K_M GGUF: 24.5GB -- tight but potentially viable on 32GB Mac
- 1M token context window
- **Excellent reasoning model for local use**

### Nemotron 3 Super (Released ~March 2026)
- 120B total params, 12B active -- MoE hybrid architecture
- 1M context window, 5x throughput vs previous gen
- **Too large for 32GB Mac** at 120B total even quantized

### Nemotron 3 Nano 30B-A3B (Released ~March 2026)
- 30B total params, 3B active
- Hybrid Mamba-2 + MoE + Attention (23 Mamba-2/MoE layers, 6 Attention layers)
- 1M context window
- **Fits 24GB RAM** -- comfortable on 32GB Mac
- GGUF, FP8, NVFP4 quantizations available

### Mistral Small 4 (Released March 16, 2026)
- 119B total params, 6.5B active (128 experts, 4 active per token)
- Unifies: instruct + reasoning (Magistral) + multimodal (Pixtral) + coding (Devstral)
- 256K context window
- **32GB Mac**: Very tight. Would need IQ2/Q3 quantization. Heavy CPU offloading needed.
- Apache 2.0 license

### Ministral 3 Series (Part of Mistral 3 family)
- 3B, 8B, 14B dense models
- Base, instruct, and reasoning variants, all with image understanding
- Apache 2.0 license
- 3B model: runs on 4GB VRAM with 4-bit quantization
- **Excellent edge/laptop models for 32GB Mac**

### Voxtral TTS (Released March 27, 2026)
- 4B parameter text-to-speech model from Mistral
- 9 languages, emotionally expressive
- Open-source, lightweight
- **Complementary tool** for local AI stack

### Leanstral (Released March 2026)
- 6B active parameter code agent for Lean 4
- Specialized niche model

### Meta Llama 4 Scout (Released ~March 2026)
- 109B total, 17B active (16 experts MoE)
- 10M token context window (industry record)
- Q4: ~55GB -- **Does NOT fit 32GB Mac** without heavy offloading
- 1.78-bit quantization: fits 24GB VRAM (but quality questionable)

### GLM-5 (Released February 11, 2026)
- 744B total params, 40-44B active (MoE)
- MIT License, trained on Huawei Ascend chips
- 200K context, 128K output tokens
- **Too large for 32GB Mac** -- needs server hardware

### Qwen3.5-35B-A3B (Released ~February 2026, but highly relevant)
- 35B total, 3B active -- MoE architecture
- 262K context (extendable to 1M)
- Multimodal (vision)
- ~22GB for full model, fits 32GB Mac easily
- 112 tok/s on RTX 3090, ~15 tok/s on MacBook Air M4 24GB
- **Note**: Currently no Qwen3.5 GGUF works in Ollama (separate mmproj files issue). Use llama.cpp.

---

## 2. INFERENCE ENGINE UPDATES (April 2026)

### Ollama 0.19 (Released March 29-31, 2026) -- MAJOR UPDATE
- **MLX backend for Apple Silicon** (preview)
- **57% faster prefill, 93% faster decode** vs 0.18 (tested with Qwen3.5-35B-A3B)
- 1,810 tok/s prefill (vs 1,154 in 0.18), 134 tok/s decode with int4
- M5 chips get additional acceleration via GPU Neural Accelerators
- No manual configuration needed -- automatic on Apple Silicon

### Ollama 0.20 (Released April 2, 2026)
- v0.20.0: April 2 | v0.20.2: April 4 | v0.20.3: April 7 | v0.20.4-rc0: April 7
- Added Gemma 4 support in MLX backend
- SentencePiece-style BPE tokenization support
- Improved argument parsing and tool call handling
- OpenClaw TUI fixes

### llama.cpp b8684 (April 7, 2026)
- Latest release build artifacts: April 6-7
- Continues to be the reference GGUF inference engine
- TurboQuant PR under review for integration
- APEX quantization works with stock llama.cpp (no patches needed)

### LM Studio 0.4.9 (April 2, 2026)
- Improved Gemma 4 tool call reliability
- Anthropic-compatible effort levels (low/medium/high/max)
- Bug fixes for UI freezes and markdown rendering

### MetalRT -- NEW ENGINE (2026)
- First complete AI inference engine for Apple Silicon: LLM + STT + TTS
- 1.10-1.19x faster than mlx-lm on decode
- 1.35-2.14x faster than llama.cpp across the board
- 658 tok/s decode speed claimed (on M4 Max 64GB)
- Speech-to-text: 714x faster than real-time
- Benchmarked on M4 Max with small models (0.6B-4B range)

### Apfel v0.6.13 (April 3, 2026) -- NEW TOOL
- Exposes Apple's on-device ~3B LLM (built into macOS 26 Tahoe) as:
  - CLI tool
  - OpenAI-compatible HTTP server
  - Interactive chat app
- Zero cost, 100% on-device, uses Neural Engine + GPU
- MIT licensed
- 513 points on Hacker News
- Limited to Apple's built-in ~3B model (4096 context)

### Cactus AI Engine (April 2, 2026) -- NEW ENGINE
- Mobile-focused inference framework
- 582 tok/s on Mac M4 for 1.2B model (76MB RAM!)
- Eliminates memory copies, leverages NPUs
- Unified multimodal SDK

### Docker Model Runner -- NEW
- Added vLLM support on macOS with Metal acceleration

---

## 3. QUANTIZATION TECHNIQUE UPDATES

### APEX (Adaptive Precision for EXpert Models) -- NEW, HOT
- MoE-aware mixed-precision quantization
- Layer-wise precision gradient: sensitive edge layers get higher precision, middle compressed more
- **Outperforms Unsloth Dynamic 2.0 on accuracy at 2x smaller size**
- Works with stock llama.cpp (no patches)
- Uses --tensor-type-file and --tensor-type flags
- Pre-quantized models available: Gemma 4, Qwen3.5, Qwen3-Coder-Next
- Gemma 4 26B-A4B APEX Balanced: 18.1GB, beats Q8_0 on perplexity (316.4 vs 337.3)
- **10-18% faster token generation** than traditional quants at same quality

### TurboQuant (Google, Published March 24, 2026)
- KV cache compression to 3-4 bits without retraining
- Heading to ICLR 2026 (late April, Rio de Janeiro)
- **Community status**: 5 independent implementations exist (2 weeks after paper)
  - One running 104B model on MacBook
  - llama.cpp PR under review
- **Official status**: Google implementation expected Q2 2026
- **llama.cpp integration**: Q3 2026 roadmap
- **Impact**: Dramatically reduces memory for long-context inference (6x reduction claimed)

### Unsloth Dynamic 2.0 -- MATURE
- Intelligent layer-wise quantization for all models (MoE and dense)
- 1.5M token calibration dataset
- Gemma 4 GGUFs already available from Unsloth

### MXFP4_MOE -- SPECIALIZED
- noctrex/Huihui-Qwen3.5-35B-A3B-abliterated-MXFP4_MOE-GGUF
- MoE-specific FP4 quantization format

---

## 4. SPECIFIC ORG STATUS CHECK

| Organization | Latest Model | Local 32GB Mac Viable? | Status |
|---|---|---|---|
| **Google DeepMind** | Gemma 4 (Apr 2) | YES - 26B-A4B (10-14GB) or 31B (20-24GB) | Released, fully supported |
| **Alibaba (Qwen)** | Qwen3.6-Plus (Apr 2), Qwen3.5-Omni (Mar 30) | 3.6-Plus: NO (API only). Qwen3.5-35B-A3B: YES | Active releases |
| **NVIDIA** | Nemotron Cascade 2 (Mar 19), Nano (Mar) | Nano 30B-A3B: YES. Cascade 2: TIGHT (24.5GB Q4) | Released |
| **Meta** | Llama 4 Scout (Mar) | TIGHT at Q4 (55GB). Needs heavy quantization | Planning more open-source models |
| **Mistral** | Mistral Small 4 (Mar 16), Ministral 3 | MS4: TIGHT (119B). Ministral 3B/8B/14B: YES | Ministral is great for Mac |
| **Microsoft** | Phi-4-multimodal (5.6B), Phi-4-mini (3.8B) | YES -- both easily fit | No new Phi-5 announced |
| **Zhipu AI** | GLM-5 (Feb 11) | NO -- 744B total | Too large |
| **DeepSeek** | V3.2 (Feb 15), "V4 Lite" rumor (Mar 9) | V3.2: NO (671B). R2: NOT RELEASED | R2/V4 expected but delayed |
| **huihui-ai** | Huihui3.5-67B-A3B (Apr 6) | YES -- 3B active params | Very active, new today |
| **Jackrong** | Qwopus3.5-*-v3, Opus-Distilled (Apr 5) | YES -- 35B-A3B model fits easily | MLX versions available |
| **Moonshot AI** | Kimi K2.5 | NO -- trillion-parameter scale | Too large for 32GB |

---

## 5. BENCHMARK & EVALUATION UPDATES

### lm-evaluation-harness (2026)
- New Metabench benchmark contributed
- Open Arabic LLM Leaderboard benchmarks added
- Unitxt Multimodality support
- New multilingual benchmarks: Spanish, Galician, Basque, Catalan
- Vision Language Model support (prototype): hf-multimodal, vllm-vlm model types
- Chat template handling improvements

### Silicon Score (siliconscore.com)
- Apple Silicon-specific benchmarking site for local LLM inference

---

## 6. APPLE SILICON / M5 SPECIFIC DEVELOPMENTS

### Apple M5 Chip (Released March 2026)
- GPU Neural Accelerators: dedicated matrix multiplication hardware
- MLX optimized at chip design phase
- Qwen3-14B-4bit: 4.06x faster time-to-first-token vs M4
- Token generation: 1.19x faster vs M4
- FLUX-dev-4bit image gen: 3.8x faster than M4

### macOS 26 (Tahoe)
- Built-in ~3B on-device LLM (exposed by Apfel)
- MLX is the preferred/official framework for LLM inference

### MLX Framework
- Now Ollama's default backend on Apple Silicon
- Supports: Gemma 4, Qwen3.5, Llama 4, all major architectures
- Neural Accelerator support on M5

---

## 7. TOP RECOMMENDATIONS FOR 32GB MAC (Newly Discovered)

### Tier 1: Excellent Fit (< 20GB, fast inference)
1. **Gemma 4 26B-A4B MoE** -- 10-14GB GGUF, 3.8B active, multimodal, Apache 2.0
2. **Qwen3.5-35B-A3B** -- ~22GB, 3B active, multimodal, 262K context
3. **Nemotron 3 Nano 30B-A3B** -- ~24GB, 3B active, Mamba-2 hybrid, 1M context
4. **Huihui3.5-67B-A3B** -- 3B active, 512 experts, brand new (Apr 6)
5. **Jackrong Qwen3.5-35B-A3B-Opus-Distilled** -- 3B active, optimized reasoning, MLX ready

### Tier 2: Good Fit (20-28GB, comfortable)
6. **Gemma 4 31B Dense** -- 20-24GB GGUF, strongest Gemma 4 quality
7. **Nemotron Cascade 2 30B-A3B** -- 24.5GB Q4, gold-medal reasoning
8. **Qwopus3.5-27B-v3** -- 27B dense, Claude-distilled quality

### Tier 3: Tight Fit (needs aggressive quantization)
9. **Llama 4 Scout** -- 109B total, needs extreme quantization (1.78-bit = 24GB)
10. **Mistral Small 4** -- 119B total, needs IQ2/Q3 quantization

### Edge/Fast Models
11. **Ministral 3 14B** -- Dense, multimodal, reasoning variant available
12. **Gemma 4 E4B** -- 4.5B, excellent quality/size ratio
13. **Phi-4-multimodal** -- 5.6B, speech+vision+text
14. **Qwopus3.5-9B-v3** -- 87.8% pass@1, great reasoning

---

## 8. KEY FINDINGS & WHAT WE MIGHT HAVE MISSED

### Confirmed NOT released yet:
- DeepSeek R2 -- still delayed, no release date
- DeepSeek V4 -- rumored April 2026, confirmed training on Huawei chips
- Huihui3.5-97B-A3B -- NOT FOUND. Only 67B-A3B exists.
- Nemotron 4 -- announced but not released
- Phi-5 -- no announcement found

### Important discoveries:
1. **APEX quantization** is a game-changer for MoE models -- better quality than Q8_0 at fraction of size
2. **Ollama 0.20** already released (April 2) with Gemma 4 + MLX support
3. **MetalRT** is a new inference engine beating both llama.cpp and MLX on decode speed
4. **TurboQuant** community implementations already running 104B models on MacBook
5. **Apfel** exposes macOS 26's built-in LLM as an OpenAI-compatible server
6. **Gemma 4 26B-A4B** is the clear "sweet spot" model for 32GB Mac in April 2026

### Inference engine landscape (April 2026):
- **Ollama 0.20** -- most user-friendly, now MLX-powered on Mac
- **llama.cpp b8684** -- most flexible, GGUF reference
- **LM Studio 0.4.9** -- best GUI experience
- **MetalRT** -- fastest decode on Apple Silicon (new challenger)
- **MLX native** -- Apple's framework, fastest with M5 chips
- **Cactus AI Engine** -- mobile/edge focused, extremely fast for small models
