# MLX for LLM Inference on Apple Silicon: Bugs, Workarounds & Status

**Research date:** April 8, 2026  
**Context:** 32GB M4 Mac, local LLM inference for coding agents  
**mlx-lm current version:** 0.31.2 (released April 7, 2026)

---

## 1. Bug Status Tracker

### Bug 1: Qwen3.5 DeltaNet/Mamba Caching Broken

**Status: OPEN (critical, unfixed)**

- **Issue #903** - "Caching doesn't seem to be working for Qwen3.5" (opened Feb 17, 2026, still OPEN)
- **Issue #980** - "Prefix cache reuse is broken for ALL hybrid-architecture models" (opened Mar 11, 2026, still OPEN)

**What happens:** The mlx-lm server reprocesses the entire prompt on every request. Cached token counter is always 0. A 10-15 second task becomes 1min 46sec. At 40K context: ~200s vs ~5s with cache reuse.

**Root cause:** Qwen3.5 uses a hybrid architecture (75% Gated DeltaNet linear attention + 25% full softmax attention, in a 3:1 pattern). The DeltaNet/Mamba layers maintain recurrent state that is fundamentally non-trimmable -- it's a compressed recurrent state that can't be split at arbitrary token boundaries. The `can_trim_prompt_cache()` check requires ALL layers to be trimmable; one non-trimmable layer causes total recomputation.

**CRITICAL FINDING: ALL Qwen3.5 sizes are hybrid.** The 0.8B, 2B, 4B, 9B, 27B dense models AND the 35B-A3B, 122B-A10B MoE models all use the same Gated DeltaNet hybrid architecture. There are NO pure-attention Qwen3.5 variants. This means the caching bug affects the entire Qwen3.5 family.

**Affected models (confirmed):**
- Qwen 3.5 (ALL sizes: 0.8B through 397B) -- Attention + DeltaNet/Mamba hybrid
- Gemma 3 (all sizes) -- 5:1 sliding + global attention pattern
- Llama 4 Scout/Maverick -- iRoPE chunked + NoPE
- GPT-OSS 120B/20B -- Sliding + full attention hybrid
- Qwen2.5-VL -- Partial sliding window

**Only working model:** MiniMax M2.5 (pure full attention MoE) -- confirmed 10.5x speedup from prefix caching.

**Performance impact (Mac Studio M3 Ultra benchmark):**

| Model | Request 1 | Request 2 | Request 3 | Cache Speedup |
|-------|-----------|-----------|-----------|---------------|
| MiniMax M2.5 | 29.33s | 6.15s | 2.79s | 10.5x |
| GPT-OSS 120B | 1.54s | 1.77s | 1.67s | None |
| Qwen 3.5 9B | 5.02s | 7.76s | 8.00s | None (slower!) |

**PR status:**
- PR #923 "Hybrid cache for Qwen3.5" -- CLOSED without merge (rejected approach)
- No replacement PR merged as of April 8

**Maintainer position:** angeloskath (maintainer) says sequential multi-turn chat works via "shorter cache" path, but prefix reuse across parallel requests sharing system prompts does NOT work. They acknowledge the agentic workflow case cannot be fixed without changing chat templates.

### Bug 2: Tool-Calling Degradation at 4-bit Quantization

**Status: CLOSED (disputed -- may still exist)**

- **Issue #1011** - "Multi-turn tool calling degrades with 4-bit/8-bit checkpoints" (opened Mar 16, closed Mar 17, 2026)
- **Issue #1061** - "Malformed tool-call output around 20k prompt tokens" (opened Mar 26, still OPEN)

**What happens:** MLX-community 4-bit checkpoints lose ability to emit structured `tool_use` blocks after ~5 rounds (~3k tokens). 8-bit degrades after ~13 rounds (~5.5k tokens). Model emits tool calls as plain text instead of structured blocks.

**Comparison:** GGUF Q4_K_XL completed 70/70 rounds with zero degradation. Cloud full-precision also 70/70.

**Issue #1011 closure context:** Maintainer (angeloskath) closed after reproducing successfully using mlx-lm's OpenAI API interface. Suggested the issue may be oMLX-specific or version-related, not fundamental to quantization. However, issue #1061 demonstrates that at ~20k tokens, mlx_lm.stream_generate() itself (no external wrappers) produces malformed XML tool-call blocks. A draft PR #1066 "Fix gated delta kernel precision" is OPEN but NOT merged.

**Cascading contamination:** Once malformed output enters conversation history, subsequent turns show dramatically increased corruption.

### Bug 3: KV Cache Cross-Contamination Between Concurrent Requests

**Status: FIXED (merged Mar 10, 2026)**

- **Issue #965** - "KV cache cross-contamination between concurrent requests" (opened Mar 8, closed Mar 10)
- **Fix:** PR #976 "Late binding caused incorrect cache checkpoint" -- MERGED Mar 10

**What happened:** At 16+ concurrent requests, response data from one prompt leaked into others. Model answered one user's question with another user's context. Agent score degraded from 0.727 to 0.456 purely from concurrency.

**Resolution:** Fixed in mlx-lm 0.31.1+. The late-binding cache checkpoint mechanism was corrected.

### Bug 4: Cache Contamination Across Conversations (0.31.0 Regression)

**Status: CLOSED but reportedly still broken**

- **Issue #975** - "Strange cache behavior with 0.31.0 in server mode" (opened Mar 9, closed)

**What happens:** After clearing chat history and starting a new conversation, model hallucinates extra tokens from previous conversations. E.g., user sends "hi", model sees "junkjunk hi" (tokens from prior session).

**Status:** Maintainer closed as "probably fixed in 0.31.1" but reporter confirmed bug persists in 0.31.1 on March 22. Root cause suspected to be off-by-one error in cache management.

### Bug 5: Prefill Much Slower Than llama.cpp

**Status: ARCHITECTURAL LIMITATION (partial workarounds exist)**

**The data (Qwen3.5-35B-A3B on M1 Max):**

| Metric | MLX (LM Studio) | GGUF (llama.cpp) |
|--------|-----------------|-------------------|
| UI-reported speed | 51-57 tok/s | 29 tok/s |
| At 8.5K context total time | 52.3s | 43.4s |
| Prefill time at 8.5K | 49.4s | 37.8s |
| Effective throughput at 8.5K | **3 tok/s** | 5.6 tok/s |
| Prefill % of total time | 94% | 87% |

The UI-reported generation speed is misleading. It only measures the decode phase, which is a small fraction of total response time. Prefill dominates real-world experience.

**Root causes:**
1. LM Studio MLX defaults to prefill chunk size of 512 (should be 2048-8192)
2. MLX models ship as bf16; M1/M2 lack native bf16 support (GGUF uses fp16)
3. Prompt caching broken for hybrid models (see Bug 1)
4. Batch processing path for prefill is younger and less optimized

### Bug 6: DeltaNet 2.7x Decode Slowdown (Embeddings)

**Status: FIXED (closed Feb 28, 2026)**

- **Issue #932** - Decode throughput dropped from ~93 tok/s to ~34 tok/s when input_embeddings deviated from vocabulary vectors.
- **Root cause:** Inputs were being automatically cast to float32 instead of original dtype.
- **Fix:** Cast embeddings back to original dtype. Straightforward user-side fix, no library patch needed.

---

## 2. What People Actually Do as Workarounds

### Option A: Switch to GGUF/llama.cpp (most common)

The most reliable workaround for Qwen3.5 specifically. GGUF Q4_K_XL handles:
- Full prompt caching (works correctly)
- 70/70 tool-calling rounds without degradation
- FlashAttention-accelerated prefill
- No bf16 issues on M1/M2

**Tradeoff:** 20-87% lower generation throughput on Apple Silicon vs MLX for pure decode. But when prefill dominates (agentic workflows, growing context), GGUF often has lower total wall-clock time.

### Option B: Use oMLX (tiered KV cache)

[oMLX](https://github.com/jundot/omlx) is an MLX-based inference server with SSD-tiered KV caching:
- Hot tier (RAM) + cold tier (SSD) for KV cache
- Cache persists across server restarts
- At 8K context: 1.7s prefill vs 49s in LM Studio MLX (**28x improvement**)
- Beats both GGUF and LM Studio MLX across tested scenarios
- Managed via macOS menu bar

**Limitations:** No specific DeltaNet/hybrid caching fix documented. Relies on mlx-lm underneath, so inherits fundamental architecture limitations. Version 0.31.0 compatibility was an open issue (#110 on oMLX repo).

### Option C: Use Rapid-MLX (DeltaNet state snapshots)

[Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) is the only engine that directly addresses DeltaNet caching:
- **DeltaNet state snapshots:** Snapshots RNN state at system prompt boundary, restores in ~0.1ms instead of re-running recurrent layers
- 2-4x TTFT improvement on cached requests for Qwen3.5
- 17 dedicated tool parsers with automatic recovery for malformed tool calls
- Tested fastest on 16/18 models vs Ollama, mlx-lm, llama.cpp
- Apache 2.0 licensed, drop-in OpenAI-compatible API

**Benchmarks (M3 Ultra 256GB):**
- Qwen3.5-35B-A3B: 83 tok/s
- Qwen3.5-122B: 44 tok/s with 100% tool reliability
- Phi-4 Mini 14B: 180 tok/s (2.3x faster than mlx-lm)

**Limitations:** Vision dependency overhead (~2.5GB extra torch+torchvision), active development, relatively new project.

### Option D: Use Ollama 0.19 (MLX backend preview)

Ollama [switched to MLX backend](https://ollama.com/blog/mlx) on March 30, 2026:
- Prefill ~1.6x faster, decode ~2x faster vs Ollama 0.18
- Benchmarks: 1,810 tok/s prefill, 112 tok/s decode (int4)
- M5 chips additionally leverage GPU Neural Accelerators

**Critical limitation:** Preview requires >32GB unified memory. Currently only supports Qwen3.5 architecture. Gemma 4 coming in Ollama 0.20. Not suitable for 32GB M4 Mac yet.

### Option E: LM Studio Prefill Patch

[Community patch](https://github.com/thornad/lmstudio-mlx-patch) for LM Studio's MLX engine:
- Change `PROMPT_PROCESSING_CHUNK_SIZE` from 512 to 8192
- Up to 1.5x faster prefill
- Modify `mlx_engine/cache_wrapper.py` and `mlx_lm/generate.py`
- Optimal value depends on RAM: 8192 for 32GB, 4096 for 16GB
- At 16384, performance regresses due to memory allocation overhead

### Option F: Use vllm-mlx

[vllm-mlx](https://github.com/waybarrios/vllm-mlx) brings vLLM's production features to Apple Silicon:
- 400+ tok/s, PagedAttention scaling
- Continuous batching, MCP tool calling
- Prefix caching that cuts multimodal latency from 21s to <1s
- Works with Claude Code

### Option G: Model-specific workarounds

- **Avoid vision variants** of Qwen3.5 in LM Studio (broken prompt caching specifically for multimodal)
- **Use fp16 models on M1/M2** instead of bf16 (or use GGUF which defaults to fp16)
- **For Gemma 4 MLX quantization:** Use [FakeRocket543/mlx-gemma4](https://github.com/FakeRocket543/mlx-gemma4) for PLE-safe quantization that handles ScaledLinear layers correctly

---

## 3. Which Models Actually Work Well in MLX Right Now?

### Working Well

| Model | Status | Notes |
|-------|--------|-------|
| **MiniMax M2.5** | Fully working | Only major model with working prefix caching. Pure full attention. |
| **Gemma 4 26B-A4B** | Working (with caveats) | Day-1 MLX support. 4-bit available. ~17GB VRAM. Needs FakeRocket patch for proper quantization. LM Studio may error on "gemma4" model type. |
| **Gemma 4 31B Dense** | Working | Pure attention, no hybrid issues. Memory-hungry at 32GB. |
| **GLM-4.7 Flash** | Working | Dense model, 4-bit and 8-bit on HuggingFace. Not reliable for multi-step tool calling. |
| **Phi-4 Mini 14B** | Working well | 180 tok/s on M3 Ultra via Rapid-MLX. No hybrid architecture. |

### Broken / Problematic

| Model | Status | Issue |
|-------|--------|-------|
| **Qwen3.5-35B-A3B** | Broken caching, tool-call degradation | All hybrid architecture bugs apply. Prefill 10-100x slower than expected. Tool calls break at 4-bit after ~5 rounds, at 8-bit after ~13 rounds. Malformed XML at ~20k tokens. |
| **Qwen3.5-9B** | Broken caching | Same hybrid DeltaNet architecture. Prefix cache shows ZERO reuse, actually gets SLOWER across requests. |
| **Qwen3.5-27B** | Broken caching | Same hybrid architecture. |
| **ALL Qwen3.5 variants** | Broken caching | The entire family (0.8B-397B) uses Gated DeltaNet hybrid. No pure-attention variants exist. |
| **Gemma 3 (all sizes)** | Broken prefix reuse | 5:1 sliding + global attention pattern. RotatingKVCache not trimmable. |
| **Llama 4 Scout/Maverick** | Broken prefix reuse | iRoPE chunked + NoPE attention. |

### Fit on 32GB M4

For 32GB unified memory (keep model weights <60%, leaving 40% for KV cache):
- Gemma 4 26B-A4B 4-bit: ~8-10GB (fits comfortably, MoE with 4B active)
- GLM-4.7 Flash 4-bit: ~4-5GB (fits easily)
- Qwen3.5-35B-A3B 4-bit: ~18-20GB (tight, Qwen team says "more than 32GB" recommended)
- Qwen3.5-9B 4-bit: ~5-6GB (fits, but caching broken)
- Dense 27B models at 4-bit: ~15-17GB (fits)

---

## 4. Recommended MLX Setup for April 2026

### If you MUST use MLX:

1. **mlx-lm version:** 0.31.2 (latest, April 7). Includes fixes for GatedDeltaNet memory leaks and cache checkpoint issues. But core hybrid caching is still broken.

2. **Best engine choice by use case:**
   - **Agentic/tool-calling with Qwen3.5:** Use Rapid-MLX (DeltaNet state snapshots)
   - **Long-context coding sessions:** Use oMLX (SSD-tiered KV cache)
   - **Quick interactive chat:** Use Ollama 0.19 (if >32GB RAM)
   - **High-throughput parallel:** Use vllm-mlx

3. **LM Studio settings if using MLX:**
   - Increase prefill chunk size to 4096-8192 via [community patch](https://github.com/thornad/lmstudio-mlx-patch)
   - Avoid vision model variants for text-only tasks
   - Use fp16 models on M1/M2 (not bf16)

4. **Model recommendations for 32GB M4 + MLX:**
   - **Best overall:** Gemma 4 26B-A4B 4-bit (working cache, MoE efficiency, good quality)
   - **Fastest small:** Phi-4 Mini 14B
   - **Avoid:** Qwen3.5 family on MLX (use GGUF instead)

### If you want reliability (recommended):

**Use GGUF via llama.cpp** for:
- Qwen3.5 models (all sizes)
- Any agentic workflow with growing context
- Tool calling reliability
- Short-output tasks (classification, RAG, function calling)

**Use MLX for:**
- Long-output generation (summaries, creative writing, code generation)
- Models with pure full attention (MiniMax M2.5, Gemma 4 dense)
- Fresh conversations with short prompts

---

## 5. Community Sentiment

### The consensus (April 2026):

**"MLX is not as mature as GGUF yet and needs a bit more time."**

Key friction points cited by r/LocalLLaMA and the broader community:
1. **Model availability lag:** GGUF appears same-day on HuggingFace; MLX conversions can take hours to days
2. **Caching is the killer issue:** Without working prefix caching, MLX's faster decode speed is meaningless for agentic workflows
3. **UI-reported speed is misleading:** The "57 tok/s" number measures only decode; effective throughput can be 3 tok/s at 8.5K context
4. **New model support is fragile:** Each new architecture (Gemma 4, Qwen3.5) requires specific fixes

### Who uses what:

- **Daily driver / production:** Most fall back to GGUF via Ollama or LM Studio's llama.cpp backend
- **Enthusiasts / benchmarking:** MLX via Rapid-MLX or oMLX
- **Mac-only dev stack with peak throughput priority:** MLX (when model architecture is compatible)
- **Cross-platform teams:** Always llama.cpp/GGUF for portability

### The Ollama MLX shift:

Ollama 0.19's switch to MLX backend is significant but limited:
- Only supports Qwen3.5 architecture currently
- Requires >32GB RAM (excludes most consumer Macs)
- Gemma 4 support coming in 0.20
- Signals long-term industry direction toward MLX on Apple Silicon

---

## 6. The Prefill Problem: Detailed Analysis

### Why MLX prefill is slow:

1. **Broken prefix caching** for hybrid architectures (the dominant factor)
2. **Small default chunk size** (512 in LM Studio, should be 4096-8192)
3. **bf16 on M1/M2** without native hardware support
4. **Younger batch processing code** compared to llama.cpp's mature FlashAttention

### Available fixes:

| Solution | Prefill Improvement | Works for Qwen3.5? | Status |
|----------|--------------------|--------------------|--------|
| **oMLX tiered KV cache** | 28x (49s -> 1.7s at 8K) | Partial (SSD caching helps, but DeltaNet state still non-trimmable) | Available now |
| **Rapid-MLX DeltaNet snapshots** | 2-4x TTFT | Yes (first solution for non-trimmable architectures) | Available now |
| **LM Studio chunk size patch** | 1.5x | Yes (helps prefill compute, not caching) | Manual patch |
| **Ollama 0.19 MLX** | 1.6x prefill | Qwen3.5 only currently | Preview, >32GB |
| **GGUF via llama.cpp** | Better prefill + working cache | Yes (full FlashAttention + prompt caching) | Mature, recommended |
| **vllm-mlx PagedAttention** | Significant at scale | Architecture-dependent | Available |

### The honest answer:

For a 32GB M4 Mac running Qwen3.5 with growing context (agentic workflows):

1. **Best option today:** GGUF Q4_K_XL via llama.cpp (or Ollama's llama.cpp backend). Working cache, working tool calls, proven reliability.
2. **Best MLX option:** Rapid-MLX with DeltaNet state snapshots. Only engine that directly addresses the hybrid caching problem.
3. **If you can switch models:** Gemma 4 26B-A4B on MLX works well -- pure MoE without the DeltaNet/Mamba complications.

---

## Sources

### GitHub Issues (mlx-lm)
- [#903 - Caching doesn't work for Qwen3.5](https://github.com/ml-explore/mlx-lm/issues/903) (OPEN)
- [#932 - 2.7x DeltaNet decode slowdown](https://github.com/ml-explore/mlx-lm/issues/932) (FIXED)
- [#965 - KV cache cross-contamination](https://github.com/ml-explore/mlx-lm/issues/965) (FIXED)
- [#975 - Strange cache behavior 0.31.0](https://github.com/ml-explore/mlx-lm/issues/975) (CLOSED, reportedly still broken)
- [#980 - Prefix cache broken for ALL hybrid models](https://github.com/ml-explore/mlx-lm/issues/980) (OPEN)
- [#1011 - Tool-calling degradation 4-bit/8-bit](https://github.com/ml-explore/mlx-lm/issues/1011) (CLOSED, disputed)
- [#1061 - Malformed tool-call at 20k tokens](https://github.com/ml-explore/mlx-lm/issues/1061) (OPEN)

### GitHub PRs (mlx-lm)
- [#923 - Hybrid cache for Qwen3.5](https://github.com/ml-explore/mlx-lm/pulls/923) (CLOSED, not merged)
- [#976 - Fix late binding cache checkpoint](https://github.com/ml-explore/mlx-lm/pulls/976) (MERGED Mar 10)
- [#1066 - Fix gated delta kernel precision](https://github.com/ml-explore/mlx-lm/pulls/1066) (OPEN, draft)

### Alternative Engines
- [oMLX - Tiered KV cache server](https://github.com/jundot/omlx)
- [Rapid-MLX - DeltaNet state snapshots](https://github.com/raullenchai/Rapid-MLX)
- [vllm-mlx - PagedAttention for Apple Silicon](https://github.com/waybarrios/vllm-mlx)
- [LM Studio prefill patch](https://github.com/thornad/lmstudio-mlx-patch)
- [FakeRocket543/mlx-gemma4 - PLE-safe Gemma 4 quantization](https://github.com/FakeRocket543/mlx-gemma4)

### Analysis & Benchmarks
- [57 tok/s on Screen, 3 tok/s in Practice (famstack.dev)](https://famstack.dev/guides/mlx-vs-gguf-apple-silicon/)
- [The Stack: Apple Silicon Local LLM Servers (Stochastic Sandbox)](https://stochasticsandbox.com/posts/the-stack-apple-silicon-local-agents-2026-03-28/)
- [2026 Mac Inference Framework Selection (MACGPU)](https://macgpu.com/en/blog/2026-mac-inference-framework-vllm-mlx-ollama-llamacpp-benchmark.html)
- [Ollama MLX Backend Announcement](https://ollama.com/blog/mlx)
- [Qwen3.5: Nobody Agrees on Attention Anymore (HuggingFace blog)](https://huggingface.co/blog/mlabonne/qwen35)
- [mlx-lm releases](https://github.com/ml-explore/mlx-lm/releases)
- [mlx-lm PyPI](https://pypi.org/project/mlx-lm/)

### Community Discussions
- [Qwen3.5 MLX vs GGUF (AI Navigate)](https://ai-navigate-news.com/en/articles/5b0c98fe-c200-4748-9c35-b0f8c5373359)
- [The Missing Piece in Apple Silicon LLM Inference (Medium)](https://medium.com/@alexandru_vasile/the-missing-piece-in-apple-silicon-llm-inference-nobody-talks-about-0236a12929d4)
- [local-llm-bench benchmarking tool](https://github.com/famstack-dev/local-llm-bench)
