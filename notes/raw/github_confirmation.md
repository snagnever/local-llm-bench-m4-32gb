# GitHub Confirmation Research - April 8, 2026

Third research pass: GitHub issues, discussions, repos confirming bugs, new tools, and performance claims for local LLM benchmarking.

---

## 1. LM Studio Bugs

### Gemma 4 Support Status
- **GGUF loading fixed**: Issue #1728 - Gemma 4 GGUF models (31B & 26B) failed with "unknown model architecture: 'gemma4'" on LM Studio 0.4.8+1. **Root cause**: bundled llama.cpp runtime 2.8.0 lacked Gemma 4 support. **Fix**: Update to llama.cpp runtime 2.10.0+. Issue closed April 3, 2026 as COMPLETED.
  - Source: https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1728
- **MLX engine lacks Gemma 4**: Issue #301 on mlx-engine and Issue #1741 on bug tracker - "Model type gemma4 not supported" error. The MLX backend (mlx_vlm) bundled with LM Studio does not yet support the gemma4 architecture. **Still open as of April 8.**
  - Source: https://github.com/lmstudio-ai/mlx-engine/issues/301
  - Source: https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1741
- **Memory growth on M4 Max**: Issue #1750 - Gemma 4 31B Q8_0 causes memory usage to grow to 221.5GB on M4 Max 64GB. System estimated 108.45GB needed (34.85GB model + 73.60GB context), auto-downscaled context from 262K to 8K tokens. **Still open, no developer response.**
  - Source: https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1750
- **Reasoning not natively handled**: Issue #1743 - LM Studio doesn't handle Gemma 4's reasoning implementation.
  - Source: https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1743

### Qwen3.5 MoE Support Status
- **MoE architecture not recognized (MLX)**: Issue #1723 - `qwen3_5_moe` architecture not detected by model scanner. Models don't appear in model list despite bundled mlx_lm supporting the architecture.
  - Source: https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1723
- **Qwen3.5 MLX engine error**: Issue #284 - "Model type qwen3_5 not supported" error in mlx-engine.
  - Source: https://github.com/lmstudio-ai/mlx-engine/issues/284
- **Speculative decoding not working**: Issue #1597 - Qwen3.5 dense models cannot be used as draft models for Qwen3.5 MoE speculative decoding.
  - Source: https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1597
- **Reasoning toggle missing**: Issue #1613 - Missing reasoning toggle and API support for Qwen 3.5-4B GGUF models.
  - Source: https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1613

**KEY TAKEAWAY**: LM Studio GGUF backend now works for Gemma 4 (after runtime update), but MLX support for both Gemma 4 and Qwen3.5 MoE is still broken/unsupported as of April 8, 2026.

---

## 2. llama.cpp Status

### Gemma 4 Support
- **Architecture merged**: PR #21309 by ngxson added Gemma 4 support (vision + MoE, no audio). Gemma 4 includes ISWA dual-cache, variable head_dim (256 for SWA / 512 for global), MoE with 128 experts top-8 plus shared expert.
  - Source: https://github.com/ggml-org/llama.cpp/pull/21309
- **Tensor shape mismatch FIXED**: Issue #21434 - `sliding_window_pattern` was read as uint32_t instead of bool, causing tensor shape errors. **Fix merged April 5, 2026** (PR #21428).
  - Source: https://github.com/ggml-org/llama.cpp/issues/21434
- **Tokenizer fixes merged**: PR #21343 (tokenizer fix) and PR #21418 (specialized Gemma 4 parser).
  - Source: https://github.com/ggml-org/llama.cpp/pull/21343
  - Source: https://github.com/ggml-org/llama.cpp/pull/21418
- **Generates garbage tokens**: Issue #21321 - Gemma 4 generates `<unused24>` tokens.
  - Source: https://github.com/ggml-org/llama.cpp/issues/21321
- **Tool calling infinite loop**: Issue #21375 - Infinite repetition loop with peg-gemma4 parser during tool calls.
  - Source: https://github.com/ggml-org/llama.cpp/issues/21375
- **Audio not supported**: Issue #21325 - Gemma 4 audio support is missing in llama.cpp.
  - Source: https://github.com/ggml-org/llama.cpp/issues/21325

### Qwen3.5 MoE Support
- **Architecture merged Feb 10, 2026**: PR #19435 and #19468 added Qwen3.5 dense and MoE support.
  - Source: https://github.com/ggml-org/llama.cpp/pull/19435
- **Perplexity degradation with f16 KV**: Issue #20035 - Default f16 KV cache causes measurable perplexity degradation because Qwen3.5 is trained in bfloat16. Clipping of dynamic range occurs.
  - Source: https://github.com/ggml-org/llama.cpp/issues/20035
- **Speculative decoding fix for hybrid SSM/MoE**: Commit 9a04ac4 fixes speculative decoding that was broken on Qwen3.5 MoE hybrid architecture.
  - Source: https://github.com/ggml-org/llama.cpp/actions/runs/22712623978
- **MTP support for dense Qwen3.5**: PR #20700 adds Multi-Token Prediction support for dense Qwen3.5 (0.8B-27B) with FastMTP vocabulary trimming. Achieves ~28 tok/s on RTX 5060 Ti.
  - Source: https://github.com/ggml-org/llama.cpp/pull/20700
- **Slower with fused gate+up**: Issue #20492 - Performance regression with `--fit on` when fused gate+up is used in Qwen3.5 35B A3B.
  - Source: https://github.com/ggml-org/llama.cpp/issues/20492

### Apple Silicon / Metal Optimizations
- **Metal4 Tensor API support MERGED**: PR #16634 by ggerganov - Initial Metal4 tensor API support targeting M5 chips. Uses Apple's MetalPerformancePrimitives (MPP) framework. **Merged Nov 6, 2025.**
  - M5 results: ~2x speedup on some workloads, ~2.5x on prompt processing with large models
  - Qwen3 testing: 4936 t/s (Metal4) vs 3073 t/s (master) on pp512
  - M4 confirmed no regressions - backward compatible
  - Source: https://github.com/ggml-org/llama.cpp/pull/16634
- **Activation rotation for better quantization MERGED**: PR #21038 by ggerganov - Walsh-Hadamard transform rotates Q, K, V activations before KV cache quantization. **Merged April 1, 2026.**
  - Qwen3 0.6B Q4_0: perplexity 62.0 -> 46.3
  - gpt-oss-20b AIME25: Q4_0 2.0% -> 21.7% with rotation
  - ~8-12% generation slowdown due to extra matrix ops
  - Source: https://github.com/ggml-org/llama.cpp/pull/21038
- **TurboQuant KV cache**: Discussion #20969 - Implementation of Google's TurboQuant (ICLR 2026) for 3.5-bit KV cache with ~4.6x compression vs FP16.
  - Source: https://github.com/ggml-org/llama.cpp/discussions/20969
- **TurboQuant-MoE**: Discussion #21135 - 8.53x compression with 100% needle recall at 128k context. Features NashMoE router, Markov trajectory predictor, PID VRAM controller. Adaptive bitwidth 1-4 bits averaging 2.1 bits/token.
  - Source: https://github.com/ggml-org/llama.cpp/discussions/21135

**KEY TAKEAWAY**: llama.cpp has basic Gemma 4 and Qwen3.5 support merged but still has active bugs (tokenizer, tool calling, perplexity). Major quantization improvements (activation rotation, TurboQuant) are landing. Metal4 API for M5 is merged with impressive speedups; M4 maintains backward compatibility.

---

## 3. MLX / mlx-lm Status

### Qwen3.5 Issues
- **Caching completely broken**: Issue #903 - KV caching fails for Qwen3.5; cached tokens counter always 0. Root cause: Mamba/DeltaNet cache cannot be trimmed, and reasoning tokens cause prompt misalignment. Users report 10-15s operations extending to 1m46s. **Multiple PRs (#906, #923) in progress.**
  - Source: https://github.com/ml-explore/mlx-lm/issues/903
- **Tool calling degrades at scale**: Issue #1011 - Qwen3.5-35B-A3B with mlx-community 4-bit/8-bit: after ~5 rounds (4-bit) or ~13 rounds (8-bit), model emits tool calls as plain text. GGUF Q4_K_XL does NOT have this problem.
  - Source: https://github.com/ml-explore/mlx-lm/issues/1011
- **Malformed tool calls at 20k tokens**: Issue #1061 - Qwen3.5-35B-A3B-4bit emits malformed tool-call output around 20k prompt tokens.
  - Source: https://github.com/ml-explore/mlx-lm/issues/1061
- **DeltaNet decode slowdown**: Issue #932 (CLOSED) - Gated DeltaNet layers (30 of 40 layers in Qwen3.5-35B-A3B) cause 2.7x slowdown when input_embeddings aren't exact vocabulary vectors (93 tok/s -> 34 tok/s). Fix: cast back to original dtype.
  - Source: https://github.com/ml-explore/mlx-lm/issues/932
- **KV cache cross-contamination**: Issue #965 - Cross-contamination between concurrent requests in mlx_lm.server.
  - Source: https://github.com/ml-explore/mlx-lm/issues/965

### Gemma 4 Status
- Gemma 4 MLX support is available via community efforts (mlx-community models on HuggingFace). Working quantized weights with trimodal validation have been published.
- LM Studio's bundled mlx-engine does NOT yet support gemma4 architecture.

**KEY TAKEAWAY**: mlx-lm has significant issues with Qwen3.5 caching (Mamba/DeltaNet hybrid makes caching very hard), tool calling degradation at scale, and concurrent request handling. DeltaNet slowdown was fixed. Gemma 4 MLX works via community but not in LM Studio yet.

---

## 4. Ollama MLX Backend Status

### Recent Releases
- **v0.20.3** (April 7, 2026) - Latest stable. Gemma 4 tool calling enhancements.
- **v0.20.4-rc2** (April 7, 2026) - Pre-release with M5 performance improvements, flash attention for Gemma4.
- **v0.20.0** (April 2, 2026) - Introduced Gemma 4 models (E2B, E4B, 26B MoE, 31B dense).
- **v0.19.0** (March 27, 2026) - **Major release**: MLX backend on Apple Silicon. Built on Apple's MLX framework for unified memory. Includes MLX periodic snapshot creation during prompt processing, KV cache memory leak fixes.
  - Source: https://github.com/ollama/ollama/releases/tag/v0.19.0

### Gemma 4 + Apple Silicon Issues
- **Flash Attention hangs**: Issue #15368 (M5 Max 128GB) - Gemma 4 31B hangs indefinitely with FA=1 and prompts >500 tokens. Root cause: hybrid attention (50 SWA + 10 global layers) with different head dimensions (256 vs 512) that FA implementation can't handle. **PR #15378 merged to fix.**
  - Without FA: only ~15 tok/s on 31B Dense (M5 Max!)
  - 26B MoE achieves ~75 tok/s
  - Source: https://github.com/ollama/ollama/issues/15368
- **MLX runner lacks Gemma 4**: Falls back to llama.cpp/GGUF runner. PR #15244 in progress for MLX support.
- **Streaming reasoning field bug**: /v1/chat/completions streaming puts content in `reasoning` field with empty `content` field. Non-streaming works fine.
- **Homebrew MLX library warnings**: Issue #14350 - MLX dynamic library not available on Homebrew installs.
  - Source: https://github.com/ollama/ollama/issues/14350

**KEY TAKEAWAY**: Ollama v0.19+ has MLX backend but Gemma 4 MLX support is still in progress (PR #15244). Flash attention fix for Gemma 4 hybrid attention was merged. Performance without FA is poor (15 tok/s on M5 Max for 31B dense). MoE variant (26B-A4B) performs much better at 75 tok/s.

---

## 5. lm-evaluation-harness (EleutherAI)

### local-chat-completions Model Type
- Registered as both "openai-chat-completions" and "local-chat-completions"
- Works with any OpenAI API-compatible local server (Ollama, LM Studio, vllm-mlx, etc.)
- **Critical limitation**: Loglikelihood and MCQ-based tasks (MMLU, HellaSwag, ARC via logprobs) are **NOT supported** with chat-completion endpoints. Only generative tasks work.
  - Source: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/API_guide.md
- Example: `lm_eval --model local-chat-completions --model_args model=meta-llama/Meta-Llama-3.1-8B-Instruct,num_concurrent=10 --apply_chat_template --tasks gsm8k`
  - Source: https://github.com/EleutherAI/lm-evaluation-harness/pull/1174

### Apple Silicon / MPS Support
- MPS backend supported: replace `--device cuda:0` with `--device mps` (requires PyTorch 2.1+)
- **Warning**: MPS backend still in early stages; correctness issues may exist. Recommended to verify model forward pass on `--device cpu` vs `--device mps` match.
- No specific Apple Silicon optimizations or known issues found for 2026.

### Known Issues with Local APIs
- Issue #2934: Confusion about how to evaluate MMLU via local-chat-completions (answer: you can't for MCQ/logprob tasks).
- Issue #3125: Errors running on locally hosted servers - configuration issues.
  - Source: https://github.com/EleutherAI/lm-evaluation-harness/issues/3125

**KEY TAKEAWAY**: lm-evaluation-harness works with local APIs for generative tasks only. For MCQ benchmarks (MMLU, HellaSwag, ARC), you need direct model access (not API). MPS backend works but may have correctness issues.

---

## 6. New/Emerging Tools

### Rapid-MLX
- **Repo**: https://github.com/raullenchai/Rapid-MLX
- **What**: OpenAI-compatible local inference server optimized for Apple Silicon, drop-in backend for Cline
- **Performance vs Ollama** (M3 Ultra 192GB benchmarks):
  - Qwen 3.5-4B: 95 tok/s vs Ollama's 38 tok/s (2.5x)
  - Qwen 3.5-9B: 66 tok/s vs Ollama's 16 tok/s (4.2x)
  - Devstral-24B: 31 tok/s vs Ollama's 13 tok/s (2.4x)
- **Key feature**: Sub-100ms TTFT via DeltaNet state snapshots
- **Install**: `pip install vllm-mlx rapid-mlx`
- **Supports**: Tool calling (Hermes, Minimax, GLM formats), reasoning models (Qwen3.5, QwQ)
  - Source: https://github.com/cline/cline/discussions/9940

### oMLX
- **Repo**: https://github.com/jundot/omlx
- **What**: MLX-based inference server with native macOS menu bar app. Tiered KV caching (hot RAM + cold SSD).
- **Key innovation**: Persists KV cache across hot (RAM) and cold (SSD) tiers. TTFT drops from 30-90s to 1-3s on long contexts.
- **Features**: Multi-model serving, VLM support, embeddings, rerankers, tool calling, structured output, Admin dashboard UI
- **API**: OpenAI + Anthropic compatible endpoints
- **Install**: Download .dmg from releases or `brew install omlx`
- **Activity**: 791 commits, actively maintained as of April 2026
  - Source: https://github.com/jundot/omlx

### vllm-mlx
- **Repo**: https://github.com/waybarrios/vllm-mlx
- **What**: OpenAI + Anthropic compatible server for Apple Silicon with continuous batching, MCP tool calling, multimodal
- **Performance**: Claims 400+ tok/s (native MLX backend)
- **Latest**: v0.2.6 (Feb 13, 2026)
- **Features**: LLMs, VLMs, audio (mlx-audio), embeddings (mlx-embeddings)
  - Source: https://github.com/waybarrios/vllm-mlx

### llama-swap
- **Repo**: https://github.com/mostlygeek/llama-swap
- **What**: Go-based proxy for dynamic model switching across local inference servers
- **How it works**: Intercepts API requests, extracts model identifier, loads correct backend config
- **Features**: OpenAI + Anthropic API support, web UI, API key auth, TTL auto-unload, groups for parallel models
- **Install**: Homebrew, WinGet, Docker, pre-built binaries
- **Zero dependencies** (Go binary)
  - Source: https://github.com/mostlygeek/llama-swap

### local-llm-bench
- **Repo**: https://github.com/famstack-dev/local-llm-bench
- **What**: Scenario-based benchmark for Apple Silicon measuring effective throughput (not just gen tok/s)
- **Formula**: `effective tok/s = output_tokens / (prefill_time + generation_time)`
- **Backends**: Ollama, LM Studio, oMLX, llama-server, MiniMax, OpenAI-compatible
- **Scenarios**: ops-agent (8-turn), doc-summary (5 single-shot), prefill-test (4 turns), creative-writing (3 turns)
- **Key finding**: At 8K context, LM Studio MLX takes 49s prefill; oMLX takes 1.7s
- **M4 results**: Not yet available, contributors invited
  - Source: https://github.com/famstack-dev/local-llm-bench

### MLXBench
- **Repo**: https://github.com/linusvwe/MLXBench
- **What**: Benchmark comparing Ollama vs vLLM-MLX vs Docker Model Runner on Apple Silicon
- **Metrics**: TTFT, tok/s throughput, e2e latency, inter-token latency, memory (RSS)
- **Concurrency testing**: levels 1, 2, 4, 8
- **Output**: Rich tables, JSON export, PNG bar charts
- **Activity**: 19 commits, last March 29, 2026
  - Source: https://github.com/linusvwe/MLXBench

### mlx_transformers_benchmark
- **Repo**: https://github.com/aukejw/mlx_transformers_benchmark
- **What**: Benchmarking LLMs on Apple Silicon with results page at https://aukejw.github.io/mlx_transformers_benchmark/
- **Note**: Default parameters don't cause thermal throttling on M4 Pro
  - Source: https://github.com/aukejw/mlx_transformers_benchmark

### benchmarks_llm_silicon
- **Repo**: https://github.com/ivanfioravanti/benchmarks_llm_silicon
- **What**: Collection of benchmarks on Apple Silicon (M2, M3 Ultra, M4 Max) using MLX, LM Studio, llama.cpp, Ollama
  - Source: https://github.com/ivanfioravanti/benchmarks_llm_silicon

---

## 7. APEX Quantization

### Overview
- **Repo**: https://github.com/mudler/apex-quant
- **Creator**: mudler / LocalAI team (creators of LocalAI)
- **License**: MIT
- **What**: MoE-aware mixed-precision quantization for GGUF models

### How It Works
1. **Tensor Classification**: Categorizes tensors by function (routed experts, shared experts, attention). Routed experts tolerate aggressive quantization because only 8 of 256 experts activate per token.
2. **Layer-wise Gradients**: Edge layers (first/last 5 of 40) get higher precision (embedding alignment, logit generation). Middle layers use lower precision.
3. **Diverse Calibration**: "I-variants" use training data spanning chat, code, reasoning, tool-calling (avoiding Wikipedia bias).

### Performance Results (Qwen3.5-35B-A3B)
| Config | Size | Perplexity | HellaSwag |
|--------|------|------------|-----------|
| I-Quality | 21.3 GB | 6.552 | 83.5% |
| I-Balanced | 23.6 GB | Best KL divergence | - |
| I-Compact | 16.1 GB | - | Fits 16GB GPU |
| Mini | 12.2 GB | Beats IQ2_M | Consumer VRAM |

Speed: 60-74.4 tok/s across configurations.

### Gemma 4 Results (April 2026)
- APEX Balanced (18.1 GB) perplexity: 316.4 vs Q8_0's 337.3 (lower = better)
- Winogrande: 54.0% (beats all baselines)
- Q8_0-level quality at 38% less model size

### Usage
- `./scripts/quantize.sh --i-quality model-f16.gguf output.gguf`
- Works with stock llama.cpp, no code changes needed
- Pre-quantized models on HuggingFace: https://huggingface.co/collections/mudler/apex-quants-gguf

### Available Models
- mudler/Qwen3.5-35B-A3B-APEX-GGUF
- mudler/gemma-4-26B-A4B-it-APEX-GGUF
- mudler/Qwen3-Coder-Next-APEX-GGUF
- mudler/Step-3.5-Flash-APEX-GGUF
- mudler/Qwen3.5-35B-A3B-APEX-TQ-GGUF (APEX + TurboQuant KV)

**KEY TAKEAWAY**: APEX is a legitimate, well-benchmarked MoE-aware quantization tool by the LocalAI team. Works with stock llama.cpp. Pre-quantized models available on HuggingFace. Best option for MoE models where uniform quantization wastes precision on inactive experts.

---

## 8. Benchmark Results: Gemma 4 + Qwen 3.5 on Apple Silicon

### M5 Max 128GB Results (from Incept5/gemma4-benchmark)
MLX 0.31.1, mlx-vlm 0.4.4, 4-bit quantization, 4k context:

| Model | Active Params | Decode | Memory |
|-------|---------------|--------|--------|
| Gemma 4 E2B | ~2B MoE | 205 tok/s | 4.7 GB |
| Gemma 4 E4B | ~4B MoE | 127 tok/s | 6.4 GB |
| Gemma 4 26B-A4B | ~4B active / 26B total MoE | 113 tok/s | 17.1 GB |
| Gemma 4 31B | 31B Dense | 27 tok/s | 22.7 GB |
| Qwen 3.5 2B | 2B Dense | 308 tok/s | 3.3 GB |
| Qwen 3.5 4B | 4B Dense | 156 tok/s | 5.2 GB |

### 256k Context Performance (4-bit, M5 Max)
| Model | Decode at 256k |
|-------|---------------|
| Gemma 4 E2B | 78 tok/s |
| Gemma 4 E4B | 27 tok/s |
| Gemma 4 26B-A4B | 30 tok/s |
| Gemma 4 31B | 7 tok/s |
| Qwen 3.5 2B | 61 tok/s |
| Qwen 3.5 4B | 29 tok/s |

### TurboQuant KV Cache Impact
- Negligible below 16k tokens
- +5-10% gains at 32-64k range
- +15-19% at 128-256k on larger models
- Minimal overhead

Source: https://github.com/Incept5/gemma4-benchmark

### Ollama Performance (M5 Max, from Issue #15368)
- Gemma 4 26B MoE: ~75 tok/s
- Gemma 4 31B Dense: ~15 tok/s (without Flash Attention)

### local-llm-bench Results (M3 Max)
- Qwen3.5-35B-A3B with oMLX: 71.3 effective tok/s (ops-agent scenario)
- Generation speeds: 55-93 tok/s across MLX engines by hardware
- **Prefill critical**: LM Studio MLX = 49s at 8K context; oMLX = 1.7s

---

## 9. Key Confirmed Bugs & Gotchas Summary

| Issue | Status | Impact | Workaround |
|-------|--------|--------|------------|
| LM Studio GGUF Gemma 4 load fail | FIXED | Update llama.cpp runtime to 2.10+ | Update runtime |
| LM Studio MLX Gemma 4 not supported | OPEN | Cannot use MLX for Gemma 4 | Use GGUF backend |
| LM Studio MLX Qwen3.5 MoE not recognized | OPEN | MoE models hidden in model list | Manual path loading? |
| llama.cpp Gemma 4 sliding_window_pattern | FIXED | Tensor shape mismatch on load | Use patched build |
| llama.cpp Gemma 4 generates garbage tokens | OPEN | Produces <unused24> tokens | - |
| llama.cpp Gemma 4 tool call infinite loop | OPEN | Server hangs during tool calls | Avoid tool calling |
| llama.cpp Qwen3.5 perplexity with f16 KV | OPEN | Quality degradation | Use bf16 or rotated quantized KV |
| mlx-lm Qwen3.5 caching broken | OPEN | 10-100x slower operations | PRs in progress |
| mlx-lm Qwen3.5 tool calling degrades | OPEN | Breaks after ~5 rounds (4-bit) | Use GGUF Q4_K_XL |
| mlx-lm KV cache cross-contamination | OPEN | Concurrent requests corrupt | Single request only |
| Ollama FA hang with Gemma 4 | FIXED | Hangs on prompts >500 tokens | PR #15378 merged |
| Ollama MLX lacks Gemma 4 | IN PROGRESS | Falls back to GGUF (slower) | PR #15244 |
| Ollama streaming reasoning field | OPEN | /v1 streaming puts content in wrong field | Use /api/chat |
| lm-eval-harness MCQ via API | BY DESIGN | Cannot eval MMLU/HellaSwag via API | Need direct model access |

---

## 10. Tool Comparison Matrix for Local Benchmarking

| Tool | Type | Apple Silicon Optimized | MLX Support | Gemma 4 | Qwen3.5 MoE | API Compatible |
|------|------|----------------------|-------------|---------|-------------|----------------|
| Ollama v0.20.3 | Runtime | Yes (MLX backend) | Yes (v0.19+) | GGUF only | Yes | OpenAI + Anthropic |
| LM Studio | Runtime | Yes | Partial (bugs) | GGUF only | GGUF only | OpenAI |
| llama.cpp (llama-server) | Runtime | Yes (Metal) | No | Yes (bugs) | Yes (bugs) | OpenAI |
| oMLX | Runtime | Yes (native MLX) | Yes | Via mlx-lm | Via mlx-lm | OpenAI + Anthropic |
| vllm-mlx | Runtime | Yes (native MLX) | Yes | Via mlx-lm | Via mlx-lm | OpenAI + Anthropic |
| Rapid-MLX | Runtime | Yes (vllm-mlx based) | Yes | Unknown | Yes | OpenAI |
| llama-swap | Proxy/Router | N/A | N/A | N/A | N/A | OpenAI + Anthropic |
| lm-eval-harness | Benchmark | MPS (limited) | No | Via API | Via API | local-chat-completions |
| local-llm-bench | Benchmark | Yes (designed for) | Via backends | Via backends | Via backends | OpenAI |
| MLXBench | Benchmark | Yes | Via backends | Via backends | Via backends | OpenAI |
| APEX-quant | Quantization | N/A | No (GGUF) | Yes | Yes | N/A |

---

## 11. Recommendations Based on Findings

### For Gemma 4 on M4 Max:
1. **Best option now**: Ollama v0.20.3+ with GGUF (FA fix merged, but watch for remaining bugs)
2. **MLX option**: Use oMLX or vllm-mlx with mlx-community weights (not LM Studio MLX)
3. **Avoid**: LM Studio MLX (gemma4 not supported), llama.cpp tool calling (infinite loop)

### For Qwen3.5 MoE on M4 Max:
1. **Best option now**: llama.cpp via Ollama with GGUF (APEX quantized for best quality/size)
2. **MLX caution**: Caching is severely broken, tool calling degrades after ~5 rounds at 4-bit
3. **GGUF Q4_K_XL** does NOT have the tool calling degradation that MLX 4-bit has

### For Benchmarking:
1. **lm-evaluation-harness** with `local-chat-completions` for generative tasks only
2. **local-llm-bench** for real-world scenario comparison across engines
3. **MLXBench** for TTFT/throughput/latency comparison at different concurrency levels
4. **Cannot use lm-eval-harness for MMLU/HellaSwag/ARC via local API** (need logprobs)

### For Quantization:
1. **APEX** for MoE models (Qwen3.5-35B-A3B, Gemma 4 26B-A4B) - best quality/size ratio
2. **Activation rotation** (merged in llama.cpp) dramatically improves Q4_0 KV cache quality
3. **TurboQuant** for KV cache compression at long contexts (128k+)
