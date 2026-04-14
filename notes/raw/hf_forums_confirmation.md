# HuggingFace Forums & AI Community Confirmation Research
## Third Research Pass - April 8, 2026

---

## 1. HuggingFace Model Discussion Pages

### 1.1 Qwen/Qwen3.5-35B-A3B Discussions
- **URL**: https://huggingface.co/Qwen/Qwen3.5-35B-A3B/discussions
- **Active discussions**: 43+ threads
- **Key thread**: "Benchmarks, tradeoffs, and live video inference across five Qwen 3.5 models" (Discussion #43)
  - 9B model beats last-gen Qwen3-VL-30B on every vision benchmark at 1/3 the size
  - 4B model retains 96% of the 27B's video understanding score
  - 27B model beats GPT-5-mini on VideoMME, MathVision, OmniDocBench
  - All five models run live video inference under 200ms per frame latency
  - Competitive positioning: superior to GPT-5-mini on vision, comparable to Claude Sonnet 4.5

### 1.2 Google Gemma-4-26B-A4B-it Discussions
- **URL**: https://huggingface.co/google/gemma-4-26B-A4B-it/discussions
- **Key thread**: "Fantastic release!" (Discussion #15)
  - **Praise**: MoE performance on laptops, outstanding multilingual, creative writing "darling" in RP communities
  - **Criticisms**: More memory-hungry than Qwen3.5's RNN hybrid, no audio in large models, missing QAT models from Gemma 3, no video support
  - Coding/agentic tasks roughly on par with Qwen 3.5
- **Discussion #7**: First community NVFP4 quantization (49GB -> 16.5GB)
- **Discussion #6**: Community resources for deployment (mobile, local, cloud)

### 1.3 NVIDIA Nemotron-Cascade-2-30B-A3B Discussions
- **URL**: https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B/discussions
- **Active discussions**: 22+ threads
- **Key thread #20**: "187 tok/s on RTX 3090, 625K Context, Agent Coding (IQ4_XS + Hermes Agent)"
  - IQ4_XS quantization by bartowski on single RTX 3090
  - Flat speed from 4K to 625K context (zero speed loss)
  - 67% faster than Qwen3.5-35B-A3B on same hardware (187 vs 112 tok/s)
  - One-shot agent coding: full GPU marketplace UI from single prompt
  - Additional user benchmarks: RX 7900 XTX: 122 tok/s, M4 Pro 48GB MLX: 88 tok/s, RTX 5090 NVFP4: ~221-230 tok/s
- **Thread #22**: MoE efficiency for agentic workflows and thinking mode trade-offs
- **Thread #9**: Community requests for official quantizations
- **Thread #6**: Criticism about deployment complexity ("There's got to be a better way")
- Released March 19, 2026; achieves IMO and IOI 2025 gold medals

### 1.4 GLM-4.7-Flash (THUDM/Z.AI)
- **Model page**: https://huggingface.co/zai-org/GLM-4.7-Flash (released under Z.AI org, not THUDM)
- 30B-A3B MoE architecture, positioned as strongest in 30B class
- GGUF available from unsloth, bartowski, lmstudio-community, others
- Q4_K_M ~10GB fits 16GB VRAM, reaching 60-100 tok/s on RTX 3090/4090
- Competitive with Sonnet 3.5 for coding; lags behind Sonnet 4.5 in complex reasoning
- **APEX GGUF available**: mudler/GLM-4.7-Flash-APEX-GGUF (4,420 downloads)
- HumanEval benchmarks for UD Q4/Q5 variants discussed in unsloth GGUF Discussion #22

### 1.5 Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated
- **URL**: https://huggingface.co/huihui-ai/Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated
- **Base model**: Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled
- **Downloads**: 136,281 last month (very high adoption)
- **Likes**: 36
- Method: abliteration using remove-refusals-with-transformers
- Requires Ollama v0.18.0+
- The underlying Jackrong model uses Claude-4.6-Opus CoT distillation via SFT with Unsloth optimization
- train_on_responses_only strategy: loss only on <think> sequences and solutions
- Fixes Qwen3.5's tendency toward excessive/repetitive reasoning on simple queries
- **No benchmark comparisons to base model provided**

### 1.6 Huihui-Qwen3.5-35B-A3B-abliterated (plain version)
- **URL**: https://huggingface.co/huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated
- **Downloads**: 62,666 last month
- **Likes**: 295
- 18 community discussions, 14 quantization variants
- Available via `ollama run huihui_ai/qwen3.5-abliterated:35b`

### 1.7 HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive
- **URL**: https://huggingface.co/HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive
- **Downloads**: 815,445 last month (MASSIVE adoption - most popular uncensored variant)
- **Likes**: 1,230+
- 27 community discussions
- Claims 0/465 refusals, "lossless uncensored"
- **Method is PROPRIETARY** - NOT standard abliteration (confirmed in Discussion #7 on 27B variant)
  - Creator stated: "Currently it's my own private methods and tools :)"
  - Took approximately one week of work for 35B variant
  - Method not publicly disclosed
- Available in 9 quantization formats (BF16 65GB to IQ2_M 11GB)
- All quants generated with importance matrix (imatrix)
- Recommended settings provided for thinking/non-thinking modes
- Compatible with llama.cpp, LM Studio, Jan, koboldcpp

---

## 2. GGUF Provider Pages

### 2.1 Unsloth/Qwen3.5-35B-A3B-GGUF
- **URL**: https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF
- **Discussion #25** (March 3, 2026 updates):
  - Q5_K_XL omitted, Q4_K_M disappeared in update
  - Recalibrated using Unsloth's own dataset (better than generic imatrix)
  - Found that quantizing attn_qkv layers negatively impacts performance - now kept at higher precision
  - Community criticism: "better to take an extra day and have a higher quality release"
- **Discussion #10** (Feb 27 GGUF Update):
  - MXFP4 layers removed from 3 quants (Q2_K_XL, Q3_K_XL, Q4_K_XL) due to quality issues
  - Tool-calling chat template fixed (added mapping validation)
  - Benchmark results invalidated: IQ4_XS, Q4_1, UD-Q4_K_XL, Q4_K_M all had mxfp4 tensors
- **Discussion #1** (Benchmarks):
  - Full benchmark data available at unsloth.ai/docs/models/qwen3.5/gguf-benchmarks
  
#### Benchmark Data (Qwen3.5-35B-A3B GGUFs):
| Quantizer | Quant | Disk (GB) | PPL | KLD 99.9% |
|-----------|-------|-----------|-----|-----------|
| Unsloth | IQ2_XXS | 9.09 | 7.716 | 4.2221 |
| Unsloth | Q2_K_XL | 12.04 | 7.0438 | 2.9092 |
| Unsloth | IQ3_XXS | 13.12 | 6.7829 | 1.5296 |
| Unsloth | Q3_K_XL | 16.06 | 6.7245 | 0.9539 |
| Unsloth | Q4_K_XL | 19.17 | 6.5918 | 0.4097 |

Key findings:
- 3-bit is especially good for balancing compression/quality
- Q4_K_XL outperforms competitors while ~8GB smaller
- ffn_* layers tolerate 3-bit well; attention layers highly sensitive
- ssm_out dramatically increases KLD when aggressively quantized
- IQ3_XXS/IQ2_XXS: 5-10% inference speed penalty vs K-quants

### 2.2 Unsloth/gemma-4-26B-A4B-it-GGUF
- Available with Dynamic 2.0 quantization
- Multiple providers: unsloth, ggml-org, lmstudio-community, bartowski

### 2.3 APEX Quantized Models from Mudler
- **CONFIRMED AVAILABLE**: Yes, APEX is fully available and production-ready
- **Full collection**: https://huggingface.co/collections/mudler/apex-quants-gguf (23 models)
- **No patches required** - works with stock llama.cpp
- **GitHub**: https://github.com/mudler/apex-quant
- **Technical report**: https://github.com/mudler/apex-quant/blob/main/paper/APEX_Technical_Report.md

#### Complete APEX Collection (23 models, as of April 2026):
| Model | Total Params | Downloads | Likes |
|-------|-------------|-----------|-------|
| Qwen3.5-35B-A3B-APEX-GGUF | 35B | 55,700 | 72 |
| gemma-4-26B-A4B-it-APEX-GGUF | 25B | 77,800 | 29 |
| Qwen3.5-35B-A3B-Claude-Distilled-APEX-GGUF | 35B | 6,300 | 9 |
| Qwen3.5-122B-A10B-APEX-GGUF | 122B | 5,710 | 9 |
| Qwen3.5-35B-A3B-APEX-TQ-GGUF | 35B | 3,010 | 10 |
| GLM-4.7-Flash-APEX-GGUF | 30B | 4,420 | 5 |
| MiniMax-M2.5-APEX-GGUF | 229B | 4,100 | - |
| Nemotron-Cascade-2-30B-A3B-APEX-GGUF | 32B | 1,910 | 3 |
| Qwen3-Coder-30B-APEX-GGUF | 31B | 3,450 | 10 |
| Qwen3-Coder-Next-APEX-GGUF | 80B | 2,590 | 6 |
| Holo3-35B-A3B-APEX-GGUF | 35B | 2,600 | 2 |
| gemma-4-26B-A4B-it-Claude-Opus-Distill-APEX-GGUF | 25B | 9,350 | 8 |
| gemma-4-26B-A4B-it-heretic-APEX-GGUF | 25B | 11,000 | 15 |
| Mistral-Small-4-119B-2603-APEX-GGUF | 119B | 1,140 | - |
| LFM2-24B-A2B-APEX-GGUF | 24B | 888 | 2 |
| Nemotron-3-Nano-30B-A3B-APEX-GGUF | 32B | 1,980 | 1 |
| NVIDIA-Nemotron-3-Super-120B-A12B-APEX-GGUF | 121B | 1,070 | - |
| Qwen3.5-397B-A17B-APEX-GGUF | 396B | 637 | 3 |
| Huihui3.5-67B-A3B-APEX-GGUF | 67B | 1,890 | 1 |
| Qwopus-MoE-35B-A3B-APEX-GGUF | 35B | 1,970 | 2 |
| Step-3.5-Flash-APEX-GGUF | 197B | 751 | - |
| Trinity-Large-Thinking-APEX-GGUF | 399B | 653 | - |

#### APEX Qwen3.5-35B-A3B Benchmark Data:
| Variant | Size | PPL | Speed | HellaSwag | MMLU | ARC | TruthfulQA |
|---------|------|-----|-------|-----------|------|-----|------------|
| APEX Quality | 21.3 GB | 6.527 | 62.3 t/s | 83.0% | 41.2% | 56.2% | 37.7% |
| APEX I-Quality | 21.3 GB | 6.552 | 63.1 t/s | 83.5% | 41.7% | 57.9% | 38.4% |
| APEX Balanced | 23.6 GB | 6.533 | 60.8 t/s | - | - | - | - |
| APEX I-Balanced | 23.6 GB | 6.548 | 61.4 t/s | - | - | - | - |
| APEX Compact | 16.1 GB | 6.783 | 69.8 t/s | - | - | - | - |
| APEX I-Compact | 16.1 GB | 6.669 | 69.8 t/s | 81.8% | 41.7% | 55.5% | 37.9% |
| APEX Mini | 12.2 GB | 7.088 | 74.4 t/s | 81.0% | 41.3% | - | - |

- **APEX Quality beats F16** on perplexity (6.527 vs 6.537) while 38% smaller than Q8_0
- **APEX Mini beats bartowski IQ2_M** on every metric despite similar size
- APEX beats Unsloth Dynamic 2.0 on accuracy at similar sizes

---

## 3. Key Confirmations

### 3.1 Is APEX quantization from mudler actually available?
**YES, CONFIRMED.** APEX is:
- Fully available on HuggingFace (23 models in collection)
- Open source: https://github.com/mudler/apex-quant
- Has a technical report
- Works with stock llama.cpp (no patches needed)
- Works with LocalAI out of the box
- Download via: `huggingface-cli download mudler/MODEL-APEX-GGUF filename.gguf --local-dir ./model`
- Use via: `llama-cli -m ./model/filename.gguf --conversation -ngl 99`

### 3.2 Has anyone done side-by-side benchmarks of base vs abliterated Qwen3.5-35B-A3B?
**PARTIALLY.** Key findings:
- **bswen.com comparison** (March 10, 2026): Tested both variants on RTX 5070 Ti 16GB
  - "Zero-loss abliteration variants showed minimal quality impact"
  - No measurable difference in reasoning or coding tasks
  - Multilingual capabilities preserved
  - Practical assessment, not controlled lab testing
- **Kaitchup/Benjamin Marie**: Abliterated 9B variants perform poorly even at Q6_K, "significantly underperforming standard Q4_K_L versions" - BUT this was for 9B, not 35B
- **HauhauCS**: Claims 0/465 refusals with "lossless" uncensoring (proprietary method, not abliteration)
- **hexgrad HF post**: Acknowledges hallucination risk with abliteration; no specific benchmark data
- **Norm-preserving biprojected abliteration**: Improved reasoning (NatInt: 21.33 vs baseline 18.72) while removing refusals

### 3.3 Community consensus on Gemma 4 MoE vs Qwen3.5 MoE for local use?
**NO CLEAR WINNER - depends on use case:**

| Dimension | Winner | Details |
|-----------|--------|---------|
| Raw benchmarks | Qwen3.5-35B-A3B | Better text reasoning, expert-science, Tau2 lead |
| Human preference (Arena) | Gemma 4 26B-A4B | 1441 vs 1400 ELO |
| Generation speed (GPU) | Mixed | HN: Gemma 150 tok/s vs Qwen 100 tok/s on 4090; Aayush: both ~135 tok/s |
| Generation speed (Mac) | Qwen3.5 | mmap works well due to hybrid RNN architecture |
| Multilingual | Gemma 4 | Community reports notably better non-English performance |
| Creative writing | Gemma 4 | "Darling in RP communities" |
| Agentic coding reliability | Qwen3.5-27B (Dense) | MoE variants less reliable on complex tasks |
| Tool calling | Qwen3.5 | Gemma 4 struggles with tool selection sequencing |
| Memory efficiency | Qwen3.5 | RNN hybrid needs less KV cache; Gemma requires more memory |
| Context window (practical) | Gemma 4 26B | 256K vs 200K on 24GB VRAM |
| Licensing | Tie | Both Apache 2.0 |

**Bottom line**: Qwen3.5-35B-A3B for coding/reasoning/agentic; Gemma 4 26B-A4B for chat/multilingual/creative.

### 3.4 Any reports of Rapid-MLX real-world usage?
**YES, CONFIRMED active and well-documented:**
- **GitHub**: https://github.com/raullenchai/Rapid-MLX
- Released March 23, 2026
- Installable via: `brew install raullenchai/rapid-mlx/rapid-mlx`
- Claimed 2-4.2x faster than Ollama on M3 Ultra
- DeltaNet state snapshots: multi-turn TTFT from 1.5s to <200ms
- 100% tool calling for Qwen, GLM, GPT-OSS, Kimi models
- 17 tool parsers with automatic recovery
- Works with Claude Code, Cursor, Aider, Continue.dev
- Integration tests for PydanticAI, smolagents, LangChain, Anthropic SDK
- Discussed in Cline project (Discussion #9940)
- Performance on M3 Ultra: Qwen3.5-35B at 83 tok/s, 9B at 108 tok/s, 122B at 44 tok/s

### 3.5 Has anyone successfully used lm-eval-harness with local models on Mac?
**YES, but with caveats:**
- lm-eval-harness supports MPS (Metal) backend via HuggingFace Transformers
- Can evaluate GGUF models via HuggingFace backend with `--model_args gguf_file=...`
- Can use `local-completions` model type to evaluate via any OpenAI-compatible API (Ollama, vLLM, etc.)
- No dedicated MLX backend tutorial found
- liteLLM provides integration documentation for lm-eval-harness with local servers
- More commonly, community uses dedicated tools:
  - **MLXBench**: Benchmark Ollama, vLLM-MLX, Docker Model Runner
  - **Anubis**: Native macOS benchmarking app with hardware telemetry
  - **local-llm-bench**: Compare MLX vs llama.cpp scenarios
  - **Silicon Score**: 309 benchmark rows across 28 Mac configs
  - **SiliconBench**: Measured tok/s by model, chip, quantization with CSV download

---

## 4. Benjamin Marie / The Kaitchup Findings

### 4.1 Qwen3.5 Quantization Articles (Series of 5+):
1. **"Qwen3.5 Quantization: Similar Accuracy, More Thinking"** (March 12, 2026)
   - INT4, NVFP4, FP8, BF16 comparison
   - 4-bit Qwen3.5 27B stronger than 9B at similar memory
   - Some layers need higher precision preservation
   
2. **"Summary of Qwen3.5 GGUF Evaluations"**
   - 750-998 samples across LiveCodeBench, GPQA Diamond, MMLU-Pro, Math500
   - 397B: "very robust to quantization" - many weights compressible
   - 35B/27B/9B: Avoid Q2, Q4 is "very safe", Q3 needs careful evaluation
   - **Abliterated models**: Popular 9B variants perform poorly even at Q6_K
   - PPL/KLD "very hard to interpret" for real-world task performance
   - KV cache quantization results behind paywall

3. **"Lessons from GGUF Evaluations: Ternary Qwen3.5, Bricked MiniMax"**
   - TQ1_0 (ternary {-1,0,+1}): benchmark error up only ~18.4%, memory 800GB->94GB
   - 397B GGUF quantizations "remarkably safe"
   - MiniMax was "bricked" by some quants

4. **"More Qwen3.5 GGUF Evals and Speculative Decoding"**
   
5. **"Qwen3.5 Medium Models: Dense vs. MoE"**
   - Direct comparison of dense vs MoE architectures

### 4.2 The Kaitchup Index
- Leaderboard for LLMs and their quantized versions
- Covers GGUF, AWQ, GPTQ, bitsandbytes (16-bit to 2-bit)
- Organized by GPU size: 8GB, 12GB, 16GB, 24GB
- Heavily multilingual evaluation
- All benchmarks on RTX 3090 for consistency

---

## 5. Unsloth Dynamic 2.0 Quantization

- Works on all models (MoE and non-MoE) - expanded from MoE-only v1
- Custom-tailored per model (Gemma 3 layers differ from Llama 4)
- Lower KL Divergence than standard imatrix and QAT quants
- **APEX comparison**: APEX claims to beat Unsloth Dynamic 2.0 on accuracy at similar sizes
  - APEX I-Quality (21.3GB) PPL 6.552 vs UD-Q8_K_XL (45.3GB) PPL 6.536
  - APEX Quality PPL 6.527 vs UD-Q4_K_L PPL 6.586 at similar file sizes

---

## 6. Ollama MLX Integration (v0.19)

- **Released**: March 31, 2026
- **Backend change**: llama.cpp -> MLX on Apple Silicon
- **Performance gains**:
  - Prefill: 1,100 -> 1,851 tok/s (1.7x faster)
  - Decode: 58 -> 134 tok/s (2.3x faster)
  - Third-party: ~230 tok/s sustained throughput
- **M5 chips**: Additional 19-27% improvement via GPU Neural Accelerators
- **Limitations**: Preview status, only Qwen3.5-35B-A3B MLX-accelerated initially, 32GB minimum RAM
- Go wrapper overhead: up to 30% vs raw MLX
- Community reception: very positive, #2 on Product Hunt (379 upvotes)
- **HN discussion**: Real-world users confirming faster performance, privacy benefits emphasized

---

## 7. Gemma 4 Community Findings (First 24 Hours)

From DEV Community and HN discussions:
- **Speed problem**: Gemma 4 26B-A4B: 11 tok/s vs Qwen3.5-35B-A3B: 60+ tok/s on same 5060 Ti 16GB (some conflicting reports)
- **ELO vs benchmarks gap**: Arena score 2150 (above GPT-OSS-120B) but roughly ties Qwen 3.5 27B on automated benchmarks
- Human preference consistently higher than benchmark scores suggest
- **Tooling issues at launch**: HF Transformers didn't recognize gemma4 architecture, PEFT couldn't handle Gemma4ClippableLinear, new mm_token_type_ids field required
- **QLoRA**: 3 bugs within first 24 hours (all fixed)
- Fine-tuning community recommended waiting several days for library updates

---

## 8. Apple Silicon Inference Ecosystem (2026)

### Framework Comparison (M4 Pro 64GB, DeepSeek V3 Q4_K_M):
| Framework | Single-User | 32-User Throughput | TTFT |
|-----------|------------|-------------------|------|
| vllm-mlx | 42 t/s | 1,150 t/s | ~120ms |
| Ollama v0.8+ | 58 t/s | 720 t/s | ~45ms |
| llama.cpp Metal | 52 t/s | 890 t/s | ~85ms |

### Agent Workload Recommendations (Stochastic Sandbox):
| Scenario | Best Choice | Rationale |
|----------|------------|-----------|
| Development | Ollama | Model library, stable API |
| Production multi-agent | vMLX | KV quant + 256 sequences |
| Long-running agents | omlx | SSD cache survives restarts |
| Cross-platform | llama.cpp server | Full parameter control |
| Non-technical teams | LM Studio | GUI + OpenAI API |

### 35B on Mac Mini 16GB ($599):
- Qwen3.5-35B via mmap: 17.3 tok/s, 81% memory free, zero swap
- Only active 3B params loaded; 90% of model on NVMe SSD
- Classification: 8.5s (Qwen) vs 1.9s (Gemma 4) - Gemma 4x faster for classification
- Critical finding: disabling thinking mode = 30x speedup on classification (30s -> <1s)

---

## 9. Simon Willison's Recent Work

- **llm-mlx plugin**: His favorite way to run local models on Mac
- **March 18, 2026**: Dan Woods ran Qwen3.5-397B at 5.5+ tok/s on 48GB M3 Max using "LLM in a Flash" technique
  - Claude Code ran 90 experiments producing MLX Objective-C and Metal code
  - Available: danveloper/flash-moe repository
- **April 6, 2026**: Covered Google AI Edge Gallery (iPhone app for Gemma 4 E2B/E4B local inference)
- **April 2, 2026**: Tested Gemma 4 via LM Studio GGUF; 31B showed generation issues, smaller models worked well
- Active participant in MLX local model ecosystem

---

## 10. Hardware Buying Guide (Julien Simon, April 2026)

| Tier | Hardware | Cost | Best For |
|------|----------|------|----------|
| <30B | RTX 5090 (32GB, 1,792 GB/s) | $5-8K system | 60-90 tok/s dense models |
| 70B | Mac Studio M4 Max (128GB, 546 GB/s) | $3,499-$3,699 | 8-15 tok/s, best value |
| 70B budget | 4x Mac Mini M4 Pro (48GB each) | $6,400-$7,200 | Batch inference via TB5 |
| 70B CUDA | RTX PRO 6000 (96GB, 1.8 TB/s) | ~$22K system | Professional workloads |
| 200B+ | Mac Studio M3 Ultra (256GB) | ~$5,999 | Llama 405B in Q4 |

MoE-specific: 30B MoE (3B active) at 234 tok/s on RTX 5090 vs ~52 tok/s for dense 32B.

---

## 11. Abliteration Quality Impact Summary

| Source | Model Size | Finding |
|--------|-----------|---------|
| bswen.com | 35B-A3B | "Zero-loss" - no measurable difference in reasoning/coding |
| Kaitchup | 9B | Abliterated 9B "significantly underperforms" standard Q4_K_L at Q6_K |
| HauhauCS | 35B-A3B | Claims "lossless" (proprietary method, not abliteration) |
| Norm-preserving abliteration | General | Improved reasoning (NatInt 21.33 vs 18.72 baseline) |
| Standard abliteration | General | Slight degradation (NatInt 18.64 vs 18.72 baseline) |
| DPO healing | Daredevil-8B | DPO fine-tuning can restore quality after abliteration damage |

**Conclusion**: Abliteration impact varies by model size. Larger models (35B+) appear more robust. For smaller models (9B), quality loss is measurable and significant. HauhauCS's proprietary method may avoid standard abliteration pitfalls.

---

## 12. Qwen3.5-35B-A3B-Claude-4.6-Opus Distillation Variants

The distillation chain:
1. **Base**: Qwen3.5-35B-A3B (Qwen official)
2. **Distilled**: Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled
   - SFT with Claude-4.6-Opus reasoning chains
   - train_on_responses_only strategy
   - Fixes Qwen3.5's verbose reasoning on simple queries
   - Unsloth-optimized training
3. **Abliterated**: huihui-ai/Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated
   - 136,281 downloads/month
   - Ollama: `huihui_ai/qwen3.5-abliterated:35b-Claude`
4. **APEX-quantized**: mudler/Qwen3.5-35B-A3B-Claude-Distilled-APEX-GGUF
   - 6,300 downloads
