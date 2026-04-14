# MoE Models on HuggingFace - Research for 32GB Apple Silicon Mac

**Date:** 2026-04-08
**Goal:** Find MoE models that fit in 32GB unified memory at Q3-Q4 quantization
**Reference:** huihui-qwen3.5-35b-a3b (35B total, 3B active) - MMLU ~90%, HumanEval ~90% at Q3

---

## Key Insight: MoE Memory Advantage

MoE models must load ALL expert weights into memory (total params), but only activate a fraction per token.
This means GGUF file size is based on TOTAL params, not active params.
For 32GB Mac: need GGUF file < ~28GB (leaving ~4GB for OS + context).

---

## TIER 1: Best Candidates (Fit at Q3-Q5, Strong Benchmarks)

### 1. Qwen3-30B-A3B (Qwen)
- **Total/Active:** 30.5B / 3.3B
- **Architecture:** 128 experts, 8 activated, 48 layers
- **Context:** 32K native, 131K with YaRN
- **Downloads:** 1.58M | **Likes:** 874
- **Key benchmarks:** Outcompetes QwQ-32B with 10x fewer active params. ~82% MMLU-Pro (CS) per community testing.
- **GGUF sizes (unsloth):**
  - Q3_K_M: 14.7 GB
  - Q4_K_M: 18.6 GB
  - Q5_K_M: 21.7 GB
  - Q6_K: 25.1 GB
  - Q8_0: 32.5 GB (tight fit)
- **mradermacher i1:** Yes - mradermacher/Qwen3-30B-A3B-i1-GGUF
- **Apple Silicon perf:** ~40 tok/s at Q4_K_M, ~64 tok/s in MLX 4-bit on M4 Max
- **Verdict:** EXCELLENT fit. Q4_K_M at 18.6GB leaves plenty of room. Q5 and Q6 also fit.

### 2. Qwen3.5-35B-A3B (Qwen) -- YOUR REFERENCE MODEL
- **Total/Active:** 35B / 3B (256 experts, 8 routed + 1 shared)
- **Architecture:** Hybrid - Gated DeltaNet + Gated Attention + MoE, 40 layers
- **Context:** 262K native, up to 1M with scaling
- **Downloads:** 3.34M | **Likes:** 1.34k
- **Benchmarks:**
  - MMLU-Pro: 85.3
  - GPQA Diamond: 84.2
  - IFEval: 91.9
  - SWE-bench Verified: 69.2
  - LiveCodeBench v6: 74.6
  - CodeForces ELO: 2028
  - C-Eval: 90.2
- **GGUF sizes (unsloth):**
  - Q3_K_S: 15.3 GB
  - Q3_K_M: 16.4 GB
  - Q4_K_M: 22 GB
  - Q5_K_M: 26.2 GB
  - Q6_K: 28.9 GB (tight)
  - Q8_0: 36.9 GB (NO FIT)
- **mradermacher i1:** Yes - mradermacher/Huihui-Qwen3.5-35B-A3B-abliterated-i1-GGUF
- **Vision:** YES - native multimodal (text + image)
- **Verdict:** EXCELLENT. Q3 at 15-16GB, Q4 at 22GB, Q5 at 26GB all fit. Best overall benchmarks in this class.

### 3. Google Gemma 4 26B-A4B-it
- **Total/Active:** 25.2B / 3.8B (128 experts, 8 active + 1 shared)
- **Architecture:** MoE, 30 layers
- **Context:** 256K tokens
- **Downloads:** 835K | **Likes:** 519
- **Benchmarks:**
  - MMLU Pro: 82.6
  - AIME 2026: 88.3
  - LiveCodeBench v6: 77.1
  - CodeForces ELO: 1718
  - GPQA Diamond: 82.3
  - MMMLU: 86.3
  - MMMU Pro (Vision): 73.8
  - MATH-Vision: 82.4
- **GGUF sizes (unsloth):**
  - Q3_K_M: 12.5 GB
  - Q4_K_M: 16.9 GB
  - Q5_K_M: 21.2 GB
  - Q6_K: 22.9 GB
  - Q8_0: 26.9 GB
- **Vision:** YES - multimodal (text + image)
- **License:** Apache 2.0
- **Verdict:** EXCELLENT. Smallest total params = smallest GGUF. Q4 at 16.9GB is very comfortable. Very strong benchmarks.

### 4. NVIDIA Nemotron-3-Nano-30B-A3B
- **Total/Active:** 30B / 3.5B (128 routed + 1 shared expert, 6 activated)
- **Architecture:** HYBRID Mamba-2 + Transformer + MoE (23 Mamba, 23 MoE, 6 GQA layers)
- **Context:** 256K default, up to 1M tokens
- **Downloads:** 1.26M | **Likes:** 704
- **Benchmarks:**
  - MMLU-Pro: 78.3
  - AIME25 (no tools): 89.1
  - AIME25 (with tools): 99.2
  - GPQA: 73.0
  - LiveCodeBench: 68.3
  - RULER-100@256k: 92.9
  - RULER-100@1M: 86.3
  - Arena-Hard-V2: 72.1
- **GGUF sizes (unsloth):**
  - Q3_K_M: 20 GB
  - Q4_K_M: 24.6 GB
  - Q5_K_M: 26.1 GB
  - Q6_K: 33.5 GB (NO FIT)
- **NOTE:** Mamba layers don't compress as well - Q2/Q3 sizes are 18.1GB floor
- **mradermacher i1:** Yes - mradermacher/Nemotron-Cascade-2-30B-A3B-i1-GGUF (for Cascade variant)
- **Verdict:** GOOD fit at Q3 (20GB) and Q4 (24.6GB). Unique Mamba architecture = great long context. Slightly weaker benchmarks than Qwen3.5.

### 5. NVIDIA Nemotron-Cascade-2-30B-A3B
- **Total/Active:** 32B / 3B
- **Architecture:** Same hybrid Mamba-2 + MoE (post-trained from Nemotron-3-Nano base)
- **Context:** up to 262K tokens
- **Downloads:** 215K | **Likes:** 466
- **Release:** March 2026
- **Benchmarks (OUTSTANDING reasoning):**
  - IMO 2025: Gold medal (35 pts)
  - IOI 2025: Gold medal (439.3 pts)
  - ICPC World Finals 2025: Gold (10/12)
  - AIME 2025: 92.4 (98.6 with TIR)
  - AIME 2026: 90.9 (95.0 with TIR)
  - LiveCodeBench v6: 87.2
  - MMLU-Pro: 79.8
  - GPQA Diamond: 76.1
  - ArenaHard v2: 83.5
  - SWE Verified (OpenHands): 50.2
  - HLE (no tool): 17.7
- **GGUF:** Same sizes as Nemotron-3-Nano (Q3_K_M ~20GB, Q4_K_M ~24.6GB)
- **mradermacher i1:** Yes
- **Verdict:** EXCELLENT. Best math/code reasoning model in this size class. Gold medals on IMO/IOI/ICPC. Fits at Q3-Q4.

### 6. Qwen3-Coder-30B-A3B-Instruct
- **Total/Active:** 30.5B / 3.3B (128 experts, 8 activated)
- **Context:** 262K native, up to 1M with Yarn
- **Downloads:** 1.11M | **Likes:** ~1k
- **Benchmarks:** SWE-bench Verified ~51.6 (OpenHands). Strong agentic coding. Exact scores not published on model card.
- **GGUF sizes:** Same as Qwen3-30B-A3B (Q3_K_M ~14.7GB, Q4_K_M ~18.6GB)
- **Verdict:** EXCELLENT for coding tasks. Same great fit as Qwen3-30B-A3B.

---

## TIER 2: Interesting But With Caveats

### 7. Qwen3-Next-80B-A3B-Instruct (Qwen)
- **Total/Active:** 80B / 3B (512 experts, 10 activated + 1 shared)
- **Architecture:** Hybrid DeltaNet + Attention + MoE, 48 layers
- **Context:** 262K native, up to 1M
- **Downloads:** 515K | **Likes:** 958
- **Benchmarks:**
  - MMLU-Pro: 80.6
  - Arena-Hard v2: 82.7
  - WritingBench: 87.3
  - LiveCodeBench v6: 56.6
  - Outperforms Qwen3-235B-A22B on some benchmarks
- **GGUF sizes (unsloth):**
  - UD-IQ1_S: 22.9 GB
  - UD-TQ1_0: 20.5 GB
  - UD-IQ2_XXS: 26.2 GB
  - Q2_K: 29.2 GB (tight)
  - Q3_K_S: 34.6 GB (NO FIT)
- **Verdict:** MARGINAL. Only fits at extreme 1-2 bit quants. Quality may suffer significantly. The 10x inference throughput advantage for long context is appealing but Q1-Q2 quantization is aggressive.

### 8. Kimi-VL-A3B-Instruct (Moonshot AI)
- **Total/Active:** 16B / 3.2B (2.8B LLM + 0.4B ViT)
- **Architecture:** MoE vision-language model with native-resolution vision encoder
- **Context:** 128K tokens
- **Downloads:** 286K | **Likes:** 258
- **Benchmarks:**
  - ScreenSpot-V2 (Agent): 92.8
  - InfoVQA: 83.2
  - MMBench-EN: 83.1
  - MathVista: 68.7
- **GGUF:** Not widely available in GGUF yet
- **Verdict:** INTERESTING for vision tasks. Very small total params = easy fit. But limited GGUF availability and older architecture.

### 9. Microsoft Phi-3.5-MoE-instruct
- **Total/Active:** 42B (16x3.8B) / 6.6B (2 experts active)
- **Architecture:** MoE decoder-only Transformer
- **Context:** 128K tokens
- **Downloads:** 105K | **Likes:** 572
- **Benchmarks:**
  - MMLU (5-shot): 78.9
  - HumanEval: 70.7
  - MATH (CoT): 59.5
  - GSM8K (CoT): 88.7
  - MBPP: 80.8
  - ARC Challenge: 91.0
- **GGUF sizes:**
  - Q2_K: ~15 GB
  - Q3_K_M: ~20 GB (estimated)
  - Q4_K_M: ~24 GB
- **License:** MIT
- **Verdict:** DECENT fit at Q2-Q4. Older model (Aug 2024). Outperformed by newer Qwen3.5 and Gemma 4 on most benchmarks. MIT license is nice.

### 10. Mixtral 8x7B (Mistral AI)
- **Total/Active:** 46.7B / ~12.9B (8 experts, 2 active)
- **Architecture:** Classic MoE, 32 layers
- **Context:** 32K tokens
- **Downloads:** Very high (legacy model)
- **Benchmarks:** MMLU ~70-71, competitive with Llama2-70B at release
- **GGUF sizes:**
  - Q3_K_M: ~20.4 GB
  - Q4_K_M: ~26 GB
- **Verdict:** LEGACY. Fits at Q3, but massively outperformed by newer models. 12.9B active params means slower inference than 3B-active models.

### 11. Mistral-Small-4-119B (Mistral AI)
- **Total/Active:** 119B / 6.5B (128 experts, 4 active)
- **Context:** 256K tokens
- **Benchmarks:** Strong across reasoning, coding, vision
- **GGUF:** noctrex MXFP4_MOE version available
- **Verdict:** TOO BIG. Even at Q2 would need ~40-50GB+. Not viable for 32GB.

---

## TIER 3: Small/Research MoE Models (Easy Fit, Lower Capability)

### 12. DeepSeek-V2-Lite (DeepSeek)
- **Total/Active:** 15.7B / 2.4B (64 routed + 2 shared experts, 6 activated)
- **Architecture:** MLA + DeepSeekMoE, 27 layers
- **Context:** 32K tokens
- **Downloads:** 245K | **Likes:** 169
- **Benchmarks:**
  - MMLU: 58.3 (base) / 55.7 (chat)
  - HumanEval: 29.9 (base) / 57.3 (chat)
  - GSM8K: 41.1 (base) / 72.0 (chat)
  - MATH: 17.1 (base) / 27.9 (chat)
- **GGUF:** Multiple sources available (mav23, lmstudio-community)
- **Estimated Q4 size:** ~9-10 GB
- **Verdict:** Easy fit but MUCH weaker benchmarks than modern models. Historical interest only.

### 13. DeepSeek-Coder-V2-Lite (DeepSeek)
- **Total/Active:** 16B / 2.4B
- **Architecture:** Same as V2-Lite but coding-focused
- **GGUF:** Available from bartowski, lmstudio-community
- **Verdict:** Same as above. Outperformed by Qwen3-Coder-30B-A3B.

### 14. OLMoE-1B-7B (Allen AI)
- **Total/Active:** 6.9B / 1B
- **Context:** Not specified (likely 4K-8K)
- **Downloads:** Moderate | **Likes:** Moderate
- **Benchmarks:** Outperforms Llama2-13B-Chat and DeepSeekMoE-16B
- **GGUF:** Official from allenai, bartowski, QuantFactory
- **Estimated Q4 size:** ~4 GB
- **License:** Fully open source (100% open)
- **Verdict:** Very easy fit but too small for serious use. Research-grade.

### 15. Qwen1.5-MoE-A2.7B (Qwen)
- **Total/Active:** 14.3B / 2.7B
- **Context:** 32K tokens
- **Downloads:** 173K | **Likes:** 223
- **Benchmarks:** Comparable to Qwen1.5-7B at 25% training cost
- **GGUF:** Available (RichardErkhov, QuantFactory)
- **Estimated Q4 size:** ~8 GB
- **Verdict:** Superseded by Qwen3-30B-A3B. Historical interest.

### 16. LLaDA-MoE-7B-A1B (inclusionAI)
- **Total/Active:** 7B / 1.4B
- **Architecture:** First open-source MoE DIFFUSION language model
- **Training:** 20T tokens
- **Benchmarks:** Comparable to Qwen2.5-3B-Instruct
- **GGUF:** Community quants available
- **Verdict:** Research novelty (diffusion LLM + MoE). Too small for production use.

### 17. LFM2-8B-A1B (Liquid AI)
- **Total/Active:** 8.3B / 1.5B
- **Architecture:** Hybrid MoE, designed for edge/on-device
- **Benchmarks:** Comparable to 3-4B dense models; faster than Qwen3-1.7B
- **GGUF:** Official from LiquidAI + unsloth + bartowski
- **Estimated Q4 size:** ~5 GB
- **Verdict:** Designed for edge devices. Very easy fit but limited capability.

### 18. DBRX (Databricks)
- **Total/Active:** 132B / 36B (16 experts, 4 active)
- **Context:** 32K tokens
- **GGUF:** mradermacher, dranger003 (imat)
- **Verdict:** TOO BIG for 32GB at any useful quant level. Legacy model (March 2024).

---

## TIER 4: Fine-tunes & Variants of Top Models

### 19. Qwen3-VL-30B-A3B-Instruct (Qwen) - Vision Language
- **Total/Active:** ~30B / 3B
- **Modality:** Text + Image + Video
- **Downloads:** 4.36M (most downloaded A3B model!)
- **GGUF:** Official from Qwen + unsloth (Thinking and Instruct variants)
- **Verdict:** Same size as Qwen3-30B-A3B. Vision variant. EXCELLENT fit.

### 20. Qwen3-Omni-30B-A3B-Instruct (Qwen) - Omni
- **Total/Active:** ~35B / 3B
- **Modality:** Text + Image + Audio + Video (input), Text + Speech (output)
- **Downloads:** 351K | **Likes:** 901
- **Audio:** 19 input languages, 10 output languages, 234ms first-packet latency
- **Verdict:** Audio-capable variant. Fits at Q3-Q4.

### 21. Qwen3-30B-A3B-Instruct-2507 / Thinking-2507
- **Updated instruct-tuned variants (July 2025)**
- **Downloads:** 1.22M (Instruct) / 236K (Thinking)
- **GGUF:** unsloth versions available
- **Verdict:** Updated post-training. Same GGUF sizes.

### 22. huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated
- **Base:** Qwen3.5-35B-A3B with refusal removal (abliteration)
- **GGUF:** mradermacher i1 + noctrex MXFP4_MOE versions
- **Verdict:** Uncensored variant. Same fit profile as Qwen3.5-35B-A3B.

### 23. HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive
- **Downloads:** 815K | **Likes:** 1.23k
- **Base:** Qwen3.5-35B-A3B
- **Verdict:** Very popular uncensored variant. Same fit.

### 24. Qwopus-MoE-35B-A3B (samuelcardillo)
- **Base:** Qwen3.5-35B-A3B fine-tuned with Jackrong's methodology
- **Focus:** Coding assistants, research agents, agentic workflows
- **GGUF:** samuelcardillo + mudler versions
- **Verdict:** Niche fine-tune. Same fit profile.

### 25. Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled
- **Base:** Qwen3.5-35B-A3B distilled from Claude Opus reasoning
- **GGUF:** noctrex MXFP4_MOE version
- **Verdict:** Interesting distillation approach. Same fit profile.

### 26. FINAL-Bench/Darwin-35B-A3B-Opus
- **Base:** Qwen3.5-35B-A3B variant
- **Verdict:** Community fine-tune. Same fit profile.

---

## TOO BIG for 32GB (Reference Only)

| Model | Total | Active | Why Listed |
|-------|-------|--------|------------|
| Qwen3.5-122B-A10B | 122B | 10B | Qwen3.5 large variant. Q2 ~50GB+. |
| DeepSeek-V3-0324 | 671B | 37B | Far too large. |
| DeepSeek-R1 | 671B | 37B | Far too large. |
| Mistral-Small-4-119B | 119B | 6.5B | Q2 ~40GB+. |
| MiniMax-M2.5 (172B-A10B REAP) | 172B | 10B | REAP-pruned but still too big. |
| Mixtral 8x22B | 141B | 39B | Far too large. |
| DBRX 132B | 132B | 36B | Far too large. |

---

## GGUF Size Comparison at Key Quant Levels (Models That Fit)

| Model | Q3_K_M | Q4_K_M | Q5_K_M | Q6_K | Q8_0 |
|-------|--------|--------|--------|------|------|
| Gemma-4-26B-A4B | 12.5 GB | 16.9 GB | 21.2 GB | 22.9 GB | 26.9 GB |
| Qwen3-30B-A3B | 14.7 GB | 18.6 GB | 21.7 GB | 25.1 GB | 32.5 GB |
| Qwen3.5-35B-A3B | 16.4 GB | 22.0 GB | 26.2 GB | 28.9 GB | 36.9 GB |
| Nemotron-3-Nano-30B-A3B | 20.0 GB | 24.6 GB | 26.1 GB | 33.5 GB | 33.6 GB |
| Phi-3.5-MoE (42B) | ~20 GB | ~24 GB | ~28 GB | N/A | N/A |
| Mixtral 8x7B (47B) | ~20 GB | ~26 GB | N/A | N/A | N/A |
| Qwen3-Next-80B-A3B | 38.3 GB | 48.5 GB | N/A | N/A | N/A |

**Bold = fits in 32GB with room for context**

---

## GGUF Quantization Providers

| Provider | Style | Notes |
|----------|-------|-------|
| **unsloth** | Dynamic 2.0 quants (UD-*) | SOTA accuracy, wide range incl. UD-IQ variants |
| **mradermacher** | Static + i1 (imatrix) | i1 = importance-matrix weighted, better quality at low bits |
| **bartowski** | imatrix-based | Popular, reliable quants |
| **lmstudio-community** | Standard quants | LM Studio compatible |
| **noctrex** | MXFP4_MOE specialized | MoE-aware quantization, better expert handling |
| **Qwen (official)** | Standard GGUF | Official but fewer quant levels |

---

## Recommendations (Ranked)

### For General Use:
1. **Qwen3.5-35B-A3B** at Q3-Q4 -- Best overall quality, multimodal, 262K context
2. **Gemma-4-26B-A4B** at Q4-Q5 -- Smallest footprint, strong benchmarks, Apache 2.0
3. **Qwen3-30B-A3B** at Q4-Q5 -- Excellent quality/size ratio

### For Math/Reasoning:
1. **Nemotron-Cascade-2-30B-A3B** at Q3-Q4 -- IMO/IOI/ICPC gold medals
2. **Qwen3.5-35B-A3B** at Q3-Q4

### For Coding:
1. **Qwen3-Coder-30B-A3B** at Q4-Q5 -- Purpose-built for coding
2. **Nemotron-Cascade-2-30B-A3B** at Q3-Q4 -- IOI gold medal
3. **Qwen3.5-35B-A3B** at Q3-Q4 -- SWE-bench 69.2

### For Long Context (>100K):
1. **Nemotron-3-Nano-30B-A3B** at Q3-Q4 -- Mamba architecture, 1M context, RULER@1M: 86.3
2. **Qwen3.5-35B-A3B** at Q3 -- 262K native

### For Vision + Language:
1. **Qwen3.5-35B-A3B** at Q3-Q4 -- Native multimodal
2. **Gemma-4-26B-A4B** at Q4-Q5 -- Image understanding
3. **Qwen3-VL-30B-A3B** at Q4-Q5 -- Dedicated VL model

### For Audio/Omni:
1. **Qwen3-Omni-30B-A3B** at Q3-Q4 -- Only option with audio I/O

### For Uncensored:
1. **HauhauCS/Qwen3.5-35B-A3B-Uncensored-Aggressive** (815K downloads, very popular)
2. **huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated** (classic abliteration)

---

## Key GGUF Sources to Download From

- Qwen3-30B-A3B: https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF
- Qwen3.5-35B-A3B: https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF
- Gemma-4-26B-A4B: https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF
- Nemotron-3-Nano: https://huggingface.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF
- Nemotron-Cascade-2: https://huggingface.co/mradermacher/Nemotron-Cascade-2-30B-A3B-i1-GGUF
- Qwen3-Coder: https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
- Qwen3-Next-80B (extreme quant): https://huggingface.co/unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF
