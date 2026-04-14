# Model Candidate List for 32GB M4 Mac Benchmarking
**Date:** 2026-04-08
**Hardware:** 32GB M4 Mac, LM Studio (GGUF), 120 GB/s bandwidth
**Reference model:** huihui-qwen3.5-35b-a3b-claude-4.6-opus-abliterated-i1 Q3_K_S (15.3GB)
**Reference scores:** MMLU 90%, DROP 95%/F1:90.6, HumanEval 90%, MATH 75%, GPQA 60%

---

## TIER 1: Must-Test (Top MoE contenders - different architectures, all fit comfortably)

### 1. Gemma-4-26B-A4B-it (Google)
- **Why:** Highest Arena ELO (1441) of any model that fits. Smallest MoE footprint.
- **Params:** 25.2B total / 3.8B active (128 experts, 16 active)
- **GGUF:** unsloth UD-IQ4_XS = 13.4GB, Q3_K_M = 12.5GB
- **Key benchmarks:** MMLU-Pro 82.6, GPQA 82.3, LiveCodeBench 77.1, AIME 88.3
- **Context:** 256K | **Multimodal:** Yes | **License:** Apache 2.0
- **Source:** unsloth/gemma-4-26B-A4B-it-GGUF

### 2. Nemotron-Cascade-2-30B-A3B (NVIDIA)
- **Why:** Gold medals on IMO/IOI/ICPC. LiveCodeBench 87.2 (best in class). Math/code beast.
- **Params:** 32B total / 3B active (hybrid Mamba-2 + MoE)
- **GGUF:** Q3_K_M ~20GB (fits with some room)
- **Key benchmarks:** AIME 2026: 90.9, LiveCodeBench 87.2, MMLU-Pro 79.8, GPQA 76.1
- **Context:** 262K | **License:** NVIDIA Open Model
- **Source:** mradermacher/Nemotron-Cascade-2-30B-A3B-i1-GGUF

### 3. GLM-4.7-Flash (Zhipu AI)
- **Why:** Fastest inference of all MoE models. Best math (AIME 91.6). MIT license.
- **Params:** 30B total / ~3B active (64 experts, 4 active + 1 shared)
- **GGUF:** Q3_K_M = 14.6GB (very comfortable fit)
- **Key benchmarks:** AIME 91.6, TAU2 79.5, GPQA 75.2, MMLU-Pro ~60 (weakness)
- **Context:** 128K | **License:** MIT
- **Source:** unsloth/GLM-4.7-Flash-GGUF

### 4. HauhauCS/Qwen3.5-35B-A3B-Uncensored-Aggressive
- **Why:** Most popular uncensored variant (815K downloads, 1230 likes). Same base as our model but different uncensoring approach and NO Claude distillation. Direct comparison.
- **Params:** 35B total / 3B active
- **GGUF:** Q3_K_S ~15.3GB (same as our reference)
- **Context:** 262K (full, not reduced to 8K like Claude-distilled)
- **Source:** mradermacher i1 GGUF or unsloth GGUF

### 5. Qwen3.5-35B-A3B (Official Base - No Fine-tune)
- **Why:** The pure base model. Benchmarks: MMLU-Pro 85.3, GPQA 84.2. Our reference model's GPQA of 60% vs base's 84.2 suggests Claude distillation + abliteration may have degraded it.
- **Params:** 35B total / 3B active
- **GGUF:** Q3_K_S = 15.3GB
- **Context:** 262K | **Multimodal:** Yes
- **Source:** unsloth/Qwen3.5-35B-A3B-GGUF

---

## TIER 2: Strong Contenders (Worth testing if Tier 1 shows promise)

### 6. Qwopus3.5-27B-v3 (Jackrong - Claude Opus distilled into dense 27B)
- **Why:** 97.56% HumanEval. Best coding distillation. Dense 27B (not MoE) so slower but potentially higher quality per active param.
- **Params:** 27B dense (all active)
- **GGUF:** ~16GB at Q4, ~13-14GB at Q3
- **Key benchmarks:** HumanEval 97.56%, Plus 95.73%
- **Context:** 8K (reduced by LoRA training)
- **Note:** We already tested qwopus-27B at ~15s for "2+2" - very slow on our hardware. Dense 27B is memory-bandwidth-bound.
- **Source:** Jackrong/Qwopus3.5-27B-v3 + mradermacher i1

### 7. Qwen3-Coder-30B-A3B (Qwen)
- **Why:** Purpose-built coding MoE. SWE-bench 51.6. 256K context.
- **Params:** 30.5B total / 3.3B active
- **GGUF:** Q3_K_S = 13.3GB (very comfortable)
- **Context:** 256K (extendable to 1M)
- **Source:** unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF

### 8. Huihui-Qwen3.5-35B-A3B-abliterated (Plain, No Claude Distillation)
- **Why:** Pure abliteration without Claude distillation. 294 likes vs 36 for Claude version. Community seems to prefer it. Full 262K context preserved.
- **Params:** 35B total / 3B active
- **GGUF:** ~15.3GB at Q3_K_S
- **Source:** mradermacher/Huihui-Qwen3.5-35B-A3B-abliterated-i1-GGUF

### 9. Qwen3.5-9B (Dense - Tiny Giant)
- **Why:** MMLU-Pro 82.5% at just 5.7GB. Beats 120B models. Absurd quality/size. Could be useful as a fast secondary model.
- **Params:** 9B dense
- **GGUF:** Q4_K_M = 5.7GB
- **Key benchmarks:** MMLU-Pro 82.5, GPQA 81.7, AIME 91.3
- **Context:** 262K | **Multimodal:** Yes
- **Source:** unsloth/Qwen3.5-9B-GGUF

### 10. Huihui3.5-67B-A3B (Merged MoE - Brand New)
- **Why:** Brand new (1 day old). 512 experts merged from 2x 35B models. Still only 3B active. If it fits, could be a hidden gem.
- **Params:** 68B total / 3B active
- **GGUF:** Unknown size. Likely ~30-35GB at Q3 (might NOT fit). Need to check.
- **Source:** huihui-ai/Huihui3.5-67B-A3B-abliterated

---

## TIER 3: Niche/Specialist (Test if time permits)

### 11. GLM-4.7-Flash-Uncensored-Heretic-NEO-CODE-Imatrix-MAX
- **Why:** Coding-optimized abliterated GLM. ~14GB at Q4.
- **Source:** HuggingFace (heretic variants)

### 12. Nemotron-3-Nano-30B-A3B (NVIDIA)
- **Why:** Unique Mamba-2 hybrid. Best long-context (RULER@1M: 86.3). But Q3 = 20GB.
- **Params:** 31.6B total / 3.5B active
- **GGUF:** Q3_K_M = 20GB (fits but tight)
- **Source:** unsloth/Nemotron-3-Nano-30B-A3B-GGUF

### 13. Phi-4-reasoning 14B (Microsoft)
- **Why:** Best math/STEM specialist in dense category. 8.9GB at Q4.
- **Context:** 16K (limited)
- **Source:** unsloth or bartowski GGUF

### 14. Mistral Small 3.2 24B (Mistral)
- **Why:** Strong instruction following. 13.1GB at Q4.
- **Context:** 128K
- **Source:** unsloth/Mistral-Small-3.2-24B-Instruct-GGUF

### 15. Qwopus3.5-9B-v3 (Small Claude distillation)
- **Why:** 87.8% HumanEval in just 9B. Claude reasoning in small package.
- **Source:** Jackrong or huihui-ai variants

### 16. Qwen3-30B-A3B (Predecessor MoE)
- **Why:** Proven model. Q4_K_M = 18.6GB. Good baseline comparison to Qwen3.5.
- **Source:** unsloth/Qwen3-30B-A3B-GGUF

---

## TIER 4: Skip (Too large, too old, or outperformed)

| Model | Reason to Skip |
|-------|----------------|
| Llama 4 Scout (109B) | 39.7GB at Q2 - doesn't fit |
| Qwen3.5-122B-A10B | 72GB+ at Q4 - doesn't fit |
| Qwen3-Next-80B-A3B | Only fits at Q1-Q2, too degraded |
| Mixtral 8x7B (47B) | Legacy, 12.9B active = slow |
| Phi-3.5-MoE (42B) | Outperformed by newer models |
| DeepSeek-V2-Lite (16B) | Much weaker than modern models |
| Gemma-4-31B Dense | 18.7GB Q4 + overhead = tight, and MoE version is better |
| Qwen3.5-27B Dense | 16.4GB + overhead = tight, slower than MoE |

---

## Quick-Probe Testing Plan

### Phase 1: Speed + Sanity Check (1-2 questions per model)
For each candidate, run:
1. **"What is 2+2?"** - measure latency/speed (tok/s)
2. **A medium MMLU-style question** - check basic reasoning works

Filter out: models that crash, take >30s for trivial questions, or produce gibberish.

### Phase 2: Mini-Benchmark (5-10 questions per survivor)
Run a mix of:
- 3x MMLU (general knowledge)
- 2x MATH (reasoning)
- 2x HumanEval (coding)
- 2x DROP (reading comprehension)
- 1x GPQA (graduate-level)

### Phase 3: Full Benchmark (20+ questions per top 3-5 models)
Same benchmark suite as our reference model for fair comparison.

---

## Key Research Insights

1. **Our GPQA score (60%) is suspiciously low** vs base model's 84.2%. The Claude distillation + abliteration pipeline may have degraded specialized reasoning. Testing the pure base model (#5) will reveal this.

2. **Arena ELO vs benchmarks diverge.** Gemma-4-26B-A4B has higher Arena ELO (1441) than Qwen3.5-35B-A3B (1400) despite lower raw benchmarks. This suggests better human-perceived quality.

3. **The 60-70% rule:** Model should be ≤60-70% of total RAM. For 32GB = target ≤19-22GB. Our Q3_K_S at 15.3GB is well within this.

4. **MoE dominates 2026.** 7 of top 10 models use MoE. The 3B-active class is the sweet spot for 32GB Macs.

5. **Abliteration has costs.** Community reports intelligence degradation and context instability. The plain base model may outperform abliterated variants on benchmarks.

6. **Memory bandwidth > chip generation** for LLM speed. M4 base (120 GB/s) is slower than M3 Max (400 GB/s).
