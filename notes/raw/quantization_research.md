# Quantization Research: Impact on LLM Quality for 32GB M4 Mac

**Research Date:** 2026-04-08
**Hardware Target:** 32GB M4 Mac (~22GB model budget), MoE models
**Primary Models:** Qwen3.5-35B-A3B, Gemma-4-26B-A4B

---

## Table of Contents

1. [Q3 vs Q4 vs Q5 vs Q6 Quality Degradation](#1-q3-vs-q4-vs-q5-vs-q6-quality-degradation)
2. [Benjamin Marie's Qwen3.5 Quantization Tests](#2-benjamin-maries-qwen35-quantization-tests)
3. [MoE-Specific Quantization](#3-moe-specific-quantization)
4. [APEX Quantization (MoE-Aware Mixed Precision)](#4-apex-quantization-moe-aware-mixed-precision)
5. [Imatrix (Importance Matrix) vs Standard Quantization](#5-imatrix-importance-matrix-vs-standard-quantization)
6. [Unsloth Dynamic 2.0 Quantization](#6-unsloth-dynamic-20-quantization)
7. [MXFP4_MOE Format (noctrex)](#7-mxfp4_moe-format-noctrex)
8. [Optimal Quantization for Our Setup](#8-optimal-quantization-for-our-setup)
9. [KV Cache and Context Window Impact](#9-kv-cache-and-context-window-impact)
10. [Flash Attention on Apple Silicon](#10-flash-attention-on-apple-silicon)
11. [Academic Research on MoE Quantization](#11-academic-research-on-moe-quantization)
12. [Key Conclusions](#12-key-conclusions)
13. [Sources](#13-sources)

---

## 1. Q3 vs Q4 vs Q5 vs Q6 Quality Degradation

### General Quality Retention by Quantization Level

From multiple sources (2025-2026 benchmarks on Llama 3.1 8B, various MoE models):

| Quant Level | Perplexity Retention | Quality vs FP16 | Speed vs Q4_K_M | Notes |
|-------------|---------------------|-----------------|-----------------|-------|
| Q3_K_S | ~88% | -12% | +30% faster | Noticeable quality loss, nuance drops |
| Q4_K_M | ~92% | -8% | Baseline | Community standard, sweet spot |
| IQ4_XS | ~95% | -5% | -20% slower | Better quality, slower inference |
| Q5_K_M | ~95% | -5% | ~same | Diminishing returns over Q4_K_M |
| Q6_K | ~97% | -3% | slightly slower | Marginal gain over Q5 |
| Q8_0 | ~99% | -1% | slower | Near-lossless |

**Critical insight:** A well-designed 4-bit format (Q4_K_M, IQ4_XS) can outperform a naive 5-bit legacy format. The quantization *method* matters more than nominal bit-width alone.

### Perplexity at Higher Quant Levels

For Qwen3.5-35B-A3B specifically (from APEX and Unsloth benchmarks):

| Quant | Size | Perplexity (PPL) | KLD 99.9% |
|-------|------|-------------------|-----------|
| F16 | 64.6 GB | 6.537 | -- |
| Q8_0 | 34.4 GB | 6.533 | 0.1033 |
| Q6_K_XL (Unsloth) | 28.2 GB | -- | 0.1437 |
| Q5_K_XL (Unsloth) | 23.2 GB | 6.5489 | 0.236 |
| Q4_K_XL (Unsloth) | 19.2 GB | 6.5918 | 0.4097 |
| Q3_K_XL (Unsloth) | 16.1 GB | -- | 0.9539 |
| IQ3_S (Unsloth) | 14.1 GB | -- | 1.4193 |
| IQ3_XXS (Unsloth) | 13.1 GB | -- | 1.5296 |
| Q2_K_XL (Unsloth) | 12.0 GB | 7.0438 | 2.9092 |
| IQ2_XXS (Unsloth) | 9.1 GB | 7.716 | 4.2221 |

**Competitor comparison:**
- bartowski Q4_K_M: KLD 0.5771 (worse than Unsloth Q4_K_XL at 0.4097)
- bartowski IQ2_XXS: PPL 9.3427 (much worse than Unsloth IQ2_XXS at 7.716)
- AesSedai IQ3_S: KLD 1.8669 (worse than Unsloth IQ3_S at 1.4193)

### Non-Linear Degradation Pattern

Quality loss is NOT linear across quantization levels:
- **Q8 to Q5:** Very small degradation (~1-3%)
- **Q5 to Q4:** Moderate but manageable (~3-5%)
- **Q4 to Q3:** More noticeable, especially in nuance and instruction-following (~5-12%)
- **Q3 to Q2:** Significant quality cliff, especially for smaller models

IFEval (instruction following) shows an interesting non-monotonic pattern: Q3_K_L reaches 79.14, Q4_K_S increases to 80.26, Q5_0 attains 80.14, but Q6_K drops to 77.63 and Q8_0 remains at 78.79. This confirms that IFEval is sensitive to format details rather than bit-width alone.

### Model Size Affects Quantization Tolerance

Red Hat's study of 500,000+ evaluations found:
- **Larger models (70B+):** Negligible degradation at 4-bit. Over 99% accuracy recovery on OpenLLM Leaderboard.
- **Smaller models (8B):** Slight variability but core semantic meaning preserved.
- **All models tested:** Over 96% recovery on the more challenging OpenLLM v2 benchmark, even at 4-bit.
- **HumanEval coding:** 8-bit achieved 99.9% recovery; 4-bit achieved 98.9%.

### "Bigger Model at Lower Quant" vs "Smaller Model at Higher Quant"

**Consensus: A bigger model at lower quantization almost always beats a smaller model at higher quantization.**

Running a 14B at Q4_K_M typically produces better results than a 7B at Q8_0, because extra parameters carry more knowledge than extra precision preserves.

**Exception:** A smaller model at higher quality beats a larger model at Q2. The breakpoint occurs at extremely aggressive quantization (Q2 and below).

**Practical rule:** Pick the biggest model that fits your RAM at Q4_K_M or above.

---

## 2. Benjamin Marie's Qwen3.5 Quantization Tests

### Source
Benjamin Marie (@bnjmn_marie on X, "The Kaitchup" newsletter) conducted extensive evaluations of Qwen3.5 GGUF quantizations using 750-998 prompts across four benchmarks: LiveCodeBench, GPQA Diamond, MMLU-Pro, and Math500.

### Key Findings

**The original claim: "Qwen3.5 35B A3B GGUF models hold up well, even down to Q3"** -- THIS IS SUPPORTED.

His specific findings by model:

**Qwen3.5-35B-A3B (MoE):** "Several quantized variants nearly match the original model's accuracy" with reasoning enabled. Proved MOST robust to quantization of all Qwen3.5 sizes. Q3 quantizations showed "less than 1 point of accuracy difference" from original weights.

**Qwen3.5-27B:** UD-Q4_K_L, IQ4_XS, and IQ4_NL "perform closely to the original." UD-IQ3_XXS was "good enough." No good Q2 option found.

**Qwen3.5-9B:** UD-Q4_K_L and Q4_K_L "perform closely to the original, consuming only ~6 GB (against ~19 GB for the original)." UD-Q3_K_L was "good enough but saves only one additional GB." Avoid UD-IQ3_XXS and all Q2 at this size.

**Qwen3.5-397B:** "Very robust to quantization. Many weights can be compressed without much impact." Both Q4 and Q3 performed very well.

### Critical MoE Finding

**"Don't quantize the shared expert!"** This was highlighted as a key explanation for performance gaps between quantization providers. Intel's approach (keeping shared experts in 16-bit) outperformed naive uniform quantization.

### Quantization with Thinking Mode

When thinking is enabled, quantized models "think more," making them more likely to hit sequence length limits. Truncation roughly doubled for quantized variants. This is particularly relevant for reasoning-heavy benchmarks.

### KV Cache Quantization

Benjamin Marie tested Q4 KV cache quantization on Qwen3.5 and found "no meaningful accuracy loss" -- but noted "it probably only works for good GGUF versions. If you begin with a poorly quantized Qwen3.5, e.g., Q2, the KV cache is already" degraded.

### His Assessment of Qwen3.5 Robustness

"Qwen3.5 is very robust to Unsloth quantization." TQ1_0 (1-bit!) "preserves the original model's accuracy extremely well." Most degradation appeared on MMLU-Pro (world knowledge), while coding and math remained more robust.

### Abliterated Models Warning

Abliterated 9B models showed "bad results" even at Q6_K, significantly underperforming standard Q4_K_L versions. This suggests abliteration + quantization compounds quality loss.

### Provider Comparison

From the 35B-A3B tweet: "Across the providers I tested, I didn't observe meaningful differences: performance varies by roughly +/-10% relative error." However, Unsloth Dynamic quants consistently showed the best KLD numbers.

---

## 3. MoE-Specific Quantization

### Why MoE Models Degrade Differently

MoE models have fundamentally different quantization characteristics compared to dense models:

1. **Structural Sparsity:** Only 8 of 256 experts (for Qwen3.5-35B-A3B) or 8 of 128 experts (for Gemma-4-26B-A4B) are active per token. Inactive experts contribute NO error to forward passes. This means aggressive quantization of routed experts is tolerable.

2. **Expert Activation Heterogeneity:** Not all experts are equally important. Research shows power-law activation distributions -- a small subset of "core" experts handles the majority of tokens while "niche" experts remain underutilized.

3. **Shared Expert Sensitivity:** The shared expert (always active) has a heavy-tailed weight distribution (kurtosis 13.10 vs 3.41 for routed experts). This means it has many outlier values that are destroyed by aggressive quantization. Shared experts MUST be quantized at higher precision.

4. **Router Sensitivity:** Expert routing is highly sensitive to quantization-induced logit perturbations. Even minor deviations in gate scores can disrupt top-k expert selection, causing tokens to be sent to wrong experts.

### Quantization Impact on Expert Routing

From EAQuant (2025-2026 research):
- Removing router alignment causes 0.9-1.3 percentage point accuracy drops under W3A4 quantization
- Specific tasks like BoolQ can drop from 69.08% to 65.75% without routing preservation
- Constraining KL regularization to top-8 experts yields optimal routing preservation
- Even at W4A4 (4-bit), routing consistency must be actively maintained

### Why MoE Models Are MORE Robust to Weight Quantization

Counter-intuitively, MoE models may actually be MORE robust to weight quantization than dense models of equivalent active size:
- Since only ~3% of experts are active per token, errors in inactive expert weights are dormant
- The effective compute path is only 3B params (for Qwen3.5-35B-A3B), so quantization noise affects fewer parameters per inference
- The router can partially compensate by adjusting which experts are selected

This explains why Benjamin Marie found Qwen3.5-35B-A3B (MoE) more robust than Qwen3.5-9B (dense) at low quantization levels.

### Practical Rule for MoE Quantization

1. **Shared experts:** Keep at Q8_0 or higher
2. **Router/gating weights:** Keep at high precision (Q6_K minimum)
3. **Routed experts:** Can tolerate aggressive quantization (Q4 or even Q3)
4. **Edge layers (first/last ~5):** More sensitive, keep at higher precision
5. **Middle layers:** Most tolerant, can go lowest

---

## 4. APEX Quantization (MoE-Aware Mixed Precision)

### What is APEX?

APEX (Adaptive Precision for EXpert Models) is a MoE-aware mixed-precision quantization technique created by mudler. It exploits three properties unique to MoE models:

1. **Structural sparsity** of routed experts (tolerate aggressive quantization)
2. **Heavy-tailed weight distribution** of shared experts (require higher precision)
3. **Non-uniform layer sensitivity** (edge layers more sensitive than middle)

### The Three-Tier Layer-Wise Precision Gradient

APEX assigns different precision based on layer position using a 5+5 edge boundary:

| Layer Range | Role | APEX Quality | APEX Compact |
|-------------|------|-------------|--------------|
| L0-4, L35-39 | Edge layers | Q6_K routed, Q8_0 shared | Q4_K routed, Q6_K shared |
| L5-9, L30-34 | Near-edge | Q5_K routed, Q8_0 shared | Q3_K routed, Q6_K shared |
| L10-29 | Middle | IQ4_XS routed, Q8_0 shared | Q3_K routed, Q6_K shared |

Testing showed 3+3 edge layers was insufficient protection, while 8+8 and 10+10 "wasted bits on insensitive layers."

### APEX Benchmark Results for Qwen3.5-35B-A3B

| Configuration | Size | PPL | KL Mean | HellaSwag | MMLU | ARC |
|--------------|------|-----|---------|-----------|------|-----|
| F16 | 64.6 GB | 6.537 | -- | 82.5% | 41.5% | 56.9% |
| Q8_0 | 34.4 GB | 6.533 | 0.0046 | 83.0% | 41.2% | 57.9% |
| **APEX Quality** | **21.3 GB** | **6.527** | 0.0114 | 83.0% | 41.2% | 56.2% |
| **APEX I-Quality** | **21.3 GB** | 6.552 | 0.0102 | **83.5%** | **41.4%** | **57.9%** |
| APEX Balanced | 23.6 GB | 6.533 | 0.0088 | 83.0% | -- | -- |
| APEX I-Balanced | 23.6 GB | 6.548 | 0.0078 | 83.0% | -- | -- |
| **APEX Compact** | **16.1 GB** | 6.783 | 0.0469 | 82.5% | -- | -- |
| APEX I-Compact | 16.1 GB | 6.669 | 0.0332 | 81.8% | -- | -- |
| APEX Mini | 12.2 GB | 7.088 | 0.0870 | 81.0% | 41.3% | 57.2% |
| Unsloth UD-Q8_K_XL | 45.3 GB | 6.536 | -- | 82.5% | 41.3% | 57.9% |
| Unsloth UD-Q4_K_L | 18.8 GB | 6.586 | -- | 82.3% | 41.1% | 59.2% |
| bartowski IQ2_M | 11.3 GB | 7.303 | -- | 80.3% | 39.6% | 56.2% |

### The "Beats F16" Claim -- IS IT REAL?

**Yes, the numbers are real, but the explanation matters.**

APEX Quality achieves perplexity of 6.527, which beats both Q8_0 (6.533) and F16 (6.537). The difference is small (0.010) but "consistent across repeated measurements."

**Why does this happen?** The three-tier precision gradient with IQ4_XS middle layers introduces "structured noise that acts as implicit regularization on the wikitext-2 evaluation distribution." This is similar to how dropout improves generalization -- controlled precision reduction in insensitive layers acts as regularization.

**Important caveat:** This is measured on wikitext-2 perplexity. It does NOT mean APEX Quality is "better than F16" in general -- it means the regularization effect happens to help on this specific evaluation benchmark. On other metrics (KL divergence), F16 and Q8_0 are still closer to the reference.

### Key Surprising Finding

**Q8_0 has the highest KL max of any model (14.709)** -- more than 2.4x higher than any APEX tier. This means APEX is actually more stable in worst-case scenarios despite lower average precision. The layered approach avoids the extreme outlier errors that uniform quantization can produce.

### Upgrading Routed Experts is Wasteful

"Upgrading routed expert weights from Q6_K to Q8_0 adds 7.5 GB with zero perplexity improvement." This confirms the insight that routed experts don't need high precision.

---

## 5. Imatrix (Importance Matrix) vs Standard Quantization

### What is Imatrix?

The importance matrix (imatrix) is computed by running a calibration dataset through the model and measuring which weights produce the largest activations. Weights that consistently produce large activations are quantized more carefully (higher precision), while less important weights can be quantized more aggressively.

### How It's Computed

1. Run a calibration dataset (300K to 1.5M tokens) through the model
2. Measure activation magnitudes at each weight
3. Generate an importance score per weight/block
4. During quantization, allocate more bits to high-importance weights

The imatrix dataset does NOT change the model's knowledge, behavior, or writing style -- it only identifies which parameters should retain more precision.

### mradermacher's i1 Quants

mradermacher (nicoboss) is a prolific GGUF quantizer on HuggingFace who produces both standard ("static") and imatrix-weighted ("i1") quantizations.

**His key claims:**
- "Weighted/imatrix quants offer higher quality than static quants at the same model size and resource usage"
- "KL-divergence, same token probability, and top token probability measurements show weighted/imatrix quants are far closer to the original unquantized model compared to static quants"
- "Even training an imatrix with random tokens will result in weighted/imatrix quants superior to same-sized static quants"
- The quality difference "is getting smaller with more bits per weight but a quality difference still remains even for Q6"

**Practical recommendation from mradermacher:**
- "i1-Q5_K_M is the first quant which based on quality measurements must be indistinguishable from the unquantized model for humans for any normal use case"
- He personally uses i1 quants for all models between 8B and 500B parameters

### When Does Imatrix Matter Most?

- **Most impact at low bit-widths:** IQ2_XXS, IQ3_XXS -- where every bit counts
- **Moderate impact at Q3-Q4:** Still measurably better KLD
- **Diminishing but present at Q5-Q6:** Quality difference exists but smaller
- **Even Q6 benefits:** Non-zero improvement detected in measurements

### Imatrix vs Standard: Quantified (Unsloth Qwen3.5-35B-A3B)

Comparing Unsloth (uses calibrated data) vs bartowski (standard):
- Unsloth Q4_K_XL: KLD 0.4097, PPL 6.5918
- bartowski Q4_K_M: KLD 0.5771 (41% worse KLD)
- Unsloth IQ2_XXS: PPL 7.716
- bartowski IQ2_XXS: PPL 9.3427 (21% worse PPL)

The gap widens dramatically at lower bit-widths.

### Imatrix "I-" Variants in APEX

APEX provides both standard and imatrix variants (prefix "I-"):
- I-variants show 10-30% lower KL divergence across all tiers
- I-Quality has best HellaSwag (83.5% vs 83.0%) and best MMLU (41.4% vs 41.2%)
- I-Compact is "the biggest imatrix winner": PPL -0.114, KL max 7.56 to 5.50, MMLU +0.8%

### Risk: Imatrix Overfitting

Unsloth warns that models can become "essentially optimized for that domain" when calibration data is too narrow. They deliberately avoid using their custom calibration dataset when benchmarking KL divergence, instead using standard Wikipedia datasets.

### 5% Slower Inference

Unsloth notes: "Imatrix definitely helps reduce KLD & PPL, at the cost of 5-10% slower inference." This is because I-quant formats (IQ4_XS, IQ3_XXS) use more complex dequantization than K-quants.

---

## 6. Unsloth Dynamic 2.0 Quantization

### What Makes It Different

Unsloth Dynamic 2.0 employs intelligent layer-specific quantization rather than uniform approaches:

1. **Per-layer customization:** "Selectively quantizes layers much more intelligently" -- different layers get different quantization types
2. **Model-specific:** "Layers quantized in Gemma 3 differ significantly from those in Llama 4" -- each architecture gets a custom scheme
3. **Large calibration:** Uses ">1.5M tokens" of "high-quality, hand-curated and cleaned data"
4. **MoE-native:** Originally "was effective only for MoE architectures" -- v2.0 expanded to all models

### UD- Format Names

The UD- prefix indicates Unsloth Dynamic quantization. Available formats include:
- **UD-IQ1_M, UD-IQ2_XXS, UD-IQ2_M:** Ultra-low-bit aggressive quants
- **UD-Q2_K_XL:** 2-bit with extended layer optimization
- **UD-IQ3_XXS, UD-IQ3_S, UD-Q3_K_XL:** 3-bit variants
- **UD-IQ4_XS, UD-IQ4_NL, UD-Q4_K_S/M/XL:** 4-bit variants
- **UD-Q5_K_S/M/XL:** 5-bit variants
- **UD-Q6_K, UD-Q6_K_XL:** 6-bit variants
- **UD-Q8_K_XL:** 8-bit dynamic

The "_XL" suffix indicates the largest/highest-quality variant at that bit level.

### Benchmark Improvements

For Gemma 3 (12B):
- Q2_K_XL: KLD reduced from 0.229671 to 0.220937 (~7.5% reduction)
- Q3_K_XL: KLD reduced from 0.087845 to 0.080617 (~8.2% reduction)

For Gemma 3 (27B):
- Q4_K_XL: 71.47% MMLU vs 71.07% for QAT baseline (while being smaller at 15.64GB vs 17.2GB)
- Q2_K_XL: 68.70% vs 67.77% standard

### KL Divergence as the Gold Standard

Unsloth argues "KL Divergence should be one of the gold standards for reporting quantization errors" rather than perplexity, because "output token values can cancel out" in perplexity, masking real distribution differences.

### How It Differs from Imatrix Alone

1. Imatrix identifies important weights; Unsloth Dynamic additionally varies the quantization TYPE per layer
2. Imatrix can overfit to calibration data domain; Unsloth uses diverse hand-curated data and benchmarks on standard Wikipedia
3. Dynamic 2.0 achieves "superior MMLU accuracy while maintaining efficiency advantages"

---

## 7. MXFP4_MOE Format (noctrex)

### What is MXFP4?

MXFP4 (Microscaling Floating Point 4-bit) is a 4-bit floating-point format standardized by the Open Compute Project (OCP) with support from NVIDIA, AMD, Microsoft, and OpenAI. It groups weights into fixed-size blocks, gives each block a shared scale factor, then stores each element as a tiny float.

### noctrex's MXFP4_MOE Approach

noctrex applies MXFP4 specifically to MoE models with a split strategy:

**Standard approach:**
- **MoE expert tensors:** MXFP4 (4-bit floating point)
- **Everything else:** Q8_0 (8-bit integer)

**Higher quality variants:**
- **MXFP4_MOE_BF:** Expert tensors in MXFP4, non-expert tensors in **BF16** (highest quality)
- **MXFP4_MOE_F:** Expert tensors in MXFP4, non-expert tensors in **FP16**

### Qwen3.5-35B-A3B MXFP4_MOE

- **File size:** 22.1 GB (both BF and F variants)
- **Quality:** BF16 variant keeps non-MoE tensors completely unquantized
- **Trade-off:** May be slower than pure quantized versions due to BF16 computation, but highest quality

### Performance

- ~98.5% of full-precision accuracy maintained
- +15% faster prompt processing vs Q4_K_M (due to smaller model size by ~1.3 GB for equivalent quality)
- Requires llama.cpp b8196 or later

### Dynamic Experiments

noctrex is experimenting with using imatrix to dynamically bump important tensors from FP4 to Q8 to FP16, creating an even more adaptive precision allocation.

### Comparison to Other Approaches

MXFP4_MOE is conceptually similar to APEX but uses a floating-point format instead of integer quantization for experts. The BF16 non-expert variant at 22.1 GB is larger than APEX Quality (21.3 GB) but keeps non-expert weights completely unquantized.

---

## 8. Optimal Quantization for Our Setup

### Memory Budget

- **32GB total** - ~8-10GB for macOS + apps = **~22-24GB for model + KV cache**
- **60-70% rule:** Model file should be no more than 60-70% of total memory = 19-22GB
- **KV cache budget:** Need 1-3GB for KV cache depending on context length and model

### Qwen3.5-35B-A3B (35B total, 3B active)

| Quant | Size | Quality | Fits? | Context Budget |
|-------|------|---------|-------|---------------|
| Q6_K_XL | 28.2 GB | Excellent | NO | -- |
| Q5_K_XL | 23.2 GB | Very good | TIGHT | ~1-2GB for KV |
| **APEX Quality** | **21.3 GB** | **Beats F16 PPL** | **YES** | **~3-5GB for KV** |
| **APEX I-Quality** | **21.3 GB** | **Best accuracy** | **YES** | **~3-5GB for KV** |
| Q4_K_XL (Unsloth) | 19.2 GB | Good | YES | ~5-7GB for KV |
| UD-Q4_K_L (Unsloth) | 18.8 GB | Good | YES | ~5-7GB for KV |
| MXFP4_MOE_BF | 22.1 GB | High (unquantized non-MoE) | YES (tight) | ~2-4GB for KV |
| **APEX Compact** | **16.1 GB** | Good | **YES (comfortable)** | **~8-10GB for KV** |
| Q3_K_XL (Unsloth) | 16.1 GB | Acceptable | YES (comfortable) | ~8-10GB for KV |
| APEX Mini | 12.2 GB | Lower | YES | ~12GB for KV |

**Recommendation for Qwen3.5-35B-A3B:**
- **Best quality that fits:** APEX I-Quality (21.3 GB) -- best accuracy benchmarks, fits with 3-5GB for KV
- **Best balance:** Unsloth UD-Q4_K_L (18.8 GB) or APEX Quality (21.3 GB)
- **Maximum context:** APEX Compact (16.1 GB) -- 8-10GB for KV, still good quality
- **If you need more room:** APEX Mini (12.2 GB) -- beats bartowski IQ2_M on every metric

### Gemma-4-26B-A4B (25.2B total, 3.8B active)

| Quant | Size | Fits? | Context Budget |
|-------|------|-------|---------------|
| Q8_0 | 26.9 GB | NO | -- |
| UD-Q6_K_XL | 23.3 GB | TIGHT | ~1-2GB for KV |
| UD-Q5_K_M | 21.2 GB | YES (tight) | ~3-5GB for KV |
| **UD-Q4_K_M** | **16.9 GB** | **YES** | **~7-9GB for KV** |
| **UD-IQ4_XS** | **13.4 GB** | **YES (comfortable)** | **~10-12GB for KV** |
| MXFP4_MOE | 16.6 GB | YES | ~7-9GB for KV |
| UD-Q3_K_M | 12.5 GB | YES (very comfortable) | ~12-14GB for KV |
| UD-IQ2_XXS | 9.88 GB | YES | ~14-16GB for KV |

**Recommendation for Gemma-4-26B-A4B:**
- **Best quality that fits:** UD-Q5_K_M (21.2 GB) if you can spare the room
- **Best balance:** UD-Q4_K_M (16.9 GB) -- recommended by Unsloth for this model
- **Maximum context:** UD-IQ4_XS (13.4 GB) or UD-Q3_K_M (12.5 GB)

**Note:** "Gemma 4 quantization doesn't have that much difference between the bits" -- quality gap between Q3 and Q4 may be smaller than for other models.

### "Q4_K_M Is the Sweet Spot" -- Still True in 2026?

**Mostly yes, but with nuance:**
- Q4_K_M remains the community standard and most-downloaded format
- For MoE models, MoE-aware quantization (APEX, MXFP4_MOE) can beat uniform Q4_K_M at the same or smaller size
- Unsloth Dynamic variants (UD-Q4_K_XL) outperform standard Q4_K_M at similar sizes
- The "sweet spot" has shifted from a single format to **"4-bit range with intelligent allocation"**

---

## 9. KV Cache and Context Window Impact

### KV Cache Memory Per Token by Model

| Model | Bytes/Token | KV at 8K ctx | KV at 32K ctx | KV at 64K ctx | Architecture |
|-------|-------------|-------------|---------------|---------------|-------------|
| Qwen3.5-35B-A3B | 20,480 | ~160 MB | ~640 MB | ~1.3 GB | Hybrid: 10 full-attention + linear-attention layers |
| Qwen3-30B-A3B | 98,304 | ~768 MB | ~3.22 GB | ~6.44 GB | Standard GQA, 48 layers |
| GLM-4.7-Flash | 54,144 | ~422 MB | ~1.77 GB | ~3.55 GB | Compressed MLA architecture |
| Nemotron-3-Nano-30B-A3B | 6,144 | ~48 MB | ~0.20 GB | ~0.40 GB | Only 6 attention layers (Mamba-2 hybrid) |

**Critical insight:** Qwen3.5-35B-A3B has remarkably efficient KV cache thanks to its hybrid attention architecture. Only 10 of 40 layers use full attention; the rest use linear attention with negligible KV storage. This makes it MUCH more memory-efficient for long contexts than its predecessor Qwen3-30B-A3B.

### Actual Measured KV Cache for Qwen3.5-35B-A3B

From real measurements with 4 parallel slots at f16:

| Context Size | KV Cache (f16) | KV Cache (q8_0) |
|-------------|---------------|----------------|
| 4,096 | 80 MiB | ~40 MiB |
| 32,768 | 640 MiB | ~320 MiB |
| 65,536 | 1,280 MiB | ~640 MiB |
| 131,072 | 2,560 MiB | ~1,280 MiB |
| 262,144 | 5,120 MiB | ~2,560 MiB |

### Usable Context on 32GB Mac

**With APEX Quality (21.3 GB model):**
- Available for KV: ~32 - 10 (OS) - 21.3 = ~0.7 GB (VERY tight!)
- With f16 KV cache: ~4K-8K context max
- With q8_0 KV cache: ~8K-16K context
- With TurboQuant turbo3: potentially 32K+ context

**More realistically, the 60-70% rule gives ~22-24GB total budget:**
- 21.3 GB model + 0.64 GB q8_0 KV = 32K context (21.94 GB total) -- fits
- 21.3 GB model + 1.28 GB q8_0 KV = 64K context (22.58 GB total) -- tight but possible

**With APEX Compact (16.1 GB model):**
- Available for KV: ~32 - 10 - 16.1 = ~5.9 GB
- With f16 KV: up to ~262K context (5.12 GB)
- With q8_0 KV: up to ~262K+ context easily
- **This is the better choice if you need long context**

**With Unsloth UD-Q4_K_L (18.8 GB model):**
- Available for KV: ~32 - 10 - 18.8 = ~3.2 GB
- With f16 KV: ~128K-160K context
- With q8_0 KV: ~262K context

### KV Cache Quantization: TurboQuant

TurboQuant (Google Research, March 2026) compresses KV cache to 3-4 bits per element with near-zero quality loss:

| Format | Compression | Quality Impact |
|--------|------------|---------------|
| turbo2 (2-bit) | 6.4x vs FP16 | Measurable loss |
| turbo3 (3-bit) | 4.6-5.1x vs FP16 | ~1.1% PPL increase |
| turbo4 (4-bit) | 3.8x vs FP16 | Negligible |
| q8_0 (baseline) | 2.0x vs FP16 | Near-zero |

With turbo3, a 70B model fits 536K tokens of context in 34GB VRAM (vs 109K with FP16 KV).

On Apple Silicon (M5 Max), turbo3 achieves 98.7-99.5% of q8_0 prefill speed. Decode is 8-36% slower at long contexts due to per-token dequantization overhead.

**KVSplit** is another approach for Apple Silicon that claims 2-3x longer contexts. Available as a llama.cpp fork.

### MoE KV Cache vs Dense Models

"MoE reduces active compute per token but KV efficiency depends on attention architecture, not sparsity alone." A dense transformer with vanilla multi-head attention would require significantly larger KV caches than models using GQA, MLA, or hybrid designs.

For MoE models, the KV cache is NOT reduced by the expert sparsity -- every token still needs its KV entry regardless of which experts were activated. The savings come from architectural choices (GQA, hybrid attention) that happen to be common in modern MoE models.

---

## 10. Flash Attention on Apple Silicon

### Current Status (2026)

- **llama.cpp:** Flash attention is supported on Metal via the `--flash-attn` or `-fa` flag
- **MLX:** Has its own optimized attention implementation
- **Ollama:** Uses llama.cpp's Metal backend (and now MLX as of 0.19)

### Memory Savings

Flash attention reduces peak memory by not materializing the full attention matrix. Instead of O(n^2) memory for the attention matrix, it uses O(n) chunked computation. This primarily helps with:
- **Prompt processing:** Most impactful during prefill of long prompts
- **Not generation:** During autoregressive generation, attention is already efficient

### Known Issues

- **Quantized KV cache + flash attention on Metal:** There have been reports of performance drops -- "both prompt processing and token generation being 1/3 slower when flash attention is enabled and one of the key or value caches is quantized" (llama.cpp issue #8918)
- This may be resolved in newer builds -- check current llama.cpp version

### Practical Impact

For a 32GB M4 Mac:
- Flash attention mainly helps during prefill of long prompts
- Memory savings allow slightly longer context windows
- Speed improvement is most visible at >8K context lengths
- MLX's approach may differ from llama.cpp's Metal flash attention implementation

---

## 11. Academic Research on MoE Quantization

### EAQuant (2025-2026) -- Expert-Aware Quantization

Three innovations for MoE models:
1. **Smoothing aggregation:** Suppresses activation outliers across MoE experts using a unified channel-wise smoothing vector
2. **Routing consistency alignment:** Dual-objective calibration minimizing both logit reconstruction error AND KL divergence between full-precision and quantized routing distributions
3. **Calibration data balance:** Ensures sparsely activated experts get adequate calibration

**Results:** 1.15-2.28% average accuracy improvement over baseline across three MoE architectures. Particularly strong gains in reasoning tasks.

Source: https://arxiv.org/abs/2506.13329

### MxMoE (ICML 2025) -- Mixed-Precision MoE Quantization

Two key insights:
1. Linear blocks exhibit varying quantization sensitivity
2. Divergent expert activation frequencies create heterogeneous computational characteristics

**Results:**
- 2.4x lower wikitext-2 perplexity than GPTQ at 2.25-bit
- Up to 3.4x speedup over full precision
- Up to 29.4% speedup over uniform quantization at equivalent accuracy with 5-bit

Source: https://arxiv.org/abs/2505.05799

### "Super Experts" Research (2025)

Research has identified "super experts" -- a small subset of experts that handle disproportionately many tokens. These should be quantized more carefully than rarely-activated "niche" experts.

### MoE-Compression (2025)

Investigates how compression error of individual experts propagates to overall model accuracy. Key finding: errors in frequently-activated experts have outsized impact on model quality.

### Dynamic Expert Quantization (2025)

Proposes maintaining frequently activated experts at high precision while aggressively quantizing rarely used experts. This adaptive strategy performs better than uniform quantization.

### Global Router Fine-Tuning (2025-2026)

A limitation of per-layer expert importance estimation is that it misses global cross-layer importance. Recent work proposes efficient global router fine-tuning that adapts routers to quantized experts, enabling optimal routing decisions post-quantization.

Source: https://openreview.net/forum?id=wAc718O8UM

---

## 12. Key Conclusions

### For Our 32GB M4 Mac Setup

1. **Qwen3.5-35B-A3B at APEX I-Quality (21.3 GB)** is the optimal quality choice. It literally beats F16 perplexity while being 3x smaller, and fits with room for ~32K context using q8_0 KV cache.

2. **If we need longer context (64K+):** Use APEX Compact (16.1 GB) or Unsloth UD-Q4_K_L (18.8 GB). Quality is still excellent.

3. **Gemma-4-26B-A4B at UD-Q4_K_M (16.9 GB)** is the recommended quant. Leaves plenty of room for KV cache. Quality difference between quant levels is reportedly small for this model.

4. **Q3 is viable for MoE models.** Benjamin Marie's testing confirms less than 1 point accuracy difference at Q3 for Qwen3.5-35B-A3B. MoE architecture provides natural quantization resilience through expert sparsity.

5. **Don't quantize the shared expert aggressively.** This is the single most important MoE quantization rule. APEX handles this automatically (Q8_0 for shared experts). Standard uniform quants (Q3_K_M, Q4_K_M) do NOT distinguish shared from routed experts.

6. **Imatrix always helps.** Use i1 quants (mradermacher) or Unsloth Dynamic over standard/bartowski quants. The gap is largest at low bit-widths but exists even at Q6.

7. **KV cache is very efficient for Qwen3.5-35B-A3B.** Thanks to hybrid attention (10 full-attention layers), KV at 32K context is only ~640 MiB (f16) or ~320 MiB (q8_0). This model is exceptionally context-friendly.

8. **Q4_K_M is still a good default** but MoE-aware formats (APEX, MXFP4_MOE) are strictly better at the same size. If available for your model, prefer them.

9. **Bigger model at lower quant beats smaller model at higher quant** -- with the caveat that Q2 and below is the exception. A 35B MoE at Q3 beats a 9B dense at Q8.

10. **KV cache quantization (q8_0) is essentially free quality** on Qwen3.5. Benjamin Marie found "no meaningful accuracy loss" from Q4 KV cache quantization. Use `--cache-type-k q8_0 --cache-type-v q8_0` flags.

---

## 13. Sources

### Benjamin Marie's Qwen3.5 Quantization Research
- [Summary of Qwen3.5 GGUF Evaluations | The Kaitchup](https://kaitchup.substack.com/p/summary-of-qwen35-gguf-evaluations)
- [Qwen3.5 Quantization: Similar Accuracy, More Thinking | The Kaitchup](https://kaitchup.substack.com/p/qwen35-quantization-similar-accuracy)
- [Choosing a GGUF Model: K-Quants, I-Quants, and Legacy Formats | The Kaitchup](https://kaitchup.substack.com/p/choosing-a-gguf-model-k-quants-i)
- [KV Cache of Small MoEs | The Kaitchup](https://kaitchup.substack.com/p/the-kv-cache-of-small-moes-qwen3)
- [@bnjmn_marie: Qwen3.5 GGUF quality tweet](https://x.com/bnjmn_marie/status/2028559740347781431)
- [@bnjmn_marie: Complete Unsloth evaluation](https://x.com/bnjmn_marie/status/2025951400119751040)
- [@bnjmn_marie: 27B evaluation](https://x.com/bnjmn_marie/status/2029227800574447958)
- [@bnjmn_marie: 9B evaluation](https://x.com/bnjmn_marie/status/2029582026450280792)
- [@bnjmn_marie: KV cache quantization](https://x.com/bnjmn_marie/status/2031063219041669390)
- [@bnjmn_marie: INT4 vs NVFP4 vs FP8 evaluation](https://x.com/bnjmn_marie/status/2032350411646992894)

### APEX Quantization
- [APEX Technical Report | GitHub](https://github.com/mudler/apex-quant/blob/main/paper/APEX_Technical_Report.md)
- [APEX-Quant GitHub Repository](https://github.com/mudler/apex-quant)
- [Qwen3.5-35B-A3B-APEX-GGUF | HuggingFace](https://huggingface.co/mudler/Qwen3.5-35B-A3B-APEX-GGUF)
- [Gemma-4-26B-A4B-it-APEX-GGUF | HuggingFace](https://huggingface.co/mudler/gemma-4-26B-A4B-it-APEX-GGUF)

### Unsloth Dynamic 2.0
- [Unsloth Dynamic 2.0 Documentation](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)
- [Unsloth Dynamic v2.0 Blog Post](https://unsloth.ai/blog/dynamic-v2)
- [Qwen3.5 GGUF Benchmarks | Unsloth](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)
- [Qwen3.5-35B-A3B-GGUF | HuggingFace](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF)
- [Gemma-4-26B-A4B-it-GGUF | HuggingFace](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)

### MXFP4_MOE (noctrex)
- [Qwen3.5-35B-A3B-MXFP4_MOE-GGUF | HuggingFace](https://huggingface.co/noctrex/Qwen3.5-35B-A3B-MXFP4_MOE-GGUF)
- [GLM-4.7-Flash-MXFP4_MOE-GGUF | HuggingFace](https://huggingface.co/noctrex/GLM-4.7-Flash-MXFP4_MOE-GGUF)

### Imatrix / mradermacher
- [Is Imatrix better than regular quants? | HuggingFace Discussion](https://huggingface.co/mradermacher/model_requests/discussions/1436)
- [Imatrix MMLU-Pro differences | HuggingFace Discussion](https://huggingface.co/mradermacher/Meta-Llama-3-8B-Instruct-i1-GGUF/discussions/1)
- [GGUF Quantization with Imatrix | The Kaitchup](https://kaitchup.substack.com/p/gguf-quantization-with-imatrix-and-q-quants)
- [Imatrix overfitting discussion | llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/5263)

### KV Cache and Memory
- [Scaling Qwen3.5 from 4K to 65K Context | lilting.ch](https://lilting.ch/en/articles/qwen35-ctx-kv-cache-vulkan-optimization)
- [TurboQuant: Extreme KV Cache Quantization | llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/20969)
- [KVSplit: Run 2-3x longer contexts on Apple Silicon | HN](https://news.ycombinator.com/item?id=44009321)
- [KV Cache Size Calculator](https://lmcache.ai/kv_cache_calculator.html)

### Academic Papers on MoE Quantization
- [EAQuant: Expert-Aware Quantization for MoE Models](https://arxiv.org/abs/2506.13329)
- [MxMoE: Mixed-precision Quantization for MoE (ICML 2025)](https://arxiv.org/abs/2505.05799)
- [Towards Global Expert-Level Mixed-Precision Quantization for MoE LLMs | OpenReview](https://openreview.net/forum?id=wAc718O8UM)
- [Dynamic Expert Quantization for Scalable MoE Inference](https://arxiv.org/html/2511.15015v1)
- [MoE-Compression: How Expert Compression Error Affects Inference](https://arxiv.org/html/2509.07727v1)
- [A Survey on Inference Optimization for MoE Models | ACM](https://dl.acm.org/doi/10.1145/3794845)
- [Unveiling Super Experts in MoE LLMs](https://arxiv.org/html/2507.23279v1)

### General Quantization Research
- [Red Hat: 500K+ evaluations on quantized LLMs](https://developers.redhat.com/articles/2024/10/17/we-ran-over-half-million-evaluations-quantized-llms)
- [GGUF Quantization: Quality vs Speed on Consumer GPUs](https://dasroot.net/posts/2026/02/gguf-quantization-quality-speed-consumer-gpus/)
- [Blind testing different quants | llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/5962)
- [Which Quantization Should I Use? A Unified Evaluation | arXiv](https://arxiv.org/pdf/2601.14277)
- [GGUF Quantization Guide | tonisagrista.com](https://tonisagrista.com/blog/2026/quantization/)
- [LLM Quantization Explained | knightli.com](https://www.knightli.com/en/2026/04/05/llm-quantization-guide-fp16-q4-q2/)
- [MXFP4 Revolution Guide](https://gigxp.com/the-mxfp4-revolution/)

### Flash Attention and Apple Silicon
- [Flash Attention on GGML/GGUF | Ollama Issue](https://github.com/ollama/ollama/issues/4051)
- [Quantized KV cache performance on Apple Silicon | llama.cpp Issue](https://github.com/ggml-org/llama.cpp/issues/8918)
- [TurboQuant in Practice | SOTAAZ](https://www.sotaaz.com/post/turboquant-practical-en)
- [Qwen3 235B and 30B Quant Benchmarking Roundup | GitHub Gist](https://gist.github.com/ubergarm/0f9663fd56fc181a00ec9f634635eb38)

### Quantization Benchmark Roundups
- [Qwen3.5 35B A3B GGUF Discussion | HuggingFace](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/discussions/10)
- [GGUF Quantizations Overview | GitHub Gist](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9)
- [llama.cpp Quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
