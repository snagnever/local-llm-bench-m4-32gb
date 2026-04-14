# GLM-5.1 Research

## Release Date
- **February 17, 2026** (arXiv: 2602.15763)
- Paper: "GLM-5: from Vibe Coding to Agentic Engineering"
- Very new - days old as of April 2026
- GLM-5.1 is an incremental update over GLM-5 (same architecture, improved scores)

## Model Size
- **Total params**: 754B (paper says 744B, HF card says 754B - likely 744B + embeddings)
- **Active params per token**: ~40B (per paper; 8 of 256 routed experts active per token + 1 shared expert)
- **Architecture**: MoE with Dynamic Sparse Attention (glm_moe_dsa)
- **Experts**: 256 routed + 1 shared, top-8 selected per token
- **Layers**: 78
- **Hidden size**: 6,144
- **Context length**: 202,752 tokens
- **License**: MIT

## Can It Fit on a 32GB Mac?
**NO.** Not even close.
- Smallest GGUF (IQ1_M, 1-bit): **206 GB**
- Smallest 4-bit GGUF (IQ4_XS): **361 GB**
- BF16: **1,510 GB**
- This model is 754B total params - no quantization can compress it to 32GB

MLX community has quantized versions but even 4-bit MLX is ~270GB+ (baa-ai/GLM-5.1-RAM-270GB-MLX).

## Smaller Variants
**NO smaller GLM-5.x variants exist** (no mini, flash, small, or lite).

The GLM family smaller models are from the previous generation:
- **GLM-4.7** - 358B total (MoE)
- **GLM-4.7-Flash** - 30B total (30B-A3B MoE) -- this is the one we test
- **glm-edge-v-5b** - 5B (vision model, unrelated)

No GLM-5-mini or GLM-5-flash has been announced.

## GGUF Availability
| Provider | Available? | Notes |
|----------|-----------|-------|
| **unsloth** | YES | unsloth/GLM-5.1-GGUF - full range IQ1_M to BF16 (206GB - 1510GB) |
| **ubergarm** | YES | ubergarm/GLM-5.1-GGUF |
| **DevQuasar** | YES | DevQuasar/zai-org.GLM-5.1-GGUF |
| **bartowski** | NO | Not found |
| **mradermacher** | NO | Not found |

All GGUFs are enormous (smallest = 206GB). Not practical for consumer hardware.

## Comparison: GLM-5.1 vs GLM-4.7-Flash

| Aspect | GLM-5.1 | GLM-4.7-Flash |
|--------|---------|---------------|
| Total params | 754B | 30B |
| Active params | ~40B | ~3B |
| Context | 202K | 131K |
| License | MIT | MIT |
| SWE-Bench | 58.4% (Pro) | 59.2% (Verified) |
| AIME | 95.3% (2026) | 91.6% (2025) |
| GPQA | 86.2% | 75.2% |
| HLE | 31.0% | 14.4% |
| BrowseComp | 68.0% | 42.8% |
| Runnable locally? | No (206GB+) | Yes (~18GB Q4) |
| Release | Feb 2026 | Jan 2025 |

GLM-5.1 is 25x larger and significantly better on reasoning/knowledge benchmarks, but GLM-4.7-Flash is a completely different weight class (designed as a small fast model).

## Full Benchmark Table (GLM-5.1 vs frontier models)

| Benchmark | GLM-5.1 | Claude Opus 4.6 | GPT-5.4 | Gemini 3.1 Pro | DeepSeek-V3.2 |
|-----------|---------|-----------------|---------|----------------|---------------|
| HLE | 31.0 | 36.7 | 39.8 | **45.0** | 25.1 |
| HLE (w/ Tools) | 52.3 | **53.1** | 52.1 | 51.4 | 40.8 |
| AIME 2026 | 95.3 | 95.6 | **98.7** | 98.2 | 95.1 |
| HMMT Feb 2026 | 82.6 | 84.3 | **91.8** | 87.3 | 79.9 |
| GPQA-Diamond | 86.2 | 91.3 | 92.0 | **94.3** | 82.4 |
| SWE-Bench Pro | **58.4** | 57.3 | 57.7 | 54.2 | - |
| NL2Repo | 42.7 | **49.8** | 41.3 | 33.4 | - |
| Terminal-Bench 2.0 | 63.5 | 65.4 | - | **68.5** | 39.3 |
| CyberGym | **68.7** | 66.6 | - | - | 17.3 |
| BrowseComp | **68.0** | - | - | - | 51.4 |
| tau3-Bench | 70.6 | 72.4 | **72.9** | 67.1 | 69.2 |

GLM-5.1 is competitive with frontier models. Best-in-class on SWE-Bench Pro (58.4%), CyberGym (68.7%), and BrowseComp (68.0%). Trails GPT-5.4 and Gemini 3.1 Pro on math.

## Key Takeaway for LiveBench
GLM-5.1 is a **frontier-class model** but only available via API (z.ai) or massive GPU clusters. Not viable for local testing. No smaller GLM-5.x variants exist yet. We should continue testing GLM-4.7-Flash as our GLM representative, and potentially add GLM-5.1 via API if z.ai provides one.
