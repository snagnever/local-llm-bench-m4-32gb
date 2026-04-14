# Twitter/X Research: Best Small/Medium LLMs for Local Inference on Apple Silicon

**Research Date:** 2026-04-08

---

## 1. Huihui-AI / Huihui Qwen Models

### Overview
huihui.ai (@support_huihui on X) is a prolific creator of abliterated (uncensored) models, primarily based on the Qwen3.5 architecture.

### Released Models (Abliterated Qwen3.5 Series)
- **Huihui-Qwen3.5-0.8B-abliterated** - Smallest variant
- **Huihui-Qwen3.5-2B-abliterated** - Uncensored via abliteration
- **Huihui-Qwen3.5-4B-abliterated** - Fine-tuned using TRL, trained on non-think mode dataset
- **Huihui-Qwen3.5-9B-abliterated** - Popular mid-size option
- **Huihui-Qwen3.5-27B-abliterated** - Dense model, excellent tool calling performance
- **Huihui-Qwen3.5-35B-A3B-abliterated** - MoE model, uncensored

### Special Models
- **Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated** - Abliterated version of Jackrong's Claude Opus distillation into the Qwen 35B-A3B MoE model
  - Source: https://x.com/support_huihui/status/2033633701595337030
- **Huihui3.5-67B-A3B** - A merged MoE model combining Qwen/Qwen3.5-35B-A3B and Hcompany/Holo3-35B-A3B. Experts expanded from 256 to 512 while still activating only 8 experts per token
  - Source: https://x.com/support_huihui/status/2040330505783218465
- **Huihui3.5-97B-A3B** - A "monster model" synthesized from THREE Qwen3.5-35B-A3B models. Described as "playable now"
  - Source: https://x.com/support_huihui/status/2041364960488677805

### Tool Calling
huihui noted that Huihui-Qwen3.5-35B-A3B-abliterated and Huihui-Qwen3.5-27B-abliterated showed "excellent" tool calling performance when tested with llama-serve and opencode.
- Source: https://x.com/support_huihui/status/2027599445794427116

### Ollama Availability
Available via Ollama with tags like:
- `ollama run huihui_ai/qwen3.5-abliterated:0.8B`
- `ollama run huihui_ai/qwen3.5-abliterated:9b`
- `ollama run huihui_ai/qwen3.5-abliterated:9b-Qwopus`
- `ollama run huihui_ai/qwen3.5-abliterated:27b-Qwopus-fp16`
- `ollama run huihui_ai/huihui3.5:67b`

---

## 2. Best Local LLM for Mac / Apple Silicon Recommendations

### By RAM Configuration

**8-16GB Macs:**
- Phi-4 Mini 3.8B - Best reasoning at low RAM
- Qwen 3 8B (Q4_K_M) - Strong instruction following, good coding, solid reasoning
- Gemma 4 E4B (8B params, 4.5B effective) - Fits comfortably on 16GB

**32GB Macs:**
- Qwen 3.5 27B or Gemma 4 26B MoE - Best balance of quality/speed
- Qwen3.5-35B-A3B (MoE) - ~21GB to run, outperforms models 6x its size
- Llama 3.1 70B at Q2_K (~23GB) - Runs but quality reduced at Q2
- In practice ~24-26GB usable for model after system overhead

**48GB+ Macs:**
- Qwen 3 32B (Q4_K_M) - Expert-level responses
- Gemma 4 31B Dense
- Qwen3.5-122B-A10B (MoE, needs aggressive quant)

### Framework Recommendations
- **Ollama** - Fastest path, easiest setup, broadest model support. Now powered by MLX on Apple Silicon (preview)
- **MLX** - Purpose-built for Apple Silicon, 10-20% better on M3/M4 chips
- **llama.cpp** - Via Ollama or standalone, broader model support
- **LM Studio** - Best visual chat UI

### Sources:
- https://apxml.com/posts/best-local-llm-apple-silicon-mac
- https://insiderllm.com/guides/best-local-llms-mac-2026/
- https://dev.to/bspann/running-llms-locally-on-macos-the-complete-2026-comparison-48fc

---

## 3. Qwen3.5 MoE / Qwen3.5-35B-A3B Discussions

### Official Announcement
Alibaba expanded the Qwen3.5 family with three new medium models:
- **Qwen3.5-27B** (Dense) - Artificial Analysis Intelligence Index: 42
- **Qwen3.5-122B-A10B** (MoE) - Intelligence Index: 42
- **Qwen3.5-35B-A3B** (MoE) - Intelligence Index: 37, most intelligent model with ~3B active params
- Source: https://x.com/ArtificialAnlys/status/2027489442697777245

### Key Tweets

**@Alibaba_Qwen (Official):**
"Qwen3.5-35B-A3B now surpasses Qwen3-235B-A22B-2507 and Qwen3-VL-235B-A22B - a reminder that better architecture, data quality [matter more than size]"
- Source: https://x.com/Alibaba_Qwen/status/2026339351530188939

**@itsPaulAi:**
"Wow they did it. In 6 months they've trained a model which is 6.7x smaller than the previous one, better in all benchmarks, available locally on a laptop. We're just at the very beginning of local LLMs."
- Source: https://x.com/itsPaulAi/status/2026354821389799720

**@SebAaltonen:**
"Qwen3.5-35B-A3B is seriously impressive. Runs on a consumer grade 24GB GPU locally and beats both GPT-5 mini and Claude Sonnet 4.5 in quality."
- Source: https://x.com/SebAaltonen/status/2026652559532814490

**@lmstudio:**
"Qwen3.5-35B-A3B is now available in LM Studio! This model outperforms previous Qwen models that are more than 6x its size. Requires about ~21GB to run locally."
- Source: https://x.com/lmstudio/status/2026406333575049485

### GGUF Quantization
@bnjmn_marie: "Qwen3.5 35B A3B GGUF models hold up well, even down to Q3. Across the providers I tested, I didn't observe meaningful differences: performance varies by roughly +/-10% relative error."
- Source: https://x.com/bnjmn_marie/status/2028559740347781431

### Benchmarks
- GPQA Diamond (graduate-level reasoning): 84.2
- Instruction following: 91.9
- APEX Quality quantization: best perplexity of any quantization (6.527, beats F16's 6.537) at just 21.3 GB
- Architecture: 256 experts per MoE layer, routing 8 experts + 1 shared expert per token across 40 transformer layers

### Speed on Apple Silicon
| Hardware | Model/Quant | tok/s |
|----------|-------------|-------|
| MacBook M5 Max 128GB | Qwen3.5-35B-A3B Q6 | 74 tok/s |
| M1 Ultra Studio 64GB | Qwen3.5-35B-A3B | 44 tok/s |
| M3 Max | Qwen3.5-35B-A3B | ~9-10 tok/s |

Sources:
- https://x.com/nix_eth/status/2032879242737045612
- https://x.com/blizaine/status/2026624163230695558

---

## 4. Abliterated Model Discussions

### What is Abliteration?
A mechanistic interpretability technique that surgically removes a model's refusal behavior by identifying and removing the "refusal direction" in the model's activation space. No retraining needed.

### Key Abliterated Models Discussed on X

**Gemma 4 Abliterated (April 2026):**
- Appeared within days of Gemma 4's April 2 launch
- 93.7% of refusal behaviors removed
- Techniques used: MPOA (Magnitude-Preserving Oblique Ablation), Biprojection + EGA, Arbitrary-Rank Ablation (ARA)
- @BrianRoemmele argued alignment in Gemma 4 is "beyond a ridiculous form of performative theater"
  - Source: https://x.com/BrianRoemmele/status/2041245561748336923
- Variants: gemma-4-31B-it-heretic, gemma-4-26B-A4B-it-heretic (MoE), gemma-4-E4B-Uncensored

**Gemma 3 Abliterated:**
- @maximelabonne: "Gemma 3 was much more resilient to refusal removal than other models like Qwen 2.5. I experimented with different recipes and improved the abliteration technique."
  - Source: https://x.com/maximelabonne/status/1901581470717608215

**Qwen3.5 Abliterated (huihui-ai):**
- Full series from 0.8B to 35B-A3B (see Section 1)
- Performance: Both abliterated and regular Qwen3.5 performed nearly identically on coding tasks, with no degradation in coding ability
- 262k native context window maintained

**Snowpiercer-15B-v4-absolute-heresy:**
- Uncensored Mistral-based model, "abliterated from restrictions"
  - Source: https://x.com/HuggingModels/status/2020877890497840535

**Notable Community Sentiment:**
@HuggingModels: "With over 44k downloads, the community clearly values its unique blend: Claude-level reasoning in an uncensored package."
- Source: https://x.com/HuggingModels/status/2038398354489741539

---

## 5. 32GB Mac / Apple Silicon LLM Recommendations

### Practical Limits
- 32GB system = ~24-26GB usable for model (after macOS overhead)
- 30B Q4 models run comfortably
- 70B models possible at aggressive Q2 quantization (~23GB) but quality notably reduced

### Top Picks for 32GB

1. **Qwen3.5-35B-A3B** (MoE, ~21GB) - Best bang for buck, beats GPT-5 mini and Claude Sonnet 4.5 per Sebastian Aaltonen
2. **Qwen3.5-27B** (Dense) - Consistent balance between memory, quality, stability
3. **Gemma 4 26B MoE** - Good balance, ~20-30 tok/s on M4
4. **DeepSeek R1 14B** (Q6_K, ~11GB) - Strong reasoning in compact package
5. **Qwen 3 14B** (Q4_K_M) - Great for M4 Pro 24GB configs

### Speed Reference on Apple Silicon
| Hardware | Model | tok/s |
|----------|-------|-------|
| M5 Max 128GB | Qwen3.5-35B-A3B Q6 | 74 |
| M5 Max 128GB | Qwen3-Coder-Next 6-bit | 67 |
| M5 Max 128GB | Llama 3.3 8B Q4 | 99 |
| M5 Max 128GB | Nemotron-3 Super Q4 | 24 |
| M1 Ultra 64GB | Qwen3.5-35B-A3B | 44 |
| M4 (general) | Gemma 4 26B MoE | 20-30 |

Source: https://x.com/nix_eth/status/2032879242737045612

---

## 6. Mixture of Experts (MoE) Small Model Discussions

### Trending MoE Models on X (2026)

**Qwen3.5-35B-A3B** - 35B total, 3B active
- The breakout star. Surpasses Qwen3-235B (6.7x larger)
- 256 experts, 8 active + 1 shared per token

**NVIDIA Nemotron 3 Nano** (30B-3A) - Hybrid SSM MoE architecture
- Novel architecture, purpose-built for edge/local
- Source: https://x.com/ctnzr/status/2000567572065091791

**IBM Granite 4.0 Tiny** - 7B MoE, 1B active
- Runs on iPhone 17 Pro, iPhone Air, iPads
- Source: https://x.com/LocallyAIApp/status/1975969475406049757

**StepFun Step 3.5 Flash** - 196B total, 11B active
- Frontier reasoning meets extreme efficiency
- Source: https://x.com/StepFun_ai/status/2018370831538180167

**Trinity Large** - 400B MoE, 13B active
- Fully open source, trained on 17T tokens
- On par with GLM-4.5 Base, significantly faster due to sparsity
- Source: https://x.com/samsja19/status/2016283855888773277

**NVIDIA Nemotron 3 Super** - 120B MoE, 12B active
- For multi-agent AI systems
- Source: https://x.com/togethercompute/status/2031831368339243454

**Tencent HY 2.0** - 406B total, 32B active, 256K context
- Source: https://x.com/TencentHunyuan/status/1996948083377332614

**Baidu ERNIE 5.0** - 2.4T MoE, <3% active per inference
- Source: https://x.com/Baidu_Inc/status/2014252300018254054

**K-EXAONE** (LG AI) - 236B MoE, 23B active, 256K context
- Hybrid attention + Multi-Token Prediction for 1.5x throughput
- Source: https://x.com/HuggingPapers/status/2008451515262841114

**Huihui3.5-67B-A3B** - 512 experts (merged from two 35B-A3B models)
**Huihui3.5-97B-A3B** - Synthesized from THREE 35B-A3B models

### Key Trend
MoE models are dominating the local LLM space in 2026. The ability to have 35B+ total parameters but only 3B active per token means you get the knowledge of a large model at the inference cost of a tiny one.

---

## 7. Recent Model Releases Generating Excitement (Especially MoE)

### Most Excited-About Models (X/Twitter Sentiment, Early 2026)

1. **Qwen3.5-35B-A3B** (Feb 2026) - Universally praised. "Beats GPT-5 mini and Claude Sonnet 4.5 in quality" on a consumer GPU
2. **Gemma 4** (April 2, 2026) - Apache 2.0 license, matches Claude Sonnet 4.5 Thinking, abliterated within days
3. **Qwen3-Coder-Next** (Feb 2026) - 80B MoE, excels at agentic coding, 256K context
4. **NVIDIA Nemotron 3 family** (Feb 2026) - Nano/Super/Ultra, hybrid SSM-MoE
5. **Trinity Large** (Jan 2026) - Fully open 400B MoE, 13B active
6. **Flash-MoE engine** - Running ~400B parameter models on iPhone 17 Pro
   - @TeksEdge: "Yes, running a ~400B parameter AI on an iPhone is 100% REAL"
   - Source: https://x.com/TeksEdge/status/2036116516123595184

### Excitement Drivers
- MoE efficiency: More knowledge, less compute
- Apple Silicon optimization: MLX + Ollama making local inference practical
- Apache 2.0 licensing: Both Gemma 4 and Qwen3.5 fully permissive
- Abliteration speed: Uncensored versions appearing within hours/days of release

---

## 8. Gemma 4 vs Qwen Comparisons

### Head-to-Head Benchmarks
| Benchmark | Gemma 4 31B | Qwen 3.5 27B |
|-----------|-------------|--------------|
| MMLU Pro | 85.2% | 86.1% |
| GPQA Diamond | 84.3% | 85.5% |
| AIME 2026 Math | 89.2% | - |
| Codeforces ELO | 2150 | - |

### Strengths
- **Gemma 4**: Better for multimodal (image+text), math competition benchmarks, competitive programming
- **Qwen 3.5**: Edge in complex reasoning (hybrid thinking mode), more size options at low end, faster inference (~35 tok/s vs ~25 tok/s on RTX 4090 Q4)

### For Local Mac Use
- Both run well on consumer hardware
- 8GB VRAM: Both Qwen 3.5 8B and Gemma 4 12B viable (with quantization)
- 6GB or less: Qwen 3.5 has more headroom with smaller variants
- Gemma 4's 26B MoE is slower (~11 tok/s) due to routing overhead on GPU

### Licensing
Both Apache 2.0 - No restrictions, full commercial use

### Sources:
- https://www.mindstudio.ai/blog/gemma-4-vs-qwen-3-5-open-weight-comparison
- https://lushbinary.com/blog/gemma-4-vs-llama-4-vs-qwen-3-5-open-weight-model-comparison/
- https://ai.rs/ai-developer/gemma-4-vs-qwen-3-5-vs-llama-4-compared

---

## 9. Claude Distillation / Reasoning Distillation Models

### Qwopus - The Standout Claude Distillation

**Model:** Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled (renamed to "Qwopus 27B")
**Creator:** Jackrong (HuggingFace)

**Key Tweets from @sudoingX (extensive testing thread):**

"spent the entire day testing Qwopus (Claude 4.6 Opus distilled into Qwen 3.5 27B) on a single RTX 3090 through Claude Code. this is my new favourite to host locally. no jinja crashes. thinking mode works natively. 29-35 tok/s. 16.5 GB."
- Source: https://x.com/sudoingX/status/2030344904774402115

"Claude Opus 4.6 reasoning distilled into Qwen 3.5 27B dense, running through Claude's own coding agent (claude code). 29-35 tok/s with thinking mode on. the jinja bug that kills thinking on base Qwen doesn't carry over."
- Source: https://x.com/sudoingX/status/2030237974286192815

"the distilled version doesn't stall and waits for tool outputs, reads them, and self-corrects when something breaks."
- Source: https://x.com/sudoingX/status/2030653597495787544

### Qwopus Benchmark Results (HumanEval 164 tasks)

**Qwopus3.5-27B-v3 vs v2 improvements:**
- Base Pass: 95.12% -> 97.56% (+2.44pp)
- Plus Pass / Strict Overall: 92.68% -> 95.73% (+3.05pp)
- Source: https://x.com/geekbb/status/2039695193960829339

**Qwopus3.5-9B-v3 (smaller variant):**
- Base pass@1: 87.80% (144/164) - outperforms both Qwen3.5-9B base (82.93%) and Claude-Distilled-v2 (82.32%)
- Plus pass@1: 82.93% (136/164)

### Other Claude Distillation Models

**Crow-9B-HERETIC** - 9B param model on Qwen 3.5 architecture, distilled from Claude Opus 4.6
- Source: https://x.com/hackernoon/status/2038794918852800531

**TeichAI - GLM-4.7-Flash** distilled from Claude 4.5 Opus (listed as popular by Unsloth)
- Source: https://x.com/UnslothAI/status/2024847369733325202

**huihui-ai Qwopus variants** - Available on Ollama:
- `huihui_ai/qwen3.5-abliterated:9b-Claude`
- `huihui_ai/qwen3.5-abliterated:9b-Qwopus`
- `huihui_ai/qwen3.5-abliterated:4b-Qwopus`
- Source: https://x.com/support_huihui/status/2035350630039191656

**Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated** - Abliterated version of the MoE Claude distillation
- Source: https://x.com/support_huihui/status/2033633701595337030

### Community Sentiment
@HuggingModels: "With over 44k downloads, the community clearly values its unique blend: Claude-level reasoning in an uncensored package."
- Source: https://x.com/HuggingModels/status/2038398354489741539

@EmadMostaque: "Distilled qwen 32b model is insane performance, will run on 16gb vram ie all new Macs absolutely fine"
- Source: https://x.com/EMostaque/status/1881334699207077937

---

## 10. Open LLM Leaderboard Discussions

### Leaderboard Status (2026)

**Tier Structure:**
- S-tier: GLM-4.7, Kimi K2.5, MiniMax M2.5
- Chinese models dominate top 10 (Qwen leading)

**Key Benchmarks:** MMLU-Pro, GPQA, MuSR, MATH, IFEval, BBH

### Notable X Discussions

**@ClementDelangue (HuggingFace CEO):**
"We burned 300 H100 to re-run new evaluations like MMLU-pro for all major open LLMs! Some learning: Qwen 72B is the king and Chinese open models are dominating overall"
- Source: https://x.com/ClementDelangue/status/1805989925080219927

**@dnhkng (March 2026):**
"I topped the HuggingFace Open LLM Leaderboard without changing a single weight. No training. No merging. No gradient descent. I duplicated 7 middle layers of Qwen2-72B and stitched it back together."
- Source: https://x.com/dnhkng/status/2035681641624920559

**Small Model Leaders:**
- Nemotron Super 49B and Step-3.5-Flash (196B MoE) achieving results that would have required 600B+ params 12 months ago
- GLM-4.7: Highest HumanEval score (94.2) of any model on the leaderboard

### Leaderboard Resources
- https://huggingface.co/open-llm-leaderboard
- https://artificialanalysis.ai/leaderboards/models
- https://llm-stats.com/leaderboards/open-llm-leaderboard

---

## Summary: Top Recommendations for 32GB Mac Apple Silicon (April 2026)

### Best Overall Local Models
1. **Qwen3.5-35B-A3B** - The consensus pick. MoE, ~21GB, beats much larger models, 44+ tok/s on M1 Ultra
2. **Qwen3.5-27B** - Dense alternative if you prefer simpler architecture, very consistent
3. **Gemma 4 31B** - Strong multimodal, Apache 2.0, great for math/code
4. **Qwopus 27B** (Claude-distilled Qwen3.5-27B) - Best for coding agents, Claude-like reasoning at 29-35 tok/s

### Best Uncensored/Abliterated
1. **huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated** - MoE, uncensored, great tool calling
2. **huihui-ai/Huihui-Qwen3.5-27B-abliterated** - Dense, uncensored
3. **Gemma 4 31B heretic/abliterated** - Multiple community variants available
4. **huihui-ai/Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated** - Abliterated Claude distillation into MoE

### Best for Coding
1. **Qwopus3.5-27B-v3** - 97.56% HumanEval base pass, Claude-quality reasoning
2. **Qwen3-Coder-Next** - 80B MoE, purpose-built for agentic coding
3. **Qwen3.5-35B-A3B** - General but strong at code

### Best Efficiency (< 16GB RAM)
1. **Qwen3.5-9B** or its abliterated variant - Sweet spot for 16GB
2. **Gemma 4 E4B** (8B, 4.5B effective) - Fits on any 16GB Mac
3. **Qwopus3.5-9B-v3** - Claude reasoning in 9B package (87.8% HumanEval)
4. **IBM Granite 4.0 Tiny** - 7B MoE, 1B active, runs on phones

### Key Trend
The MoE architecture revolution means 2026 local LLMs deliver frontier-quality results on consumer hardware. The Qwen3.5-35B-A3B is the defining model of this era - 35B total params, 3B active, fits in 21GB, and beats models 6x its size.
