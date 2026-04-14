# Twitter/X Confirmation Pass - April 2026

Third research pass to confirm or challenge prior findings using X/Twitter and web sources.
Date: 2026-04-08

---

## 1. "Gemma 4 26B-A4B has Arena ELO 1441, highest of models that fit 32GB"

**STATUS: CONFIRMED**

- Arena.ai official post confirms: Gemma 4 26B-A4B debuts at #6 on the open source leaderboard (Arena ELO ~1441)
- Gemma 4 31B ranks #3 among open models (#27 overall) with ELO ~1452
- The 26B MoE requires only ~18GB at 4-bit or ~28GB at 8-bit, fitting comfortably in 32GB
- Source: @arena (https://x.com/arena/status/2039782449648214247)
- Source: @Raullen (https://x.com/Raullen/status/2041260332505522291)
- Source: Threads thread with exact scores (https://www.threads.com/@smiling.ank.1/post/DWo9ZfXj7Wn/)

**CORRECTION/NUANCE**: The 31B Dense model at ELO 1452 also fits in 32GB at 4-bit quant (reportedly ~18GB with aggressive quantization). So the 26B-A4B is NOT the highest Arena scorer that fits 32GB - the 31B Dense at 4-bit also fits. However, the 26B MoE is much faster due to only 4B active params.

---

## 2. "APEX quantization beats F16 perplexity for Qwen3.5-35B-A3B"

**STATUS: CONFIRMED - STRONGLY**

- Creator Ettore Di Giacinto (@mudler_it) posted detailed results on X
- APEX Quality achieves perplexity 6.527 on wikitext-2-raw, beating F16's 6.537 at only 21.3 GB
- APEX is 2x smaller than Unsloth Dynamic 2.0 while maintaining better accuracy on MoE architectures
- Half the size of Q8_0 with perplexity comparable to or better than F16
- Works with stock llama.cpp - no custom builds needed
- APEX GGUFs now published for 7 MoE models: Gemma 4 26B-A4B, GLM-4.7-Flash, Holo3-35B-A3B, Qwen3.5-35B-A3B
- Source: @mudler_it (https://x.com/mudler_it/status/2039364812463853708)
- Source: @mudler_it update (https://x.com/mudler_it/status/2040685981011948008)
- Source: @TeksEdge Gemma 4 APEX results (https://x.com/TeksEdge/status/2040149162076029066)
- Source: HuggingFace APEX technical report (https://github.com/mudler/apex-quant/blob/main/paper/APEX_Technical_Report.md)

**KEY DETAIL**: On Gemma 4 26B-A4B, APEX Balanced (18.1 GB) achieved perplexity 316.4 vs Q8_0's 337.3. All APEX variants show 10-18% faster token generation than equivalent size quants.

---

## 3. "Nemotron-Cascade-2-30B-A3B won IMO/IOI/ICPC gold medals"

**STATUS: CONFIRMED**

- NVIDIA research paper and VentureBeat coverage confirm gold medal performance:
  - IMO 2025: 35/42 points (gold medal level)
  - IOI 2025: 439.28/600 (gold medal level)
  - ICPC World Finals 2025: 10/12 problems solved, #4 Gold medal placement
- Only the second open model to reach this tier, after DeepSeek-V3.2-Speciale (which has 20x more params)
- IMO solutions evaluated by human expert (IMO 2015 Gold medalist)
- Real user benchmarks on RTX 3090: 187 tok/s flat from 4K to 625K context, 67% faster than Qwen3.5 35B-A3B
- Source: @ychenNLP NVIDIA release (https://x.com/ychenNLP/status/2036140633690350052)
- Source: @sudoingX RTX 3090 testing (https://x.com/sudoingX/status/2038260014490714430)
- Source: VentureBeat article (https://venturebeat.com/orchestration/nvidias-nemotron-cascade-2-wins-math-and-coding-gold-medals-with-3b-active)

**CAVEAT for Apple Silicon**: The 187 tok/s benchmark is on RTX 3090, not Apple Silicon. Q4_K_M at 24.5GB does NOT fit in 24GB VRAM. Users should use Bartowski IQ4_XS at 18.17GB.

---

## 4. "GLM-4.7-Flash is the fastest MoE inference"

**STATUS: PARTIALLY CONFIRMED - NUANCED**

- GLM-4.7-Flash is 31B/3B total/active parameters
- It is confirmed as the most intelligent open weights model under 100B total params on Artificial Analysis Intelligence Index
- Best performing 30B model on SWE-Bench and GPQA with 200K context
- Became the most downloaded model on Unsloth AI
- On a cluster of 4x M4 Pro Mac Minis via exo labs: 100 tok/s (aiming for 200 tok/s)
- On M4 128GB: used for Cline vs Claude-Code comparison
- Source: @ArtificialAnlys (https://x.com/ArtificialAnlys/status/2014518379114332204)
- Source: @alexocheema (https://x.com/alexocheema/status/2013694573910937980)
- Source: @goinggodotnet M4 testing (https://x.com/goinggodotnet/status/2015132760479003074)

**CORRECTION**: "Fastest MoE inference" is a broad claim. GLM-4.7-Flash is the top-performing 30B-class MoE for quality, but raw inference speed depends on the MoE model's active param count. With 3B active params, it's comparable to other 3B-active MoE models. The claim about speed is more about its quality-per-token-speed ratio.

---

## 5. "Abliteration compounds quality loss with quantization"

**STATUS: PARTIALLY CONFIRMED - MORE NUANCED THAN STATED**

- Academic paper (arxiv 2512.13655) provides detailed analysis:
  - MoE models show substantial reasoning degradation post-abliteration (safety experts contribute to reasoning pipeline)
  - Dense models show negligible or slightly positive effects from abliteration
  - Mathematical reasoning most sensitive: GSM8K scores show range of -18.81 percentage points
- Quantization interaction: Standard quantization (Q4_K_M) does NOT compound with abliteration noticeably. Models quantized to Q4_K_M perform comparably to F16 variants post-abliteration
- HOWEVER: Aggressive KV cache quantization (Q8_0) in abliterated models CAN produce incorrect analysis in long contexts
- New technique: MPOA (Magnitude-Preserving Orthogonal Ablation) minimizes quality degradation
- Newer technique: ARA (reported in Heretic tool) - claimed even less "brain damage" than abliteration
- Gemma-4-31B-JANG_4M-CRACK: full abliteration with only -2% MMLU drop (74.5%), 93.7% HarmBench compliance
- Source: arxiv paper (https://arxiv.org/html/2512.13655)
- Source: @leftcurvedev_ Gemma 4 JANG (https://x.com/leftcurvedev_/status/2040596306075193412)
- Source: @umiyuki_ai on ARA (https://x.com/umiyuki_ai/status/2040727062743511256)

**KEY FINDING**: The claim is too broad. For dense models, abliteration + quantization is fine. For MoE models, abliteration itself causes more damage than quantization does. The compounding effect is specifically about MoE architectures and aggressive KV cache quantization.

---

## 6. "HauhauCS Uncensored-Aggressive is the most popular uncensored Qwen3.5"

**STATUS: CONFIRMED**

- Knut Jaegersberg confirmed: 0/465 refusals, fully uncensored with zero capability loss
- Trending on HuggingFace, covered by HackerNoon
- Available in 9B and 27B variants
- Users in Japan and other regions actively testing and sharing results
- Competitor: huihui.ai releases abliterated versions of many Qwen3.5 sizes (0.8B, 2B, 9B, 27B) but these appear less popular
- Source: @JagersbergKnut (https://x.com/JagersbergKnut/status/2029598249904923002)
- Source: @hackernoon (https://x.com/hackernoon/status/2036903474282024993)
- Source: @aimodelsfyi (https://x.com/aimodelsfyi/status/2030135002881720718)

**NOTE**: For Gemma 4, the new JANG_4M-CRACK and Heretic variants are the emerging uncensored favorites (93.7% HarmBench, -2% MMLU).

---

## 7. "Ollama 0.19 switched to MLX backend"

**STATUS: CONFIRMED - WITH IMPORTANT CAVEATS**

- Official Ollama announcement on March 31, 2026: switched from Metal backend to MLX
- Performance gains: ~2x faster decode (58 -> 112 tok/s), ~1.6x faster prefill (1154 -> 1810 tok/s)
- With int4 quant: 1851 tok/s prefill, 134 tok/s decode
- Awni Hannun (MLX creator) celebrated the integration
- Source: @ollama (https://x.com/ollama/status/2038835449012351197)
- Source: @awnihannun (https://x.com/awnihannun/status/2038837352739844552)
- Source: 9to5Mac (https://9to5mac.com/2026/03/31/ollama-adopts-mlx-for-faster-ai-performance-on-apple-silicon-macs/)
- Source: Ollama blog (https://ollama.com/blog/mlx)

**KNOWN BUGS/ISSUES:**
1. Gemma 4 31B hangs with flash attention enabled + prompts >500 tokens (CPU/GPU drops to 0%)
2. Gemma 4 falls back to llama.cpp runner when MLX errors - only ~15 tok/s for 31B, ~75 tok/s for 26B MoE
3. Homebrew installs show "MLX dynamic library not available" warning on every command
4. Large model inference saturates unified memory bus, causing Thunderbolt display resets
5. Limited model support in preview: only Qwen3.5-35B-A3B confirmed MLX-accelerated initially
6. Gemma 4 31B Q8_0 causes runaway memory growth (221.5 GB) and instability
- Source: GitHub issue #15368 (https://github.com/ollama/ollama/issues/15368)
- Source: GitHub issue #14350 (https://github.com/ollama/ollama/issues/14350)

**RECOMMENDATION**: Ollama 0.19 MLX is in preview. Check model compatibility before relying on MLX acceleration.

---

## 8. "Rapid-MLX is fastest on Apple Silicon"

**STATUS: PARTIALLY CONFIRMED**

- One major post by @Raullen claims: tested across 18 models vs Ollama, mlx-lm, llama.cpp - fastest on 16 of them
- Features: DeltaNet state snapshots for multi-turn (TTFT from 1.5s to <200ms), 100% tool-calling accuracy
- Reported benchmarks on Mac Studio M3 Ultra:
  - 122B model: 57 tok/s
  - Coder-Next 80B: 74 tok/s, 0.10s TTFT
  - 35B: 83 tok/s
  - 9B: 108 tok/s (2.3x faster than Ollama)
- Source: @Raullen (https://x.com/Raullen/status/2035004368102322433)

**CAVEAT**: This is primarily from a single promotional post. No widespread independent benchmarks found confirming these numbers. The claim needs more verification. Ollama 0.19 with MLX may narrow the gap significantly since the comparison was likely against pre-MLX Ollama.

---

## 9. "MLX is 3x faster than llama.cpp for MoE"

**STATUS: PARTIALLY CONFIRMED - NUMBER VARIES**

- General consensus: MLX is significantly faster than llama.cpp on Apple Silicon
- Academic study found: MLX (~230 tok/s) vs llama.cpp (~150 tok/s) - roughly 1.5x, not 3x for general workloads
- Chris Prucha posted: "Apple MLX is 2X+ faster than llama.cpp's backend" on M3 Max 64GB (MLX-4bit: 9.3 tok/s vs GGUF: 4.5 tok/s = ~2x)
- For MoE specifically: The speedup may be larger due to MLX's unified memory architecture being better suited to MoE sparse activations
- Ollama blog (post-MLX): Shows roughly 2x decode improvement over previous llama.cpp backend
- Source: @chrisprucha (https://x.com/chrisprucha/status/1868086906258694554)
- Source: Academic comparison paper (https://arxiv.org/pdf/2511.05502)
- Source: famstack.dev analysis (https://famstack.dev/guides/mlx-vs-gguf-apple-silicon/)

**CORRECTION**: The "3x" figure is likely inflated. Consensus is closer to 1.5-2x for general workloads. For MoE models specifically, the advantage may approach 2-3x in some configurations, but "3x" as a blanket statement is not well-supported.

**IMPORTANT NUANCE from @TeksEdge**: Qwen3.5 tool-calling in MLX variants degrades after 5-10 rounds, while GGUF (Q4_K_XL) holds steady at 70/70 success over longer contexts. So MLX may be faster but GGUF may be more reliable for agentic workloads.

---

## 10. "LM Studio has bugs loading Gemma 4 and Qwen3.5 MoE"

**STATUS: CONFIRMED - ACTIVELY BEING FIXED**

- Multiple GitHub bug reports confirm:
  - Gemma 4 GGUF (31B and 26B): "Failed to load model" - required llama.cpp 2.10.0+
  - Gemma 4 MLX: "Model type gemma4 not supported" ValueError
  - Gemma 4 31B Q8_0: Abnormal memory growth to 221.5 GB, system becomes unresponsive
  - Qwen3.5: RAG jinja rendering bug ("No user query found in messages") - NOW FIXED
  - Qwen3.5: Tool calling support improved in latest version
- Fix: LM Studio 0.4.9 coming with improved tool calling + llama.cpp runtime update needed
- Users advised: run `lms runtime update --all` before loading Gemma 4
- Source: Bug tracker #1728 (https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1728)
- Source: Bug tracker #1741 (https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1741)
- Source: Bug tracker #1750 (https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1750)
- Source: @yagilb (https://x.com/yagilb/status/2039750894921593223)

---

## 11. "simple-evals is deprecated"

**STATUS: PARTIALLY CONFIRMED**

- The simple-evals GitHub README states it "will not be actively maintained" and they are "not accepting new evals"
- However, it's not formally "deprecated" - just not maintained
- OpenAI's primary evaluation framework is now the Evals Dashboard (https://evals.openai.com/) with Trace Evals, Datasets, and Prompt Optimization
- The main openai/evals repo (separate from simple-evals) is still active
- Source: GitHub simple-evals repo (https://github.com/openai/simple-evals)
- Source: OpenAI Evals platform (https://evals.openai.com/)

**CORRECTION**: "Deprecated" is too strong. It's "no longer actively maintained" but still functional. For local model evaluation, it was never the primary tool anyway.

---

## 12. "lm-evaluation-harness is the standard"

**STATUS: CONFIRMED FOR RESEARCH/LEADERBOARD USE**

- Torchtune ships with lm-evaluation-harness integration for eval of finetunes
- Used widely by HuggingFace Open LLM Leaderboard
- However, for local/personal testing, the community uses a mix:
  - lm-evaluation-harness for formal benchmarks
  - HuggingFace Community Evals (new: datasets now host live leaderboards)
  - Manual testing with specific prompts
  - Benjamin Marie's manual GGUF evaluation methodology
- Source: @haileysch__ (https://x.com/haileysch__/status/1780283749667623381)
- Source: @ben_burtenshaw Community Evals (https://x.com/ben_burtenshaw/status/2019795723378942295)

---

## 13. "MMLU is contaminated, use MMLU-Pro"

**STATUS: LARGELY CONFIRMED**

- Ben Burtenshaw (HuggingFace): "Eval scores in 2026 are broken. MMLU at 91%+, GSM8K at 94%+, yet models still can't handle basic multi-step tasks"
- MMLU-Pro creator Wenhu Chen positioned it as "more difficult, robust and reasoning-driven"
- MMLU-Pro uses 10 options instead of 4, making random guessing less effective
- Qwen3.5 quantization evaluations (Benjamin Marie) note: "Most of the degradation is on MMLU Pro" - suggesting it's a more sensitive benchmark
- Community consensus: MMLU is saturated/contaminated, MMLU-Pro is better but not perfect
- Source: @ben_burtenshaw (https://x.com/ben_burtenshaw/status/2019795723378942295)
- Source: @WenhuChen (https://x.com/WenhuChen/status/1790597967319007564)
- Source: @bnjmn_marie (https://x.com/bnjmn_marie/status/2025951400119751040)

---

## 14. NEW models released in April 2026 that could be contenders

### Gemma 4 (April 2, 2026)
- 4 variants: E2B, E4B, 26B-A4B MoE, 31B Dense
- Apache 2.0 license
- 31B Dense: Arena ELO ~1452 (#3 open), 89.2% AIME 2026
- 26B MoE: Arena ELO ~1441 (#6 open), fits in 18GB at 4-bit
- **CONTENDER**: Yes, strong contender for 32GB Macs

### Holo3-35B-A3B (March 2026)
- H Company's computer-use focused MoE model
- Based on Qwen3.5 architecture, 35B/3B total/active
- 77.8% on OSWorld-Verified (desktop agent benchmark)
- Apache 2.0 license, open weights
- APEX GGUFs available
- **CONTENDER**: Yes, specifically for agentic/computer-use workloads

### Gemma-4-31B-JANG_4M-CRACK (April 2026)
- Abliterated Gemma 4 31B with only -2% MMLU drop
- 93.7% HarmBench compliance, 18GB mixed-precision MLX
- Uses CRACK + MPOA technique
- **CONTENDER**: For uncensored use cases, replaces HauhauCS models for Gemma 4

### Google LiteRT-LM (April 7, 2026)
- New inference framework for edge devices
- Production-ready, open-source
- Focus on mobile/IoT deployment
- **RELEVANT**: New inference engine option for small models

### Apfel (April 3, 2026)
- Unlocks Apple's built-in 3B parameter on-device LLM (Apple Foundation Model)
- CLI tool + OpenAI-compatible API server
- No download needed, runs on Neural Engine + GPU
- 4096-token context, mixed 2/4-bit quantization
- **CONTENDER**: Not for benchmarking (limited model), but notable as zero-setup baseline

---

## 15. New inference engines or major updates in last 2 weeks

### Ollama 0.19 (March 31, 2026)
- MLX backend for Apple Silicon - 2x speedup
- Still in preview, limited model support

### MetalRT by RunAnywhereAI
- Claims fastest LLM decode: 658 tok/s on M4 Max
- Full pipeline: LLM + STT + TTS
- Beats MLX on decode speed
- Source: @sanchitmonga22 (https://x.com/sanchitmonga22/status/2030923381181223384)

### Flash-MoE
- Enables running ~400B MoE models on iPhone 17 Pro and Macs
- Streams weights from SSD, only loads active experts into memory
- Simon Willison: Qwen 3.5 397B-A17B running at ~5.7 tok/s using 5.5 GB active memory on M3 Mac
- Source: @TeksEdge (https://x.com/TeksEdge/status/2036116516123595184)
- Source: @simonw (https://x.com/simonw/status/2034365182751973541)

### Docker Model Runner + vLLM
- vLLM now supported in Docker Model Runner on macOS for MLX models
- OpenAI-compatible API workflow
- Source: @Docker (https://x.com/Docker/status/2028470592899354929)

### LM Studio 0.4.9 (coming)
- Improved tool calling for Gemma 4 and Qwen3.5
- Updated llama.cpp runtime

---

## 16. New quantization techniques beyond APEX

### APEX (Adaptive Precision for EXpert Models) - CONFIRMED STATE OF THE ART for MoE
- Layer-wise precision gradient + MoE-aware tensor classification
- Beats F16 perplexity at fraction of size
- Works with stock llama.cpp

### DASLab GGUF Quantization Toolkit
- First open-source toolkit bringing GPTQ + EvoPress to GGUF format
- Heterogeneous quantization based on importance
- Source: @DAlistarh (https://x.com/DAlistarh/status/1967527772429177299)

### Unsloth Dynamic 2.0
- Per-tensor mixed-bit quantization
- Qwen3.5 shows "near-lossless accuracy under 4-bit weight and KV cache quantization"
- TQ1_0 preserves accuracy extremely well
- Source: @bnjmn_marie (https://x.com/bnjmn_marie/status/2025951400119751040)

### MPOA (Magnitude-Preserving Orthogonal Ablation)
- Not a quantization technique per se, but minimizes quality degradation from abliteration
- Preserves norm of weights when removing restriction direction vectors

### ARA (new, in Heretic tool)
- Newer than MPOA, claimed less "brain damage" than standard abliteration
- Applied to Gemma 4 already
- Source: @umiyuki_ai (https://x.com/umiyuki_ai/status/2040727062743511256)

### NVFP4
- NVIDIA's 4-bit floating point format
- Supported in Ollama 0.19
- Maintains model accuracy while reducing memory bandwidth

---

## 17. Apple Silicon M4-specific optimizations

### M4 Max Performance
- MLX inference 27% faster than M3 Max (72 vs 56 tok/s for Gemma 2 9B 4-bit)
- M4 Pro: 4x faster than M1 (104 vs 25 tok/s for Llama 3.2 3B 4-bit)
- MetalRT achieves 658 tok/s decode on M4 Max
- Source: @ronaldmannak (https://x.com/ronaldmannak/status/1854889772134936960)

### M5 Max (new data point!)
- Justin Kalland benchmarks: Qwen3.5-35B-A3B Q6 at 74 tok/s on M5 Max 128GB
- Qwen3-Coder-Next 6-bit: 67 tok/s
- Source: @nix_eth (https://x.com/nix_eth/status/2032879242737045612)

### Apple Neural Engine Reverse-Engineering
- ANE found to be "ridiculously efficient" but CoreML adds 2-4x overhead
- WWDC rumored CoreAI update may fix this overhead
- Source: @ronaldmannak (https://x.com/ronaldmannak/status/2028560995875168292)

### Distributed MLX Improvements
- Awni Hannun: "Distributed inference in MLX on Apple silicon will be much faster in Tahoe 26.2"
- Thunderbolt 5 clusters (80Gbps) running distributed inference
- Source: @awnihannun (https://x.com/awnihannun/status/1999596403472105975)

### Apple Foundation Model via Apfel
- Built-in ~3B param model, mixed 2/4-bit quant
- Runs on Neural Engine + GPU, no download needed
- 4096 token context, OpenAI-compatible API
- Source: (https://github.com/Arthur-Ficial/apfel)

### Key M4 Speed Benchmarks (from various sources)
| Model | Hardware | Quant | Speed |
|-------|----------|-------|-------|
| Qwen3.5-35B-A3B | M3 Max 128GB | 4-bit | 91.6 tok/s |
| Qwen3.5-35B-A3B | M5 Max 128GB | Q6 | 74 tok/s |
| Qwen3.5-122B-A10B | M3 Max 128GB | 4-bit | 39.9 tok/s |
| Gemma 4 26B-A4B | M4 Mini 32GB | 4-bit | ~75 tok/s (GGUF fallback) |
| Nemotron Cascade 2 | RTX 3090 | IQ4_XS | 187 tok/s |

---

## SUMMARY OF CORRECTIONS TO PRIOR FINDINGS

1. **ELO 1441**: Confirmed, but 31B Dense at 4-bit also fits 32GB with higher ELO (1452)
2. **APEX beats F16**: Strongly confirmed (6.527 vs 6.537 perplexity)
3. **Nemotron gold medals**: Confirmed (IMO 35/42, IOI 439/600, ICPC 10/12)
4. **GLM-4.7-Flash fastest**: Partially - fastest quality-wise, not necessarily raw speed
5. **Abliteration compounds**: Nuanced - worse for MoE, fine for dense; MPOA/ARA mitigate
6. **HauhauCS most popular**: Confirmed for Qwen3.5; JANG_4M-CRACK emerging for Gemma 4
7. **Ollama 0.19 MLX**: Confirmed, but buggy (Gemma 4 hangs, limited model support)
8. **Rapid-MLX fastest**: Single-source claim, needs verification; Ollama MLX may close gap
9. **MLX 3x faster**: Closer to 1.5-2x; MLX has tool-calling reliability issues vs GGUF
10. **LM Studio bugs**: Confirmed, being actively fixed (need runtime update + v0.4.9)
11. **simple-evals deprecated**: "Not actively maintained" - not formally deprecated
12. **lm-evaluation-harness standard**: Yes for formal benchmarks; community uses mix
13. **MMLU contaminated**: Confirmed - saturated at 91%+; MMLU-Pro is preferred

## NEW DISCOVERIES NOT IN PRIOR RESEARCH

- **Holo3-35B-A3B**: New MoE model specifically for computer-use/agentic tasks (77.8% OSWorld)
- **MetalRT**: Claims 658 tok/s decode on M4 Max, potentially fastest Apple Silicon engine
- **Flash-MoE**: Enables 400B models on iPhone/Mac by streaming weights from SSD
- **Apfel**: Unlocks Apple's built-in 3B LLM via CLI (no download needed)
- **ARA technique**: Newer than MPOA for abliteration with less quality loss
- **JANG_4M-CRACK**: Gemma 4 31B uncensored with only -2% MMLU, uses CRACK+MPOA
- **M5 Max benchmarks**: First speed data available (74 tok/s for Qwen3.5-35B-A3B Q6)
- **MLX tool-calling degradation**: MLX variants lose tool-calling accuracy after 5-10 rounds; GGUF more reliable
- **Gemma 4 very robust to quantization**: "More than Qwen3.5" per Benjamin Marie's evaluation
- **LiteRT-LM**: Google's new edge inference framework (April 7, 2026)
- **DASLab GGUF Toolkit**: GPTQ + EvoPress brought to GGUF format
