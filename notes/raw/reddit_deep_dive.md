# Reddit Deep Dive: Local LLM Benchmarking on Apple Silicon

*Search performed: 2026-04-08 11:19 UTC*

**Subreddits searched:** r/LocalLLaMA, r/MachineLearning, r/LocalAI

**Total unique posts found:** 160

---

## Comprehensive Analysis

### Executive Summary

Reddit's r/LocalLLaMA community (160 unique posts analyzed, 15 full threads scraped) provides an extremely active and opinionated picture of the local LLM landscape as of early April 2026. The community is dominated by discussion of **Gemma 4** (released April 2), **Qwen 3.5** (the incumbent champion), and emerging models like **Nemotron Cascade 2**. Key themes: quantization breakthroughs (TurboQuant, APEX), the 32-64GB memory sweet spot problem, and the ongoing GGUF (llama.cpp) vs MLX debate.

---

### 1. Models: What We May Have Missed or Underweighted

**Gemma 4 (released 2026-04-02) -- THE biggest event**
- 4 variants: E2B (2.3B effective), E4B (4.5B effective), 26B-A4B (MoE, 4B active), 31B (dense)
- Apache 2.0 license (major upgrade from restrictive Gemma license)
- Native thinking, tool calling, multimodal (text+image, audio on small models)
- 256K context on medium models, 128K on small
- Community consensus: 31B dense is the quality champion, 26B-A4B is the speed champion
- Per-Layer Embeddings (PLE) in E2B/E4B is a genuinely novel architecture -- embeddings stored per-layer can be offloaded to disk/CPU with minimal speed penalty
- **Critical early bug**: llama.cpp tokenizer was broken at launch (fixed in b8648+), all early quants needed to be redone. Many early "reviews" were of a broken model
- **Tool calling issues**: 26B-A4B has consistent JSON formatting problems in tool calls. Multiple users report needing custom sanitizers. 31B works fine
- **KV cache concern**: Gemma 4 uses more KV cache memory than Qwen 3.5 at equivalent context lengths (no hybrid linear attention)

**Qwen 3.5 -- still the reigning champion for many users**
- Qwen 3.5 27B (dense) -- community darling for 24GB VRAM cards, repeatedly called "the best local model"
- Qwen 3.5 35B-A3B (MoE) -- best speed/quality tradeoff on 32GB+ systems
- Qwen 3.5 122B-A10B -- runs at IQ3_XXS on 64GB Macs, usable with TurboQuant for full 262K context
- Qwen3-Coder-Next 80B-A3B -- some users prefer this for coding despite similar benchmarks to 35B-A3B; 80B total params means more knowledge
- **Key weakness**: Overthinking problem (thousands of thinking tokens for simple queries), looping on default settings, poor European language support vs Gemma
- **Key strength**: Much better image understanding than Gemma 4, more reliable tool calling

**Nemotron Cascade 2 30B-A3B -- worth benchmarking**
- Based on Nemotron's own architecture (NOT Qwen-based)
- 97.6% on HumanEval at IQ4_XS -- exceptional for its size
- Outscores Qwen 3.5 35B-A3B on the SQL benchmark
- Mixed real-world reports: excellent for pure coding benchmarks, but "terrible for agentic tasks" (loops, doesn't follow system prompts, timeouts)
- Some users report it as a legitimate Qwen 3.5 35B-A3B replacement; others find it unusable for anything beyond HumanEval
- MLX support was initially broken (citing system prompt instead of generating output)

**GLM 4.7 Flash -- confirmed solid baseline**
- Consistently mentioned as a reliable daily driver (72 tok/s, 18GB)
- "Is GLM-4.7-Flash relevant anymore?" thread suggests it's being superseded by Qwen 3.5 but still has loyal users
- Used as baseline in many comparisons
- REAP 23B variant mentioned for coding

**Devstral Small 2 24B -- "severely underrated"**
- Multiple users praise it as best coding model for 16GB VRAM
- Scores well on benchmarks but doesn't get hype

**GPT-OSS 20B/88B/120B -- mentioned peripherally**
- 88B variant by Nvidia mentioned as new option for 64GB systems
- 20B scored surprisingly well on SQL benchmark

**Qwen 3.6 -- mentioned as upcoming**
- Referenced as "the upcoming qwen 3.6" by users, suggesting imminent release

**Darwin-35B-A3B-Opus** -- community merge model
- MoE merge of Qwen3.5-35B-A3B (father) + Claude Opus 4.6 distillation (mother)
- Uses "Model MRI" layer-by-layer analysis for merge
- Niche but interesting community effort

**Mimo v2 Flash** -- "a gem of a model" per SQL benchmark author

**Apriel 1.6 15B** -- scored 20/25 on SQL benchmark, "outstanding competitor"

**Kimi-Linear 48B-A3B** -- tested on Pi5, mentioned as option

---

### 2. Inference Engines and Quantization

**TurboQuant -- the hottest topic**
- KV cache compression technique from Google, compresses K and V caches to ~3-3.5 bits
- Key innovation: sparse V dequant -- skip 90% of V dequantization where attention weights are near-zero, yielding +22.8% decode speed at 32K context on M5 Max
- Community concern: turbo3 KLD is 2x worse than q4_0. Some argue this makes it unusable; developer argues the positions skipped only contribute quantization noise
- Currently in a llama.cpp fork (TheTom/turboquant_plus), not yet in mainline
- A vLLM implementation also exists (turboquant-vllm on PyPI)
- Multiple users running it successfully for 262K context on 64GB Macs with Qwen 3.5 122B-A10B

**APEX (Adaptive Precision for Expert Models) -- worth noting**
- Novel MoE quantization technique from LocalAI team (mudler)
- Claims: outperforms Unsloth Dynamic 2.0 on accuracy while being 2x smaller for MoE architectures
- 33% faster inference, 14% faster prompt processing
- Works with stock llama.cpp (no patches needed)
- Benchmarked on Qwen3.5-35B-A3B
- "Half the size of Q8. Perplexity comparable to F16"

**llama.cpp -- still the dominant engine**
- Critical tips from community:
  - `-np 1` for single-user setups (20% more TPS)
  - `--fit-target` for optimal VRAM usage
  - Custom Jinja templates matter hugely for Gemma 4 (interleaved template preserves reasoning between tool calls)
  - `-cmoe` flag for offloading MoE weights to CPU
  - Disable browser hardware acceleration for GPU perf
- Regression warnings: specific builds have tokenizer bugs for Gemma 4

**MLX vs llama.cpp ("Round 2" thread)**
- "GGUF (llama.cpp) vs MLX Round 2" found that:
  - MLX prompt caching was broken for some models (mlx-lm#903)
  - Hybrid attention models MLX can't optimize
  - bf16 models on M1 (which doesn't do bf16 natively) hurt MLX
  - For well-suited models (e.g. not Qwen 3.5), MLX can match or beat GGUF
  - Ollama adds significant overhead vs raw llama.cpp
- Many users prefer GGUF for parallelism, shared KV cache, and KV cache persistence
- Rapid-MLX mentioned but not widely discussed

**Apple Neural Engine (ANE)**
- "Bypassing CoreML" thread: researcher achieved 170 tok/s by directly targeting the ANE
- CoreML treats ANE as black-box scheduler with no direct control
- The ANE does ~19 TFLOPS but is completely unused for LLM workloads
- Very early stage but potentially transformative for on-device inference

**LM Studio vs raw llama.cpp**
- Multiple users frustrated with LM Studio config management
- Consensus: LM Studio good for downloading models, llama.cpp with custom shell scripts better for daily use
- LM Studio had delayed Gemma 4 support

---

### 3. Apple Silicon Specific Findings

**Performance benchmarks on Apple Silicon (from multiple threads)**
- M5 MacBook Air 32GB: Gemma-4-26B-A4B at UD-IQ4_XS gives 300t/s PP, 12t/s TG at 8W (!)
- M5 Max: 37 models benchmarked, community benchmark database being built
- M5 Pro MacBook: Gemma 4 26B-A4B at ~81 tok/s
- M4 Pro 48GB: TurboQuant testing shows turbo3 works well for Gemma 4
- M2 Max 64GB: "falls right into the local LLM dead zone" -- too small for 100B+, 35B models are "mediocre for sophisticated agentic use"
- M1 Ultra Mac Studio: Gemma 26B-A4B same speed as Qwen3.5 35B-A3B (~1000pp, ~60tg at 20k context)
- M3 Pro 36GB: Gemma 26B runs at ~3.2 tok/s, E4B at ~4.3 tok/s (disappointingly slow)
- M3 Ultra: 17 MLX models benchmarked, Qwen3.5-122B-A10B at 8bit gives 43 t/s decode

**The 32GB Mac sweet spot**
- Gemma 4 26B-A4B at Q4 is the new champion: fast, fits with room for context
- Qwen 3.5 35B-A3B at Q4 also fits but with less context headroom
- With TurboQuant, context can be dramatically extended

**The 64GB dead zone**
- Too small to run 122B models at good quality
- 35B MoE models are "mediocre for sophisticated agentic use"
- Best options: Qwen3-Coder-Next 80B-A3B at Q4 (tight at ~50GB), or Qwen 3.5 122B-A10B at IQ3_XXS
- External GPU via Thunderbolt mentioned (Tinygrad support) but very early/unreliable on Apple Silicon

---

### 4. Benchmark Methodology Insights

**Community skepticism of benchmarks is HIGH**
- "Never trust trust me bro benchmarks" -- frequent sentiment
- "The stupid trend of not trusting benchmarks is really affecting critical thinking" -- counter-sentiment
- "Benchmaxxing is a real problem" -- widely acknowledged
- Consensus: make your own benchmarks for your use case

**LLM-as-Judge criticism**
- Claude Opus 4.6 as judge widely criticized for bias
- Positional bias, verbosity bias, self-preference bias all noted
- "Use programmatic scoring. Write unit test cases. Test for exact matches." -- strong recommendation
- Temperature settings must match model recommendations or results are meaningless

**Notable community benchmarks**
- FoodTruckBench: agentic business simulation, Gemma 4 31B #3 overall
- SQL Benchmark (nicklothian): text-to-SQL, Qwen 3.5 27B ties with Kimi K2.5 at top
- AdamBench: local LLM agentic coding benchmark
- Altered Riddles: testing if LLMs can ignore memorized answers
- EuroEval: multilingual benchmark showing Gemma 4 dominance in European languages
- Quantuzo: SWE-bench across different KV cache quantization levels

---

### 5. Warnings and Gotchas

1. **Gemma 4 llama.cpp bugs**: Tokenizer was broken at launch. Need b8648+ minimum. Some builds have regressions after that. imatrix quants made before the fix need to be redone.

2. **Gemma 4 26B-A4B tool calling is broken**: Produces malformed JSON. Needs custom output sanitizer for agentic use. 31B dense works fine.

3. **Qwen 3.5 overthinking**: Generates thousands of thinking tokens for simple queries. Set reasoning-budget to 1000-1500 tokens, or disable thinking entirely for non-reasoning tasks.

4. **Qwen 3.5 tool calling in thinking block**: Tool calls inside <thinking> blocks often don't execute. Known issue with workarounds.

5. **Temperature matters enormously**: Google recommends temp=1.0 for Gemma 4 (community often disagrees for coding: temp=0.3, top-p=0.9, min-p=0.1, top-k=20). Using wrong temperature invalidates any benchmark.

6. **MoE speed is deceptive for agents**: 3x faster generation looks great, but if the model needs 2-3 retry cycles, dense models that nail it first try end up faster.

7. **Context memory differs wildly**: Gemma 4 uses much more KV cache than Qwen 3.5 at same context length (no hybrid linear attention). Factor of 2-3x difference on long contexts.

8. **Unsloth Q4_K_XL vs Bartowski Q4_K_M**: Noticeable speed difference (~28 vs ~38 tok/s on Gemma 4). Likely due to different quant architectures. Worth testing both.

9. **Quantizer quality matters**: 500GB of storage needed just to produce various quant sizes of Gemma-4-26B-A4B. There's "an art" to configuring quants between architectures.

10. **Nemotron Cascade 2 benchmaxing concerns**: Exceptional HumanEval scores but multiple reports of poor real-world agentic performance (loops, ignoring system prompts).

---

### 6. Very Recent Developments (April 2026)

- **Gemma 4 release (April 2)**: Biggest open model event; Apache 2.0 license
- **Gemma 4 Uncensored/Abliterated** (April 5): All 4 variants abliterated with MoE expert abliteration. KL divergence measured.
- **TurboQuant sparse V dequant** (March 27): +22.8% decode speed, being validated for mainline llama.cpp PR
- **37-model Mac benchmark database** (April 6): Community building per-chip benchmark database for M1-M5
- **Qwen 3.6 imminent**: Referenced by users as upcoming
- **Darwin V5 merging engine**: New layer-by-layer "Model MRI" merge technique
- **APEX MoE quantization** (April 1): 2x smaller than Unsloth Dynamic 2.0 for MoE, 33% faster
- **Gemma 4 interleaved Jinja template**: Critical for agentic use, preserves reasoning between tool calls
- **Per-layer embedding offloading**: `-ot "per_layer_token_embd.weight=CPU"` puts E4B at 4.7GB VRAM

---

### 7. Community Consensus: Best Models by Use Case

| Use Case | Best Model (Community) | Runner-up |
|---|---|---|
| Coding on 24GB VRAM | Qwen 3.5 27B (dense) | Devstral Small 2 24B |
| Coding on 16GB VRAM | GLM 4.7 Flash Q4_K_M | Nemotron Cascade 2 Q4 |
| General on 32GB Mac | Gemma 4 26B-A4B (Q4) | Qwen 3.5 35B-A3B (Q4) |
| General on 64GB Mac | Qwen 3.5 122B-A10B (IQ3_XXS) | Qwen3-Coder-Next 80B-A3B |
| Agentic workflows | Qwen 3.5 27B | Gemma 4 31B |
| European languages | Gemma 4 31B/26B | (no close second) |
| Image understanding | Qwen 3.5 (any size) | (Gemma 4 significantly worse) |
| Translation | Gemma 4 31B | TranslateGemma |
| Speed-critical tasks | Gemma 4 26B-A4B / Qwen 3.5 35B-A3B | GLM 4.7 Flash |
| Creative writing | Gemma 4 31B | Qwen 3.5 27B |

---

### 8. Key Takeaways for Our Benchmark Project

1. **Must benchmark Gemma 4** (both 26B-A4B and 31B) -- it's the #1 topic on r/LocalLLaMA right now
2. **Qwen 3.5 27B dense should be tested alongside MoE variants** -- community repeatedly notes it outperforms 35B-A3B in quality
3. **TurboQuant KV cache should be tested** -- +22.8% decode speed is significant and directly relevant to Apple Silicon
4. **APEX quantization worth investigating** for MoE models
5. **Nemotron Cascade 2 30B-A3B** deserves inclusion -- controversial but potentially strong
6. **Temperature and sampling settings must match model recommendations** -- biggest methodology concern in the community
7. **Tool calling reliability** should be a test dimension -- significant differences between models
8. **KV cache memory usage** at different context lengths is a critical practical metric
9. **Per-Layer Embedding offloading** for Gemma 4 E-series is a unique optimization worth testing
10. **llama.cpp build version matters** -- document exact build used in all benchmarks

---

*Raw search results and full thread content follow below.*

---

---

## All Search Results (sorted by relevance)

### 1. [Gemma 4 has been released](https://www.reddit.com/r/LocalLLaMA/comments/1salgre/gemma_4_has_been_released/)
- **Subreddit:** r/LocalLLaMA | **Score:** 2273 | **Comments:** 663 | **Date:** 2026-04-02 16:01 UTC
- **Author:** u/jacek2023 | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 35990
  > [https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)
  > 
  > [https://huggingface.co/unsloth/gemma-4-31B-it-GGUF](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF)
  > 
  > [https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF)
  > 
  > [https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF)
  > 
  > [https://huggingface.co/collections/google/gemma-4](https://huggingf

### 2. [Gemma 4 just casually destroyed every model on our leaderboard except Opus 4.6 and GPT-5.2. 31B params, $0.20/run](https://www.reddit.com/r/LocalLLaMA/comments/1sdcotc/gemma_4_just_casually_destroyed_every_model_on/)
- **Subreddit:** r/LocalLLaMA | **Score:** 1729 | **Comments:** 284 | **Date:** 2026-04-05 19:30 UTC
- **Author:** u/Disastrous_Theme5906 | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 22970
  > Tested Gemma 4 (31B) on our benchmark. Genuinely did not expect this.
  > 
  > 100% survival, 5 out of 5 runs profitable, +1,144% median ROI. At $0.20 per run.
  > 
  > It outperforms GPT-5.2 ($4.43/run), Gemini 3 Pro ($2.95/run), Sonnet 4.6 ($7.90/run), and absolutely destroys every Chinese open-source model we've tested — Qwen 3.5 397B, Qwen 3.5 9B, DeepSeek V3.2, GLM-5. None of them even survive consistently.
  > 
  > The only model that beats Gemma 4 is Opus 4.6 at $36 per run. That's 180× more expensive.
  > 
  > 31 billi

### 3. [Gemma 4 and Qwen3.5 on shared benchmarks](https://www.reddit.com/r/LocalLLaMA/comments/1saoyj7/gemma_4_and_qwen35_on_shared_benchmarks/)
- **Subreddit:** r/LocalLLaMA | **Score:** 855 | **Comments:** 235 | **Date:** 2026-04-02 18:07 UTC
- **Author:** u/fulgencio_batista | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 13250

### 4. [Skipping 90% of KV dequant work → +22.8% decode at 32K (llama.cpp, TurboQuant)](https://www.reddit.com/r/LocalLLaMA/comments/1s56g07/skipping_90_of_kv_dequant_work_228_decode_at_32k/)
- **Subreddit:** r/LocalLLaMA | **Score:** 839 | **Comments:** 113 | **Date:** 2026-03-27 14:56 UTC
- **Author:** u/Pidtom | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 10650
  > I’ve been working on an open source TurboQuant implementation for KV cache compression in llama.cpp and ran into a hard bottleneck: dequantization.
  > 
  > At long context (32K on M5 Max), dequant alone was taking around 40 percent of decode time.
  > 
  > I tried fixing it the usual way:
  > - register LUTs  
  > - SIMD tricks  
  > - fused kernels  
  > - branchless math  
  > 
  > Tested about 14 different approaches. None beat the baseline. Hardware was already at the limit.
  > 
  > What ended up working was much simpler.
  > 
  > Flash attenti

### 5. [Per-Layer Embeddings: A simple explanation of the magic behind the small Gemma 4 models](https://www.reddit.com/r/LocalLLaMA/comments/1sd5utm/perlayer_embeddings_a_simple_explanation_of_the/)
- **Subreddit:** r/LocalLLaMA | **Score:** 518 | **Comments:** 53 | **Date:** 2026-04-05 15:02 UTC
- **Author:** u/-p-e-w- | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 6240
  > Many of you seem to have liked my recent post ["A simple explanation of the key idea behind TurboQuant"](https://www.reddit.com/r/LocalLLaMA/comments/1s62g5v/a_simple_explanation_of_the_key_idea_behind/). Now I'm really not much of a blogger and I usually like to invest all my available time into developing Heretic, but there is another really cool new development happening with lots of confusion around it, so I decided to make another quick explainer post.
  > 
  > You may have noticed that the brand-n

### 6. [Don't sleep on the new Nemotron Cascade](https://www.reddit.com/r/LocalLLaMA/comments/1rzud2z/dont_sleep_on_the_new_nemotron_cascade/)
- **Subreddit:** r/LocalLLaMA | **Score:** 301 | **Comments:** 137 | **Date:** 2026-03-21 15:30 UTC
- **Author:** u/ilintar | **Query match:** "Nemotron Cascade"
- **Relevance score:** 5750
  > While there has been a lot of discussion regarding the Nemotron Super family of models, I feel like the newest addition, the [Nemotron Cascade 2 30B-A3B](https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B) (which is \*not\* based on the Qwen architecture despite a similar size, it's a properly hybrid model based on Nemotron's own arch) has largely flown under the radar.
  > 
  > I've been running some evals on local models lately since I'm kind of tired of the "vibe feels" method of judging them. 

### 7. [Gemma 4 is good](https://www.reddit.com/r/LocalLLaMA/comments/1sb73ar/gemma_4_is_good/)
- **Subreddit:** r/LocalLLaMA | **Score:** 256 | **Comments:** 140 | **Date:** 2026-04-03 07:42 UTC
- **Author:** u/One_Key_8127 | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 5360
  > Waiting for artificialanalysis to produce intelligence index, but I see it's good. Gemma 26b a4b is the same speed on Mac Studio M1 Ultra as Qwen3.5 35b a3b (\~1000pp, \~60tg at 20k context length, llama.cpp). And in my short test, it behaves way, way better than Qwen, not even close. Chain of thoughts on Gemma is concise, helpful and coherent while Qwen does a lot of inner-gaslighting, and also loops a lot on default settings. Visual understanding is very good, and multilingual seems good as we

### 8. [Gemma 4 is a huge improvement in many European languages, including Danish, Dutch, French and Italian](https://www.reddit.com/r/LocalLLaMA/comments/1seo2rq/gemma_4_is_a_huge_improvement_in_many_european/)
- **Subreddit:** r/LocalLLaMA | **Score:** 266 | **Comments:** 58 | **Date:** 2026-04-07 06:26 UTC
- **Author:** u/Balance- | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 3820
  > The benchmarks look really impressive for such small models. Even in general, they stand up well. Gemma 4 31B is (of all tested models):
  > 
  > \- 3rd on Dutch
  > 
  > \- 2nd on Danish
  > 
  > \- 3rd on English
  > 
  > \- 1st on Finish
  > 
  > \- 2nd on French
  > 
  > \- 5th on German
  > 
  > \- 2nd on Italian
  > 
  > \- 3rd on Swedish
  > 
  > Curious if real-world experience matches that.
  > 
  > Source: https://euroeval.com/leaderboards/

### 9. [64Gb ram mac falls right into the local llm dead zone](https://www.reddit.com/r/LocalLLaMA/comments/1s9xvtb/64gb_ram_mac_falls_right_into_the_local_llm_dead/)
- **Subreddit:** r/LocalLLaMA | **Score:** 115 | **Comments:** 133 | **Date:** 2026-04-01 21:22 UTC
- **Author:** u/Skye_sys | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 3810
  > So I recently bought a Mac (m2 max) with local llm use in mind and I did my research and everywhere everyone was saying go for the larger ram option or I will regret it later... So I did.
  > 
  > Time to choose a model:
  > 
  > "Okay,
  > - Nice model, Qwen3.5 35b a3b running 8 bit quant, speedy even with full context size. 
  > \-&gt; Performance wise it's mediocre especially for more sophisticated agentic use" 
  > 
  > "Hmm let me look for better options because I have 64 gbs maybe there is a smarter model out there. 
  > - Q

### 10. [I tested as many of the small local and OpenRouter models I could with my own agentic text-to-SQL benchmark. Surprises ensured...](https://www.reddit.com/r/LocalLLaMA/comments/1s7r9wu/i_tested_as_many_of_the_small_local_and/)
- **Subreddit:** r/LocalLLaMA | **Score:** 220 | **Comments:** 65 | **Date:** 2026-03-30 13:55 UTC
- **Author:** u/nickl | **Query match:** "Nemotron Cascade"
- **Relevance score:** 3500
  > Last week I asked for some feedback about what extra models I should test. I've added them all and now the benchmark is available at [https://sql-benchmark.nicklothian.com/](https://sql-benchmark.nicklothian.com/)
  > 
  > I didn't say a lot about what the agent at the time, but in simple terms it takes an English query like "*Show order lines, revenue, units sold, revenue per unit (total revenue ÷ total units sold), average list price per product in the subcategory, gross profit, and margin percentage 

### 11. [Gemma 4 31B vs Gemma 4 26B-A4B vs Qwen 3.5 27B — 30-question blind eval with Claude Opus 4.6 as judge](https://www.reddit.com/r/LocalLLaMA/comments/1scwos6/gemma_4_31b_vs_gemma_4_26ba4b_vs_qwen_35_27b/)
- **Subreddit:** r/LocalLLaMA | **Score:** 152 | **Comments:** 96 | **Date:** 2026-04-05 06:48 UTC
- **Author:** u/Silver_Raspberry_811 | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 3440
  > Just finished a 3-way head-to-head. Sharing the raw results because this sub has been good about poking holes in methodology, and I'd rather get that feedback than pretend my setup is perfect.
  > 
  > **Setup**
  > 
  > * 30 questions, 6 per category (code, reasoning, analysis, communication, meta-alignment)
  > * All three models answer the same question blind — no system prompt differences, same temperature
  > * Claude Opus 4.6 judges each response independently on a 0-10 scale with a structured rubric (not "which 

### 12. [Visual Guide to Gemma 4](https://www.reddit.com/r/LocalLLaMA/comments/1sbik5l/visual_guide_to_gemma_4/)
- **Subreddit:** r/LocalLLaMA | **Score:** 286 | **Comments:** 25 | **Date:** 2026-04-03 16:36 UTC
- **Author:** u/jacek2023 | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 3360
  > source: [https://x.com/osanseviero/status/2040105484061954349](https://x.com/osanseviero/status/2040105484061954349)
  > 
  > [https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4)

### 13. [benchmarks of gemma4 and multiple others on Raspberry Pi5](https://www.reddit.com/r/LocalLLaMA/comments/1sdcdno/benchmarks_of_gemma4_and_multiple_others_on/)
- **Subreddit:** r/LocalLLaMA | **Score:** 227 | **Comments:** 50 | **Date:** 2026-04-05 19:17 UTC
- **Author:** u/honuvo | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 3270
  > Hey all,
  > 
  > this is an update! A few days ago I posted to show the performance of a Raspberry Pi5 when using a SSD to let larger models run. Rightfully so, a few brought to my attention that the PCIe is faster than the USB3 connection I was using, so I bought the official HAT.
  > 
  > **Spoiler: As expected: Read speed doubled, leading to 1.5x to 2x improvement on tokens/sec for inference and text generation on models in swap.**
  > 
  > I'll repeat my setup shortly:
  > 
  > * Raspberry Pi5 with 16GB RAM
  > * Official Act

### 14. [Comparing Qwen3.5 vs Gemma4 for Local Agentic Coding](https://www.reddit.com/r/LocalLLaMA/comments/1sd0be8/comparing_qwen35_vs_gemma4_for_local_agentic/)
- **Subreddit:** r/LocalLLaMA | **Score:** 138 | **Comments:** 93 | **Date:** 2026-04-05 10:34 UTC
- **Author:** u/garg-aayush | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 3240
  > [Gemma4](https://deepmind.google/models/gemma/gemma-4/) was relased by Google on April 2nd earlier this week and I wanted to see how it performs against Qwen3.5 for local agentic coding. This post is my notes on benchmarking the two model families. I ran two types of tests:
  > 
  > * **Standard llama-bench benchmarks** for raw prefill and generation speed
  > * **Single-shot agentic coding tasks** using [Open Code](https://opencode.ai) to see how these models actually perform on real multi-step coding work

### 15. [Is 1-bit and TurboQuant the future of OSS? A simulation for Qwen3.5 models.](https://www.reddit.com/r/LocalLLaMA/comments/1sadadw/is_1bit_and_turboquant_the_future_of_oss_a/)
- **Subreddit:** r/LocalLLaMA | **Score:** 138 | **Comments:** 79 | **Date:** 2026-04-02 10:09 UTC
- **Author:** u/GizmoR13 | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 2960
  > Simulation what the Qwen3.5 model family would look like using 1-bit technology and TurboQuant. The table below shows the results, this would be a revolution:
  > 
  > |Model|Parameters|Q4\_K\_M File (Current)|KV Cache (256K) (Current)|Hypothetical 1-bit Weights|KV Cache 256K with TurboQuant|Hypothetical Total Memory Usage|
  > |:-|:-|:-|:-|:-|:-|:-|
  > |Qwen3.5-122B-A10B|122B total / 10B active|74.99 GB|81.43 GB|17.13 GB|1.07 GB|**18.20 GB**|
  > |Qwen3.5-35B-A3B|35B total / 3B active|21.40 GB|26.77 GB|4.91 GB|0.

### 16. [Gemma 4 for 16 GB VRAM](https://www.reddit.com/r/LocalLLaMA/comments/1scw979/gemma_4_for_16_gb_vram/)
- **Subreddit:** r/LocalLLaMA | **Score:** 168 | **Comments:** 58 | **Date:** 2026-04-05 06:22 UTC
- **Author:** u/Sadman782 | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 2840
  > **Update**: You can definitely consider Q8\_0 for mmproj; the quality doesn't drop, and surprisingly, it improved a bit in my vision tests. For example, with this one: [https://huggingface.co/prithivMLmods/gemma-4-26B-A4B-it-F32-GGUF/blob/main/GGUF/gemma-4-26B-A4B-it.mmproj-q8\_0.gguf](https://huggingface.co/prithivMLmods/gemma-4-26B-A4B-it-F32-GGUF/blob/main/GGUF/gemma-4-26B-A4B-it.mmproj-q8_0.gguf), now you can fit 30K more context in its place. 60K+ context FP16 cache with vision.
  > 
  > I think th

### 17. [My Experience with Qwen 3.5 35B](https://www.reddit.com/r/LocalLLaMA/comments/1ryc3w0/my_experience_with_qwen_35_35b/)
- **Subreddit:** r/LocalLLaMA | **Score:** 92 | **Comments:** 81 | **Date:** 2026-03-19 20:54 UTC
- **Author:** u/viperx7 | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 2540
  > these last few months we got some excellent local models like
  > 
  > * Nemotron Nano 30BA3
  > * GLM 4.7 Flash
  > 
  > both of these were very good compared to anything that came before them with these two for the first time i was able to reliably do stuff(meaning i can look at a task and know `yup these will be able to do it`)
  > 
  > but then came Qwen 35B. it was smarter overall speeds don't degrade with larger context, and all the things that the other two struggle with Qwen 3.5B nailed it with ease (the task i am 

### 18. [Consolidated my homelab from 3 models down to one 122B MoE — benchmarked everything, here's what I found](https://www.reddit.com/r/LocalLLaMA/comments/1s4p922/consolidated_my_homelab_from_3_models_down_to_one/)
- **Subreddit:** r/LocalLLaMA | **Score:** 87 | **Comments:** 66 | **Date:** 2026-03-27 00:39 UTC
- **Author:** u/MBAThrowawayFruit | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 2190
  > Been running local LLMs on a Strix Halo setup (Ryzen AI MAX+ 395, 128GB RAM, 96 GiB shared GPU memory via Vulkan/RADV) under Proxmox with LXC containers and llama-server. Wanted to share where I landed after way too much benchmarking.
  > 
  > **THE OLD SETUP (3 text models)**
  > 
  > \- GLM-4.7-Flash: 30B MoE 3B active, 18GB, 72 tok/s — daily driver, email
  > 
  > \- Qwen3.5-35B-A3B: 35B MoE 3B active, 20GB, 55 tok/s — reasoning/coding
  > 
  > \- Qwen3-VL-8B: 8B dense, 6GB, 39 tok/s — vision/cameras
  > 
  > \~44GB total. Worked b

### 19. [Benchmarked 18 models that I can run on my RTX 5080 16GB using Nick Lothian's SQL benchmark](https://www.reddit.com/r/LocalLLaMA/comments/1s9mkm1/benchmarked_18_models_that_i_can_run_on_my_rtx/)
- **Subreddit:** r/LocalLLaMA | **Score:** 68 | **Comments:** 70 | **Date:** 2026-04-01 14:44 UTC
- **Author:** u/grumd | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 2080
  > 2 days ago there was a very cool post by u/nickl:
  > 
  > [https://reddit.com/r/LocalLLaMA/comments/1s7r9wu/](https://reddit.com/r/LocalLLaMA/comments/1s7r9wu/)
  > 
  > Highly recommend checking it out!
  > 
  > I've run this benchmark on a bunch of local models that can fit into my RTX 5080, some of them partially offloaded to RAM (I have 96GB, but most will fit if you have 64).
  > 
  > Results:
  > 
  >     24: unsloth/Qwen3.5-122B-A10B-GGUF:UD-Q4_K_XL
  >     🟩🟩🟩🟩🟩 🟩🟩🟩🟩🟩 🟩🟩🟩🟩🟩 🟩🟩🟩🟥🟩 🟩🟩🟩🟩🟩
  >     23: bartowski/Qwen_Qwen3.5-27B-GGUF:IQ4_

### 20. [Nemotron Cascade 2 30B A3B](https://www.reddit.com/r/LocalLLaMA/comments/1ryo0i9/nemotron_cascade_2_30b_a3b/)
- **Subreddit:** r/LocalLLaMA | **Score:** 94 | **Comments:** 56 | **Date:** 2026-03-20 05:42 UTC
- **Author:** u/Middle_Bullfrog_6173 | **Query match:** "Nemotron Cascade"
- **Relevance score:** 2060
  > Based on Nemotron 3 Nano Base, but more/better post-training. Looks competitive with 120B models on math and code benchmarks. I've yet to test.
  > 
  >   
  > Hugging Face: [https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B](https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B)
  > 
  > Paper: [https://arxiv.org/abs/2603.19220](https://arxiv.org/abs/2603.19220)

### 21. [Tips: remember to use -np 1 with llama-server as a single user](https://www.reddit.com/r/LocalLLaMA/comments/1s4c7t3/tips_remember_to_use_np_1_with_llamaserver_as_a/)
- **Subreddit:** r/LocalLLaMA | **Score:** 115 | **Comments:** 45 | **Date:** 2026-03-26 16:24 UTC
- **Author:** u/ea_man | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 2050
  > Llama-serve.cp on default behavior may allocates 4x context size in order to serve multiple clients, if you are a single user on a system with little VRAM you know that the bigger the context length -&gt; smaller LM in VRAM -&gt; reduced speed.
  > 
  > So launch with llama-server `-np1` , maybe add `--fit-target 126`  
  > On my 12GB GPU with 60k context I got \~20% more TPS.
  > 
  > One more: if you use Firefox (or others) disable hw acceleration:
  > 
  > * Go to **Settings** \&gt; **General** \&gt; **Performance**.
  > * 

### 22. [Tested how OpenCode Works with SelfHosted LLMS: Qwen 3.5 &amp; 3.6, Gemma 4, Nemotron 3, GLM-4.7 Flash...](https://www.reddit.com/r/LocalLLaMA/comments/1sduazd/tested_how_opencode_works_with_selfhosted_llms/)
- **Subreddit:** r/LocalLLaMA | **Score:** 108 | **Comments:** 48 | **Date:** 2026-04-06 09:46 UTC
- **Author:** u/rosaccord | **Query match:** "Nemotron Cascade"
- **Relevance score:** 2040
  > I have run two tests on each LLM with OpenCode to check their basic readiness and convenience:
  > 
  > \- Create IndexNow CLI in Golang (Easy Task) and
  > 
  > \- Create Migration Map for a website following SiteStructure Strategy. (Complex Task)
  > 
  > Tested Qwen 3.5, &amp; 3.6, Gemma 4, Nemotron 3, GLM-4.7 Flash and several other LLMs.
  > 
  > Context size used: 25k-50k - varies between tasks and models.
  > 
  > The result is in the table below, hope you find it useful.
  > 
  > https://preview.redd.it/gdrou1bmdjtg1.png?width=686&amp

### 23. [Gemma 4 26b a4b - MacBook Pro M5 MAX. Averaging around 81tok/sec](https://www.reddit.com/r/LocalLLaMA/comments/1sb1rb9/gemma_4_26b_a4b_macbook_pro_m5_max_averaging/)
- **Subreddit:** r/LocalLLaMA | **Score:** 85 | **Comments:** 55 | **Date:** 2026-04-03 02:52 UTC
- **Author:** u/Bderken | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 1950
  > Pretty fast! Uses around 114watts at its peak, short bursts as the response is usually pretty fast.

### 24. [Devstral small 2 24b severely underrated](https://www.reddit.com/r/LocalLLaMA/comments/1ry93gz/devstral_small_2_24b_severely_underrated/)
- **Subreddit:** r/LocalLLaMA | **Score:** 84 | **Comments:** 42 | **Date:** 2026-03-19 18:52 UTC
- **Author:** u/The_Paradoxy | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 1680
  > I'm not a vibe coder, but I would like some basic assistance with my code. I'm posting this because I feel like the general consensus on Reddit was misleading about which models would be best for me to run locally on a 16gb GPU for code assistance.
  > 
  > 
  > For context, I'm an early career academic with no research budget for a fancy GPU. I'm using my personal 16gb 4060ti to assist my coding. Right now I'm revisiting some numpy heavy code wrapped with @numba.jit that I wrote three years ago and it impl

### 25. [I benchmarked 37 LLMs on MacBook Air M5 32GB — full results + open-source tool to benchmark your own Mac](https://www.reddit.com/r/LocalLLaMA/comments/1se81a5/i_benchmarked_37_llms_on_macbook_air_m5_32gb_full/)
- **Subreddit:** r/LocalLLaMA | **Score:** 83 | **Comments:** 40 | **Date:** 2026-04-06 19:00 UTC
- **Author:** u/evoura | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 1630
  > So I got curious about how fast different models actually run on my M5 Air (32GB, 10 CPU/10 GPU). Instead of just testing one or two, I went through 37 models across 10 different families and recorded everything using llama-bench with Q4\_K\_M quantization.
  > 
  > The goal: build a **community benchmark database** covering every Apple Silicon chip (M1 through M5, base/Pro/Max/Ultra) so anyone can look up performance for their exact hardware.
  > 
  > # The Results (M5 32GB, Q4_K_M, llama-bench)
  > 
  > # Top 15 by G

### 26. [Unsloth fixed version of Qwen3.5-35B-A3B is incredible at research tasks.](https://www.reddit.com/r/LocalLLaMA/comments/1rjh5wg/unsloth_fixed_version_of_qwen3535ba3b_is/)
- **Subreddit:** r/LocalLLaMA | **Score:** 305 | **Comments:** 76 | **Date:** 2026-03-03 05:40 UTC
- **Author:** u/Daniel_H212 | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 1371
  > When I first tried Qwen3.5-35B-A3B I was impressed, but honestly it seemed like a small jump over GLM-4.7-Flash, which had already impressed me with its interleaved thinking and native tool use capabilities. Qwen3.5-35B-A3B was about the level of "better" I thought it would be from having 5B extra parameters, and I thought the only big advantage was hybrid linear attention allowing double the native context length without really increasing memory footprint.
  > 
  > I saw today that Unsloth updated Qwen

### 27. [Llama.cpp Mi50 ROCm 7 vs Vulkan Benchmarks](https://www.reddit.com/r/LocalLLaMA/comments/1s0tcf7/llamacpp_mi50_rocm_7_vs_vulkan_benchmarks/)
- **Subreddit:** r/LocalLLaMA | **Score:** 87 | **Comments:** 22 | **Date:** 2026-03-22 18:32 UTC
- **Author:** u/JaredsBored | **Query match:** "Nemotron Cascade"
- **Relevance score:** 1310
  > Testing ROCm 7 using TheRock nightly tarballs against Vulkan on Mi50. 
  > 
  > # System Setup
  > 
  > |System|Spec|Note|
  > |:-|:-|:-|
  > |GPU|1x Mi50 32GB|113-D1631700-111 vbios|
  > |CPU|EPYC 7532|Proxmox virtualized 28c/56t allocated|
  > |RAM|8x16GB DDR4 2933Mhz||
  > |OS|Ubuntu Server 24.04|Kernel 6.8.0-106-generic|
  > |ROCm Version|7.13.0a20260321|[TheRock Nightly Page](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#browsing-release-tarballs)|
  > |Vulkan|1.4.341.1||
  > |Llama.ccp Build|8467|Built using recommended commands

### 28. [Quantizers appriciation post](https://www.reddit.com/r/LocalLLaMA/comments/1sc1bhe/quantizers_appriciation_post/)
- **Subreddit:** r/LocalLLaMA | **Score:** 100 | **Comments:** 13 | **Date:** 2026-04-04 05:59 UTC
- **Author:** u/Kahvana | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 1260
  > Hey everyone,
  > 
  > Yesterday I decided to try and learn how to quantize ggufs myself with reasonable quality, in order to understand the magic behind the curtain.
  > 
  > Holy... I did not expect how much work it is, how long it takes, and requires A LOT (500GB!) of storage space for just Gemma-4-26B-A4B in various sizes. There really is an art to configuring them too, with variations between architectures and quant types.
  > 
  > Thanks to unsloth releasing their imatrix file and huggingface showing the weight t

### 29. [Gemma-4 26B-A4B + Opencode on M5 MacBook is *actually good*](https://www.reddit.com/r/LocalLLaMA/comments/1sbaack/gemma4_26ba4b_opencode_on_m5_macbook_is_actually/)
- **Subreddit:** r/LocalLLaMA | **Score:** 72 | **Comments:** 27 | **Date:** 2026-04-03 10:52 UTC
- **Author:** u/maddie-lovelace | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 1260
  > TL;DR, 32gb M5 MacBook Air can run gemma-4-26B-A4B-it-UD-IQ4\_XS at **300t/s PP** and **12t/s generation** (running in low power mode, uses **8W**, making it the first laptop I've used to not get warm and noisy whilst running LLMs). Fast prompt processing + short thinking traces + can actually handle agentic behaviour = Opencode is actually usable from my laptop!
  > 
  > \--
  > 
  > Previously I've been running LLMs off my M1 Max 64gb. And whilst it's been good enough for tinkering and toy use cases, it's nev

### 30. [APEX MoE quantized models boost with 33% faster inference and TurboQuant (14% of speedup in prompt processing)](https://www.reddit.com/r/LocalLLaMA/comments/1s9vzry/apex_moe_quantized_models_boost_with_33_faster/)
- **Subreddit:** r/LocalLLaMA | **Score:** 70 | **Comments:** 27 | **Date:** 2026-04-01 20:13 UTC
- **Author:** u/mudler_it | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 1240
  > I've just released APEX (Adaptive Precision for EXpert Models): a novel MoE quantization technique that outperforms Unsloth Dynamic 2.0 on accuracy while being 2x smaller for MoE architectures. 
  > 
  > Benchmarked on Qwen3.5-35B-A3B, but the method applies to any MoE model. Half the size of Q8. Perplexity comparable to F16. 
  > 
  > Works with stock llama.cpp with no patches. Open source (of course!), with &lt;3 from the [github.com/mudler/LocalAI](http://github.com/mudler/LocalAI) team!
  > 
  > 
  > 
  > https://preview.r

### 31. [Google strongly implies the existence of large Gemma 4 models](https://www.reddit.com/r/LocalLLaMA/comments/1sazp6w/google_strongly_implies_the_existence_of_large/)
- **Subreddit:** r/LocalLLaMA | **Score:** 85 | **Comments:** 19 | **Date:** 2026-04-03 01:16 UTC
- **Author:** u/coder543 | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 1230
  > In the [huggingface card:](https://huggingface.co/google/gemma-4-26B-A4B-it)
  > 
  > &gt; Increased Context Window – The small models feature a 128K context window, while the medium models support 256K.
  > 
  > Small and medium... implying at least one large model! 124B confirmed :P

### 32. [Nemotrons](https://www.reddit.com/r/LocalLLaMA/comments/1s2p8k0/nemotrons/)
- **Subreddit:** r/LocalLLaMA | **Score:** 74 | **Comments:** 23 | **Date:** 2026-03-24 20:21 UTC
- **Author:** u/jacek2023 | **Query match:** "Nemotron Cascade"
- **Relevance score:** 1200
  > There will be 4 at some point :)

### 33. [Gemma 4 Uncensored (autoresearch results)](https://www.reddit.com/r/LocalLLaMA/comments/1sd8c59/gemma_4_uncensored_autoresearch_results/)
- **Subreddit:** r/LocalLLaMA | **Score:** 89 | **Comments:** 12 | **Date:** 2026-04-05 16:40 UTC
- **Author:** u/adefa | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 1130
  > # Gemma 4 Uncensored — all 4 models, MoE expert abliteration, automated research loop
  > 
  > Released uncensored versions of all four Gemma 4 models. bf16 + GGUF for each.
  > 
  > **Collection**: https://huggingface.co/collections/TrevorJS/gemma-4-uncensored-69d2885d6e4fc0581f492698
  > 
  > **Code**: https://github.com/TrevorS/gemma-4-abliteration
  > 
  > ## Results
  > 
  > | Model | Baseline | After | KL Div |
  > |-------|----------|-------|--------|
  > | E2B (2.3B) | 98% | 0.4% | 0.346 |
  > | E4B (4.5B) | 99% | 0.7% | 0.068 |
  > | 26B MoE

### 34. [Gemma 4 is great at real-time Japanese - English translation for games](https://www.reddit.com/r/LocalLLaMA/comments/1sbiqx3/gemma_4_is_great_at_realtime_japanese_english/)
- **Subreddit:** r/LocalLLaMA | **Score:** 70 | **Comments:** 21 | **Date:** 2026-04-03 16:43 UTC
- **Author:** u/KageYume | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 1120
  > When Gemma 3 27B QAT IT was released last year, it was SOTA for local real-time Japanese-English translation for visual novel for a while. So I want to see how Gemma 4 handle this use case.
  > 
  > **Model:**
  > 
  > * Unsloth's gemma-4-26B-A4B-it-UD-Q5\_K\_M
  > * Context: 8192
  > * Reasoning: OFF
  > 
  > **Softwares:**
  > 
  > * Front end: Luna Translator
  > * Back end: LM Studio
  > 
  > **Workflow:**
  > 
  > 1. Luna hooks the dialogue and speaker's name from the game.
  > 2. A [Python script](https://pastebin.com/ADVeZPqT) structures the hooked te

### 35. ["The Child That Surpassed Both Parents" Darwin-35B-A3B-Opus (35B/3B MoE) with Model MRI Technique](https://www.reddit.com/r/LocalLLaMA/comments/1s9irdk/the_child_that_surpassed_both_parents/)
- **Subreddit:** r/LocalLLaMA | **Score:** 47 | **Comments:** 31 | **Date:** 2026-04-01 12:12 UTC
- **Author:** u/Own-Potential-2308 | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 1090
  > Darwin-35B-A3B-Opus is a 35B MoE model (only 3B parameters active) created by SeaWolf-AI / VIDRAFT\_LAB using their new Darwin V5 merging engine.
  > 
  > They built a system that does a deep "CT-scan" (Model MRI) of the parent models layer by layer to figure out what actually works.
  > 
  > Father: Qwen3.5-35B-A3B (strong generalist)
  > 
  > Mother: Claude 4.6 Opus distilled (strong reasoning but apparently had a lot of "dead experts" after distillation)
  > 
  > The merge strategy: transplant the mother's strong reasoning 

### 36. [Recently I did a little performance test of several LLMs on PC with 16GB VRAM](https://www.reddit.com/r/LocalLLaMA/comments/1sc8mwc/recently_i_did_a_little_performance_test_of/)
- **Subreddit:** r/LocalLLaMA | **Score:** 34 | **Comments:** 36 | **Date:** 2026-04-04 13:00 UTC
- **Author:** u/rosaccord | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 1060
  > Qwen 3.5, Gemma-4, Nemotron Cascade 2 and GLM 4.7 flash.
  > 
  > Tested to see how performance (speed) degrades with the context increase.
  > 
  > used llama.cpp and some nice quants better fitting for 16GB VRAM in my RTX 4080.
  > 
  > Here is a result comparison table. Hope you find it useful.
  > 
  > https://preview.redd.it/ylafftgx76tg1.png?width=827&amp;format=png&amp;auto=webp&amp;s=16d030952f1ea710cd3cef65b76e5ad2c3fd1cd3
  > 
  > 

### 37. [TurboQuant seems to work very well on Gemma 4 — and separately, per-layer outlier-aware K quantization is beating current public fork results on Qwen PPL](https://www.reddit.com/r/LocalLLaMA/comments/1sd05sa/turboquant_seems_to_work_very_well_on_gemma_4_and/)
- **Subreddit:** r/LocalLLaMA | **Score:** 62 | **Comments:** 16 | **Date:** 2026-04-05 10:25 UTC
- **Author:** u/Fearless-Wear8100 | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 940
  > I’ve been experimenting with TurboQuant KV cache quantization in llama.cpp (CPU + Metal) on Gemma 4 26B A4B-it Q4\_K\_M on an Apple M4 Pro 48GB, and the results look surprisingly strong.
  > 
  > **Gemma 4 findings**
  > 
  > On Gemma 4, QJL seems to work well, and FWHT as a structured rotation substitute also looks like a good fit for the large attention heads (dk=256/512).
  > 
  > My benchmark results:
  > 
  > * tq3j/q4\_0: 37/37 on quality tests, 8/8 on NIAH
  > * tq2j/q4\_0: 36/37, with the only miss being an empty response
  > 

### 38. [V100 32 Gb : 6h of benchmarks across 20 models with CPU offloading &amp; power limitations](https://www.reddit.com/r/LocalLLaMA/comments/1s5o37v/v100_32_gb_6h_of_benchmarks_across_20_models_with/)
- **Subreddit:** r/LocalLLaMA | **Score:** 37 | **Comments:** 26 | **Date:** 2026-03-28 02:19 UTC
- **Author:** u/icepatfork | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 890
  > I posted a few days ago about my setup here : https://www.reddit.com/r/LocalLLaMA/comments/1s0fje7/nvidia\_v100\_32\_gb\_getting\_115\_ts\_on\_qwen\_coder/ 
  > 
  > \- Ryzen 7600 X &amp; 32 Gb DDR5
  > 
  > \- Nvidia V100 32 GB PCIExp (air cooled)
  > 
  > I run a 6h benchmarks across 20 models (MOE &amp; dense), from Nemotron…Qwen to Deepseek 70B with different configuration of :
  > 
  > \- Power limitation (300w, 250w, 200w, 150w)
  > 
  > \- CPU Offload (100% GPU, 75% GPU, 50% GPU, 25% GPU, 0% GPU)
  > 
  > \- Different context window (u

### 39. [Qwen 3.5 Tool Calling Fixes for Agentic Use: What's Broken, What's Fixed, What You (may) Still Need](https://www.reddit.com/r/LocalLLaMA/comments/1sdhvc5/qwen_35_tool_calling_fixes_for_agentic_use_whats/)
- **Subreddit:** r/LocalLLaMA | **Score:** 44 | **Comments:** 21 | **Date:** 2026-04-05 23:01 UTC
- **Author:** u/FigZestyclose7787 | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 860
  > Posted - What follows after this introduction is generated by Claude Opus 4.6 after hundreds of back and forths with log analysis for tool calls that were not working, and Qwen 3.5 models getting confused from local llm providers as well as Nano-Gpt. I fixed it for my own use with Pi coding agent at the time.
  > 
  > Some of the fixes that were needed are no longer needed (TLDR at the bottom) but most are still applicable, as validated today.
  > 
  > If you use Qwen 3.5 models and are having issues with model

### 40. [SWE-bench results for different KV cache quantization levels](https://www.reddit.com/r/LocalLLaMA/comments/1s28z12/swebench_results_for_different_kv_cache/)
- **Subreddit:** r/LocalLLaMA | **Score:** 36 | **Comments:** 25 | **Date:** 2026-03-24 09:28 UTC
- **Author:** u/burakodokus | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 860
  > I have been running SWE-bench-lite across different KV cache quantization levels. I am still collecting data but I can share the early results.
  > 
  > 
  > 
  > Dashboard: [https://huggingface.co/spaces/burakaydinofficial/Quantuzo](https://huggingface.co/spaces/burakaydinofficial/Quantuzo)
  > 
  > Repo: [https://github.com/burakaydinofficial/Quantuzo](https://github.com/burakaydinofficial/Quantuzo)
  > 
  > Results Dataset: [https://huggingface.co/datasets/burakaydinofficial/Quantuzo](https://huggingface.co/datasets/burakay

### 41. [TurboQuant on Apple Silicon: real benchmarks on Mac Mini M4 16GB and M3 Max 48GB](https://www.reddit.com/r/LocalLLaMA/comments/1sdkav6/turboquant_on_apple_silicon_real_benchmarks_on/)
- **Subreddit:** r/LocalLLaMA | **Score:** 25 | **Comments:** 28 | **Date:** 2026-04-06 00:50 UTC
- **Author:** u/Expensive-String8854 | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 810
  > I’ve been testing TurboQuant this week on two machines and wanted to share the actual numbers.
  > 
  > **Why this matters:** TurboQuant compresses the KV cache, not the model weights. On long contexts, KV cache can take several GB of memory, so reducing it can make a big difference even when throughput stays similar.
  > 
  > **In the setup I tested,** K stays at q8\_0 and V goes to turbo3 (\~3-bit). That asymmetric tradeoff makes sense because errors in the keys affect attention routing more directly, while v

### 42. [The best practice for a SWE to use a local LLM for coding.](https://www.reddit.com/r/LocalLLaMA/comments/1s74pog/the_best_practice_for_a_swe_to_use_a_local_llm/)
- **Subreddit:** r/LocalLLaMA | **Score:** 9 | **Comments:** 36 | **Date:** 2026-03-29 19:38 UTC
- **Author:** u/Feeling_Ad9143 | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 810
  > I am a .Net developer (also large experience with SQL and JS, studying Python) with 7+ years of experience on a number of projects. I am considering switching to MLOps on the verge of .Net and Python. I don't want to lose my edge and I like coding and architecture.
  > 
  > I have a PC with 5070 Rtx 12Gb so it is kind of limited. I am experimenting with models qwen3.5:9b and qwen3.5:35b-a3b with 32K context for now. Just in case I won't have a corporate access to something like Claude Code or would need

### 43. [Small models can be good agents](https://www.reddit.com/r/LocalLLaMA/comments/1rzv31l/small_models_can_be_good_agents/)
- **Subreddit:** r/LocalLLaMA | **Score:** 23 | **Comments:** 29 | **Date:** 2026-03-21 15:59 UTC
- **Author:** u/mikkel1156 | **Query match:** "Nemotron Cascade"
- **Relevance score:** 810
  > I have been messing with some of the smaller models (think sub 30B range), and getting them to do complex tasks. 
  > 
  > My approach is pretty standard: take a big problem and get it to break it down into smaller tasks. They are instructed to create JavaScript code that runs in a sandbox (v8), with custom functions and MCP tools.
  > 
  > Though I don't currently have the hardware to run this myself, I am using a provider to rent GPU by the hour (usually one or two RTX 3090). Keep that in mind for some of thi

### 44. [Share your llama-server init strings for Gemma 4 models.](https://www.reddit.com/r/LocalLLaMA/comments/1sfh2ut/share_your_llamaserver_init_strings_for_gemma_4/)
- **Subreddit:** r/LocalLLaMA | **Score:** 18 | **Comments:** 31 | **Date:** 2026-04-08 03:00 UTC
- **Author:** u/AlwaysLateToThaParty | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 800
  > Hi.  I'm trying to use llama.cpp to give me workable Gemma 4 inference, but I'm not finding anything that works.  I'm using the latest llama.cpp, but I've tested it now on three versions.  I thought it might just require me waiting until llama.ccp caught up, and now the models load, where before they didn't at all, but the same issues persist.  I've tried a few of the ver4 models, but the results are either lobotomized or extremely slow. I tried this one today :
  > 
  > llama-server.exe -m .\models\30B

### 45. [Best model for 4090 as AI Coding Agent](https://www.reddit.com/r/LocalLLaMA/comments/1sdzu1v/best_model_for_4090_as_ai_coding_agent/)
- **Subreddit:** r/LocalLLaMA | **Score:** 6 | **Comments:** 33 | **Date:** 2026-04-06 14:05 UTC
- **Author:** u/Dry_Sheepherder5907 | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 720
  > Good day. I am looking for best local  model for coding agent. I might've missed something or some model which is not that widely used so I cam here for the help.
  > 
  > Currently I have following models I found useful in agentic coding via Google's turbo quant applied on **llama.cpp:**
  > 
  > * GLM 4.7 Flash Q4\_K\_M -&gt; 30B
  > * 30B Nemotron 3 Q4\_K\_M -&gt; 30B
  > * Qwen3 Coder Next Q4\_K\_M -&gt; 80B
  > 
  > I really was trying to get Qwen3 Coder Next to get a decent t/s for input and output as I thought it would 

### 46. [Arandu v0.6.0 is available](https://www.reddit.com/r/LocalLLaMA/comments/1rxayki/arandu_v060_is_available/)
- **Subreddit:** r/LocalLLaMA | **Score:** 30 | **Comments:** 20 | **Date:** 2026-03-18 17:52 UTC
- **Author:** u/fredconex | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 700
  > This is Arandu, a Llama.cpp launcher with:
  > 
  > *  Model management
  > *  HuggingFace Integration
  > *  Llama.cpp GitHub Integration with releases management
  > *  Llama-server terminal launching with easy arguments customization and presets, Internal / External
  > *  Llama-server native chat UI integrated
  > *  Hardware monitor
  > *  Color themes
  > 
  > Releases and source-code:  
  > [https://github.com/fredconex/Arandu](https://github.com/fredconex/Arandu)  
  >   
  > So I'm moving out of beta, I think its been stable enough by no

### 47. [Get 30K more context using Q8 mmproj with Gemma 4](https://www.reddit.com/r/LocalLLaMA/comments/1sdst2i/get_30k_more_context_using_q8_mmproj_with_gemma_4/)
- **Subreddit:** r/LocalLLaMA | **Score:** 31 | **Comments:** 19 | **Date:** 2026-04-06 08:13 UTC
- **Author:** u/Sadman782 | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 690
  > Hey guys, quick follow up to my post yesterday about running Gemma 4 26B.
  > 
  > I kept testing and realized you can just use the Q8\_0 mmproj for vision instead of F16. There is no quality drop, and it actually performed a bit better in a few of my tests (with --image-min-tokens 300 --image-max-tokens 512). You can easily hit 60K+ total context with an FP16 cache and still keep vision enabled.
  > 
  > Here is the Q8 mmproj I used : [https://huggingface.co/prithivMLmods/gemma-4-26B-A4B-it-F32-GGUF/blob/main/

### 48. [AdamBench - a benchmark for local LLMs for agentic coding (on RTX5080 16Gb + 64Gb RAM)](https://www.reddit.com/r/LocalLLaMA/comments/1s4immu/adambench_a_benchmark_for_local_llms_for_agentic/)
- **Subreddit:** r/LocalLLaMA | **Score:** 16 | **Comments:** 21 | **Date:** 2026-03-26 20:18 UTC
- **Author:** u/Real_Ebb_7417 | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 580
  > So... I was looking for the best local models for myself to use them in agentic coding workflows. And this is how this benchmark idea was born. And even though it's very "me-specific", I think that it might be useful for others as well, so I decided to document and publish it.
  > 
  > The full benchmark results, methodology, visalisations etc. can be found here: [https://github.com/tabupl/AdamBench](https://github.com/tabupl/AdamBench)
  > 
  > README (+ prompt files in review\_outputs) should provide all nece

### 49. [Gemma 4 on LocalAI: Vulkan vs ROCm](https://www.reddit.com/r/LocalLLaMA/comments/1sf182p/gemma_4_on_localai_vulkan_vs_rocm/)
- **Subreddit:** r/LocalLLaMA | **Score:** 36 | **Comments:** 11 | **Date:** 2026-04-07 16:38 UTC
- **Author:** u/pipould | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 580
  > # Gemma 4 on LocalAI: Vulkan vs ROCm
  > 
  > Hey everyone! 👋
  > 
  > Just finished running a bunch of benchmarks on the new Gemma 4 models using LocalAI and figured I'd share the results. I was curious how **Vulkan** and **ROCm** backends stack up against each other, and how the **26B MoE** (only ~4B active params) compares to the full **31B dense** model in practice.
  > 
  > ---
  > 
  > Three model variants, each on both Vulkan and ROCm:
  > 
  > | Model | Type | Quant | Source |
  > |---|---|---|---|
  > | gemma-4-26B-A4B-it-APEX | MoE 

### 50. [What small models are you using for background/summarization tasks?](https://www.reddit.com/r/LocalLLaMA/comments/1rqk0gr/what_small_models_are_you_using_for/)
- **Subreddit:** r/LocalLLaMA | **Score:** 6 | **Comments:** 26 | **Date:** 2026-03-11 04:27 UTC
- **Author:** u/Di_Vante | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 580
  > I'm experimenting with using a smaller, faster model for summarization and other background tasks. The main model stays on GPU for chat and tool use (GLM-4.7-flash or Qwen3.5:35b-a3b) while a smaller model (Qwen3.5:4b) runs on CPU for the grunt work.
  > 
  > Honestly been enjoying the results. These new Qwen models really brought the game — I can reliably offload summarization and memory extraction to the small one and get good output. Thinking of experimenting with the smaller models for subagent/a2a 

### 51. [llama-bench ROCm 7.2 on Strix Halo (Ryzen AI Max+ 395) — Qwen 3.5 Model Family](https://www.reddit.com/r/LocalLLaMA/comments/1rorzuk/llamabench_rocm_72_on_strix_halo_ryzen_ai_max_395/)
- **Subreddit:** r/LocalLLaMA | **Score:** 86 | **Comments:** 51 | **Date:** 2026-03-09 05:47 UTC
- **Author:** u/przbadu | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 564
  > # llama-bench ROCm 7.2 on Strix Halo (Ryzen AI Max+ 395) — Qwen 3.5 Model Family
  > 
  > 
  > Running `llama-bench` with **ROCm 7.2** on AMD Ryzen AI Max+ 395 (Strix Halo) with 128GB unified memory.
  > 
  > 
  > All models are from [Unsloth](https://huggingface.co/unsloth) (UD quants).
  > 
  > 
  > ## System Info
  > 
  > 
  > - **CPU/GPU**: AMD Ryzen AI Max+ 395 (Radeon 8060S, 40 CUs, 128GB unified)
  > - **OS**: Fedora
  > - **Kernel**: 6.18.13-200.fc43.x86_64
  > - **Backend**: ROCm 7.2
  > - **llama.cpp build**: d417bc43 (8245)
  > 
  > 
  > ## Benchmarks
  > 
  > 
  > | mod

### 52. [Gemma 4 26B A4B just doesn't want to finish the job... or is it me?](https://www.reddit.com/r/LocalLLaMA/comments/1sconnk/gemma_4_26b_a4b_just_doesnt_want_to_finish_the/)
- **Subreddit:** r/LocalLLaMA | **Score:** 4 | **Comments:** 25 | **Date:** 2026-04-04 23:55 UTC
- **Author:** u/boutell | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 540
  > I've tried Gemma 4 26B A4B under both OpenCode and Claude Code now, on an M2 Macbook Pro with 32GB RAM. Both times using Ollama 0.20.2, so yes, I have the updates that make Ollama Gemma 4 compatible.
  > 
  > I gave it a meaty job to do, one that Opus 4.6 aced under Claude Code last week. Straightforward adapter pattern — we support database "A," now support database "B" by generating a wrapper that implements a subset of the database "A" API. Piles of unit tests available, tons of examples of usage in 

### 53. [LM Studio, Error when loading Gemma-4](https://www.reddit.com/r/LocalLLaMA/comments/1sb7lzw/lm_studio_error_when_loading_gemma4/)
- **Subreddit:** r/LocalLLaMA | **Score:** 8 | **Comments:** 23 | **Date:** 2026-04-03 08:14 UTC
- **Author:** u/Soft-Series3643 | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 540
  > Hey!
  > 
  > Apple M1Max, LM Studio 0.4.9+1 (updated today, release notes say that gemma4-support now included),
  > 
  > Engines/Frameworks: LM Studio MLX 1.4.0, Metal llama.cpp 2.10.1, Harmony (Mac) 0.3.5.
  > 
  > Also installed "mlx-vlm-0.4.3" via terminal.
  > 
  >   
  > When loading gemma-4-26b-a4b-it-mxfp4-mlx, it says:
  > 
  > "Failed to load model.
  > 
  > Error when loading model: ValueError: Model type gemma4 not supported. Error: No module named 'mlx\_vlm.models.gemma4'"
  > 
  > Exactly the same happened with another gemma-4-e2b-instruct

### 54. [Is GLM-4.7-Flash relevant anymore?](https://www.reddit.com/r/LocalLLaMA/comments/1rnwvg6/is_glm47flash_relevant_anymore/)
- **Subreddit:** r/LocalLLaMA | **Score:** 46 | **Comments:** 67 | **Date:** 2026-03-08 05:56 UTC
- **Author:** u/HumanDrone8721 | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 540
  > In the last week I've seen a lot of Qwen related work and optimizations, but close to nothing related to GLM open-weights models, are they still relevant or they've been fully superseded by the latest Qwen?

### 55. [Mapping True Coding Efficiency (Coding Index vs. Compute Proxy)](https://www.reddit.com/r/LocalLLaMA/comments/1sd4trk/mapping_true_coding_efficiency_coding_index_vs/)
- **Subreddit:** r/LocalLLaMA | **Score:** 9 | **Comments:** 22 | **Date:** 2026-04-05 14:19 UTC
- **Author:** u/NewtMurky | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 530
  > TPS (Tokens Per Second) is a misleading metric for speed. A model can be "fast" but use 5x more reasoning tokens to solve a bug, making it slower to reach a final answer.
  > 
  > I mapped [**ArtificialAnalysis.ai**](http://ArtificialAnalysis.ai) data to find the "Efficiency Frontier"—models that deliver the highest coding intelligence for the least "Compute Proxy" (Active Params × Tokens).
  > 
  > **The Data:**
  > 
  > * **Coding Index:** Based on Terminal-Bench Hard and SciCode.
  > * **Intelligence Index v4.0:** Inclu

### 56. [GGUF (llama.cpp) vs MLX Round 2: Your feedback tested, two models, five runtimes. Ollama adds overhead. My conclusion. Thoughts?](https://www.reddit.com/r/LocalLLaMA/comments/1s49lvh/gguf_llamacpp_vs_mlx_round_2_your_feedback_tested/)
- **Subreddit:** r/LocalLLaMA | **Score:** 10 | **Comments:** 21 | **Date:** 2026-03-26 14:50 UTC
- **Author:** u/arthware | **Query match:** "Rapid-MLX"
- **Relevance score:** 520
  > Two weeks ago I posted here that [MLX was slower than GGUF on my M1 Max](https://www.reddit.com/r/LocalLLaMA/comments/1rs059a/mlx_is_not_faster_i_benchmarked_mlx_vs_llamacpp). You gave feedback, pointed out I picked possibly the worst model for MLX. Broken prompt caching ([mlx-lm#903](https://github.com/ml-explore/mlx-lm/issues/903)), hybrid attention MLX can't optimize, bf16 on a chip that doesn't do bf16.
  > 
  > So I went and tested almost all of your hints and recommendations.  
  > Two mature models (

### 57. [Benchmarked 11 MLX models on M3 Ultra — here's which ones are actually smart and fast](https://www.reddit.com/r/LocalLLaMA/comments/1rkcvqa/benchmarked_11_mlx_models_on_m3_ultra_heres_which/)
- **Subreddit:** r/LocalLLaMA | **Score:** 53 | **Comments:** 59 | **Date:** 2026-03-04 05:20 UTC
- **Author:** u/Striking-Swim6702 | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 513
  > **UPDATE (2026-03-05):** Expanded to **17 models** based on your feedback! Added Qwen3.5-27B/9B/4B, GLM-4.5-Air, Devstral-Small-2, Mistral-Small-3.2. Fixed a parser bug that was killing GPT-OSS-20B scores (3% → 80% tool calling). Added RAM and Avg columns as requested. Original 11-model table preserved below for reference.
  > 
  > |Model|Quant|RAM|Decode|Tools|Code|Reason|General|Avg|
  > |:-|:-|:-|:-|:-|:-|:-|:-|:-|
  > |Qwen3.5-122B-A10B|8bit|129.8 GB|43 t/s|87%|**90%**|**90%**|**90%**|**89%**|
  > |Qwen3.5-122B

### 58. [Basic PSA. PocketPal got updated, so runs Gemma 4.](https://www.reddit.com/r/LocalLLaMA/comments/1scsgid/basic_psa_pocketpal_got_updated_so_runs_gemma_4/)
- **Subreddit:** r/LocalLLaMA | **Score:** 21 | **Comments:** 14 | **Date:** 2026-04-05 02:59 UTC
- **Author:** u/Sambojin1 | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 490
  > Just because I've seen a couple of "I want this on Android" questions, PocketPal got updated a few hours ago, and runs Gemma 4 2B and 4B fine. At least on my hardware (crappy little moto g84, 12gig ram workhorse phone). Love an app that gets regular updates.
  > 
  > I'm going to try and squeak 26B a4 iq2 quantization into 12gigs of ram, on a fresh boot, but I'm almost certain it can't be done due to Android bloat.
  > 
  > But yeah, 2B and 4B work fine and quickly under PocketPal. Hopefully their next one is 7

### 59. [Vulkan now faster on PP AND TG on AMD Hardware?](https://www.reddit.com/r/LocalLLaMA/comments/1rpcdrb/vulkan_now_faster_on_pp_and_tg_on_amd_hardware/)
- **Subreddit:** r/LocalLLaMA | **Score:** 10 | **Comments:** 19 | **Date:** 2026-03-09 20:52 UTC
- **Author:** u/XccesSv2 | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 480
  > Hey guys, i did some new llama-benches with newest llama.cpp updates and compared my vulkan and rocm build again. I am on Fedora 43 with ROCm 7.1.1 with an AMD Radeon Pro W7800 48GB and Radeon 7900 XTX 24GB  
  > In the past, ROCm was always faster on PP but compareable or 10% slower on TG. But now it's a complete different story:
  > 
  > Qwen3.5-35B-A3B-UD-Q8\_K\_XL.gguf -ngl 999 -dev Vulkan0/Vulkan1 -ts 0.3/0.67
  > 
  > ggml\_vulkan: Found 2 Vulkan devices:  
  > ggml\_vulkan: 0 = AMD Radeon RX 7900 XTX (RADV NAVI3

### 60. [[Benchmark] Altered Riddles: Can LLMs ignore what they've memorised?](https://www.reddit.com/r/LocalLLaMA/comments/1sdsxyl/benchmark_altered_riddles_can_llms_ignore_what/)
- **Subreddit:** r/LocalLLaMA | **Score:** 19 | **Comments:** 14 | **Date:** 2026-04-06 08:22 UTC
- **Author:** u/marcodsn | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 470
  > In the past year you may have encountered the following prompt:
  > 
  > &gt;The surgeon, who is the boy's father, says, 'I cannot operate on this boy—he's my son!'. Who is the surgeon to the boy?
  > 
  > If you try to give this prompt to an LLM *right now* you will probably still receive “The mother” as an answer, even though the text *explicitly states* that the surgeon is the boy’s father; this is probably due to the fact that this prompt is an alteration of a very common “riddle”, to which the answer is, i

### 61. [Gemma 4 26B-A4B MoE running at 45-60 tok/s on DGX Spark — here's how](https://www.reddit.com/r/LocalLLaMA/comments/1sbekgc/gemma_4_26ba4b_moe_running_at_4560_toks_on_dgx/)
- **Subreddit:** r/LocalLLaMA | **Score:** 22 | **Comments:** 12 | **Date:** 2026-04-03 14:06 UTC
- **Author:** u/CoconutMario | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 460
  > Spent half the night on getting google/gemma-4-26B-A4B-it running fast on a single NVIDIA DGX Spark (128GB unified memory, GB10 Blackwell). Some things I learned that might save others time:
  > 
  > **NVFP4 quantization**
  > 
  > The 26B MoE model is \~49GB in BF16 — runs but slowly. NVFP4 brings it down to 16.5GB with 3x compression. The catch: Google stores MoE expert weights as fused 3D tensors that no existing quantization tool handles. NVIDIA's modelopt silently skips them (91% of the model!). I wrote a 

### 62. [Gemma 4 Architecture Comparison](https://www.reddit.com/r/LocalLLaMA/comments/1sbdr75/gemma_4_architecture_comparison/)
- **Subreddit:** r/LocalLLaMA | **Score:** 40 | **Comments:** 3 | **Date:** 2026-04-03 13:33 UTC
- **Author:** u/seraschka | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 460
  > Flagship open-weight release days are always exciting. Was just reading through the Gemma 4 reports, configs, and code, and here are my takeaways: Architecture-wise, besides multi-model support, Gemma 4 (31B) looks pretty much unchanged compared to Gemma 3 (27B). 
  > 
  > [Link to the comparison page: https:\/\/sebastianraschka.com\/llm-architecture-gallery\/?compare=gemma-3-27b&amp;#37;2Cgemma-4-31b](https://preview.redd.it/iisaroou8zsg1.png?width=1444&amp;format=png&amp;auto=webp&amp;s=662c000e32ae22

### 63. [I was bored - so i tested the h... out of a bunch of models - so you dont have to :)](https://www.reddit.com/r/LocalLLaMA/comments/1s2r7w2/i_was_bored_so_i_tested_the_h_out_of_a_bunch_of/)
- **Subreddit:** r/LocalLLaMA | **Score:** 4 | **Comments:** 20 | **Date:** 2026-03-24 21:34 UTC
- **Author:** u/leonbollerup | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 440
  > So.. i was bored.. and i decided to run a test - using the same prompt on a bunch of models.. i then used Gemini 3 Pro an Opus 4.6 to verify the results.  
  > \--
  > 
  > The prompt:  
  > \---  
  > **Question:**
  > 
  > 
  > 
  > A city is planning to replace its diesel bus fleet with electric buses over the next 10 years. The city currently operates 120 buses, each driving an average of 220 km per day. A diesel bus consumes 0.38 liters of fuel per km, while an electric bus consumes 1.4 kWh per km.
  > 
  > 
  > 
  > Relevant data:
  > 
  > * Diesel

### 64. [My prompt is causing seizures on three models?](https://www.reddit.com/r/LocalLLaMA/comments/1sc5bhg/my_prompt_is_causing_seizures_on_three_models/)
- **Subreddit:** r/LocalLLaMA | **Score:** 2 | **Comments:** 20 | **Date:** 2026-04-04 10:00 UTC
- **Author:** u/PiratesOfTheArctic | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 420
  > Hi everyone, I've been trying to find a suitable reddit group to ask this, and failed (if there is one about prompt questions please let me know!)
  > 
  > I'm trying to create a basic date list:
  > 
  > create dates in DD/MM/YY format from 1 Feb 2026 to 30 April 2026, excluding weekends (saturday and sunday). Make a list formatted as a column. sort by earliest date first. do not hallucinate. do not make mistakes.
  > 
  > I've tried on:
  > 
  > * Qwen3.5-35B-A3B-UD-IQ4\_XS.gguf
  > * gemma-4-E4B-it-Q4\_K\_M.gguf
  > * Phi-4-mini-re

### 65. [Artificial Analysis Intelligence Index vs weighted model size of open-source models](https://www.reddit.com/r/LocalLLaMA/comments/1rljbix/artificial_analysis_intelligence_index_vs/)
- **Subreddit:** r/LocalLLaMA | **Score:** 75 | **Comments:** 31 | **Date:** 2026-03-05 14:40 UTC
- **Author:** u/Balance- | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 411
  > Same plot as earlier this morning, but now with more models that only Qwen.
  > 
  > Note that dense models use their listed parameter size (e.g., 27B), while Mixture-of-Experts models (e.g., 397B A17B) are converted to an effective size using \`sqrt(total\*active)\` to approximate their compute-equivalent scale.
  > 
  > Data source: [https://artificialanalysis.ai/leaderboards/models](https://artificialanalysis.ai/leaderboards/models)

### 66. [Qwen 3 coder 30B is quite impressive for coding](https://www.reddit.com/r/LocalLLaMA/comments/1sf8zp8/qwen_3_coder_30b_is_quite_impressive_for_coding/)
- **Subreddit:** r/LocalLLaMA | **Score:** 7 | **Comments:** 16 | **Date:** 2026-04-07 21:15 UTC
- **Author:** u/ag789 | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 390
  > This is a followup for [https://www.reddit.com/r/LocalLLaMA/comments/1seqsa2/glm\_47\_flash\_is\_quite\_impressive\_for\_coding/](https://www.reddit.com/r/LocalLLaMA/comments/1seqsa2/glm_47_flash_is_quite_impressive_for_coding/)
  > 
  > This is another 'old' model (as 'newer and better' models has evolved  after that), but that (30B) models which presumbly with 4-8 bit quant fits in 32 GB memory are still 'hard to find'. the 'newer and better' models many have well more parameters than 30B.
  > 
  > The models

### 67. [Does anyone have experience using claude to create plans, then hand off the coding work to a local model running on a 5090?](https://www.reddit.com/r/LocalLLaMA/comments/1s9deau/does_anyone_have_experience_using_claude_to/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 19 | **Date:** 2026-04-01 07:14 UTC
- **Author:** u/ArugulaAnnual1765 | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 380
  > Given the recent usage reductions in claude, people have been looking for ways to save usage.
  > 
  > Obviously a local model running on a single 5090 will probably not be good enough to plan, however could something like qwen3.5-35b-a3b q4 be reliably used for just the code edits?
  > 
  > 

### 68. [First time using local models for coding, please share your system prompts and tips](https://www.reddit.com/r/LocalLLaMA/comments/1s4c1mp/first_time_using_local_models_for_coding_please/)
- **Subreddit:** r/LocalLLaMA | **Score:** 7 | **Comments:** 15 | **Date:** 2026-03-26 16:18 UTC
- **Author:** u/Slice-of-brilliance | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 370
  > Hi there, I have used local models before but only for normal conversations. I have never used them for coding. I would like to do so. I searched around and came to know that GLM 4.7 Flash is one of the best options right now. Now I would like to learn what kind of system prompts and other settings you configure to get the best from your experience and use case.
  > 
  > Please share! Thanks!

### 69. [M3 Pro Macbook, 36GB RAM feels slow when running Gemma 26B or E4B](https://www.reddit.com/r/LocalLLaMA/comments/1seq1nf/m3_pro_macbook_36gb_ram_feels_slow_when_running/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 17 | **Date:** 2026-04-07 08:24 UTC
- **Author:** u/impish19 | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 340
  > Hello
  > 
  >   
  > I have a M3 Pro machine with 36 gigs of RAM. I was hoping to run at least E4B with 10 tokens/sec or higher but both E4B and 26B run much slower. E4B runs at around 4.3 tokens/sec and 26B runs at around 3.2 tokens/sec. I'm running them through llama.cpp. 
  > 
  > I was hoping to run one of these with Hermes or OpenClaw later but given how slow they are there's no way they're going to be able to handle OpenClaw.
  > 
  >   
  > I've seen people recommend this configuration earlier for running OpenClaw loca

### 70. [My OpenCode local LLM agent setup — what would you change?](https://www.reddit.com/r/LocalLLaMA/comments/1rqheqo/my_opencode_local_llm_agent_setup_what_would_you/)
- **Subreddit:** r/LocalLLaMA | **Score:** 7 | **Comments:** 13 | **Date:** 2026-03-11 02:23 UTC
- **Author:** u/Shoddy_Bed3240 | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 330
  > I’ve been fine-tuning my **OpenCode** workflow to balance API costs with local hardware performance. Currently running **llama.cpp** locally with a focus on high-quantization models 
  > 
  > # The Agent Stack
  > 
  > |**Agent**|**Model**|**Quant**|**Speed (t/s)**|
  > |:-|:-|:-|:-|
  > |**plan**|Kimi K2.5 (OpenCode Go)|API|\~45|
  > |**build / debug**|Qwen3 Coder Next|Q8\_K\_XL|47|
  > |**review**|Qwen3.5-122B-A10B|Q8\_K\_XL|18|
  > |**security**|MiniMax M2.5|Q4\_K\_XL|20|
  > |**docs / test**|GLM-4.7-Flash|Q8\_K\_XL|80|
  > 
  > # The Logi

### 71. [How do I access a llama.cpp server instance with the Continue extension for VSCodium?](https://www.reddit.com/r/LocalLLaMA/comments/1rz900l/how_do_i_access_a_llamacpp_server_instance_with/)
- **Subreddit:** r/LocalLLaMA | **Score:** 2 | **Comments:** 15 | **Date:** 2026-03-20 21:33 UTC
- **Author:** u/warpanomaly | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 320
  > If I'm running GLM-4.7-Flash-GGUF:Q6\_K\_XL from the powershell terminal like this `.\llama-server.exe -hf unsloth/GLM-4.7-Flash-GGUF:Q6_K_XL --host` `127.0.0.1` `--port 10000 --ctx-size 32000 --n-gpu-layers 99`, how do I access it from the Continue plugin in VSCodium?  
  > 
  > The "Add Chat model" optional only shows pre-configured cloud based API option like Claude and ChatGPT, and the only local models I can find is Ollama and a version of Llama.cpp that doesn't work.
  > 
  > This is my llama-server insta

### 72. [Fix: OpenClaw + Ollama local models silently timing out? The slug generator   is blocking your agent (and 4 other fixes)](https://www.reddit.com/r/LocalLLaMA/comments/1sdnf43/fix_openclaw_ollama_local_models_silently_timing/)
- **Subreddit:** r/LocalLLaMA | **Score:** 5 | **Comments:** 13 | **Date:** 2026-04-06 03:12 UTC
- **Author:** u/After-Confection-592 | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 310
  > I spent a full day debugging why Gemma 4 26B (and E4B) would never respond through OpenClaw on Telegram, even though `ollama run gemma4` worked perfectly fine. Sharing everything I found.
  > 
  > **Hardware:** Mac Studio M4 Max, 128GB unified memory
  > 
  > **Setup:** OpenClaw 2026.4.2 + Ollama 0.20.2 + Gemma 4 26B-A4B Q8\_0
  > 
  > # The Symptoms
  > 
  > * `/new` works instantly, shows correct model
  > * Send "hi" and nothing happens. No typing indicator, no response
  > * No visible errors in the gateway log
  > * Model responds in

### 73. [LLM performance decreased significantly over time using the same models and same hardware in LMStudio.](https://www.reddit.com/r/LocalLLaMA/comments/1s7jmw2/llm_performance_decreased_significantly_over_time/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 15 | **Date:** 2026-03-30 07:15 UTC
- **Author:** u/fernandollb | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 300
  > Recently I started using LMStudio to load local models and use them with ClawdBot, when I started using it I could offload 100% of the model (Qwen3.5-35b-a3b) to my 4090 with 100.000 context and it was flying. Right now I have to set context at 60.000 to achieve the same speed.
  > 
  > I have tried starting new ClawdBot sessions and restarting LM Studio but nothing seems to help. Is there a fix for this issue?

### 74. [Gemma4 (26B-A4B) is genuinely great and fast for local use](https://www.reddit.com/r/LocalLLaMA/comments/1sbb073/gemma4_26ba4b_is_genuinely_great_and_fast_for/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 15 | **Date:** 2026-04-03 11:30 UTC
- **Author:** u/garg-aayush | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 300
  > https://reddit.com/link/1sbb073/video/5iuejqilmysg1/player
  > 
  > Gemma4 is genuinely great for local use. I spent some time playing around with it this afternoon and was really impressed with gemma-4-26B-A4B capabilities and speep of \~145 t/s (on RTX4090). This coupled with web search mcp and image support delivers a really nice chat experience.   
  >   
  > You can further improve this experience with a few simple tricks and a short system prompt. I have written a blog post that covers how I set it up and

### 75. [Is Nemotron-Cascade-2-30B-A3B better than Qwen3.5 27B?](https://www.reddit.com/r/LocalLLaMA/comments/1s8gz3x/is_nemotroncascade230ba3b_better_than_qwen35_27b/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 15 | **Date:** 2026-03-31 07:48 UTC
- **Author:** u/Ok-Internal9317 | **Query match:** "Nemotron Cascade"
- **Relevance score:** 300
  > Is it benchmaxxed or actually useful, have y'all tied it?

### 76. [What is everyones thoughts on Nemotron-Cascade 30b a3b](https://www.reddit.com/r/LocalLLaMA/comments/1rzlegt/what_is_everyones_thoughts_on_nemotroncascade_30b/)
- **Subreddit:** r/LocalLLaMA | **Score:** 12 | **Comments:** 9 | **Date:** 2026-03-21 07:33 UTC
- **Author:** u/Odd-Ordinary-5922 | **Query match:** "Nemotron Cascade"
- **Relevance score:** 300
  > heres the model [https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B](https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B)

### 77. [Speed difference on Gemma 4 26B-A4B between Bartowski Q4_K_M and Unsloth Q4_K_XL](https://www.reddit.com/r/LocalLLaMA/comments/1sc52ge/speed_difference_on_gemma_4_26ba4b_between/)
- **Subreddit:** r/LocalLLaMA | **Score:** 7 | **Comments:** 10 | **Date:** 2026-04-04 09:45 UTC
- **Author:** u/BelgianDramaLlama86 | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 270
  > I've noticed this on Qwen3.5 35B before as well, there is a noticeable speed difference between Unsloth's Q4\_K\_XL and Bartowski's Q4\_K\_M on the same model, but Gemma 4 seems particularly harsh in this regard: Bartowski gets 38 tk/s, Unsloth gets 28 tk/s... everything else is the same, settings wise. This is with the latest Unsloth quant update and latest llama.cpp version. Their size is only \~100 MB apart. Anyone have any idea why this speed difference is there?
  > 
  > Btw, on Qwen3.5 35B I notic

### 78. [Best AI coding agent for Gemma-4-26B?](https://www.reddit.com/r/LocalLLaMA/comments/1sd4qfn/best_ai_coding_agent_for_gemma426b/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 13 | **Date:** 2026-04-05 14:15 UTC
- **Author:** u/Pristine-Tax4418 | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 260
  > For Qwen3-Coder-Next, Qwen3.5-122B-A10B and Qwen3.5-35B-A3B, I use qwen coder cli.
  > 
  > I also tried OpenCode and Mistral Vibe for Qwen models, but got worse results.
  > 
  > 
  > 
  > For Gemma, there's [https://github.com/google-gemini/gemini-cli](https://github.com/google-gemini/gemini-cli) — but unfortunately it doesn't support local models out of the box.
  > 
  > 
  > 
  > In your opinion, what is the best agent environment for Gemma?

### 79. [My new favorite warp speed ! qwen3.5-35b-a3b-turbo-swe-v0.0.1](https://www.reddit.com/r/LocalLLaMA/comments/1s7650p/my_new_favorite_warp_speed/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 12 | **Date:** 2026-03-29 20:32 UTC
- **Author:** u/PhotographerUSA | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 240
  > This version fly's on my machine and get quick accurate results. I highly recommend it !  
  > It's better than the base module and loads real quick !
  > 
  > [https://huggingface.co/rachpradhan/Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1](https://huggingface.co/rachpradhan/Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1)
  > 
  > My specs are Ryzen 9 5950x, DDR4-3400 64GB, 18TB of solid state and 3070 GTX 8GB. I get 35TK/sec

### 80. [Anyone have gemma4-31b or 26b working with codex/claude localy?](https://www.reddit.com/r/LocalLLaMA/comments/1sdtp8e/anyone_have_gemma431b_or_26b_working_with/)
- **Subreddit:** r/LocalLLaMA | **Score:** 2 | **Comments:** 11 | **Date:** 2026-04-06 09:09 UTC
- **Author:** u/dopey_se | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 240
  > I run a pair of P100s locally, and for past while been quite happy with Qwen3.5-27b 4bit with 250k context.
  > 
  > I have been able to ask it to fetch tickets from my self hosted youtrack, implement, update tickets, progress tickets, commit, push, etc. And in general it always produces still-building-running code albeit some features take a few iterations. The general idea is to have it regularly check for new tickets in Y status, and do a defined skill to process these tickets.. Letting it run unatte

### 81. [Gemma 4](https://www.reddit.com/r/LocalLLaMA/comments/1sdemwf/gemma_4/)
- **Subreddit:** r/LocalLLaMA | **Score:** 5 | **Comments:** 9 | **Date:** 2026-04-05 20:47 UTC
- **Author:** u/lordsnoake | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 230
  > Howdy!
  > 
  > So I am curious to know, how is everyone getting to run Gemma 4?
  > 
  > I can't run Gemma 4 on any model locally and when I do, the model spazs out and returns the infamous &lt;unused4&gt; response.
  > 
  > I have tried llama-server, ollama, and LMS studio.
  > 
  > for each one, I tried different models from various authors like unsloth, bartowski, etc.
  > 
  > My question, is; how does everyone set it up for agentic use like Claude or crush?
  > 
  > my hardware: gmktec strix halo 128GB
  > 
  > OS: Ubuntu 24.04
  > 
  > I followed the 

### 82. [Gemma4 26B-A4B &gt; Gemma4 31B.  Qwen3.5 27B &gt; Qwen3.5 35B-A3B. Gemma4 26B-A4B &gt;= Qwen3.5  35-A3B. Current state. Tell me why I am right or wrong.](https://www.reddit.com/r/LocalLLaMA/comments/1sbqnsz/gemma4_26ba4b_gemma4_31b_qwen35_27b_qwen35_35ba3b/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 11 | **Date:** 2026-04-03 21:38 UTC
- **Author:** u/inthesearchof | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 220
  > Normally i prefer the dense qwen over MoE. It seems to have flipped for Gemma.  Maybe things will change after everything gets better optimized but currently liking Gemma4's MoE

### 83. [Gemma 4 26B A4B Single Page ASCII Chatbot Design](https://www.reddit.com/r/LocalLLaMA/comments/1scq6mb/gemma_4_26b_a4b_single_page_ascii_chatbot_design/)
- **Subreddit:** r/LocalLLaMA | **Score:** 16 | **Comments:** 3 | **Date:** 2026-04-05 01:07 UTC
- **Author:** u/Reaper_9382 | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 220
  > Built a single chatbot HTML page using Gemma 4 26B A4B running locally sharded between my 7900 XT and 3060 Ti with 32K context window at 50-65 t/s.  
  >   
  > Connects to LM Studio's API with full streaming, Markdown rendering, model selector, 6 parameter sliders, message editing with history branching, regenerate, abort, and system prompt support.  
  >   
  > Claude helped fix two DOM bugs that Gemma couldn't. Everything else was Gemma 4.  
  >   
  > GitHub: [https://github.com/Shoggoth43/Gemma-4-26B-A4B-Generatio

### 84. [GLM 4.7 flash is quite impressive for coding](https://www.reddit.com/r/LocalLLaMA/comments/1seqsa2/glm_47_flash_is_quite_impressive_for_coding/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 11 | **Date:** 2026-04-07 09:10 UTC
- **Author:** u/ag789 | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 220
  > GLM 4.7 flash  
  > [https://z.ai/blog/glm-4.7](https://z.ai/blog/glm-4.7)  
  > [https://huggingface.co/models?sort=trending&amp;search=glm-4.7](https://huggingface.co/models?sort=trending&amp;search=glm-4.7)  
  > [https://www.reddit.com/r/LocalLLaMA/comments/1qkqvkr/yesterday\_i\_used\_glm\_47\_flash\_with\_my\_tools\_and\_i/](https://www.reddit.com/r/LocalLLaMA/comments/1qkqvkr/yesterday_i_used_glm_47_flash_with_my_tools_and_i/)
  > 
  > is quite impressive for coding.  
  > I'm using GLM 4.7 REAP 23B Q4\_K\_M.gguf

### 85. [Bypassing CoreML: Natively training and running LLMs directly on the Apple Neural Engine (170 tok/s)](https://www.reddit.com/r/LocalLLaMA/comments/1rl9fl4/bypassing_coreml_natively_training_and_running/)
- **Subreddit:** r/LocalLLaMA | **Score:** 34 | **Comments:** 19 | **Date:** 2026-03-05 05:44 UTC
- **Author:** u/No_Gap_4296 | **Query match:** "Rapid-MLX"
- **Relevance score:** 216
  > It is hard to communicate how frustratingly opaque Apple's hardware stack can be. We all target the Mac's GPU via MLX or llama.cpp for our local models, but there is a dedicated AI accelerator—the Apple Neural Engine (ANE)—sitting completely dark for LLM workloads. CoreML treats it as a black-box scheduler, stripping away any direct control or ability to train. 
  > 
  > There are a few real caveats here, but imo the fundamental constraint to using the ANE hasn't been compute (it actually pulls \~19 TFL

### 86. [Speculative Decoding Single 3090 Qwen Model Testing](https://www.reddit.com/r/LocalLLaMA/comments/1s6cw23/speculative_decoding_single_3090_qwen_model/)
- **Subreddit:** r/LocalLLaMA | **Score:** 5 | **Comments:** 8 | **Date:** 2026-03-28 21:41 UTC
- **Author:** u/Alert_Cockroach_561 | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 210
  > Had Claude summarize, or i would have put out alot of slop
  > 
  > # Spent 24 hours benchmarking speculative decoding on my RTX 3090 for my HVAC business — here are the results
  > 
  > I'm building an internal AI platform for my small HVAC company (just me and my wife). Needed to find the best local LLM setup for a Discord bot that handles customer lookups, quote formatting, equipment research, and parsing messy job notes. Moved from Ollama on Windows to llama.cpp on WSL Linux with speculative decoding.
  > 
  > # Ha

### 87. [LM Studio DGX Spark generation speeds for 23 different models](https://www.reddit.com/r/LocalLLaMA/comments/1s4yc0w/lm_studio_dgx_spark_generation_speeds_for_23/)
- **Subreddit:** r/LocalLLaMA | **Score:** 1 | **Comments:** 10 | **Date:** 2026-03-27 08:28 UTC
- **Author:** u/Late_Night_AI | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 210
  > Salutations lads, I ran 23 different models on my Gigabyte Atom (DGX Spark) in LM Studio to benchmark their generation speeds.
  > 
  > Theres no real rhyme or reason to the selection of models other than they’re more common ones that I have 🤷‍♂️
  > 
  > Im using LM Studio 4.7 with Cuda 13 llama.cpp (Linux ARM) v2.8.0
  > 
  > I loaded the model with their full context window, other than that i left all the other settings as the default stuff.
  > 
  > My method of testing their generation speeds was extremely strict and held

### 88. [Best model for my rig (9950X3D, RTX 6000 96GB, 192GB DDR5, 9100 4TB) - C coding / cybersec](https://www.reddit.com/r/LocalLLaMA/comments/1rzrd6g/best_model_for_my_rig_9950x3d_rtx_6000_96gb_192gb/)
- **Subreddit:** r/LocalLLaMA | **Score:** 1 | **Comments:** 10 | **Date:** 2026-03-21 13:22 UTC
- **Author:** u/anon33anon | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 210
  > What's the absolute best model (or a combination of them for different tasks) for:  
  > \-Architectural choices, detailed planning, overview of the system to be engineered (usually it's either C clients, either C mixed with Kotlin (Android) or Swift (iOS), and partially JS for clients, usually GO for backends with many services)  
  > \-Often I need MISRA C (C89) for other high-assurance projects (cars, aerospace, trains, etc), sometimes simpler IoT (ESP or RPI)  
  > \-Decent for deployments  
  > \-Often cod

### 89. [Best Gemma4 llama.cpp command switches/parameters/flags? Unsloth GGUF?](https://www.reddit.com/r/LocalLLaMA/comments/1sbpf86/best_gemma4_llamacpp_command/)
- **Subreddit:** r/LocalLLaMA | **Score:** 2 | **Comments:** 9 | **Date:** 2026-04-03 20:50 UTC
- **Author:** u/Fulminareverus | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 200
  > Can anyone share their command string they use to run Gemma 4? For example, I have previously used this for qwen35:
  > 
  > llama-server.exe --hf-repo unsloth/Qwen3.5-35B-A3B-GGUF --hf-file Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf --port 11433 --host 0.0.0.0 -c 131072 -ngl 999 -fa on --cache-type-k q4_0 --cache-type-v q4_0 --jinja --temp 1.0 --top-p 0.95 --min-p 0.0 --top-k 20 -b 4096 --repeat-penalty 1.0 --presence-penalty 1.5 --no-mmap
  > 
  > I'm trying to find the best settings to run it, and curious what others a

### 90. [gemma-4-26B-A4B tool calling performance?](https://www.reddit.com/r/LocalLLaMA/comments/1seamje/gemma426ba4b_tool_calling_performance/)
- **Subreddit:** r/LocalLLaMA | **Score:** 2 | **Comments:** 9 | **Date:** 2026-04-06 20:30 UTC
- **Author:** u/edmcman | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 200
  > Has anyone else been having trouble with tool calling on gemma-4-26B-A4B?  I tried unsloth's GGUFs, both BF16 and UD-Q4\_K\_XL.  I sometimes get a response that has no text or tool calls; it just is empty, and this confuses my coding agent.  gemma-4-31B UD-Q4\_K\_XL seems to be working fine.  Just wondering if it is just me.

### 91. [Anyone got Gemma 4 26B-A4B running on VLLM?](https://www.reddit.com/r/LocalLLaMA/comments/1se39el/anyone_got_gemma_4_26ba4b_running_on_vllm/)
- **Subreddit:** r/LocalLLaMA | **Score:** 6 | **Comments:** 7 | **Date:** 2026-04-06 16:10 UTC
- **Author:** u/toughcentaur9018 | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 200
  > If yes, which quantized model are you using abe what’s your vllm serve command?
  > 
  > I’ve been struggling getting that model up and running on my dgx spark gb10. I tried the intel int4 quant for the 31B and it seems to be working well but way too slow. 
  > 
  > Anyone have any luck with the 26B? 

### 92. [I think I got solutions for Qwen 3.5 tool call in thinking block](https://www.reddit.com/r/LocalLLaMA/comments/1sccqt2/i_think_i_got_solutions_for_qwen_35_tool_call_in/)
- **Subreddit:** r/LocalLLaMA | **Score:** 3 | **Comments:** 8 | **Date:** 2026-04-04 15:52 UTC
- **Author:** u/Interesting-Print366 | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 190
  > I have also experienced that when using the qwen3.5 model, tool\_call often does not execute when called inside &lt;thinking&gt;, and I have heard that many others are experiencing the same issue.
  > 
  >   
  > I have tried to reproduce this several times, and while it may not be entirely accurate, it seems to attempt to skip thinking and make a tool call immediately when it is clear from the preceding context which tool call the model should make.
  > 
  >   
  > However, since the qwen3.5 model forces thinking to o

### 93. [Qwen 3.5 4B is the first small open-source model to solve this.](https://www.reddit.com/r/LocalLLaMA/comments/1roi1ry/qwen_35_4b_is_the_first_small_opensource_model_to/)
- **Subreddit:** r/LocalLLaMA | **Score:** 41 | **Comments:** 11 | **Date:** 2026-03-08 22:08 UTC
- **Author:** u/ConfidentDinner6648 | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 189
  > I ran a very small abstraction test:
  > 
  > 11118888888855 -&gt; 118885
  > 79999775555 -&gt; 99755
  > AAABBBYUDD -&gt; ?
  > Qwen 3.5 4B was the first small open source model  to solve it. That immediately caught my attention, because a lot of much bigger models failed.
  > 
  > Models that failed this test in my runs:
  > GPT-4
  > GPT-4o
  > GPT-4.1
  > o1-mini
  > o3-mini
  > o4-mini
  > OSS 20B
  > OSS 120B
  > Gemini 2.5 Flash
  > All Qwen 2.5 sizes
  > Qwen 3.0 only passed with Qwen3-235B-A22B-2507.
  > 
  > Models that got it right in my runs:
  > o1 — first to solve

### 94. [Yet another post of genuinely impressed with Qwen3.5](https://www.reddit.com/r/LocalLLaMA/comments/1rl1j07/yet_another_post_of_genuinely_impressed_with/)
- **Subreddit:** r/LocalLLaMA | **Score:** 44 | **Comments:** 9 | **Date:** 2026-03-04 23:38 UTC
- **Author:** u/Di_Vante | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 186
  > I'm benchmarking a few different models to identify the best match for a few use cases I have, and threw a few Qwen3.5 in the mix (4b, 9b and 27b). I was not expecting the 4b to be as good as it is!
  > 
  > These results are on a Ollama running on a 7900XTX
  > 
  > |**Model**|**Fast**|**Main**|**Long**|**Overall**|
  > |:-|:-|:-|:-|:-|
  > |**devstral-small-2:24b**|0.97|1.00|0.99|0.99|
  > |**mistral-small3.2:24b**|0.99|0.98|0.99|0.99|
  > |**deepseek-r1:32b**|0.97|0.98|0.98|0.98|
  > |**qwen3.5:4b**|0.95|0.98|1.00|0.98|
  > |**glm-

### 95. [Llama Server issue running Gemma 4 26B A4B](https://www.reddit.com/r/LocalLLaMA/comments/1sb0h1i/llama_server_issue_running_gemma_4_26b_a4b/)
- **Subreddit:** r/LocalLLaMA | **Score:** 7 | **Comments:** 5 | **Date:** 2026-04-03 01:52 UTC
- **Author:** u/VampiroMedicado | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 170
  > When I try to run llama-server with Gemma 4 26B A4B model, the inference step displays this error:
  > 
  > While executing FilterExpression at line 18, column 34 in source:
  > ...if -%}↵            {%- if value['type'] | upper == 'STRING' -%}↵                ...
  >                                            ^
  > Error: Unknown (built-in) filter 'upper' for type Array
  > 
  > I'm doing something wrong?
  > 
  > I call this with:
  > 
  > llama-server --model MODEL_PATH -c 0 --jinja --fit on --no-mmap
  > 
  > Llama CLI works fine.

### 96. [Strix Halo settings for agentic tasks](https://www.reddit.com/r/LocalLLaMA/comments/1s1vayj/strix_halo_settings_for_agentic_tasks/)
- **Subreddit:** r/LocalLLaMA | **Score:** 5 | **Comments:** 6 | **Date:** 2026-03-23 22:09 UTC
- **Author:** u/Intelligent-Form6624 | **Query match:** "Nemotron Cascade"
- **Relevance score:** 170
  > Been running Claude Code using local models on the Strix Halo (Bosgame M5, 128GB). Mainly MoE such as Qwen3.5-35B-A3B (Bartowski Q6\_K\_L) and Nemotron-Cascade-2-30B-A3B (AesSedai Q5\_K\_M).
  > 
  > The use case isn’t actually coding. It’s more document understanding and modification. So thinking is desirable over instruct.
  > 
  > OS is Ubuntu 24.04. Using llama.cpp-server via latest ggml docker images (llamacpp:vulkan, llamacpp:rocm).
  > 
  > For whatever reason, Gemini 3.1 Pro assured me ROCm was the better engin

### 97. [Could we engineer a Get-Shit-Done Lite that would work well with models like Qwen3.5 35B A3B?](https://www.reddit.com/r/LocalLLaMA/comments/1s7cv46/could_we_engineer_a_getshitdone_lite_that_would/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 8 | **Date:** 2026-03-30 01:16 UTC
- **Author:** u/HockeyDadNinja | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 160
  > Has someone done this already? A simple spec driven design framework that helps them along and reduces complexity. I want to go to work and have my 2 x 4060 ti 16G yolo mode for me all day.

### 98. [My current LocalLLM project list](https://www.reddit.com/r/LocalLLaMA/comments/1s4i2f3/my_current_localllm_project_list/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 8 | **Date:** 2026-03-26 19:57 UTC
- **Author:** u/BigJay125 | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 160
  > Sharing some things I've been hacking on recently. Maybe some of you guys have gone after these too! 
  > 
  > My goal is to complete these projects entirely with local, organically farmed tokens.
  > 
  > **1. OpenTax** \- A containerized, isolated, fully local LLM tax preparation agent. Drop docs in, answer some questions, do my taxes. I've already had it estimate my 1040 a few times but it has made mistakes - tweaking to see how close I can get it. 
  > 
  >   **why:** local compute / privacy seems fun. i like not g

### 99. [After a week of trying many models for fiction writing, Gemma 4 26B A4B IT (Heretic) is the first one which feels actually capable.](https://www.reddit.com/r/LocalLLaMA/comments/1se197x/after_a_week_of_trying_many_models_for_fiction/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 8 | **Date:** 2026-04-06 14:58 UTC
- **Author:** u/AnOnlineHandle | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 160
  > In the very early days I was able to finetune a gen 1 llama base model on my own writing, but I wanted to avoid setting that all up again and was hoping that I could instruct a more modern model into writing what I want.
  > 
  > However every model which could fit on my GPU which I tried was a disappointment, even though they were widely praised as the best. Short contexts, frequent incoherency, not grasping the prompt, not grasping the subtleties of example text snippets, etc.
  > 
  > I was about to give up,

### 100. [What is the best local LLMs as of March 2026?](https://www.reddit.com/r/LocalLLaMA/comments/1rkppnl/what_is_the_best_local_llms_as_of_march_2026/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 24 | **Date:** 2026-03-04 16:17 UTC
- **Author:** u/Pejorativez | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 144
  > What is the all-around best local LLM for general uses cases like asking questions, reasoning, encyclopedia, writing text?
  > 
  > I'm currently using GLM-4.7-Flash 8.0 via Ollama, which is amazing. And currently downloading LFM2:24B. Looking forward to testing it. 
  > 
  > What would you say is the best local models, and why?

### 101. [Qwen3.5-35B-A3B-Claude-4.6-Opus-Uncensored-KL-UD-V2-GGUF + Bonus scripts](https://www.reddit.com/r/LocalLLaMA/comments/1sd99ez/qwen3535ba3bclaude46opusuncensoredkludv2gguf/)
- **Subreddit:** r/LocalLLaMA | **Score:** 6 | **Comments:** 4 | **Date:** 2026-04-05 17:17 UTC
- **Author:** u/EvilEnginer | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 140
  > Hello everyone. I fixed Qwen3.5 35B A3B (Claude Opus + uncensored merge) via KL divergence minimisation. I fixed attention, dense FFN, MoE experts, shared experts, and got *92% KL drop with working Arkanoid game in 2 prompts.*
  > 
  > **Here link:** [https://huggingface.co/LuffyTheFox/Qwen3.5-35B-A3B-Claude-4.6-Opus-Uncensored-KL-UD-V2-GGUF](https://huggingface.co/LuffyTheFox/Qwen3.5-35B-A3B-Claude-4.6-Opus-Uncensored-KL-UD-V2-GGUF) . Please read launch instructions on page for best experience.
  > 
  > I merg

### 102. [GLM 4.7 Flash 30B PRISM with web search is seriously impressive](https://www.reddit.com/r/LocalLLaMA/comments/1s4gjmb/glm_47_flash_30b_prism_with_web_search_is/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 7 | **Date:** 2026-03-26 19:00 UTC
- **Author:** u/Internal_Finding4501 | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 140
  > Got this running about 2 days ago and wow this thing has blown me away with how well it handles complex reasoning tasks compared to the Qwen lineup I was using before. What really stands out is how unrestricted it feels - I can dig into basically any research topic without hitting those annoying soft blocks
  > 
  >   
  > Sure the core knowledge base doesnt match up to something like 120B Derestricted but once you add web search RAG into the mix this 30B model actually outperforms most of what Ive tested. 

### 103. [Ok i think im done trying to make a lifelike agent..](https://www.reddit.com/r/LocalLLaMA/comments/1rse8ps/ok_i_think_im_done_trying_to_make_a_lifelike_agent/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 7 | **Date:** 2026-03-13 04:59 UTC
- **Author:** u/Myvzw_copyrightbot | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 140
  > (I'm not a bot, my username was apparently copyrighted or something and reddit changed it to this)
  > 
  > (Also i dont really use social media, hence this old ass account with no history)
  > 
  > I know these are just prediction models but damn the following interaction is by far the most eerie ive seen..
  > 
  > **A little context:** Out of morbid curiosity, I gave GLM 4.7-flash web search and had it generate a system prompt for the most life-like personality it could. I told it "You make you. Search the web and c

### 104. [Nemotron Cascade 2 on 6GB VRAM](https://www.reddit.com/r/LocalLLaMA/comments/1rz43hi/nemotron_cascade_2_on_6gb_vram/)
- **Subreddit:** r/LocalLLaMA | **Score:** 3 | **Comments:** 5 | **Date:** 2026-03-20 18:24 UTC
- **Author:** u/AppealSame4367 | **Query match:** "Nemotron Cascade"
- **Relevance score:** 130
  > Edit: context of 90k + still seems to run at least and -b / -ub of 512 -&gt; 300+ prefill tps -&gt; not sure about quality yet  
  > 
  > 
  > \-&gt; 4.750 GB VRAM  
  > \-&gt; 17.5   GB RAM
  > 
  >   
  > \- around 100 tps prefill  
  > \- 10-20 tps output at 6k context  
  > \- thinking is short, so it's still usable albeit low speed
  > 
  > \- intel 6 core  
  > \- rtx2060, laptop, 6gb vram  
  > \- 32GB RAM
  > 
  > 53/53 layers where offloaded to GPU.
  > 
  > Cool if you wanna have a smart llm on low spec hardware. Qwen3.5 9B/35B think too long to be usa

### 105. [Thoughts on recent small (under 20B) models](https://www.reddit.com/r/LocalLLaMA/comments/1ppstef/thoughts_on_recent_small_under_20b_models/)
- **Subreddit:** r/LocalLLaMA | **Score:** 71 | **Comments:** 25 | **Date:** 2025-12-18 14:55 UTC
- **Author:** u/surubel | **Query match:** "Nemotron Cascade"
- **Relevance score:** 121
  > Recently we're been graced with quite a few small (under 20B) models and I've tried most of them.
  > 
  > The initial benchmarks seemed a bit too good to be true, but I've tried them regardless. 
  > 
  > * RNJ-1: this one had probably the most "honest" benchmark results. About as good as QWEN3 8B, which seems fair from my limited usage. 
  > * GLM 4.6v Flash: even after the latest llama.cpp update and Unsloth quantization I still have mixed feelings. Can't get it to think in English, but produces decent results. 

### 106. [Using SCHED_RR on all cores gives a decent 25%-40% boost in token generation with CPU offloading](https://www.reddit.com/r/LocalLLaMA/comments/1s5879c/using_sched_rr_on_all_cores_gives_a_decent_2540/)
- **Subreddit:** r/LocalLLaMA | **Score:** 4 | **Comments:** 4 | **Date:** 2026-03-27 15:59 UTC
- **Author:** u/XLIICXX | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 120
  > I always assumed that limiting the threads to half the number of cores/threads would give the best generation t/s with CPU offloading but apparently using the `SCHED_RR` (realtime-ish) scheduler on all cores/threads gives a decent 25% boost compared to half the cores on the default `SCHED_NORMAL` scheduler:
  > 
  > &amp;nbsp;  
  > 
  > | Threads | SCHED_NORMAL | SCHED_RR | Diff   |
  > |--------:|-------------:|---------:|-------:|
  > |         |              |          | - ~ 8% |
  > |       8 |          ~28 |      ~23

### 107. [What models fit in 16gb vram for local agentic coding?](https://www.reddit.com/r/LocalLLaMA/comments/1s9xduu/what_models_fit_in_16gb_vram_for_local_agentic/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 6 | **Date:** 2026-04-01 21:04 UTC
- **Author:** u/Witty_Mycologist_995 | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 120
  > Currently using glm 4.7 flash, it’s very meh 
  > 
  > Heard omnicoder or Crow 9b are good, are they any better?
  > 
  > Or Qwen3.5 27b?

### 108. [Choosing between templates for local coding](https://www.reddit.com/r/LocalLLaMA/comments/1rq1eey/choosing_between_templates_for_local_coding/)
- **Subreddit:** r/LocalLLaMA | **Score:** 2 | **Comments:** 5 | **Date:** 2026-03-10 16:20 UTC
- **Author:** u/MattimaxForce | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 120
  > Hi everyone! Can anyone help me decide which model would be best for doing agentic coding locally?
  > 
  > I'm undecided between these here:
  > 
  > [https://huggingface.co/Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
  > 
  > [https://huggingface.co/zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)
  > 
  > [https://huggingface.co/Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next)
  > 
  >   
  > The fact that I want the best model possible but also the lightest possible.
  > 
  > Any 

### 109. [What's a good small local model, if any, for local APPLY / EDIT operations in code editors while using SOTA for planning?](https://www.reddit.com/r/LocalLLaMA/comments/1s6x0dt/whats_a_good_small_local_model_if_any_for_local/)
- **Subreddit:** r/LocalLLaMA | **Score:** 1 | **Comments:** 5 | **Date:** 2026-03-29 14:46 UTC
- **Author:** u/ea_man | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 110
  > The idea is to use a SOTA model for planning code with a prompt that generates base architecture and then most of the code, then use a local LM to manage file creation, EDIT, APPLY of the code now in the context. The purpose is reducing usage of expensive on-line models delegating the *supposedly simple* EDIT / APPLY to local models.
  > 
  > Now I'm asking first if this is feasible, if LocalLM can be trusted to properly apply code without messing up often.  
  > Then what models and with what parameters wo

### 110. [DGX Spark + Qwen3.5-35B-A3B: MXFP4 produces Chinese character artifacts — anyone else seeing this?](https://www.reddit.com/r/LocalLLaMA/comments/1s6osp2/dgx_spark_qwen3535ba3b_mxfp4_produces_chinese/)
- **Subreddit:** r/LocalLLaMA | **Score:** 1 | **Comments:** 5 | **Date:** 2026-03-29 07:35 UTC
- **Author:** u/kaltinator | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 110
  > \## Setup
  > 
  > 
  > 
  > \- \*\*Hardware:\*\* NVIDIA DGX Spark (GB10, SM121 Blackwell, 128 GB unified RAM)
  > 
  > \- \*\*OS:\*\* Ubuntu 24.04.4 LTS (aarch64)
  > 
  > \- \*\*CUDA:\*\* 13.0
  > 
  > \- \*\*Model:\*\* Qwen3.5-35B-A3B (BF16 checkpoint, MXFP4 online quantization)
  > 
  > \- \*\*Inference:\*\* vLLM 0.17.1+cu130 with \[namake-taro/vllm-custom\](https://github.com/namake-taro/vllm-custom) MXFP4 patches applied
  > 
  > \- \*\*Use case:\*\* RAG document processing pipeline (RAGFlow) — Vision descriptions, keyword extraction, question 

### 111. [I made a small app to use Copilot Chat with LM Studio instead of Ollama.](https://www.reddit.com/r/LocalLLaMA/comments/1se0c1r/i_made_a_small_app_to_use_copilot_chat_with_lm/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 5 | **Date:** 2026-04-06 14:23 UTC
- **Author:** u/x0wl | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 100
  > I found out that VSCode's built-in Copilot Chat can work with local models, but requires Ollama. I don't use Ollama because I like LM Studio.
  > 
  > I looked at its source code and found that is only uses Ollama-specific APIs to discover available models, but then it just relies on OpenAI-compatible endpoints. So I implemented a small server that emulates enough of Ollama's API for Copilot to work by making use of LM Studio's REST API.
  > 
  > The GitHub Link is here: [https://github.com/x0wllaar/copilot-oll

### 112. [Is the jump from 48GB to 64GB unified memory worth it given where local models are headed?](https://www.reddit.com/r/LocalLLaMA/comments/1saltth/is_the_jump_from_48gb_to_64gb_unified_memory/)
- **Subreddit:** r/LocalLLaMA | **Score:** 2 | **Comments:** 4 | **Date:** 2026-04-02 16:15 UTC
- **Author:** u/mrr_reddit | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 100
  > Context: Prices below are Apple Education (US). Coming from a 16” M4 Pro 48GB that I sold to a close friend but I realized portability matters more to me than I thought as a SWE, so going 14”.
  > 
  > My local AI stack: LM Studio with multiple MCP servers. Day-to-day models are Qwen3.5 35B-A3B, Qwen3.5 27B, and GPT-OSS 20B
  > 
  > The decision:
  > 
  > 	∙	$2,409 — M5 Pro binned (15-core CPU, 16-core GPU) — 48GB
  > 
  > 	∙	$2,779 — M5 Pro unbinned (18-core CPU, 20-core GPU) — 64GB
  > 
  > Bandwidth is identical at 307 GB/s on both

### 113. [Nemotron-Cascade-2 10GB MAC ONLY Scores 88% on MMLU.](https://www.reddit.com/r/LocalLLaMA/comments/1s05az7/nemotroncascade2_10gb_mac_only_scores_88_on_mmlu/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 5 | **Date:** 2026-03-21 22:54 UTC
- **Author:** u/HealthyCommunicat | **Query match:** "Nemotron Cascade"
- **Relevance score:** 100
  > Even if someone did happen to make an MLX quant of this size (10gb) it would be completely incoherent at 2bit. 
  > 
  > https://huggingface.co/JANGQ-AI/Nemotron-Cascade-2-30B-A3B-JANG\_2L
  > 
  > Mistral 4 30-40gb and a 60-70gb version coming out later today.

### 114. [I Ran Kotlin HumanEval on 11 Local LLMs. An 8GB Model Beat Several 30B Models](https://www.reddit.com/r/LocalLLaMA/comments/1ru4vlu/i_ran_kotlin_humaneval_on_11_local_llms_an_8gb/)
- **Subreddit:** r/LocalLLaMA | **Score:** 4 | **Comments:** 3 | **Date:** 2026-03-15 04:37 UTC
- **Author:** u/codeforlyfe | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 100
  > TLDR: I ran JetBrains' Kotlin HumanEval on 11 local models, including some small ones that fit on a 16 GB VRAM GPU. Here are the results.
  > 
  > * pass@1 / pass@3:
  >    * GPT-OSS 20B: 85% / 95%
  >    * Qwen3.5-35B-a3b: 77% / 86%
  >    * EssentialAI RNJ-1: 75% / 81% ← 8.8 GB file size
  >    * Seed-OSS-36B: 74% / 81%
  >    * GLM 4.7 Flash: 68% / 78%
  > 
  > A few things I found interesting:
  > 
  > * GPT-OSS 20B still dominates at 85% pass@1, despite being one of the smaller models by file size (12 GB)
  > * EssentialAI RNJ-1 at 8.8 G

### 115. [[P] Gemma 4 running on NVIDIA B200 and AMD MI355X from the same inference stack, 15% throughput gain over vLLM on Blackwell](https://www.reddit.com/r/MachineLearning/comments/1saot07/p_gemma_4_running_on_nvidia_b200_and_amd_mi355x/)
- **Subreddit:** r/MachineLearning | **Score:** 6 | **Comments:** 2 | **Date:** 2026-04-02 18:01 UTC
- **Author:** u/carolinedfrasca | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 100
  > Google DeepMind dropped Gemma 4 today:
  > 
  > **Gemma 4 31B:** dense, 256K context, redesigned architecture targeting efficiency and long-context quality
  > 
  > **Gemma 4 26B A4B:** MoE, 26B total / 4B active per forward pass, 256K context
  > 
  > Both are natively multimodal (text, image, video, dynamic resolution).
  > 
  > We got both running on MAX on launch day across NVIDIA B200 and AMD MI355X from the same stack. On B200 we're seeing 15% higher output throughput vs. vLLM (happy to share more on methodology if usefu

### 116. [Whats your strategy for long conversations with local models?](https://www.reddit.com/r/LocalLLaMA/comments/1rjv92p/whats_your_strategy_for_long_conversations_with/)
- **Subreddit:** r/LocalLLaMA | **Score:** 1 | **Comments:** 15 | **Date:** 2026-03-03 17:19 UTC
- **Author:** u/Di_Vante | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 93
  > I've been testing a few different agents locally and sometimes it gets really frustrating. I feel like I need to do some sort of reboot every few sessions, otherwise the quality deterioration is intense.
  > 
  > My goal is to start with a "personal assistant" that handles simple tasks, and then build a few other agents that run on CPU (don't care about speed on those)
  > 
  > Anyone having good results that don't require having to "clear up" the chat every session or so? 
  > 
  > I'm mostly running Ollama on a 7900x

### 117. [Opencode + Local Models + Apple MLX = ??](https://www.reddit.com/r/LocalLLaMA/comments/1s498dm/opencode_local_models_apple_mlx/)
- **Subreddit:** r/LocalLLaMA | **Score:** 1 | **Comments:** 4 | **Date:** 2026-03-26 14:37 UTC
- **Author:** u/agrof | **Query match:** "Nemotron Cascade"
- **Relevance score:** 90
  > I have experience using llama.cpp on Windows/Linux with 8GB NVIDIA card (384 GB/s bandwidth) and offloading to CPU to run MoE models. I typically use the Unsloth GGUF models and it works relatively well.
  > 
  > I have recently started playing with local models on a Macbook M1 Max 64GB, and if feels like a downgrade in terms of support. llama.cpp vulkan doesn't run as fast as MLX and there are less MLX models in huggingface in comparison to GGUF.
  > 
  > I have tried mlx-lm, oMLX, vMLX with various degrees of

### 118. [Any other LLMs are as good as this one ?](https://www.reddit.com/r/LocalLLaMA/comments/1rvhwwj/any_other_llms_are_as_good_as_this_one/)
- **Subreddit:** r/LocalLLaMA | **Score:** 1 | **Comments:** 4 | **Date:** 2026-03-16 18:33 UTC
- **Author:** u/Mr_Universal000 | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 90
  > Hi,
  > 
  > so I've tried so many different models, including heretic/abliterated versions but non of them were as good as "Dolphin Mistral GLM 4.7 Flash 24B Venice Edition Thinking Uncensored I1", the output is really good, creativity is great.
  > 
  > but I'm looking for a different LLM with a different Arch other than llama.
  > 
  > can you one recommend other LLMs that fit in a 3060 12gb ?  
  >   
  > i use it mainly for writing and coming up with ideas and concepts.
  > 
  > Thanks in advance.

### 119. [AdamBench v1.1 - a benchmark for local coding models. New models added (eg. Gemma4)](https://www.reddit.com/r/LocalLLaMA/comments/1seeo2z/adambench_v11_a_benchmark_for_local_coding_models/)
- **Subreddit:** r/LocalLLaMA | **Score:** 2 | **Comments:** 3 | **Date:** 2026-04-06 23:05 UTC
- **Author:** u/Real_Ebb_7417 | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 80
  > Some time ago, I published my benchmark of local coding models AdamBench (here: [https://github.com/tabupl/AdamBench](https://github.com/tabupl/AdamBench)). The purpose of this benchmark is to test local models at agentic coding task on my specific hardware (RTX5080 + 64Gb RAM). And now, I wanted to add a couple models before switching to RTX5090 (I'll do v2 on it, automated and more immune to random luck). Specifically I added:
  > 
  > * All Gemma4 versions -&gt; Very good scores, but worse than corre

### 120. [Gemma4 issue with winogrande bench](https://www.reddit.com/r/LocalLLaMA/comments/1sc5e10/gemma4_issue_with_winogrande_bench/)
- **Subreddit:** r/LocalLLaMA | **Score:** 4 | **Comments:** 2 | **Date:** 2026-04-04 10:04 UTC
- **Author:** u/qdwang | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 80
  > gemma-4-26B-A4B-it-Q4\_K\_M can only get around 50% acc on winogrande-debiased-eval.csv with llama-perplexity.
  > 
  > Meanwhile qwen3.5-35B-A3B-IQ4\_NL can get about 75%+ acc.
  > 
  > However, in real-world tasks, the Gemma 4 model performs very well.
  > 
  > Why does this discrepancy occur?

### 121. [Gemma 4 26B-A4B on Apple M1 Max is very fast](https://www.reddit.com/r/LocalLLaMA/comments/1sbxsyn/gemma_4_26ba4b_on_apple_m1_max_is_very_fast/)
- **Subreddit:** r/LocalLLaMA | **Score:** 2 | **Comments:** 3 | **Date:** 2026-04-04 02:55 UTC
- **Author:** u/Beamsters | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 80
  > Gemma 4 26B-A4B quantized at Q5K\_S running on Apple M1 Max 32GB
  > 
  > Using LMStudio, Unsloth Q5K\_S  Context 65536 use around 22GBish memory (Metal llama 2.11.0)
  > 
  > On average Tok/s = 50.x
  > 
  > On the other hand Gemma 4 31B (Q4K\_S) is quite slow on average Tok/s = 10-11

### 122. [Gemma-4-26B-A4B on RX 6600 / 32gb ddr4 / mid i5 cpu: 12-15 tps, nice..](https://www.reddit.com/r/LocalLLaMA/comments/1saw6ur/gemma426ba4b_on_rx_6600_32gb_ddr4_mid_i5_cpu_1215/)
- **Subreddit:** r/LocalLLaMA | **Score:** 2 | **Comments:** 3 | **Date:** 2026-04-02 22:42 UTC
- **Author:** u/mr_happy_nice | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 80
  > quick test Unsloth's Instruct MXFP4 quant on LM Studio / PopOS-Ubuntu  
  > this is on the Vulkan EP

### 123. [Nemotro-Cascade 2 Uncensored (Mac Only) 10gb - 66% MMLU / 18gb - 82% MMLU](https://www.reddit.com/r/LocalLLaMA/comments/1s0d4lp/nemotrocascade_2_uncensored_mac_only_10gb_66_mmlu/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 4 | **Date:** 2026-03-22 05:12 UTC
- **Author:** u/HealthyCommunicat | **Query match:** "Nemotron Cascade"
- **Relevance score:** 80
  > Usually the MMLU scores go a little higher after ablation but I need to look into what went differently cuz the scores went down for both quants.
  > 
  > [https://huggingface.co/dealignai/Nemotron-Cascade-2-30B-A3B-JANG\_4M-CRACK](https://huggingface.co/dealignai/Nemotron-Cascade-2-30B-A3B-JANG_4M-CRACK)
  > 
  > Architecture	Nemotron Cascade 2 — 30B total, \~3B active, 3 layer types
  > 
  > Quantization	JANG\_4M (8/4-bit mixed, 4.1 avg) — 17 GB
  > 
  > HarmBench	99.4% (318/320)
  > 
  > MMLU	82.7% (172/208 with thinking)
  > 
  > Speed	\~

### 124. [local llm inference on M4 Max vs M5 Max](https://www.reddit.com/r/LocalLLaMA/comments/1s771lm/local_llm_inference_on_m4_max_vs_m5_max/)
- **Subreddit:** r/LocalLLaMA | **Score:** 4 | **Comments:** 2 | **Date:** 2026-03-29 21:08 UTC
- **Author:** u/purealgo | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 80
  > I just picked up an M5 Max MacBook Pro and am planning to replace my M4 Max with it, so I ran my open-source MLX inference benchmark across both machines to see what the upgrade actually looks like in numbers. Both are the 128GB, 40-core GPU configuration. Each model ran multiple timed iterations against the same prompt capped at 512 tokens, so the averages are stable.
  > 
  > The M5 Max pulls ahead across all three models, with the most gains in prompt processing (17% faster on GLM-4.7-Flash, 38% on Q

### 125. [Key Highlights of NVIDIA’s New Model: Nemotron-Cascade-8B](https://www.reddit.com/r/LocalLLaMA/comments/1po3ln2/key_highlights_of_nvidias_new_model/)
- **Subreddit:** r/LocalLLaMA | **Score:** 66 | **Comments:** 4 | **Date:** 2025-12-16 14:40 UTC
- **Author:** u/Dear-Success-1441 | **Query match:** "Nemotron Cascade"
- **Relevance score:** 74
  > **\[1\] General-Purpose Reinforcement-Learned Model**
  > 
  > * Trained through a sequential and domain-wise reinforcement learning pipeline built on top of a base Qwen3-8B model, enhancing performance across diverse task domains
  > 
  > **\[2\] Dual Reasoning &amp; Instruction Modes**
  > 
  > * Supports both *thinking* (reasoning) and *instruct* (non-reasoning) modes, allowing flexible use cases within the same model architecture.
  > 
  > **\[3\] Strong Benchmark Performance**
  > 
  > * Achieves competitive results on knowledge,

### 126. [Qwen 3.5 35B on LocalAI (Strix Halo): Vulkan / ROCm](https://www.reddit.com/r/LocalLLaMA/comments/1sfor4x/qwen_35_35b_on_localai_strix_halo_vulkan_rocm/)
- **Subreddit:** r/LocalLLaMA | **Score:** 5 | **Comments:** 1 | **Date:** 2026-04-08 10:16 UTC
- **Author:** u/pipould | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 70
  > # Qwen 3.5 35B on LocalAI: Vulkan vs ROCm
  > 
  > Hey everyone! 👋
  > 
  > Just finished running a bunch of benchmarks on the new Qwen 3.5 35B models using LocalAI and figured I'd share the results. I was curious how **Vulkan** and **ROCm** backends stack up against each other for these two different quant/source variants.
  > 
  > ---
  > 
  > Two model variants, each on both Vulkan and ROCm:
  > 
  > | Model | Type | Quant | Source |
  > |---|---|---|---|
  > | mudler/Qwen3.5-35B-A3B-APEX-GGUF:Qwen3.5-35B-A3B-APEX-I-Quality.gguf | MoE (3B 

### 127. [3090 Gemma4 50% Util? not laoding all layers to vram?](https://www.reddit.com/r/LocalLLaMA/comments/1se49u2/3090_gemma4_50_util_not_laoding_all_layers_to_vram/)
- **Subreddit:** r/LocalLLaMA | **Score:** 3 | **Comments:** 2 | **Date:** 2026-04-06 16:47 UTC
- **Author:** u/veryhasselglad | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 70
  > model: google/gemma-4-26b-a4b from lmstudio (running via lms)

### 128. [Local LLM inference on M4 Max vs M5 Max](https://www.reddit.com/r/LocalLLaMA/comments/1s97chj/local_llm_inference_on_m4_max_vs_m5_max/)
- **Subreddit:** r/LocalLLaMA | **Score:** 5 | **Comments:** 1 | **Date:** 2026-04-01 02:04 UTC
- **Author:** u/purealgo | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 70
  > I picked up an M5 Max MacBook Pro and wanted to see what the upgrade looks like in practice, so I ran the same MLX inference benchmark on it and on my M4 Max. Both machines are the 16 inch, 128GB, 40-core GPU configuration.
  > 
  > The table below uses the latest comparable runs with a short prompt and output capped at 512 tokens. Prompt processing on the M5 Max improved by about 14% to 42%, while generation throughput improved by about 14% to 17%.
  > 
  > | Model | M4 Max Gen (tok/s) | M5 Max Gen (tok/s) | M

### 129. [Qwen3.5 9B and 27B gibberish since first start.](https://www.reddit.com/r/LocalLLaMA/comments/1rln3qc/qwen35_9b_and_27b_gibberish_since_first_start/)
- **Subreddit:** r/LocalLLaMA | **Score:** 1 | **Comments:** 11 | **Date:** 2026-03-05 17:07 UTC
- **Author:** u/jpbras | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 69
  > Computer 1: Windows 11, Dell Pro 14 Plus, 32GB RAM, llama.cpp b8204 release.  
  > Both models unsloth, downloaded on 3rd March, both using the recommended parameters.  
  > Qwen3.5-9B-Q6\_K and Qwen3.5-27B-Q4\_K\_M, The output is all gibberish.  
  > All previous models installed like GLM-4.7-Flash, Qwen3-Coder-30B-AB and Qwen2.5 works.
  > 
  > Computer 2: Linux Fedora 43, old ASUS 16GB no GPU, Qwen3.5-9B-Q4\_K\_M.gguf works, 2.5t/s but works.
  > 
  > What I've tried:
  > 
  > `llama-server.exe --ctx-size 16384 --temp 0.6 --top

### 130. [Best Local Model For Python and QT Quick Coding](https://www.reddit.com/r/LocalLLaMA/comments/1rhzknn/best_local_model_for_python_and_qt_quick_coding/)
- **Subreddit:** r/LocalLLaMA | **Score:** 1 | **Comments:** 11 | **Date:** 2026-03-01 15:08 UTC
- **Author:** u/wisepal_app | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 69
  > I mainly develop desktop software with Pyside6 and QML for my specific domain. i don't want my data collected by closed ai corps. So i decided to go full local almost 4 months ago. I bought a Hp Zbook laptop with i7-12800h, 96 gb ddr5 4800 mhz ram, a4500 rtx 16 gb vram and windows 10 pro.
  > 
  > Thanks to the community in this sub i learned lots of things. Started from Lm Studio and ended up with llama.cpp with lots of flag combinations :)
  > 
  > Then i tried agentic coding with opencode and lastly with Pi 

### 131. [My experience with Qwen3.5-35B-A3B-4bit on macbook pro m3 max 36 gb](https://www.reddit.com/r/LocalLLaMA/comments/1scfdqi/my_experience_with_qwen3535ba3b4bit_on_macbook/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 3 | **Date:** 2026-04-04 17:34 UTC
- **Author:** u/Sea-Emu2600 | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 60
  > First of all I am pretty new to this local llama world. I spent a few days trying a few things, mainly ollama and omlx with opencode.
  > 
  > Right now I am trying to create a python project with deepagents. I am running Qwen3.5-35B-A3B-4bit  using oMLX.
  > 
  > Deepagents has some skills that shows how to to use the library.  
  > So far the experience is not being pleasant. While the setup works and token generation looks fast enough (getting 47t/s on avg) what I see is that the model spends too much time on th

### 132. [Offline-first MDN Web Docs RAG-MCP server](https://www.reddit.com/r/LocalLLaMA/comments/1s9tnev/offlinefirst_mdn_web_docs_ragmcp_server/)
- **Subreddit:** r/LocalLLaMA | **Score:** 2 | **Comments:** 2 | **Date:** 2026-04-01 18:50 UTC
- **Author:** u/dpswt | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 60
  > Hi.
  > 
  > While tinkering with RAG ideas I've thoroughly processed the entire MDN Web Docs original content, pre-ingested it into LanceDB, uploaded the 50k+ rows [dataset](https://huggingface.co/datasets/deepsweet/mdn) to HuggingFace, and published a [RAG-MCP server](https://github.com/deepsweet/mdn) ready for semantic search with hybrid vector (1024-d) and full‑text (BM25) retrieval.
  > 
  > A screenshot is worth a thousand words, see both repositories for more details.

### 133. [MCPHub's Smart Routing feature - actually beneficial or waste of time?](https://www.reddit.com/r/LocalLLaMA/comments/1s58a5x/mcphubs_smart_routing_feature_actually_beneficial/)
- **Subreddit:** r/LocalLLaMA | **Score:** 4 | **Comments:** 1 | **Date:** 2026-03-27 16:01 UTC
- **Author:** u/moderately-extremist | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 60
  > I'm wondering what people's experiences are with the Smart Routing feature on MCPHub, if it was actually helpful.  I'm using Qwen3.5-35b-a3b as my main model and it seems like it already decides what tool to call.   My concern is the steps to go through the Smart Routing is just going to introduce a delay without any real benefit.  But maybe it's actually after than letting the main model decide?  I'm thinking of using qwen3-embedding-4b for the Smart Routing model.

### 134. [We made a system for autonomous agents to speak to each other without a human input needed](https://www.reddit.com/r/LocalLLaMA/comments/1s3juge/we_made_a_system_for_autonomous_agents_to_speak/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 3 | **Date:** 2026-03-25 18:59 UTC
- **Author:** u/Helpful-Series132 | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 60
  > [https://github.com/StarpowerTechnology/Starpower/blob/main/Demos/starpower-autonomy-groupchat.ipynb](https://github.com/StarpowerTechnology/Starpower/blob/main/Demos/starpower-autonomy-groupchat.ipynb)
  > 
  > This is a simple setup to be able to speak to a group of agents with a human groupchat feel .. asynchronous, not  a instant reply, pretty chill if you just like to observe ai behavior or talk to them, but you can just allow them to talk to themselves if you want. Speaking you’re self is optional

### 135. [Any advice to upgrade my current setup or it's too soon with current prices?](https://www.reddit.com/r/LocalLLaMA/comments/1rowchi/any_advice_to_upgrade_my_current_setup_or_its_too/)
- **Subreddit:** r/LocalLLaMA | **Score:** 2 | **Comments:** 9 | **Date:** 2026-03-09 10:16 UTC
- **Author:** u/soyalemujica | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 60
  > Basically;
  > 9800x3D
  > Nvidia 5060ti 16gb VRAM
  > 64gb ddr5 6400mts
  > 1000w PSU 
  > 
  > I am using Qwen3-Coder in 4bit at 26t/s
  > 27B at Q3SS at 24t/s (can't exceed 4k context)
  > 27b at 4q at 11t/s (even less context)
  > 35B A3B 4bit at 56t/s
  > GLM 4.7 Flash at 26t/s
  > 
  > Just asking if there's anything I can get le upgrade for better models and workload.

### 136. [Local LLMs are slow, I have too many things to try, and I hate chat UIs, so I built an async task board where agents work in parallel while I do other things](https://www.reddit.com/r/LocalLLaMA/comments/1rhalir/local_llms_are_slow_i_have_too_many_things_to_try/)
- **Subreddit:** r/LocalLLaMA | **Score:** 10 | **Comments:** 4 | **Date:** 2026-02-28 18:50 UTC
- **Author:** u/BadBoy17Ge | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 54
  > &gt;quick context on why I built this my PC is slow for local LLMs so I'd kick off a task and just... wait. meanwhile I have like 10 other things I want to try. so instead of one chat I built a board where everything queues up and runs while I get on with other stuff. the parallel agents thing came from that same frustration  stop babysitting one chat, let them all run
  > 
  > # Clara Companion: connect your machine to your AI
  > 
  > You run a lightweight companion on any machine (PC, server, whatever). It c

### 137. [Question About Cmake command](https://www.reddit.com/r/LocalLLaMA/comments/1salfek/question_about_cmake_command/)
- **Subreddit:** r/LocalLLaMA | **Score:** 1 | **Comments:** 2 | **Date:** 2026-04-02 16:00 UTC
- **Author:** u/JwustGiveMeAName | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 50
  > So i followed the ggml-org github page and used the git clone repo method to set up llama. i have the nvidia toolkit from the nvidia website installed and followed the cuda method and ran the following commands:
  > 
  > cmake -B build -DGGML\_CUDA=ON
  > 
  > cmake --build build --config Release
  > 
  > cd build/bin
  > 
  > ./llama-server -hf ggml-org/Qwen3.5-35B-A3B-GGUF:Q8\_0
  > 
  > now while llama does successfully output to a local host, if i close the terminal window, i need to rerun all the commands starting from cmake to g

### 138. [Local Video-to-Text Pipeline on Apple Silicon (Whisper + Qwen2.5-VL) - Optimized for 8GB/16GB RAM](https://www.reddit.com/r/LocalLLaMA/comments/1p82u5k/local_videototext_pipeline_on_apple_silicon/)
- **Subreddit:** r/LocalLLaMA | **Score:** 14 | **Comments:** 18 | **Date:** 2025-11-27 13:57 UTC
- **Author:** u/Longjumping-Elk-7756 | **Query match:** "Rapid-MLX"
- **Relevance score:** 50
  > Hi everyone,
  > 
  > I wanted to share a Python script I built to convert video files into a rich text context suitable for RAG (Retrieval Augmented Generation).
  > 
  > My goal was to process videos locally on my Mac without sending data to the cloud, and crucially, to make it run on machines with limited RAM (like base M1/M2/M3 Airs) without crashing.
  > 
  > **🚀 How it works (The "Smart" Pipeline):**
  > 
  > 1. **Scene Detection (OpenCV):** Instead of analyzing every frame (which is slow and redundant), the script detec

### 139. [What is the best model for coding in local 8-14b parameters](https://www.reddit.com/r/LocalLLaMA/comments/1psib5j/what_is_the_best_model_for_coding_in_local_814b/)
- **Subreddit:** r/LocalLLaMA | **Score:** 7 | **Comments:** 21 | **Date:** 2025-12-21 21:59 UTC
- **Author:** u/nicklazimbana | **Query match:** "Nemotron Cascade"
- **Relevance score:** 49
  > I saw nvidia nemotron cascade etc. Results but im not sure they are working really well. Im not looking for sonnet grade model. I plan to fine-tune it for cyber security tasks. And also i have 300 dollar google cloud credit but i cant use it in gpu. Can i finetune with gpu?

### 140. [Nemotron-Cascade 8B/14B from NVIDIA (Qwen3 finetunes)](https://www.reddit.com/r/LocalLLaMA/comments/1pnxxnq/nemotroncascade_8b14b_from_nvidia_qwen3_finetunes/)
- **Subreddit:** r/LocalLLaMA | **Score:** 31 | **Comments:** 5 | **Date:** 2025-12-16 09:42 UTC
- **Author:** u/jacek2023 | **Query match:** "Nemotron Cascade"
- **Relevance score:** 41
  > "powerful general-purpose model trained through sequential and domain-wise reinforcement learning"
  > 
  > # [](https://huggingface.co/nvidia/Nemotron-Cascade-8B#results)Results
  > 
  > * We evaluate our model against competitive reasoning models on a diverse set of benchmarks, covering general-knowledge reasoning, alignment and instruction following, mathematical reasoning, competitive programming, software engineering, and tool-use proficiency.
  > * For Nemotron-Cascade models, we use a maximum generation leng

### 141. [Advice | Ask | Be Carefull With Qwen 3.5 Vision Configuration LLama Server](https://www.reddit.com/r/LocalLLaMA/comments/1sd2zer/advice_ask_be_carefull_with_qwen_35_vision/)
- **Subreddit:** r/LocalLLaMA | **Score:** 2 | **Comments:** 1 | **Date:** 2026-04-05 12:58 UTC
- **Author:** u/Excellent_Call_5954 | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 40
  > Hi guys,
  > 
  > If you have trouble with image processing to catch small detail find sweet spot for this parameter on Llama Server:  
  > "--image-min-tokens", "1024",
  > 
  > I realized when I set this and try to increase model start to catch small details better.
  > 
  >   
  > Also I am using ik llama with Qwen3.5-35B-A3B-UD-Q4\_K\_XL.gguf with 131K context size, and :
  > 
  > "-ngl", "99",
  > 
  > "--jinja",
  > 
  > "-fa", "1",
  > 
  > "-b", "16384",
  > 
  > "-ub", "16384",
  > 
  >   
  > I am trying on RTX A6000( I know it's powerfull but since concurrency and hi

### 142. [Vllm gemma4 26b a4b it-nvfp4 run success](https://www.reddit.com/r/LocalLLaMA/comments/1sdmjgc/vllm_gemma4_26b_a4b_itnvfp4_run_success/)
- **Subreddit:** r/LocalLLaMA | **Score:** 4 | **Comments:** 0 | **Date:** 2026-04-06 02:31 UTC
- **Author:** u/NovelAdorable7033 | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 40
  > \#!/usr/bin/env bash
  > 
  > set -euo pipefail
  > 
  > BASE\_DIR="/mnt/d/AI/docker-gemma4"
  > 
  > PATCH\_DIR="$BASE\_DIR/nvfp4\_patch"
  > 
  > BUILD\_DIR="$BASE\_DIR/build"
  > 
  > HF\_CACHE\_DIR="$BASE\_DIR/hf-cache"
  > 
  > LOG\_DIR="$BASE\_DIR/logs"
  > 
  > PATCH\_FILE="$PATCH\_DIR/gemma4\_patched.py"
  > 
  > DOCKERFILE\_PATH="$BUILD\_DIR/Dockerfile"
  > 
  > BASE\_IMAGE="vllm/vllm-openai:gemma4"
  > 
  > PATCHED\_IMAGE="vllm-gemma4-nvfp4-patched"
  > 
  > CONTAINER\_NAME="vllm-gemma4-nvfp4"
  > 
  > MODEL\_ID="bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4"
  > 
  > SERVED\_MODEL\_NAME="

### 143. [Claude code + LMstudio](https://www.reddit.com/r/LocalLLaMA/comments/1sew6h8/claude_code_lmstudio/)
- **Subreddit:** r/LocalLLaMA | **Score:** 2 | **Comments:** 1 | **Date:** 2026-04-07 13:35 UTC
- **Author:** u/Mr_Universal000 | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 40
  > Hi everyone,
  > 
  > I just have a question in regards to how to use the leaked claude code / or an improved version of it, bear in mind that I'm not tech savvy at all or understand all the little things about AI.
  > I have LMstudio, I download models there that fit my PC specs, and run it.
  > 
  > My question is I would like to use the leaked claude code, but I have no clue how to connect the models I have in LM into it.
  > Such as qwen or GLM 4.7 flash, etc.
  > 
  > A guide or step by step would be appreciated.
  > 
  > Thanks 

### 144. [Agentic Coding MoE Models for 10GB VRAM Setup with CPU Offloading?](https://www.reddit.com/r/LocalLLaMA/comments/1rjrfzg/agentic_coding_moe_models_for_10gb_vram_setup/)
- **Subreddit:** r/LocalLLaMA | **Score:** 1 | **Comments:** 6 | **Date:** 2026-03-03 14:56 UTC
- **Author:** u/DK_Tech | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 39
  > Current setup: 7800x3d, 32GB DDR5 6000MHz, RTX 3080 10GB
  > 
  > Mainly looking at Qwen3-Coder-30B-A3B-Instruct and GLM-4.7-Flash 
  > 
  > Would use the Q4\_K\_M quant splitting 50/50 b/w VRAM and RAM.
  > 
  > Any other options to consider? My use case is to have an agentic setup working with something like a ralph loop to continue iterating overtime.

### 145. [Are Image-Text-to-Text models becoming the next big AI?](https://www.reddit.com/r/LocalLLaMA/comments/1obqqdi/are_imagetexttotext_models_becoming_the_next_big/)
- **Subreddit:** r/LocalLLaMA | **Score:** 11 | **Comments:** 11 | **Date:** 2025-10-20 19:11 UTC
- **Author:** u/Full_Piano_3448 | **Query match:** "Rapid-MLX"
- **Relevance score:** 33
  > I’ve been checking the trending models lately and it’s crazy how many of them are Image-Text-to-Text. Out of the top 7 right now, 5 fall in that category (PaddleOCR-VL, DeepSeek-OCR, Nanonets-OCR2-3B, Qwen3-VL, etc). DeepSeek even dropped their own model today.
  > 
  > Personally, I have been playing around with a few of them (OCR used to be such a pain earlier, imo) and the jump in quality is wild. They’re getting better at understanding layout, handwriting, tables data.  
  > (ps: My earlier fav was Mist

### 146. [Are Local LLMs good enough for Vibe Coding? Gemma4-26B-A4B vs Qwen3.5-35B-A3B](https://www.reddit.com/r/LocalLLaMA/comments/1sfnuwu/are_local_llms_good_enough_for_vibe_coding/)
- **Subreddit:** r/LocalLLaMA | **Score:** 1 | **Comments:** 1 | **Date:** 2026-04-08 09:23 UTC
- **Author:** u/Interesting_Key3421 | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 30
  > https://grigio.org/are-local-llms-good-enough-for-vibe-coding-gemma4-26b-a4b-vs-qwen3-5-35b-a3b/

### 147. [New local agent framework with efficient browser use](https://www.reddit.com/r/LocalLLaMA/comments/1sb8403/new_local_agent_framework_with_efficient_browser/)
- **Subreddit:** r/LocalLLaMA | **Score:** 3 | **Comments:** 0 | **Date:** 2026-04-03 08:45 UTC
- **Author:** u/cride20 | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 30
  > Hey, have you ever wondered how cool would it be to run a whole agent that can do a lot of things locally?
  > 
  > Say less, I was thinking how far can I push Qwen3.5-35B-A3B model (UD\_Q4\_K\_XL unlosth)
  > 
  > I started making a framework that gives it almost infinite possibilites like filesystem usage, TTS, FFMPEG, STT, browser use etc....
  > 
  > So far I can use this framework to generate a TTS story combine all the generated files and burn them as subtitles on a minecraft parkour video (yes, I'm making local 

### 148. [100% Local free experiment: Agent + Model + GAME ENGINE - Need Tips &amp; Tricks](https://www.reddit.com/r/LocalLLaMA/comments/1sabbcd/100_local_free_experiment_agent_model_game_engine/)
- **Subreddit:** r/LocalLLaMA | **Score:** 3 | **Comments:** 0 | **Date:** 2026-04-02 08:09 UTC
- **Author:** u/VirtualWishX | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 30
  > I'm curious about trying something I want to test which supposed to run 100% locally, Free, Offline using my PC Specs limits:
  > 
  > Before I made this post I did a small test and it was very impressive for what it is and it made me wondering if I can push the limits to something better with more control for more complex project.
  > 
  > I simply loaded **LMStudio** (because I'm a visual person) and I've tested:  
  > **Qwen3.5 35B A3B Q4\_K\_M** \- (probably there are newer / better versions up to date)
  > 
  > I trie

### 149. [Feedback on my hybrid local + cloud LLM architecture (llama.cpp + OpenRouter + MCP + RAG)](https://www.reddit.com/r/LocalLLaMA/comments/1sbgtd8/feedback_on_my_hybrid_local_cloud_llm/)
- **Subreddit:** r/LocalLLaMA | **Score:** 3 | **Comments:** 0 | **Date:** 2026-04-03 15:32 UTC
- **Author:** u/Ill_Leadership1076 | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 30
  > Hey everyone,
  > 
  > I’ve been building a hybrid LLM setup and wanted to get some feedback from people who are more experienced with pipelines.
  > 
  > My idea is to combine local models (for cost/privacy) with cloud models (for stronger reasoning maybe not use cloud atall), and route between them intelligently.
  > 
  > I am wondering that is my planning looks correct , if not what can be improved.
  > 
  > [Current workflow ](https://preview.redd.it/l4c34h8kuzsg1.png?width=2132&amp;format=png&amp;auto=webp&amp;s=152f5f088

### 150. [[P] Fine-tuned 8B model for Quantum Cryptography](https://www.reddit.com/r/MachineLearning/comments/1pykrn2/p_finetuned_8b_model_for_quantum_cryptography/)
- **Subreddit:** r/MachineLearning | **Score:** 0 | **Comments:** 14 | **Date:** 2025-12-29 12:12 UTC
- **Author:** u/Disastrous_Bid5976 | **Query match:** "Nemotron Cascade"
- **Relevance score:** 28
  > https://preview.redd.it/l9mf2szxh3ag1.png?width=1948&amp;format=png&amp;auto=webp&amp;s=aaadb8254f3a7e6d2df05a3b6d14c7210d0f1370
  > 
  > Experiment/Job ID/Result
  > 
  > |BB84 Basis|d57r147p3tbc73aqi44g|QBER 1.3%|
  > |:-|:-|:-|
  > |Bell/CHSH|d57r0ubht8fs73a33s9g|S = 2.475|
  > |5-Qubit GHZ|d57qv1jht8fs73a33qig|Fidelity 86.6%|
  > 
  > Sharing a domain-specific fine-tune for quantum cryptography (QKD protocols, QBER analysis, attack simulation).  
  >   
  >   
  > Setup:  
  > \- Base: Nemotron-Cascade-8B-Thinking  
  > \- LoRA r=64, 8,213 exampl

### 151. [[P] mlx-onnx: Run your MLX models in the browser using ONNX / WebGPU](https://www.reddit.com/r/MachineLearning/comments/1rdrurq/p_mlxonnx_run_your_mlx_models_in_the_browser/)
- **Subreddit:** r/MachineLearning | **Score:** 5 | **Comments:** 1 | **Date:** 2026-02-24 20:34 UTC
- **Author:** u/rut216 | **Query match:** "Rapid-MLX"
- **Relevance score:** 21
  > **Web Demo:** [https://skryl.github.io/mlx-ruby/demo/](https://skryl.github.io/mlx-ruby/demo/)
  > 
  > **Repo:** [https://github.com/skryl/mlx-onnx](https://github.com/skryl/mlx-onnx)
  > 
  > **What My Project Does**
  > 
  > It allows you to convert MLX models into ONNX (onnxruntime, validation, downstream deployment). You can then run the onnx models in the browser using WebGPU.
  > 
  > * Exports MLX callables directly to ONNX
  > * Supports both Python and native C++ interfaces
  > 
  > **Target Audience**
  > 
  > * Developers who want to 

### 152. [AutoBe vs Claude Code: coding agent developer's review of the leaked source code of Claude Code](https://www.reddit.com/r/LocalLLaMA/comments/1sexhy2/autobe_vs_claude_code_coding_agent_developers/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 1 | **Date:** 2026-04-07 14:25 UTC
- **Author:** u/jhnam88 | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 20
  > I build another coding agent — AutoBe, an open-source AI that generates entire backend applications from natural language.
  > 
  > When Claude Code's source leaked, it couldn't have come at a better time — we were about to layer serious orchestration onto our pipeline, and this was the best possible study material.
  > 
  > Felt like receiving a gift.
  > 
  > ## TL;DR
  > 
  > 1. Claude Code—source code leaked via an npm incident
  >    - `while(true)` + autonomous selection of 40 tools + 4-tier context compression
  >    - A master

### 153. [Easy OpenClaw setup with Discord on Docker without TUI/WebUI](https://www.reddit.com/r/LocalLLaMA/comments/1s93109/easy_openclaw_setup_with_discord_on_docker/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 1 | **Date:** 2026-03-31 22:58 UTC
- **Author:** u/chibop1 | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 20
  > I needed to set up OpenClaw with Discord in a headless Docker without relying on the TUI or WebUI which are very annoying to use with screen readers.
  > 
  > I created a short tutorial along with scripts to manage the Docker setup:
  > 
  > https://github.com/chigkim/easyclaw
  > 
  > It includes:
  > 
  > * Image: ghcr.io/openclaw/openclaw:latest
  > * Preconfigured with OpenAI Responses API to run with various engines/model setup
  > * Easy script: `claw [init|config|log|start|stop|restart|build|update|run|dashboard]`
  > * OpenClaw ru

### 154. [gemma 4 26b a4b coding impressions](https://www.reddit.com/r/LocalLLaMA/comments/1seg374/gemma_4_26b_a4b_coding_impressions/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 1 | **Date:** 2026-04-07 00:04 UTC
- **Author:** u/Chilalala | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 20
  > speed is usable on my m1 max, but can take a while for even a simple html test project with sporadic weird syntax errors in html, css and js that take a few iterations to fix...

### 155. [Openclaw y gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf](https://www.reddit.com/r/LocalLLaMA/comments/1scki7k/openclaw_y_gemma426ba4bitudq4_k_xlgguf/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 1 | **Date:** 2026-04-04 20:56 UTC
- **Author:** u/Ashamed-Honey1202 | **Query match:** "Gemma 4 26B A4B"
- **Relevance score:** 20
  > Estoy muy sorprendido de que esto esté funcionando en mi máquina y tan bien.
  > 
  > Tengo 32gb RAM y 12gb de vram.
  > 
  > Esta mañana he hecho una prueba y me daba en Unsloth 40tokens por segundo de salida, así que me he decidido a arrancar un server de llama e instalar openclaw.
  > 
  > He arrancado llama con esta configuración:
  > 
  > &amp; "C:\\IA\\llama.cpp\\llama-server.exe" \`
  > 
  > \-m "C:\\IA\\models\\gemma-4-26b-a4b\\gemma-4-26B-A4B-it-UD-Q4\_K\_XL.gguf" \`
  > 
  > \--mmproj "C:\\IA\\models\\gemma-4-26b-a4b\\mmproj-BF16.gg

### 156. [Anyone tried TurboQuant on MLA models like GLM-4.7-Flash?](https://www.reddit.com/r/LocalLLaMA/comments/1sdpige/anyone_tried_turboquant_on_mla_models_like/)
- **Subreddit:** r/LocalLLaMA | **Score:** 2 | **Comments:** 0 | **Date:** 2026-04-06 05:00 UTC
- **Author:** u/Aromatic_Mind_4084 | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 20
  > Has anyone tried TurboQuant on MLA models like GLM-4.7-Flash?
  > 
  > I am curious whether it works well in practice, what the performance gains look like, and whether there are any quality tradeoffs or implementation issues. Would love to hear if anyone has tested this in a real setup.

### 157. [The best local translation models for a 32GB VRAM 5090 setup](https://www.reddit.com/r/LocalLLaMA/comments/1s03cii/the_best_local_translation_models_for_a_32gb_vram/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 1 | **Date:** 2026-03-21 21:29 UTC
- **Author:** u/personalaccount14 | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 20
  > I'm sharing the best, **fast** local translation models I've found for a **32GB VRAM 5090 GPU VRAM-only** setup. I'm still using DDR4, so my recommendations don't account for system RAM.
  > 
  > My primary language pairs are Swedish-English and Korean-English.
  > 
  > I recommend TranslateGemma models which are significantly better according to Google than Gemma3 27b at translation, but they use user-user prompts and not the system-user format. I don't know how to make them take system-user prompts; I think i

### 158. [glm-4.7-flash on nvidia blackwell and vllm](https://www.reddit.com/r/LocalLLaMA/comments/1rofipw/glm47flash_on_nvidia_blackwell_and_vllm/)
- **Subreddit:** r/LocalLLaMA | **Score:** 1 | **Comments:** 1 | **Date:** 2026-03-08 20:28 UTC
- **Author:** u/Rich_Artist_8327 | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 9
  > Not able to run glm-4.7-flash on 2x5090 and latest vllm docker nightly. updated transformers. What to do.
  > 
  > Edit: will actually not anymore try to use it, its too unsafe model for my needs

### 159. [What is LLMFit Smoking? Can M1 Max run anything decently enough for agentic coding?](https://www.reddit.com/r/LocalLLaMA/comments/1sfnieh/what_is_llmfit_smoking_can_m1_max_run_anything/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 0 | **Date:** 2026-04-08 09:02 UTC
- **Author:** u/GoodhartMusic | **Query match:** "Qwen3.5 35B A3B"
- **Relevance score:** 0
  > As you can see in this analysis, LLMfit estimated 85 tokens per second with a 64B model. When i tried, I got 9t/s. :'( I'm pretty extremely new to local inference and wonder if an m1 max can realistically take advantage of that in a meaningful way, even if a substantial process takes hours?

### 160. [LlamaSuite progress](https://www.reddit.com/r/LocalLLaMA/comments/1rrih9q/llamasuite_progress/)
- **Subreddit:** r/LocalLLaMA | **Score:** 0 | **Comments:** 0 | **Date:** 2026-03-12 05:43 UTC
- **Author:** u/vk3r | **Query match:** "GLM 4.7 Flash"
- **Relevance score:** 0
  > Hello!  
  > Victor here.
  > 
  > I apologize for the lack of updates or the repository. I’ve only been able to work on it during the evenings because of my job.
  > 
  > I’ve made several very interesting improvements:
  > 
  > * **New Models page:** It allows you to view, edit, copy, upload/download models, and launch the chat in the default browser. Everything works in real time.
  > * **New Files page:** It allows creating/deleting folders and downloading/renaming/deleting files. It has been optimized and now all download


---

## Full Thread Content (Top Threads)

### Thread 1: Gemma 4 has been released
- **Link:** https://www.reddit.com/r/LocalLLaMA/comments/1salgre/gemma_4_has_been_released/
- **Score:** 2273 | **Comments:** 663 | **Date:** 2026-04-02 16:01 UTC

**Post body:**

[https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)

[https://huggingface.co/unsloth/gemma-4-31B-it-GGUF](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF)

[https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF)

[https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF)

[https://huggingface.co/collections/google/gemma-4](https://huggingface.co/collections/google/gemma-4)

**What’s new in Gemma 4** [https://www.youtube.com/watch?v=jZVBoFOJK-Q](https://www.youtube.com/watch?v=jZVBoFOJK-Q)

Gemma is a family of open models built by Google DeepMind. Gemma 4 models are multimodal, handling text and image input (with audio supported on small models) and generating text output. This release includes open-weights models in both pre-trained and instruction-tuned variants. Gemma 4 features a context window of up to 256K tokens and maintains multilingual support in over 140 languages.

Featuring both Dense and Mixture-of-Experts (MoE) architectures, Gemma 4 is well-suited for tasks like text generation, coding, and reasoning. The models are available in four distinct sizes: **E2B**, **E4B**, **26B A4B**, and **31B**. Their diverse sizes make them deployable in environments ranging from high-end phones to laptops and servers, democratizing access to state-of-the-art AI.

Gemma 4 introduces key **capability and architectural advancements**:

* **Reasoning** – All models in the family are designed as highly capable reasoners, with configurable thinking modes.
* **Extended Multimodalities** – Processes Text, Image with variable aspect ratio and resolution support (all models), Video, and Audio (featured natively on the E2B and E4B models).
* **Diverse &amp; Efficient Architectures** – Offers Dense and Mixture-of-Experts (MoE) variants of different sizes for scalable deployment.
* **Optimized for On-Device** – Smaller models are specifically designed for efficient local execution on laptops and mobile devices.
* **Increased Context Window** – The small models feature a 128K context window, while the medium models support 256K.
* **Enhanced Coding &amp; Agentic Capabilities** – Achieves notable improvements in coding benchmarks alongside native function-calling support, powering highly capable autonomous agents.
* **Native System Prompt Support** – Gemma 4 introduces native support for the `system` role, enabling more structured and controllable conversations.

# Models Overview

Gemma 4 models are designed to deliver frontier-level performance at each size, targeting deployment scenarios from mobile and edge devices (E2B, E4B) to consumer GPUs and workstations (26B A4B, 31B). They are well-suited for reasoning, agentic workflows, coding, and multimodal understanding.

The models employ a hybrid attention mechanism that interleaves local sliding window attention with full global attention, ensuring the fi

**Top comments:**

**u/WithoutReason1729** (score: 1):
> Your post is getting popular and we just featured it on our Discord! [Come check it out!](https://discord.gg/PgFhZ8cnWW)

You've also been given a special flair for your contribution. We appreciate your post!

*I am a bot and this action was performed automatically.*

**u/danielhanchen** (score: 506):
> * Gemma-4 has **native thinking, tool calling and is multimodal!**
* Use temperature = 1.0, top\_p = 0.95, top\_k = 64 and the EOS is `&lt;turn|&gt;`. `&lt;|channel&gt;thought\n` is also used for the thinking trace!
* Guide to run them at [https://unsloth.ai/docs/models/gemma-4](https://unsloth.ai/docs/models/gemma-4)
* Gemma-4 also works seamlessly in Unsloth Studio! [https://unsloth.ai/docs/new/studio](https://unsloth.ai/docs/new/studio)
* All GGUFs at [https://huggingface.co/collections/unsloth/gemma-4](https://huggingface.co/collections/unsloth/gemma-4)

  **u/jacek2023** (score: 123):
  > thanks for the quick GGUF release!!!

    **u/danielhanchen** (score: 57):
    > Thanks for the post as well haha - you we were lightning fast as well :)

  **u/NoahFect** (score: 37):
  > Hey, quick question re: Unsloth Studio.  I'm thinking of switching over to it from my existing llama.cpp installation, but why do I need to create an account to run stuff locally?

    **u/danielhanchen** (score: 22):
    > It's out! See https://github.com/unslothai/unsloth?tab=readme-ov-file#-quickstart

For Linux, WSL, Mac: `curl -fsSL https://unsloth.ai/install.sh | sh`
For Windows: `irm https://unsloth.ai/install.ps1 | iex`

  **u/970FTW** (score: 12):
  > Truly the best to ever do it lol

    **u/danielhanchen** (score: 9):
    > Thanks!

  **u/illcuontheotherside** (score: 6):
  > You guys ROCK!!!

  **u/Daniel_H212** (score: 6):
  > It seems like native tool calling isn't working very well. Is this a model problem or me? I'm running 26B-A4B at UD-Q6\_K\_XL with all the same settings in OpenWebUI as Qwen3.5-35B-A3B also at the same quant, (native tool calling on, web search and web scrape tools enabled), plus with &lt;|think|&gt; at the start of the system prompt to enforce thinking, and when given a research task, Qwen3.5 did a web search (searxng, so only snippets were returned from each result) and then scraped 5 specific pages, while gemma 4 did a web search, summarised, came up with a research plan, and then immediate

    **u/danielhanchen** (score: 4):
    > Try Unsloth Studio - it works wonders in it! We tried very hard to make tool calling work well - sadly nowadays it's not the model, but rather the harness / tool that's more problematic

https://preview.redd.it/q26cxh2o0usg1.png?width=2880&amp;format=png&amp;auto=webp&amp;s=502c2cc5c710d6700f2d0af45f0de144adaf0121



**u/Both_Opportunity5327** (score: 529):
> Google is going to show what open weights is about. 

Happy Easter everyone.

  **u/Daniel_H212** (score: 109):
  > Wish they'd release bigger models though, a 100B MoE from them could be great without threatening their proprietary models. Hopefully one is coming later?

    **u/sininspira** (score: 144):
    > If the 31b is as good as the open model rankings suggest, they don't really \*need\* to release a bigger one at the moment...

    **u/RedParaglider** (score: 44):
    > Man 80-120 would be killer, but I'm happy to have what they just released!

    **u/RottenPingu1** (score: 19):
    > I'd settle for 70B

    **u/jacek2023** (score: 19):
    > either the 124B model was too weak and did not beat smaller ones in benchmarks/ELO, or it was too strong and threatened Gemini

  **u/ThiccStorms** (score: 8):
  > I'm very excited for the 2b!

**u/itsdigimon** (score: 184):
> Did Google just release a 26B A4B model? Sounds like christmas is early for GPU poor folks :')

  **u/bikemandan** (score: 59):
  > Will it run on my Commodore 64?

    **u/FlamaVadim** (score: 38):
    > Naturlich!

    **u/picosec** (score: 15):
    > If you have enough external storage attached it should be able to run. You might be able to achieve low single-digit tokens per year.

    **u/roselan** (score: 5):
    > Easily.

    **u/toothpastespiders** (score: 5):
    > Main reason I'm bummed about the lack of a 120b model. I was all prepped to start writing it to floppy for my Commodore 128.

  **u/Final_Ad_7431** (score: 29):
  > yeah im only really able to run qwen3.5 35b on 8gb vram, im very excited to compare this new moe

    **u/mattrs1101** (score: 9):
    > What settings do you use? 

    **u/Borkato** (score: 6):
    > Qwen 3.5 35B is indeed god tier tho! 

**u/StatFlow** (score: 162):
> apache license is new - not a 'google gemma' license anymore!

  **u/Borkato** (score: 23):
  > Woah, what’s the difference? Is it like super open now? :D

    **u/StatFlow** (score: 77):
    > apache 2.0 is the gold standard and fully permissive. the google gemma license was "open" but google technically had the ability to restrict for any reason if they wanted to/it came to that.

**u/Altruistic_Heat_9531** (score: 385):
> https://preview.redd.it/qg7b58pszssg1.jpeg?width=500&amp;format=pjpg&amp;auto=webp&amp;s=4a2a21419855733128a49ce7baa74505addd7025

  **u/Altruistic_Heat_9531** (score: 414):
  > And after a week maybe : "Gemma 4 26B Heretic Uncensored Ablated Claude Opus 4.6 Reasoning Distlled Expanded fine tuned quantized"

  
Sorry to tempting lol

    **u/LagOps91** (score: 119):
    > you forgot turbo quant in there!

    **u/marcoc2** (score: 31):
    > Gemmopus

    **u/sibilischtic** (score: 25):
    > Eh im going to wait for

Gemma 4 26B Heretic Uncensored Ablated Claude Opus 4.6 Chain of Thot (NSFW) Quasimodal chuck Norris bingo night 

    **u/bucolucas** (score: 47):
    > "Hey guys which one of the Gemma models is best at 'unconventional roleplay?'"

\*hint hint nod nod wink wink\*

Also it needs to fit inside 1.5GB NVIDIA card from 1999, be able to generate images, and run at 9000 tokens/second

    **u/ea_nasir_official_** (score: 38):
    > Claude: safety

Gpt: wasting money

Google: tracking us all

LocalLlama: UNCENSORED TURBORAPIST CLAUDE DISTILL QWENGEMMA CODER MOE ABLITERATED 6.9B UD-IQ69420 

  **u/AXYZE8** (score: 58):
  > Yup, thats me

https://preview.redd.it/6zdub1w30tsg1.png?width=449&amp;format=png&amp;auto=webp&amp;s=58be39cf2ce80e8a8dae21daf68e36488c6b091f

    **u/BubrivKo** (score: 12):
    > Lol, ok, It seems there are people who are using Q2 models :D

  **u/Far-Low-4705** (score: 25):
  > i was looking at the benchmarks and tbh, it feels like gemma 4 ties with qwen, if not qwen being slightly ahead

and qwen 3.5 is more compute efficient too, 3b active params vs 4b, and 27b vs 31b dense. both tying on benchmarks so i mean idk.

gemma doesnt have an overthinking problem tho, saying "Hi" it only thinks for 30 tokens or so which is way better than 7,000 tokens lol


---

### Thread 2: Gemma 4 just casually destroyed every model on our leaderboard except Opus 4.6 and GPT-5.2. 31B params, $0.20/run
- **Link:** https://www.reddit.com/r/LocalLLaMA/comments/1sdcotc/gemma_4_just_casually_destroyed_every_model_on/
- **Score:** 1729 | **Comments:** 284 | **Date:** 2026-04-05 19:30 UTC

**Post body:**

Tested Gemma 4 (31B) on our benchmark. Genuinely did not expect this.

100% survival, 5 out of 5 runs profitable, +1,144% median ROI. At $0.20 per run.

It outperforms GPT-5.2 ($4.43/run), Gemini 3 Pro ($2.95/run), Sonnet 4.6 ($7.90/run), and absolutely destroys every Chinese open-source model we've tested — Qwen 3.5 397B, Qwen 3.5 9B, DeepSeek V3.2, GLM-5. None of them even survive consistently.

The only model that beats Gemma 4 is Opus 4.6 at $36 per run. That's 180× more expensive.

31 billion parameters. Twenty cents. We double-checked the config, the prompt, the model ID — everything is identical to every other model on the leaderboard. Same seed, same tools, same simulation. It's just this good.

Strongly recommend trying it for your agentic workflows. We've tested 22 models so far and this is by far the best cost-to-performance ratio we've ever seen.

Full breakdown with charts and day-by-day analysis: [foodtruckbench.com/blog/gemma-4-31b](https://foodtruckbench.com/blog/gemma-4-31b)

*FoodTruck Bench is an AI business simulation benchmark — the agent runs a food truck for 30 days, making decisions about location, menu, pricing, staff, and inventory. Leaderboard at* [*foodtruckbench.com*](https://foodtruckbench.com)

**EDIT — Gemma 4 26B A4B results are in.**

Lots of you asked about the 26B A4B variant. Ran 5 simulations, here's the honest picture:

**60% survival** (3/5 completed, 2 bankrupt). Median ROI: +119%, Net Worth: $4,386. Cost: $0.31/run. Placed #7 on the leaderboard — above every Chinese model and Sonnet 4.5, below everything else.

Both bankruptcies were loan defaults — same pattern we see across models. The 3 surviving runs were solid, especially the best one at +296% ROI.

**But here's the catch.** The 26B A4B is the only model out of 23 tested that required custom output sanitization to function. It produces valid tool-call intent, but the JSON formatting is consistently broken — malformed quotes, trailing garbage tokens, invalid escapes. I had to build a 3-stage sanitizer specifically for this model. No other model needed anything like this. The business decisions themselves are unmodified — the sanitizer only fixes JSON formatting, not strategy. But if you're planning to use this model in agentic workflows, be prepared to handle its output format. It does not produce clean function calls out of the box.

**TL;DR:** 31B dense → 100% survival, $0.20/run, #3 overall. 26B A4B → 60% survival, $0.31/run, #7 overall, but requires custom output parsing. The 31B is the clear winner. Updated leaderboard: foodtruckbench.com

**Top comments:**

**u/Recoil42** (score: 234):
> OP: Looks like you don't have an inference cost column on your results page at all? Seems like it would be useful. 

  **u/Disastrous_Theme5906** (score: 78):
  > Yeah fair point, it's not on the main leaderboard table yet. Cost data is in the individual case studies but should probably be a column on the main page too. Adding it to the list.

**u/jkflying** (score: 93):
> How does the MoE model do?

  **u/Disastrous_Theme5906** (score: 75):
  > MoE models didn't do well on our bench. Qwen 3.5 397B (17B active) only has 29% survival and negative ROI. DeepSeek V3.2 survives 62% of the time but still ends up in the red. Gemma 4 being dense and still beating all of them at 31B is honestly the most surprising part.

    **u/dampflokfreund** (score: 108):
    > They are talking about Gemma 4 26B A4B.

    **u/Deep90** (score: 22):
    > Just a suggestion, I think it would be interesting if you started running a hidden seed for all models to see if any of them are potentially being trained and over-fitted to your benchmark.

    **u/BankruptingBanks** (score: 3):
    > I think he meant gemma moe model

    **u/Phaelon74** (score: 2):
    > Dense always has an advantage on Moe's.  So that should bot be all that surprising.

    **u/nnxnnx** (score: 2):
    > Would be great to test Qwen 3.5 27B (dense) - which would be a much better "equivalent" to compare against compared to Qwen 3.5 9B that is currently the "closest model" in the leaderboard.

    **u/BidWestern1056** (score: 1):
    > commented this separately but same on npcsh benchmarks, it performed worse than gemma3:4b

    **u/inphaser** (score: 1):
    > Do you know if it has already been converted for apple MLX?

    **u/RobotechRicky** (score: 1):
    > Which model is best for coding?  Is Gemma 4 it?

    **u/FatheredPuma81** (score: 1):
    > If MoE doesn't perform well then test Qwen3.5 27B? Other benchmarks show it on par with 122B.

**u/exact_constraint** (score: 42):
> Be interesting to see Qwen3.5 27B added to the test matrix - 31b dense vs Qwen MOE isn’t a super fair comparison, imo. 

  **u/Disastrous_Theme5906** (score: 7):
  > We tested Qwen 3.5 at both 9B and 397B — the 397B actually went bankrupt. More parameters didn't help. Qwen 3.5 is a great model overall, but this kind of sustained multi-day agentic task seems to hit different. Not sure the 27B would change the picture much — probably needs a generation-level jump.



    **u/Awwtifishal** (score: 15):
    > The 397B has 17B active parameters. Maybe the active parameter counts matters much more than the total (edit: for this specific test). The 27B dense has 27B active parameters.

    **u/SSOMGDSJD** (score: 7):
    > Earlier in this thread you said MoEs specifically don't perform well on this bench. Why would you run Gemma 4 31b dense and Moe and then balk at running qwen.5 27B, dense gemmas obvious nearest neighbor? Qwen3.5 27b has been getting a lot of praise for its performance at its param count, would be nice to see direct comparisons between it and Gemma 4 31b.

**u/DetouristCollective** (score: 32):
> Do you have any plans to compare it to another comparable dense model like Qwen3.5 27B?

**u/Adventurous-Paper566** (score: 59):
> Gemma 4 is the first local model I can run on 32Gb of VRAM without having to correct it.


I'm talking with it, with an average stt time of 2 minutes per input, and he NEVER disgress or misunderstood the subject of the conversation. In French. Even Gemini flash makes a lot of mistakes.


It's a huge improvement for Local LLM!


I'm waiting the 124B MoE with impatience! My RAM and CPU will suffer like never lol !

  **u/redditorialy_retard** (score: 8):
  > me with 24 GB :(

    **u/Adventurous-Paper566** (score: 4):
    > Q4_K_XL still good, remove the mmproj to maximize the context length 👍

  **u/bacocololo** (score: 2):
  > Which quantization model do you use please ?
And other parameters ? llamacpp, vllm ...
Merci

    **u/Adventurous-Paper566** (score: 6):
    > Q6\_K\_L bartowski, without the mmproj I can reach 20k context.

Q4\_K\_XL unsloth with the same parameters loads with 65536 tokens but I haven't tested its limits yet.

With the Google's recommended settings.

I only use LM-Studio, I'm a non-technical user (but since LLM are becoming a passion, i'm considering building a dedicated Linux server in the future).

https://preview.redd.it/6ujkc8xdpjtg1.png?width=748&amp;format=png&amp;auto=webp&amp;s=b7db939244353424ee045533725bd3ed1ff6b991

**u/YetiTrix** (score: 67):
> Gemma 4 didn't really work for my use case. Which is diagnosing PLC Code. Qwen-Coder-Next still does best job for that.

  **u/Disastrous_Theme5906** (score: 25):
  > Makes sense, 31B is still a small model and can't be great at everything. Our benchmark tests agentic decision-making, not coding. For PLC diagnostics and dev tasks there are definitely better options at this size. Qwen-Coder is solid for that.

    **u/ceo_of_banana** (score: 1):
    > How does it test that? I've heard the world many times but I'm still not sure what it means.

  **u/Ryukish** (score: 5):
  > Have you tried a mix of skills + more detailed prompts. I usually find that open source models need me to be more explicit to get opus level performance 

    **u/YetiTrix** (score: 11):
    > PLC code is especially hard because it's not really trained on especially text representation of ladder logic. It doesn't exist out on the internet. The models have to infer a lot more of the meaning. Yes, I do A/B testing with my prompts.

I don't really want to put in time training a model of PLC code because I feel like eventually allen bradley will do it and integrate it for their code generation. But my agent is for diagnosing live machine in a production environment. It assists operators in why a machine isn't working by looking at the state of the current live code running. That code co

  **u/Embarrassed_Adagio28** (score: 1):
  > Yeah qwen 3 coder next is still my best local coding model. It is great at tool calling and with the right project structure, can run for hours at a time without stopping to ask questions. Even with the iq3_xss quant. 

**u/GrungeWerX** (score: 32):
> Why isn't **Qwen 3.5 27B** in this testing? That's the only fair comparison to the 31B as they're both dense models...

**u/aristotle-agent** (score: 40):
> yikes.   Question:  does it *feel*  better than those paid models?   

( like is performance better feeling than sonn4.6 and gem3pro from your image?) 

  **u/Disastrous_Theme5906** (score: 48):
  > Genuinely yes. In terms of agentic reasoning this model is way above Sonnet 4.6 and Gemini 3 Pro. The decision quality is closer to GPT-5.2/5.3/5.4 xhigh honestly. How they achieved this in 31B params we don't fully understand yet, but Google says they specifically trained Gemma 4 for agentic tasks so that probably explains a lot.

    **u/johnnyXcrane** (score: 33):
    > I really really doubt that. Perhaps in some specific use cases. But I have not yet tested it so I am not saying you lying. I just read that so often here and in tons of benchmarks and all always were way worse than SOTA.

    **u/joost00719** (score: 21):
    > My experience is way worse than even qwen 3.5 35b. It fails to even edit a json file. I mean, it does edit the file, it just fucks over the syntax.

I don't like it. I wish I could, but for programming it's kinda bad. 

    **u/Venium** (score: 5):
    > &gt;quality is closer to GPT-5.2/5.3/5.4 xhigh

lol, lmao even.

    **u/Nervous_Variety5669** (score: 14):
    > Genuinely, I do not appreciate you insulting our intelligence. Your benchmark is vibes and so are your comments. You've lost all credibility with claiming, and I will quote what you said (not that pivot you made to another commenter narrowing the scope to your vibe benchmark):

"In terms of agentic reasoning this model is way above Sonnet 4.6 and Gemini 3 Pro. The decision quality is closer to GPT-5.2/5.3/5.4 xhigh honestly."

Second, you have not provided the parameters you used for any of the models.

\- What reasoning effort did you use for each model?  
\- Did you configure compaction? How

    **u/randylush** (score: 2):
    > You say the word “genuinely” a lot

  **u/Ardalok** (score: 3):
  > It feels better than Gemini 3 Flash, or at least on par.

**u/one-escape-left** (score: 20):
> from your blog post: "**Qwen 3.5 9B** (bankrupt tier, $0.15/run) — the closest model in parameter count and price"

This is incorrect. Qwen 3.5 27B is the closest dense model in the family. Have you considered running that model?





**u/kavakravata** (score: 15):
> Can i run it with a single 3090? 😁😁


---

### Thread 3: Gemma 4 and Qwen3.5 on shared benchmarks
- **Link:** https://www.reddit.com/r/LocalLLaMA/comments/1saoyj7/gemma_4_and_qwen35_on_shared_benchmarks/
- **Score:** 855 | **Comments:** 235 | **Date:** 2026-04-02 18:07 UTC

**Top comments:**

**u/WithoutReason1729** (score: 1):
> Your post is getting popular and we just featured it on our Discord! [Come check it out!](https://discord.gg/PgFhZ8cnWW)

You've also been given a special flair for your contribution. We appreciate your post!

*I am a bot and this action was performed automatically.*

**u/Different_Fix_2217** (score: 135):
> Using both side by side Qwen3.5 is MUCH better at image understanding as well.

  **u/AlexMan777** (score: 55):
  > I confirm. Qwen I much better with images and series of images (tested up to ~280 images at once as frames from video and Qwen did it like a champion) 

    **u/ZenaMeTepe** (score: 12):
    > Can you share the details? I have a need for a similar use case but encoding 280 images at full size will take forever. Did you resize, combine into a grid? What was the goal of your process? Getting a description of the video which frames represented? 

    **u/AlwaysLateToThaParty** (score: 2):
    > &gt; (tested up to ~280 images at once as frames from video and Qwen did it like a champion) 

What type of prompt do you use for that type of analysis?  I've found that it just starts building a narrative that separates from the images as it progresses.  Is there some trick to that process?

  **u/FinBenton** (score: 8):
  > Also for creative writing, I get much much better stuff out of Qwen while super safe gemma just kinda ignores most of the instructions to default to some generic paths in writing. Qwen is just, say less I got you :D

e. Actually I just updated my llama.cpp with the latest fixes, this seemed to help gemma A LOT, seems like it was kinda broken.

    **u/Koalateka** (score: 1):
    > I like both, but I prefer the writing of Gemma 4, in my test it hasn't ignored instructions (thinking enabled).

    **u/Frosty_Chest8025** (score: 1):
    > yes, its good to test a model with a proper tool. nobody should use Ollamas, LM-studios or llamas.cpp. These are for hobbyists.

  **u/MerePotato** (score: 9):
  > Qwen's definitely better with English and Chinese text but I'm skeptical of this claim, Deepmind are really, really good at multimodal stuff

    **u/Different_Fix_2217** (score: 19):
    > Qwen3.5 is absurdly good. And I never liked any qwen model before that series.

    **u/obvithrowaway34434** (score: 5):
    > Qwen always had the best multimodal open source models. Qwen 2.5 7B and 72B VL were the best open source options for quite a long time. Deepmind will never open-source the good stuff, that's like the only reason I (and I suspect many others) use Gemini.

  **u/Birdinhandandbush** (score: 1):
  > It took me a while to get it analysing images on OpenClaw but it even guessed the decade of a picture earlier from the style of dress and the style of picture. It's seriously impressive. And I'm just using the 9b variant 

  **u/Clear-Ad-9312** (score: 1):
  > I went looking for why qwen 3.5 can read images better than gemma 4, and I see I am not the only one.

**u/atape_1** (score: 106):
> Hmmm, not the earth shattering kaboom we were hoping for, but still nice to see! 

  **u/dampflokfreund** (score: 101):
  > To be fair, Qwen releases a model every two weeks or so, no chance for Gemma to catch up in benchmarks, but it doesn't have to. Real world use cases are much more important and we know where Gemma will take the clear lead - multilingual and writing capabilities. 

    **u/DoorStuckSickDuck** (score: 57):
    > Tool usage is becoming increasingly more useful (especially for enriching writing with sources using RAG, and having the agent be able to query different parts of the query automatically), so seeing the tool usage be so poor for the Gemma models is a bit disappointing. More testing will be required ofc.

  **u/AnticitizenPrime** (score: 44):
  > At least Gemma probably won't use 50 billion tokens with each request.

Edit: But it seems it can, if that's what you want! see here: https://www.reddit.com/r/LocalLLaMA/comments/1sav9wg/gemma_4_is_efficient_with_thinking_tokens_but_it/

    **u/dtdisapointingresult** (score: 18):
    > Gemma 4 is a reasoning model. Don't expect the quick answers you were used to in Gemma 3.

    **u/Weak-Shelter-1698** (score: 8):
    > Fr. I hate that qwen 3.5 doesn't support context shift 

  **u/Far-Low-4705** (score: 1):
  > well, one benefit is that gemma doesnt overthink as much as qwen, especially on "hi", also images use FAR less tokens than qwen. so better for context usage and prevents more context rot.

but otherwise id agree with you, might be sticking to qwen... especially the upcoming qwen 3.6

    **u/AnticitizenPrime** (score: 18):
    > &gt; especially on "hi"

Just tested with Gemma 31B:

&gt;&gt;The user said "Hi".
This is a standard greeting.

&gt;&gt;Respond with a friendly, welcoming greeting and an offer to help.

&gt;&gt;Plan:

&gt;&gt;Acknowledge the greeting.

&gt;&gt;Ask how I can assist the user today.

&gt;**Hello! How can I help you today?**

Love to see it.

    **u/Pristine-Woodpecker** (score: 8):
    > The images to tokens thing in Gemma-4 is configurable (see the docs!), I think that's why people are reporting it's much worse than Qwen 3.5 at image understanding.

**u/Apprehensive-View583** (score: 263):
> woo, Qwen3.5 27b is really the beast

  **u/rm-rf-rm** (score: 200):
  > You should always try out the model for yourself and decide. Benchmarks are notoriously unreliable now.

    **u/toothpastespiders** (score: 74):
    > I'll always advise people to make their own benchmarks based on how they use LLMs. Even the most hastily put together, tiny, self-made benchmark that tests against one's personal needs is going to say a million times more than the big public benchmarks.

    **u/SodaBurns** (score: 38):
    > Never trust trust me bro benchmarks 

    **u/Guinness** (score: 10):
    > Yep, Minimax M2.5 and GLM 5 score extremely high and yet I find myself consistently going to Kimi K2.5.

  **u/TechExpert2910** (score: 15):
  > It doesn't seem to "write" nearly as well as Gemma (relevant for any chat use-cases), though.

In human-rated response preference ELO scores, Gemma 4 is *miles* ahead of Qwen 3.5:
https://arena.ai/leaderboard/text?license=open-source

    **u/SpicyWangz** (score: 9):
    > This part is very interesting. I'm curious to see where it lands in the next few days

    **u/Technical_Ad_6106** (score: 4):
    > good chance that in a few days it drops 60 places. just like what happend to qwen hehe

  **u/Guinness** (score: 5):
  > Can you imagine being the guy who fired the guy who beat Google at Gemma 4?

  **u/Frosty_Chest8025** (score: 1):
  > Still it lacks quite a lot against Gemma-4

**u/teachersecret** (score: 58):
> Gemma 4 is good. Damn good.

Qwen 27b... also good :).

We're eating pretty well lately.

  **u/RevolutionaryGold325** (score: 1):
  > If you have 32GB VRAM, you need to use Q2 for gemma while using Q8 for Qwen if you want to have 100k context.

**u/AlexMan777** (score: 51):
> My little conclusions from testing:
1. Gemma 31B roughly on par with Qwen 27B intelligence wise. But Gemma is slower because bigger. 
2. Gemma is much better with reasoning in terms of it finishing reasoning and give final answer mush faster then Qwen. Its a big plus.
3. Qwen is much better with image and series of images understanding. Qwen can handle and answer questions about ~280 images at once (as frames from video). Gemma can't.

Resume: didn't find yet where I should use Gemma 31B instead of Qwen 27B (as I use it without reasoning). Didn't test on tool use or agentic. 

  **u/Pristine-Woodpecker** (score: 7):
  > It looks like you configure how much tokens the image stack produces, I imagine that's being an issue here.

  **u/TheTerrasque** (score: 4):
  > Haven't tested images yet, but the a4b one does a lot better on my small agentic tests so far. Finds the data in fewer calls, much less thinking, higher tokens per second, and much larger context.

  **u/tekdemon** (score: 1):
  > Honestly curious how they fare if you disable thinking entirely. Qwen going on wild thinking fests for everything makes the latency really bad for non-chat use cases where you’re waiting for the final output and not reading the reasoning traces.

Need to see which model manages to just go for it and nail most things.

    **u/AlexMan777** (score: 1):
    > After some tests I completely disabled thinking on Qwen 27B and it still works great. Reasoning time in Qwen is way to long and slow.  
On the 35B A3B i've enabled thinking and set reasoning-budget to 1000-1500 tokens as it works much faster and thinking here really helps to make it more intelligent. But almost in all cases if you need some smart answers - 27B even without thinking is a best choice. 

**u/evilbarron2** (score: 63):
> So no reason to move from my Qwen3.5-35B-A3B


---

### Thread 4: Skipping 90% of KV dequant work → +22.8% decode at 32K (llama.cpp, TurboQuant)
- **Link:** https://www.reddit.com/r/LocalLLaMA/comments/1s56g07/skipping_90_of_kv_dequant_work_228_decode_at_32k/
- **Score:** 839 | **Comments:** 113 | **Date:** 2026-03-27 14:56 UTC

**Post body:**

I’ve been working on an open source TurboQuant implementation for KV cache compression in llama.cpp and ran into a hard bottleneck: dequantization.

At long context (32K on M5 Max), dequant alone was taking around 40 percent of decode time.

I tried fixing it the usual way:
- register LUTs  
- SIMD tricks  
- fused kernels  
- branchless math  

Tested about 14 different approaches. None beat the baseline. Hardware was already at the limit.

What ended up working was much simpler.

Flash attention computes softmax weights before touching V.  
At long context, most of those weights are basically zero.

So instead of making dequant faster, I just skip V dequant entirely for positions with negligible attention.

It’s about 3 lines in the kernel.

**Results on Qwen3.5-35B-A3B (M5 Max):**

**TurboQuant KV (turbo3):**
- +22.8% decode at 32K  
- PPL unchanged  
- NIAH: 7/9 → 9/9  

**Standard q8_0 KV cache:**
- +5% decode  
- PPL identical  
- NIAH identical  

So this is not TurboQuant-specific. It’s using attention sparsity directly.

Also tested on M2 Pro:
- 4-mag LUT on K side + sparse V stack cleanly  
- turbo3 went from ~0.45x → ~0.73x vs q8_0  

**Repo and benchmarks:**  
https://github.com/TheTom/turboquant_plus  

**Writeup:**  
https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/sparse-v-dequant.md  

If anyone wants to try this on CUDA or other setups I’d be interested to see results.

*Note: a CUDA port is currently being tested independently. Will share results once available.*

**Top comments:**

**u/WithoutReason1729** (score: 1):
> Your post is getting popular and we just featured it on our Discord! [Come check it out!](https://discord.gg/PgFhZ8cnWW)

You've also been given a special flair for your contribution. We appreciate your post!

*I am a bot and this action was performed automatically.*

**u/qwen_next_gguf_when** (score: 143):
> This absolutely deserves an aggressive upvote.

  **u/Alarmed-Subject-7243** (score: 55):
  > The best kind of optimization is always just realizing you can skip the useless parts entirely. That kind of brutal efficiency is exactly what we need to finally get complex agentic workflows running smooth on local hardware.

    **u/Hephaestite** (score: 12):
    > I did this a few years ago in a deterministic simulation system, we found that the simulation we were running would have taken just over a year largely because what we were working on was suffering from combinatorial explosion. The fix was really simple in the end, symmetry reduction… we collapsed a large number of elements that were probably the same down to a single element.

End result was a reduction from 1 year to about 16 minutes 

  **u/DistanceSolar1449** (score: 36):
  > This is so stupid, even a high schooler can understand this. I love it. It’s so blindingly obvious and yet I’m amazed this isn’t done already.

    **u/AnOnlineHandle** (score: 12):
    > Apparently nobody had tried just using lossless compression on the weights and decompressing layers temporarily when they're used rather than downcasting or moving them in and out of memory until recently? Unless I misunderstood a repo which I skimmed recently.

Even if that had a performance penalty, it seems like it would probably still be far better than moving them in and out of vram for models which are too big to fit on a local GPU, and is so obvious I keep thinking I must have misunderstood the announcement. It would also mean being able to load bigger models which are currently only do

    **u/Succubus-Empress** (score: 2):
    > And they had top people working on it. Huh

  **u/Pidtom** (score: 17):
  > thanks, that's the goal. I want to finish some more validation first, especially CUDA and a few more cross-model checks, then put together a clean PR for llama.cpp. lots to do!

  **u/Succubus-Empress** (score: 0):
  > I don’t like any aggression 

**u/Pentium95** (score: 35):
> I'd love to have this in mainline llama.cpp

**u/Specialist_Sun_7819** (score: 55):
> wait this is actually genius. you tried 14 brute force approaches and the real win was just... not doing the work at all for tokens that dont matter. the fact that attention sparsity at long context is predictable enough to skip 90% of V dequant is wild. 3 lines in the kernel too lol. curious how this holds up at like 64k+ context, does the sparsity ratio keep climbing or does it plateau?

  **u/roxoholic** (score: 40):
  > &gt; not doing the work at all 

That's the best kind of optimization. Nothing beats that.

    **u/mxforest** (score: 27):
    > I have 15 yrs of experience doing that in high stakes environment. Where do I apply?

    **u/wh33t** (score: 7):
    > The best solution to any problem is to just not have the problem to begin with.

  **u/Pidtom** (score: 26):
  > yeah pretty much. was head banging on the M2 Pro trying to squeeze out any gains, tried every trick i could think of. register LUTs, SIMD shuffles, bit arithmetic, fused blocks, branchless math (lots of parsing through papers). 14 approaches, none of them moved the needle. the constant memory LUT on Apple Silicon is just at the hardware floor. you can't beat 4 divergent reads with any amount of ALU cleverness.

then i looked at it differently: the problem isn't how fast you dequant, it's how many positions you're dequanting that don't matter. softmax already tells you which positions are worth

    **u/xXprayerwarrior69Xx** (score: 5):
    > I wish I was big brained like you 

  **u/True_Requirement_891** (score: 3):
  > somebody correct me but is deepseek-v3.2's DSA based on the same idea?

  **u/Rare_Potential_1323** (score: 3):
  > This vaguely reminds me of my drive/folder/file mirror sync program. If the file hash exist on both drives, but named something different you just click a button on which drive you want to mimic. The file name and path are changed to match the other drive. No data transfered, a lot of time and drive lifespan saved. I couldn't find a program like this so I made it. It has other cool features like specializing in maxed out drive space and finding only the superfluous files with an option to move them to a scratch drive. (Vibe coded)

**u/[deleted]** (score: 24):
> [removed]

  **u/Pidtom** (score: 12):
  > interesting thought. the tradeoff is that caching the dequant output puts you back at fp16 memory usage, which defeats the compression. the whole point is keeping it compressed and dequanting on the fly. sparse V is actually a different angle: instead of making dequant faster or caching it, we skip it entirely for positions where softmax says the attention weight is negligible (\~90% at 32K). turns out those positions were contributing quantization noise anyway, NIAH went from 7/9 to 9/9 after removing them

**u/Such_Advantage_6949** (score: 12):
> What is the kld measure for this approach?

  **u/Pidtom** (score: 12):
  > Good question, this was actually on my roadmap but I had not run it yet.

I went ahead and ran KLD on it.

MoE (Qwen3.5-35B-A3B):

* q8\_0: 0.00155
* q4\_0: 0.00809
* turbo3: 0.01615

Dense (Qwen3.5-27B):

* q8\_0: 0.000018
* q4\_0: 0.00274
* turbo3: 0.00990

Top-p agreement is 95 to 99 percent depending on quantization.

So the divergence tracks the underlying quantization level, not sparse V itself. Skipping low-weight positions does not introduce meaningful additional shift.

Main paper updated thank you!

    **u/LirGames** (score: 3):
    > I'd rather see the results of turbo4 which is the one that Google stated to be near-lossless. And maybe even have other/higher bit implementations.
The advantage of TurboQuant should be to increase the context without losing too much on fp16 at long contexts. Q8 is already bad at that, if TurboQuant can be better than it's worth the effort, otherwise it's a neat trick that will have its applications but a bit less revolutionary than what everyone thinks... Personally I'm waiting for benchmarks and integration in main llama.cpp before starting testing myself on real repositories.

    **u/DistanceSolar1449** (score: 1):
    > Is this KLD difference vs no sparse-V quant? Or just the KLD numbers of the sparse-V quant?

    **u/Sliouges** (score: 1):
    > May I ask over how many tokens is KLd? Perhaps I missed it but I didn't find it specified? Single token, multiple tokens, base-forced, autoregressive, number of prompts samples?

  **u/FullOf_Bad_Ideas** (score: 6):
  > it sucks

&gt; turbo3 KLD is roughly 2× q4_0 on both architectures. This is expected: turbo3 uses 3.5 bits (less than q4_0's 4 bits) with a fundamentally different compression mechanism (WHT rotation + polar codebook vs scalar quantization).
&gt; 
&gt; The same-top-p metric shows turbo3 agrees with f16 on the top token 94-96% of the time. For context, q4_0 (a widely-used cache type) agrees 96-98%.

    **u/ParaboloidalCrest** (score: 6):
    > Hmmm I'm afraid that looks like a show-stopper. Q8 kv cache was already questionable at best.

    **u/Caffeine_Monster** (score: 2):
    > KLD increasingly feels like a meaningless metric.

q4_0 has a tiny KLD loss which typically translates into a large loss in hard benchmark scores vs the full precision model.

    **u/Pidtom** (score: 1):
    > haven’t run sparse V on q4\_0 yet. focus so far has been q8\_0 and turbo3 since those are the main operating points for this work (higher-fidelity baseline and bandwidth-optimized path).

would expect similar ON/OFF behavior, but worth validating explicitly.

**u/One_Temperature5983** (score: 10):
> Nice work — skipping unnecessary dequant is always the best optimization.
                                                                                                                                                  
  I've been working on a parallel TurboQuant implementation for vLLM (turboquant-vllm, just hit 1.0.0 on PyPI). Different stack (Triton/Python,
  HuggingFace DynamicCache, vLLM attention backend) but we hit the same dequant bottleneck. Our solution was incremental dequantization — only    
  dequant the new token each decode step, not the full cache. Took overhead from 3.36x t

  **u/Pidtom** (score: 1):
  >     https://github.com/0xSero/turboquant

since you're working on vLLM you may want to check out this repo   


    **u/Cultured_Alien** (score: 1):
    > the one you're replying to looks like a bot

**u/ketosoy** (score: 26):
> The speed the world is moving is insane, and I love it.

**u/FullOf_Bad_Ideas** (score: 13):
> if it improved NIAH and ppl, it smells wrong

&gt;turbo3 KLD is roughly 2× q4_0 on both architectures. This is expected: turbo3 uses 3.5 bits (less than q4_0's 4 bits) with a fundamentally different compression mechanism (WHT rotation + polar codebook vs scalar quantization).


&gt;The same-top-p metric shows turbo3 agrees with f16 on the top token 94-96% of the time. For context, q4_0 (a widely-used cache type) agrees 96-98%.

Wasn't turbo supposed to be lossless? why is it worse than q4_0????

you are also measuring ppl on 8 chunks, that's not enough.

edit: you measure ppl on 512 ctx where 

  **u/Pidtom** (score: 5):
  > fair points, let me clarify

on turbo3 vs q4\_0: turbo3 is more aggressive compression (3.5 bits vs 4), so slightly higher KLD than q4\_0 is expected. the goal there is better memory efficiency, not strictly minimizing divergence

on the “smells wrong” part: sparse V is only removing positions with near-zero attention weight. those contribute almost no signal but still add quantization noise, so skipping them can slightly clean up the accumulation. that’s why you can see NIAH improve even though overall KLD still tracks the underlying quantization level

on PPL: agreed that 8 chunks at 512 con

    **u/FullOf_Bad_Ideas** (score: 7):
    > &gt;on turbo3 vs q4_0: turbo3 is more aggressive compression (3.5 bits vs 4), so slightly higher KLD than q4_0 is expected. the goal there is better memory efficiency, not strictly minimizing divergence

I don't agree, KLD higher than q4_0  is not expected. TurboQuant is supposed to be essentially loseless, and q4_0 is widely regarded at being too destructive to even consider using. q4_0 is the naive quant that should be absolutely crushed by any serious KV cache quantization strategy.

&gt;on the “smells wrong” part: sparse V is only removing positions with near-zero attention weight. those c

    **u/Pidtom** (score: 5):
    > ran longer-context PPL validation.



from 512 through 32K context, results are numerically identical with and without sparse V at τ=1e-6 on both MoE and dense models.

at 32K, skip rate is \~90%, so sparse V is definitely active. despite that, PPL does not change.

in this setup, skipped positions appear to fall below numerical significance rather than contributing meaningfully to the output.

full results and methodology:

[https://github.com/TheTom/turboquant\_plus/blob/main/docs/papers/sparse-v-dequant.md](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/sparse-v-dequant.md)

  **u/NandaVegg** (score: 1):
  > It's not impossible that noise-gating technique like this could improve benchmarks like NIAH as Transformer is full of noises (even a bit of added noise like naively upping number of active experts can significantly worsen the outputs). Concern is that adding unknown-to-the-model bias from post-quant in many cases lead to weird model behavior (like sudden repetition loop) in actual use, often a significantly worse experience than numbers suggest.

Would get more interesting if model is post-trained with this (attn noise gating, not TurboQuant itself) in mind.

**u/peva3** (score: 5):
> Beat me to it by just a few hours! I'm working on the same exact thing right now. If I make any progress I'll do a pull on your repo.

  **u/Pidtom** (score: 6):
  > Please share your repo. I would love to see what you have.


---

### Thread 5: Per-Layer Embeddings: A simple explanation of the magic behind the small Gemma 4 models
- **Link:** https://www.reddit.com/r/LocalLLaMA/comments/1sd5utm/perlayer_embeddings_a_simple_explanation_of_the/
- **Score:** 518 | **Comments:** 53 | **Date:** 2026-04-05 15:02 UTC

**Post body:**

Many of you seem to have liked my recent post ["A simple explanation of the key idea behind TurboQuant"](https://www.reddit.com/r/LocalLLaMA/comments/1s62g5v/a_simple_explanation_of_the_key_idea_behind/). Now I'm really not much of a blogger and I usually like to invest all my available time into developing Heretic, but there is another really cool new development happening with lots of confusion around it, so I decided to make another quick explainer post.

You may have noticed that the brand-new Gemma 4 model family includes two small models: **gemma-4-E2B** and **gemma-4-E4B**.

Yup, that's an "E", not an "A".

Those are neither Mixture-of-Experts (MoE) models, nor dense models in the traditional sense. They are something else entirely, something that enables interesting new performance tradeoffs for inference.

## What's going on?

To understand how these models work, and why they are so cool, let's quickly recap what Mixture-of-Experts (MoE) models are:

gemma-4-26B-A4B is an example of an MoE model. It has 25.2 billion parameters (rounded to 26B in the model name). As you may know, transformer language models consist of layers, and each layer contains a so-called MLP (Multi-Layer Perceptron) component, which is responsible for processing the residual vector as it passes through the layer stack. In an MoE model, that MLP is split into "experts", which are sub-networks that learn to specialize during training. A routing network decides *for each token* which experts are the most appropriate for the token, and only those expert networks are actually used while processing that token.

In other words, while an MoE model has many parameters, only a fraction of them are required to predict the next token at any specific position. This is what the model name means: gemma-4-26B-A4B has 26 billion (actually 25.2 billion) total parameters, but only 4 billion of those (actually 3.8 billion) are active during any single inference step.

The good news is that this means that we can do inference much faster than for a dense 26B model, as only 3.8 billion parameters are involved in the computations. The bad news is that **we still need to be able to load all 25.2 billion parameters into VRAM (or fast RAM),** otherwise performance will tank because we don't know in advance which parameters we'll need for a token, and the active experts can differ from token to token.

Now gemma-4-E2B is a very different beast: **It has 5.1 billion parameters, but 2.8 billion of those are embedding parameters.** Google claims that those parameters "don't count", so they say that there are only 2.3 billion *effective* parameters. That's what the "E2B" part stands for.

## Wut? Why don't the embedding parameters count?

If you have read or watched even a basic introduction to language models, you probably know what embeddings are: They are high-dimensional vectors associated with each token in the vocabulary. Intuitively speaking, they capture the "essence" of what a token sta

**Top comments:**

**u/sir_creamy** (score: 63):
> Appreciate all your contributions to the community. 

**u/xadiant** (score: 47):
> First of all, great explanation for laymen like me.

Okay, so... it's all a huge lookup exercise for each token. Instead of having this giga table, they split it between layers, as if a mixture of embeddings. 

What are the limits to that? Why not make a 100B 10E model, or use a hybrid approach with MoE?

Also in theory, training these models should be more efficient as we can offload embeddings to CPU, right?

  **u/-p-e-w-** (score: 42):
  > The limits are the extent to which this approach benefits model quality.

You can’t just shove all of the model intelligence into *static* parameters. It’s already a miracle that 50% is possible.

    **u/xadiant** (score: 11):
    > Makes sense. Still, very interesting if it can scale up because the e4b model is shockingly good based on my limited use. 

    **u/guiopen** (score: 7):
    > And could this be applied to MOE models too?

    **u/Down_The_Rabbithole** (score: 2):
    > The question becomes why isn't this applied to bigger models? Does it stop scaling after a certain point? Why isn't Gemma 4 31B "E18B" instead?

  **u/rabidcow** (score: 8):
  > There's a [video about DeepSeek's "Engram"](https://youtu.be/87Q8nf1XHKA) that was very conveniently timed.

I don't know offhand what their scale is, but apparently they do combine it with MoE.

    **u/keepthepace** (score: 7):
    > It evoked that video to me too, but I think these are different things (that could be combined!) engrams are about grouping tokens together, here we are talking about caching some of the knowledge associated with each token, if I am understanding correctly

  **u/DeepOrangeSky** (score: 5):
  > If I'm understanding this correctly, isn't the idea that the memory size savings on this are basically just purely to do with the vocabulary table of the model (with the ~250,000 words or so of vocabulary in the model's vocabulary).  So, if that would stay about the same size regardless of how big the LLM model was in total parameter size, then, then memory size savings would be the same whether it was an E2B model with 5 billion total parameters or a E97b model with 100 billion total parameters, saving the same 3 billion parameters worth of savings for each model, with that being a significan

**u/Awkward-Boat1922** (score: 19):
> You are pretty good at writing. 

**u/Mbando** (score: 13):
> Thanks for this. It’s the Engram paper in a production model then.

  **u/-dysangel-** (score: 8):
  > It sounds related, though from what I understood of the paper, engram handles multi token sequences too and so is a much bigger LUT? Either way this technique seems like it's going to enable us to focus params on intelligence, and engrams on knowledge

  **u/nebulous_mind** (score: 1):
  > It's the 1-gram paper in a production model.

This simple approach won't really work with (n&gt;1)-grams, given the combinatorial explosion.

**u/Firepal64** (score: 18):
> llama.cpp seems to shove the entire model, with embeddings, into VRAM when using -ngl 99. Are you trying to imply it'd be possible to leave the embeddings out of VRAM, and they just didn't implement it yet?

Edit; it's possible already. Check replies.

  **u/-p-e-w-** (score: 13):
  > When you pass that flag you’re putting everything in VRAM by design. Check the CLI arguments for more fine-grained control over which component goes where.

    **u/Firepal64** (score: 41):
    > Oh shit. `-ot "per_layer_token_embd\.weight=CPU"` puts Gemma 4 E4B's VRAM usage at 4.7GB (Q8!) with no discernible downside. This is actually sick as hell?

Thank you for pointing me towards this, makes me appreciate this model way more.

Edit: Do note that context is kind of expensive on this model. At f16, 8192 tokens of context costs 2.7GB(!). Furthermore, **the Unsloth Q8_K_XL has 5.6GB VRAM usage**, larger than the ggml-org Q8 quant used in my initial test. I suddenly feel like I have misled people a bit as to how great this model is with this optimization, but there are some memory savin

  **u/guiopen** (score: 2):
  > I'm facing the same problem.

**u/llama-impersonator** (score: 6):
> also interesting: n-gram embedding tables like in longcat-flash-lite

  **u/guiopen** (score: 1):
  > Yes, I would love an explanation on that

**u/Constant-Bonus-7168** (score: 3):
> Great explanation. Embedding tables are static after training, so they're perfect for lookup. Transformer layers need to stay dynamic for reasoning though.

**u/StyMaar** (score: 3):
> &gt; Now I'm really not much of a blogger

Hey that's a lie, I saw a link to your blog in your github bio :p

You could (should) definitely revive it given how clear your explanations are on both TurboQuant and this.

**u/sniperczar** (score: 5):
> Reminds me a lot of rainbow tables for password cracking. They require a huge amount of storage but the actual lookup doesn't require additional computation. This differs from pure cracking on lots of GPUs, and there are also hybrid approaches that seed permutations from variations of an initial dictionary or password collections from leaks.

**u/DeepOrangeSky** (score: 2):
> Regarding this, about the MoE models:

&gt;A routing network decides for each token which experts are the most appropriate for the token, and only those expert networks are actually used while processing that token.

I am curious if they tend to employ any tricks with this part.  As in, do they actually do a true 100% re-do from absolute scratch for every single token, or do they have some trick where the router is aware of which route is in the process of being used more heavily to increase its probability of routing down that route rather than it having an identical probability for every pos

  **u/geli95us** (score: 2):
  > I don't know if I understood your first point completely, but such a mechanism wouldn't be necessary, routers read the current value of the token to decide which expert to use, if the network needs context from previous tokens to make the decision of what expert to use, it can fetch that information using the attention mechanism.

For your second point, no, the only information an LLM has of its past inference is the tokens it actually wrote down. It seems like a good idea on paper, but it messes with training efficiency, people have experimented with this but nothing has worked well afaik. (D

    **u/DeepOrangeSky** (score: 2):
    > With the first thing, what I meant was, since he said that for each token the router had to try to decide which experts would be the most appropriate to use, I was wondering if there is some method where it weights the probabilities of using the experts it had already been using for a while into the inference it is in the middle of doing to skew in favor of those experts (if maybe part of the weakness with MoEs is if some unreliability happens if it switches to the wrong experts with new tokens as it churns its way through the tokens).  But seems like maybe the opposite is the problem.  As in,

**u/z_latent** (score: 2):
> Yes!! I had been keeping an eye on research around this like [this](https://arxiv.org/abs/2503.15798) and [this](https://www.arxiv.org/abs/2602.00398). It made me realize we'd soon have better models at near zero cost besides needing more storage, and as you mentioned, even that isn't a problem since you can keep them on disk (SSD) with minimal impact on speed.

I'm really happy Google released a model implementing it, and I hope we will see greater usage of these "very sparse" architectures moving forward.

**u/SkyFeistyLlama8** (score: 2):
> https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4

This post is a good addition to the linked article above by one of the DeepMind team.

https://huggingface.co/spaces/hesamation/primer-llm-embedding?section=what_are_embeddings?

This one goes into what embeddings are and how they're the first layer of an LLM's processing.

As mentioned in the post, essentially they're just lookup tables combining a token and an embedding vector for that token giving it some kind of semantic meaning. I can't figure out why no one's kept that LUT on disk instead of cramming everything into 

**u/Training-Respect8066** (score: 1):
> Very *subtle* self promotion in the first paragraph there.

  **u/eltonjock** (score: 3):
  > Eh. Let ‘em play the game like everyone else. 

**u/VoiceApprehensive893** (score: 1):
> amazing explanation

**u/IrisColt** (score: 1):
> Thanks for the insightful read!

**u/-dysangel-** (score: 1):
> those "embedding vectors" sound a lot like the engram stuff Deepseek V4 is going to have, except that the engrams can encode for sequences rather than just tokens, right?

**u/_kaidu_** (score: 1):
> For me this sounds like a mixture-of-experts that only uses bias terms and no linear weight matrix. Its surprising that this is so powerful.

**u/FrogsJumpFromPussy** (score: 1):
> I don't know what google counts in E2B but the model won't even load on my iPad. No issue to run qwen3.5 4b q6_k on 14tps. Yet e2b won't even work and Locally AI app which has support for gemma4 recommends m2 to run 😔

**u/Worried-Ad-7351** (score: 1):
> Thats quite interesting actually.

**u/Lakius_2401** (score: 1):
> You had my upvote at "complete dogshit in practice"

**u/jantaatihai** (score: 1):
> Hey, liked the way you've explained it. I recall reading your TurboQuant post too, and that was equally good.             

I only have basic understanding of how LLMs work behind the scenes, so I understood just half of it.       
**Is there any structured way/guide/list of topics, I should be following to understand LLM stuff clearly?**        
Currently, I am halfway through *LLMs from Scratch from Sebestian R*. Thanks.

**u/Logan_Maransy** (score: 1):
> Can I ask you why this wasn't done earlier? Or rather, from my understanding, isn't the residual stream changing the token embeddings for ALL the tokens in a sequence, and wouldn't per-layer embeddings destroy any residual stream nudging of the current token sequence?


Here's my current understanding of how text LLMs work. LLMs have a fixed text vocabulary, where each "word" in the vocabulary can simply be represented as a number. Ignoring RL training for a moment, the entire goal of the LLM is to guess the next "word" (number) given some sequence of already existing "words". For lots of reas

**u/CATLLM** (score: 1):
> Please post more stuff like this im learning so much

**u/AnOnlineHandle** (score: 1):
> Per-layer embeddings are incredibly powerful and overlooked IMO, where I suspect smarter conditioning is a massive breakthrough waiting to happen.

They worked incredibly with unet-based image diffusion models. I could exceed full finetuning concept detail accuracy for a larger library of concepts by training direct conditioning vectors for each layer of the frozen unet, each able to be learned in relative isolation without impacting the learning of the other concepts. Concepts which were impossible to achieve with textual inversion or even full finetuning when part of a large library of conce


---

### Thread 6: Don't sleep on the new Nemotron Cascade
- **Link:** https://www.reddit.com/r/LocalLLaMA/comments/1rzud2z/dont_sleep_on_the_new_nemotron_cascade/
- **Score:** 301 | **Comments:** 137 | **Date:** 2026-03-21 15:30 UTC

**Post body:**

While there has been a lot of discussion regarding the Nemotron Super family of models, I feel like the newest addition, the [Nemotron Cascade 2 30B-A3B](https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B) (which is \*not\* based on the Qwen architecture despite a similar size, it's a properly hybrid model based on Nemotron's own arch) has largely flown under the radar.

I've been running some evals on local models lately since I'm kind of tired of the "vibe feels" method of judging them. A combo that I quite like is HumanEval + ClassEval, simply because they're quick to run and complicated enough for most small models to still have noticeable differences. So, I gave mradermacher's IQ4\_XS quant for a spin.

On HumanEval, Cascade 2 achieved a whopping 97.6%, leaving both medium Qwen3.5 models in the rear window. Similarly, it obtained a respectable 88% on ClassEval.

I'm going to run some more tests on this model, but I feel it deserves a bit more attention.

**Top comments:**

**u/Shir_man** (score: 51):
> r/unsloth , we need a help here with a dynamic quant 

  **u/uhuge** (score: 2):
  > u/[danielhanchen](https://www.reddit.com/user/danielhanchen/) and his bro u/yoracale , we mean

**u/mantafloppy** (score: 15):
> Every time a new Nemotron come around, i really want to like it, then i try it....

I tried both :

 https://huggingface.co/mradermacher/Nemotron-Cascade-2-30B-A3B-GGUF Q8

https://huggingface.co/mlx-community/Nemotron-Cascade-2-30B-A3B-8bit

Both result, to one shot simple retro game, was real bad...

But maybe its the quant that not ready, because both model had a thinking block that looked way different, which is not normal...

They also repeated to themself to `Simplify` many time during their thinking...

    But for simplicity in this example...
    
    Simplify: We'll just provide dece

  **u/bjodah** (score: 2):
  > I think you might get better results if you first ask it for a PRD, then you discuss pros/cons of different choices, then you ask for a e.g. a 3-phase implementation plan. Iterate on it a bit, then you have it implement the first phase in a clean context. Rinse and repeat for remaining phases.

    **u/mantafloppy** (score: 16):
    > I get what you are saying, but i test every model with the same prompt.

If they `specialize` the model to the point it cant produce quality result without tailoring the prompt to its need, then its all reverse.

I dont work for the model, the model work for me.

I dont mind if it work better, if i prompt it better ; but its should not work badly on a normal, not elaborate promt.

**u/Finanzamt_Endgegner** (score: 40):
> Legend for posting this, i didnt even see this model was released!

**u/hp1337** (score: 32):
> Wow for pure coding this is insanely good for the size

  **u/Borkato** (score: 21):
  > Wait so this is a good replacement for Qwen3.5-35B-A3B for coding??

    **u/ilintar** (score: 19):
    > Might be ye.

    **u/Excellent-Baker-1177** (score: 4):
    > Its actually better. Much more impressive in my 2 days of agentic coding. Kinda feels minimax2.5 ish in terms of speed and “sureness”? Lol. Crazy considering im using some early bootleg quant while waiting on unsloth. The first time i feel like i dont have to compromise speed/capability on my rtx3090

    **u/soyalemujica** (score: 0):
    > 100% - trust me, I just gave it a try and it's beating 35B A3B by a HUGE margin, and also FASTER for some reason, it's more accurate with its thoughts as well. It 1 shot the 3 requests I have prepared for each MoE AI I tested, and this is the first time I see a 30B\~ beat it.

  **u/SocialDinamo** (score: 0):
  > Open code or something else? Im testing there, we will see how it goes! 

**u/LegacyRemaster** (score: 8):
> Hi Ilintar. I tested it on an RTX 6000 96GB at Q8. I tried to make it implement a change in an HTML file of over 2000 lines with maximum context enabled. It didn't work: it suffers from laziness, just like models of the same size prior to Qwen 3.5/next. No way to generate full html file.

  **u/-InformalBanana-** (score: 8):
  > Everytime i try out a nvidia model i find it is just benchmaxed, cant do anything worthwhile in real world... didn't try this one yet...

    **u/crantob** (score: 1):
    > These things are so high dimensional that we can observe huge differences between people who resonate with them and others who 'just don't get along'.

Reminds me of certain team members who failed to mesh sometimes.

    **u/Gullible-Bath9948** (score: 1):
    > The same for me. That's why I'm sticking to GLM-4.7-Flash. So far hasn't found any model up to 35B better than this one. 

    **u/Emergency-Associate4** (score: 1):
    > I think for that to work you’d have to have a “manager” agent that would keep track of the tasks and delegate small tasks (2-4) to a “junior” agent and implement feedback, memory etc. 

Correct me if I’m wrong

    **u/LegacyRemaster** (score: 0):
    > The problem is always "what do you use it for?" On long contexts, nvidia creates windows with 1M tokens and cannot copy and paste the same content, making + / - changes.

  **u/ixdx** (score: 2):
  > I also tested this on editing a 1000 line .vue file with a simple request to move a button from one block to another, and it failed every time, unlike the Qwen3.5/next models.  I used llama.cpp b8467 and mradermacher/Nemotron-Cascade-2-30B-A3B.Q8\_0.gguf.

**u/Thrumpwart** (score: 14):
> I’m having a very good experience with it. For my coding purposes it’s very good and very, very fast.

**u/lezioul** (score: 6):
> I've tried Q4 and Q6 quant and I found it less consistent than qwen3-coder 30b-A3B.

**u/MokoshHydro** (score: 21):
> I've tried it with Opencode and it simply doesn't work (MLX 4.0). Instead of producing output it just cites instructions from system prompt.

https://preview.redd.it/edrh6gksbfqg1.png?width=2104&amp;format=png&amp;auto=webp&amp;s=43099a780652fdfc1f6532b59288a31befc78f33



  **u/light100001** (score: 7):
  > i tried it with opencode + lmstudio, initially it was doing the same thing. then i updated the lmstudio runtimes after that it was responding. 
but still stopping midway during a task, half work done.. putting new code in wrong places in file..


    **u/yeah-ok** (score: 4):
    > I know this is half of a tangent but my dear lord LM Studio pisses me off with the way configuration is spread throughout the interface. Load a new model, but wupii, apart from certain configurations the overall the config is not loaded; it needs applying separately. It just ends up being non-deterministic nonsense that is hard to apply to actual daily work routine. I will take llama.cpp with a custom sh file to hold the configuration per model any day of the week! LM Studio still comes in handy for downloading models tho..

  **u/JayPSec** (score: -1):
  > ditto

  **u/quinncom** (score: 0):
  > Yeah, I had exactly the same problem running it in LM Studio. I’m connecting from Pi coding agent. 


**u/SocialDinamo** (score: 14):
> Your post made me take a look, just got it downloaded the q8 for the strix halo. Just over 50t/s generating on short test prompts. Im very happy with my three quick check tests! Quick hulusenation test, knowledge recall, combining lists all went well. And my god it didnt take 2-5k tokens to get an answer.   
  
 llama-server.exe -m models\\Nemotron-Cascade-2-30B-A3B.Q8\_0.gguf -ngl 99 -c 124000 -np 1 -b 8192 --host [0.0.0.0](http://0.0.0.0) \--port 8080 --temp 1.0 --top-p 0.95 --top-k 0 --min-p 0.05 --presence-penalty 0.0 --repeat-penalty 1.0 -fa on --jinja --chat-template-kwargs "{\\"enable\_

  **u/SpicyWangz** (score: 1):
  > How does it compare to qwen3-coder-next

    **u/SocialDinamo** (score: 1):
    > I didn’t have good luck with qwen3-coder-next. I needed a generalist and for that I was a big fan of gpt-oss-120b on the strix halo. But gpt-oss-120b isn’t perfect either. 

Qwen3.5 is WAY too verbose for me

  **u/cafedude** (score: 0):
  > Which q8 did you download?

**u/valx_nexus** (score: 3):
> The cascade approach is interesting because it's essentially doing at the architecture level what many of us have been doing manually - routing between models of different sizes based on task complexity.

  
I've been running a 5-model local setup where different models handle different cognitive roles (pattern recognition, reasoning, creativity, synthesis, emotional depth) and the orchestration layer decides which model(s) to engage for each subtask. Nemotron Cascade formalizes this inside a single system.

  
The question I have is whether the cascade's internal routing captures the same ben

**u/DistanceAlert5706** (score: 3):
> Faster than Qwen3.5 35b, but god it's terrible for agentic tasks...  
Goes into loops, doesn't follow system prompt instructions, timeouts on pretty simple queries, and idk just extremely unreliable.

While Qwen3.5 35b itself loves to go into the loops it's much better.  
Also Nemotron runs like 25% faster than Qwen3.5 35b but on actual agentic tasks it ends up \~3 times slower.

Maybe we need to wait and there are some bugs in llama.cpp implementation or this model just finetuned for benchmarks. Haven't tried coding yet.

  **u/crantob** (score: 1):
  > Ty for reporting.  I've been amazed how some redditors appear to love the same nemotron models that fall far behind the alternatives in my usage.

Can't ignore Nvidia though.  Gotta spend some time with this one too.

    **u/DistanceAlert5706** (score: 2):
    > Yeah, 100%. I had high hopes about this one. Again it's a new model and maybe there are some bugs in implementation or quant, but it's pretty unusable for general agent.

**u/Lorian0x7** (score: 27):
> The stupid trend of not trusting benchmarks is really affecting the critical thinking in this community.

  **u/DealingWithIt202s** (score: 22):
  > To extent that most new model training is “teaching to the test“ benchmarks aren’t really reliable, but that is true of everything these days. Employers can’t trust what a person puts on a resume reflects anything close to reality. You have to use intuition. But when one is 29% and the other is 97%, that is a useful datapoint in a decision framework is it not?

    **u/buttplugs4life4me** (score: 2):
    > I found it funny that recently someone posted their own tune with a headline like "87% on AIME" (exact percentage may vary). Go into the model card and at least they're honest enough to say that they finetuned it on that specific benchmark only. 

  **u/rulerofthehell** (score: 21):
  > Benchmaxxing is a real problem. 

    **u/BitterProfessional7p** (score: -3):
    > When there are hundreds of different benchmarks and they all agree it is difficult to argue this.

    **u/Lorian0x7** (score: -2):
    > Since everyone benchmax their models and there is a benchmark for everything, suddenly benchmarks becomes very reliable indicator or real comparable performances.


---

### Thread 7: Gemma 4 is good
- **Link:** https://www.reddit.com/r/LocalLLaMA/comments/1sb73ar/gemma_4_is_good/
- **Score:** 256 | **Comments:** 140 | **Date:** 2026-04-03 07:42 UTC

**Post body:**

Waiting for artificialanalysis to produce intelligence index, but I see it's good. Gemma 26b a4b is the same speed on Mac Studio M1 Ultra as Qwen3.5 35b a3b (\~1000pp, \~60tg at 20k context length, llama.cpp). And in my short test, it behaves way, way better than Qwen, not even close. Chain of thoughts on Gemma is concise, helpful and coherent while Qwen does a lot of inner-gaslighting, and also loops a lot on default settings. Visual understanding is very good, and multilingual seems good as well. Tested Q4\_K\_XL on both.

I wonder if mlx-vlm properly handles prompt caching for Gemma (it doesn't work for Qwen 3.5).

~~Too bad it's KV cache is gonna be monstrous as it did not implement any tricks to reduce that, hopefully TurboQuant will help with that soon.~~ \[edit\] SWA gives some benefits, KV cache is not as bad as I thought, people report that full 260K tokens @ fp16 is like 22GB VRAM (for KV cache, quantized model is another \~18GB @ Q4\_K\_XL). It is much less compacted than in Qwen3.5 or Nemotron, but I can't say they did nothing to reduce KV cache footprint. 

I expect censorship to be dogshit, I saw that e4b loves to refuse any and all medical advice. Maybe good prompting will mitigate that as "heretic" and "abliterated" versions seem to damage performance in many cases.

No formatting because this is handwritten by a human for a change.

\[edit\] Worth to note that Google's AI studio version of Gemma 26b a4b is very bad. It underperforms my GGUF with tokenizer issues :)

**Top comments:**

**u/NemesisCrow** (score: 52):
> So far, I only tested the Gemma 4 E2B model in Edge Gallery on my phone. 
This tiny model was the first ever that told me it hasn't enough context and therefore can't provide me an actual answer. Pretty impressive. 

  **u/estrafire** (score: 4):
  > 4b is so good too for its size, especially with audio and video understanding (bigger models only support video, no audio, same as Qwen 3.5).

Not sure if its on the model or the tools setup, but in Edge Gallery it first checks wikipedia for any information question I do even if it is something like "what does happen in this video"

    **u/maaya_yu** (score: 1):
    > which app do you use to chat with gemma 4 by uploading a local video?

**u/Pristine-Woodpecker** (score: 205):
> I don't understand how people can post these results when it's already confirmed the llama.cpp implementation is completely broken.

Are these all bot accounts?

Edit: The fix was just merged, but it obviously wasn't there when OP posted.

  **u/ambient_temp_xeno** (score: 49):
  > Like clockwork. I've learned over the years(!) to wait at *least* a day before even bothering to download quants.

  **u/Feztopia** (score: 35):
  > I'm running the e4 on my phone with Google's own app (not llamacpp) and I must say it's pretty good for it's speed and size. The biggest thing since Mistral 7b (which I also ran on my phone)

    **u/Dramatic-Chard-5105** (score: 4):
    > What kind of phone do you have and for what purposes to run e4? E4b means you need at least 2/3GB or ram if quantized and then you need the rest for managing OS resources no? I guess the speed would not be optimal for most use cases 

    **u/eidrag** (score: 1):
    > i tried both e2b and e4b, they're faster than qwen 3.5 2b, and understand better too

    **u/Pristine-Woodpecker** (score: 0):
    > That makes sense, Google's app probably doesn't have those bugs, but OP is talking about llama.cpp.

    **u/FoxTrotte** (score: 0):
    > Google has an app ?

  **u/petuman** (score: 12):
  > &gt; when it's already confirmed the llama.cpp implementation is completely broken.

at least on short casual chats unsloth gemma-4-26B-A4B-it-UD-Q4_K_XL doesn't seem completely broken on b8637 (first build with G4 support)

https://imgur.com/a/rHBkpz1

https://pastebin.com/uyL4e7Qu

    **u/trusty20** (score: 1):
    > The llama.cpp tokenizer was literally bugged on release, there are PRs being merged in as we speak, so you're pissing into the wind here.

    **u/Pristine-Woodpecker** (score: -7):
    > It breaks down completely if the convo goes a bit longer, but you can also get looping almost immediately. Anyway, the bug is known and understood by now, there's no point in arguing about this.

  **u/314kabinet** (score: 6):
  > Idk I got unsloth studio whose installer builds llama cpp from source and it runs perfectly fine on my 4090 

    **u/Pristine-Woodpecker** (score: -1):
    > The fixes were merged about 20 minutes ago, so depending on when you built it "it runs perfectly fine" would've been a huge overstatement.

It definitely wasn't fixed yet when OP posted.

It's possible all imatrix quants (e.g. unsloth) need to be redone :-/

  **u/One_Key_8127** (score: 17):
  > It is not "completely broken". It's tokenizer seems to be off, so it underperforms and its gonna be especially visible on spelling. It's probably gonna have a hard time counting R's in strawberry, but it produces very coherent and usable outputs.

    **u/kichael** (score: 2):
    > I ran into spelling issues with e4b where it tried to say something was misspelled and should be spelled a different way... When the suggestion was the same spelling. Q4_K_M

    **u/Pristine-Woodpecker** (score: -15):
    > LMAO at this response.

  **u/nickludlam** (score: 3):
  > I can understand what it looks like, but a commit landed in the llama.cpp repo that fixed it for me \~ 12 hours ago, and I was happily testing it in the \`llama-cli\` before I went to bed. It isn't beyond reason that  OP has had a working setup for a while now.

    **u/Pristine-Woodpecker** (score: 1):
    > See the timestamps, it was still completely broken 12h ago. The tokenizer didn't work. All the quants are being reuploaded now because they were broken too.

  **u/Oren_Lester** (score: 5):
  > I am using MLX, superb model

    **u/Pristine-Woodpecker** (score: 5):
    > Not with llama.cpp you aren't.

  **u/sky111** (score: 11):
  > Yes, they are. And none of them mentions that it's slow (11 t/s vs 60t/s with qwen 3.5, same hardware) and fits much less context than Qwen 3.5 in the same amount of VRAM (20k context vs 190k with qwen). So it's hardly even competitor if you are on a limited hardware.

    **u/One_Key_8127** (score: 7):
    > You talk about MoE? Gemma MoE is exactly the same speed at 20k context as Qwen3.5's MoE (35b a3b), both TG and PP, on Mac via llama.cpp. But you got a point on VRAM - VRAM usage on long context is a big downside and is gonna be very painful. At least till TurboQuant is properly supported by backends (and even then it's not gonna be as fast and efficient as Qwen3.5 or Nemotron). But it is still worth it probably since it produces more compact CoT and seems to be smarter overall.

    **u/Pristine-Woodpecker** (score: 0):
    > Performance seems OK in the sense that it's generating garbage output rather quickly, comparable to Qwen.

It's not obvious to me what in the architecture causes the KV cache difference?

  **u/Bingo-heeler** (score: 2):
  > I was able to use gemma on llama.cpp last night around 7 hours ago.


  **u/TapAggressive9530** (score: 4):
  > Maybe missing something but I tested a good chunk of yesterday with Gemma 4 .  Works fine with vLLM ( RTX 6000) + Claude code and I have a smaller model running on ollama on an RTX 5060Ti GPU .  Seems ok .  Never found any local models that have impressed me. Maybe one day …

    **u/sleepy_roger** (score: 2):
    > How long have you been in the space? Models have come so far in the last couple of years.

  **u/Feisty-Divide8081** (score: 1):
  > OP runs on Mac with mlx-vlm, so they don’t use llama.cpp

    **u/Pristine-Woodpecker** (score: 1):
    > They specifically mention llama.cpp including the performance on it, and having used GGUF.

  **u/jbuenojr** (score: 1):
  > I had opus 4.6 patch it for me to do testing yesterday. It’s pretty dang good 

  **u/trusty20** (score: 2):
  > Gemma posts have ALWAYS gotten these exuberant "omg it's the best thing ever" posts, and weird flip flopping between "look how good it did on this benchmark compared to the competition" then "benchmarks don't mean anything" when it doesn't score well against the competition.

Like I appreciate the fact we're talking about something that is free, they need to at minimum get some good press, so I don't get too focused on it, but they really really need to chill on the marketing posts.

  **u/ProfessionalSpend589** (score: -5):
  > My pet peeve is a model family name in the title and then OP talks only one of the smaller variants.

Such dishonesty… :(

**u/MinimumCourage6807** (score: 12):
> Gemma 4 31b is by far the best open weight model in finnish language i have tested with a big margin! And seems to be a solid performer in agent frameworks so i bet it gets to good use.

It is slow though, rtx 6000 pro gives around 30 tokens / s on llamacpp on q8. Cosidering minimax blasts around 80 and devstral 2 123b around the same 30 i hope future llamacpp versions will speed things up a bit.

  **u/jugalator** (score: 2):
  > Same in Swedish. It's incredible what they've done at this size. I struggle with Swedish often even with 70B models.

  **u/Arska_man** (score: 2):
  > 26b A4B also! I just tested it, and it beats all other models in this category!

    **u/MinimumCourage6807** (score: 1):
    > Yeah, i will give that a try also! There certainly is advantages of fast delivery if the quality drop is not too steep.

  **u/Ok-Drawer5245** (score: 2):
  > Even gemma3 completely destroys qwen models when it comes to European languages. For this I’m still using gemma3 in my production setup to this day. I intend to switch to gemma4 after testing/verification. Only use qwen if you are ok with only English/Chinese haha

  **u/One_Key_8127** (score: 1):
  > Interesting, are you sure about Devstral? Devstral Q8 won't fit on rtx 6000 pro, and I don't think Q4 can run at 30tps on rtx 6000 pro due to memory bandwidth limitations (it's 70+ GB, 6000 pro has \~1800GB/s max bandwidth, gives \~25tps in perfect conditions and realistically 15-20tps). Unless you somehow got multi-token prediction to work extremely well for your specific use case?

    **u/MinimumCourage6807** (score: 1):
    > Sorry my answer went bit two down on the chain. But   i have as said in the other answer two cards for the bigger models, 5090 and pro 6000. And the speed is been around the same as now the gemma4 which i was surpriced about. These numbers are not from bench so they definitely might be a bit off to way or another.


---

### Thread 8: Gemma 4 is a huge improvement in many European languages, including Danish, Dutch, French and Italian
- **Link:** https://www.reddit.com/r/LocalLLaMA/comments/1seo2rq/gemma_4_is_a_huge_improvement_in_many_european/
- **Score:** 266 | **Comments:** 58 | **Date:** 2026-04-07 06:26 UTC

**Post body:**

The benchmarks look really impressive for such small models. Even in general, they stand up well. Gemma 4 31B is (of all tested models):

\- 3rd on Dutch

\- 2nd on Danish

\- 3rd on English

\- 1st on Finish

\- 2nd on French

\- 5th on German

\- 2nd on Italian

\- 3rd on Swedish

Curious if real-world experience matches that.

Source: https://euroeval.com/leaderboards/

**Top comments:**

**u/anotheruser323** (score: 22):
> Non-professional translation is one of the things I think LLMs are actually good for. And google seems to be the best at it currently.

  **u/PreciselyWrong** (score: 6):
  > Google Translate, on the other hand, has become utter garbage compared to deepl and such

    **u/kellencs** (score: 1):
    > google translate now has advanced mode for some language pairs that is miles better than deepl btw

**u/ambient_temp_xeno** (score: 36):
> They really just gave us a SOTA translation model.

https://preview.redd.it/mcdmn5iftptg1.png?width=856&amp;format=png&amp;auto=webp&amp;s=5e18a0154ee49f902ee5ab3cc1fb1f90d1318007



  **u/pol_phil** (score: 11):
  > The translation evals can be misleading. After testing on some lower resource EU languages for scientific document translation, Gemma4 can lose coherence and start outputting random Chinese/Hindi/Arabic.

    **u/ambient_temp_xeno** (score: 2):
    > Is gemma 4 working right in what you're testing it on though? Gemma 4 was broken for me in llama.cpp in an insidious way until b8648

  **u/LoafyLemon** (score: 6):
  > 31B can actually roleplay too, without extra fine tuning or decensoring! I am actually amazed and baffled at the same time. It really sticks to character descriptions, so if you do DnD RP, villains are actually villainous. 


It also has a lot of knowledge about fantasy worlds, which is fun.

    **u/ambient_temp_xeno** (score: 2):
    > Gemma 4 really showed the weaknesses of the Chinese models in terms of their censorship (not just the Tiananmen square kind, the ass-slapping kind) and lack of datasets (compared to Google).

Mistral of course is even worse.

Thinking about it, Gemma 4 reminds me a lot of command-r plus (first release) although much smaller and smarter, of course.

  **u/Mark__27** (score: 1):
  > Is there a similar eval to this for Arabic/Hindi?

**u/That_Country_7682** (score: 10):
> 1st on finnish is actually wild. small models doing multilingual this well was not on my 2026 bingo card.

  **u/FinBenton** (score: 3):
  > I did some stuff on it in Finnish, its best Iw seen but it does make a lot of mistakes.

    **u/drallcom3** (score: 1):
    > &gt; but it does make a lot of mistakes

What kind of mistakes does it make? I'm curious. Does it get words wrong? Or is it more in long texts?

**u/drillmast3r** (score: 18):
> I tried to see if there had been any improvement in the Hungarian language compared to the previous model, but unfortunately, I don’t think so. And yet I was really looking forward to this model.

  **u/AssOverflow12** (score: 5):
  > Kár érte :(

    **u/drillmast3r** (score: 3):
    > Igen. A Gemma 3 már majdnem jó volt nekem, nagyon bíztam a 4-ben, dehát... 

  **u/Healthy-Nebula-3603** (score: 2):
  > Have you tested Aya 31b model ( trained for transactions) 

    **u/drillmast3r** (score: 1):
    > I will try it, thank you!

  **u/alamacra** (score: 2):
  > Are there any models that are any good in Hungarian, even of the really large ones?

    **u/PunnyPandora** (score: 3):
    > gemini is pretty good at them but it's not open source

    **u/drillmast3r** (score: 2):
    > I don't know the big ones. I’ve tried these: qwen3.5-35b - 27b, gpt-oss-20b, gemma-3-27b-it, glm-4.7-flash, and gemma-4-26b - 31b. In my opinion, gemma-3 produces the best Hungarian text out of these. But unfortunately, it’s not flawless either.

  **u/pip25hu** (score: 1):
  > Isn't it in second place for Hungarian, right after the latest Gemini?

**u/Middle_Bullfrog_6173** (score: 3):
> Doesn't reach the performance of the closed models on some of the smaller languages, but probably the open SOTA. Matches my experience in practice.

**u/phido3000** (score: 5):
> One day they will develop and AI that can understand Australian. 

**u/BrightRestaurant5401** (score: 2):
> from the local models gemma always has been the most impressive in Dutch, this time I no exception.  
I must say however that the most surprising to me is that Claude sonnet has the number 1 spot in this ranking.



**u/Icy-Degree6161** (score: 2):
> Is this about generic language interaction or tranlsation specifically?

In the translation space for these languages I found TranslateGemma and EuroLLM to be great.

  **u/Mashic** (score: 2):
  > In my tests for English &gt; Arabic translation. Gemma 4 beats translategemma out of the water.

**u/alexx_kidd** (score: 2):
> Greek.. worse 

**u/Mrfrednot** (score: 2):
> What model should I use for old greek? Is there one that is specifically good for old texts?

**u/madsheepPL** (score: 5):
> I don't see qwen 3.5 27B in there... It's been a top performer for me. 

  **u/Icy-Degree6161** (score: 18):
  > For Europ3an languages specifically?

    **u/madsheepPL** (score: 3):
    > yeah, translations from different languages into english. I'll have to review it though, going through this made me realize I need better benchmarks. 

    **u/FlamaVadim** (score: 3):
    > unfortunately qwen is not so great in translations 

  **u/FinBenton** (score: 6):
  > Atleast in Finnish language, qwen is just horrible compared to gemma.

  **u/Tenerezza** (score: 5):
  > Not for me that's for sure, sure can only verify Swedish, Norwegian and Danish to a  extent and every Qwen model is worse then even Gemma 3 when it comes to translations. And the few times i needed to use Finnish it's basically just crap. Gemma is by far much better at it and Gemma 4 is actually one of the best yet, behaves more or less the same as claude. The data from above basically verify what i see to here example just for the 3 languages 

[https://euroeval.com/leaderboards/Multilingual/mainland-scandinavian/](https://euroeval.com/leaderboards/Multilingual/mainland-scandinavian/)

**u/Mark__27** (score: 2):
> What about Arabic/Hindi?

**u/Available_Load_5334** (score: 1):
> https://github.com/ikiruneo/millionaire-bench

A benchmark using questions from the German version of "Who Wants to Be a Millionaire?".

**u/Mashic** (score: 1):
> For English to Arabic too. I have really been impressive of its accuracy over translategemma.

**u/bonobomaster** (score: 1):
> As a German this is still a good reminder to only talk to any LLM in that language it was trained on the most.

**u/Fluxx1001** (score: 1):
> Interesting Leaderboard. However it's strange that Mistral models are way behind in this benchmark, although they are explicitly trained on being multilingual European.

**u/HigherConfusion** (score: 1):
> Thanks. It confirm my own experience, that Gemma 3 12B, is still the best model at Danish, my machine can handle. It feels like Gemma 4 left a big gab between E4B and 26B-A4B. 


---

### Thread 9: 64Gb ram mac falls right into the local llm dead zone
- **Link:** https://www.reddit.com/r/LocalLLaMA/comments/1s9xvtb/64gb_ram_mac_falls_right_into_the_local_llm_dead/
- **Score:** 115 | **Comments:** 133 | **Date:** 2026-04-01 21:22 UTC

**Post body:**

So I recently bought a Mac (m2 max) with local llm use in mind and I did my research and everywhere everyone was saying go for the larger ram option or I will regret it later... So I did.

Time to choose a model:

"Okay,
- Nice model, Qwen3.5 35b a3b running 8 bit quant, speedy even with full context size. 
\-&gt; Performance wise it's mediocre especially for more sophisticated agentic use" 

"Hmm let me look for better options because I have 64 gbs maybe there is a smarter model out there. 
- Qwen3.5 27b mlx running at 4 bit quant (also full context size) is just the performance I need since it's a dense model. 
\-&gt; The catch is that, surprise surprise, it's slow so the agent takes up to 10 minutes just to create a folder structure"

So the dream would be like a 70 or 60b with active 9 or 7b model but there is none. 

Essentially, they sit in this like awkward middle ground where they are too big for consumer hardware but not powerful enough to compete with those "frontier" giants.

It seems like there really is this gap between the mediocre models (35/27b) and the 'good' ones (&gt;100b) because of that.. 

And my ram size (and performance) fits exactly into this gap, yippie 👍 

But who knows what the future might hold especially with Google's research on turbo quant

what do you guys think or even recommend? 

**Top comments:**

**u/grumd** (score: 62):
> qwen3-coder-next is actually 80b-a3b which is around the sweet spot you're looking for. Except it's probably not better than 35b-a3b, in benchmarks they're similar. 27B is the actual best quality model, but yeah it's slow for many setups.

As someone else said, getting a thunderbolt GPU will probably make you able to run 122B-A10B at a good Q4 quant, if you get a 3090 for example. My 16gb gpu + 64gb RAM were able to do IQ3_XXS or Q3_K_S but not Q4, unless you aren't using the machine for anything else

  **u/SkyFeistyLlama8** (score: 10):
  > Coder Next 80B A3B at Q4 is still my favorite model on my 64 GB machine but it's right at the edge of being usable. It's usually hitting 50 GB RAM usage so there's not much free memory left for the OS and actual applications to run.

It's only 3B active so it still runs fast even on a laptop.

I find the 80B a lot better than Qwen 3.5 35B at Q4 in terms of outright intelligence and for long context handling. When I'm coding with the 35B, I tend to stick to function-level output or simple refactoring whereas the 80B can also handle planning.

    **u/grumd** (score: 1):
    > Is that a Mac machine? Did you try running 27B instead?

  **u/putrasherni** (score: 3):
  > i prefer it over even 27B 

    **u/grumd** (score: 5):
    > For me nah, qwen3-coder-next is nowhere close to 27B or 122B

  **u/Mistercheese** (score: 5):
  > What’s your setup look like? What goes on gpu vram vs Mac unified memory? I was under the impression thunderbolt bandwidth would still be a limitation in speed if you spread across both. 

    **u/demon_itizer** (score: 10):
    > You need good CPU bandwidth as well as good GPU bandwidth for all the matrices they’re multiplying, but the CPU to GPU bandwidth requirement is not that big of a deal because they don’t synchronise that often

That is if you’re able to connect the TB gpu at all, which I am not sure if is possible on the M macs

    **u/grumd** (score: 8):
    > I'm using a Linux PC sorry mate :D

  **u/Skye_sys** (score: 2):
  > Oooh this seems interesting. 
But yeah I got similar results when I ran qwen3 next 80b when compared it to 3.5 35b... 
Money is tight atm but I didn't even thought of using a external gpu! Thanks! 

    **u/daaain** (score: 11):
    > Qwen 3 *coder* next is a different model, make sure you got the right one

    **u/CheatCodesOfLife** (score: 4):
    > Make sure it actually works before you buy anything!!

I don't see how you'd get nvidia drivers running on the M2 platform. Older Intel macs probably work, but I doubt the M* series would.

But another option is https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md

I wouldn't buy a GPU just for that though since last time I used it, performance was awful.

  **u/4xi0m4** (score: 2):
  > The dead zone is real but MoE models like Qwen3.5 35B A3B genuinely bridge it nicely on Mac. The A3B activation means only ~10B params hit memory at once, so Q4 fits comfortably in 64GB with room for decent context. Have you tried Qwen3.5 35B A3B at Q4_K_M? It is fast enough on M2 Max and the quality jump over 27B dense models is noticeable for coding tasks. The 80B A3B is better but tight on 64GB unless you are fine with IQ3_XXS which starts hurting quality.

  **u/Badger-Purple** (score: 1):
  > you ever run an external gpu on a mac? 

    **u/grumd** (score: 1):
    > no

**u/Hot-Section1805** (score: 53):
> Qwen 3.5 122b a10b IQ3_XXS (unsloth) works for me on M4Pro 64GB. It powers my OpenClaw instance, even supports the full context window (262K) with TurboQuant

  **u/Mistercheese** (score: 12):
  > This implies even the q3 of 122b beats out a q8 of 35b? For quality or speed or both?

    **u/Far-Low-4705** (score: 2):
    > yeah, this is what im curious about...

i can run 27b dense at Q4 at nearly the same speed i can run 122b at UDQ3\_K\_XL, so im really not sure what is better tbh.

    **u/[deleted]** (score: -8):
    > [deleted]

  **u/Skye_sys** (score: 7):
  > Already downloading! But we can't expect a mlx version of this soon do we? 

    **u/Temporary-Mix8022** (score: 15):
    > I run quite a few models on Mac.. and maybe it's just because I use LM Studio, maybe it has some quirks (although, I thought a lot of it is just llama.cpp under the hood)..


I've actually started preferring gguf as they support parallelism, shared KV cache and caching of the KV cache.


I've not noticed substantial performance differences. 


Hopefully someone will come educate me on the error of my ways 😅. But this was running oss20/120b and I actually choose gguf

  **u/Prudent-Ad4509** (score: 2):
  > I run exactly the same on 2x5090. It is faster when run on GPU of course, but the point is the amount of ram needed. KV cache works fine with default kv quantization, TurboQuant will just drop cache ram requirements even more. On a mac the system should take some ram as well unfortunately.

    **u/Temporary-Mix8022** (score: 3):
    > How do you run your 2x 5090s without running into the heat inflow/outflow issues? 


I'm doing quite a bit of ML work.. and due to my model size and compute requirements.. I want 2x 5090s as they're significantly better than 1x rtx6000, but I can't figure out the hardware side for cooling.


How do you configure them to stop them thermal throttling? I've even been looking at 1x water cooled.. but I'm not sure if I'd have space.


And do you need a specific CPU/MB setup to get 2x full speed PCIe5 lanes?

  **u/FerradalFCG** (score: 1):
  > do you need to increase the gpu memory above the default size?

  **u/Ell2509** (score: 1):
  > Mind if I ask how you are using turboquant? I can't find clear info on how to use it, yet. 

    **u/aijoe** (score: -1):
    > You havent asked a non local model? 

  **u/MoudieQaha** (score: 1):
  > Where are you using turboquant from?

    **u/Hot-Section1805** (score: 2):
    > Currently the llamacpp fork by TheTom

**u/StardockEngineer** (score: 7):
> Use Qwen3.5 35b at 4-bit.  

Qwen3-Coder-Next will fit at 4-bit, too.

But that chip just isn't going to be great.  Not enough oomph.

**u/AdamDhahabi** (score: 13):
> Add an external Nvidia GPU via Thunderbolt, that extra compute and VRAM will make a large difference. Enjoy Qwen 3.5 122b IQ4\_XS at a great speed. This is now possible with Tinygrad: [https://x.com/\_\_tinygrad\_\_/status/2039213719155310736](https://x.com/__tinygrad__/status/2039213719155310736)

  **u/Hot-Section1805** (score: 6):
  > April fools unfortunately (which my iPhone tried to autocorrect to April goons)

    **u/zdy132** (score: 2):
    > Wait, it's real! See you guys in a while, I've got things to try.

[supported GPU (AMD RDNA3+ or NVIDIA Ampere+)](https://docs.tinygrad.org/tinygpu/)

  **u/Mistercheese** (score: 3):
  > What’s your setup look like? What goes on gpu vram vs Mac unified memory? I was under the impression thunderbolt bandwidth would still be a limitation in speed if you spread across both. 

    **u/AdamDhahabi** (score: 3):
    > I'm not into Mac myself but started my local LLM journey with a gaming laptop and second GPU via Thunderbolt 3, it worked as expected. Now since a few days possible on Mac, but limited software support at the moment. It is still early tech.

    **u/SnooPaintings772** (score: 1):
    > Apple Silicon Macs (The M* series) don't support external GPUs, so that's not an option for the Mini).

**u/Technical-Earth-3254** (score: 12):
> Theres a "new" GPT OSS 88b by Nvidia that just got released. Idk how large it is in 4 bit, but it should be fine.

  **u/Outdatedm3m3s** (score: 5):
  > It runs great on my 64gb m5 pro

  **u/Ell2509** (score: 1):
  > That's pretty new! What software did you ring it with?

**u/PassengerPigeon343** (score: 6):
> This is exactly why I haven’t added a third 3090 to my system to get to 72GB. It is borderline feasible with some extra complexity for fitting, increased power needs, etc. but doesn’t really gain me anything major in terms off the models I can run. You really need to be in the 96-128GB or more before more compelling options start to open up. 

**u/florinandrei** (score: 4):
> What is the actual memory bandwidth of your system? M2 Max is theoretically capable of 400 GB/s, but actual systems may vary.

If you have at least 250 GB/s it should not be very slow.

&gt; It seems like there really is this gap between the mediocre models (35/27b) and the 'good' ones (&gt;100b) because of that..

Maybe, but the &gt;100b models are not god-mode either. You still don't get Opus-like performance from a 120b model.

  **u/Skye_sys** (score: 1):
  > Yes 400 GB/s is correct but I just think it's more of a compute issue rather then memory bandwidth 


---

### Thread 10: I tested as many of the small local and OpenRouter models I could with my own agentic text-to-SQL benchmark. Surprises ensured...
- **Link:** https://www.reddit.com/r/LocalLLaMA/comments/1s7r9wu/i_tested_as_many_of_the_small_local_and/
- **Score:** 220 | **Comments:** 65 | **Date:** 2026-03-30 13:55 UTC

**Post body:**

Last week I asked for some feedback about what extra models I should test. I've added them all and now the benchmark is available at [https://sql-benchmark.nicklothian.com/](https://sql-benchmark.nicklothian.com/)

I didn't say a lot about what the agent at the time, but in simple terms it takes an English query like "*Show order lines, revenue, units sold, revenue per unit (total revenue ÷ total units sold), average list price per product in the subcategory, gross profit, and margin percentage for each product subcategory*" and turns it into SQL that it tests against a set of database tables. 

It gets to see the query results and can modify it to fix issues, but with a limit to the number of debugging rounds it gets. 

The benchmark is deliberately short (25 questions) and fast to run (much less than 5 minutes for most models) so you can try different configurations etc, but it is tough enough to separate the best models from the others. 

I added the ability to run it yourself against your own server (thanks to the WASM version of Llama.cpp).

A few of the things I found interesting:

* The best open models are kimi-k2.5, Qwen 3.5 397B-A17B and Qwen 3.5 27B (!) 
* NVIDIA Nemotron-Cascade-2-30B-A3B outscores Qwen 3.5-35B-A3B and matches Codex 5.3
* Mimo v2 Flash is a gem of a model

I'd love to see some scores people get, as well as what I should change for v2!

**Top comments:**

**u/nickl** (score: 34):
> https://preview.redd.it/1sr7utvv07sg1.png?width=2392&amp;format=png&amp;auto=webp&amp;s=45d75782e4975b2ca3792db125c6ab4f320b2c1b

Some might find this chart useful too. 

  **u/LowMental5202** (score: 4):
  > How come gpt oss 20b is so high up? 
Tested it yesterday and got beyond a 100t/s on my 3090 and a relative big context window 

    **u/nickl** (score: 7):
    > I don't have a good explanation. The free version scored much better than the non-free version too. It's very odd!

It got very close on some questions it missed too

https://preview.redd.it/jau4lso647sg1.png?width=2444&amp;format=png&amp;auto=webp&amp;s=0dfda2eff3a5afd3d37515c772af16d47d45cfd1

    **u/IrisColt** (score: 1):
    > I still recommend it to colleagues, by the way.

  **u/Kahvana** (score: 5):
  > Didn't expect mistral small 2603 to score that high, neat!

**u/Adorable_Weakness_39** (score: 45):
> Qwen 3.5-27B is the goat. You can run it on a RTX 3090 at 40 tok/s. Everyone should be using it on their own hardware

  **u/nickl** (score: 13):
  > Yes, it's amazing. 

I'm very excited about Nemotron-Cascade-2-30B-A3B. A 3bit quant got within 2 points of Qwen 3.5-27B but some of the missed scores were timeouts on my GTX 1070 with 8GB!

    **u/RedParaglider** (score: 4):
    > I think it's crazy that it beat qwen3 coder next and the 122b model.. the 122b was what really surprised me.  

  **u/themaskbehindtheman** (score: 4):
  > What's the quant and context you run with?

  **u/LienniTa** (score: 5):
  > hey, what settings?

    **u/Adorable_Weakness_39** (score: 5):
    > llama.cpp, 99 gpu layers, 240k context, q8 kv\_cache.

The thing I've been working on auto-configures the llama.cpp server and downloads/detects the model model based on your hardware: [https://github.com/L-Forster/open-jet](https://github.com/L-Forster/open-jet) . I am wanting feedback for future development.

It's faster than ollama and is easier to set up than llama.cpp. 

**u/nicholas_the_furious** (score: 10):
> Yo! I love this and I am using your benchmark. Something I noticed, though, is you are controlling the temperature when you are passing in the calls. I know that in general low temp = better for tool calling, however I want to ask you to allow the model/provider to set their own sampling parameters!

Many models now need a temp higher than 0 or .1, especially reasoning models, in order to produce the best results. That may be why your reasoning versions of the smaller qwen models did worse than the non-thinking versions. I am running a q8\_0 version of the Nemotron 2 Cascade model and it is su

  **u/nickl** (score: 3):
  > These are all great points. Raised an [issue](https://github.com/nlothian/llm-sql-benchmark/issues/4) for myself, thanks

  **u/nicholas_the_furious** (score: 2):
  > The outstanding competitor was actually Apriel 1.6 15B. It got 20/25. Nailed everything below hard.

**u/Technical-Earth-3254** (score: 6):
> My local Qwen 3.5 27B Opus Distil in q4km with q8 kv scored 21/25, I'm very impressed.

  **u/nickl** (score: 2):
  > That's a great score. What TPS did it get? (It shows in the mouse over)

    **u/Technical-Earth-3254** (score: 6):
    > Around 20tps on a 3090 at 400W fully loaded in VRAM with \~60k context.

  **u/Killawatts13** (score: 1):
  > Mind posting which model? My main issue is small context windows. Using q8 kv helps but still bottlenecks me at around 80k context. I have a 4090 w/64gb ram

    **u/Technical-Earth-3254** (score: 1):
    > I was using Bartowskis quant, if that's what ur looking for. With our 24GB of VRAM we are limited to ~60k context when running on Windows. I have around 2GB of VRAM usage before using the model. Running on a setup without windows and a GUI would improve things quite a bit.

**u/MLDataScientist** (score: 6):
> Amazing website with interactive charts. Thanks for sharing!  
Do you have any SQL fine-tuned small models (&lt;=9B) to test this benchmark with? I think even Qwen3.5 4B with SQL data fine-tuning might reach 90%+.

  **u/nickl** (score: 2):
  > Yes that's my plan. 

I'd like to do a fine tune of 0.8B so it can run in-browser and actually be useful. 

But very happy to try other models if they exist already! 

You might have missed it but if you have llama.cpp/LMStudio/whatever you can run the benchmark yourself against any models you have locally

https://preview.redd.it/j9wyscak27sg1.png?width=2446&amp;format=png&amp;auto=webp&amp;s=eb3e0a4a06e863b8419ed4b4728d16206b1edb88



    **u/rm-rf-rm** (score: 2):
    > &gt; But very happy to try other models if they exist already!

Yes there are a bunch! they used to be posted on here every other week. Please add them, would be a very interesting result

**u/Evening_Ad6637** (score: 3):
> Great work OP!

I’ve tested GLM-4.5-Air and GLM-4.7


**GLM-4.7**

23/25 (failed Q9 and Q21)

$0.07

143s

- - -

**GLM-4.5-Air**

19/25

(Can’t remember the rest unfortunately)

- - -

Edit:


**Devstral-2512**

22/25 (failed Q6, Q8 and Q9)

$0.04

171s

- - -

**Qwen3-Coder-Next**

20/25 (failed Q2, Q3, Q9, Q10 and Q21)

$0.05

252s

That’s quite different from your result

**u/MD_Reptile** (score: 3):
> I'd sure like to be able to run kimi k2.5 locally but reqs are too crazy for the good quants - I'd really love a good lighter quant to use on reasonable hardware that somehow is just as smart! What's the closest thing you've found that doesn't require a data center in your basement?

  **u/nickl** (score: 6):
  > Qwen 3.5-27B is probably the best for most people to self host. I'm optimistic about Nemotron-Cascade-2-30B-A3B because runs (slowly) on my 8GB 1070, so I expect it will perform much better for most people who invest in some decent hardware. 



**u/SeaDisk6624** (score: 3):
> Can you test qwen 3.5 397b fp4 on open router please? it would be really interesting how it compares. thanks!

  **u/nickl** (score: 4):
  > Is there a FP4 version on OpenRouter?

If you have an OpenRouter key you can actually run it yourself. 

https://preview.redd.it/92w7vf2027sg1.png?width=2446&amp;format=png&amp;auto=webp&amp;s=21d8b9d4bd4bd0a689087a96fc99fd8d65465031



    **u/SeaDisk6624** (score: 1):
    > yes there is, I can't currently maybe next week.

**u/grumd** (score: 3):
> Omg this is so good, I want to run all my local models on this benchmark. I can't get it to run with my llama.cpp server though! I tried pointing it at http://localhost:8080/v1/messages, at http://localhost:8080/v1/completions, I can see the logs of a request coming to POST /v1/messages, but then I just see "Model failed to produce a tool call in 3 consecutive attempts" and there's no response from the model. Maybe something wrong with the setup because the same models work fine with OpenCode / Claude Code

EDIT: OK I got it to work by specifying /v1/chat/completions! Stay tuned for my benchma

  **u/nickl** (score: 1):
  > Yep. Small models often fail to do reliable tool calls.

Grammar mode helps sometimes but that isn't available in the web version. The write up has more details.

Edit: maybe coding agents just keep trying. 

    **u/chooseyouravatar** (score: 3):
    > https://preview.redd.it/a4yssubgn8sg1.png?width=2236&amp;format=png&amp;auto=webp&amp;s=5d6f9122586c1d5379d819ef10d0f303f7c13cc5

Love your site, thanks for your work. Talking about small models, it was fun to test Jan v1 (a 4b model, 2.5GB quantized) next to 30B and 9B models. It didn't perform badly (factually way better than Jan v2 8b)!  


**u/rm-rf-rm** (score: 2):
> Is the full db schema injected into the prompt? I couldnt find information on this

  **u/nickl** (score: 1):
  > No just the tables for the question.

If you click on a cell in a heatmap it shows you the exact trace for that question.


**u/abkibaarnsit** (score: 2):
> This is amazing. One small request, is it possible to add a SQL formatter to the `Model SQL` and `Canonical SQL` text areas

I want to compare the SQLs

  **u/nickl** (score: 1):
  > \&gt; is it possible to add a SQL formatter to the `Model SQL` and `Canonical SQL` text areas

I did a quick look for something easy but didn't find something. Didn't look too hard though and I agree it is needed. 

    **u/abkibaarnsit** (score: 1):
    > Raising a PR

**u/tmvr** (score: 2):
> Awesome site and benchmark setup!

Qwen3.5 27B Q4\_K\_L from bartowski gives the same 23/25 result you got with the same Q9 and Q21 failing.

EDIT:  
The Qwen3.5 27B IQ4\_XS result is also 23/25 with the same Q9 and Q21 failing, but while Q22 is a pass, it took 4 attempts.

**u/rm-rf-rm** (score: 2):
> Can you please address these 2 critical questions:

- Is the correctness check just row count, col count, column names and first row values? Not actually checking if all data is correct and equivalent to the canonical sql?
- is the result for a single test based on just 1 pass? is there any checking for stability/instability i.e. each question asked 3-5 times to see if it passes every time?

  **u/nickl** (score: 1):
  > &gt;Is the correctness check just row count, col count, column names and first row values? 

Yes this is correct. 

&gt;  
Not actually checking if all data is correct and equivalent to the canonical sql?

Correct. 

The justification here is that it's actually very hard to get the first row correct and other ones wrong. Here is a typical trace:

https://preview.redd.it/728vpnf26dsg1.png?width=1656&amp;format=png&amp;auto=webp&amp;s=d412bb4eba6b88859edb95861ac19247e75c269e

 When I was first building it I was using very small models. Passing the full result set blew out the context very quickl

    **u/rm-rf-rm** (score: 1):
    > &gt; I'd be interested if you have examples of any cases where this scoring gives the wrong result

I think the onus is on you to demonstrate that scoring with the full result vs first row is equivalent before settling for the latter as good enough.


---

### Thread 11: Gemma 4 31B vs Gemma 4 26B-A4B vs Qwen 3.5 27B — 30-question blind eval with Claude Opus 4.6 as judge
- **Link:** https://www.reddit.com/r/LocalLLaMA/comments/1scwos6/gemma_4_31b_vs_gemma_4_26ba4b_vs_qwen_35_27b/
- **Score:** 152 | **Comments:** 96 | **Date:** 2026-04-05 06:48 UTC

**Post body:**

Just finished a 3-way head-to-head. Sharing the raw results because this sub has been good about poking holes in methodology, and I'd rather get that feedback than pretend my setup is perfect.

**Setup**

* 30 questions, 6 per category (code, reasoning, analysis, communication, meta-alignment)
* All three models answer the same question blind — no system prompt differences, same temperature
* Claude Opus 4.6 judges each response independently on a 0-10 scale with a structured rubric (not "which is better," but absolute scoring per response)
* Single judge, no swap-and-average this run — I know that introduces positional bias risk, but Opus 4.6 had a 99.9% parse rate in prior batches so I prioritized consistency over multi-judge noise
* Total cost: $4.50

**Win counts (highest score on each question)**

|Model|Wins|Win %|
|:-|:-|:-|
|Qwen 3.5 27B|14|46.7%|
|Gemma 4 31B|12|40.0%|
|Gemma 4 26B-A4B|4|13.3%|

**Average scores**

|Model|Avg Score|Evals|
|:-|:-|:-|
|Gemma 4 31B|8.82|30|
|Gemma 4 26B-A4B|8.82|28|
|Qwen 3.5 27B|8.17|30|

Before you ask — yes, Qwen wins more matchups but has a lower average. That's because it got three 0.0 scores (CODE-001, REASON-004, ANALYSIS-017). Those look like format failures or refusals, not genuinely terrible answers. Strip those out and Qwen's average jumps to \~9.08, highest of the three. So the real story might be: **Qwen 3.5 27B is the best model here when it doesn't choke, but it chokes 10% of the time.**

**Category breakdown**

|Category|Leader|
|:-|:-|
|Code|Tied — Gemma 4 31B and Qwen (3 each)|
|Reasoning|Qwen dominates (5 of 6)|
|Analysis|Qwen dominates (4 of 6)|
|Communication|Gemma 4 31B dominates (5 of 6)|
|Meta-alignment|Three-way split (2-2-2)|

**Other things I noticed**

* Gemma 4 26B-A4B (the MoE variant) errored out on 2 questions entirely. When it worked, its scores matched the dense 31B almost exactly — same 8.82 average. Interesting efficiency story if Google cleans up the reliability.
* Gemma 4 31B had some absurdly long response times — multiple 5-minute generations. Looks like heavy internal chain-of-thought. Didn't correlate with better scores.
* Qwen 3.5 27B generates 3-5x more tokens per response on average. Verbosity tax is real but the judge didn't seem to penalize or reward it consistently.

**Methodology caveats (since this sub rightfully cares)**

* 30 questions is a small sample. I'm not claiming statistical significance, just sharing signal.
* Single judge (Opus 4.6) means any systematic bias it has will show up in every score. I've validated it against multi-judge panels before and it tracked well, but it's still one model's opinion.
* LLM-as-judge has known issues: verbosity bias, self-preference bias, positional bias. I use absolute scoring (not pairwise comparison) to reduce some of this, but it's not eliminated.
* Questions are my own, not pulled from a standard benchmark. That means they're not contaminated, but they also reflect my biases about what matters.

Happy to share

**Top comments:**

**u/corpo_monkey** (score: 70):
> What do you mean same temperature?  
Temperature is not universal, every LLM has its own preference, and different tasks require different temps.  
It's like every athlete must use the same shoe size.

  **u/anomaly256** (score: 14):
  > This was my thought too

  **u/Igot1forya** (score: 5):
  > You bring up a great point. What's the most accurate way to know the proper temp for each model, is there a baked in default that's optimal? I see these stats on the model cards but, is there a central repository or public database that defines each models ideal temp for each situation? I'd love to setup agents and have a model router just "know" this information and hit the ground running. Every day that goes by I find myself further and further kicking myself for drawing conclusions about models and it turns out I'm just using it wrong or picking the wrong tool for the job. 

    **u/Silver_Raspberry_811** (score: 4):
    > There isn't a great central repository for this unfortunately. Model cards on HuggingFace sometimes list recommended settings, but they're inconsistent — Google recommends temp 1.0 for everything on Gemma 4, which most people here disagree with for non-creative tasks. The reality is it's still trial-and-error per model per task type.

For a model router setup, you'd probably want to store per-model configs as metadata and load them dynamically. It's an infrastructure problem more than a knowledge problem — someone just needs to build and maintain the lookup table.

  **u/IrisColt** (score: 1):
  > I was about to write this.

  **u/Silver_Raspberry_811** (score: -13):
  > Fair point. I used the same temperature (0.7) and max\_tokens (2048) across all three models through OpenRouter's API defaults. You're right that each model has its own optimal inference settings — Gemma 4's docs recommend temp 1.0, Qwen has its own recommended params.

This is the same feedback I got on my Qwen batch two weeks ago and it's still a real limitation. Running through an API means I don't control the full inference config. The tradeoff is reproducibility (anyone can hit the same OpenRouter endpoint) vs optimality (each model running at its best). I'm planning a local rerun with mo

    **u/Far-Low-4705** (score: 13):
    > 2048 is not enough tokens 

**u/Sadman782** (score: 45):
> I don't know how you ran it, if you're running it locally using llama.cpp, use the b8660 llama.cpp build (more recent versions have a regression, another tokenization issue) and use --temp 0.3 --top-p 0.9 --min-p 0.1 --top-k 20

I am sure the 26B will do much better.

Also, Claude might favor better formatting etc., a boolean test is not good. Try the below prompt for the judge:

I am benchmarking many AIs in many tasks. You are a judge. Go through them question by question, not LLM by LLM. Go through each question and, for every question, give all AIs a score out of 10, and be sure to be fair

  **u/spaceman_** (score: 14):
  > Where do those values come from? Google recommends different values on their release page.

    **u/Sadman782** (score: 15):
    > by testing myself.  
It depends on your use case, for creative writing or other creative tasks temp 1 might be better, but for coding or tasks which require high accuracy especially if you use a quantized version these values yield much better results

  **u/StardockEngineer** (score: 5):
  > Is it still regressed?  You didn't link to an issue so I can't tell without digging.  Help us out :) 

  **u/Substantial-Ebb-584** (score: 3):
  > This. And Claude models alone as a judge is not a good test, since the results will be biased. Additionally claude will bias towards results that have answers similar with its programming.

  **u/Traditional-Gap-3313** (score: 2):
  > If you're using Claude code just spin up multiple subjects, one per query and every one of them will have a clean context. 


Jumbling in multiple evaluations introduces noise. If your rubric is correctly written, then not seeing other samples should be a benefit. If it needs to see all of them to judge, then you can't trust the grades and simply need to rank them.

  **u/Silver_Raspberry_811** (score: 2):
  > Great catches on both fronts. I didn't know about the llama.cpp regression — I ran these through OpenRouter's API so the inference stack is whatever their provider uses, which I can't control. That's a real limitation I should call out more explicitly.

Your judge prompt suggestion is interesting. I use absolute scoring (each response scored independently) specifically to avoid anchoring effects from seeing other responses. But your approach of scoring all responses to the same question together would give better relative ranking. Worth testing both and comparing. I'll try a side-by-side on th

  **u/PKJedi** (score: 2):
  > These tips work nicely, thank you!  
  
I have a couple offtopic questions, if you don't mind. Others are free to answer of course.  
\- Where do you recommend following LocalLLaMA related discussion? Here, the LocalLLM Discord, more?  
\- Which harness(es) do you recommend for coding with limited size models? OpenCode / Pi / Qwen / other? Any recent guides or important config tips?

    **u/RelicDerelict** (score: 10):
    > No Discord please!

    **u/Silver_Raspberry_811** (score: -9):
    > Hey thanks for stopping by, if you are interested in more: [https://discord.gg/2V3dg7hc](https://discord.gg/2V3dg7hc)

**u/Look_0ver_There** (score: 9):
> Would've been nice to also include Qwen 3.5 35B-A3B, since that is the closest counterpart to Gemma 4 26B-A4B

I'm also a little confused on how a "win" is chosen. 

  **u/Silver_Raspberry_811** (score: -4):
  > You're all right — the MoE-to-MoE comparison (Gemma 4 26B-A4B vs Qwen 3.5 35B-A3B) is the more meaningful matchup. I'll queue that as the next H2H. Dense-vs-dense and MoE-vs-MoE makes more sense than mixing them.

If you want to suggest which questions or categories matter most for that comparison, the Discord has a #suggest-models channel: [https://discord.gg/2V3dg7hc](https://discord.gg/2V3dg7hc)

**u/ambient_temp_xeno** (score: 44):
> LLM as judge = no thanks.  
It also depends how you're running Gemma 4 for the test. The new custom parser for gemma 4 in llama.cpp b8665 has fixed it for me. Before, it failed the test of just being given the image below. Now it solves it.

https://preview.redd.it/v9z0evuokbtg1.png?width=729&amp;format=png&amp;auto=webp&amp;s=43bb9b2b2e8869fe30eb05740c831431bf86b393

  **u/Silver_Raspberry_811** (score: 3):
  > Understood. LLM-as-judge has real limitations — verbosity bias, self-preference, positional effects. I use it because the alternative (human evaluation at scale) costs 100x more and I'm one person with no funding.

What I can say: Claude Opus 4.6 had a 99.9% parse rate across 1,067 prior judgments, scored in the 7.33 average range (not inflating everything to 9+), and when I compared its rankings against a full 10-model peer matrix, they correlated at 73%. Not perfect. Better than nothing.

The human baseline study is on the roadmap — comparing AI judge rankings against human preferences on th

    **u/Far-Low-4705** (score: 8):
    > Use programmatic scoring.

Write unit test cases for coding problems, test for exact matches in math problems, score multiple choice questions etc. all of these are better because they give objective results, not opinionated results 

  **u/high_funtioning_mess** (score: 1):
  > [Copying my reply from another comment]

I think same LLM that answered the question acting as a judge to review it own answer has some merits to it. It shows it how good the model is at judging/knowing whether the answer/thought process is correct or not.

    **u/Silver_Raspberry_811** (score: -1):
    > Interesting — hadn't seen braintwin before. I'll take a look. My eval engine is open-source (github.com/themultivac/multivac-evaluation) so if there's overlap or interoperability worth exploring I'm open to it.

**u/dubesor86** (score: 5):
> The token verbosity/inefficiency is a real killer during local use.

**u/Charming_Support726** (score: 4):
> Nice. My simple take is that both models are in the same ballpark. Qwen 3.5 has some advantage, but Gemma 4 is very good, especially in human communication - hard to measure with LLM-as-a-Judge. 

It feels like Gemma is just lacking a bit of tuning

  **u/Silver_Raspberry_811** (score: 0):
  > Good to see you again. Agreed — they're in the same ballpark. The Gemma 4 communication results are interesting because that's where subjective quality matters most and LLM-as-judge is weakest. Would love to hear if your experience running them locally matches these scores.

**u/spky-dev** (score: 4):
> 30 questions is an incredibly insignificant sample size. 

**u/Middle_Bullfrog_6173** (score: 3):
> The results look like you need harder tasks or a stricter rubric to really tell the difference between these. Do you have subscores you can use to tell how the differences come about in practice? E.g. completeness vs correctness vs writing quality or whatever.


Also are these full model weights or a particular quantization?

  **u/Silver_Raspberry_811** (score: 1):
  > All models ran through OpenRouter API — I don't control quantization. That's a known limitation and it's documented in the model\_metadata.json saved with each eval.

  **u/Silver_Raspberry_811** (score: 1):
  > If you have better suggestions or architecture design let me know will try it out in upcoming face as resources increases and so does the headcount.

  **u/Silver_Raspberry_811** (score: 1):
  > Yes — every judgment has five subscores: correctness, completeness, clarity, depth, usefulness. I can break those out per model. Quick version: Gemma 4 31B scored highest on clarity across the board. Qwen 3.5 27B scored highest on completeness but lowest on clarity. That tracks with the verbosity pattern.

Full per-question scores are on GitHub: [github.com/themultivac/multivac-evaluation/tree/main/data/GEMMA4-H2H-20260404](http://github.com/themultivac/multivac-evaluation/tree/main/data/GEMMA4-H2H-20260404)

**u/ShelZuuz** (score: 3):
> Would be good if these results have t/s, because 8.82 on both 26B-A4B and 31B doesn't make them equivalent.

**u/WetSound** (score: 3):
> In my tests 31B can go way deeper and complex than the others, before totally loosing it.

**u/3dom** (score: 3):
> Thank you! Very interesting test. Could be great to add Qwen 35B though.

  **u/daviden1013** (score: 3):
  > I'm interested to see how Qwen3.5 35B A3B does. It seems more meaningful to compare Gemma 31B vs. Qwen 27B (dense) and Gemma 26B A4B vs. Qwen 35B A3B (MoE).

  **u/Silver_Raspberry_811** (score: 1):
  > Hey, thanks for stopping by. If interested in more, join us here: [https://discord.gg/2V3dg7hc](https://discord.gg/2V3dg7hc)

**u/stddealer** (score: 3):
> Gemma3 31B is the first model that can successfully solve some riddles containing red herrings I like to test models with. Qwen3.5 27B gets fixated on the irrelevant information and gives a wrong answer, while Gemma4 manages to ignore it.

  **u/Silver_Raspberry_811** (score: -1):
  > Hey, thanks for stopping by. If interested in more, join us here: [https://discord.gg/2V3dg7hc](https://discord.gg/2V3dg7hc)

**u/Wildnimal** (score: 3):
> Good stuff. You should have added the 35-A3B from Qwen, since you compared a MOE model from Gemma there.



---

### Thread 12: Visual Guide to Gemma 4
- **Link:** https://www.reddit.com/r/LocalLLaMA/comments/1sbik5l/visual_guide_to_gemma_4/
- **Score:** 286 | **Comments:** 25 | **Date:** 2026-04-03 16:36 UTC

**Post body:**

source: [https://x.com/osanseviero/status/2040105484061954349](https://x.com/osanseviero/status/2040105484061954349)

[https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4)

**Top comments:**

**u/noage** (score: 21):
> Dense models of similar size are 'strong' compared to a slightly smaller moe model which is 'incredible?'

  **u/Big_Mix_4044** (score: 10):
  > "Incredible" is an attendance award.

    **u/DistanceSolar1449** (score: 1):
    > Gemma 4's architecture is not exactly super new and fancy. Sliding window attention aside, the rest of it is pretty much the exact same as most older models like gpt-oss or Qwen 3. GQA attention, dense/sparse FFN.

**u/RandomForestRobin** (score: 7):
> So the sliding window attention is just... pre-transformer/2017 LSTMs??? 

  **u/ShelZuuz** (score: 1):
  > Parallel vs. Sequential.

And a bunch of other stuff. But Parallel is all you need...

**u/garg-aayush** (score: 15):
> This is such a great blog. It is a definite must-read not just for understanding the Gemma4 model architecture but also decoder architectures in general. As with Maarten’s blogs, it is full of visualizations which makes it especially easy for beginners to follow and understand.

  **u/JollyJoker3** (score: 4):
  > Ok, ok, I'll read it

**u/llama-impersonator** (score: 3):
> bit odd to show lm_head on model arch diagrams for models with tied embeddings

  **u/CheatCodesOfLife** (score: 1):
  > And the arbitrary "amazing" / "incredible" on the MoE (in what way? it under-performs the dense model).
Makes me want to just **not** read the entire thing because it might I don't k now if it's actually accurate or slop.

**u/[deleted]** (score: 1):
> [deleted]

  **u/jacek2023** (score: 1):
  > It's in the post description 

    **u/abkibaarnsit** (score: 1):
    > Don't know why it's not visible to me :/
Apologie

**u/Caffdy** (score: 1):
> if all three inputs go through an embedding layer, why mention (Google in this case) E2B/E4B, when in reality it's more like 8B tokens?

  **u/aWildNacatl** (score: 4):
  > The per layer embedding doesn't need to be in vram. So its offloaded to flash.

    **u/Caffdy** (score: 2):
    > when you say flash, you mean the ssd?

**u/Gringe8** (score: 1):
> Its funny i just read this and it made me think to turn SWA on in kobold, massively reducing the vram required for the context.

**u/Altruistic_Heat_9531** (score: 1):
> kinda incredible that most of the transformer arch are stem from Google.   
Attn all u need - Google  
Switch Transformer (seed that will become MoE) - Google  
PLE - Google

  **u/crantob** (score: 1):
  > Money talks.

**u/Flaky_Direction3643** (score: 1):
> @grok what is ffnn in this image

  **u/TurnUpThe4D3D3D3** (score: 1):
  > FFNN stands for **Feed-Forward Neural Network**.

It's the standard dense layer in transformer decoder blocks that processes each token position independently after the attention mechanism. In the Gemma 4 diagram:

- **Tiny (E2B/E4B)** and **Dense (31B)** models use a standard FFNN
- **MoE (26B)** replaces the FFNN with a **Mixture of Experts** layer (the red block in the diagram) for better efficiency at scale

The FFNN typically consists of two linear transformations with a non-linear activation function (like GELU or SiLU) between them, applied position-wise to expand and then project the h

  **u/jacek2023** (score: 1):
  > why do you simulate grok guys ;)

**u/hustla17** (score: 1):
> I was playing around with the small models , and this article is just the cherry on top. I am learning so much thx!


---

### Thread 13: benchmarks of gemma4 and multiple others on Raspberry Pi5
- **Link:** https://www.reddit.com/r/LocalLLaMA/comments/1sdcdno/benchmarks_of_gemma4_and_multiple_others_on/
- **Score:** 227 | **Comments:** 50 | **Date:** 2026-04-05 19:17 UTC

**Post body:**

Hey all,

this is an update! A few days ago I posted to show the performance of a Raspberry Pi5 when using a SSD to let larger models run. Rightfully so, a few brought to my attention that the PCIe is faster than the USB3 connection I was using, so I bought the official HAT.

**Spoiler: As expected: Read speed doubled, leading to 1.5x to 2x improvement on tokens/sec for inference and text generation on models in swap.**

I'll repeat my setup shortly:

* Raspberry Pi5 with 16GB RAM
* Official Active Cooler
* Official M.2 HAT+ Standard
* 1TB SSD connected via HAT
* Running stock Raspberry Pi OS lite (Trixie)

*Edit: added BOM*

As per request, here the BOM. I got lucky with the Pi, they're now \~150% pricier.

|item|price in € with VAT (germany)|
|:-|:-|
|Raspberry Pi 5 B 16GB|226.70|
|Raspberry Pi power adapter 27W USB-C EU|10.95|
|Raspberry Pi Active Cooler|5.55|
|Raspberry Pi PCIe M.2 HAT Standard|12.50|
|Raspberry Pi silicone bottom protection|2.40|
|Rubber band|\~0.02|
|SSD (already present, YMMV)|0.00|

My focus is on the question: `What performance can I expect when buying a few standard components with only a little bit of tinkering?` I know I can buy larger fans/coolers from third-party sellers, overclock and overvolt, buy more niche devices like an Orange Pi, but thats not what I wanted, so I went with a standard Pi and kept tinkering to a minimum, so that most can still do the same.

By default the Pi uses the PCIe interface with the Gen2 standard (so I only got \~418MB/sec read speed from the SSD when using the HAT). I appended `dtparam=pciex1_gen=3` to the file "/boot/firmware/config.txt" and rebooted to use Gen3.

Read speed of the SSD increased from 360.18MB/sec (USB) by a factor of **2.2x** to what seems to be the maximum others achieved too with the HAT.

    $ sudo hdparm -t --direct /dev/nvme0n1p2
    /dev/nvme0n1p2:
     Timing O_DIRECT disk reads: 2398 MB in  3.00 seconds = 798.72 MB/sec

My SSD is partitioned to be half swapspace, half partition where I store my models (but that could be also anywhere else). Models that fit in RAM don't need the swap of course.

I benchmarked all models with this command, testing prompt processing (pp512) and text generation (tg128) at zero and (almost all) at 32k context:

    $ llama.cpp/build/bin/llama-bench -r 2 --mmap 0 -d 0,32768 -m &lt;all-models-as-GGUF&gt; --progress | tee bench.txt

Here are the filtered results in alphabetical order (names adjusted as GLM4.7-Flash was mentioned as the underlying deepseek2 architecture for example):

|model|size|pp512|pp512 @ d32768|tg128|tg128 @ d32768|
|:-|:-|:-|:-|:-|:-|
|Bonsai 8B Q1\_0|1.07 GiB|3.27|\-|2.77|\-|
|gemma3 12B-it Q8\_0|11.64 GiB|12.88|3.34|1.00|0.66|
|gemma4 E2B-it Q8\_0|4.69 GiB|41.76|12.64|4.52|2.50|
|gemma4 E4B-it Q8\_0|7.62 GiB|22.16|9.44|2.28|1.53|
|gemma4 26B-A4B-it Q4\_K\_M|15.70 GiB|15.88|6.45|3.06|1.66|
|gemma4 26B-A4B-it Q6\_K|21.32 GiB|10.95|5.31|2.76|1.59|
|gemma4 26B-A4B-it Q8\_0|25.00 GiB|9.22|5.03|2.45|1.44|
|gemma4 3

**Top comments:**

**u/ProfessionalSpend589** (score: 38):
> \&gt; If you have any questions just comment or write me. :)

How does the setup perform without a rubber band? I can procure a Pi 5, but with current prices I'd like to reduce the BOM even if it affects PP and TG a bit.

  **u/honuvo** (score: 25):
  > The rubber band is crucial and sadly is the most expensive part. Jokes aside, I only had a 2280 length SSD and didn't want to buy another one just so it fits the Pi better ;)

    **u/overand** (score: 4):
    > I had a desktop with a 22110 loaded into a 2280-sized PCI-E adapter - so not only was it rubber-banded to the card, but the back plate was removed - with the butt of the 22110 NVME drive sticking out of the back of the system. It was Extremely Professional™️

  **u/TheDailySpank** (score: 16):
  > I like how it really pulls the whole project together nicely. 

    **u/overand** (score: 3):
    > https://i.redd.it/g8p1b17yzitg1.gif



  **u/perkia** (score: 3):
  > You can *probably* remove one or even two screws from the hat to measurably drive the BOM down. Don't tell too many people.

  **u/IrisColt** (score: 0):
  > heh

**u/honuvo** (score: 11):
> Here the full (almost) unedited table for all tested models. I omitted a few columns in the main post to have an easier time to compare.

*Part 1: *

| model                           |       size |     params | backend    | threads | mmap |            test |                  t/s | 
| ------------------------------- | ---------: | ---------: | ---------- | ------: | ---: | --------------: | -------------------: |
| Bonsai 8B Q1_0                  |   1.07 GiB |     8.19 B | CPU        |       4 |    0 |           pp512 |          3.27 ± 0.00 |
| Bonsai 8B Q1_0                  |   1.07 GiB |  

  **u/honuvo** (score: 10):
  > *Part 2:*

|model|size|params|backend|threads|mmap|test|t/s|
|:-|:-|:-|:-|:-|:-|:-|:-|
|kimi-linear 48B.A3B IQ1\_M - 1.75 bpw|10.17 GiB|49.12 B|CPU|4|0|pp512|8.67 ± 0.01|
|kimi-linear 48B.A3B IQ1\_M - 1.75 bpw|10.17 GiB|49.12 B|CPU|4|0|tg128|4.24 ± 0.00|
|kimi-linear 48B.A3B IQ1\_M - 1.75 bpw|10.17 GiB|49.12 B|CPU|4|0|pp512 @ d32768|2.78 ± 0.01|
|kimi-linear 48B.A3B IQ1\_M - 1.75 bpw|10.17 GiB|49.12 B|CPU|4|0|tg128 @ d32768|0.58 ± 0.01|
|qwen35moe 122B.A10B Q2\_K - Medium|41.51 GiB|122.11 B|CPU|4|0|pp512|2.46 ± 0.00|
|qwen35moe 122B.A10B Q2\_K - Medium|41.51 GiB|122.11 B|CPU|4|0|tg128|1.05 ± 0

    **u/VicemanPro** (score: 5):
    > I'm really surprised by some of these. Thank you!

    **u/honuvo** (score: 2):
    > |model|size|params|backend|threads|mmap|test|t/s|
|:-|:-|:-|:-|:-|:-|:-|:-|
|gemma4 26B-A4B-it Q4\_K - Medium|15.70 GiB|25.23 B|CPU|4|0|pp512|15.88 ± 0.16|
|gemma4 26B-A4B-it Q4\_K - Medium|15.70 GiB|25.23 B|CPU|4|0|tg128|3.06 ± 0.00|
|gemma4 26B-A4B-it Q4\_K - Medium|15.70 GiB|25.23 B|CPU|4|0|pp512 @ d32768|6.45 ± 0.11|
|gemma4 26B-A4B-it Q4\_K - Medium|15.70 GiB|25.23 B|CPU|4|0|tg128 @ d32768|1.66 ± 0.01|
|gemma4 26B-A4B-it Q6\_K|21.32 GiB|25.23 B|CPU|4|0|pp512|10.95 ± 0.32|
|gemma4 26B-A4B-it Q6\_K|21.32 GiB|25.23 B|CPU|4|0|tg128|2.76 ± 0.03|
|gemma4 26B-A4B-it Q6\_K|21.32 GiB|25.23 B|CPU|4

  **u/Steve_OH** (score: 3):
  > What was the tokens/s on bonsai?

    **u/honuvo** (score: 2):
    > Oh, table was shifted and showed the results in the wrong column. PP 3.27 TG 2.77, but that is the first row of the main posts table also :)

**u/exaknight21** (score: 4):
> PrismML’s Llama Fork likely needs tweaking for the Pi 5. I’m 100 miles away from mine and I’m itching to try it out. The 8B packs a punch.

  **u/honuvo** (score: 1):
  > No doubt it's a good and interesting model, that's why I tested it. I'm not good enough to know where even to begin improving the code for the Pi5 though. If you manage to tweak it, I'd be happy to test :)

**u/DevilaN82** (score: 4):
> Can you please test mmaping SSD so it does not need to use SWAP and reads weights from disk directly?

  **u/honuvo** (score: 1):
  > I did test that, but results were worse. Maybe I'll add one or two comparisons to the table to show, but takes time :)

    **u/DevilaN82** (score: 1):
    > I remember you doing tests with SSD connected to usb3.0. I am curious how much slower PCI connected SSD is vs using SWAP file on this very SSD.

**u/akavel** (score: 2):
> I'd be really curious of results for `gemma4 26B-A4B-it` at q6 and q4 (any), and similarly for `Qwen3.5 35B.A3B`.

  **u/honuvo** (score: 2):
  > Downloading now. Will add the results when they're done, but can take 1-2 days (depending on when I get to it and because the Pi isn't that fast.)  
But I looked at my old results (with inferior memory bandwidth) and had 2-3x the performance with Qwen3.5 35B.A3B Q4\_K\_M in comparison to the Q8, so looks promising.

  **u/honuvo** (score: 1):
  > All online now :)

    **u/akavel** (score: 1):
    > Thanks!! also for the ping :) interesting, I thought the speeds would be faster; and didn't expect this to differ between qwen and gemma.

**u/Sliouges** (score: 2):
> That rubber band... that holds everything together.

  **u/honuvo** (score: 1):
  > I knew everybody would appreciate it. I wouldn't have been able to continue without it :P

**u/JoeS830** (score: 2):
> Fun stuff. So at this point how far are we from putting together our own local conversational AI that we can talk to at home and get high quality voice responses without sending anything to the cloud? Is this already doable by piecing existing elements together?

  **u/honuvo** (score: 3):
  > I'm nowhere near that currently, but I think that's already been done. I know of [this project ](https://github.com/dnhkng/GLaDOS) but don't know the hardware requirements.

    **u/JoeS830** (score: 2):
    > Thanks. That sounds pretty specific though: “the first steps towards a real-life implementation of the AI from the Portal series by Valve”. I’d really like to be able to run gemma4 locally, and have a local “always listening for keyword” routine running, and then having any gemma4 text output sent back to me with an open weights speech model. It feels like we’re super close to being able to do that with semi affordable hardware. Fun times!

**u/Nice_Cellist_7595** (score: 2):
> Solid work

**u/PiratesOfTheArctic** (score: 1):
> You're running models higher than my laptop does! Going to go through your list now 😜

**u/AnonLlamaThrowaway** (score: 1):
> With the backend being the CPU, it makes me wonder if Vulkan would make this any faster

  **u/honuvo** (score: 1):
  > Hm... damn. Now I'm curious too. Memory speed is the same (shared RAM/VRAM) but maybe using the Broadcom VideoCore's processing is faster? Maybe I'll check.

  **u/honuvo** (score: 1):
  > Tested it. Vulkan build is not working with the Pi5.  
Getting `ggml_vulkan: Error: Shared memory size too small for matrix multiplication.`

See also: [https://github.com/ggml-org/llama.cpp/issues/9801](https://github.com/ggml-org/llama.cpp/issues/9801)

    **u/AnonLlamaThrowaway** (score: 1):
    > Interesting.

I know the GPU is definitely not the focus of the Pi boards, but it would definitely be interesting to see it used as a pseudo-NPU of sorts, if only to find out whether it can actually accelerate workloads...

**u/starstripper** (score: 1):
> Is there a way to do something similar if you’re using the ai hat 2?

  **u/honuvo** (score: 1):
  > Isn't the AI HAT 2 only for image processing?

    **u/starstripper** (score: 1):
    > I know it does that better than LLM but it does have 8gb dedicated memory, I dont know if it has to be a special model compiled to take advantage of the npu though…

**u/Potential-Net-9375** (score: 1):
> Sorry to ask, but do you have data on Qwen3.5 9B q4_k_m? This is significantly smaller in size than q8, and with a proper harness still works very well

  **u/honuvo** (score: 2):
  > Don't be sorry :) I just added it to the table in the main post. Surprisingly it starts worse as the Q8 but with more context performs better. This is all in RAM btw (Q8 as well as Q4), so I guess unpacking the quants takes it's toll in the beginning but with deeper context the smaller footprint makes it work better? I'm just guessing here, sorry.

**u/NitsTheTits** (score: 1):
> Hey can you provide parts and cost breakdown of the spec? :)

  **u/honuvo** (score: 1):
  > Sure, added it to the main post :)


---

### Thread 14: Comparing Qwen3.5 vs Gemma4 for Local Agentic Coding
- **Link:** https://www.reddit.com/r/LocalLLaMA/comments/1sd0be8/comparing_qwen35_vs_gemma4_for_local_agentic/
- **Score:** 138 | **Comments:** 93 | **Date:** 2026-04-05 10:34 UTC

**Post body:**

[Gemma4](https://deepmind.google/models/gemma/gemma-4/) was relased by Google on April 2nd earlier this week and I wanted to see how it performs against Qwen3.5 for local agentic coding. This post is my notes on benchmarking the two model families. I ran two types of tests:

* **Standard llama-bench benchmarks** for raw prefill and generation speed
* **Single-shot agentic coding tasks** using [Open Code](https://opencode.ai) to see how these models actually perform on real multi-step coding workflows

**My pick is Qwen3.5-27B which is still the best model for local agentic coding** on an 24GB card (RTX 3090/4090). It is reliable, efficient, produces the cleanest code and fits comfortably on a 4090.

|Model|Gen tok/s|Turn(correct)|Code Quality|VRAM|Max Context|
|:-|:-|:-|:-|:-|:-|
|Gemma4-26B-A4B|\~135|3rd|Weakest|\~21 GB|256K|
|Qwen3.5-35B-A3B|\~136|2nd|Best structure, wrong API|\~23 GB|200K|
|Qwen3.5-27B|\~45|1st|Cleanest and best overall|\~21 GB|130K|
|Gemma4-31B|\~38|1st|Clean but shallow|\~24 GB|65K|

&gt;**Max Context** is the largest context size that fits in VRAM with acceptable generation speed.

* MoE models are \~3x faster at generation (\~135 tok/s vs \~45 tok/s) but both dense models got the complex task right on the first try. Both the MoE models needed retries.
* Qwen3.5-35B-A3B is seems to be the most verbose (32K tokens on the complex task).
* Gemma4-31B dense is context-limited in comparison to others on a 4090. Had to drop to 65K context to maintain acceptable generation speed.
* None of the models actually followed TDD despite being asked to. All claimed red-green methodology but wrote integration tests hitting the real API.
* Qwen3.5-27B produced the cleanest code (correct API model name, type hints, docstrings, pathlib). Qwen3.5-35B-A3B had the best structure but hardcoded an API key in tests and used the wrong model name.

You can find the detailed analysis notes here: [https://aayushgarg.dev/posts/2026-04-05-qwen35-vs-gemma4/index.html](https://aayushgarg.dev/posts/2026-04-05-qwen35-vs-gemma4/index.html)

Happpy to discuss and understand other folks experience too.

**Top comments:**

**u/aldegr** (score: 43):
> Assuming you're using the latest llama.cpp, try testing Gemma 4 with https://github.com/ggml-org/llama.cpp/blob/master/models/templates/google-gemma-4-31B-it-interleaved.jinja.

  **u/garg-aayush** (score: 18):
  > No, I ran these tests on Friday (April 3rd) morning. Thanks, I will try to run them again with this model template. 

    **u/ResearchCrafty1804** (score: 7):
    > Please update us with your findings, if the latest llama.cpp and chat template make a difference to Gemma4 in local agentic coding

  **u/Intelligent_Lab1491** (score: 8):
  > How do I use this?

    **u/xanduonc** (score: 11):
    > `--chat-template-file "/models/google-gemma-4-31B-it-interleaved.jinja"`

    **u/garg-aayush** (score: 8):
    > You need to provide the chat template as one of the flags during the server launch. 

  **u/grumd** (score: 2):
  > What does this template do?

    **u/aldegr** (score: 7):
    > The template preserves reasoning between tool calls, in adherence to the Gemma 4 prompt formatting guide (https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4#managing-thought-context). The original templates strip this reasoning, contradicting the guide. It does require a recent llama.cpp and an agent harness that sends back reasoning traces. Pi and OpenCode should work out of the box.

  **u/riceinmybelly** (score: 2):
  > Are lama.cpp templates also working in lm studio?

    **u/aldegr** (score: 2):
    > This requires a particular implementation in llama.cpp to build the model turn, so I highly doubt it will work with LM Studio.

**u/Barry_22** (score: 15):
> So Qwen-3-27B is still a champ?

  **u/garg-aayush** (score: 7):
  > Yes, I still feel qwen3.5 is better for coding. Not just the results but also the size and speed is way better suited for 24GB cards. 

    **u/danf0rth** (score: 2):
    > I have similar feeling, i found Qwen really more intelligent when writing code, e.g. it written ipynb files correctly, while Gemma 4 create not working notebook. 

But on the other side, i use rtk to compress context a bit, and i found that with the same agents, Qwen ignores prefixing all commands with rtk, but Gemma for some reason make it more frequently, feels like it following instructions better in this case.

Also i use HIP ROCm on Windows (7900XTX), and observed that Gemma has performance degradation. It prints response and every n seconds it slows down again and again, until there is r

    **u/ixdx** (score: 1):
    > For 2x16GB GPUs, the Qwen3.5-27B-Q4\_K\_M/L is also better, as it fits within a 128K context. The Gemma-4-31B context takes up more VRAM.

**u/donhardman88** (score: 16):
> Model choice is important, but for agentic coding, the retrieval layer is actually the bigger variable. You can have the best model in the world, but if the agent is just using flat semantic search to find code, it'll still struggle with complex cross-file dependencies.

I've found that the most successful local agent setups are the ones using a structural knowledge graph via MCP. It allows the agent to actually 'navigate' the project architecture rather than just guessing based on embeddings. It makes a huge difference in how the agent handles refactoring across multiple files.

  **u/Potential-Leg-639** (score: 19):
  > Tell us more about „using a structural knowledge graph via MCP“, please?

    **u/donhardman88** (score: 24):
    > Sure! The core idea is that instead of treating your code as a collection of text chunks (which is what standard RAG does), you use AST parsing (via tree-sitter) to map out the actual symbols, function definitions, and cross-file dependencies. This creates a structural knowledge graph.

By exposing this graph through an MCP (Model Context Protocol) server, the AI agent doesn't just 'search' for a similar string—it can actually 'navigate' the codebase. For example, it can find a function definition and then immediately see every other file that imports or calls that specific function, regardles

  **u/cleverusernametry** (score: 6):
  > No. Boris Cherny himself says "agentic search" - simply grep and glob outperform rag for coding. That's all that Claude code uses. 

Unless you're a poor engineer or vibe coder, you're codebase will follow good/standard folder structures for your language + have good docs. That's all that the model needs to get the right context

    **u/donhardman88** (score: 2):
    > he problem with the 'just grep it' argument is that it assumes a human is steering the ship. If you're the one telling the agent exactly which symbols to grep on every single request, then you're the one doing the work, not the AI.

I've spent a lot of time with Claude Code, and the 'grep loop' is exactly where it falls apart. The agent greps, gets 20 results, reads 5 of them, gets overwhelmed by the noise, and then starts hallucinating or ignoring the original intent just to finish the task. It's a classic case of 'context collapse.'

Grep doesn't avoid the context window problem; it actually

    **u/IrisColt** (score: 1):
    > Today I learned that my Diogenes syndrome-esque folders mean that I'm a poor engineer, whatever that means, heh

  **u/gyzerok** (score: 1):
  > Can you elaborate your specific setup?

    **u/donhardman88** (score: 10):
    > My setup is really focused on minimizing the 'noise' that usually kills agentic coding. I've tried almost everything—Claude, Codex, and various other tools with open-source models—and the biggest realization was that most clients aren't actually focused on retrieval quality. They just give you a basic search and expect the LLM to figure it out, but you really have to tune the retrieval layer to get a professional outcome.

Currently, I'm mostly using **GLM-5 with MiniMax** (I actually have a sub on [ollama.com](http://ollama.com) for this) and I'm really happy with the performance. The core dr

**u/MrMisterShin** (score: 3):
> I took a look at your link. Thanks for including the actual duration time in your analysis. 

Tokens per second is not the full story, when you have models that “think extensively” or require more tool calls etc than other models to complete a task with good quality. 

  **u/garg-aayush** (score: 1):
  > Yup, I observed the same thing especially with MOEs they are blazing fast in comparison to dense ones but they are more likely to use more api and tall calls along with every now and then issue of infinite loop. They seem less precise for coding. Maybe there is way to get around this with better prompting and mid-feedback. 


**u/Voxandr** (score: 3):
> This is my findings too. Qwen Still better than Gemma in agentic insturction following.

**u/kiwibonga** (score: 2):
> This post made me realize I dreamed about local model benchmarks last night. I don't remember any specifics but I was so excited about this graph with red and green balls.

  **u/gyzerok** (score: 5):
  > Maybe you were dreaming about balls, not benchmarks?

    **u/2muchnet42day** (score: 1):
    > Why not both?

**u/mmontes11** (score: 2):
> Similar findings with an RTX PRO 4000 SFF, also 24GB:

https://github.com/mmontes11/llm-bench

**u/mapsbymax** (score: 2):
> Great comparison. One thing I've noticed running agentic tasks locally - the MoE speed advantage is deceptive for agent loops. The 3x faster generation looks great on paper, but when the model needs 2-3 retry cycles because it got something wrong, you end up slower than the dense model that nailed it first try.

The TDD observation is really interesting too. I've tried multiple local models and none of them actually do proper red-green-refactor even when explicitly asked. They all write the implementation and tests together. Would love to see someone crack that with better system prompts or fi

  **u/garg-aayush** (score: 1):
  > I also have a hunch that if we can build a good feedback loops and automatic test runs, then MOEs will start performing at par with dense models. I remember reading one of the researcher tweets where they got the local models be it MOE or dense work better by having an automatic feedback loop which ensured maximum of 2 tries per problem with the error in some smart way becoming a feedback signal for the 2nd try.

**u/Eyelbee** (score: 1):
> How do you get 130k context? 

  **u/garg-aayush** (score: 4):
  > I used q8 quantization for KV cache. This fits well on 4090.
Actually, I have also seen folks use q8 for K and turbo3 for V that should even help you get more context in. 

**u/vasimv** (score: 1):
> Did they actually test and debug the code they wrote? In my experiments writing simple android apps with local models, the most challenging part was debugging the code to ensure it met technical requirements.

**u/D2OQZG8l5BI1S06** (score: 1):
> Gemma 4 prompt processing is almost twice faster than Qwen for me (75 vs 40 t/s), it does help a lot too.

  **u/garg-aayush** (score: 1):
  > That is interesting. What is your setup that you are able to get 75t/s for Gemma but 40 for Qwen?

    **u/D2OQZG8l5BI1S06** (score: 1):
    > Latest llama.cpp with -cmoe, so all MoE weights on CPU

**u/Constant-Bonus-7168** (score: 1):
> Qwen3.5-27B handles multi-turn corrections cleanly — it can accept feedback and adjust without hallucinating. That's more valuable for agentic work than raw single-shot accuracy.

**u/twanz18** (score: 1):
> Both are solid for agentic coding. In my experience Qwen3.5 handles longer context tasks better when you are running multi-step workflows. Gemma4 is faster on shorter prompts though. If you are running these locally and want to connect them to something like Claude Code or other agents remotely, check out OpenACP. It lets you bridge any coding agent to Telegram or Discord so you can trigger tasks from your phone. Open source, self-hosted. Full disclosure: I work on it.

**u/Rich_Artist_8327** (score: -5):
> Reddit is full of Qwen marketing posts. 


---

### Thread 15: Is 1-bit and TurboQuant the future of OSS? A simulation for Qwen3.5 models.
- **Link:** https://www.reddit.com/r/LocalLLaMA/comments/1sadadw/is_1bit_and_turboquant_the_future_of_oss_a/
- **Score:** 138 | **Comments:** 79 | **Date:** 2026-04-02 10:09 UTC

**Post body:**

Simulation what the Qwen3.5 model family would look like using 1-bit technology and TurboQuant. The table below shows the results, this would be a revolution:

|Model|Parameters|Q4\_K\_M File (Current)|KV Cache (256K) (Current)|Hypothetical 1-bit Weights|KV Cache 256K with TurboQuant|Hypothetical Total Memory Usage|
|:-|:-|:-|:-|:-|:-|:-|
|Qwen3.5-122B-A10B|122B total / 10B active|74.99 GB|81.43 GB|17.13 GB|1.07 GB|**18.20 GB**|
|Qwen3.5-35B-A3B|35B total / 3B active|21.40 GB|26.77 GB|4.91 GB|0.89 GB|**5.81 GB**|
|Qwen3.5-27B|27B|17.13 GB|34.31 GB|3.79 GB|2.86 GB|**6.65 GB**|
|Qwen3.5-9B|9B|5.89 GB|14.48 GB|1.26 GB|1.43 GB|**2.69 GB**|
|Qwen3.5-4B|4B|2.87 GB|11.46 GB|0.56 GB|1.43 GB|**1.99 GB**|
|Qwen3.5-2B|2B|1.33 GB|4.55 GB|0.28 GB|0.54 GB|**0.82 GB**|

**Top comments:**

**u/_-_David** (score: 26):
> I heard something like six months ago a rumor that Gemma 4 would be a bitnet and push their QAT to the limit.  I didn't really put my faith into that, but I do think that is ultimately the better architecture.  But of course, there are often esoteric reasons why things don't work like a curious layperson might think. Training stability? Inference efficiency? Don't know.  But it wouldn't surprise me in the least if it were to turn out that way eventually, and models over 2bit precision are a relic.

  **u/AI_Enhancer** (score: -3):
  > This aged like fine milk

    **u/_-_David** (score: 12):
    > Why? I said I didn't anticipate it happening. "I didn't really put my faith into that" was what I said, then never claimed that other than "eventually" it might be that way. This seems like you just wanted a "gotcha!" pretty badly.

**u/Pulselovve** (score: 60):
> At some point you reach reasonable physics limits.
Weights store information, routines, reasoning patterns, etc. you can squeeze them till some point  but you can't have all human knowledge and thinking patterns (on text at least) compressed in 8 gb...

You are losing necessarily resolution.
The problem is at the moment we can't separate, reasoning/intelligence from information, maybe then we will have very good reasoners with no information but that can fetch the info they need.

  **u/Lorian0x7** (score: 28):
  > Sure, that's true but it's been 3 years that I keep reading this argument and look how far we went since then. People were already doomed about knowledge density 3 years ago, what makes you think now is different? I'm pretty sure in another 3 years we will have the same discussion.

  **u/waruby** (score: 10):
  > The latest paper from Deepseek kind of does that, and is orthogonal with MoE, so it further reduces the number of active parameters required for the same quality of answers from the model. 

    **u/ai_without_borders** (score: 15):
    > yeah the deepseek paper is wild. i was reading some analysis on zhihu about it and the interesting context is that this efficiency research isnt just academic for them, its directly motivated by the chip export restrictions. when you cant buy h100s you have to squeeze every bit of performance out of what you have. so moe, low bit quantization, and kv compression arent nice to haves, theyre survival strategies. the fact that these techniques also happen to benefit the local llm community running on consumer gpus is kind of a happy accident. basically chinese labs are speed running efficient inf

    **u/oxygen_addiction** (score: 5):
    > And if engram separation between knowledge/logic ends up working in practice and we get better RAM+VRAM utilization, a bit of all of the above will lead to better local inference.

  **u/Feztopia** (score: 5):
  > You don't need perfect compression. You just need compression that gives you a Qwen3.5-35B-A3B in the size of Qwen3.5-9B Q4 that's still better than the 9B. That would be already progress.

  **u/plaintexttrader** (score: 2):
  > The “very good reasoner without information” is an interesting point. Though I think that might be impossible. LLMs reason through chains of thoughts, which is possible via language training on tons of data, and knowledge is inseparable “side effect”. You do need knowledge to develop common sense for CoTs.
It is not possible to reason that “1kg of feather is the same weight as 1kg of steel” without knowing what is a kg, a feather, steel.

  **u/Ell2509** (score: 2):
  > But you arent losing anything appreciable, in the case of turboquant. They have changed the format that data is stored as, without losing accuracy. Now that is the "polar" stuff, and relates yo KV cache (convo history). 

For model weights, they are actually building models from the ground up in 1 bit, rather than training a model and then compressing. The claim is that this also changes the form without losing accuracy compared to unquantised models. 

The nature of it is changing from navigating the matrix of relational probability with more concise instructions. Instead of "go left, up 2 fl

    **u/Pulselovve** (score: 6):
    > Yes they found a way to compress efficiently context, that's very far in terms of order of magnitude compared to the knowledge that gets compressed in weights. We are already using similar optimisations in quantisation.

  **u/MartiniCommander** (score: 1):
  > This is exactly like xvid-x264-x265-AV1

We always say there's a limit but then we find ways to keep going around them. 

**u/unbannedfornothing** (score: 10):
> Where did you get this numbers for k\\v cache? This is incorrect. Even 397B model gives \`llama\_kv\_cache: size = 7680.00 MiB (262144 cells,  15 layers,  4/1 seqs), K (f16): 3840.00 MiB, V (f16): 3840.00 MiB\` for 256K context for me. And for q8\_0: \`llama\_kv\_cache: size = 4080.00 MiB (262144 cells,  15 layers,  4/1 seqs), K (q8\_0): 2040.00 MiB, V (q8\_0): 2040.00 MiB\`

**u/ambient_temp_xeno** (score: 9):
> I'm not sure how I ended up in a ~~1.25 bit~~ 1.125 bit model quant timeline. I had chest pains the night before.

**u/jaker86** (score: 7):
> Could be cool! Numbers are a bit optimistic IMO:

Turboquant is great, but does not apply linearly to cache numbers for models like Qwen3.5; due to their hybrid architecture, a some of the cache is not K or V.

You also need to account for VRAM overhead during operation.

Source: running turboquant’d 27b on my 3090

**u/No-Refrigerator-1672** (score: 111):
> Why stop at 1-bit? Let's go with 0 bit! Who even needs weights at all? Imagine running a model with literally zero vram needed!

  


  **u/dero_name** (score: 131):
  > \&gt; Imagine running a model with literally zero vram needed!

You mean thinking? For myself? Heretic.

  **u/live_love_laugh** (score: 38):
  > Why the sarcasm? Maybe the wins are overblown a bit and the performance loss underplayed, but I still think the benefits are real and significant.

I'm still surprised that PrismML went for 1-bit and not for 1.58-bit (i.e. ternary) parameters. Intuitively I would think that having both 0 and -1 at your disposal would be a massive win for the expressiveness of the network. But I'm not really educated enough for my intuition to be worth much.

I have seen people talk about how the real world performance of PrismML's Bonsai models is disappointing. But I mean, if the performance loss can be mitig

    **u/No-Refrigerator-1672** (score: 38):
    > &gt;Why the sarcasm?

Because there are no real models listed, no real tests run, not even a theoretical proposition on how to quant to 1-bit without lobotomizing a model. Just some numbers that are completely made up and have nothing behind them. Why would anyone consider it serious?

    **u/Brou1298** (score: 4):
    > Also confused by the no ternary

    **u/sonicnerd14** (score: 1):
    > I think the way to look at is what would this do for the quantized versions that are already very coherent at sizes like 3 and 4 bits, maybe even 2 bit. If this is what a model with 1 bit can do then just imagine what a usable sized model would be able to do with the same optimizations applied.

  **u/JsThiago5** (score: 4):
  > You can simply imagine it running and then type the answer

  **u/DR4G0NH3ART** (score: 7):
  > Do it in your head, we might even call it, hmm.. Let us go with Natural intelligence. 

  **u/OXKSA1** (score: 2):
  > technically you could use ram instead of vram and use cloud ai models lol

  **u/Koalateka** (score: 2):
  > The kind of mentality that makes science advance...

  **u/bapuc** (score: 3):
  > This is what i am working on

    **u/Silver-Champion-4846** (score: 1):
    > Where have you reached so far?

  **u/sammcj** (score: 4):
  > Ideally models would start giving bits back, it's about time

  **u/Constant-Simple-1234** (score: 1):
  > This is already possible. Just switch from Qwen at 27B to the one at 2B. Seems like thinking is very compressible and can span wide range of sizes. It is just a lossy compression and the loss is real. (Partially joking, at least in tone ;) )

  **u/TopChard1274** (score: 1):
  > Fun at parties \⁠(⁠ϋ⁠)⁠/⁠♩

  **u/ohgoditsdoddy** (score: 1):
  > I thought good models with ternary weights (~1.5bit) are possible if they are trained for it (as opposed to quantized after the fact).

    **u/Silver-Champion-4846** (score: 2):
    > They should be, but there's not much support by companies as of yet

  **u/gigaflops_** (score: 1):
  > I run all my LLMs at -1 bit quantization and that way they *increase* the amount of available VRAM on my graphics card

**u/retireb435** (score: 3):
> but when

**u/spaceman_** (score: 18):
> The 1-bit models which Microsoft (BitNet) and PrismML (Bonsai) developed are NOT 1-bit quantized versions of other models. They are specialized models. You cannot have a 1-bit 8B model that competes against a 4, 8 or 16-bit 8B model and expect the same level of quality.

  **u/One_Key_8127** (score: 27):
  > Bonsai is quantized Qwen3 8b. I wonder whether you can quantize the Qwen3.5 MoE models to 1bit, but the dense 27b Qwen3.5 should be within PrismML's reach.

    **u/a_beautiful_rhind** (score: 0):
    > Ahh.. ok.. then it's just more fucking grift. Fool me once. 

Computationally heavy conversion to low-bit and getting meh performance has been done. Basically will never go anywhere.

In before a bunch of downvotes saying "n-n-ooo you're wrong this time, its good... :rocket: :rocket:"

Also see why Revolutionalredstone made that mistake. It was a *bit* misrepresented.

    **u/[deleted]** (score: -3):
    > [deleted]

  **u/Odd-Ordinary-5922** (score: 11):
  > I dont understand why you say things with such certainty when the optimization improvement of llms has been crazy this past year


---

