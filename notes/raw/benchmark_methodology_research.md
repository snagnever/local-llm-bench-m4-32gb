# LLM Benchmark Methodology Research (April 2026)

Deep research on the current state-of-the-art for benchmarking local LLMs, focused on practical evaluation on a 32GB M4 Mac.

---

## 1. Is simple-evals Still Good?

### Status: Deprecated as of July 2025

OpenAI's simple-evals is **no longer actively maintained**. The repository states: "simple-evals will no longer be updated for new models or benchmark results." It will continue to host reference implementations for HealthBench, BrowseComp, and SimpleQA only.

### What simple-evals Includes
- MMLU, MATH, GPQA, DROP, MGSM, HumanEval, SimpleQA, BrowseComp, HealthBench
- Zero-shot, chain-of-thought prompting (no few-shot or role-play)
- Sensitive to prompting style -- different libraries get different results on the same benchmarks

### Limitations
- **No longer updated** for new models or results
- PRs and Issues are not actively monitored
- Non-OpenAI model results were rarely updated even when active
- No unified setup mechanism; each eval requires separate configuration
- Limited benchmark coverage compared to alternatives
- No native support for GGUF/local model formats

### Verdict
simple-evals was a lightweight, opinionated reference implementation. It is now effectively abandoned. For ongoing local evaluation work, you need something else.

### Better Alternatives

| Tool | Status | Benchmarks | Local Model Support | Notes |
|------|--------|-----------|-------------------|-------|
| **lm-evaluation-harness** (EleutherAI) | Actively maintained (v0.4.9+) | 60+ benchmarks, 400+ tasks | HF, GGUF, vLLM, SGLang, API | De facto standard; powers HF Open LLM Leaderboard |
| **DeepEval** (Confident AI) | Active | 16+ benchmarks (MMLU, HumanEval, GSM8K, IFEval, DROP, etc.) | Any LLM via custom class | Python-native, pytest-like interface, easy setup |
| **LiveBench** | Active (monthly updates) | 18 tasks across 6 categories | API-compatible models | Contamination-free, no LLM judge needed |
| **MathArena** | Active | AIME 2025/2026, math olympiads | vLLM, API | Uncontaminated math evaluations |

**Recommendation: lm-evaluation-harness is the clear standard for local model benchmarking in 2026.** It supports GGUF models directly, runs on Apple Silicon (MPS backend), and covers nearly every benchmark you'd want.

Sources:
- [OpenAI simple-evals GitHub](https://github.com/openai/simple-evals)
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [DeepEval Benchmarks](https://deepeval.com/docs/benchmarks-introduction)

---

## 2. Which Benchmarks Matter Most in 2026?

### Benchmark-by-Benchmark Analysis

#### MMLU-Pro (Harder MMLU)
- **What:** 12,000+ questions, 14 domains, 10 answer choices (vs MMLU's 4)
- **Run locally:** Yes. GitHub repo with eval scripts: `cd scripts/examples/ && sh eval_llama_2_7b.sh`
- **Open code:** Yes ([TIGER-AI-Lab/MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro))
- **Also in:** lm-evaluation-harness
- **Verdict:** Strong upgrade from MMLU. MMLU is saturated (frontier models >88%); MMLU-Pro has ~2% prompting sensitivity and much better discrimination

#### LiveCodeBench v6 (Contamination-Free Coding)
- **What:** 1,055 problems (v6, April 2025) from LeetCode/AtCoder/CodeForces contests, tagged with release dates
- **Run locally:** Yes, but requires vLLM for open models (GPU-oriented). Uses Python 3.11
- **Open code:** Yes ([LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench))
- **Key feature:** Time-segmented evaluation -- test only on problems released after model's training cutoff
- **Limitation on Mac:** vLLM dependency is CUDA-focused; would need to use API mode or adapt for llama.cpp server
- **Verdict:** Best coding benchmark available. Contamination-free by design

#### TAU2-Bench (Agentic Tasks)
- **What:** Evaluates conversational agents on tool use and policy adherence across airline, retail, telecom, banking domains
- **Run locally:** Yes via `uv sync && tau2 run --domain airline --agent-llm <model> --user-llm <model>`
- **Open code:** Yes ([sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench))
- **Also:** [tau2-bench-verified](https://github.com/amazon-agi/tau2-bench-verified) (corrected/verified version by Amazon)
- **Limitation:** Requires a user-simulator LLM (API cost) in addition to the model being tested
- **Verdict:** Good for agentic evaluation but complex setup; requires two LLM endpoints

#### SWE-bench Verified (Software Engineering)
- **What:** 500 human-validated real GitHub issues across 12 Python repos
- **Run locally:** Yes, but requires Docker for reproducible environments (~150GB+ disk for images)
- **Open code:** Yes ([SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench))
- **Limitation:** Very resource-intensive; Docker-based execution; primarily tests agentic coding pipelines, not raw model capability
- **Verdict:** Gold standard for SWE capability but overkill for quick local model comparison. Better for evaluating full agent systems

#### AIME 2025/2026 (Math Competition)
- **What:** 30 problems per year from American Invitational Mathematics Examination; integer answers 000-999
- **Run locally:** Yes via MathArena (`uv run python scripts/run.py --comp aime/aime_2025 --models <model>`)
- **Open code:** Yes ([eth-sri/matharena](https://github.com/eth-sri/matharena)); datasets on HuggingFace
- **Key feature:** Uncontaminated -- new problems each year with known release dates
- **Limitation:** Only 30 questions -- statistically noisy (see Section 3)
- **Verdict:** Excellent for math reasoning but small sample. Use alongside MATH-500

#### Arena-Hard v2 (Open-Ended Quality)
- **What:** 500 challenging user queries + 250 creative writing queries from Chatbot Arena
- **Run locally:** Yes ([lmarena/arena-hard-auto](https://github.com/lmarena/arena-hard-auto))
- **Limitation:** **Requires an LLM judge** (GPT-4.1 or Gemini-2.5 recommended). Measures relative win-rate against a baseline model
- **Verdict:** Highest correlation with human preferences (Chatbot Arena). Worth running but judge dependency adds cost/complexity

#### IFEval (Instruction Following)
- **What:** ~500 prompts with verifiable constraints ("write >400 words", "mention AI 3+ times")
- **Run locally:** Yes, available in lm-evaluation-harness
- **Open code:** Yes (part of lm-eval-harness)
- **Key feature:** **No LLM judge needed** -- purely rule-based verification
- **Verdict:** Highly recommended. Cheap, fast, objective, no judge bias. Tests a capability that matters for practical use

#### RULER (Long Context)
- **What:** Synthetic benchmark testing real context utilization (needle-in-haystack, multi-hop tracing, aggregation)
- **Run locally:** Yes ([NVIDIA/RULER](https://github.com/NVIDIA/RULER))
- **Limitation:** Requires Docker (NVIDIA PyTorch container); designed for NVIDIA GPUs
- **Verdict:** Important if testing long-context claims but heavy infrastructure requirements. Not practical for quick Mac-based evaluation

#### HLE (Humanity's Last Exam)
- **What:** 2,500 expert-level questions across dozens of subjects; both multiple-choice and short-answer
- **Run locally:** Yes (`pip install -r requirements.txt` in hle_eval directory)
- **Open code:** Yes ([centerforaisafety/hle](https://github.com/centerforaisafety/hle)), MIT license
- **Key feature:** Extremely difficult -- frontier models score ~3-10%
- **Limitation:** Uses OpenAI API interface for evaluation; would need adaptation for local models
- **Verdict:** Interesting for testing ceiling capability but may be too hard for smaller local models to show meaningful differentiation

### Recommended Benchmark Suite for Local Evaluation on M4 Mac

**Tier 1 -- Run These (easy, fast, no LLM judge):**
| Benchmark | Questions | Judge Needed | In lm-eval-harness | Why |
|-----------|-----------|-------------|--------------------|----|
| MMLU-Pro | 12,000+ | No | Yes | Knowledge + reasoning, not saturated |
| IFEval | ~500 | No (rule-based) | Yes | Instruction following, objective |
| GSM8K | 1,319 | No | Yes | Math reasoning baseline |
| HumanEval | 164 | No (execution) | Via DeepEval | Code generation |

**Tier 2 -- Run These (moderate setup):**
| Benchmark | Questions | Judge Needed | Why |
|-----------|-----------|-------------|-----|
| GPQA Diamond | 198 | No | Hard graduate-level questions |
| MATH-500 | 500 | No (exact match) | Mathematical problem solving |
| AIME 2025/2026 | 30 | No (integer match) | Math competition, uncontaminated |
| DROP | 9,536 | No | Reading comprehension + discrete reasoning |

**Tier 3 -- If You Have Time/Resources:**
| Benchmark | Questions | Judge Needed | Why |
|-----------|-----------|-------------|-----|
| Arena-Hard v2 | 750 | Yes (GPT-4.1/Gemini) | Best human-preference proxy |
| LiveCodeBench v6 | 1,055 | No (execution) | Contamination-free coding |
| LiveBench | 18 tasks | No (objective) | Monthly-refreshed, contamination-free |
| HLE | 2,500 | Yes | Frontier difficulty |

Sources:
- [MMLU-Pro GitHub](https://github.com/TIGER-AI-Lab/MMLU-Pro)
- [LiveCodeBench GitHub](https://github.com/LiveCodeBench/LiveCodeBench)
- [TAU2-bench GitHub](https://github.com/sierra-research/tau2-bench)
- [SWE-bench GitHub](https://github.com/SWE-bench/SWE-bench)
- [MathArena GitHub](https://github.com/eth-sri/matharena)
- [Arena-Hard-Auto GitHub](https://github.com/lmarena/arena-hard-auto)
- [IFEval in lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/ifeval/README.md)
- [NVIDIA RULER](https://github.com/NVIDIA/RULER)
- [HLE GitHub](https://github.com/centerforaisafety/hle)
- [LiveBench GitHub](https://github.com/LiveBench/LiveBench)

---

## 3. Sample Size: Is 20 Questions Enough?

### Short Answer: No. 20 questions is statistically unreliable.

### The Math

For a binary benchmark (correct/incorrect) with n=20 questions:

**95% Confidence Interval width using Bernoulli SE:**
- If a model scores 70% (14/20): SE = sqrt(0.7 * 0.3 / 20) = 0.102
- 95% CI = 70% +/- 20.1 percentage points = **[49.9%, 90.1%]**
- This means a "70%" score could actually be anywhere from 50% to 90%

**If a model scores 80% (16/20):**
- SE = sqrt(0.8 * 0.2 / 20) = 0.089
- 95% CI = 80% +/- 17.5pp = **[62.5%, 97.5%]**

**Comparison: two models scoring 70% vs 80% on 20 questions have completely overlapping confidence intervals.** You cannot distinguish them.

### What the Research Says

Key findings from Cameron Wolfe's "Applying Statistics to LLM Evaluations" and the Benchmark-squared paper:

1. **CLT fails below n=100.** The Central Limit Theorem-based confidence intervals become unreliable with fewer than ~100 datapoints. With n=20, standard statistical methods don't even apply properly.

2. **Minimum practical threshold: 100-200 questions** for basic reliability with CLT methods. For comparing models with small differences, you need 500+.

3. **The quadratic relationship:** Detecting a performance gap half the size requires **4x the samples**. If you want to distinguish a 5% difference (e.g., 75% vs 80%), you need roughly 4x more samples than to distinguish a 10% difference.

4. **BIG-Bench guidelines:** Minimum 32 evaluation samples per task, though authors are encouraged to create much larger tasks.

5. **Clustered questions:** If your 20 questions cluster into related topics, effective sample size is even smaller. One study found that accounting for question clusters increased standard error by **3x**.

### Practical Recommendations for Your Setup

| Sample Size | What You Can Detect | Useful For |
|------------|--------------------:|------------|
| 20 | ~22pp differences | Quick sanity check only; cannot compare models |
| 50 | ~14pp differences | Rough ordering of very different models |
| 100 | ~10pp differences | Basic model comparison with caveats |
| 200 | ~7pp differences | Reasonable model comparison |
| 500 | ~4.4pp differences | Good statistical power for most comparisons |
| 1000+ | ~3pp differences | Publication-quality evaluation |

### Variance Reduction Techniques (to get more from fewer questions)

1. **Paired comparison:** When comparing two models, test them on the *same* questions. This leverages inter-model correlation (typically r=0.3-0.7) for a "free" variance reduction. Report paired differences, not independent scores.

2. **Multiple samples per question (resampling):** Generate K outputs per question. Reduces within-question variance by factor of K. Total variance becomes: `Var = (Var_between + Var_within/K) / n`

3. **Token probability scoring:** Where possible, use next-token log-probabilities instead of sampling. This eliminates within-question stochastic variance entirely.

4. **Factorized Active Querying (FAQ):** Recent technique delivers up to **5x effective sample size gains** by adaptively selecting which questions to ask.

### Bottom Line

**With 20 questions, you have a vibe check, not a benchmark.** Increase to at least 100 for basic reliability, 200+ for meaningful comparison. The full benchmark sets (MMLU-Pro: 12K, IFEval: 500, GSM8K: 1.3K) exist for good reason.

If resource-constrained, use a stratified subset of 100-200 questions from each benchmark rather than 20 across all benchmarks.

Sources:
- [Applying Statistics to LLM Evaluations](https://cameronrwolfe.substack.com/p/stats-llm-evals) -- Cameron Wolfe
- [Benchmark-squared: Systematic Evaluation of LLM Benchmarks](https://arxiv.org/abs/2601.03986)
- [Efficient Evaluation with Statistical Guarantees](https://arxiv.org/abs/2601.20251)

---

## 4. Tools the Community Actually Uses

### lm-evaluation-harness (EleutherAI) -- The Standard

**Status in 2026:** Still the de facto standard. Powers HuggingFace Open LLM Leaderboard. Current version: v0.4.9+.

**Key features:**
- 60+ benchmarks, 400+ tasks/subtasks
- GGUF model support via HF backend (pass `gguf_file=model.gguf` in model_args)
- Apple Silicon MPS support via HuggingFace backend
- API backend (OpenAI-compatible, Anthropic, etc.)
- vLLM and SGLang backends
- Chain-of-thought stripping (`think_end_token` argument)
- CLI refactored with subcommands: `lm_eval run`, `lm_eval ls`, `lm_eval validate`
- YAML config support for reproducible evaluation setups
- Lighter base install -- backends installed separately: `pip install lm_eval[hf]`

**Running GGUF on Mac:**
```
pip install "lm_eval[hf]"
lm_eval run --model hf \
  --model_args pretrained=/path/to/model,gguf_file=model.gguf,tokenizer=<tokenizer> \
  --tasks mmlu_pro,ifeval,gsm8k \
  --device mps \
  --batch_size 1
```

**Important GGUF tip:** Always provide a separate tokenizer path. Without it, HF attempts to reconstruct the tokenizer from the GGUF file, which can take hours or hang indefinitely.

**Alternative approach:** Run model as llama-server (OpenAI-compatible API) and use the API backend:
```
# Terminal 1: Start llama.cpp server
llama-server --model model.gguf --port 8080 --n-gpu-layers 99

# Terminal 2: Run evaluation
lm_eval run --model local-completions \
  --model_args model=local,base_url=http://localhost:8080/v1 \
  --tasks mmlu_pro,ifeval
```

### DeepEval -- Easy Python-Native Alternative

```python
from deepeval.benchmarks import MMLU, IFEval
benchmark = MMLU()
benchmark.evaluate(model=your_custom_llm, batch_size=5)
print(benchmark.overall_score)
```
- 16+ benchmarks including MMLU, IFEval, HumanEval, GSM8K, DROP, HellaSwag
- Custom LLM class wraps any model endpoint
- Pytest-like interface for CI/CD integration
- No Apple Silicon-specific optimizations but works via API

### LocalScore (Mozilla) -- Hardware Performance Only

- Measures speed, not quality: prompt processing tok/s, generation tok/s, TTFT
- Open source (Apache 2.0), Mozilla-backed
- Built on Llamafile/llama.cpp
- Download-and-run CLI executable
- Apple Silicon M-series supported
- Score: 1000 = excellent, 250 = passable, <100 = poor
- **Not a quality benchmark** -- only measures inference performance

### SiliconBench -- Apple Silicon Specific

- Tracks LLM tok/s benchmarks specifically on Apple Silicon hardware
- Available at [siliconbench.radicchio.page](https://siliconbench.radicchio.page/)
- Speed benchmarks only, not quality

### LM Studio Benchmarking

LM Studio itself does not have built-in quality benchmarks. Third-party tools exist:
- **lm-studio-benchmark** (GitHub: Shamik07) -- coding task evaluation
- **lm-speedometer** -- simple tok/s measurement
- Best approach: Use LM Studio's OpenAI-compatible API endpoint with lm-evaluation-harness or DeepEval

### simple-evals vs lm-eval-harness Comparison

| Feature | simple-evals | lm-eval-harness |
|---------|-------------|-----------------|
| Status | Deprecated (July 2025) | Active (v0.4.9+) |
| Benchmarks | 9 | 400+ tasks |
| GGUF support | No | Yes (via HF backend) |
| Apple Silicon | No | MPS support |
| API models | OpenAI-focused | OpenAI, Anthropic, TextSynth, custom |
| Community | Minimal | Huge (hundreds of papers) |
| Maintenance | Bug fixes only | Active development |
| Prompt style | Zero-shot CoT only | Configurable few-shot, CoT, chat templates |

Sources:
- [lm-evaluation-harness GitHub](https://github.com/EleutherAI/lm-evaluation-harness)
- [DeepEval GitHub](https://github.com/confident-ai/deepeval)
- [LocalScore](https://www.localscore.ai/blog)
- [SiliconBench](https://siliconbench.radicchio.page/)
- [LM Studio benchmark tool](https://github.com/Shamik07/lm-studio-benchmark)
- [Arena-Hard-Auto](https://github.com/lmarena/arena-hard-auto)

---

## 5. Evaluation Pitfalls

### Quantization Effects on Benchmarks

Quantization does NOT affect all benchmarks equally. Specific degradation patterns from research on GGUF quantized models:

#### Quality Retention by Quantization Level (GGUF)

| Quant | Quality vs FP16 | Notes |
|-------|-----------------|-------|
| Q8_0 | ~99% | Effectively lossless for most tasks |
| Q5_K_M | ~95-97% | Best quality/speed balance |
| Q4_K_M | ~92% | Good for most tasks, notable drops in some |
| Q3_K_M | ~85% | Noticeable degradation |
| Q2_K | ~70% | Significant quality loss |

#### Which Benchmarks Are Most Sensitive to Quantization?

**Most resilient (Q4 still works well):**
- GSM8K (math word problems): Retains 84-87% of baseline at Q4. Step-by-step arithmetic is structurally resilient
- BBH (logical reasoning): ~90% accuracy retention at Q4_K_M
- Code generation (HumanEval): Q4_K_M matches AWQ and BitsandBytes at ~51.8% pass@1

**Most sensitive (avoid Q4):**
- IFEval (instruction following): >10% accuracy loss at INT4/Q4 levels. Highly sensitive
- C-Eval / multilingual tasks: 15-20% reduction at Q4_K_M. Language-specific embeddings suffer
- MMLU (knowledge/factual): Sensitive; lower-bit GGUF is risky for knowledge-intensive tasks

**Practical recommendation:**
- For quality-critical benchmarks: Use Q5_K_M minimum, Q8_0 preferred
- For coding/math: Q4_K_M acceptable
- For instruction following / knowledge: Q5_K_M or higher
- General rule: Run the largest model that fits at Q8 rather than squeezing a bigger model at Q4

### Context Window Effects

- Models perform worse at longer contexts even within their advertised window
- RULER benchmark shows effective context is often much shorter than claimed
- For benchmarks: Most standard benchmarks use short contexts (<4K tokens), so this is rarely an issue
- If testing long-context: Use RULER or LiveBench's data analysis tasks

### Contamination / Data Leaking

**Contaminated benchmarks (avoid for reliable comparison):**
- MMLU -- widely used in training data; scores are inflated
- HumanEval -- problems available on GitHub for years
- HellaSwag, ARC -- likely in most training sets
- GSM8K -- increasingly contaminated

**Clean / contamination-resistant benchmarks:**
- **LiveCodeBench** -- time-segmented, new problems from weekly contests
- **LiveBench** -- monthly-refreshed questions from recent sources
- **AIME 2025/2026** -- new competition problems each year
- **HLE** -- expert-created, not in training data; rolling version (HLE-Rolling) adds fresh questions
- **MMLU-Pro** -- newer, less likely contaminated, 10-way choice reduces guessing
- **IFEval** -- verifiable constraints, harder to memorize

**Best practice:** Use time-stamped benchmarks and compare models only on problems released after their training cutoff.

### No Benchmarks Specifically for Quantized Models

There are no dedicated "quantization benchmarks." The standard approach is:
1. Run the same benchmark on full-precision and quantized versions
2. Measure the degradation per task category
3. The degradation pattern is model-specific and task-specific

Sources:
- [LLM Quantization Guide: GGUF vs AWQ vs GPTQ vs bitsandbytes](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/)
- [Benchmarking Quantized LLMs: What Works Best for Real Tasks](https://www.ionio.ai/blog/llm-quantize-analysis)
- [Profiling LLM Inference on Apple Silicon: A Quantization Perspective](https://arxiv.org/abs/2508.08531)
- [LiveCodeBench](https://livecodebench.github.io/)
- [LiveBench](https://livebench.ai/livebench.pdf)

---

## 6. Grading Methodology

### Using Gemini Flash as an LLM Judge: Mostly Reliable, With Caveats

#### What the Research Says (2025-2026)

The Sage evaluation framework (December 2025) provides the most comprehensive analysis:

**Judge model ranking (best to worst on consistency):**
1. Gemini-2.5-Pro (best overall: IPI 0.072, TOV 1.091)
2. **Gemini-2.5-Flash** (strong second: IPI 0.087, TOV 1.326)
3. Qwen3-235B-A22B
4. GPT-5-Chat
5. Claude models

**Gemini Flash specifics:**
- On easy distinctions (clearly better/worse responses): 65.2% consistency between scoring methods
- On hard distinctions (similar quality responses): Only 31.5% consistency
- All models show ~200% degradation from easy to hard evaluation scenarios
- Gemini-2.5-Flash is a strong, cost-effective judge for clear quality differences but unreliable for subtle ones

**Known biases in all LLM judges:**
- Positional bias (answer order affects judgment): Ranges from 25% to 76% inconsistency depending on model
- Verbosity bias (longer = judged as better)
- Self-enhancement bias (models prefer their own outputs)
- Format inconsistency: Direct scoring vs pairwise comparison gives different results

#### Best Grading Approaches in 2026

**Tier 1: Objective/Rule-Based (Preferred -- no judge needed)**
- Exact match (AIME, MATH, SimpleQA)
- Code execution / test passing (HumanEval, LiveCodeBench, SWE-bench)
- Verifiable constraints (IFEval -- "write >400 words")
- Multiple-choice accuracy (MMLU-Pro, GPQA)

**Tier 2: LLM-as-Judge (When Objective Grading Isn't Possible)**
- Use the strongest available model: Gemini-2.5-Pro > Gemini-2.5-Flash > GPT-4.1
- Use explicit rubric prompting: Reduces inconsistency by 16% (IPI) and 11% (TOV)
- Use pairwise comparison rather than point-wise scoring for open-ended tasks
- If using a grading scale, use 0-5 (research shows highest human-LLM alignment on this scale)
- Run both orderings (A,B and B,A) and check for positional consistency
- **Panel of diverse judges** (multiple different models): 7-15% improvement over single judge
- **Avoid debate-based multi-agent judging:** 45-158% degradation vs single judges

**Tier 3: Avoid**
- Using the same model as both generator and judge (self-enhancement bias)
- Single-score rubrics without explicit criteria
- Debate-style multi-judge setups (ChatEval approach performs worse)

#### Gemini Flash Verdict for Your Setup

Gemini Flash is a **reasonable choice** as a judge, ranking 2nd overall in consistency research. It's the most cost-effective strong judge available. However:
- Use it only where objective grading is impossible
- Always provide explicit rubrics in judge prompts
- For close comparisons, run both orderings
- Be aware that subtle quality differences may not be reliably detected
- Consider upgrading to Gemini-2.5-Pro for critical evaluations (it's meaningfully better)

### Self-Consistency / Majority Voting

**Should you use it?** Yes, for benchmarks that involve generation (not multiple-choice).

Recent advances (2025):
- **Ranked Voting** outperforms simple majority voting by considering alternative answer rankings
- **Reasoning-Aware Self-Consistency (RASC):** Reduces sample usage by ~70% while maintaining accuracy through criteria-based stopping and weighted voting
- **Self-certainty based voting:** Lightweight metric that rivals traditional self-consistency

**Practical recommendation:**
- For multiple-choice benchmarks: Not needed (use log-probabilities if available)
- For generation benchmarks (coding, math): Use majority@K with K=3-5 samples
- For resource-constrained setups: Use self-certainty or RASC to reduce required samples
- Report both single-sample and majority-vote scores

Sources:
- [Are We on the Right Way to Assessing LLM-as-a-Judge?](https://arxiv.org/html/2512.16041v1)
- [Grading Scale Impact on LLM-as-a-Judge](https://arxiv.org/html/2601.03444v1)
- [JudgeBench: Evaluating LLM-Based Judges](https://openreview.net/forum?id=G0dksFayVq)
- [Ranked Voting Self-Consistency](https://arxiv.org/abs/2505.10772)
- [Reasoning Aware Self-Consistency](https://aclanthology.org/2025.naacl-long.184/)

---

## 7. Recommended Action Plan

### Immediate Changes (High Impact, Low Effort)

1. **Switch from simple-evals to lm-evaluation-harness** as your evaluation framework
2. **Increase sample size to at least 100 per benchmark** (use `--limit 100` in lm-eval if full set is too slow)
3. **Add IFEval** -- free, fast, objective, no judge needed, and tests practical capability
4. **Replace MMLU with MMLU-Pro** -- better discrimination, less saturated
5. **Always run paired comparisons** -- same questions for all models, report paired differences

### Medium-Term Improvements

6. **Add contamination-free benchmarks:** AIME 2025/2026 (math), LiveCodeBench (code)
7. **Run GGUF models at Q5_K_M or Q8_0** for benchmarking (even if deploying at Q4)
8. **Report confidence intervals** alongside point scores
9. **Use log-probabilities** where possible for multiple-choice tasks (eliminates sampling variance)

### Setup for M4 Mac with 32GB

**Option A: lm-eval-harness with GGUF directly**
```
pip install "lm_eval[hf]"
lm_eval run --model hf \
  --model_args pretrained=./models/,gguf_file=model-Q5_K_M.gguf,tokenizer=<base-model-name> \
  --tasks mmlu_pro,ifeval,gsm8k_cot,gpqa_diamond,drop \
  --device mps --batch_size 1
```

**Option B: llama.cpp server + API backend (more flexible)**
```
# Start server
llama-server -m model.gguf -ngl 99 --port 8080

# Run evaluation
lm_eval run --model local-chat-completions \
  --model_args model=local,base_url=http://localhost:8080/v1,tokenizer_backend=huggingface \
  --tasks mmlu_pro,ifeval \
  --apply_chat_template
```

**Option C: DeepEval (simplest Python API)**
```python
from deepeval.benchmarks import MMLU, IFEval, HumanEval
for bench in [MMLU(), IFEval(), HumanEval()]:
    bench.evaluate(model=my_local_llm)
    print(f"{bench.__class__.__name__}: {bench.overall_score}")
```

### What Not to Do

- Do not rely on 20-question subsets for model comparison
- Do not use MMLU alone (saturated, contaminated)
- Do not benchmark at Q4 and assume results apply to Q8 (or vice versa)
- Do not use a weak LLM judge for subtle quality comparisons
- Do not ignore confidence intervals when claiming Model A > Model B
- Do not mix prompting styles between models (use same chat template approach for all)

---

## Summary Table: Key Recommendations

| Question | Answer |
|----------|--------|
| Is simple-evals still good? | No -- deprecated July 2025. Switch to lm-eval-harness |
| Best tool for local eval? | lm-evaluation-harness (EleutherAI) with GGUF/MPS support |
| Is 20 questions enough? | No -- minimum 100, ideally 200-500 per benchmark |
| Best judge model? | Gemini-2.5-Pro > Gemini-2.5-Flash > GPT-4.1. But prefer objective scoring |
| Most important benchmarks? | MMLU-Pro, IFEval, GSM8K, GPQA Diamond (objective); Arena-Hard v2 (judged) |
| Cleanest benchmarks? | LiveCodeBench, LiveBench, AIME 2025/2026, HLE-Rolling |
| Quantization for benchmarks? | Q5_K_M minimum; Q8_0 preferred; Q4 is risky for knowledge/instruction tasks |
| Majority voting? | Yes for generation tasks (K=3-5); not needed for multiple-choice |
| Apple Silicon tools? | lm-eval-harness (MPS), LocalScore (speed only), SiliconBench (speed only) |
