# Test Parameters Research for Local LLM Benchmarking on Apple Silicon

Research compiled: 2026-04-08

---

## 1. Context Window Settings

### Current Setup: 8192 context

### Is 8192 enough for all benchmarks?

**Short answer: YES for input, but may be limiting for chain-of-thought output in thinking models.**

Analysis of each benchmark's context requirements based on the codebase (simple-evals) and LiveBench:

| Benchmark | Input Token Estimate | Output Needed | 8192 Sufficient? |
|-----------|---------------------|---------------|-------------------|
| MMLU | ~200-400 tokens (question + 4 choices) | ~100-300 (short reasoning + letter) | YES |
| GPQA | ~300-800 tokens (graduate-level Q + 4 choices) | ~200-500 (reasoning + letter) | YES |
| DROP | ~1000-2500 tokens (passage + few-shot examples + question) | ~100-500 | YES, but tight with 3 few-shot examples |
| MATH | ~100-500 tokens (problem statement) | ~500-5000 (chain-of-thought) | MARGINAL for thinking models |
| HumanEval | ~200-500 tokens (function signature + docstring) | ~200-1000 (code) | YES |
| SimpleQA | ~50-200 tokens (factual question) | ~100-500 | YES |
| BrowseComp | ~200-500 tokens | ~200-500 | YES |
| LiveBench | Varies by category | Varies | YES for most tasks |

**Key findings:**
- GPQA questions are multiple-choice with expert-level content but not extremely long passages. The GPQA dataset has 198 (diamond), 448 (main), or 546 (extended) questions -- the questions themselves fit comfortably in 8K.
- DROP uses reading comprehension passages plus 3 few-shot examples. The simple-evals implementation includes passage context + 3 training examples as few-shot, which can push input to ~2000-3000 tokens. Still fits in 8K.
- MATH problems themselves are short, but chain-of-thought output from thinking models can be very long (thousands of tokens).
- LiveBench uses temperature=0 by default (no force_temperature), max_tokens defaults to 4096.

### Qwen3.5 Context Recommendations

From the official Qwen3.5-35B-A3B model card:
- **Native context length**: 262,144 tokens
- **Recommended minimum context**: 128,000 tokens to preserve thinking capabilities
- **Recommended max output for complex problems**: 81,920 tokens
- **Recommended max output for general queries**: 32,768 tokens

**Critical insight**: Qwen recommends 128K minimum context to preserve thinking capabilities. Running at 8192 is **far below** this recommendation. However, this applies primarily to thinking mode where the model generates long internal reasoning chains. For non-thinking mode (instruct mode), 8192 is more reasonable.

### How Context Size Affects Speed and Memory on Apple Silicon

**Memory impact:**
- KV cache is allocated **upfront** in llama.cpp based on the `-c` (context) parameter, not dynamically
- Setting context to 8192 when you only use 2000 tokens **still allocates memory for 8192 tokens**
- For the Qwen3.5-35B-A3B model with GQA: ~200MB of KV cache per slot at 8K context
- Doubling context from 4096 to 8192 adds several hundred MB of KV cache memory
- Going from 8192 to 32768 would add ~1-2GB more KV cache

**Speed impact:**
- Prompt processing (prefill) scales roughly linearly with context length
- Token generation speed is minimally affected by context length until very long contexts (>32K)
- Flash attention helps maintain speed at longer contexts
- On M4 with 32GB: 8K context is very comfortable, 32K is feasible, 128K would be tight

**Does setting 8192 when model supports 262K waste memory?**
- NO -- llama.cpp allocates KV cache based on the `-c` parameter, NOT the model's maximum
- Setting `-c 8192` only allocates KV cache for 8192 tokens regardless of model capability
- This is the correct approach: use the minimum context you need

### Recommendations for Context Window

| Use Case | Recommended Context |
|----------|-------------------|
| Non-thinking benchmarks (MMLU, GPQA, HumanEval) | 8192 is sufficient |
| Thinking mode benchmarks (MATH, reasoning) | 16384-32768 minimum |
| Complex chain-of-thought | 32768+ |
| Full Qwen3.5 thinking capability | 65536-131072 (if memory allows) |

**Practical recommendation for 32GB M4**: Use 16384 context as a balanced default. This provides headroom for chain-of-thought while keeping KV cache memory modest. For thinking mode, consider 32768 if memory allows.

---

## 2. Temperature and Sampling

### Current Setup: temperature=0.5 (in simple-evals ChatCompletionSampler default)

### What the frameworks use:

| Framework/Benchmark | Default Temperature | Notes |
|-------------------|-------------------|-------|
| OpenAI simple-evals (ChatCompletionSampler) | 0.5 | Hardcoded default in the sampler class |
| LiveBench (gen_api_answer.py) | 0 | Default when no force_temperature and no required_temperature |
| EleutherAI lm-eval-harness | 0.0 | Greedy decoding for most tasks |
| HuggingFace Open LLM Leaderboard | 0.0 for most; 0.5-0.7 for code (pass@k) | |
| Qwen3.5 official (thinking mode) | 0.6-1.0 | 1.0 for general, 0.6 for precise coding |
| Qwen3.5 official (non-thinking mode) | 0.7-1.0 | 0.7 general, 1.0 reasoning tasks |
| GPQA (simple-evals, n_repeats=4-10) | 0.5 (inherits sampler default) | Multiple repeats to average variance |
| MATH (simple-evals, n_repeats=10-16) | 0.5 (inherits sampler default) | Multiple repeats to average variance |

### Temperature 0 vs 0.5 for Benchmarking

**Temperature 0 (Greedy Decoding):**
- Performs argmax at each step -- always picks most probable token
- Maximizes reproducibility (though NOT perfectly deterministic due to floating-point issues, batching effects, and hardware differences)
- Best for: factual tasks, classification, data extraction, multiple-choice
- Used by: LiveBench, EleutherAI lm-eval-harness, most leaderboards
- Drawback: Can get stuck in repetitive loops; may not explore solution space for open-ended tasks

**Temperature 0.5:**
- OpenAI's simple-evals default -- a deliberate choice
- Introduces mild randomness while remaining fairly deterministic
- Better for chain-of-thought tasks where diverse reasoning paths help
- simple-evals compensates with n_repeats (4-16 repetitions) and averages
- Drawback: Less reproducible; requires multiple runs

**Temperature 0 is NOT perfectly deterministic because:**
1. Floating-point precision issues cause different rounding across runs
2. Batching effects: different concurrent requests change computation order
3. Hardware differences: BF16 precision is sensitive to tensor parallel size, batch size, GPU types
4. Quantization: GGUF quantized models have additional numerical noise

### Qwen3.5-Specific Temperature Settings

From the official model card:

**Thinking Mode:**
- General tasks: temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5
- Precise coding: temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0

**Non-Thinking (Instruct) Mode:**
- General tasks: temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5
- Reasoning tasks: temperature=1.0, top_p=1.0, top_k=40, presence_penalty=2.0

**Important**: These are Qwen's official recommendations. Using temperature=0 goes against their guidance and may produce suboptimal results due to the model being trained/aligned with these specific temperature ranges.

### top_p, top_k, min_p Settings

**Do they matter for benchmarks?**
- **top_p**: At temperature=0, top_p is irrelevant (greedy decoding ignores sampling). At temperature>0, top_p=0.95 is standard.
- **top_k**: Qwen3.5 recommends top_k=20 (thinking) or top_k=40 (non-thinking reasoning). Most frameworks ignore top_k or set it very high.
- **min_p**: Relatively new parameter. Qwen3.5 sets min_p=0.0 (disabled).
- **presence_penalty**: Qwen3.5 uniquely recommends high presence_penalty (1.5-2.0) to prevent repetition. This is unusual and model-specific.

### Recommendations for Temperature

| Approach | Temperature | Pros | Cons |
|----------|------------|------|------|
| **Greedy (reproducible)** | 0 | Deterministic, standard for leaderboards | May not match Qwen3.5 training distribution |
| **Qwen3.5 official** | 0.6-1.0 | Matches model's intended behavior | Non-reproducible, needs multiple runs |
| **Simple-evals default** | 0.5 | Balanced; matches OpenAI's methodology | Needs n_repeats for reliability |
| **Compromise** | 0.3 | Mostly deterministic with slight exploration | Non-standard |

**Practical recommendation**: 
- For **comparable leaderboard results**: temperature=0 (matches LiveBench, lm-eval-harness)
- For **best model performance**: Follow Qwen3.5 official settings (temp 0.6-1.0 depending on task)
- For **our setup (quick benchmarking)**: temperature=0 is most practical since we run only 20 examples -- variance from sampling would be too high at n=20 with temp>0

---

## 3. max_tokens Setting

### Current Setup: max_tokens=2048

### Is 2048 enough?

| Benchmark | Typical Output Length | 2048 Sufficient? | Recommended |
|-----------|---------------------|-------------------|-------------|
| MMLU | 50-300 tokens | YES | 1024 |
| GPQA | 100-500 tokens | YES | 1024 |
| DROP | 50-500 tokens | YES | 1024 |
| MATH (non-thinking) | 200-2000 tokens | MARGINAL | 4096 |
| MATH (thinking mode) | 2000-10000+ tokens | NO | 16384-32768 |
| HumanEval | 100-800 tokens | YES | 2048-4096 |
| SimpleQA | 50-500 tokens | YES | 1024 |
| LiveBench (all) | Varies | See below | 4096 (LiveBench default) |

### Benchmark-Specific Analysis

**MATH with Chain-of-Thought:**
- Research shows accurate answers require output length exceeding a problem-specific threshold
- Average: fewer than 10,000 tokens for most math problems
- Qwen3 technical report: up to 38,912 tokens per problem for thinking mode
- **2048 will truncate many thinking-mode math solutions**

**HumanEval:**
- Most solutions are under 500 tokens
- Complex solutions (dynamic programming, etc.) can reach 800-1200 tokens
- 2048 is adequate; 4096 provides comfortable headroom
- Standard evaluation setups use max_new_tokens=4096

**GPQA:**
- Multiple choice -- answer is a single letter
- Chain-of-thought reasoning typically 200-500 tokens
- 2048 is more than adequate

**LiveBench defaults:**
- gen_api_answer.py defaults to max_tokens=4096
- Model configs can override: QwQ-32B uses 16000-31000, Qwen3 thinking uses 38912

### Qwen3.5 Official Recommendations
- General queries: max_tokens=32,768
- Complex benchmarks (math/programming competitions): max_tokens=81,920

### Recommendations for max_tokens

| Mode | Recommended max_tokens |
|------|----------------------|
| Non-thinking benchmarks | 4096 (matches LiveBench default) |
| Thinking mode benchmarks | 16384 minimum, 32768 ideal |
| Qwen3.5 full thinking | 32768-81920 |

**Practical recommendation**: Increase from 2048 to **4096** minimum. For thinking mode, use **16384**. The only cost of higher max_tokens is slightly more memory for the output buffer, but generation stops when the model outputs EOS anyway -- unused max_tokens don't consume significant resources.

---

## 4. Number of Parallel Slots

### Current Setup: 1 thread (sequential)

### Does parallelism affect quality?

**Key findings from llama.cpp discussions:**

1. **Quality is NOT affected by parallelism** -- each slot processes independently and produces the same output regardless of how many other slots are active (assuming temperature=0)
2. **However**, with temperature>0, different batching patterns can cause floating-point order-of-operations differences, potentially changing outputs slightly
3. **Speed per request decreases** with more parallel slots because:
   - KV cache memory is multiplied by number of slots
   - GPU compute is shared across slots
   - CPU-side sampling becomes a bottleneck

### Memory Impact of Parallel Slots

For Qwen3.5-35B-A3B (MoE with GQA) at 8K context:
- ~200MB KV cache per slot
- 1 slot: ~200MB KV overhead
- 2 slots: ~400MB KV overhead  
- 4 slots: ~800MB KV overhead
- On 32GB M4: model itself takes ~15-20GB (Q3_K_S), leaving ~12-17GB for KV cache and overhead

### Recommendations for Parallel Slots

**For benchmarking quality**: 1 slot is optimal
- Eliminates any potential for batching-related non-determinism
- Maximizes per-request speed (all GPU compute dedicated to one request)
- Most reproducible results

**For benchmarking throughput** (if time-constrained):
- 2-4 slots is reasonable on 32GB M4 at 8K context
- Beyond 4 slots, diminishing returns and potential memory pressure
- Per-request latency increases, but total throughput improves

**Practical recommendation**: Keep at 1 for benchmarking. Speed is not the bottleneck compared to result quality. If running hundreds of examples, 2 slots could save time without quality impact (at temp=0).

---

## 5. System Prompt

### Current Setup: "You are a helpful assistant." (OPENAI_SYSTEM_MESSAGE_API)

### Analysis of System Prompt Choices

**What the frameworks use:**

| Framework | System Prompt |
|-----------|--------------|
| simple-evals (API models) | "You are a helpful assistant." |
| simple-evals (ChatGPT models) | "You are ChatGPT, a large language model trained by OpenAI..." |
| simple-evals (Claude models) | CLAUDE_SYSTEM_MESSAGE_LMSYS (custom) |
| LiveBench | No system prompt by default (none set in gen_api_answer.py unless question has system_prompt) |
| EleutherAI lm-eval-harness | No system prompt (bare prompts) |

**Key insight**: LiveBench actually uses **no system prompt** by default -- the system prompt is only included if the question itself specifies one. The simple-evals setup adds "You are a helpful assistant." via the ChatCompletionSampler.

### Model-Specific System Prompt Considerations

**Qwen3.5:**
- No specific system prompt recommended in the model card
- The model supports system messages through standard chat templates
- For thinking mode, the model uses `<think>` tags automatically based on template configuration, not system prompt
- Recommended: Use "You are a helpful assistant." or no system prompt

**Abliterated models (Huihui):**
- Abliteration removes refusal behavior
- Standard system prompts work fine
- No special system prompt needed

**Claude-distilled models:**
- The Claude-distilled Qwen3.5 uses `<think>` tags for chain-of-thought
- Standard system prompt works; the thinking behavior comes from the fine-tuning, not the system prompt

**MoE models:**
- No special system prompt considerations for MoE architecture
- The expert routing is internal to the model; system prompts don't affect it

### Recommendations for System Prompt

**"You are a helpful assistant."** is a good universal default:
- Matches OpenAI simple-evals methodology
- Works across all model types
- Doesn't bias the model toward any specific behavior

**Alternative: No system prompt** (matches LiveBench and lm-eval-harness)
- Some models perform differently without a system prompt
- Slightly more "raw" evaluation of model capabilities
- Matches how most leaderboard evaluations are run

**Practical recommendation**: Keep "You are a helpful assistant." for consistency with simple-evals. For LiveBench comparison, consider removing the system prompt to match their methodology.

---

## 6. Thinking Mode / Chain of Thought

### Qwen3.5 Thinking Mode

**Critical finding**: Qwen3.5 operates in thinking mode **by default**. It generates `<think>...</think>` content before producing final responses.

**How to control:**
- Enable: Default behavior, or explicitly set `enable_thinking: true` in chat template kwargs
- Disable: Set `enable_thinking: false` in chat template kwargs
- **Qwen3.5 does NOT support `/think` and `/no_think` soft switching** (unlike Qwen3)

**For benchmarking:**
- Thinking mode produces longer, more detailed responses
- Increases token usage significantly (potentially 5-50x more output tokens)
- May improve accuracy on reasoning tasks (MATH, GPQA) but wastes tokens on simple tasks (MMLU)
- Requires much higher max_tokens (16384-81920 vs 2048-4096)
- Requires larger context window (128K recommended by Qwen)

**With Claude-distilled Qwen3.5 (Huihui model):**
- The model uses `<think>` tags inherited from Claude distillation
- This is NOT the same as Qwen3.5's native thinking mode
- The `<think>` behavior is baked into the fine-tune, not controllable via template
- The thinking content is generated within the normal output stream

### Base Qwen3.5 vs Claude-Distilled

| Feature | Base Qwen3.5 | Huihui Claude-Distilled |
|---------|-------------|------------------------|
| Thinking mode | Native, template-controlled | Baked into fine-tune |
| `<think>` tags | Via chat template | Always present in output |
| Disable thinking | `enable_thinking: false` | Cannot easily disable |
| Token usage | Controllable | Always includes thinking |
| Quality on reasoning | Better with thinking=true | Thinking included by default |

### Gemma 4 Thinking Mode

- **YES, Gemma 4 supports thinking mode**
- Enable with `enable_thinking=True` in the processor/template
- Uses `<|think|>` token in the system prompt
- Available on all sizes (E2B, E4B, 26B-A4B, 31B)
- Disable by removing the `<|think|>` token from system prompt
- "Dramatically improves performance on complex tasks without extra fine-tuning"

### GLM-4.7 Thinking Mode

- **YES, GLM-4.7 supports thinking mode** (interleaved thinking since GLM-4.5)
- Supports **turn-level thinking control**: disable for lightweight turns, enable for complex reasoning
- "Preserved Thinking" mode: retains thinking blocks across multi-turn conversations
- GLM-4.7 achieves 42.8% on HLE benchmark (+12.4% over GLM-4.6)

### Recommendations for Thinking Mode

**For fair benchmarking comparison:**
- Run each model in its **optimal mode** -- thinking mode for reasoning, non-thinking for simple tasks
- OR run all models with thinking disabled for apple-to-apple comparison on raw capability
- Document which mode was used

**For our setup (Claude-distilled Qwen3.5):**
- Thinking is always on (baked into fine-tune)
- Increase max_tokens to at least 4096, ideally 8192-16384
- Increase context window to at least 16384
- The `<think>` content will be included in the response; ensure answer extraction handles it

---

## 7. Number of Examples / Statistical Significance

### Current Setup: 20 questions per benchmark

### Statistical Analysis of n=20

**Standard Error at n=20:**
For binary accuracy (correct/incorrect):
- SE = sqrt(p(1-p)/n)
- At 50% accuracy: SE = sqrt(0.5 * 0.5 / 20) = 0.112 = **11.2%**
- 95% CI = score +/- 1.96 * SE = +/- **21.9%**
- At 70% accuracy: SE = sqrt(0.7 * 0.3 / 20) = 0.102 = **10.2%**
- 95% CI = score +/- **20.1%**

**What this means:**
- If a model scores 70% on 20 questions, the 95% CI is approximately **50% to 90%**
- A 10% difference between models (e.g., 65% vs 75%) is **NOT statistically significant** at n=20
- You need ~25% difference between models to be confident the difference is real

### Minimum Sample Sizes for Various Confidence Levels

| Desired CI Width (95%) | Required n | Assumed accuracy ~50% |
|------------------------|-----------|----------------------|
| +/- 20% | 25 | Very rough estimate |
| +/- 10% | 100 | Minimum for research |
| +/- 5% | 400 | Standard for publications |
| +/- 3% | ~1000 | High confidence |

**Key research findings:**
- CLT-based confidence intervals become unreliable below ~100 samples
- At n < 100, confidence intervals are systematically too narrow (overconfident)
- The sample size required grows quadratically: detecting a gap half the size requires 4x samples
- Bootstrap methods are recommended over CLT for small samples

### What Established Benchmarks Use

| Benchmark | Total Questions | Typical Subset for Evaluation |
|-----------|----------------|------------------------------|
| MMLU | 14,042 test | Full (but subjects vary: 100-1000 per subject) |
| GPQA Diamond | 198 | Full (198) |
| GPQA Main | 448 | Full or subset |
| DROP | ~9,536 dev | Full or large subset |
| MATH | 5,000 test / 500 (MATH-500) | 500 (MATH-500 common) |
| HumanEval | 164 | Full (164) |
| SimpleQA | ~4,326 | Full or large subset |
| LiveBench | Varies per release | Full per category |

### simple-evals n_repeats Strategy

The simple-evals codebase uses an interesting approach to handle variance:
- **GPQA**: n_repeats=10 (default), each of 198 questions run 10 times = 1,980 evaluations
- **MATH**: n_repeats=10-16 (default), massive repetition for reliable averages
- **MMLU**: n_repeats=1 (14K+ questions, large enough)
- **DROP**: n_repeats=1 (9.5K questions)
- **HumanEval**: 1 sample per task (pass@1)

This means simple-evals effectively runs **1,980 evaluations for GPQA** and potentially **80,000 evaluations for MATH** to get reliable numbers. Our 20 questions is dramatically fewer.

### Variance Expected at n=20

**Example**: If true accuracy is 70%:
- With n=20: Expected results range from ~50% to ~90% across random subsets
- With n=100: Expected results range from ~61% to ~79%
- With n=500: Expected results range from ~66% to ~74%

**Across multiple runs** (if temperature > 0):
- Stochastic variance adds on top of sampling variance
- Total variance = between-question variance + within-question variance

### Recommendations for Sample Size

**n=20 is useful for:**
- Quick directional comparison ("Model A is clearly better/worse than Model B")
- Detecting large differences (>25 percentage points)
- Rapid iteration during development
- Getting a rough sense of model capability

**n=20 is NOT sufficient for:**
- Publishing reliable benchmark scores
- Detecting moderate differences (5-15 percentage points)
- Ranking closely-matched models
- Statistical significance claims

**Practical recommendations:**
1. **Keep n=20 for initial screening** -- it's fast and catches major issues
2. **For final comparison**: increase to 50-100 examples per benchmark
3. **Report confidence intervals**: Always note "n=20, 95% CI: +/- ~20%"
4. **For critical decisions**: Use the full benchmark (all available questions)
5. **Run multiple trials if using temp>0**: At least 3 trials, report mean and std

---

## 8. Flash Attention and Other Optimizations

### Current Setup: flash_attention enabled in LM Studio

### Flash Attention on Apple Silicon

**Should it be enabled?**
- **YES** -- flash attention is generally beneficial on Apple Silicon
- Reduces memory usage for the attention computation
- Improves speed, especially at longer context lengths
- llama.cpp supports flash attention via the `-fa` flag
- LM Studio enables it as an option

**Caveats:**
- Flash attention with partial GPU offloading can have issues in some llama.cpp versions
- When using flash attention, full GPU offloading (`-ngl 99`) is recommended
- Metal FlashAttention 2.0 implementations can be up to 94% faster than standard ggml

**Recommendation**: Keep flash_attention enabled. Ensure full GPU offloading.

### num_experts Setting for MoE Models

**Qwen3.5-35B-A3B architecture:**
- Total parameters: 35B
- Active parameters: 3B (A3B = Active 3B)
- This is a Mixture of Experts model

**Should num_experts match the architecture?**
- The model automatically uses the correct number of active experts based on its architecture
- In llama.cpp, you can override with `--override-kv llama.expert_used_count=int:N`
- **Do NOT change this unless you know what you're doing**
- Reducing active experts: faster but lower quality
- Increasing active experts: not possible beyond the architecture's design

**MoE-specific optimizations:**
- `--n-cpu-moe` flag: keeps MoE weights of first N layers on CPU (saves GPU memory)
- Batch size tuning: defaults may be too small for MoE models on CPU+GPU
- Recommended: `-b 2048 -ub 512` for MoE models

### Other Optimizations

**KV Cache Quantization:**
- llama.cpp supports `-ctk q8_0` and `-ctv q8_0` for KV cache quantization
- Reduces KV cache memory by ~50% with minimal quality loss
- Useful if running with larger context windows

**Metal GPU Offloading:**
- Always use `-ngl 99` or equivalent to offload all layers to GPU
- On M4 32GB: the Q3_K_S quantization of Qwen3.5-35B-A3B should fit entirely in GPU
- Partial offloading causes major speed degradation

**Batch Size:**
- `-b 512` (logical batch) and `-ub 512` (physical batch) are reasonable defaults
- Larger batch sizes improve prompt processing speed but use more memory
- For single-request benchmarking, default batch sizes are fine

### Recommendations Summary

| Setting | Recommendation | Reason |
|---------|---------------|--------|
| Flash attention | ENABLE | Speed + memory improvement |
| GPU offloading | Full (`-ngl 99`) | Required for good performance |
| num_experts | DO NOT CHANGE | Use model's default |
| KV cache quantization | Consider q8_0 if memory-tight | Minimal quality loss |
| Batch size | Default (512) | Sufficient for single-slot benchmarking |

---

## Summary: Optimal Benchmarking Configuration

### Recommended Settings for Non-Thinking Mode

```
Context window:  8192-16384  (sufficient for most benchmarks)
Temperature:     0           (reproducible, matches leaderboard standards)
max_tokens:      4096        (matches LiveBench default)
Parallel slots:  1           (maximum quality and reproducibility)
System prompt:   "You are a helpful assistant."  (or none, to match LiveBench)
Flash attention: ENABLED
GPU offloading:  FULL (-ngl 99)
```

### Recommended Settings for Thinking Mode (if enabled)

```
Context window:  32768-65536     (Qwen recommends 128K, but constrained by 32GB RAM)
Temperature:     0.6             (Qwen3.5 official for precise tasks)
top_p:           0.95
top_k:           20
presence_penalty: 1.5            (Qwen3.5 specific, prevents repetition)
max_tokens:      16384-32768     (thinking chains can be very long)
Parallel slots:  1
System prompt:   "You are a helpful assistant."
Flash attention: ENABLED
GPU offloading:  FULL
```

### Recommended Settings Matching Qwen3.5 Official (Non-Thinking)

```
Context window:  16384+
Temperature:     0.7             (general tasks)
top_p:           0.8
top_k:           20
presence_penalty: 1.5
max_tokens:      4096-8192
```

### Critical Changes from Current Setup

| Parameter | Current | Recommended | Priority |
|-----------|---------|-------------|----------|
| max_tokens | 2048 | **4096** (non-thinking) / **16384** (thinking) | HIGH |
| Temperature | 0.5 | **0** (leaderboard) or **0.6-0.7** (Qwen optimal) | MEDIUM |
| Context window | 8192 | **16384** (if thinking mode active) | MEDIUM |
| Sample size | 20 | **50-100** (for reliable comparison) | LOW (for screening, 20 is OK) |
| Parallel slots | 1 | 1 (keep) | N/A |
| Flash attention | enabled | enabled (keep) | N/A |
| System prompt | "You are a helpful assistant." | Keep or remove | LOW |

---

## Sources

### Context Window and Memory
- [llama.cpp Performance on Apple Silicon](https://github.com/ggml-org/llama.cpp/discussions/4167)
- [Practical Long-Context LLM Inference with llama.cpp](https://nullmirror.com/en/blog/2025-11-01-practical-long-context-llm-inference-with-llama.cpp/)
- [LLM VRAM Requirements Explained](https://techtactician.com/llm-gpu-vram-requirements-explained/)
- [llama.cpp KV Cache Discussion](https://github.com/ggml-org/llama.cpp/discussions/9784)
- [llama.cpp Memory Allocation](https://github.com/ggml-org/llama.cpp/discussions/9936)
- [MLX vs llama.cpp on Apple Silicon](https://groundy.com/articles/mlx-vs-llamacpp-on-apple-silicon-which-runtime-to-use-for-local-llm-inference/)

### Temperature and Sampling
- [Does Temperature 0 Guarantee Deterministic Outputs?](https://www.vincentschmalbach.com/does-temperature-0-guarantee-deterministic-llm-outputs/)
- [Why Temperature=0 Doesn't Guarantee Determinism](https://mbrenndoerfer.com/writing/why-llms-are-not-deterministic)
- [How to Get Consistent LLM Outputs in 2025](https://www.keywordsai.co/blog/llm_consistency_2025)
- [LLM Temperature, Top-P, Top-K Guide](https://amitray.com/llm-parameters-temperature-top-p-top-k-guide/)

### Model-Specific Settings
- [Qwen3.5-35B-A3B Model Card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [Qwen3.5 Thinking Mode Discussion](https://huggingface.co/Qwen/Qwen3.5-9B/discussions/13)
- [Qwen3.5 /think /no_think Support](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/discussions/23)
- [Gemma 4 Thinking Mode](https://ai.google.dev/gemma/docs/capabilities/thinking)
- [GLM-4.7 Thinking Mode](https://docs.z.ai/guides/capabilities/thinking-mode)
- [Gemma 4 Practical Guide](https://dev.to/arshtechpro/gemma-4-a-practical-guide-for-developers-2co5)

### Statistical Methodology
- [Applying Statistics to LLM Evaluations](https://cameronrwolfe.substack.com/p/stats-llm-evals)
- [Benchmark2: Systematic Evaluation of LLM Benchmarks](https://arxiv.org/pdf/2601.03986)
- [On Robustness and Reliability of Benchmark-Based Evaluation](https://arxiv.org/pdf/2509.04013)

### Parallel Inference
- [Optimal Parameters for Parallel Inference](https://github.com/ggml-org/llama.cpp/discussions/18308)
- [llama.cpp Parallelization Explanation](https://github.com/ggml-org/llama.cpp/discussions/4130)
- [Benchmarking llama.cpp Parallelism](https://medium.com/@ferraricorneloup.teo/how-many-developers-can-one-gpu-serve-benchmarking-llama-cpp-parallelism-on-a40-gpus-0ea2a8c36045)

### MoE Optimization
- [Guide to Optimizing MoE Inference in llama.cpp](https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0)
- [Performant MoE CPU Inference with GPU Acceleration](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)
- [DavidAU MoE Expert Management Guide](https://huggingface.co/DavidAU/How-To-Set-and-Manage-MOE-Mix-of-Experts-Model-Activation-of-Experts)

### Benchmarking Frameworks
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [OpenAI simple-evals](https://github.com/openai/simple-evals)
- [HuggingFace Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- [EleutherAI lm-eval-harness max_new_tokens Issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/2730)
- [30 LLM Evaluation Benchmarks and How They Work](https://www.evidentlyai.com/llm-guide/llm-benchmarks)

### Flash Attention
- [Metal FlashAttention 2.0](https://engineering.drawthings.ai/p/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c)

### Benchmark-Specific
- [GPQA Paper](https://arxiv.org/abs/2311.12022)
- [DROP Paper](https://arxiv.org/abs/1903.00161)
- [HumanEval Paper](https://arxiv.org/abs/2107.03374)
- [LLM Response Length Over Time (Epoch AI)](https://epoch.ai/data-insights/output-length/)
- [Relationship Between Reasoning and Performance](https://arxiv.org/html/2502.15631v1)
