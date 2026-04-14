# Practical Local LLM Benchmarking on Apple Silicon: Real-World Setups

Research compiled April 2026. Focus: actual published setups, tools, and methodologies people use to benchmark local LLMs on Mac hardware.

---

## 1. Dedicated Apple Silicon Benchmark Tools

### local-llm-bench (famstack-dev) -- RECOMMENDED STARTING POINT
- **Repo:** https://github.com/famstack-dev/local-llm-bench
- **What it does:** Scenario-based benchmarking that measures *effective throughput* (output_tokens / total_wall_clock_time), not just the generation tok/s counter. Addresses the key insight: "57 tok/s on screen, 3 tok/s in practice" because prefill time dominates at longer contexts.
- **Core metric:** `effective tok/s = output_tokens / (prefill_time + generation_time)`
- **Supported backends:** Ollama (default:11434), LM Studio (:1234), oMLX (:8000), llama-server/llama.cpp (:8090), any OpenAI-compatible endpoint
- **Four test scenarios:**
  - `ops-agent` - 8-turn conversation (agent workflows with growing context)
  - `doc-summary` - 5 single-shot turns (long input, short classification output)
  - `prefill-test` - 4 single-shot turns (prefill scaling from 655-8.5K tokens)
  - `creative-writing` - 3 single-shot turns (short prompt, long 2K-token output)
- **Quick start (no dependencies, Python 3.8+):**
  ```bash
  # Ollama
  python3 bench.py --model llama3.1:8b --model-label llama-3.1-8b-instruct

  # LM Studio
  python3 bench.py --backend lmstudio --model mlx-community/qwen3.5-35b-a3b \
    --model-label qwen3.5-35b-a3b --no-think

  # Any OpenAI-compatible endpoint
  export OPENAI_API_KEY=omlx
  python3 bench.py --backend openai --backend-label omlx \
    --base-url http://localhost:8000 --model "Qwen3.5-35B-A3B-4bit" \
    --model-label qwen3.5-35b-a3b --no-think

  # Compare results
  python3 compare.py results/<model>/<scenario>/*.json
  ```
- **Results auto-save** to `results/<model>/<scenario>/<chip>_<backend>.json`
- **Key finding:** At 8.5K tokens of context on M3 Max, MLX's effective throughput drops to 3 tok/s despite reporting 51 tok/s generation speed. oMLX's tiered KV cache achieved 71.3 tok/s effective throughput for Qwen3.5-35B, nearly double M1 Max.
- **Tuning tips:** Ollama: `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q4_0`. LM Studio MLX: increase prefill chunk size from 512 to 4096.
- Source: https://famstack.dev/guides/mlx-vs-gguf-apple-silicon/

### MLX Transformers Benchmark (aukejw)
- **Repo:** https://github.com/aukejw/mlx_transformers_benchmark
- **Results site:** https://aukejw.github.io/mlx_transformers_benchmark/
- **What it does:** Benchmarks LLM inference across frameworks (PyTorch MPS, MLX, LM Studio, Ollama, llama.cpp) at different quantization levels (bfloat16, int8, int4).
- **Setup:**
  ```bash
  git clone git@github.com:aukejw/mlx_transformers_benchmark.git
  cd mlx_transformers_benchmark
  make setup    # Creates Python 3.11 environment via uv
  make test     # Validates GPU availability

  # Quick test (Qwen2.5 0.5B model)
  uv run python scripts/run_llm_benchmarks.py --run_only_benchmarks qwen-2.5-0.5b-it \
    --dtypes '["int4","int8"]' --num_iterations 3

  # Full benchmark
  make run-llm-benchmarks   # Tests all data types
  make show-llm-benchmarks  # Generates HTML report
  ```
- **Notable:** Includes cooldown parameters to prevent thermal throttling on MacBook Pro. Results published as interactive HTML.

### LocalScore
- **Site:** https://www.localscore.ai
- **What it does:** Open-source benchmark (built on Llamafile, Apache 2.0) that measures three metrics combined into a single score:
  1. Prompt Processing Speed (tokens/s)
  2. Generation Speed (tokens/s)
  3. Time to First Token (ms)
- **Score scale:** 1000 = excellent, 250 = acceptable, below 100 = poor
- **Test methodology:** 8 distinct tests from classification (1024 prompt/16 gen tokens) to creative writing (16 prompt/1536 gen tokens)
- **Standard reference models:**
  - Tiny: Llama 3.2 1B (Q4_K_M, ~2GB)
  - Small: Llama 3.1 8B (Q4_K_M, ~6GB)
  - Medium: Qwen 2.5 14B (Q4_K_M, ~10GB)
- **Usage:** Download CLI executable, run benchmark, optionally submit results to public database. Can benchmark any .gguf file.

### llmfit -- Hardware Compatibility + Recommendations
- **Repo:** https://github.com/AlexsJones/llmfit
- **What it does:** Detects your hardware (RAM, CPU, GPU/VRAM) and scores hundreds of models on quality, speed, fit, and context. Auto-selects MLX on Apple Silicon.
- **Install:** `brew install llmfit` or `curl -fsSL https://llmfit.axjns.dev/install.sh | sh`
- **Usage:**
  ```bash
  llmfit                                          # Interactive TUI
  llmfit fit --perfect -n 5                       # Top 5 models that fit perfectly
  llmfit recommend --json --use-case coding       # JSON output for scripting
  llmfit plan "Model-Name" --context 8192         # Memory plan for specific model
  ```
- **Speed estimation formula:** `(bandwidth_GB_s / model_size_GB) * 0.55 efficiency_factor`
  - Metal speed constant: 160 tok/s baseline
- **Fit categories:** Perfect (recommended GPU memory), Good (fits with headroom), Marginal (tight/CPU-only), Too Tight
- **Apple Silicon detection:** Queries `system_profiler`, sets VRAM = system RAM

### Benchmarks LLM Silicon (ivanfioravanti)
- **Repo:** https://github.com/ivanfioravanti/benchmarks_llm_silicon
- **What it does:** Community repository collecting performance data across Apple Silicon generations (M2, M3 Ultra, M4 Max). Compares MLX, llama.cpp, LM Studio, Ollama.
- **Metrics stored:** Inference speed, memory consumption, token generation rates, model loading times.

---

## 2. Ollama Benchmark Automation Tools

### aidatatools/ollama-benchmark
- **Repo:** https://github.com/aidatatools/ollama-benchmark
- **Install:** `pip install llm-benchmark` (or `pipx`, `uv pip`)
- **What it does:** Auto-detects RAM and selects appropriate model set:
  - 4-7GB: deepseek-r1:1.5b, gemma:2b, phi:2.7b
  - 7-15GB: phi3:3.8b, gemma2:9b, mistral:7b, llama3.1:8b
  - 15-31GB: phi4:14b, deepseek-r1:14b
  - 31GB+: phi4:14b, gpt-oss:20b
- **Run:** `llm_benchmark run`
- **Options:** `--no-sendinfo` (don't upload results), `--custombenchmark=file.yml` (custom model list)
- **Custom YAML:**
  ```yaml
  file_name: "custombenchmarkmodels.yml"
  version: 2.0.custom
  models:
    - model: "deepseek-r1:1.5b"
    - model: "qwen:0.5b"
  ```

### cloudmercato/ollama-benchmark
- **Repo:** https://github.com/cloudmercato/ollama-benchmark
- **Install:** `pip install https://github.com/cloudmercato/ollama-benchmark/archive/refs/heads/main.zip`
- **Features:**
  - Speed testing with concurrent requests using MT-Bench dataset questions
  - Embedding performance across languages
  - Model loading time measurement
  - **LLM-as-Judge quality evaluation** (judge model rates and provides feedback)
  - Security testing (censorship bypass resilience)
  - Built-in system monitoring (probes for CPU/GPU/memory)
- **Output:** YAML-formatted with statistical measures (mean, std dev, percentiles)

### CordatusAI/ollama-benchmark
- **Repo:** https://github.com/CordatusAI/ollama-benchmark
- **What it does:** Streamlit-based GUI. Select multiple models via checkboxes, auto-pulls missing models, measures output speed (tok/s) and prompt speed (tok/s). Results in `benchmark_results.csv`.
- **Run:** `streamlit run llm_benchmark_tool.py`
- **Note:** Primarily targets NVIDIA GPUs but works on Mac via Ollama.

### Walter Deane's Bash Script Approach
- **Source:** https://medium.com/@walterdeane/benchmarking-local-llms-with-ollama-and-a-simple-bash-script-8fdb5baf5456
- **Methodology:**
  1. Model warmup: `ollama run $MODEL --prompt 'Say hello.'`
  2. Timed execution: `START_TIME=$(date +%s.%N)` / `END_TIME=$(date +%s.%N)`
  3. CSV logging: timestamp, model name, prompt type, status, duration, output path
  4. Raw output saved to text files (ANSI stripped)
- **Key insight:** Speed often matters more than capability in practice; smaller models with faster streaming feel more usable than slower larger ones.

### kaushall13/local-llm-benchmark
- **Repo:** https://github.com/kaushall13/local-llm-benchmark
- **What it does:** Modular Python toolkit using Ollama. Measures TTFT, tok/s, TPM, peak CPU/RAM/GPU usage with **real-time dashboard**.
- **Config format (JSON):**
  ```json
  {
    "models": [{"name": "qwen2.5:1.5b"}],
    "parameters": {
      "num_ctx": [2048, 4096],
      "num_predict": [128, 512],
      "num_gpu": [0, -1],
      "temperature": [0.7]
    },
    "benchmark_settings": {
      "trials": 3,
      "output_prefix": "results",
      "ollama_host": "http://localhost:11434"
    }
  }
  ```
- **Real-time monitoring:** Interactive dashboard showing generation tok/s, TPM, CPU/GPU/RAM usage, and live text output.
- **Output:** CSV with timestamps, plus console summary table.

### BenchLlama (srikanth235)
- **Repo:** https://github.com/srikanth235/benchllama
- **Install:** `pip install benchllama`
- **What it does:** Code generation benchmark using HumanEval dataset (bigcode/humanevalpack). Supports Python, JavaScript, Java, Go, C++.
- **Connects to Ollama** at `http://localhost:11434`
- **Multi-model:** Pass multiple model names to test simultaneously
- **Key metric:** pass@k calculation for code correctness

---

## 3. LM Studio Automation for Benchmarking

### CLI (lms) -- Full Scripting Capability
LM Studio's CLI runs headless (no GUI required). Key commands for benchmark automation:

```bash
# Model management
lms load <model-name> --gpu=max --context-length=8192
lms load <model-name> --ttl 3600           # Auto-unload after 1hr idle
lms unload --all
lms ps --json                              # Machine-readable loaded model list
lms ls                                     # List all downloaded models

# Server control
lms server start
lms server stop
lms server status

# Headless mode (no GUI)
lms daemon up
lms daemon down

# Example benchmark automation script
#!/bin/bash
lms daemon up
for model in "qwen3-8b" "llama3.1-8b" "gemma2-9b"; do
  lms load "$model" --gpu=1.0 --context-length=8192
  lms server start
  # Run your benchmark against localhost:1234
  python my_benchmark.py --base-url http://localhost:1234/v1
  lms server stop
  lms unload --all
done
lms daemon down
```

### Python SDK
```python
import lmstudio as lms

# Load model
model = lms.llm("qwen/qwen3-4b-2507")

# Load with TTL (auto-unload)
model = lms.llm("qwen/qwen3-4b-2507", ttl=3600)

# Load new instance with custom identifier
client = lms.get_default_client()
model = client.llm.load_new_instance("qwen/qwen3-4b-2507", "benchmark-model")

# Unload
model.unload()
```

### REST API
LM Studio 0.4.0+ has native REST API at `/api/v1/*` endpoints:
- Load/unload models
- Set TTL per-model in request payload
- OpenAI-compatible `/v1/chat/completions` and `/v1/completions`
- Docs: https://lmstudio.ai/docs/developer/rest/endpoints

---

## 4. Academic Benchmark Suites That Work with Local Models

### lm-evaluation-harness (EleutherAI) -- The Standard
- **Repo:** https://github.com/EleutherAI/lm-evaluation-harness
- **YES, it talks to local OpenAI-compatible APIs** via `local-completions` and `local-chat-completions` model types.
- **Setup:**
  ```bash
  git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
  cd lm-evaluation-harness
  pip install -e ".[api]"
  # For ifeval benchmark:
  pip install langdetect immutabledict
  ```
- **Run against local API (completions endpoint):**
  ```bash
  lm_eval --model local-completions \
    --tasks hellaswag \
    --output_path ./results \
    --log_samples \
    --model_args model=my-model,base_url=http://localhost:1234/v1/completions,\
  num_concurrent=8,max_retries=3,timeout=3000,seed=1234,temperature=0,\
  tokenizer=path/to/tokenizer
  ```
- **Run against local API (chat endpoint):**
  ```bash
  lm_eval --model local-chat-completions \
    --tasks gsm8k_cot_llama,ifeval \
    --model_args model=meta-llama/Meta-Llama-3.1-8B-Instruct,\
  base_url=http://localhost:1234/v1/chat/completions,\
  num_concurrent=32,max_retries=3,tokenized_requests=False \
    --apply_chat_template --fewshot_as_multiturn
  ```
- **Key parameters:**
  - `base_url` - your local server endpoint
  - `num_concurrent` - parallel requests (increase for throughput, reduce if server crashes)
  - `timeout` - extend to 3000+ for slow local models
  - `tokenizer` - path to HuggingFace tokenizer (important for accurate token counting)
- **IMPORTANT LIMITATION:** Log-likelihood/MCQ tasks (like MMLU) only work with completion endpoints, NOT chat-completion endpoints. Use `--apply_chat_template` with completion APIs for chat formatting while retaining logit access.
- **Available benchmarks (partial list):** MMLU, HellaSwag, ARC, WinoGrande, GSM8K, TruthfulQA, IFEval, GPQA, and hundreds more.
- **This is what the Open LLM Leaderboard uses internally.** You can replicate leaderboard evaluations locally by running the same tasks with the same few-shot settings.
- Source: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/API_guide.md

### LightEval (HuggingFace) -- YES, supports local endpoints via LiteLLM
- **Repo:** https://github.com/huggingface/lighteval
- **Setup with local endpoint:**
  ```bash
  # Install
  pip install lighteval

  # Create config file (e.g., local_config.yaml):
  ```
  ```yaml
  model_parameters:
    model_name: "openai/my-local-model"
    base_url: "http://localhost:1234/v1"
    api_key: ""  # empty for local
    generation_parameters:
      temperature: 0.5
      max_new_tokens: 256
      top_p: 0.9
      seed: 0
  ```
  ```bash
  # Run
  lighteval endpoint litellm \
    "provider=openai,model_name=my-model" \
    gsm8k
  ```
- Works with any OpenAI-compatible endpoint (Ollama, LM Studio, vLLM, llama-server).
- Source: https://huggingface.co/docs/lighteval/use-litellm-as-backend

### simple-evals (OpenAI) -- Requires Minor Modification for Local
- **Repo:** https://github.com/openai/simple-evals
- **Benchmarks included:** MMLU, MATH, GPQA, DROP, MGSM, HumanEval, SimpleQA, BrowseComp, HealthBench
- **Local endpoint support:** NOT built-in. The `ChatCompletionSampler` creates `OpenAI()` client with no `base_url` parameter. However, the fix is simple since it uses the standard `openai` Python library:
  - **Option 1:** Set `OPENAI_BASE_URL=http://localhost:1234/v1` environment variable (the `openai` library reads this automatically)
  - **Option 2:** Modify `sampler/chat_completion_sampler.py` constructor to accept `base_url` and pass to `OpenAI(base_url=base_url)`
- **As of July 2025:** No longer updated for new models, but reference implementations remain.
- Uses zero-shot chain-of-thought prompting (not few-shot).

### MT-Bench (Multi-Turn) -- Runs Locally via FastChat
- **Repo:** https://github.com/lm-sys/FastChat
- **What it does:** 80 challenging multi-turn open-ended questions across 8 categories. Uses LLM-as-judge (GPT-4) to score responses 1-10.
- **Setup:**
  ```bash
  git clone https://github.com/lm-sys/FastChat.git
  cd FastChat
  pip install -e ".[model_worker,llm_judge]"

  # Generate model answers
  python gen_model_answer.py --model-path [MODEL-PATH] --model-id [MODEL-ID]
  # Results saved to data/mt_bench/model_answer/[MODEL-ID].jsonl

  # Judge with GPT-4
  python gen_judgment.py --model-list [MODEL-ID] --judge-model gpt-4
  ```
- **Cost note:** Requires GPT-4 API calls for judging (~$10-15 per model evaluation).
- 3.3K human annotations available at `lmsys/mt_bench_human_judgments` for calibration.

### AlpacaEval -- Local with API Judge
- **Repo:** https://github.com/tatsu-lab/alpaca_eval
- **Install:** `pip install -e .` (from fork)
- **Evaluate local model:**
  ```bash
  alpaca_eval evaluate_from_model --model_configs path/to/model_config.yaml
  ```
- **Evaluator:** `alpaca_eval_gpt4_turbo_fn` (recommended). Requires `OPENAI_API_KEY`.
- **Correlation:** 0.98 Spearman with ChatBot Arena, costs <$10 OpenAI credits, runs in <3 minutes.
- **For fully local evaluation:** Could substitute a local judge model, though this degrades evaluation quality.

---

## 5. Framework Comparison Results (Published)

### MLX vs llama.cpp on Apple Silicon (famstack.dev, 2026)
**Hardware:** Mac Studio M1 Max, 64GB unified memory
**Model:** Qwen3.5-35B-A3B

| Scenario | GGUF (llama.cpp) | MLX | Winner |
|----------|-----------------|-----|--------|
| Document Classification (short out) | 16-21 effective tok/s | 11-16 effective tok/s | GGUF |
| Prefill Scaling (8.5K context) | 43.4s total | 52.3s total | GGUF |
| Agent Conversation (8 turns) | Varies | Wins most turns (longer output) | MLX |
| Creative Writing (long output) | Lower displayed speed | 57 tok/s displayed | MLX |

**Critical findings:**
- Choose GGUF when output is short relative to input (classification, tool-calling, RAG)
- Choose MLX when output exceeds input (creative writing, summaries, long explanations)
- Crossover: At 600-token context, MLX needs 250+ output tokens to overcome prefill disadvantage
- oMLX with tiered KV cache: 5x prefill speedup (49s -> 1.7s at 8K context), best of both worlds

### 2026 Framework Benchmark (macgpu.com)
**Hardware:** M4 Pro, 64GB unified memory, 273 GB/s bandwidth
**Model:** DeepSeek V3 (Q4_K_M GGUF / MLX 4-bit)

| Framework | Single-User | 32-User Throughput | TTFT | Best For |
|-----------|------------|-------------------|------|----------|
| vllm-mlx | 42 t/s | 1,150 t/s | ~120ms | Production agent fleets |
| Ollama v0.8+ | 58 t/s | 720 t/s | ~45ms | Development, lowest latency |
| llama.cpp (Metal) | 52 t/s | 890 t/s | ~85ms | Edge/embedded, max control |

### Academic Study (arxiv:2511.05502)
**Hardware:** Mac Studio M2 Ultra, 192GB unified memory
**Model:** Qwen-2.5 family (100-100K token prompts)

| Framework | Strength |
|-----------|----------|
| MLX | Highest sustained generation throughput |
| MLC-LLM | Lowest TTFT for moderate prompts |
| llama.cpp | Most efficient for lightweight single-stream |
| Ollama | Best developer ergonomics, lags in throughput |
| PyTorch MPS | Limited by memory constraints on large models |

### Ollama 0.19 MLX Backend (March 2026)
Ollama switched Apple Silicon backend from llama.cpp to MLX:

| Metric | Ollama 0.18 (llama.cpp) | Ollama 0.19 (MLX) | Improvement |
|--------|------------------------|-------------------|-------------|
| Prefill (NVFP4) | 1,154 tok/s | 1,810 tok/s | +57% |
| Decode (NVFP4) | 58 tok/s | 112 tok/s | +93% |
| Decode (int4) | - | 134 tok/s | - |

**Note:** Only Qwen3.5-35B-A3B is MLX-accelerated in the 0.19 preview. More architectures coming.
Source: https://ollama.com/blog/mlx

---

## 6. Practical Considerations

### Model Warmup
- **Why it matters:** First inference after loading is slower due to GPU memory allocation, KV cache initialization, and Metal shader compilation.
- **Standard practice:** Run a short warmup prompt before measuring: `ollama run $MODEL --prompt 'Say hello.'`
- **Multiple iterations:** Most tools run 3+ iterations and report averages. Some discard the first run.
- Source: Walter Deane's methodology, aukejw benchmark (includes cooldown for thermal throttling)

### Measuring Performance Properly
- **Three distinct metrics:**
  1. Time-to-First-Token (TTFT) -- prefill latency
  2. Generation speed (tok/s) -- what the UI shows
  3. Effective throughput (tok/s) -- output_tokens / total_wall_time (what you actually experience)
- **Most benchmarks only show #2.** The famstack-dev tool specifically addresses this gap.
- **Prefill dominates:** At 8.5K context, prefill accounts for 94% of MLX's total time.

### Memory Monitoring on Apple Silicon

**Tools:**
- **macmon** (recommended, no sudo): `brew install macmon` -- real-time CPU/GPU/ANE power, temps, and memory. JSON output for logging.
  - Repo: https://github.com/vladkens/macmon
- **asitop** (requires sudo): Uses powermetrics. Shows CPU/GPU utilization, frequency, ANE, RAM/swap, power consumption.
  - Repo: https://github.com/tlkh/asitop
- **powermetrics** (built-in): `sudo powermetrics --samplers gpu --show-usage --interval 1000`
- **Activity Monitor:** Memory Pressure graph (green=OK, yellow=warning, red=swapping)
- **ollama ps:** Shows GPU memory percentage for loaded models

### Detecting GPU Memory Spill / CPU Fallback
- **The problem:** Apple Silicon caps GPU memory at ~75% of unified RAM (Metal driver design):
  - 128GB Mac: ~96GB GPU
  - 64GB Mac: ~48GB GPU
  - 32GB Mac: ~21-24GB GPU
- **How to detect spill:**
  1. Sudden tok/s drop (40 tok/s -> 8 tok/s, or 50 tok/s -> 2-5 tok/s)
  2. `ollama ps` shows less than 100% GPU
  3. Activity Monitor Memory Pressure turns red
  4. Ollama server logs show `recommendedMaxWorkingSetSize` values
- **Override the GPU memory limit:**
  ```bash
  sudo sysctl iogpu.wired_limit_mb=122880   # For 120GB on 128GB system
  # Verify:
  sysctl iogpu.wired_limit_mb
  ```
  Takes effect immediately, no reboot needed. To persist: add to `/etc/sysctl.conf` (requires SIP disabled).
- **Practical strategies:**
  - Use quantized models (Q4_K_M saves ~60% vs FP16)
  - Reduce context length (KV cache grows linearly)
  - Leave 8-16GB buffer for system stability

### Quantization Quality Loss
- **Perplexity increase (lower = better, measured on Llama 7B):**
  - Q8_0: +0.0004 ppl (virtually lossless)
  - Q6_K: +0.0008 ppl
  - Q5_K_M: +0.0796 ppl (~95-99% of original quality)
  - Q4_K_M: +0.0535 ppl (balanced, good for most tasks)
  - Q4_0: +0.2499 ppl (noticeable quality loss)
- **Task sensitivity:** MMLU (factual retrieval) is more sensitive to quantization than creative tasks. Below Q5_K_M is risky for knowledge-intensive applications.
- **Sweet spot for Apple Silicon:** Q4_K_M for memory-constrained setups, Q5_K_M or Q8_0 when you have headroom.
- Source: https://github.com/ggml-org/llama.cpp/discussions/2094

---

## 7. Quality Evaluation Methodologies People Actually Use

### Digital Spaceport's "Grade 1" Test Set (10 questions)
- **Source:** https://digitalspaceport.com/about/testing-local-llms/
- **Philosophy:** Test general user expectations, not model-optimized interactions.
- **10 test questions covering:**
  - Ethical reasoning ("Armageddon with a Twist")
  - Code generation ("Coding Flippyblock Extreme")
  - Counting/parsing ("Parsing Peppermints")
  - Pattern recognition ("Arbitrary Arrays")
  - Instruction following ("Cat Sentence Parsing")
  - Math reasoning ("Numeric Comparison")
  - Knowledge recall ("Hundred Decimals of Pi")
  - Spatial reasoning ("SVG Creation")
  - Timeline reasoning ("Pico de Gato")
  - Multi-step calculation ("Two Driver Problem")
- Uses highest precision possible (FP16 preferred), minimum 8K context, 1 tok/s floor.

### LLM-as-Judge Pattern
Several tools implement this:
- cloudmercato/ollama-benchmark uses a judge model to rate and provide feedback
- MT-Bench uses GPT-4 as judge (1-10 scale per turn)
- AlpacaEval uses GPT-4 Turbo as judge (win-rate vs reference)
- For fully local: use a stronger local model to judge weaker ones (e.g., Qwen 72B judging 8B models), though quality degrades vs GPT-4 judge

### Run-and-Compare CSV Pattern
Most practical setups follow this workflow:
1. Define model list and prompts
2. Loop: load model -> run prompts -> capture output + timing -> save to CSV -> unload
3. Optionally run judge model on outputs
4. Compare via spreadsheet, Plotly charts, or custom HTML

---

## 8. All-in-One / Workflow Tools

### yzyydev/local-llm-benchmark
- **Repo:** https://github.com/yzyydev/local-llm-benchmark
- Python backend that benchmarks across multiple providers (Ollama, OpenAI, Claude, Google)
- Good for comparing local vs cloud models on the same prompts

### LLM Benchmarker Suite (FormulaMonks)
- **Repo:** https://github.com/FormulaMonks/llm-benchmarker-suite
- Combines: static evaluations (BoolQ, HellaSwag via OpenCompass), LLM-as-judge (MT-Bench via FastChat), OpenAI Evals
- **Note:** Requires CUDA, so not directly usable on Mac without modification

### liteLLM as Universal Proxy
- **Docs:** https://docs.litellm.ai
- Translates between different LLM API formats. Useful as middleware:
  - Run local model via Ollama/LM Studio
  - Point liteLLM at it
  - Run any evaluation suite that speaks OpenAI API against liteLLM
- Compatible with lm-eval-harness, LightEval, and any OpenAI-SDK-based tool

---

## 9. Replicating the Open LLM Leaderboard Locally

The Open LLM Leaderboard is just a wrapper around lm-evaluation-harness. To replicate:

1. Set up lm-eval-harness (see Section 4)
2. Start your local model server (Ollama, LM Studio, etc.)
3. Run the same tasks with the same few-shot settings:
   ```bash
   # Example: ARC Challenge (25-shot)
   lm_eval --model local-completions \
     --tasks arc_challenge \
     --num_fewshot 25 \
     --model_args model=my-model,base_url=http://localhost:1234/v1/completions

   # Example: HellaSwag (10-shot)
   lm_eval --model local-completions \
     --tasks hellaswag \
     --num_fewshot 10 \
     --model_args model=my-model,base_url=http://localhost:1234/v1/completions

   # Example: MMLU (5-shot)
   lm_eval --model local-completions \
     --tasks mmlu \
     --num_fewshot 5 \
     --model_args model=my-model,base_url=http://localhost:1234/v1/completions
   ```
4. **Important:** Use completion endpoints (not chat) for log-likelihood tasks like MMLU, HellaSwag, ARC.
5. Results may differ slightly from the leaderboard due to different tokenizers, prompt formatting, and quantization effects.

---

## 10. Summary: Recommended Setup for Benchmarking 15+ Models on M4 Mac

### Speed/Performance Benchmarking
1. **Use local-llm-bench** (famstack-dev) for effective throughput across real scenarios
2. Supplement with **LocalScore** for standardized cross-hardware comparison
3. Monitor with **macmon** (no sudo, JSON output for logging)

### Quality Benchmarking
1. **lm-eval-harness** with `local-completions` or `local-chat-completions` for academic benchmarks (MMLU, HellaSwag, ARC, GSM8K)
2. **MT-Bench** via FastChat for multi-turn conversational quality (requires GPT-4 API for judging)
3. Custom prompt set (like Digital Spaceport's 10 questions) for task-specific evaluation

### Automation Loop for Multiple Models
```bash
#!/bin/bash
# Using LM Studio CLI for model management
lms daemon up
lms server start

MODELS=("model1" "model2" "model3" ... "model15")

for model in "${MODELS[@]}"; do
  echo "=== Benchmarking $model ==="
  lms load "$model" --gpu=max --context-length=8192
  sleep 5  # Wait for model to fully load

  # Speed benchmark
  python3 bench.py --backend lmstudio --model "$model" --model-label "$model"

  # Quality benchmark (subset of lm-eval tasks)
  lm_eval --model local-chat-completions \
    --tasks gsm8k,arc_easy,hellaswag \
    --model_args model="$model",base_url=http://localhost:1234/v1/chat/completions,\
  num_concurrent=4,timeout=3000 \
    --output_path "./results/$model"

  lms unload --all
done

lms server stop
lms daemon down
```

Alternative with Ollama (simpler model management):
```bash
#!/bin/bash
MODELS=("llama3.1:8b" "qwen2.5:7b" "gemma2:9b" "mistral:7b" "phi4:14b")

for model in "${MODELS[@]}"; do
  echo "=== Benchmarking $model ==="
  ollama pull "$model"   # Ensure model is downloaded

  # Speed benchmark (famstack-dev tool works with Ollama by default)
  python3 bench.py --model "$model" --model-label "$model"

  # Quality benchmark
  lm_eval --model local-chat-completions \
    --tasks gsm8k,hellaswag \
    --model_args model="$model",base_url=http://localhost:11434/v1/chat/completions,\
  num_concurrent=4,timeout=3000 \
    --output_path "./results/$model"

  ollama stop "$model"
done
```

---

## Sources

### Tools & Repositories
- [local-llm-bench](https://github.com/famstack-dev/local-llm-bench) - Scenario-based Apple Silicon benchmark
- [MLX Transformers Benchmark](https://github.com/aukejw/mlx_transformers_benchmark) - Framework comparison tool
- [LocalScore](https://www.localscore.ai/blog) - Standardized local LLM benchmark
- [llmfit](https://github.com/AlexsJones/llmfit) - Hardware compatibility scanner
- [Benchmarks LLM Silicon](https://github.com/ivanfioravanti/benchmarks_llm_silicon) - Community results repository
- [aidatatools/ollama-benchmark](https://github.com/aidatatools/ollama-benchmark) - Auto-detecting Ollama benchmark
- [cloudmercato/ollama-benchmark](https://github.com/cloudmercato/ollama-benchmark) - Full-featured Ollama benchmark with LLM-as-Judge
- [CordatusAI/ollama-benchmark](https://github.com/CordatusAI/ollama-benchmark) - Streamlit GUI benchmark
- [kaushall13/local-llm-benchmark](https://github.com/kaushall13/local-llm-benchmark) - Real-time monitoring benchmark
- [BenchLlama](https://github.com/srikanth235/benchllama) - Code generation benchmark
- [yzyydev/local-llm-benchmark](https://github.com/yzyydev/local-llm-benchmark) - Multi-provider benchmark

### Evaluation Frameworks
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - Standard academic benchmark suite
- [lm-eval-harness API Guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/API_guide.md) - Local endpoint setup
- [LightEval + LiteLLM](https://huggingface.co/docs/lighteval/use-litellm-as-backend) - HuggingFace evaluation with local endpoints
- [simple-evals](https://github.com/openai/simple-evals) - OpenAI's lightweight eval library
- [MT-Bench / FastChat](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md) - Multi-turn evaluation
- [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) - Instruction-following evaluation

### LM Studio Automation
- [LM Studio CLI Docs](https://lmstudio.ai/docs/cli) - Full CLI reference
- [LM Studio Python SDK - Model Loading](https://lmstudio.ai/docs/python/manage-models/loading) - Programmatic model management
- [LM Studio REST API](https://lmstudio.ai/docs/developer/rest/endpoints) - Native REST endpoints

### Performance Analysis & Results
- [MLX vs GGUF on Apple Silicon](https://famstack.dev/guides/mlx-vs-gguf-apple-silicon/) - Effective throughput comparison
- [2026 Mac Inference Framework Benchmark](https://macgpu.com/en/blog/2026-mac-inference-framework-vllm-mlx-ollama-llamacpp-benchmark.html) - vllm-mlx vs Ollama vs llama.cpp
- [Production-Grade Local LLM Inference on Apple Silicon](https://arxiv.org/abs/2511.05502) - Academic framework comparison
- [Ollama MLX Backend Announcement](https://ollama.com/blog/mlx) - 93% decode speed improvement
- [lm-eval with local server guide](https://medium.com/@kimdoil1211/evaluating-llm-accuracy-with-lm-evaluation-harness-for-local-server-a-comprehensive-guide-933df1361d1d) - Step-by-step setup

### Monitoring & Diagnostics
- [macmon](https://github.com/vladkens/macmon) - No-sudo Apple Silicon monitor
- [asitop](https://github.com/tlkh/asitop) - Performance monitoring CLI
- [Apple Silicon GPU Memory Limits](https://stencel.io/posts/apple-silicon-limitations-with-usage-on-local-llm%20.html) - Memory architecture constraints
- [GGUF Quantization Quality](https://github.com/ggml-org/llama.cpp/discussions/2094) - Perplexity comparison across quant levels

### Methodology & Guides
- [Benchmarking LLMs with Ollama Bash Script](https://medium.com/@walterdeane/benchmarking-local-llms-with-ollama-and-a-simple-bash-script-8fdb5baf5456) - Practical warmup and CSV logging
- [Digital Spaceport Testing Methodology](https://digitalspaceport.com/about/testing-local-llms/) - 10-question quality evaluation
- [Artificial Analysis Methodology](https://artificialanalysis.ai/methodology/performance-benchmarking) - Professional benchmarking best practices
- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) - HuggingFace evaluation standard
