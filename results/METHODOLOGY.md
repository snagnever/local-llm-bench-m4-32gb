# Benchmark Methodology

**Current harness:** `scripts/bench2.py` (v2, full logging)
**Previous version:** `scripts/bench.py` had bugs — see AUDIT_REPORT.md

## Hardware
- MacBook Air M4, 32GB RAM, 120 GB/s bandwidth, fanless
- Engine: LM Studio (llama.cpp GGUF backend)
- MoE models only (dense 24B+ thermal throttles to 100°C on fanless Air)

## Parameters (set by bench2.py)
- `temperature=0` (deterministic)
- `max_tokens=32768` (large enough for thinking models — Gemma 4 uses up to ~11k tokens/question on hard MATH)
- `context_length=32768+` (set in LM Studio model config, recommend 65536 if you have RAM)
- `random.seed(42)` (same questions for all models, enforced in bench2.py's dataset loaders)
- `timeout=1800` (30 min per question — some thinking models need this on Level 5 MATH)

## Pre-Flight Checklist (BEFORE every benchmark run)

Use `scripts/lms.py check` to automate most of this:

1. **Verify model loaded**: `python3 scripts/lms.py status`
2. **Warm up**: `python3 scripts/lms.py warmup` — sends "What is 2+2?" and verifies response
3. **Verify parameters**: bench2.py prints `max_tokens=X` in its pre-flight output, confirm it matches what you expect (should be 32768 for MATH/GPQA runs)
4. **Temperature check**: `macmon pipe | head -1` — GPU should be < 50°C before starting
5. **Verify dataset loads**: `python3 scripts/bench2.py <benchmark> --examples 100 --only 1` — runs exactly one question to test the full pipeline
6. **Check disk space**: Dataset cache needs ~300MB in `~/.cache/huggingface/` (all 5 datasets together)
7. **READ THE SCRIPT**: Before any long run, read the actual script file to verify parameters. Never trust "it's probably fine."

## During Run

bench2.py automatically handles logging, but you should still:

1. **Monitor output**: Every question prints: question number, elapsed time, prompt_tokens, completion_tokens (think + visible), tok/s, GPU temp, RAM, swap, correct/wrong, running score
2. **Pipe through tee**: `python3 scripts/bench2.py math --examples 100 2>&1 | tee results/runs/gemma_math_console.log`
3. **Watch for errors**: `finish_reason: length` (truncation), timeouts, empty responses, API errors
4. **Watch temperature**: GPU should stay < 85°C. If > 90°C, something is wrong. If > 95°C, abort.
5. **Never run two models simultaneously** — they compete for GPU/memory, both slow down, and logs get tangled
6. **Don't walk away without logging** — the JSONL file is written live, but stdout also captures real-time progress

## Data bench2.py saves per run

Both files are written to `results/runs/`:

### 1. Per-question JSONL (`{benchmark}_{model}_{timestamp}.jsonl`)
One line per question with:
- `run_name`, `benchmark`, `model`, `question_num`, `dataset_idx`, `timestamp`
- Question metadata: `level`, `type`, `subject`, `entry_point`
- API params: `max_tokens_sent`, `temperature`, `context_length`
- Token accounting: `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `visible_tokens`, `total_tokens`, `finish_reason`
- Timing: `elapsed_s`, `tok_s`
- Grading: `correct`, `extracted_answer`, `expected_answer`, `grade_reason`
- System state before/after: `gpu_temp_c`, `cpu_temp_c`, `process_ram_mb`, `swap_used_mb`
- Model config: `model_arch`, `model_quant`, `model_size_gb`, `model_params`
- Full raw response text (for re-grading/audit)

### 2. Summary JSON (`{benchmark}_{model}_{timestamp}_summary.json`)
- Score, correct/total, truncated count, elapsed
- Full model config, hardware info, initial/final system state
- Run parameters (seed, temperature, max_tokens, timeout)
- List of questions run (for targeted reruns)

### Why detail data matters
- Lets you verify scoring is correct after the fact
- Lets you compare specific questions across models
- Lets you debug answer extraction issues
- Lets you verify speed claims
- Without it, you have to trust the script was correct at runtime — and we know from AUDIT_REPORT that the old bench.py wasn't

## Benchmark-Specific Notes

### MMLU
- Dataset: `cais/mmlu`, "all" split (14,042 questions, we sample 100)
- Scoring: Extract letter (A/B/C/D) from response, compare to ground truth
- Prompt: Question + 4 choices + "Answer with just the letter."
- No known issues

### HumanEval
- Dataset: `openai/openai_humaneval` (164 problems, we sample 100)
- Scoring: Execute generated code + test cases, check subprocess exit code
- 10-second execution timeout per problem
- Strips markdown code fences from response
- **Critical**: Must use the REAL dataset — a very early bench.py version had hardcoded fake questions (now fixed in bench2.py)

### MATH
- Dataset: `EleutherAI/hendrycks_math`, all 7 subjects concatenated (5,000 problems, we sample 100)
- Scoring: Extract `\boxed{answer}` from response using brace-depth counting (not regex, which can't handle nested braces)
- Comparison: LaTeX-normalized (`\frac83` == `\frac{8}{3}`, `\left(x\right)` == `(x)`)
- **Critical**: `max_tokens=32768` is required for thinking models. Old bench.py had `max_tokens=4096` hardcoded which truncated 25-32% of Gemma/huihui responses before they reached `\boxed{}`. Fixed in bench2.py.

### DROP
- Dataset: `ucinlp/drop`, validation split (9,535 questions, we sample 100)
- Scoring: Substring match — any expected answer appears in response (case-insensitive)
- Generous scoring — model can output extra text as long as answer is included

### GPQA
- Dataset: `fingertap/GPQA-Diamond` (198 questions, we sample 100)
- Scoring: Extract letter (A/B/C/D) from response
- **Warning**: Heavy thinking models may spiral on some questions (Gemma 4 uses 3,000-16,000+ tokens per question on some). 100q runs can take 10+ hours. Qwen3-Coder handles it fine (no thinking). huihui-ref handles it (moderate thinking).

## After Run

1. **Verify scores against output**: `head -5 results/runs/{run_name}.jsonl` — check a few questions manually
2. **Update the master results**: `python3 scripts/update_master_data.py` — aggregates latest runs into `extracted/master_summary.json`
3. **Regenerate charts**: execute `results/benchmark_analysis.ipynb` to refresh PNGs
4. **Update FINAL_100Q_RESULTS.md** with new scores if significant
5. **Cool-down**: Wait for GPU to drop below 50°C before next run (especially between different models)

## Statistical Notes

- n=20: Useless. CI ±20pp. A "vibe check" at best.
- n=100: Meaningful. CI ±~10pp. Good enough for model comparison.
- n=500+: Precise. CI ±~4pp. Not worth the time for local model testing.
- Always use the same random seed (42) for all models so they get identical questions.

## Cross-Machine Benchmarking

To run bench2.py against LM Studio on a different machine (e.g. Windows PC with RTX 3080 Ti serving over the network):

1. On the remote machine, enable LM Studio "Serve on network" in settings
2. Find the remote IP: `ipconfig` (Windows) or `ifconfig` (Linux/Mac)
3. On your Mac, point bench2.py at the remote:
   ```bash
   LMSTUDIO_URL=http://192.168.1.100:1234/v1 python3 scripts/bench2.py math --examples 100
   ```
   (Note: bench2.py currently hardcodes the URL. See `remote-setup/` for a patched version.)
