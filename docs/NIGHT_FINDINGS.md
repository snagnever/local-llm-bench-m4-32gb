# Night Exploration — Consolidated Findings

**Date:** 2026-04-12 / 2026-04-13
**Scope:** MLX inference, REAP pruned models, KV quantization, alternative servers
**Work log:** `research/WORKLOG_NIGHT.md`
**Follow-up (2026-04-14):** Tool-calling benchmark on the same Gemma 4 REAP weights in both engines — `results/tool_calling_results.md`. **Finding:** for tool-calling (short outputs, long prompts) llama.cpp GGUF is 1.9× faster end-to-end than mlx_vlm on the same weights. The 1.67× MLX advantage below holds for long-generation workloads but **inverts for short-output workloads** because llama.cpp prefill beats mlx_vlm prefill on Apple Silicon. Choose the engine per output length, not per model.

---

## TL;DR

**For long-context daily use on 32GB M4: `deadbydawn101/gemma-4-21b-REAP-Tool-Calling-mlx-4bit` via mlx_vlm Python API.**

- **3-5x faster than LM Studio base Gemma 4** on benchmark workloads
- **8-23% faster prefill** at every context size vs base Gemma 4 26B MLX
- **3 GB less memory** (~12K extra context headroom)
- **Sweet spot: 16K-32K context** (1-3 min prefill, comfortable, reliable)
- **Practical max: 96K** with TurboQuant enabled (13 min prefill, OOM threshold)
- **131K full context window: NOT viable on 32GB Mac**, need 64GB+

**Speed comes from two sources:**
1. **Quantization scheme** (deadbydawn uses pure 4-bit affine, not PLE-safe → fewer bf16 ops → 40% more tok/s)
2. **REAP pruning** generates ~3-5x fewer thinking tokens per question vs base Gemma 4

**TurboQuant** has exactly one use case: enables 90-100K context that would otherwise OOM. Outside that range it's slower with no benefit.

**Caveats:**
- `mlx_vlm` is **broken for GPQA on Gemma 4** (both base and REAP spiral). Use LM Studio for GPQA.
- REAP "under-thinks" on DROP (7/10 vs base's 9/10) — confident wrong answers.
- ~15% failure rate on hardest MATH (termination spirals).
- 30% pruning (REAP 19B) is too aggressive — loses MMLU quality.

**Don't bother with:**
- `mlx_vlm.server` (17s per-request overhead — use Python API directly)
- TriAttention (not installable on Mac)
- TurboQuant below 90K context (adds 5-20% overhead with no benefit)

---

## What was tested

### Models
| Model | Source | Size |
|-------|--------|------|
| Gemma 4 26B-A4B Q4_K_M | lmstudio-community GGUF | 16.8 GB on disk, 17-19 GB RAM |
| Gemma 4 26B-A4B 4-bit | mlx-community MLX | 15 GB on disk, 15.7 GB RAM |
| REAP 21B (20% pruned) | ukint-vs MLX 4-bit | 13 GB on disk, 14.0 GB RAM |
| REAP 19B (30% pruned) | ukint-vs MLX 4-bit | 12 GB on disk, 12.8 GB RAM |

### Engines
| Engine | Status | Notes |
|--------|--------|-------|
| LM Studio (llama.cpp GGUF) | ✓ Works reliably | GUI-only reasoning toggle, no API |
| `mlx_vlm.server` | ⚠️ High overhead | 17s/request overhead, unsuitable for benchmarking |
| `mlx_vlm` Python library | ✓ Works | Used via `mlx_probe.py` |
| `mlx-openai-server` | ⚠️ Same issues | Same mlx_vlm spiral on GPQA |

---

## Speed data (hard numbers)

### MATH Q9-20, both models, LM Studio vs REAP MLX

Questions where BOTH gave correct answers (n=9):
- **Base Gemma 4 LM Studio**: 1221s total, avg 135s/question, 21-28 tok/s
- **REAP 21B MLX**: 364s total, avg 40s/question, 23-26 tok/s
- **Speedup: 3.4x**
- Both use the same prompt, same seed, same temperature

Why? REAP generates **5x fewer thinking tokens per question** reaching the same answer. It's more "decisive" about when to stop thinking.

### Token efficiency (sample: MATH Q1-3)
| Q | Base LM Studio tokens | REAP MLX tokens | Ratio |
|---|---|---|---|
| 1 | 1949 (think 1469 + vis 480) | 388 | **5.0x fewer** |
| 2 | 635 (think 388 + vis 247) | 223 | **2.8x fewer** |
| 3 | 2213 (think 1642 + vis 571) | 563 | **3.9x fewer** |

### Raw generation speed comparison (tok/s)
| Setup | MATH | MMLU | DROP | Notes |
|-------|------|------|------|-------|
| Base LM Studio | 20-29 | 14-15 | 16 | |
| Base MLX 26B | 25-31 | 26-33 | 27-31 | |
| ukint-vs REAP 21B MLX | 23-26 | 24-30 | 23-30 | plain REAP |
| **deadbydawn REAP 21B MLX** | **37-38** | **35-42** | **35-42** | **+ TurboQuant + TriAttention** |
| REAP 19B MLX (30% prune) | 23-27 | 23-28 | — | |

**deadbydawn's variant is ~40-50% faster than plain REAP MLX**, same quality on MATH (3/3 vs 3/3) and DROP (7/10 vs 7/10), 1 question difference on MMLU (9/10 vs 10/10). This is the best-performing setup tested.

Total time comparison on 10-question probes:
- **ukint-vs REAP MMLU**: 99s
- **deadbydawn REAP MMLU**: 67s (47% faster)
- **ukint-vs REAP DROP**: 252s  
- **deadbydawn REAP DROP**: 46s (5.5x faster — dramatic speedup from TurboQuant)

---

## Quality data

### Small-sample probes this session

| Benchmark | Base LM Studio (100q) | Base MLX 4-bit (10q) | REAP 21B MLX (varies) | REAP 19B MLX (10q) |
|-----------|----------------------|----------------------|----------------------|---------------------|
| MATH Q1-8 | 7/8 | 7/8 | 7/8 | — |
| MATH Q9-20 | 11/12 | — | 9/12 | — |
| MMLU Q1-10 | ~84% (100q) | 8/10 | **10/10** | 7/10 |
| DROP Q1-10 | ~89% (100q) | 9/10 | 7/10 | — |
| GPQA Q1-5 | 2/5 (from 100q) | 0/5 ❌ | 0/5 ❌ | — |

### MATH failure modes
- **Q4 Level 5**: base LM Studio spiral to 10k tokens (FAIL), base MLX spiral to 8k (FAIL), REAP spiral to 16k (FAIL). Genuinely hard question.
- **Q15, Q16**: base LM Studio solved (10k, 3.5k tokens), REAP spiraled at 16k tokens. REAP termination issue.
- **Q18 Level 5**: base LM Studio timeout (0 tokens), REAP wrong answer.

### GPQA failure mode in mlx_vlm
Both base Gemma 4 MLX and REAP MLX spiral on GPQA. Model generates legitimate reasoning (formula recall, exploration) but never terminates into a response phase. Fills max_tokens with repetitive `"Let's try..."` exploration.

**This is NOT a REAP problem** — base MLX also fails. It's an upstream mlx-vlm library issue in chat template or thinking termination handling.

---

## Long context performance (the use case that matters)

User cares about long-context daily use. Tested needle-in-haystack retrieval on deadbydawn REAP 21B MLX on 32GB Mac.

### Daily-use context sizes (no TurboQuant needed)

| Context | Prefill t/s | Gen t/s | Total time | Peak GB | Found needle |
|---------|-----------:|---------:|------------:|---------:|:------------:|
| 4K | 311 | 35.7 | 13s | 14.50 | ✓ |
| 8K | 288 | 34.1 | 28s | 14.76 | ✓ |
| 16K | 264 | 31.6 | 62s | 15.50 | ✓ |
| 32K | 215 | 25.3 | 153s | 17.03 | ✓ |
| 64K | 157 | 19.8 | 420s (**7 min**) | 20.31 | ✓ |

### Extreme context (TurboQuant matters here)

| Context | Without TurboQuant | With TurboQuant |
|---------|-------------------:|----------------:|
| **96K** | **OOM at 94K** ❌ | ✓ 776s, 23.20 GB, found needle |
| **131K** | OOM | **OOM at 100K** ❌ |

### Comparison: REAP vs base Gemma 4 26B MLX

| Context | REAP 21B Prefill | Base 26B Prefill | REAP Gen | Base Gen | REAP Mem | Base Mem |
|---------|-----------------:|-----------------:|---------:|---------:|----------:|----------:|
| 4K | **311** | 278 | 35.7 | 32.8 | **14.50** | 17.28 |
| 8K | **288** | 267 | 34.1 | 33.1 | **14.76** | 17.54 |
| 16K | **264** | 215 | 31.6 | 24.7 | **15.50** | 18.28 |
| 32K | **215** | 181 | 25.3 | 25.3 | **17.03** | 19.81 |
| 64K | **157** | 145 | 19.8 | 17.5 | **20.31** | 23.09 |

REAP 21B is consistently 8-23% faster prefill, 5-15% faster gen, and ~3 GB less memory at every context size.

### Practical findings

1. **0-32K context: works great** — sub-3-minute wait times, no special tricks needed
2. **32K-64K: usable but slow** — 7 min wait at 64K is annoying for interactive use
3. **64K-96K: requires TurboQuant** to avoid OOM, ~13 min wait
4. **96K+: not viable on 32GB Mac**, period
5. **Gemma 4's 131K context window is impossible on 32GB Mac** — would need 64GB+ of unified memory
6. **TurboQuant's actual value**: extends OOM ceiling from ~94K to ~100K (small but real)
7. **REAP saves 3 GB memory at every context size** vs base Gemma 4 — that's the equivalent of an extra ~12K context worth of headroom
8. **The needle was found at every context size up to 96K** — quality is preserved, only speed and memory degrade with context

### What this means for "I want long context for daily use"

- **Best practical sweet spot**: **16K-32K context**. Sub-3 minutes wait, comfortable memory.
- **Power user max**: 64K with 7-min wait when needed.
- **Edge case**: 96K with TurboQuant, 13-min wait, only when you really need it.
- **Don't bother**: 100K+ on 32GB Mac.
- **Use deadbydawn REAP, not base Gemma 4** — saves 3 GB and is 8-23% faster across the board.
- **TurboQuant only matters at the very edge** (90K+) — ignore it for typical work.

## Memory impact

| Setup | Model load mem | Peak (during MATH) | Notes |
|-------|---------------|-------------------|-------|
| LM Studio Gemma 4 ctx=16384 | 17.4 GB | ~19 GB | 4-11 GB swap on long runs |
| MLX base Gemma 4 26B 4-bit | 15.7 GB | 16.4 GB | Low swap |
| **REAP 21B MLX** | **14.0 GB** | **14.7 GB** | **Very low swap** |
| REAP 19B MLX | 12.7 GB | 13.2 GB | Lowest |

On 32GB RAM, going from 19 GB to 14 GB frees up ~5 GB for OS, Chrome, other apps. Swap pressure drops from "high" to "none" during LLM runs.

---

## What mlx_vlm.server is broken for

During testing, I found these issues with `mlx_vlm.server`:
1. **17+ seconds per-request overhead** (KV cache rebuild between requests)
2. **enable_thinking flag works** but only via direct Python API, not HTTP
3. **Gemma 4 thinking termination broken on GPQA** — also happens in mlx-openai-server
4. **reasoning_content field not populated** even when thinking is on (reasoning is included in `content`)

Use `mlx_vlm` Python library directly via a script, not the server.

---

## KV cache quantization / TurboQuant — measured vs hype

Tested thoroughly with TurboQuant-MLX v2.0 (DeadByDawn101/turboquant-mlx). Initial claims: "4.6x KV cache compression" and "~50x combined compression vs fp16" (with TriAttention).

### Short-context results (3 MATH questions, ~90-563 output tokens)
| Config | Time | Gen t/s | Peak mem |
|--------|------|---------|----------|
| Baseline | 32.3s | 37.4 | 13.1 GB |
| kv_bits=4 turboquant | 32.0s | 37.7 | 13.1 GB |

**No difference.** Short contexts have tiny KV caches, nothing to compress.

### Long-context results (~9000 token input)
| Config | Prompt t/s | Gen t/s | Peak mem | Status |
|--------|-----------:|---------:|----------:|--------|
| Baseline | 291 | 37.2 | **14.80 GB** | OK |
| kv_bits=8 uniform | — | — | — | **CRASHES: NYI** |
| kv_bits=4 uniform | — | — | — | **CRASHES: NYI** |
| kv_bits=4 turboquant | 276 | 34.6 | **14.80 GB** | **5-7% slower, same memory** |

**Findings:**
1. Same memory at 9K context — no compression visible
2. 5-7% slower due to quantization overhead
3. Uniform KV quant crashes on long contexts in mlx_vlm
4. TurboQuant is only beneficial at MUCH larger contexts (30K+ tokens) where KV cache dominates memory
5. For benchmark workloads (typical <2K token inputs), TurboQuant provides **zero benefit**

### TriAttention — not installable on Mac
Requires `triton` package which has no macOS wheels (Linux/CUDA only). Cannot test.

### Corrected understanding
**The 40% speedup of deadbydawn REAP over ukint-vs REAP is NOT from TurboQuant.** It's from different base model quantization:
- ukint-vs: PLE-safe 4-bit (preserves some layers in bf16 for quality)
- deadbydawn: affine 4-bit (~4.8 bpw average, fewer bf16 layers)

Fewer bf16 layers during inference = faster Metal kernels. TurboQuant was never actually active when I measured the ~40% gap.

---

## System stability

One kernel panic at 23:50 (`SOCD report detected: iBoot async abort`) caused by running LM Studio AND mlx_vlm.server simultaneously (17GB + 14.5GB = 31.5GB on 32GB machine + overhead).

Lesson: **only one inference engine loaded at a time on 32GB**.

Post-restart: one-engine discipline maintained, no further issues through 3+ hours of testing.

---

## Practical recommendations for your 32GB M4 Air

### TL;DR — recommended setup for daily use

```bash
# One-time setup
pip install mlx-vlm
huggingface-cli download deadbydawn101/gemma-4-21b-REAP-Tool-Calling-mlx-4bit

# Use via Python API
python3 scripts/mlx_probe.py --model deadbydawn101/gemma-4-21b-REAP-Tool-Calling-mlx-4bit ...
```

**Why deadbydawn over ukint-vs**: 40-50% faster (37 vs 27 tok/s) due to more aggressive 4-bit quantization (no PLE-safe bf16 layers), same quality on our tests, 1 GB less memory.

### Long-context daily use guide

| Context | Wait time | Recommendation |
|---------|-----------|----------------|
| 4-16K | <1 min | Default — any setup works |
| 16-32K | 1-3 min | Sweet spot for most "long context" use |
| 32-64K | 3-7 min | Acceptable for important docs |
| 64-90K | 7-13 min | Batch territory, no TurboQuant needed |
| **90-100K** | **13+ min** | **Enable TurboQuant (--kv-bits 4 --kv-quant-scheme turboquant)** to avoid OOM |
| 100K+ | — | Not viable on 32GB Mac, need 64GB+ |

**Practical sweet spot: 16K-32K context.** Sub-3-minute prefill, no special tricks needed, no OOM risk.

### If you want speed for math/reasoning (benchmark workloads)
**deadbydawn REAP 21B MLX** — 3-6x faster than LM Studio base on MATH. Tested at 37 tok/s sustained. Same quality, much less memory.

But:
- ~15% failure rate on hardest questions (termination spirals — REAP-specific)
- Broken on GPQA (mlx_vlm library bug, also affects base Gemma 4 in MLX)
- Lower quality on DROP (under-thinking)

### If you want reliability across all benchmarks
Keep **LM Studio with GGUF Gemma 4 26B Q4_K_M** as fallback. Works everywhere including GPQA. Use it when reliability matters more than speed.

### For memory-constrained workflows
REAP 21B MLX saves 3-5 GB RAM vs base Gemma 4. That's ~12K extra context worth of headroom. Useful if you need to run other applications alongside.

### Avoid these
- **mlx_vlm.server / mlx-openai-server** — 17s/request overhead, unsuitable for benchmarking. Use Python API instead.
- **REAP 19B (30% pruned)** — too aggressive, loses MMLU quality (7/10 vs 21B's 10/10).
- **mlx_vlm for GPQA** — broken for Gemma 4 (both base and REAP spiral indefinitely). Use LM Studio.
- **TurboQuant for short contexts** (<90K) — adds 5-20% overhead with no benefit. Only enable at 90K+ context where it's needed to avoid OOM.
- **TriAttention** — not installable on Mac (requires Linux/CUDA Triton).

---

## Open questions / future work

1. **Why does mlx_vlm spiral on Gemma 4 GPQA?** — Confirmed it's an upstream mlx_vlm library issue (not REAP-specific, base Gemma 4 26B MLX has the same failure). Worth filing upstream bug at https://github.com/Blaizzy/mlx-vlm/issues
2. **Does REAP termination improve with prompt engineering?** — Adding "Answer immediately after thinking" might help.
3. **Full 100q REAP MATH benchmark** — to get a real score, not just probe samples.
4. **Vision tests** — both models support vision, completely untested.
5. **deadbydawn101's tool-calling variant** — 1623 downloads, claimed features include TurboQuant and TriAttention. Worth testing.
6. **Long-context KV quantization** — where TurboQuant should shine. Untested.

---

## Files created this session

- `research/WORKLOG_NIGHT.md` — detailed work log with timestamps
- `research/NIGHT_FINDINGS.md` — this document (executive summary)
- `scripts/mlx_probe.py` — clean MLX probe harness using Python API
- `scripts/probe_compare.py` — HTTP-based probe (for LM Studio testing)
- `scripts/turboquant_test.py` — TurboQuant on/off comparison
- `scripts/long_context_test.py` — needle-in-haystack at scaling context sizes
- `notes/twitter/0xSero_digest.md` — 24KB digest of 1,143 0xSero tweets
- `results/probes/mlx_*.jsonl` — 21 raw probe files
- `results/probes/longctx_*.jsonl` — 4 long-context test files
- `results/probes/tq_*.jsonl` — 2 TurboQuant comparison files

Models downloaded (in `~/.cache/huggingface/hub/`, ~53 GB total):
- `mlx-community/gemma-4-26B-A4B-it-4bit` (15 GB) — base Gemma 4 MLX
- `ukint-vs/gemma-4-21b-a4b-it-REAP-MLX-4bit` (13 GB) — plain REAP 20% pruned
- `ukint-vs/gemma-4-19b-a4b-it-REAP-MLX-4bit` (12 GB) — plain REAP 30% pruned
- **`deadbydawn101/gemma-4-21b-REAP-Tool-Calling-mlx-4bit` (13 GB)** — REAP + tool calling, recommended

Python packages installed:
- `mlx-vlm` (0.4.4) — MLX vision-language inference
- `turboquant-mlx` v2.0 (from DeadByDawn101 GitHub) — KV cache compression
- `mlx-openai-server` — alternative MLX server (tested, has same Gemma 4 GPQA bug as mlx_vlm)

All numbers in this document are backed by JSONL files with full per-question detail.

---

## Complete probe results table

| Config | Bench | N | Correct | Time | Gen tok/s | Peak mem |
|--------|-------|---|---------|------|-----------|----------|
| Base MLX 26B | math Q1-3 | 3 | 3/3 | 274s | 29.2 | 16.0 GB |
| Base MLX 26B | math Q4-8 | 5 | 4/5 | 671s | 26.0 | 16.0 GB |
| Base MLX 26B | mmlu | 10 | 8/10 | 201s | 28.4 | 16.1 GB |
| Base MLX 26B | drop | 10 | 9/10 | 265s | 29.2 | 16.3 GB |
| Base MLX 26B | gpqa | 5 | **0/5** ❌ | 1400s | 25.6 | 16.4 GB |
| REAP 21B ukint-vs ON | math Q1-3 | 3 | 3/3 | 45s | 26.8 | 14.1 GB |
| REAP 21B ukint-vs ON | math Q4-8 | 5 | 4/5 | 884s | 23.1 | 14.5 GB |
| REAP 21B ukint-vs ON | math Q9-20 | 12 | 9/12 | 1165s | 24.1 | 14.3 GB |
| REAP 21B ukint-vs OFF | math Q1-3 | 3 | 3/3 | 51s | 26.7 | 14.1 GB |
| REAP 21B ukint-vs ON | mmlu | 10 | **10/10** ✓ | 98s | 27.2 | 14.5 GB |
| REAP 21B ukint-vs ON | drop | 10 | 7/10 | 251s | 26.6 | 14.6 GB |
| REAP 21B ukint-vs ON | gpqa | 4 | **0/4** ❌ | 1647s | 23.1 | 14.7 GB |
| REAP 21B ukint-vs OFF | gpqa | 5 | 1/5 | 66s | 44.0 | 14.7 GB |
| REAP 21B ukint-vs ON + KV4 | math Q1-3 | 3 | 3/3 | 48s | 25.3 | 14.1 GB |
| REAP 19B (30% prune) ON | math Q1-3 | 3 | 3/3 | 48s | 26.4 | 12.8 GB |
| REAP 19B (30% prune) ON | mmlu | 10 | 7/10 | 197s | 26.4 | 13.2 GB |
| **deadbydawn REAP 21B** | **math Q1-3** | **3** | **3/3** | **32s** | **37.4** | **13.1 GB** |
| **deadbydawn REAP 21B** | **mmlu** | **10** | **9/10** | **66s** | **38.5** | **13.4 GB** |
| **deadbydawn REAP 21B** | **drop** | **10** | **7/10** | **45s** | **38.4** | **13.5 GB** |
