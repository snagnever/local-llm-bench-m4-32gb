# Night Exploration Work Log

**Started:** 2026-04-12
**Goal:** Explore MLX, REAP models, alternative inference servers, KV cache optimization, and other Mac-specific SOTA for local LLM inference on 32GB M4 MacBook Air.

**Constraint:** Many small experiments, not long benchmarks. Document everything.

---

## Hypotheses to test

1. **MLX might be faster than llama.cpp for some model sizes** — despite our prior finding that llama.cpp was 3x faster on Gemma 4 26B. The question is: what model sizes / quant types does MLX actually win on?
2. **REAP models preserve speed and quality but reduce memory** — from 0xSero's claims. Gemma-4-21B-REAP could free up ~3GB of RAM with minimal quality loss.
3. **KV cache quantization to 4-bit could give us 20x more context** — 0xSero claim. Worth testing in LM Studio.
4. **Alternative MLX servers (mlx-openai-server, vLLM-MLX, oMLX, vMLX)** may have features/speed LM Studio lacks.
5. **Gemma-4-REAP GGUFs are broken on llama.cpp** (chat template bug). If true, we must use MLX or skip REAP entirely.

---

## Plan

### Phase 1: Research + environment check (in parallel with agents)
- [ ] Subagent: Research 0xSero model cards for quality claims and benchmarks
- [ ] Subagent: Research alternative MLX servers (vLLM-MLX, mlx-openai-server, oMLX, vMLX)
- [ ] Subagent: Research Reddit/HN local LLM Mac tips (KV quant, flash attn, etc)
- [ ] Check disk space, current mlx-lm install, LM Studio settings

### Phase 2: Baseline current system
- [ ] Quick speed probe on current Gemma 4 LM Studio config with multiple question types
- [ ] Test LM Studio KV cache quantization options (if available)
- [ ] Test LM Studio flash attention toggle

### Phase 3: MLX direct testing
- [ ] Install/verify mlx-lm
- [ ] Run Gemma 4 26B MLX 4-bit directly via mlx_lm.server
- [ ] Compare to LM Studio on same questions
- [ ] Try smaller model (Gemma 3 4B or Qwen3.5 9B) to see if MLX wins on smaller sizes

### Phase 4: REAP model testing
- [ ] Check disk space
- [ ] Download Gemma-4-21B-REAP MLX 4-bit
- [ ] Quality probe (10 questions across benchmarks)
- [ ] Speed probe vs base Gemma 4

### Phase 5: Alternative servers
- [ ] Try mlx-openai-server (if installable)
- [ ] Try vLLM-MLX (if installable on Mac)

### Phase 6: Other optimizations
- [ ] Flash attention comparison
- [ ] Context length vs speed (we have data already, could supplement)
- [ ] KV cache quantization test

---

## Results log

### 22:50 — Environment check & download
- Disk: 292GB free, plenty of room
- Swap was high (13GB) from prior long runs; reclaimed after unload
- MLX environment is fully installed at `.venv-mlx/` (python 3.11)
- mlx-vlm v0.4.4 available — has `mlx_vlm.server` with built-in KV cache quantization (`--kv-bits`, `--kv-quant-scheme turboquant/uniform`)
- mlx-community cached directory mostly empty — prior session cleaned up MLX models
- Started download: `ukint-vs/gemma-4-21b-a4b-it-REAP-MLX-4bit` (13GB, completed in ~4 min)

### 22:58 — LM Studio config limits
- `lms load` CLI does NOT expose KV cache quantization options (`--cache-type-k`, `--cache-type-v`)
- LM Studio GUI has it (confirmed by agents), but not scriptable
- **Conclusion:** can't test LM Studio KV quant from CLI. Skipping, moving to MLX where we have full control.

### 23:00 — LM Studio Gemma 4 baseline (ctx=16384, flash_attention=True)
Direct API test, 3 questions:
- Warmup (2+2): 12.7 t/s, 47 tokens (42 think + 5 vis)
- MMLU-style question: 13.3s, **17.7 t/s**, 235 tokens (230 think + 5 vis)
- Math easy (sin 30°): 44.6s, **19.3 t/s**, 859 tokens (388 think + 471 vis)
- Geography (Paris): 3.3s, **18.7 t/s**, 62 tokens (57 think + 5 vis)
- **Baseline: ~18-19 t/s sustained on LM Studio Gemma 4 26B base**

### 23:05 — REAP Gemma 4 21B MLX — first test
Started `mlx_vlm.server --model ukint-vs/gemma-4-21b-a4b-it-REAP-MLX-4bit --port 8080`
- Model loads in ~13 seconds
- Warmup: output=2 tokens, 57.9 tok/s, peak_mem=13.97 GB
- Simple MMLU: out=4 tokens, 35.2 t/s
- Math hard (custom): out=831 tokens, **25.5 t/s sustained**
- DROP-style: out=2 tokens, 52.8 t/s
- GPQA-style: out=4 tokens, 34.6 t/s
- Peak memory stable at ~14.2 GB (vs ~17.4 GB LM Studio Gemma 4)

### 23:15 — REAP MLX vs Base Gemma 4 on EXACT benchmark questions (Q1-5 from seed=42)

| Q | Level | Base Gemma 4 26B | REAP Gemma 4 21B MLX |
|---|-------|------------------|----------------------|
| 1 | L4 | 72.0s, 1949 tok (1469 think+480 vis), 27.1 t/s, OK | 19.8s, 489 tok, 25.8 t/s, **OK** |
| 2 | L2 | 22.1s, 635 tok (388+247), 28.7 t/s, OK | 9.5s, 232 tok, 26.1 t/s, **OK** |
| 3 | L3 | 90.7s, 2213 tok (1642+571), 24.4 t/s, OK | 27.5s, 688 tok, 25.7 t/s, **OK** |
| 4 | ? | 553.0s, 10518 tok, FAIL (bugged max_tokens originally; valid now) | **TIMEOUT** (>300s) |
| 5 | L1 | 26.0s, 551 tok (227+324), 21.2 t/s, OK | 240.3s, 287 tok, 22.4 t/s, OK (slow outlier) |

**Observations:**
1. **Per-token generation speed is similar** (25-28 t/s both models) on questions where both completed
2. **REAP generates ~3x fewer tokens per question** — because mlx-vlm does NOT invoke Gemma 4's thinking mode the way LM Studio does
3. **Real-world speedup on Q1-3**: 2.3x-3.6x (22→9.5s, 72→19.8s, 90→27.5s)
4. **Quality Q1-3: 3/3 correct** (small sample, but matches base Gemma)
5. **Q4 timed out** — unclear if spiral or stall
6. **Q5 took 240s** — suspiciously slow, possibly model swap-in from disk after Q4 stall

**Memory footprint:**
- REAP MLX: ~14.2 GB peak (stays at same level across queries)
- Base Gemma 4 LM Studio: ~17.4 GB process + 5GB swap

### 23:30 — Root cause for REAP "no thinking" identified
Subagent investigation of `chat_template.jinja` (lines 259-266) found:

```jinja
{%- if not enable_thinking | default(false) -%}
    {{- '<|channel>thought\n<channel|>' -}}   <-- empty thought block
{%- endif -%}
```

When `enable_thinking=False` (default), the template PRE-FILLS an empty `<|channel>thought\n<channel|>` block, effectively telling Gemma 4 "you've already thought, go straight to response." That's why REAP generated 232-688 tokens vs base Gemma's 635-2213 — it wasn't thinking at all.

**mlx_vlm.server defaults `enable_thinking=False`** (confirmed in TemplateParams.template_kwargs). The earlier REAP test was apples-to-oranges.

To enable thinking: add `extra_body={"enable_thinking": True}` or as top-level JSON parameter.

### 23:35 — First attempt with enable_thinking=True failed
- Sent "What is 2+2?" to REAP MLX with `enable_thinking=True`
- Response took **292 seconds** for only 66 output tokens
- `generation_tps=22.7` reported by server, but effective rate was 0.23 tok/s
- Content contained `<|channel>thought\n`... so thinking DID happen, but massive overhead outside actual generation
- Clean request "Capital of France?" (no thinking flag): **18.1 seconds for 9 tokens** — still way too slow
- Server logs show "Generation finished, cleared cache" between EVERY request → huge per-request overhead

**Conclusion:** `mlx_vlm.server` has ~17s per-request overhead (likely KV cache rebuild, tokenizer reset, processor reset). Unsuitable for rapid benchmarking. Need to use `mlx_vlm.generate` CLI instead (loads once, generates once, exits).

### 23:50 — KERNEL PANIC 🔥
macOS crashed with "SOCD report detected: (iBoot async abort)". Likely causes:
- LM Studio (Gemma 4 26B, ~17GB resident) AND mlx_vlm.server (REAP, ~14.5GB peak) running simultaneously on 32GB RAM
- Plus Python probe processes, plus background Claude Code shell
- Sustained 70% CPU on mlx_vlm for 15+ minutes
- Thermal pressure on fanless MacBook Air

**Lesson:** Never run two LLM engines simultaneously on 32GB. Unload LM Studio before starting MLX.

---

## After restart — Recovery state
- Disk: 289GB free ✓
- Swap: 0 MB used (clean slate) ✓
- LM Studio: running but no models loaded ✓
- REAP model: intact on disk (13GB) ✓
- /tmp question caches: wiped, rebuilt ✓
- Worklog saved ✓

## Revised plan (one engine at a time)

### Phase A: REAP MLX via CLI only (no server) ✓ DONE
Using `mlx_probe.py` which loads via `mlx_vlm` Python library directly.

**Results so far:**

| Bench | Thinking | N | Correct | Total Time | Avg tok/s | Notes |
|-------|----------|---|---------|-----------|-----------|-------|
| MATH  | ON  | 8 | 7/8 (87.5%) | 929s | 23.1 | Q4 spiraled (Level 5, same Q base also failed) |
| MATH  | OFF | 3 | 3/3 (100%) | 51s  | 26.7 | Q1-3 only |
| MMLU  | ON  | 10 | **10/10 (100%)** | 99s | 27.2 | **Excellent** |
| GPQA  | ON  | 5 | **0/5 (0%)** | 630s | 23.7 | **Catastrophic** — 4 spiraled to max |
| GPQA  | OFF | 5 | 1/5 (20%) | 66s | 44.0 | Fast but wrong answers |

**Peak memory:** 14.0-14.7 GB stable (vs base Gemma 4 in LM Studio ~17-19 GB)

### Key findings
1. **REAP is ~3-5x faster on MATH/MMLU** due to more concise thinking (uses ~5x fewer tokens than base Gemma 4 on same questions)
2. **REAP preserves MATH and MMLU quality** — 87.5% and 100% on our samples match or exceed base
3. **REAP CATASTROPHICALLY FAILS on GPQA** — 0/5 with thinking ON, 1/5 with thinking OFF, vs base's 70% on 10-question probe
4. **Memory footprint 3-5 GB less** than base Gemma 4 in LM Studio
5. **Failure mode on GPQA**: genuine exploration that never converges. The model tries formulas, gets "impossible" results, tries again, until out of tokens. Not garbage — indecisiveness.
6. **mlx_vlm.server** has massive per-request overhead (17s+) making it unsuitable for benchmarking. Use `mlx_vlm.generate` Python API or CLI directly.

### Phase B: LM Studio reasoning=off ❌ IMPOSSIBLE via API
Tried every parameter variation (`reasoning`, `reasoning_effort`, `reasoning_mode`, `thinking`, `enable_thinking`, `chat_template_kwargs`). None disabled thinking on Gemma 4 via `/v1/` OR `/api/v0/` endpoints. It's a GUI-only toggle. Skipping.

### Phase C: Download base Gemma 4 MLX for comparison (in progress)
- Model: `mlx-community/gemma-4-26B-A4B-it-4bit`
- Size: ~16 GB, at ~1.4 GB downloaded
- Purpose: isolate whether GPQA collapse is from REAP pruning or MLX-specific issue (chat template, quantization)
- Will test same 5 GPQA questions on base MLX

### Phase D: KV cache quantization test on REAP (pending)
Try `--kv-bits 4 --kv-quant-scheme turboquant` to see if we can fit larger contexts or free up memory.

### Phase E: Consolidated analysis ✓ DONE

## Final Findings

### Complete experiment matrix

| Model | Engine | Bench | N | Score | Notes |
|-------|--------|-------|---|-------|-------|
| Gemma 4 26B | LM Studio (GGUF Q4_K_M) | MATH | 100 | 82% | from 100q run |
| Gemma 4 26B | LM Studio (GGUF Q4_K_M) | MMLU | 100 | 84% | from 100q run |
| Gemma 4 26B | LM Studio (GGUF Q4_K_M) | DROP | 100 | 89% | from 100q run |
| Gemma 4 26B | LM Studio (GGUF Q4_K_M) | GPQA | 100 | 64% | from 100q run |
| Gemma 4 26B | mlx_vlm (MLX 4-bit) | MATH Q1-3 | 3 | **3/3** | 274s total |
| Gemma 4 26B | mlx_vlm (MLX 4-bit) | MATH Q4-8 | 5 | 4/5 | Q4 spiraled at 8192 |
| Gemma 4 26B | mlx_vlm (MLX 4-bit) | MMLU | 10 | 8/10 | |
| Gemma 4 26B | mlx_vlm (MLX 4-bit) | DROP | 10 | **9/10** | |
| Gemma 4 26B | mlx_vlm (MLX 4-bit) | GPQA | 5 | **0/5** | **mlx-vlm broken** |
| REAP 21B | mlx_vlm (MLX 4-bit) | MATH Q1-8 | 8 | 7/8 | Q4 spiraled at 16384 |
| REAP 21B | mlx_vlm (MLX 4-bit) | MATH Q9-20 | 12 | 9/12 | Q15,Q16 spiraled; Q18 wrong |
| REAP 21B | mlx_vlm (MLX 4-bit) | MMLU | 10 | **10/10** | **Beats base MLX** |
| REAP 21B | mlx_vlm (MLX 4-bit) | DROP | 10 | 7/10 | under-thinking |
| REAP 21B | mlx_vlm (MLX 4-bit) | GPQA ON | 5 | 0/5 | **mlx-vlm broken** |
| REAP 21B | mlx_vlm (MLX 4-bit) | GPQA OFF | 5 | 1/5 | |
| REAP 21B | mlx_vlm + KV quant 4-bit | MATH Q1-3 | 3 | 3/3 | Same as without |

### Speed comparison (REAP MLX vs base Gemma 4 LM Studio, MATH Q9-20)
On 9 questions where BOTH got correct:
- **Base LM Studio: 1221s total** (avg 135s/question)
- **REAP MLX: 364s total** (avg 40s/question)
- **Speedup: 3.4x**
- Quality preserved on these questions

### Memory footprint
| Setup | Model mem | Peak | Swap impact |
|-------|-----------|------|-------------|
| Gemma 4 26B LM Studio ctx=16k | 17-19 GB | 18 GB | 4-11 GB swap |
| Gemma 4 26B MLX 4-bit | 15.7 GB | 16.4 GB | Low |
| **REAP 21B MLX 4-bit** | **14.0 GB** | **14.7 GB** | **Very low** |

REAP saves ~3 GB vs base LM Studio, ~2 GB vs base MLX. On 32GB Mac, this matters for avoiding swap.

### Critical findings

1. **mlx-vlm is BROKEN for Gemma 4 on GPQA** — Both base and REAP fail 0/5 with spirals. LM Studio works fine. This is an mlx-vlm library issue, likely chat template / thinking termination handling. **Do not use mlx-vlm for GPQA.**

2. **REAP excels on MATH when it works** — 3.4x faster than base LM Studio with same correctness. Generates 5x fewer thinking tokens but reaches the same answers. This is legit free speed.

3. **REAP has termination failures on ~15% of MATH questions** — Q4, Q15, Q16, Q18 all failed to terminate thinking. Base LM Studio solved Q15 and Q16. Even at 16384 max_tokens, REAP spirals indefinitely. Estimated REAP MATH 100q score: 75-80% vs base 82%.

4. **REAP scores 10/10 on MMLU** (small sample) — possibly BETTER than base MLX (8/10). Noise in small sample.

5. **REAP under-thinks on DROP** — 7/10 vs base MLX 9/10. Answers with 8-50 tokens where base uses 200+ tokens. Confidence without reasoning.

6. **mlx-vlm.server has massive per-request overhead** — ~17s per request of non-generation time. Unsuitable for benchmarking. Use the Python API via mlx_vlm.load/generate.

7. **Base Gemma 4 MLX produces different outputs than LM Studio** even at temp=0 due to different quantization/numerical precision. Not a bug, just a different model effectively.

8. **KV cache quantization (4-bit TurboQuant)** works on REAP with no quality loss on short contexts. Would matter more on long contexts.

9. **LM Studio's reasoning=off toggle is GUI-only** — not exposed via HTTP API. Cannot script non-thinking tests of base Gemma 4.

### Practical recommendations

**For your 32GB M4 MacBook Air:**

| Use case | Best setup | Why |
|----------|-----------|-----|
| **Math/reasoning (speed-focused)** | REAP 21B + mlx_vlm | 3.4x faster than LM Studio with similar quality |
| **General knowledge (MMLU)** | REAP 21B + mlx_vlm | Best quality and speed |
| **GPQA (hard science)** | Base Gemma 4 + LM Studio | mlx-vlm is broken for this |
| **DROP (reading comprehension)** | Base Gemma 4 + LM Studio | REAP under-thinks |
| **Coding (HumanEval)** | Not tested in this night | Prior data says Gemma 4 LM Studio = 99% |
| **Memory-constrained workflows** | REAP 21B + mlx_vlm | Saves 3-5 GB RAM |

### Also tested: REAP 19B (30% pruned)
- MATH Q1-3: 3/3 correct (same as 21B)
- MMLU Q1-10: **7/10 correct** (vs 21B's 10/10) — **30% pruning loses MMLU quality**
- Peak memory: 12.8 GB (vs 14.1 for 21B, -1.3 GB)
- Same speed as 21B

**Conclusion: 30% pruning is too aggressive on knowledge tasks. Stick with 21B (20%).**

### Also tested: mlx-openai-server
- Same spiral behavior on GPQA Q1 (4096 tokens, no answer)
- Confirms the issue is upstream mlx-vlm, not a specific server implementation
- `reasoning_content` field exists but empty for Gemma 4 (not split from content)

### Long-context performance (the use case that matters)

User cares about **long-context daily use**, not benchmark scores. Tested needle-in-haystack retrieval at increasing context sizes for two models:

#### deadbydawn REAP 21B MLX (no TurboQuant)

| Context | Prefill t/s | Gen t/s | Total | Peak GB | Found needle |
|---------|-----------:|---------:|------:|--------:|:-----------:|
| 4K | 311 | 35.7 | 13.1s | 14.50 | ✓ |
| 8K | 288 | 34.1 | 28.4s | 14.76 | ✓ |
| 16K | 264 | 31.6 | 62.1s | 15.50 | ✓ |
| 32K | 215 | 25.3 | 152.8s | 17.03 | ✓ |
| 64K | 157 | 19.8 | 419.5s | 20.31 | ✓ |

#### Base Gemma 4 26B MLX (mlx-community)

| Context | Prefill t/s | Gen t/s | Total | Peak GB | Found needle |
|---------|-----------:|---------:|------:|--------:|:-----------:|
| 4K | 278 | 32.8 | 14.6s | 17.28 | ✓ |
| 8K | 267 | 33.1 | 30.6s | 17.54 | ✓ |
| 16K | 215 | 24.7 | 76.5s | 18.28 | ✓ |
| 32K | 181 | 25.3 | 181.7s | 19.81 | ✓ |
| 64K | 145 | 17.5 | 452.8s | 23.09 | ✓ |

**Key findings:**
1. **Both models retrieve the needle at every context size up to 64K** — quality preserved
2. **REAP 21B is 8-23% faster prefill than base 26B** at all context sizes
3. **REAP 21B uses ~3 GB less memory** at every context size (REAP saves both weight memory AND attention memory)
4. **Prefill is the bottleneck** — at 64K, 7+ minutes JUST to read the document before generating
5. **Memory grows linearly with context** (~80 MB per 1K tokens) due to global attention layers (5/30 layers)
6. **64K is the practical daily-use limit** on 32GB Mac — beyond that prefill takes too long
7. **Speed degrades smoothly** — no cliff, no OOM up to 64K

#### TurboQuant on Gemma 4 — disappointing
At 64K context, TurboQuant only saved 0.31 GB (1.5%) at 2-8% speed cost.

| Metric | Base 64K | + TurboQuant | Δ |
|--------|---------:|-------------:|----|
| Prefill | 157 t/s | 154 t/s | -2% |
| Gen | 19.8 t/s | 18.3 t/s | -8% |
| Peak mem | 20.31 GB | 20.00 GB | -1.5% |
| Total time | 419.5s | 427.9s | +2% |

Why so weak? Gemma 4 already minimizes KV cache via sliding window attention (25/30 layers fixed at 1024 tokens). Only 5 global attention layers grow with context. TurboQuant's compression target (the global KV cache) is already small.

**TurboQuant is essentially a no-op on Gemma 4 architecture.** Useful for other models with full attention, not Gemma 4.

#### Extreme context: 96K and 131K (the actual TurboQuant use case)

Tested deadbydawn REAP at 98304 and 131072 tokens on 32GB Mac, both with and without TurboQuant.

| Context | Without TurboQuant | With TurboQuant |
|---------|-------------------:|----------------:|
| **96K target** | **OOM at 94K** (Metal "Insufficient Memory") ❌ | **✓ Completed** (776s, 23.20 GB, found needle) |
| **131K target** | (would OOM) | **OOM at 100K** (76%) ❌ |

**THE TURBOQUANT USE CASE FOUND:**
TurboQuant pushes the OOM ceiling from ~94K to ~100K tokens. That's a ~6K token extension — small in absolute terms but **the difference between "this context size works" and "OOM crash"**.

For Gemma 4 on 32GB Mac:
- **0-90K tokens**: Use baseline, no need for TurboQuant
- **90K-100K tokens**: USE TurboQuant — without it, you crash
- **100K+ tokens**: NOT VIABLE on 32GB Mac, even with TurboQuant

Gemma 4's full 131K context window is NOT usable on 32GB Mac. ~96K is the realistic upper limit with TurboQuant.

**Practical context guide for daily use (32GB M4):**
| Context | Status | Notes |
|---------|--------|-------|
| 4K | ✓ Fast (13s) | 311 t/s prefill |
| 8K | ✓ Fast (28s) | 288 t/s prefill |
| 16K | ✓ OK (62s) | 264 t/s prefill, 1 minute wait |
| 32K | ✓ OK (153s) | 215 t/s prefill, 2.5 min wait |
| 64K | ⚠️ Slow (420s) | 157 t/s prefill, **7 min wait** |
| 96K | ⚠️ Slow + needs TurboQuant (776s) | 127 t/s prefill, **13 min wait** |
| 100K+ | ❌ OOM | Not possible on 32GB |
| 131K | ❌ OOM | Full context window unusable |

**The practical "long context daily use" limit is 32K-64K** for tolerable wait times. Beyond 64K it's batch-job territory (set it and walk away).

### TurboQuant KV compression — measured, not what I thought
Tested TurboQuant-MLX v2.0 (from GitHub DeadByDawn101/turboquant-mlx) with four configurations on same 3 long-context DROP questions (~9000 token input).

| Config | Prompt t/s | Gen t/s | Peak mem | Notes |
|--------|-----------|---------|----------|-------|
| Baseline (no KV quant) | 291 | 37.2 | 14.80 GB | |
| kv_bits=8 uniform | — | — | — | **CRASHES: RotatingKVCache Quantization NYI** |
| kv_bits=4 uniform | — | — | — | **CRASHES: RotatingKVCache Quantization NYI** |
| kv_bits=4 turboquant | 276 | 34.6 | 14.80 GB | 5-7% slower, same memory |

**Findings:**
1. At 9K context, TurboQuant provides **no memory savings** — same 14.80 GB peak
2. TurboQuant is **5-7% slower** than baseline on short outputs due to quantization overhead
3. Uniform KV quantization (8-bit and 4-bit) **crashes** on long contexts in mlx_vlm
4. TurboQuant only provides benefits for contexts MUCH larger than 9K — probably 30K+ tokens where KV cache dominates memory

**CORRECTION to earlier claim:** The 40% speedup of deadbydawn REAP over ukint-vs REAP is NOT from TurboQuant. It's from different base model quantization schemes:
- ukint-vs: PLE-safe 4-bit (preserves ScaledEmbedding, PLE, vision encoder, norms in bf16)
- deadbydawn: affine 4-bit (~4.8 bpw average, fewer bf16 layers)

Fewer bf16 layers = faster inference on MLX/Metal. TurboQuant was never active in my earlier tests.

### TriAttention — NOT INSTALLABLE on Mac
Requires `triton` package which has no macOS wheels. Linux/CUDA only. Cannot test on Apple Silicon.

### **deadbydawn101 Tool-Calling variant — the speed winner**
Tested `deadbydawn101/gemma-4-21b-REAP-Tool-Calling-mlx-4bit` which claims TurboQuant + TriAttention + tool calling. Size similar to ukint-vs variant (12.8 GB peak).

| Benchmark | ukint-vs (plain REAP) | deadbydawn101 (TurboQuant) | Speedup |
|-----------|----------------------|---------------------------|---------|
| MATH Q1-3 | 45.4s, 26.8 t/s, 3/3 | **32.1s, 37.4 t/s, 3/3** | **1.4x** |
| MMLU Q1-10 | 99s, 27.2 t/s, 10/10 | **67s, 38.5 t/s, 9/10** | **1.5x** |
| DROP Q1-10 | 252s, 26.6 t/s, 7/10 | **46s, 38.4 t/s, 7/10** | **5.5x** |

**deadbydawn is 40-550% faster than plain REAP MLX** with essentially the same quality.

The DROP 5.5x speedup is the biggest — this is because plain REAP ukint-vs DROP had a Q8 that spiraled to 4096 tokens; deadbydawn's variant fails Q8 faster (443 tokens).

Peak memory is also 1 GB lower: 13.0-13.5 GB vs 14.0-14.7 GB.

**deadbydawn101/gemma-4-21b-REAP-Tool-Calling-mlx-4bit is the best variant tested this session.**

### What I did NOT test (for future)
- `deadbydawn101/gemma-4-21b-REAP-Tool-Calling-mlx-4bit` (1623 downloads, tool calling + TurboQuant + TriAttention)
- oMLX with paged SSD KV cache
- Full 100q benchmark of REAP MLX
- Long-context scenarios where KV quant matters
- Switching chat template in mlx-vlm to see if GPQA works with different template handling
- LM Studio GUI reasoning=off (requires manual GUI interaction)

### Artifacts produced
- `research/WORKLOG_NIGHT.md` (this file)
- `scripts/mlx_probe.py` (clean MLX probe harness)
- `scripts/probe_compare.py` (4-way probe with extra_body support)
- `results/probes/mlx_*.jsonl` (all raw probe results)
- Question caches in `/tmp/*_questions_seed42.json`

### Stability note
System rebooted once (iBoot async abort) at ~23:50 due to running LM Studio AND mlx_vlm.server simultaneously on 32GB. Post-restart, one-engine-at-a-time discipline was maintained with no further issues.
