# Testing Plan — Mac Studio M4 Max 128 GB Fork

> **Fork-specific.** Upstream targets a fanless MacBook Air M4 32 GB. This fork retargets the same harness at a Mac Studio M4 Max with 128 GB unified memory. That changes which models are interesting and removes most of the thermal / memory constraints upstream had to work around.

## Why this plan differs from upstream

| Constraint | Upstream (M4 Air 32 GB) | This fork (Mac Studio M4 Max 128 GB) |
|---|---|---|
| Practical model size ceiling | ~17 GB GGUF | ~80 GB weights (60–70 GB safe with KV) |
| Dense > 24B feasibility | Thermal throttles at 100 °C, fanless | Fine — active cooling, no throttling |
| Need REAP / aggressive prune | Yes, only way to fit Gemma 4 | No — full-precision 27B fits easily |
| Loaded models at once | One | Up to two (JIT swap recommended past that) |
| Best path | GGUF (llama.cpp) primarily | MLX primarily, GGUF for A/B on tool calls |

Upstream's finding *"llama.cpp for short-output tool calls, MLX for long decode"* should be re-verified at this rig's larger model sizes — it isn't obviously transferable.

## LM Studio inventory (verified 2026-05-18)

Source: `lms ls`, `lms ps`, and `GET /v1/models` against `http://192.168.68.124:1234/v1`. Sizes are on-disk weights.

**Daily-driver candidates** (the ones this plan actually benchmarks):

| Model ID (use with API) | Format | Quant | Arch | Size | Vision | Tools | Status |
|---|---|---|---|---|:---:|:---:|---|
| `qwen/qwen3-coder-next` | MLX safetensors | **6-bit** (4-bit variant also on disk) | qwen3_next (80B/3B MoE) | 64.76 GB | — | ✓ | Available |
| `qwen3.6-27b` | MLX safetensors | 6-bit | qwen3_5 (27B dense) | 22.80 GB | ✓ | ✓ | ✓ Loaded (65k ctx) |
| `qwen3.6-35b-a3b@6bit` | MLX safetensors | 6-bit | qwen3_5_moe (35B/3B) | 29.09 GB | ✓ | ✓ | Available |
| `qwen3.6-35b-a3b@8bit` | MLX safetensors | 8-bit | qwen3_5_moe (35B/3B) | 37.75 GB | ✓ | ✓ | Available (new) |
| `gemma-4-26b-a4b-it-mlx@4bit` | MLX safetensors | 4-bit | gemma4 (26B/4B MoE) | 15.64 GB | ✓ | ✓ | Available |
| `gemma-4-26b-a4b-it-mlx@6bit` | MLX safetensors | 6-bit | gemma4 (26B/4B MoE) | 21.81 GB | ✓ | ✓ | Available (new) |
| `gemma-4-31b-it-mlx` | MLX safetensors | 8-bit | gemma4 (31B dense) | 33.80 GB | ✓ | ✓ | Available (new) |
| `gemma-4-e4b-it-mlx` | MLX safetensors | 8-bit | gemma4 (4B) | 8.97 GB | ✓ | ✓ | Available |
| `deepseek-v4-flash-dq` | MLX safetensors | 2-bit DQ | deepseek_v4 | 96.53 GB | — | — | ⚠ Tight fit; needs strict context cap |
| `text-embedding-nomic-embed-text-v1.5` | GGUF | Q4_K_M | nomic-bert | 84 MB | — | — | Embeddings (out of scope) |

**Out-of-scope local variants** (community abliterated / alternate re-quants — listed so they're not mistaken for daily-driver candidates; see "NOT in scope" below):

| Model ID | Provenance | Size |
|---|---|---|
| `qwen3.6-27b-paro` | z-lab | 18.80 GB |
| `qwen3.6-27b-ud-mlx` | unsloth, 4-bit | 26.21 GB |
| `qwen3.6-27b-jang_4m-crack` | dealignai, 4-bit | 17.55 GB |
| `gemma-4-31b-jang_4m-crack` | dealignai, 8-bit | 22.69 GB |

**Notable deltas vs. the lineup in [`../local-llm-reference.md`](../local-llm-reference.md):**
- `qwen/qwen3-coder-next` is **6-bit / 64.76 GB**, not 4-bit / 44.9 GB. Reference doc is stale on this — fix it there too.
- `gemma-4-26b-a4b-it-mlx` is **now on disk in both 4-bit and 6-bit** — was a wishlist item; the 6-bit (21.81 GB) is new since the previous inventory pass and enables a same-model quant A/B.
- `qwen3.6-35b-a3b@8bit` (37.75 GB) is new — gives a 6↔8-bit quant A/B at the same active-param count as coder-next.
- `gemma-4-31b-it-mlx` (dense 31B, 8-bit, 33.80 GB) is new and not in any prior plan revision — fills the dense-Gemma slot above 26B-A4B.
- `gemma-4-e4b` is now **MLX 8-bit / 8.97 GB**, not GGUF Q4_K_M / 6.33 GB as previously documented. Update the reference doc accordingly.
- `deepseek-v4-flash` (4-bit, 151 GB) **has been removed from disk** — the Phase 3 "decision needed" entry is resolved; only the DQ variant remains.
- `nvidia/nemotron-3-nano-omni` (GGUF Q4_K_M, 26 GB) **has been removed from disk** — drop from the plan.
- All MLX models in the lineup advertise `vision: true` and `trainedForToolUse: true` per model metadata — the "vision: to add" line in the reference doc is also stale.

## Goal

For each model worth keeping a slot in rotation, produce:
1. **Knowledge score** across the 5-benchmark suite (MMLU, HumanEval, MATH, DROP, GPQA) at n=100.
2. **Coding score** on LiveCodeBench `release_v6` (contamination-resistant rolling window through ~Apr 2025) at n=50 — added 2026-05-18; supersedes HumanEval as the primary frontier-comparable coding signal.
3. **Tool-calling score** on the jdhodges 40 + Veerman 12 suite.
4. **Speed numbers** at the context length we actually use (65 536 for agentic loops).
5. **Engine A/B** (LM Studio GGUF vs MLX) where both formats are available.

Output: an updated `results/FINAL_100Q_RESULTS.md` and `results/tool_calling_results.md` reflecting this rig, plus an addendum (`results/M4_MAX_128GB_NOTES.md`) covering the larger-model regime.

## Models — prioritized

Order is by daily-driver value: finish a phase before touching the next one.

### Phase 1 — current daily drivers (must-test)

| # | Model ID | Role | Size | Why test |
|---|---|---|---|---|
| 1 | `qwen/qwen3-coder-next` (6-bit) | Agentic coder, default | 64.76 GB | Daily driver. Upstream couldn't run this at all. Need to know: how does it actually score on the same harness? Verify it's worth 65 GB over the smaller MoE alternatives. |
| 2 | `qwen3.6-27b` (6-bit dense) | Reasoning specialist | 22.80 GB | Claimed "Opus 4.5 parity" on SWE-bench Verified — needs internal validation, not marketing. |
| 3 | `qwen3.6-35b-a3b@6bit` | Fast generalist | 29.09 GB | Sits between the other two; need to know if it's actually pulling weight or if the other two cover its slot. |

### Phase 2 — already-on-disk candidates (high-value, free of download cost)

| # | Model ID | Why |
|---|---|---|
| 4 | `gemma-4-26b-a4b-it-mlx@4bit` | Upstream's **knowledge-quality winner** (Gemma 4 26B-A4B, 83.6 % avg across all 5 benches). MLX version is now local — was wishlist when the plan was first written. Cheap to test (15.64 GB) and could displace 35B-A3B in the "fast generalist" slot if it lives up to upstream's numbers. |
| 5 | `gemma-4-26b-a4b-it-mlx@6bit` (variant of #4) | Same weights, heavier quant (21.81 GB). A/B against the 4-bit — directly tests whether 4→6-bit closes the gap to upstream's GGUF Q4_K_M numbers, or if Gemma is quant-robust enough that the extra 6 GB doesn't pay off. |
| 6 | `gemma-4-31b-it-mlx` (8-bit dense) | New on disk: dense Gemma 4 31B at 8-bit (33.80 GB). Fills the dense-model slot above qwen3.6-27b. Worth a knowledge run to see if dense-31B-at-8-bit beats MoE-26B-A4B-at-4-bit on the upstream knowledge benches. |
| 7 | `gemma-4-e4b-it-mlx` (8-bit) | Small Gemma 4 (4B, 8.97 GB MLX) with vision + tools. Test as a candidate for FIM/quick-call slot — does it cover anything 27B/35B-A3B don't, at a fraction of the load time? |
| 8 | `qwen/qwen3-coder-next@4bit` (variant of #1) | Same weights, lighter quant (~44 GB). A/B against the 6-bit to quantify the quality vs. memory trade. |
| 9 | `qwen3.6-35b-a3b@8bit` (variant of #3) | Same weights, heavier quant (37.75 GB). A/B against the 6-bit — quantifies whether the extra 8.7 GB moves knowledge or tool-calling scores enough to justify it over the 6-bit at the same active-param count. |

### Phase 3 — fit / feasibility experiments

| # | Model ID | Plan |
|---|---|---|
| 10 | `deepseek-v4-flash-dq` (2-bit DQ, 96.53 GB) | Load-test first: cap context at 32 768, confirm it loads and stays under 128 GB with margin. If yes, run **tool-calling only** as the cheapest signal. Full knowledge suite only if it actually fits long-context workloads. |

> *Previous Phase 3 entry for `deepseek-v4-flash` (4-bit, 151 GB) is resolved — model has been removed from disk as of 2026-05-18. Only the DQ variant above remains in scope for this phase.*

### Phase 4 — watchlist (not on disk; test on arrival)

From [`../local-llm-reference.md`](../local-llm-reference.md):
- `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` — FIM autocomplete (speed_probe only; full suite not appropriate for FIM workload).
- `mlx-community/Qwen2.5-VL-72B-Instruct-4bit` — only if current MLX models' built-in vision proves insufficient (probe needed first to know).
- `mlx-community/Kimi-K2.6-Thinking-*` distilled — watch for a fitting distillation size.
- `0xSero/Gemma-4-21B-REAP` (GGUF) — upstream's daily-driver winner. Optional reproducibility check; lower priority now that the un-pruned Gemma 4 26B-A4B is local.

## Current status (2026-05-19, Phase 1 complete)

Snapshot of what's been run on this rig so far. **Bold** = run, value is the headline score; `—` = not run. Sources: [`benchmarks/runs/*_summary.json`](benchmarks/runs/), [`results/speed_probe/`](results/speed_probe/), [`../benchmarking/local-llm-bench/results/`](../benchmarking/local-llm-bench/results/). Full write-up: [`results/M4_MAX_128GB_NOTES.md`](results/M4_MAX_128GB_NOTES.md).

### Accuracy + coding + tool-calling

| Phase | Model | HumanEval | LCB v6 | MMLU | MATH | DROP | GPQA (raw) | jdhodges (40) | veerman (12) |
|---|---|---|---|---|---|---|---|---|---|
| 1 #1 | `qwen/qwen3-coder-next` (6-bit) | **89 %** | — | **76 %** | **84 %** | **83 %** | **37 %** | **90 %** (18.6 t/s) | **83.3 %** (35.4 t/s) |
| 1 #2 | `qwen3.6-27b` (6-bit dense) | **93 %** | — | **88 %** | **88 %** | **90 %** | **70 %** ⚠ 15 trunc | **95 %** (14.5 t/s) | **83.3 %** (18.4 t/s) |
| 1 #3 | `qwen3.6-35b-a3b@6bit` | **87 %** | — | **83 %** ⚠ 2 trunc | **89 %** ⚠ 2 trunc | **89 %** | **65 %** ⚠ 23 trunc | **97.5 %** (72.4 t/s) | **75.0 %** (85.8 t/s) |
| 2 #4 | `gemma-4-26b-a4b-it-mlx@4bit` | — | — | — | — | — | — | — | — |
| 2 #5 | `gemma-4-26b-a4b-it-mlx@6bit` | — | — | — | — | — | — | — | — |
| 2 #6 | `gemma-4-31b-it-mlx@8bit` | — | — | — | — | — | — | — | — |
| 2 #7 | `gemma-4-e4b-it-mlx@8bit` | — | — | — | — | — | — | — | — |
| 2 #8 | `qwen/qwen3-coder-next@4bit` | — | — | — | — | — | — | — | — |
| 2 #9 | `qwen3.6-35b-a3b@8bit` | — | — | — | — | — | — | — | — |
| 3 #10 | `deepseek-v4-flash-dq` | — | — | — | — | — | — | — | — |

⚠ = run had truncations (response hit `max_tokens=32768` before emitting the final answer letter; counted as wrong). See [Truncation finding](#truncation-finding-gpqa--thinking-models) below for exact question lists. **Raw GPQA scores under-count true ability**: 27b corrected ceiling ≈78-85 %, 35b-a3b corrected ceiling ≈75-83 %.

### Speed (probe + throughput)

| Phase | Model | Speed probe (3-q) | creative-writing | doc-summary | ops-agent | prefill-test |
|---|---|---|---|---|---|---|
| 1 #1 | `qwen/qwen3-coder-next` (6-bit) | **1.5 s total** (2 macmon samples — noisy) | **67.8 eff t/s** / 70.2 gen | **45.6** / 69.9 | **55.8** / 67.9 | **20.6** / 68.0 |
| 1 #2 | `qwen3.6-27b` (6-bit dense) | **74.4 s total** (70 samples) | **19.9 eff t/s** / 20.7 gen | **11.2** / 20.2 | **16.2** / 20.0 | **4.0** / 20.4 |
| 1 #3 | `qwen3.6-35b-a3b@6bit` | **15.8 s total** (≈90 tok/s sustained) | **86.9 eff t/s** / 91.6 gen | **55.9** / 91.7 | **71.4** / 85.5 | **22.9** / 85.5 |
| 2 #4–9 | — | — | — | — | — | — |
| 3 #10 | `deepseek-v4-flash-dq` | — | — | — | — | — |

> Pre-2026-05 speed-probe entries for `devstral-small-2-24b`, `gemma-4-26b-a4b`, `glm-4.7-flash`, `qwen3-coder-30b-a3b`, `qwen3.5-9b`, `qwen3.5-35b-a3b` (+ huihui/HauhauCS abliterated variants, `nemotron-cascade-2`) exist in `results/speed_probe/` but were collected on the upstream M4 Air 32 GB rig before this fork. They're useful as a lineup baseline ([`results/speed_probe_all_10.md`](results/speed_probe_all_10.md)) but **not directly comparable** to the M4 Max 128 GB numbers above. Throughput results in [`../benchmarking/local-llm-bench/results/`](../benchmarking/local-llm-bench/results/) for `gemma-3-12b`, `glm-4.7-flash`, `llama-3.1-8b`, `mistral-small-3.1-24b`, `qwen2.5-vl-7b`, `qwen3-30b`, `qwen3-30b-a3b-instruct-2507`, `qwen3-vl-8b`, `qwen3.5-35b-a3b` (+ `jang_4k` variant), `smolvlm2-2.2b` are likewise from the upstream rig — keep for reference, don't re-bench unless they re-enter the plan.

### Phase 1 outcomes

- **`qwen3.6-27b` is the quality king.** Leads or ties every bench in the set; knowledge avg 85.8 % beats upstream's quality winner (Gemma 4 26B-A4B at 83.6 %). Cost: dense → ~20 tok/s → 37.6 h full suite, dominated by GPQA (22.2 h alone, 15 truncations).
- **`qwen3-coder-next` is the speed/agentic king.** Knowledge avg 73.8 % is the lowest, but it shipped the full suite in **1.9 h** (19× faster than 27b) and has competitive tool-calling. Daily-driver assignment confirmed: default for OpenCode/Cline/agentic loops.
- **`qwen3.6-35b-a3b` is the MoE-thinking middle.** Best jdhodges (97.5 %), best MATH (89 %), runs ~3× faster than 27b for ~3pp less knowledge. Right slot when you need *thinking-mode reasoning at MoE speed*.
- **Upstream finding *knowledge ≠ tool-calling* confirmed.** 35b-a3b leads jdhodges, trails Veerman; 27b ties both but is slow.

### Truncation finding (GPQA + thinking models)

**Discovery during Phase 1:** bench2.py's default `max_tokens=32 768` is **insufficient for GPQA on thinking Qwen3.6 models**. GPQA requires a single-letter final answer *after* the model's full reasoning chain — unlike MATH (where `\boxed{answer}` can be emitted mid-chain) or HumanEval (where the code is the answer). When the model spirals beyond 32k thinking tokens, no letter ever lands and the harness records `FAIL`.

Truncations also appeared in **MATH** (1 on 27b, 2 on 35b-a3b) and **MMLU** (2 on 35b-a3b) but at much lower rates — the chained-reasoning answer format only matters for GPQA.

**Affected question IDs (verified against `benchmarks/runs/gpqa_*_<timestamp>.jsonl`):**

| Model | Trunc count | Question IDs (`bench2.py gpqa --only` syntax) |
|---|---|---|
| `qwen3.6-27b` | 15 / 100 | `2,15,24,28,36,40,43,44,46,51,55,71,80,87,92` |
| `qwen3.6-35b-a3b@6bit` | 23 / 100 | `2,4,12,15,21,24,28,32,43,46,52,53,55,60,69,70,71,80,84,87,89,92,96` |

**11 questions both models truncated on** — the hardest cluster in the suite: `2, 15, 24, 28, 43, 46, 55, 71, 80, 87, 92`. These are likely the test-set's hardest physics/chemistry/biology MCQs and worth tagging for any future GPQA work on thinking models.

**Why it hurts the score:** truncated questions are counted as wrong, so 27b's recorded 70 % is the floor — 11 of the 15 truncations would only need to flip from `null` to the correct letter for 27b to hit 81 % (still below the upper bound of "all 15 right" = 85 %). 35b-a3b similar: raw 65 %, ceiling 88 % if all 23 truncations were correct (unlikely; realistic ≈75-83 %).

## Next steps — prioritized

Concrete actions, ranked by **(value × confidence) / cost**:

### Step A — rerun truncated GPQA questions with `--max-tokens 65536` (high value, well-defined, deferred from Phase 1)

```bash
# 27b — 15 questions
python3 scripts/bench2.py gpqa --examples 100 --model qwen3.6-27b \
  --only 2,15,24,28,36,40,43,44,46,51,55,71,80,87,92 --max-tokens 65536

# 35b-a3b@6bit — 23 questions
python3 scripts/bench2.py gpqa --examples 100 --model "qwen3.6-35b-a3b@6bit" \
  --only 2,4,12,15,21,24,28,32,43,46,52,53,55,60,69,70,71,80,84,87,89,92,96 --max-tokens 65536
```

- **Cost:** ~7–10 h on 27b dense (each question may still spiral up to 65k tokens at 20 tok/s ≈ 54 min worst-case), ~3–5 h on 35b-a3b MoE.
- **Expected outcome:** corrected GPQA scores. Either confirms 27b ≈80 % / 35b-a3b ≈78 % (defensible numbers) or reveals that some questions truly *can't* be solved by these models even with infinite tokens.
- **Risk:** the 65 536 cap is also the model's loaded context length. A few questions may still truncate. If that happens, the residual count is the floor of "unanswerable at this scale" and the published numbers should reflect that explicitly.
- **Harness verification needed first:** confirm `bench2.py` actually merges `--only` re-runs into the existing summary (vs. writing a separate run). If it writes a fresh summary, the chart script + `M4_MAX_128GB_NOTES.md` need to point at the merged result, not the original.

### Step B — patch `max_tokens` default for GPQA before Phase 2

For any future thinking-model GPQA run (Phase 2's Gemma 4 candidates and any Phase 4 thinking model), use `--max-tokens 65536` by default. Minimal patch to `scripts/bench2.py`:

```python
# In the gpqa branch of run_bench(), if max_tokens is unset, default to 65536
if args.benchmark == "gpqa" and args.max_tokens == MAX_TOKENS:
    args.max_tokens = 65536
```

Or just always pass `--max-tokens 65536` in the run command. Either way, **don't** let Phase 2 inherit the silent 32k cap.

### Step C — add LiveCodeBench retroactively to the Phase 1 daily drivers

Phase 1 was started before LCB v6 entered the plan (added 2026-05-18). For full Phase 1 closure:

```bash
for m in qwen/qwen3-coder-next qwen3.6-27b qwen3.6-35b-a3b@6bit; do
  python3 scripts/bench2.py livecodebench --examples 50 --lcb-version release_v6 --model "$m"
done
```

- **Cost:** ~30–90 min per model (~45 min for coder-next, ~3 h for 27b, ~1.5 h for 35b-a3b). Total ~5 h.
- **Value:** gives a contamination-resistant coding signal comparable across the rig and frontier providers; HumanEval is largely saturated and no longer first-party-reported.

### Step D — start Phase 2 with Gemma 4 26B-A4B (quality candidate)

Already on disk in both 4-bit (15.64 GB) and 6-bit (21.81 GB). Upstream's knowledge-quality winner.

- **Cost:** ~12–20 h per quant per full suite (it's a thinking model — and the GPQA `max_tokens` rule from Step B applies).
- **Decision point afterward:** does it beat or tie 27b on the knowledge benches at a fraction of the wall-clock? If yes, Phase 2 #5 (the 6-bit) becomes the new candidate for the "knowledge / quality generalist" slot in [`local-llm-reference.md`](../local-llm-reference.md).

### Step E — pre-Phase-2 throughput sweep (decoupled from the knowledge suite)

The "Speed characterization" section already lists 7 model-labels with `bench.py --backend lmstudio` pending. ~15-25 min per model, **~2–3 h total**. Doing this first means the speed table is full before the multi-day knowledge runs start, and the operator/user has the speed answer for "is X usable day-to-day" without waiting for accuracy data.

### Step F — defer until later

- `deepseek-v4-flash-dq` fit-test (Phase 3): only relevant if some Phase 2 model demonstrates a clear quality ceiling and we need to push further. Otherwise it's a 96 GB resident that locks out everything else.
- Engine A/B (LM Studio MLX vs. llama.cpp GGUF): need to pull a GGUF first. Not on the critical path; pair with whichever model has both formats available when the moment is right (likely Gemma 4 26B-A4B).
- Phase 4 watchlist arrivals: nothing to do until something lands.

### Recommended next session

If running interactively: **Step E (throughput sweep, ~2-3 h) → Step A 35b-a3b leg only (~3-5 h)**, in that order. Confirms 35b-a3b's true GPQA before committing to the longer 27b rerun, and fills the speed table for the entire plan in the same day. Save Step A 27b for an unattended overnight block.

If running headless overnight: **Step A both legs back-to-back (~10-15 h)**. Lower complexity, single decision point at end-of-day, leaves the morning clean for Step C (LCB) or starting Phase 2.

## Benchmarks — what to run, in what order

### Per-model order (each model, in this sequence)

1. **Speed probe** (~10 min) — `scripts/speed_probe.py`. 10 quick questions. Confirms the model loads, responds, gives a rough tok/s. If this is broken, stop and fix before committing to a multi-hour run.
2. **Tool calling** (~15–30 min) — `scripts/tool_call_bench.py` on jdhodges + Veerman suites. Fastest signal for "is this usable in OpenCode/Cline/Aider day-to-day?" — and the upstream finding that knowledge ≠ tool-calling rank means this matters independently.
3. **HumanEval** (~30–60 min) — quickest of the knowledge benches; sanity check on code generation before committing to MATH/GPQA. Likely saturated for current models (89–93 % on Phase 1 daily-drivers); keep for backward comparison with upstream's numbers, but use LiveCodeBench as the load-bearing coding metric.
4. **LiveCodeBench** (~30–90 min at n=50) — `scripts/bench2.py livecodebench --examples 50 --lcb-version release_v6`. The primary coding signal going forward: contamination-resistant (LeetCode/AtCoder/Codeforces problems through ~Apr 2025), pass@1 (all hidden tests must pass), reported by every current frontier provider (DeepSeek, Qwen, Claude, GPT-5, Gemini) so cross-rig comparison is real. Use `release_v5` if v6 is suspected to overlap a model's pre-training cutoff.
5. **MMLU** (~30–60 min) — broad knowledge, fast scoring (single letter).
6. **MATH** (~2–8 h) — slow for thinking models; long pole.
7. **DROP** (~1–2 h) — reading comprehension, generous substring scoring.
8. **GPQA** (~2–10 h) — hardest, slowest. Run last. Skip on first pass if a model has clearly already qualified or disqualified itself.

Rationale: each step is a gate. Cheap signal first; expensive only if the cheap signal warrants it.

### Cross-model order

Run benchmarks **model-major, not benchmark-major**: complete the full suite on Qwen3-Coder before starting Qwen3.6-27B. This way an interrupted run leaves at least one model fully characterized, instead of three models partially characterized.

## Engine A/B (LM Studio GGUF vs MLX)

Re-test upstream's "llama.cpp short, MLX long" rule for any model that has both formats available on this rig:

- For each format, run tool-calling (short) and MATH (long) at n=100.
- Compare wall-clock and quality.
- Record in `results/engine_comparison.md` as a separate section ("M4 Max 128 GB findings").

Likely candidates on the current inventory:
- `qwen/qwen3-coder-next` **@4bit vs @6bit** — same engine, different quant (covered in Phase 2 #8).
- `gemma-4-26b-a4b-it-mlx` **@4bit vs @6bit** — same engine, different quant (covered in Phase 2 #5). A separate head-to-head vs. a GGUF Q4_K_M Gemma 4 26B-A4B would still be valuable if downloaded, since it tests engine *and* quant family in one shot.
- `qwen3.6-35b-a3b` **@6bit vs @8bit** — same engine, different quant (covered in Phase 2 #9).
- All other current models are MLX-only on disk; cross-engine A/B would require pulling a GGUF first.

## Speed characterization (effective tok/s, `local-llm-bench`)

Separate measurement track from the per-model speed_probe (which only confirms "loads and responds"). The scenario-based harness at [`../benchmarking/local-llm-bench`](../benchmarking/local-llm-bench/) measures **effective throughput** — output tokens divided by total wall-clock including prefill — across four realistic workloads (ops-agent, doc-summary, prefill-test, creative-writing). This is what we actually wait for in agentic loops, and prefill dominates as context grows.

Run for every model in the plan that doesn't already have an M4 Max result in [`../benchmarking/local-llm-bench/results/`](../benchmarking/local-llm-bench/results/). Where it overlaps with the per-model order's step 1 (speed_probe), prefer this bench — if it completes the 4 scenarios cleanly, the speed_probe step is satisfied by definition.

### Already characterized on this rig (M4 Max 128 GB, LM Studio MLX)

| Plan ref | Model | Result folder | Backend label |
|---|---|---|---|
| Phase 1 #1 | `qwen/qwen3-coder-next` (6-bit) | [`results/qwen3-coder-next-6bit/`](../benchmarking/local-llm-bench/results/qwen3-coder-next-6bit/) | `lmstudio` |
| Phase 1 #2 | `qwen3.6-27b` (6-bit dense) | [`results/qwen3.6-27b-dense-mlx-6bit/`](../benchmarking/local-llm-bench/results/qwen3.6-27b-dense-mlx-6bit/) | `lmstudio` |
| Phase 1 #3 | `qwen3.6-35b-a3b@6bit` | [`results/qwen3.6-35b-a3b/`](../benchmarking/local-llm-bench/results/qwen3.6-35b-a3b/) | `lmstudio-mlx` |

### Still to run

| Plan ref | Model ID (API) | Suggested `--model-label` | Notes |
|---|---|---|---|
| Phase 2 #4 | `gemma-4-26b-a4b-it-mlx@4bit` | `gemma-4-26b-a4b-mlx-4bit` | First Gemma 4 speed numbers on this rig. |
| Phase 2 #5 | `gemma-4-26b-a4b-it-mlx@6bit` | `gemma-4-26b-a4b-mlx-6bit` | Pair with #4 for a same-model quant-vs-throughput plot. |
| Phase 2 #6 | `gemma-4-31b-it-mlx` | `gemma-4-31b-dense-mlx-8bit` | Dense 31B at 8-bit — speed cost vs. 26B-A4B is the question. |
| Phase 2 #7 | `gemma-4-e4b-it-mlx` | `gemma-4-e4b-mlx-8bit` | Small model; should top the throughput table. |
| Phase 2 #8 | `qwen/qwen3-coder-next@4bit` | `qwen3-coder-next-4bit` | Pair with the already-done 6-bit run to quantify quant→throughput. |
| Phase 2 #9 | `qwen3.6-35b-a3b@8bit` | `qwen3.6-35b-a3b-8bit` | Pair with the already-done 6-bit run (same caveat). |
| Phase 3 #10 | `deepseek-v4-flash-dq` | `deepseek-v4-flash-dq-2bit` | **Only if it loads cleanly under the 32 768 context cap** — skip the 8K prefill-test turn if it OOMs; record what works. |

### Per-run command template

```bash
# From the local-llm-bench directory, against the LM Studio host
python3 bench.py \
  --backend lmstudio \
  --base-url http://192.168.68.124:1234/v1 \
  --model <model-id-from-table> \
  --model-label <label-from-table>
```

Add `--no-think` only for thinking-mode models (Qwen3.5-family per upstream README); the Qwen3.6 / Gemma 4 / DeepSeek-V4 lineups in this plan do not need it by default — verify per model before assuming. Results auto-save to `results/<model-label>/<scenario>/m4-max-128gb-40gpu_lmstudio.json`.

### Estimated wall-clock

~15–25 min per model for the four default scenarios (longer for thinking models). All 7 to-do entries ≈ **2–3 hours** of mostly-unattended runs — much cheaper than the knowledge suite, so worth front-loading: do all of these before starting Phase 2 knowledge benches, so the speed table is complete by the time the quality numbers come in.

## Hardware/operational rules (delta from upstream METHODOLOGY.md)

Most of upstream's `results/METHODOLOGY.md` still applies. Differences for this rig:

- **Thermal abort threshold**: upstream aborts at GPU > 95 °C. On the Mac Studio that almost never happens; treat sustained > 85 °C as the watch line and > 95 °C as abort.
- **Memory**: keep `weights + KV < 80 GB`. With JIT in LM Studio, only one of {coder-next, 27B, 35B-A3B, gemma-26B-A4B} should be resident at benchmark time. Two simultaneous large MLX models has caused queue stalls before — see [local-llm-reference.md](../local-llm-reference.md) troubleshooting.
- **DeepSeek-V4-Flash-DQ exception**: at 96.53 GB weights, it must be the **only** loaded model. Cap context at 32 768 and watch wired memory with `sudo memory_pressure`.
- **Context length on load**: set explicitly. 65 536 for tool-calling and knowledge benches (matches daily-driver agentic-loop setting). Don't max to 256k+ — KV cache reservation was the root cause of past stalls.
- **Pre-flight**: run `python3 scripts/lms.py check` before each run, as upstream requires.
- **Verify model IDs**: use the exact strings from `GET /v1/models`, not the `mlx-community/...` paths in older configs. The harness sends `model: <id>` to LM Studio; mismatched IDs return 404.

## Deliverables

After phase 1:

1. `results/runs/` — per-question JSONL for each (model × benchmark) pair (harness auto-writes).
2. `results/extracted/master_summary.json` — regenerated via `scripts/update_master_data.py`.
3. `results/tool_calling_results.md` — append a "M4 Max 128 GB, Phase 1" section with the 3 daily-driver models alongside upstream's table.
4. `results/FINAL_100Q_RESULTS.md` — same.
5. `results/M4_MAX_128GB_NOTES.md` (new) — observations specific to the larger-model regime: which models actually justify their memory, how much the engine A/B finding generalizes, where KV cache became the bottleneck.
6. `results/charts/` — regenerated via the analysis notebook.
7. `../benchmarking/local-llm-bench/results/<model-label>/` — effective-tok/s scenario JSONs for every newly-benched model (see "Speed characterization" above).

After phase 2/3: update the same files; no new structure.

## What's explicitly NOT in scope

- Long-context needle-in-haystack — upstream has `scripts/long_context_test.py`; defer until we have a workload that demands it.
- TurboQuant KV-cache quantization — only matters when memory-constrained. We aren't, except possibly for DeepSeek-V4-Flash-DQ.
- Cross-machine comparisons against the M4 Air or external GPUs — not the question being asked.
- Building new benchmarks. Reuse upstream harness as-is; if a bug is found, fix it and credit in `AUDIT_REPORT.md`.
- Re-running the embeddings model (`nomic-embed-text-v1.5`) — different workload class, not what this harness measures.
- Abliterated / uncensored community re-quants on disk (`qwen3.6-27b-paro`, `qwen3.6-27b-ud-mlx`, `qwen3.6-27b-jang_4m-crack`, `gemma-4-31b-jang_4m-crack`) — different workload class (alignment-stripped); not what this harness measures, and they'd pollute the "daily-driver value" ranking.

## Estimated total wall-clock

Per model, full suite (tool-calling + 5 knowledge benches at n=100 + LiveCodeBench at n=50): roughly **9–32 h** depending on whether the model "thinks" (LCB adds ~30–90 min).

**Phase 1 actuals (2026-05-17 → 2026-05-19):**
- `qwen/qwen3-coder-next` — 1.9 h (no thinking, no truncations)
- `qwen3.6-27b` — 37.6 h (dense thinking model; 22.2 h was GPQA alone, of which 7.2 h was the 15 truncated questions consuming the full 32k cap)
- `qwen3.6-35b-a3b@6bit` — 14.7 h (MoE thinking; 6.5 h GPQA, ~4.4 h was the 23 truncated questions)
- **Total Phase 1: ~54 h across 3 models.** Outside the plan's original 8-30 h/model estimate on the high end, driven by GPQA truncation behavior.

**Phase 2 forward estimate** — three full-suite candidates (Gemma 4 26B-A4B @4bit, Gemma 4 31B dense, Gemma 4 E4B) ≈ **2–4 days**, plus three quant-A/B variants (Gemma 4 26B-A4B @6bit, coder-next @4bit, Qwen3.6-35B-A3B @8bit) that only need tool-calling + 1–2 knowledge benches to detect quant drift ≈ another **1–2 days**. **Apply the `max_tokens=65 536` rule for GPQA on all thinking models going forward** (Step B in next-steps) — should shave hours per run by avoiding the spiral-to-32k-then-fail loop.

Plan around overnight blocks; `scripts/run_tool_call_overnight.sh` already exists as a template.
