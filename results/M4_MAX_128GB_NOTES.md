# M4 Max 128 GB — Phase 1 Notes

**Hardware:** Mac Studio M4 Max, 128 GB unified memory, macOS 26.3
**Backend:** LM Studio 0.4.x + MLX (safetensors), context 65 536, temperature 0, seed 42
**Window:** 2026-05-17 → 2026-05-19
**Harness:** patched fork of upstream `scripts/{bench2.py, tool_call_bench.py}` (in-repo path layout instead of upstream's parent-dir layout — see scripts/m4max_charts.py for the comparator that pulls these runs).

This addendum sits alongside upstream's [FINAL_100Q_RESULTS.md](FINAL_100Q_RESULTS.md) and [tool_calling_results.md](tool_calling_results.md). It does **not** replace them — upstream's numbers are M4 Air 32 GB / GGUF Q4_K_M; ours are M4 Max 128 GB / MLX safetensors.

## Phase 1 results — all 3 daily drivers, n=100 each

| Bench | `qwen/qwen3-coder-next` (6-bit, 80B/3B MoE) | `qwen3.6-27b` (6-bit dense) | `qwen3.6-35b-a3b` (6-bit, 35B/3B MoE) |
|---|---|---|---|
| Tool calls jdhodges (40) | 90 % | 95 % | **98 %** |
| Tool calls Veerman (12) | 83 % | 83 % | 75 % |
| **Tool calls combined (52)** | 88.5 % | **92.3 %** | **92.3 %** |
| HumanEval | 89 % | **93 %** | 87 % |
| MMLU | 76 % | **88 %** | 83 % |
| MATH | 84 % | 88 % | **89 %** |
| DROP | 83 % | **90 %** | 89 % |
| GPQA (raw) | 37 % | **70 %** (15 truncated) | 65 % (23 truncated) |
| GPQA truncations | 0 | 15 / 100 (15 %) | 23 / 100 (23 %) |
| **Knowledge avg (5 benches)** | 73.8 % | **85.8 %** | 82.6 % |
| Wall-clock total (Phase 1) | ~1.9 h | ~37.6 h | ~14.7 h |

Charts: [chart_m4max_phase1_scores.png](charts/chart_m4max_phase1_scores.png), [chart_m4max_phase1_throughput.png](charts/chart_m4max_phase1_throughput.png), [chart_m4max_phase1_time.png](charts/chart_m4max_phase1_time.png) — regenerate any time with `python scripts/m4max_charts.py`.

## Key takeaways

### 1. `qwen3.6-27b` dense is the quality king, by a wide margin

- Best on **5 of 6 benches** outright (TC tied, HE, MMLU, DROP, GPQA) and tied with 35b-a3b on tool calling.
- Knowledge avg **85.8 %** — beats upstream's quality winner Gemma 4 26B-A4B (83.6 %) by 2pp on the same suite.
- GPQA **70 %** (raw, with 15 truncated questions counted as wrong) is +6pp over upstream Gemma 4 26B (64 %) and +33pp over coder-next. With the 15 truncations fixed, true score likely 78-85 %.
- **Costs:** ~20 tok/s decode, dense (every parameter active per token). Full suite ran 37.6 h — most of that GPQA (22 h) and HumanEval (3.1 h) where thinking-mode token counts dominate.

### 2. `qwen3-coder-next` is the speed/agentic-tool king, not the quality king

- Knowledge avg **73.8 %** is the lowest of the three — 6-bit 80B/3B MoE doesn't beat smaller dense / smaller MoE on knowledge.
- GPQA **37 %** is the floor: it emits **zero think tokens**, so it can't reason through grad-level science MCQs. The harness logs `think=0` for every coder-next response across every bench.
- However: tool calling **88.5 %** (vs upstream 30B-A3B's 0% bug), MATH **84 %** (best in upstream's set, on par with this rig's 27b), HE **89 %** — and it ran the full suite in **1.9 h** vs 27b's 37.6 h. **19× faster than 27b for ~12pp less knowledge score.**
- Verdict matches the daily-driver assignment in [`../../local-llm-reference.md`](../../local-llm-reference.md): default for OpenCode / Cline / agentic loops; switch to 27b when you need accuracy on a single hard question.

### 3. `qwen3.6-35b-a3b` is the goldilocks middle, but doesn't dominate

- Knowledge avg **82.6 %** is between coder-next (73.8 %) and 27b (85.8 %).
- Best **tool calling jdhodges** in the set (98 %), worst **Veerman** (75 %) — combined 92.3 % ties 27b.
- Best **MATH** in the set (89 %) — within ±1pp of 27b at **3× the speed**.
- Worst **HumanEval** of the three (87 %).
- **Speed:** ~90 tok/s — 4.5× faster than 27b dense, 3× faster than coder-next on knowledge benches.
- **Verdict:** the right choice when you need *thinking-mode reasoning at MoE speed* — e.g. for the "fast generalist" slot. Drop-in replacement for coder-next when you specifically need a thinking model. Probably the right default for chat with reasoning.

### 4. Knowledge ranking does NOT correlate with tool-calling ranking (upstream finding confirmed)

- 27b leads knowledge, ties tool calling.
- 35b-a3b leads jdhodges (98 %), trails on Veerman (75 %) — the multi-turn / ambiguous-intent Veerman cases trip it more than the single-call jdhodges cases.
- coder-next leads no knowledge bench, but is competitive on tool calling (88.5 %) — and is what the OpenCode/Claude-Code-style scaffolds were trained for.

### 5. Engine A/B (LM Studio + MLX vs llama.cpp GGUF)

Phase 1 didn't run a head-to-head — all three models are MLX-only on this rig. Upstream's finding ("llama.cpp for short outputs, MLX for long decode") is not yet re-tested at these sizes. Deferred to Phase 2 (need to pull GGUF Gemma 4 26B-A4B for A/B against the local MLX 4-bit).

## Truncation finding — GPQA on thinking models

**The most surprising operational finding** of Phase 1 was that bench2.py's default `max_tokens=32768` is **insufficient for GPQA on thinking Qwen3.6 models**.

GPQA's answer format (single letter) means the model has to finish its full reasoning chain *before* emitting the final answer. MATH's `\boxed{}` answer can be emitted mid-chain — so MATH only truncated 1-2 questions per run; GPQA truncated 15-23 %.

Both 27b and 35b-a3b truncated **the same questions** — Q2, Q15, Q24, Q28, Q36, Q43, Q46, Q51, Q55, Q71, Q80, Q87, Q92 are common between them. This isn't a per-model defect; it's per-question hardness consuming the thinking budget.

**Recorded effect:** truncated questions are graded `FAIL`, so the published scores undercount true ability:

- 27b GPQA: raw 70 %, **true score (rerun) likely 78-85 %**.
- 35b-a3b GPQA: raw 65 %, **true score likely 75-83 %**.

**Mitigation for Phase 2:**
1. Raise `max_tokens` to 65 536 (= model's context length) for the GPQA bench specifically. Other benches don't need it.
2. The 15-23 truncated questions can be re-run with `bench2.py gpqa --only N1,N2,...` using `--max-tokens 65536`. Cost: ~7-10 h on 27b dense, ~3-4 h on 35b-a3b MoE.
3. This is queued as a follow-up; the headline numbers above already give a defensible floor.

## Operational findings

- **Memory:** at our sizes (15-65 GB), the 128 GB ceiling never became a constraint. Wired memory peaked at 88 GB during coder-next + KV cache; swap stayed at 66-68 MB throughout — no swap pressure.
- **Thermals:** GPU 50-60 °C sustained throughout (vs upstream's 95 °C+ thermal-abort issue on the M4 Air). Active cooling means full runs without throttling.
- **JIT model loading:** `lms load --ctx 65536` reliably takes 8-10 s on a cold swap. With JIT enabled in LM Studio settings, between-model swaps are seamless.
- **Tool-call harness path:** upstream's `tool_call_bench.py` and `tool_call_report.py` use `SCRIPT_DIR.parent.parent / "research" / "benchmarks" / "tool_calling"`, which doesn't match this fork's layout. Patched to `SCRIPT_DIR.parent / "results" / "tool_calling"` — see those scripts.
- **Charts script:** `scripts/m4max_charts.py` is a new fork-only addition; safe to re-run any time. Pulls every `_summary.json` in `benchmarks/runs/`, overlays upstream's published numbers, writes three PNGs into `results/charts/`.

## Per-model raw data

All `_summary.json` and `.jsonl` files are in `benchmarks/runs/`:

| Model | Run prefix |
|---|---|
| `qwen/qwen3-coder-next` | `*_qwen_qwen3-coder-next_20260517_*` |
| `qwen3.6-27b` | `*_qwen3.6-27b_20260517_*` / `_20260518_*` |
| `qwen3.6-35b-a3b@6bit` | `*_qwen3.6-35b-a3b@6bit_20260519_*` |

Tool-calling YAMLs (jdhodges + Veerman) are unchanged from upstream — `results/tool_calling/`.

## Next phases

Plan defined in [TESTING_PLAN.md](../TESTING_PLAN.md):

- **Phase 1 ✅ complete** (this document).
- **Phase 2 pending:** `gemma-4-26b-a4b-it-mlx@4bit` + `@6bit` (4↔6-bit A/B), `gemma-4-31b-it-mlx` (8-bit dense), `gemma-4-e4b-it-mlx` (small/fast), `qwen3-coder-next@4bit` (6↔4-bit quant A/B), `qwen3.6-35b-a3b@8bit` (6↔8-bit quant A/B).
- **Phase 3 pending:** `deepseek-v4-flash-dq` (96.53 GB 2-bit DQ, tool-call only as fit-test).
- **Phase 4 watchlist:** Qwen2.5-Coder-7B FIM, Qwen2.5-VL-72B, Kimi-K2.6-Thinking distill, Gemma 4 21B REAP.
