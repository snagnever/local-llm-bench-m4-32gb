# M4 Max 128 GB — Phase 1 + Phase 2 Notes

**Hardware:** Mac Studio M4 Max, 128 GB unified memory, macOS 26.3
**Backend:** LM Studio 0.4.x + MLX (safetensors), context 65 536, temperature 0, seed 42
**Phase 1 window:** 2026-05-17 → 2026-05-19 (Qwen daily-driver trio)
**Phase 2 window:** 2026-05-20 → 2026-05-22 (Gemma 4 family — 4 models)
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

## Phase 2 results — Gemma 4 family, n=100 (LCB n=50)

Four models, executed model-major 2026-05-20 → 2026-05-22 per
[`docs/benchmark-plans/2026-05-20-gemma-4-phase-2.md`](../../../docs/benchmark-plans/2026-05-20-gemma-4-phase-2.md).

| Bench | `gemma-4-26b-a4b@4bit` | `gemma-4-26b-a4b@6bit` | `gemma-4-31b` (8-bit dense) | `gemma-4-e4b` (4B, 8-bit) |
|---|---|---|---|---|
| Tool calls jdhodges (40) | **98 %** | **98 %** | **98 %** | 88 % |
| Tool calls Veerman (12) | 83 % | 83 % | 83 % | 67 % |
| **Tool calls combined (52)** | 95.2 % | 95.2 % | 95.2 % | 82.7 % |
| HumanEval | **98 %** | 97 % | 95 % | 91 % |
| LiveCodeBench v6 (n=50) | 66 % (8 LCB TRUNC, post-Step-B) | **80 %** (1 LCB TRUNC, post-Step-B) | 76 % | 68 % |
| MMLU | 78 % | 78 % | 77 % | 65 % |
| MATH | 80 % (1 TRUNC) | 83 % | 79 % | 14 % ⚠ |
| DROP | 79 % | 79 % | **85 %** | 65 % |
| GPQA (raw, `--max-tokens 65536`) | 47 % (1 TRUNC) | 53 % (2 TRUNC) | 48 % | 34 % |
| **Knowledge avg (HE+MMLU+MATH+DROP+GPQA)** | 76.4 % | **78.0 %** | 76.8 % | 53.8 % |
| Effective throughput, ops-agent (eff / gen t/s) | **80.7 / 100.3** | 66.6 / 80.8 | 10.3 / 13.7 | 62.9 / 70.9 |
| Wall-clock total (full suite) | ~3.0 h | ~3.0 h | ~5.0 h | ~2.5 h |

Throughput breakdown across all 4 scenarios:

| Scenario | @4bit eff/gen | @6bit eff/gen | 31B eff/gen | E4B eff/gen |
|---|---|---|---|---|
| creative-writing | 99.9 / 106.0 | 80.5 / 85.3 | 13.6 / 13.9 | 69.5 / 73.7 |
| doc-summary | 55.8 / 107.6 | 46.9 / 85.9 | 8.0 / 14.3 | 46.1 / 75.4 |
| ops-agent | 80.7 / 100.3 | 66.6 / 80.8 | 10.3 / 13.7 | 62.9 / 70.9 |
| prefill-test | 17.9 / 99.2 | 20.5 / 80.9 | 3.3 / 13.6 | 29.1 / 69.3 |

## Phase 2 key takeaways

### 6. `gemma-4-26b-a4b@6bit` is the Gemma flagship

- Wins or ties every Gemma A/B: **LCB v6 80 %** (after Step B reruns, +14 vs @4bit's 66 %), **MATH 83 %**, **GPQA 53 %** (+6 vs @4bit), **HumanEval 97 %**.
- Knowledge avg 78.0 % — top of the Gemma family but still **−7.8pp below `qwen3.6-27b`** (85.8 %).
- Decode throughput 80.8 gen t/s — 4× faster than 27b dense (20.7) for ~8pp less knowledge. Strong "fast generalist" candidate.

### 7. `@4bit` is the speed king of the rig, ties on coding/tool-calling

- **Fastest decode on the rig**: 100.3 ops-agent gen t/s, 106 creative-writing — beats every Phase 1 model.
- Identical tool-calling (98 % / 83 %) and HumanEval (98 %, +1 vs @6bit) to the larger quant.
- The quant cost shows on the **hard** benches: LCB **66 % vs @6bit's 80 %** (Step B reruns: only 1 of 9 truncations recovered at 65 k; 8 still truncate → real model limits), GPQA **47 % vs 53 %**. Closer to a useful "fast coder" slot than a knowledge generalist.

### 8. `gemma-4-31b` dense is a cost-trap on this rig

- Decode **6× slower than @6bit** (13.7 vs 80.8 gen t/s) — full suite took ~5 h.
- Quality is **indistinguishable or worse** vs `@6bit` on every bench except DROP (+6pp). HumanEval −2, LCB −2, MMLU −1, MATH −4, GPQA −5.
- 33.80 GB on disk vs `@6bit`'s 21.81 GB; pays the dense tax for no return.
- **Verdict:** skip in normal rotation. The DROP +6 win is its only headline.

### 9. `gemma-4-e4b` is FIM / quick-call only — not a generalist

- HumanEval 91 %, jdhodges 88 % — useful for fast call-and-format work.
- **MATH collapses to 14 %** — the 4B size simply can't handle hard symbolic math. MMLU 65 % (−13 vs @6bit), Veerman 67 % (−16), GPQA 34 %.
- Throughput is **not** dramatically faster than @4bit MoE (70.9 vs 100.3 ops-agent gen t/s) because MLX optimises MoE A3B routing well — the small-dense model has no inference edge here.
- **Verdict:** matches the original "FIM / quick-call slot" brief; **not a daily-driver fallback**.

### 10. No Gemma beats `qwen3.6-27b` on knowledge

- Best Gemma (`@6bit`) trails 27b by: **MMLU −10pp** (78 vs 88), **MATH −5pp**, **DROP −11pp**, **GPQA −17pp raw**. Even `31B dense`'s DROP +6 over `@6bit` (85) still trails 27b (90).
- 27b retains the "knowledge generalist" slot. Gemma 4 26B-A4B `@6bit` enters the rotation as **"fast generalist with vision"** (Gemmas all advertise vision; 27b does too but is too slow to be the agentic default).

### 11. Gemma 4 truncation profile differs from Qwen 3.6

- **Qwen 3.6 thinking models truncated GPQA at 32 768** (final-letter format + spirals). Phase 1: 15-23 % truncation on Qwen GPQA.
- **Gemma 4 does NOT emit thinking tokens** (`think=0` in every response) — but **truncates LCB on hard problems** because it writes long, exhaustive code (often re-deriving full helpers). Phase 2: 18 % LCB truncation on @4bit, 8 % on @6bit, 0 % on 31B/E4B.
- **Operational rule going forward (updated 2026-05-24 after Step B reruns):** for Gemma 4, raise `--max-tokens` on **LiveCodeBench** to **65 536** by default. But: most 32k truncations are NOT cap-too-tight — they're real model limits. `@4bit` got only +2 pp from the rerun (64 → 66, 1 of 9 recovered); `@6bit` also +2 pp (78 → 80, 1 of 4 recovered). The "ceiling ≈ 86 %" projection in the Phase 2 plan was optimistic — actual recovery is modest.
- Two GPQA questions on `@6bit` truncated even at the 65 536 cap (Q60, Q78) — both `exp=C got=None`. Forms the "unanswerable at this scale" floor for those two MCQs.

## Phase 2 truncations — explicit list (post-Step-B)

| Model | Bench | Truncated Qs | Notes |
|---|---|---|---|
| `gemma-4-26b-a4b@4bit` | LCB v6 | 8, 11, 19, 28, 39, 44, 45, 46 | All 8 at 65 k cap — real model limits; not cap-too-tight. Q2 was recovered (FAIL → OK at 65 k) |
| `gemma-4-26b-a4b@4bit` | MATH | 64 | Not yet rerun |
| `gemma-4-26b-a4b@4bit` | GPQA | 1 | At 65 536 cap — unrecoverable at this scale |
| `gemma-4-26b-a4b@6bit` | LCB v6 | 19 | At 65 k cap — real spiral. Q8/Q15 now answer cleanly under 10 k but wrong; Q28 recovered (FAIL → OK at 65 k) |
| `gemma-4-26b-a4b@6bit` | GPQA | 60, 78 | Both at 65 536 cap — unrecoverable |
| `gemma-4-31b-it-mlx` | — | 0 truncations | |
| `gemma-4-e4b-it-mlx` | — | 0 truncations | |

## Phase 2 wall-clock actuals

- `gemma-4-26b-a4b@4bit` — ~3.0 h (note: interrupted mid-MATH and resumed; nohup-detached driver script used after that)
- `gemma-4-26b-a4b@6bit` — ~3.0 h
- `gemma-4-31b-it-mlx` — ~5.0 h (dense 31B decode tax)
- `gemma-4-e4b-it-mlx` — ~2.5 h
- **Total Phase 2: ~13.5 h** — well under the testing-plan's 2-4 days forward estimate, because Gemma 4 doesn't spiral the way Qwen thinking models did.

## Slot updates from Phase 2

To carry into [`docs/local-llm-reference.md`](../../../docs/local-llm-reference.md):

| Slot | Phase 1 incumbent | Phase 2 verdict |
|---|---|---|
| **Agentic coder / default** | `qwen3-coder-next@6bit` | **Unchanged.** Gemma `@4bit` is fast and codes well (HE 98, jd 98) but knowledge floor (MMLU 78, GPQA 47) is too low for general agentic use. |
| **Knowledge generalist** | `qwen3.6-27b@6bit` | **Unchanged.** No Gemma comes within 7pp on knowledge avg. |
| **Fast generalist (chat + reasoning)** | `qwen3.6-35b-a3b@6bit` | **`gemma-4-26b-a4b@6bit` is now a viable peer** for non-vision work, and the strict winner if vision is needed. Knowledge avg 78 vs 35b-a3b's 82.6, but +3pp on HumanEval and ties on tool-calling. Pick by workload. |
| **FIM / quick-call** | (open) | **`gemma-4-e4b-it-mlx`** fills the slot. Useful for autocomplete and one-shot tool calls only; don't ask it MATH. |
| **Vision tasks** | (open — all MLX models advertise vision) | **`gemma-4-26b-a4b@6bit`** by default; `@4bit` if memory is constrained. (Vision quality not yet benchmarked — workload-specific.) |
| **`gemma-4-31b-it-mlx`** | — | **Demote / skip.** 6× slower than @6bit with no quality return. Worth keeping on disk only as a reproducibility reference. |

## Phase 1 LCB backfill — added 2026-05-24

Plan: [`docs/benchmark-plans/2026-05-22-livecodebench-phase-1.md`](../../../docs/benchmark-plans/2026-05-22-livecodebench-phase-1.md).
Closes the empty LCB column in the master table; no other benches rerun.

| Model | Cap | LCB v6 (n=50) | Truncations | Wall-clock |
|---|---|---|---|---|
| `qwen/qwen3-coder-next` (6-bit) | 32 768 → 65 536 on Q19 rerun | **56 %** (28/50) | **0** (after rerun) | 42 min + 10 min Q19 rerun |
| `qwen3.6-27b` (6-bit dense)     | 65 536 | **62 %** (31/50) | 1 (Q3 atcoder/medium abc365_c) | 16.2 h |
| `qwen3.6-35b-a3b@6bit`          | 65 536 | **54 %** (27/50) | **6** (after Q4 rerun) | 4.6 h + 14 min Q4 rerun |

**Headline:** rank order matches knowledge — `27b > coder-next > 35b-a3b`.
HumanEval saturation suspected and confirmed (HE spread 87–93 collapsed to
8pp on LCB). `27b` is the only Phase 1 model to clear 60 % on LCB v6 and is
now the strongest local coding signal on this rig, displacing the prior
"coder-next is good enough" assumption when correctness > speed.

**Q19 rerun (coder-next, atcoder/hard abc354_d)**: originally truncated at
32 k cap. Reran at 65 k → completed in 36 k tokens (no truncation) but the
answer was still wrong. Q19 is a **genuine model failure**, not a cap
artifact. Net result: coder-next's LCB is now a clean **56 %, 0 truncations,
0 errors** — the most defensible Phase 1 number.

**Q4 rerun (35b-a3b, leetcode/easy 3429)**: originally showed as HTTP 400
right after Q3's spiral — suspected to be a cascade victim (LM Studio still
processing Q3 when Q4 was sent). Reran solo on a clean LM Studio state →
spiraled to 65 k, real truncation. So Q4 is a **real model limit**, not a
backend hiccup. Net result: 35b-a3b's truncation count is now **6** (was
"5 + 1 HTTP 400"); score is unchanged at 27/50.

**Q3 (atcoder/medium abc365_c)** spiraled on every Phase 1 model — `27b` and
`35b-a3b` truncated at 65 k; `coder-next` failed it too (FAIL). It's the
canonical "hard for all thinking budgets" LCB v6 case on this rig and worth
tagging in any future LCB work.

### Wall-clock vs plan

| Plan | Actual | Notes |
|---|---|---|
| 5 h total | **~21 h compute + ~12 h driver idle** | 27b alone consumed 16.2 h of compute |
| 30–60 min coder-next | 42 min | ✅ on plan |
| 1.5–3 h 35b-a3b@6bit | 4.6 h | Over plan, 5 spirals cost the budget |
| 3–6 h 27b | 16.2 h | Far over plan — denser thinking + raised cap meant ~20-min average per question, with 10 questions over 15 min each |

The under-estimate on 27b was the raised-cap × ~20 t/s combination: a
spiral at 65 k tokens takes ~55 min on dense. Future planning for any
≤ 25 t/s thinking model at the raised cap should budget ~20 min/q average
and ~55 min/spiral.

## Bench-harness issues found in Phase 2 + Phase 1 LCB backfill

1. **`tools/local-llm-bench/bench.py` `--base-url`** must **not** include `/v1` — the script appends `/v1/chat/completions` itself. The testing-plan command on line 322 (`--base-url http://192.168.68.124:1234/v1`) doubles the path and silently fails the context-window pre-flight. Use `http://127.0.0.1:1234` (bare host:port).
2. **`scripts/tool_call_bench.py --suite`** only accepts `jdhodges` or `veerman` — there is no `both`. Run twice if you want both suites.
3. **`bench2.py` writes a fresh summary per run** rather than merging when you re-run with `--only`. To produce a single canonical score after a truncation rerun, the per-question JSONLs need to be reconciled manually. After both the Phase 1 LCB backfill and the Phase 2 Step B reruns there are five canonical merged summaries (chart script picks them by alphabetical-last sort — `MERGED` beats any timestamped name):
   - `benchmarks/runs/livecodebench_qwen_qwen3-coder-next_MERGED_summary.json` (original 50-q + Q19 rerun at 65 k)
   - `benchmarks/runs/livecodebench_qwen3.6-35b-a3b@6bit_MERGED_summary.json` (3 batch JSONLs + Q4 rerun)
   - `benchmarks/runs/livecodebench_qwen3.6-27b_MERGED_summary.json` (14 batch JSONLs)
   - `benchmarks/runs/livecodebench_gemma-4-26b-a4b-it-mlx@4bit_MERGED_summary.json` (original 50-q + 9-Q Step B rerun at 65 k)
   - `benchmarks/runs/livecodebench_gemma-4-26b-a4b-it-mlx@6bit_MERGED_summary.json` (original 50-q + 4-Q Step B rerun at 65 k)
4. **`bench2.py` hardcoded 1800 s urlopen timeout was too short for slow thinking models at the raised cap.** When 27b dense exceeded 30 min on a spiral, `urlopen` aborted but LM Studio kept processing — the next request queued behind the still-running inference and also timed out, cascading into back-to-back failures and wedging the run. **Fix landed:** `bench2.py` now reads `BENCH_TIMEOUT` env var (seconds) for the per-request timeout. Set `BENCH_TIMEOUT=3600` when running any ≤ 25 t/s thinking model at `--max-tokens 65536`. Default unchanged (1800 s).
5. **Long-running `bench2.py` invocations launched via the Claude Code `Bash run_in_background` harness silently died around the 2–3 h mark** regardless of the `timeout` flag passed. No traceback, no exit notification — the python process simply vanished and the per-question JSONL stopped growing. Reproduced 3× on different runs. **Workaround:** detached driver pattern. See `.bench-logs/run-27b-lcb-remaining.sh` (`nohup` → PPID=1) — it ran 11 sequential 3-question batches over ~11 h without interruption. Recommended for any future single-leg bench run expected to exceed ~2 h.
6. **The `Config` and per-question `model_arch/model_quant/model_size_gb/model_params` fields reflect whatever model is resident at `bench2.py` startup, not the model the API request hit.** `get_model_config()` is called once at run start; the per-question recording inherits that snapshot even if LM Studio JIT-swaps to the requested `--model` on the first request. Not a correctness issue (the actual model that responded is the one the API was asked for) but it's misleading in the JSONL — the `metric_*` fields in the canonical merged summaries above were verified against the request body, not the stale snapshot.

## Next phases

Plan defined in [TESTING_PLAN.md](../TESTING_PLAN.md):

- **Phase 1 ✅ complete** + **LCB backfill ✅ complete** (added 2026-05-24, this document).
- **Phase 2 ✅ complete** (Gemma 4 family — 4 models — this document).
- **Phase 2 quant-A/B variants pending:** `qwen3-coder-next@4bit` (vs done @6bit), `qwen3.6-35b-a3b@8bit` (vs done @6bit).
- **Phase 3 pending:** `deepseek-v4-flash-dq` (96.53 GB 2-bit DQ, tool-call only as fit-test).
- **Phase 4 watchlist:** Qwen2.5-Coder-7B FIM, Qwen2.5-VL-72B, Kimi-K2.6-Thinking distill, Gemma 4 21B REAP.
- **Follow-up A ✅ — LCB truncation reruns for Gemma `@4bit` and `@6bit`** (Step B from testing-plan, done 2026-05-24). `@4bit`: 1 of 9 recovered (Q2), score 64 → 66, 8 truncations still at 65 k cap. `@6bit`: 1 of 4 recovered (Q28), 2 of 4 now answer cleanly but still wrong (Q8, Q15), 1 still truncates (Q19), score 78 → 80. Wall-clock 116 min + 21 min. Canonical MERGED summaries written. Key finding: most Gemma 4 LCB truncations at 32 k are real model limits, not cap-too-tight artifacts.
- **Follow-up B — Phase 1 LCB cheap-recovery reruns (done 2026-05-24).**
  `coder-next` Q19 at 65 k → completed cleanly but still wrong (real model
  failure, not cap artifact). `35b-a3b@6bit` Q4 at 65 k on clean LM Studio
  state → still truncated (real spiral, not HTTP 400 cascade). Both reruns
  improved data hygiene, neither changed any score. Remaining open question
  is whether a higher cap (≥ 96 k) recovers 35b-a3b's 5 deep-thinking spirals
  (Q3, Q23, Q33, Q39, Q44) — estimated ~1.5–2 h at 96 k, +4–10 pp upside.
  Skipped for now per cost/value trade-off.

## Terminal-Bench 2.0 — Phase A (tie-breaker, 2026-05-24 → 2026-05-26)

Plan: [`docs/benchmark-plans/2026-05-24-terminal-bench-phase-a-plus-b.md`](../../../docs/benchmark-plans/2026-05-24-terminal-bench-phase-a-plus-b.md).
First on-rig measurements of T-Bench 2.0 — the only **agentic shell-loop**
signal on this rig, complements the static benches above.

| # | Model | Score | PASS / FAIL | Errored (timeout) | Wall-clock |
|---|---|---|---|---|---|
| A1 | `qwen/qwen3-coder-next` (6-bit) | **32.6 %** (vendor 36.2) | 29 / 60 | 43 | 16.8 h |
| A2 | `gemma-4-26b-a4b-it-mlx@6bit` | **21.3 %** | 19 / 69 (1 ungraded) | 40 | 14.4 h |

**Operational notes:**

- **`--agent-timeout-multiplier 0.5`** — bounded the 14 outlier tasks
  declaring >60-min agent budgets (1 task at 200 min). Without this cap,
  leg 1's first 3 tasks alone consumed 144 min for zero passes; total
  wall-clock would have been 4–5 days/leg instead of 14–17 h. Scores are
  therefore a **defensible floor**: tasks that genuinely need >15 min agent
  time on this rig are graded FAIL. Plan estimate: full-budget would lift
  scores ≤5 pp.
- **Harness**: Harbor 0.8.0 + terminus-2 agent + LiteLLM → LM Studio
  OpenAI-compat endpoint, container env=docker linux/amd64 (Rosetta).
  Concurrency=1 (single-resident-model rule). Adapter
  `scripts/harbor_to_summary.py` post-processes `<job_dir>/result.json` →
  `benchmarks/runs/tbench_*_summary.json`. Per-leg driver scripts under
  `.bench-logs/run-tbench-*.sh` (pattern derived from
  `run-27b-lcb-remaining.sh`).
- **Result.json schema gotcha**: Harbor 0.8.0 always excludes
  `trial_results` from the on-disk `result.json` (periodic AND final
  writes). Source of truth for score is
  `stats.evals[<key>].reward_stats.reward["1.0"|"0.0"]` — list of trial
  names per reward value. First adapter pass reported 0.0% before this was
  fixed; current adapter uses `reward_stats` directly.
- **Task-order bias**: T-Bench 2.0's first ~5 tasks (make-mips-interpreter,
  circuit-fibsqrt, build-pov-ray, overfull-hbox, video-processing) are all
  in the hardest decile by `timeout_sec`. Both legs spent ~2 h at 0/N PASS
  before scoring anything. Future timeline projections from the first
  10 tasks will under-estimate by ~3× — wait for the inflection at task
  ~15 before extrapolating final score.
- **`AgentTimeoutError` ≠ chain bug**: 43/89 fails on leg 1 and 40/89 on
  leg 2 are graded-as-FAIL timeouts (model spent the full agent budget
  without solving). The non-timeout fails are graded-FAIL with model
  responses returned cleanly — verifier scored 0. Chain is healthy; the
  task set is genuinely hard for ≤30B local models.

**Takeaways:**

1. **Coding-best vs knowledge-best diverge on agentic shell.** Phase 2
   LCB had `gemma-4-26b-a4b@6bit` (80 %) crushing `coder-next` (56 %).
   T-Bench inverts: coder-next 32.6 % > Gemma 21.3 %. **The agentic loop
   reveals what static benches miss** — tool-driven multi-turn behavior
   is coder-next's design target, and the gap shows.
2. **Vendor claim is honest for coder-next.** 32.6 % measured vs 36.2 %
   vendor = 3.6 pp gap, consistent with MLX 6-bit quant cost on a model
   the vendor likely measured at BF16 or close.
3. **Phase A → B decision** (plan §4): leg 1 = 32.6 % (>25 %), leg 2 =
   21.3 % (between 10 and 25 %). Neither rule fires exactly; both legs
   produce real positive signal (>>10 %), one is solidly in vendor-parity
   range. **Proceeding to full Phase B** to fill the remaining 5 local
   rows — the bench is producing meaningful local signal worth measuring
   across the whole candidate set.

Per-trial data: `benchmarks/runs/tbench_qwen-qwen3-coder-next_*.{jsonl,_summary.json}`,
`benchmarks/runs/tbench_gemma-4-26b-a4b-it-mlx-6bit_*.{jsonl,_summary.json}`.
Raw Harbor jobs: `.bench-logs/tbench-runs/{coder-next,gemma-26b-a4b-6bit}/`.

## Terminal-Bench 2.0 — Phase B (full backfill, 2026-05-26 → 2026-05-29)

All 7 local models measured. Full table:

| # | Model | Score | PASS / FAIL | Errored | Wall-clock | Per-task mean |
|---|---|---|---|---|---|---|
| A1 | `qwen/qwen3-coder-next` (6-bit) | **32.6 %** (vendor 36.2) | 29 / 60 | 43 | 16.8 h | 11.3 min |
| B5 | `qwen3.6-27b` dense (6-bit) | **31.5 %** | 28 / 61 | 58 | 18.9 h | 12.7 min |
| B3 | `qwen3.6-35b-a3b@6bit` | **28.1 %** | 25 / 64 | 47 | 15.6 h | 10.5 min |
| B4 | `gemma-4-31b-it-mlx` (8-bit dense) | **22.5 %** | 20 / 69 | 49 | 17.4 h | 11.7 min |
| A2 | `gemma-4-26b-a4b@6bit` | **21.3 %** | 19 / 69 | 40 | 14.4 h | 9.7 min |
| B2 | `gemma-4-26b-a4b@4bit` | **20.2 %** | 18 / 68 | 44 | 16.0 h | 10.8 min |
| B1 | `gemma-4-e4b` (4B/8-bit) | **4.5 %** | 4 / 85 | 10 | 6.4 h | 4.3 min |

**Total Phase A+B wall-clock: ~105 h across 7 legs** — under the plan's
~100 h B-full estimate, but only because the 0.5x agent-timeout cap shaved
worst-case 4–5× on the 14 long-timeout outlier tasks. Without the cap, full
B would have taken ~250+ h.

### What the agentic loop reveals that static benches miss

This is the headline finding — T-Bench cleanly inverts LCB at the top:

| Model | LCB v6 rank | T-Bench rank | Δ |
|---|---|---|---|
| `gemma-4-26b-a4b@6bit` | **1st** (80 %) | 5th (21.3 %) | **−4 spots** |
| `gemma-4-31b` dense | 2nd (76 %) | 4th (22.5 %) | −2 spots |
| `gemma-4-26b-a4b@4bit` | 3rd (66 %) | 6th (20.2 %) | −3 spots |
| `qwen3.6-27b` dense | 4th (62 %) | **2nd** (31.5 %) | +2 spots |
| `qwen/qwen3-coder-next` | 5th (56 %) | **1st** (32.6 %) | **+4 spots** |
| `qwen3.6-35b-a3b@6bit` | 6th (54 %) | 3rd (28.1 %) | +3 spots |

The Qwen models — trained with explicit agentic-loop targets — all gain
ranks. The Gemma models — trained as general-purpose / one-shot — all lose
ranks. This was the resolution the post-Phase-2 analysis predicted but
couldn't pin down without a multi-turn agent signal on the rig.

### Per-leg findings

1. **`coder-next` 32.6 % vs vendor 36.2 %** — 3.6 pp gap, consistent with
   MLX 6-bit quant cost on a model the vendor likely measured at BF16. The
   vendor's branding for it as the agentic-default is honest.
2. **`qwen3.6-27b` dense 31.5 %** — the knowledge king lands #2 on T-Bench
   too. 1.1 pp behind coder-next; 6× slower decode (~20 t/s vs 67 t/s). For
   one-shot hard problems it's still the right pick; for high-turn-count
   agentic loops, coder-next's speed advantage compounds.
3. **`qwen3.6-35b-a3b@6bit` 28.1 %** — F1 thinking-format guard PASSED
   easily (threshold was ≤ 5 % to abort 27b). Thinking format works fine
   with terminus-2 on the LM Studio → LiteLLM chain. The earlier MoE-MLX
   tool-call regression concern (`qwen3.6-35b-a3b` on `mlx-community` 4-bit
   checkpoints) does not appear on the 6-bit variant.
4. **`gemma-4-31b` dense 22.5 %** — the Phase 2 LCB verdict generalizes:
   31b dense ties or loses to 26B-A4B@6bit on agentic shell too. Demote /
   skip in normal rotation holds.
5. **Quant A/B on 26B-A4B: @4bit 20.2 % vs @6bit 21.3 % — 1.1 pp gap.**
   The 14 pp LCB penalty from 4-bit doesn't bite on agentic shell. Likely
   because T-Bench's bottleneck is multi-turn coordination + tool use, not
   the kind of one-shot algorithmic correctness LCB tests. Operationally:
   for agentic-only workloads on this rig, `gemma-4-26b-a4b@4bit` is a
   reasonable substitute for `@6bit` at 1/2 the weights size — but
   `coder-next` beats both by 11–12 pp anyway.
6. **`gemma-4-e4b` 4.5 %** — confirmed 4B size floor on T-Bench. The agent
   runs cleanly (10 errored vs 40+ on bigger models — the model isn't
   timing out, it just doesn't solve), but the verifier scores 0 on 85/89
   tasks. Useful as FIM / quick-call slot only, not agentic.

### Operational findings

- **`--agent-timeout-multiplier 0.5` is essential on this rig.** Without
  it, leg 1's first 3 tasks alone consumed 144 min for zero passes; the
  14 outlier tasks declaring >60-min agent budgets would have made each
  leg 60+ h. Published scores are a defensible floor; plan estimate is
  full-budget would lift ≤5 pp per leg.
- **First ~5 tasks of T-Bench 2.0 are all in the hardest decile by
  declared timeout.** Every leg sees 0 passes for the first 2-3 h.
  Don't extrapolate from the early window — wait for task ~15 before
  projecting final score.
- **Two Harbor 0.8.0 quirks to know about:** (1) on-disk `result.json`
  excludes `trial_results` even at job completion — score must come from
  `stats.evals[*].reward_stats`. (2) The `--agent-timeout-multiplier` flag
  multiplies the per-task `timeout_sec` declared in each `task.toml`; it
  doesn't impose a flat cap. Run-time bound is therefore the sum of
  multiplier × declared-timeout, not a fixed budget.
- **macOS lacks `setsid`.** The plan template referenced
  `nohup setsid bash ...` from prior LCB driver scripts but that fails on
  macOS Darwin. Subshell double-fork `( nohup bash ...sh > /tmp/...log 2>&1 & )`
  works and gives PPID=1 (launchd). Same F3 silent-kill protection.

Per-trial JSONLs and summaries: `benchmarks/runs/tbench_*.{jsonl,_summary.json}`
(seven pairs). Raw Harbor jobs: `.bench-logs/tbench-runs/<job_name>/`.

## Phase 3 #10 — DeepSeek V4 Flash (blocked 2026-05-29 → OOM fixed, partial sweep 2026-05-30)

Plan: [`docs/benchmark-plans/2026-05-29-deepseek-v4-flash-phase-3.md`](../../../docs/benchmark-plans/2026-05-29-deepseek-v4-flash-phase-3.md).

> **⏩ Current status (2026-05-30): OOM fixed; knowledge sweep partially run.** The
> original blocker is resolved (see the two addenda below). With the
> [`cache-materialize`](../../../patches/mlx-lm-deepseek-v4-cache-materialize.patch) patch the
> model now benches cleanly on a single long-lived server. Measured so far: **MMLU 44 %,
> GPQA 24 %, HumanEval 48 %** (n=100 each), tool-calling jdhodges **40/40 completed (8 correct)**.
> Still pending: MATH, DROP, LiveCodeBench, tool-calling Veerman, Terminal-Bench, throughput.
> Plan for the rest: [`docs/benchmark-plans/2026-05-30-deepseek-v4-flash-remaining-benches.md`](../../../docs/benchmark-plans/2026-05-30-deepseek-v4-flash-remaining-benches.md).
> The numbers in the historical "blocked" narrative below are superseded by **Addendum 2**.

Status (historical, 2026-05-29): **full sweep aborted at Step 3b (tool-calling jdhodges).** The
runtime stack does not produce reliable inference at scale on this rig
yet; bench numbers below are partial and not comparable.

### What ran

| Step | Bench | Outcome |
|---|---|---|
| Pre-flight | mlx_lm.server + CLI generate | ✅ smoke chat returned coherent reply at warm-load |
| 3a | speed_probe (3 prompts) | ✅ 26 t/s steady-state on the code prompt; cold-load fine |
| 3b | tool_call_bench jdhodges (40) | ⚠ **12.5 % (5/40)** — see "why this is a floor, not a score" below |
| 3b | tool_call_bench veerman (12) | Skipped — same blocker would apply |
| 3c–h | bench2.py knowledge + LCB | Skipped — same blocker |
| 4 | bench.py throughput | Skipped — same blocker |
| 5 | T-Bench 2.0 | Skipped — multi-turn agent loop guarantees the blocker fires |

### The blocker: `RuntimeError: [metal::malloc] Resource limit (499000) exceeded`

The DeepSeek V4 model port in [mlx-lm PR #1192](https://github.com/ml-explore/mlx-lm/pull/1192)
uses an MLA compressor + indexer (`deepseek_v4.py`) in its attention path. The
M4 Max Metal device reports a hardware `resource_limit: 499000` (checked via
`mx.device_info()`).

> **Root cause corrected 2026-05-30.** This is a *count of **live resident**
> Metal buffers* (the allocator's `num_resources_` vs `resource_limit_` /
> `ResidencySet`), **not** "per command buffer" and **not** cross-request cache
> accumulation. The model leaks **~1 live buffer per layer (43 layers) per
> decode step** in the compressor/indexer, growing linearly until the count cap
> is hit at **~11,300 generated tokens regardless of prompt length**. Ablation:
> ~83% compressor/indexer, ~17% core attention; MoE/hyper-connections don't leak.
> Full evidence in [`docs/deepseek-v4-flash-metal-oom-investigation.md`](../../../docs/deepseek-v4-flash-metal-oom-investigation.md)
> §2/§2.4 and the fix plan's Phase 1.5 / Phase 2-revised. The original
> single-forward-pass framing below is superseded.

| Memory observation | Value |
|---|---|
| Metal `max_recommended_working_set_size` | 115.4 GB |
| Metal `memory_size` | 128 GB |
| Metal `max_buffer_length` | 86.6 GB |
| Metal **`resource_limit`** | **499000** ← the blocker |

`resource_limit` is a **count**, not a byte size — no env var
(`set_memory_limit`, `set_wired_limit`, `set_cache_limit`) raises it; it's
fixed by the Apple Metal driver / device class. The fix has to be in the
mlx-lm port: chunking the indexer so no single command buffer references
> 499 000 resources. Filed below.

### Why the jdhodges 12.5 % is a floor, not a score

5 OK / 40 fresh cases (`benchmarks/runs/toolcall_jdhodges__Users_vitor_.lmstudio_models_mlx-community_DeepSeek-V4-Flash-2bit-DQ_20260529_121729_summary.json`).
The 5 passes are all in the `edge_cases` category (5/8 = 62.5 %) — cases
where the correct answer is **plain prose, no tool call** (greetings,
definitions, vague reminders). Every other category scored 0 % flat:

| Category | Pass / Total |
|---|---|
| tool_selection | 0 / 8 |
| argument_accuracy | 0 / 8 |
| multi_tool | 0 / 8 |
| edge_cases | **5 / 8** |
| format_compliance | 0 / 8 |

Two failure modes mixed inside the 35 misses, both expected:

1. **`no_tool_called`** — mlx_lm.server logged `WARNING - Received tools
   but model does not support tool calling`. Consistent with the inventory
   ([`docs/testing-plan.md:54`](../../../docs/testing-plan.md)) marking
   this model as **Tools: —**. The 2-bit DQ checkpoint isn't a
   tool-calling fine-tune.
2. **`request_error: Connection error.`** — 15+ cases timed out (350–960 s
   each) because the Metal `resource_limit` aborted the request mid-decode
   on the server, the urlopen kept waiting on a half-open socket, and
   eventually saw the connection drop. **49 `RuntimeError [metal::malloc]
   Resource limit (499000) exceeded`** entries in
   `.bench-logs/mlx-server-deepseek-v4.log`.

The first ~19 cases ran cleanly (all `no_tool_called`, prose-only output,
no Metal errors) before the prompt cache crossed the resource-count
threshold. That's the smoking gun — cold-cache works, warm-cache fails.

### Wall-clock cost

- Server warm-load (96 GB model): ~5 min cold to first inference.
- Speed probe: 2.5 s for 3 prompts.
- Tool-call jdhodges: **161.5 min** wall-clock (vs ~30 min projected) —
  ~110 min was urlopen waiting for hung requests after the Metal cap fired.
- Total session: ~3 h before pulling the plug.

### Implications for the testing plan

- **Phase 3 #10 cannot complete as written.** Restart-per-bench would
  dodge it for short benches (HumanEval, MMLU, DROP), but every long
  bench (MATH n=100, GPQA n=100, LCB n=50) accumulates enough cache to
  trip mid-run, and T-Bench's multi-turn loop fires it on every task.
- **Operational rule for any future try:** if running this model on this
  rig at all, restart `mlx_lm.server` between requests — there is no
  in-process knob to reset the prompt cache without dropping the model.
  Use [Step D quant A/B](../../../docs/testing-plan.md#step-d--phase-2-quant-ab-variants)
  candidates instead for the freed compute time.
- **Upstream dependency:** mlx-lm PR #1192 (DeepSeek V4 architecture) or
  a successor needs to chunk the indexer's per-forward-pass resource
  count before benching this checkpoint becomes worthwhile again.

### Artifacts

- jdhodges summary: `benchmarks/runs/toolcall_jdhodges__Users_vitor_.lmstudio_models_mlx-community_DeepSeek-V4-Flash-2bit-DQ_20260529_121729_summary.json`
- jdhodges per-case: `benchmarks/runs/toolcall_jdhodges__Users_vitor_.lmstudio_models_mlx-community_DeepSeek-V4-Flash-2bit-DQ_20260529_121729.jsonl`
- Speed probe: `results/speed_probe/_Users_vitor_.lmstudio_models_mlx-community_DeepSeek-V4-Flash-2bit-DQ_20260529_121622_results.json`
- Server log with Metal errors: `.bench-logs/mlx-server-deepseek-v4.log`
- Tool-call driver log: `.bench-logs/toolcall-jdhodges-deepseek-v4-flash.log`
- Setup guide (unchanged): [`docs/deepseek-v4-flash-setup.md`](../../../docs/deepseek-v4-flash-setup.md)
- Detached drivers (unused, kept for the next try):
  `.bench-logs/run-deepseek-v4-flash-{math,gpqa}.sh`,
  `.bench-logs/run-tbench-deepseek-v4-flash.sh`

### Addendum (later same day) — chunked indexer patch + restart-per-batch attempt

After the blocker was documented, two follow-up workarounds were attempted:

1. **Vendor patch `patches/mlx-lm-deepseek-v4-indexer-chunk.patch`** — chunks the MLA indexer over `n_heads` with `mx.eval` + `mx.clear_cache` between chunks. Reduced the OOM rate from 49 → 3 (chunk=8) → 8 (chunk=2). Did **not** fully solve it: even with chunk=2, single-request OOMs migrated to other ops (notably `mx.random.seed` at server bookkeeping) and wedge the device for subsequent requests.
2. **Restart-per-batch operational wrapper** — `.bench-logs/run-deepseek-v4-flash-toolcall-jdhodges-restart-loop.sh` kills + restarts `mlx_lm.server` between each 8-case batch. Ran 3 of 5 batches before being stopped. Per-batch:

| Batch | Score | OOMs in server log | Wall-clock |
|---|---|---|---|
| `sel_*` (short prompts ~75-105 tok) | 0/8 clean prose, 0 OOMs | 0 | 8.7 min |
| `arg_*` (longer prompts, tool-arg schemas) | 0/8 (mostly Connection errors) | ~16 | 32.9 min |
| `multi_*` (longest prompts) | 0/8 (2 prose, 6 errors) | ~26 | 33.4 min |

**Key finding from these attempts:** the restart-per-batch wrapper helps batches with short prompts (`sel_*` ran clean) but does **not** help batches with longer prompts — meaning **single-request OOMs are a real failure mode independent of cross-request cache accumulation**. The bench's first OOM in the un-patched run (case 20, `multi_email_after_calendar_read`) is the same multi-tool category that fails inside a fresh-server batch.

Full investigation + fix plan are now in dedicated docs:
- [`docs/deepseek-v4-flash-metal-oom-investigation.md`](../../../docs/deepseek-v4-flash-metal-oom-investigation.md) — root cause, all test runs, hypotheses, external signals (PR #1192 stalled since 2026-05-01; spicyneuron's 4000-token reproducer and `fix-ds4` fork)
- [`docs/deepseek-v4-flash-metal-oom-fix-plan.md`](../../../docs/deepseek-v4-flash-metal-oom-fix-plan.md) — confidence-ordered hypothesis-application plan with exact edits, apply commands, and pass/fail tests for each step

**Daily-driver implications (FIXED 2026-05-30):** the per-decode-step live-buffer leak is
**fixed and reproducer-verified.** Fix = [`patches/mlx-lm-deepseek-v4-cache-materialize.patch`](../../../patches/mlx-lm-deepseek-v4-cache-materialize.patch),
one hunk in `DeepseekV4Model.__call__` that `mx.eval`s every per-layer cache array each
forward (cuts the un-detached lazy graphs in `PoolingCache`/`RotatingKVCache` and their
batched variants). The forced-generation reproducer now streams **19,989 tokens clean at
31.3 t/s with 0 Metal OOMs** (baseline died at 11,314; no throughput regression), leak
slope 205 → 7 KB/step. (Dead ends en route: patching `PoolingCache` then `BatchPoolingCache`
both still OOMed on the *server* path — the BatchGenerator uses `Batch*` caches and the
dominant batch leak is `BatchRotatingKVCache`; the single model-forward choke point covers
all cache classes at once. The earlier "H1 per-layer eval" idea also FAILED — can't reclaim
live buffers.) **Done-bar #1 also PASSED:** full 40-case jdhodges sweep on one long-lived
server = **40/40 completed, 0 Metal OOMs, 19.8 min** (vs unpatched 49 OOMs, aborted at case
20). Tool-call score 8/40 (all `edge_cases` — model isn't a tool-caller, unchanged by fix).
Remaining: 30-turn chat (#3, manual) + an optional full Phase 3 #10 knowledge/throughput
sweep now that the runtime is stable. See investigation doc §2 and fix plan Phase 2-revised (R5).

### Addendum 2 (2026-05-30) — knowledge-bench results + upstream submission

With the runtime stable, the knowledge sweep was re-run on a **single long-lived patched
server** (no restart wrapper), greedy `temp=0`, thinking=OFF, per-request `max_tokens` capped
(2048 MMLU / 4096 GPQA+HumanEval) to bound the separate 2-bit degeneration runaway. This run
doubled as the pre-submission OOM soak: **300 requests, 0 `metal::malloc`, 0 errors, ~2h44m**.

| Bench | n | Score | Degenerate (TRUNC) | Wall-clock | Metal OOMs |
|---|---|---|---|---|---|
| MMLU | 100 | **44 %** | 0 | 16 min | 0 |
| GPQA | 100 | **24 %** | 36 | 96 min | 0 |
| HumanEval | 100 | **48 %** | 15 | 52 min | 0 |
| Tool-calling jdhodges (40) | 40 | 8/40 (**20 %**) | — | 19.8 min | 0 |
| Tool-calling Veerman (12) | 12 | 2/12 (**17 %**) | — | 5.5 min | 0 |
| **Soak total** | **300** | — | 51 | **~2h44m** | **0** |

**Remaining-bench queue** (post-soak, single long-lived server, same config, 0 OOMs):

| Bench | n | Score | TRUNC | Wall-clock | Metal OOMs |
|---|---|---|---|---|---|
| DROP | 100 | **71 %** — best knowledge result; extractive QA survives 2-bit | 0 | 16 min | 0 |
| MATH | 100 | **47 %** — floor; 41 % degenerated to the cap at temp=0 | 41 | 105 min | 0 |
| LiveCodeBench v6 | 50 | **6 %** — floor; 80 % degenerated, 2-bit can't sustain long codegen | 40 | 171 min | 0 |

> **Tool-calling is N/A on this build, not a quality signal.** The MLX conversion ships a
> 24-line `chat_template.jinja` with **no tools branch** and no tool special tokens, so
> `mlx_lm.server` logs *"model does not support tool calling"* and **drops the `tools` array on
> every request** — the model never sees a tool schema and can only answer in prose
> (`no_tool_called`). The jdhodges 8/40 + Veerman 2/12 passes are all prose-is-correct edge
> cases. `tool_combined` = 10/52 (19.2 %) is plotted for completeness but reflects the missing
> template (a conversion gap, fixable), **not** the model's inherent tool ability. 2-bit quant
> would further hurt structured emission even with a proper template.

> **✅ Tool calling RECOVERED — and the native format makes it excellent (2026-05-31).**
> The conversion ships no tool template. Two configs were tested on the *same 2-bit checkpoint*:
>
> | Tool format | jdhodges | Veerman | combined |
> |---|---|---|---|
> | Hermes `<tool_call>` (workaround template + `deepseek_json` parser) | 33/40 (82 %) | 6/12 (50 %) | 39/52 (75 %) |
> | **Native DSML** (official template #16 + `deepseek_dsml` parser) | **39/40 (98 %)** | **9/12 (75 %)** | **48/52 (92 %)** |
>
> Native **DSML matches the best full-size local on this rig** (qwen3.6-35b-a3b, 98 % jdhodges).
> The Hermes "partial multi-tool" misses were a **format tax, not a 2-bit ceiling** — in DSML the
> model emits parallel calls natively (multi_tool 3/8 → **8/8**). Both configs hit the same `>`-token
> BPE-merge gotcha in mlx-lm's marker matching (json_tools fix: ml-explore/mlx-lm#1335 / #1336; the
> `deepseek_dsml` parser uses the same prefix-marker trick). Use **native DSML** going forward.
> Analysis: [`docs/benchmark-plans/2026-05-30-deepseek-v4-flash-tool-template.md`](../../../docs/benchmark-plans/2026-05-30-deepseek-v4-flash-tool-template.md).

Reading it:
- **OOM fix vindicated under sustained load.** 51 of the 300 requests ran the full token cap
  (long/degenerate generations — the hardest case for the residency leak) on a server that
  never restarted, with zero residency errors. Strongest cross-request evidence to date.
- **Scores are the 2-bit DQ quality floor**, not a runtime issue — well below the Gemma/Qwen
  locals (MMLU 65–88, GPQA 34–70, HumanEval 87–98 on this rig). Orthogonal to the OOM fix.
- **Degeneration is long-form only**: 0 % on short MMLU answers, 36 %/15 % on the
  longer-output GPQA/HumanEval (repetition looping + some legit-but-rambling answers
  guillotined at the cap). No sampling knob fixes it; 4-bit (the real remedy) exceeds 128 GB.

**Upstream submission (2026-05-30):** the cache-materialize fix was filed upstream —
issue [ml-explore/mlx-lm#1332](https://github.com/ml-explore/mlx-lm/issues/1332),
PR [Blaizzy/mlx-lm#25](https://github.com/Blaizzy/mlx-lm/pull/25) (against the #1192 head
branch), and a heads-up comment on [#1192](https://github.com/ml-explore/mlx-lm/pull/1192#issuecomment-4585428668).
Standalone writeup: [`docs/deepseek-v4-flash-metal-oom-upstream-writeup.md`](../../../docs/deepseek-v4-flash-metal-oom-upstream-writeup.md).

**Still pending** (see [`docs/benchmark-plans/2026-05-30-deepseek-v4-flash-remaining-benches.md`](../../../docs/benchmark-plans/2026-05-30-deepseek-v4-flash-remaining-benches.md)):
MATH, DROP, LiveCodeBench v6, Terminal-Bench 2.0, the 4 throughput
scenarios. Charts (`results/charts/chart_m4max_phase1_*.png`) regenerated to include the
measured cells; blank cells = not yet measured.

## MiniMax-M2.5-3bit — feasibility ABORTED (GPU kernel panic ×3, 2026-07-03 → 07-04)

Plan: [`docs/benchmark-plans/2026-07-03-minimax-m2.5-feasibility.md`](../../../docs/benchmark-plans/2026-07-03-minimax-m2.5-feasibility.md).

> **⛔ VERDICT: NO-GO on this rig/OS.** `mlx-community/MiniMax-M2.5-3bit` (93 GiB weights,
> `minimax_m2` arch, 256E/8A MoE, 62 layers, no MLA) loads and generates coherently, and
> its *quality* is strong — but under sustained inference it **reproducibly hard
> kernel-panics the Mac Studio** (three times), in Apple's GPU driver
> (`IOGPUFamily` / `IOGPUGroupMemory` / `AGXG16X`), across **every** config tried.
> Deployment is impossible while the host crashes. Do **not** re-test on this stack.

### Cheap-signal results — partial (sweep never completed; all crashed out)

| Bench | Result | Notes |
|---|---|---|
| Tool calls jdhodges (40) | **97.5 %** (39/40) | strong mechanics |
| Tool calls veerman (12) | **58.3 %** (7/12) | under-agency, prompt-addressable — A/B nudge was a trade (+agentic / −mechanics), not a win |
| HumanEval | **95.8 % raw / 97.2 % hang-adj** | cut at 72/100 (parallel-4 dead-request hangs) |
| LiveCodeBench v6 | **68 % raw / 74 % hang-adj** (26/38) | crashed 3× before finishing 50 |
| MMLU | — | abandoned (host crashes) |

Config: ctx 32768, parallel 1, temp 0, seed 42. "hang-adj" excludes `p=0 c=0`
dead-request timeouts (infra failures, not wrong answers). All numbers are **partial
and not fully comparable** — the run never completed.

### The blocker: reproducible GPU-driver kernel panic ×3

| # | Config | Memory at crash | Panic |
|---|---|---|---|
| 1 | ctx 65000 / parallel 4 / fp16 KV | ~ceiling | `remove_memory_object() memory object not found` @IOGPUGroupMemory.cpp:323 |
| 2 | ctx 32768 / parallel 1 / fp16 KV | **OK** (0 % compressor) | `pending memory object … non pending hash` @:528 |
| 3 | ctx 32768 / parallel 1 / **KV-quant 8-bit** | **OK** | same @:528; panicked task = `LM Studio Helper (GPU)` |

`IOGPUFamily 129.3.2 / AGXG16X 345.20.4`, macOS 25D125 (Darwin 25.3.0), M4 Max T6041.
A reproducible **Apple GPU-driver bug** in `IOGPUGroupMemory`'s object-tracking hash,
triggered by MLX's Metal alloc/free pattern for this model — **independent of parallelism,
context, memory pressure, and KV quantization.** Nothing application-side fixes it:
memory tuning, config tuning, and KV quant were all tried and all crashed (KV quant only
*delayed* it, surviving ~21 long generations). Correlates with long-generation / large-KV
load: tool-calling and HumanEval (short gens) never crashed; LCB's 20k–32k-token reasoning
spirals did. Soft precursor = the intermittent `p=0 c=0` dead-request hangs.

**Only external changes could revisit it:** an Apple macOS/GPU-driver update, an MLX/LM
Studio release that changes the Metal allocation pattern, or a **different runtime** (GGUF
via llama.cpp — different Metal path, untested, a separate investigation). Full detail:
the plan doc's "Kernel panic — THREE TIMES" section. Contrast with DeepSeek-V4 (§ above):
that was a fixable *mlx-lm buffer leak*; this is a *driver-level panic* with no
application-side remedy.

## agents-a1-xl-mlx — cheap-signal + coding/knowledge tail (2026-07-04 → 07-05)

**Qwen3.5 MoE** (`qwen3_5_moe` arch, self-IDs as "Qwen3.5 / Alibaba Tongyi"),
MLX 6-bit, 27.8 GB on disk (29.90 GB resident), ctx 131712. Comfortable-fit class
(~48 GB resident with KV, no swap) — **not** the memory/panic class that blocked
DeepSeek-V4 / MiniMax-M2.5. Ran the entire cheap tail with **zero crashes**.
Plan + full write-up: [`docs/benchmark-plans/2026-07-04-agents-a1-xl.md`](../../../docs/benchmark-plans/2026-07-04-agents-a1-xl.md).

| Signal | Score | Notes |
|---|---|---|
| jdhodges (40) | **92.5%** (37/40) | sel 7/8 · args 8/8 · multi 6/8 · edge 8/8 · format 8/8; 7.4 min |
| Veerman (12) | **83.3%** (10/12) | action 6/7 · **restraint 2/2** · hard 2/3; ties the leaders |
| HumanEval | **97%** (97/100) | 1 trunc; 75 min; ties gemma@6bit 97 |
| MMLU | **82%** (82/100) | 3 trunc; 104 min (2× 65k-cap spirals @ ~17 min) |
| LiveCodeBench v6 | **64%** (32/50) | 2 trunc; **240 min** (2× 65k-cap spirals @ ~29 min) |
| Speed | ~40 t/s think / ~65–80 short | MoE; heavy reasoning inflates wall-clock |

**Headline:** strong well-rounded MoE — top-tier tool-calling + HumanEval, near-top
MMLU (< 27b 88), solid mid-pack LCB (> coder-next 56 / 27b 62 / 35b-a3b 54; < gemma@6bit 80).
**Caveat = thinking tax:** emits reasoning tokens on everything (109 on "2+2", 18k on a
"leetcode/easy", 65k-cap spirals on both MMLU and LCB) → far slower than a same-size
non-thinking model; a full MATH/DROP/GPQA sweep would be 20–40 h (Qwen-3.6-dense phenotype).
**Gate:** marginal pass on coding only (LCB 64% vs 27b 62%, +2 pp; MMLU 82% < 85% miss) →
**expensive tail DEFERRED** (thin justification, already well-characterized).
**Slot:** solid mid-tier all-rounder; does **not** displace coder-next (agentic speed),
27b (knowledge), or gemma@6bit (coding). Best fit = tool-calling generalist, but the
thinking tax makes it slower than coder-next for real agentic loops.

Operational note: arrived co-resident with hermes-4-70b + qwen3.6-27b (~110 GB weights,
swap maxed, `Spill=YES`); unloaded both per the single-large-model residency rule before
benching (swap 19.9 GB → 166 MB). All numbers above are single-model / clean-state.

## kimi-dev-72b — cheap-signal gate ABORTED (speed 7 t/s, 2026-07-05)

`unsloth/Kimi-Dev-72B-GGUF` UD-Q6_K_XL (arch `qwen2` / Qwen2.5-72B base, 73B dense,
62.55 GiB weights, 67.16 GB resident at ctx 32768 / parallel 1). SWE-bench Verified
60.4 % is its headline (SOTA open-source at release). **Aborted at the speed step of
the cheap-signal ladder** — never reached graded coding runs.

| Signal | Result | Notes |
|---|---|---|
| Load | ✅ clean, 36 s | Stock llama.cpp 2.23.1; the red LM Studio arch badge was benign. |
| Pre-flight | ✅ PASS | Warmup answers "4". |
| **Speed** | **~7 tok/s** | 3 runs: 7.0 / 7.0 / 7.1 t/s (trivial), 6.9 / 7.2 / 7.2 (mmlu). Compute-bound (GPU 100 %, 54 W); memory state (87 GB no-swap vs 135 GB swapping) did **not** move the number. |
| Tool-calling | ✗ no structured calls | Given a tool + explicit instruction, emitted prose *about* calling it inside `◁think▷`, `tool_calls: []`. Not a tool-calling fine-tune → floor, like `deepseek-v4-flash-dq`. |
| Reasoning | mandatory `◁think▷` spirals | **Non-standard markers** (not `<think>`) → LM Studio does **not** parse them (`reasoning_tokens: 0`); raw reasoning lands in `content`. Spirals even on "2+2" (180 tok, cut mid-think at the probe cap). |

**Verdict — NO-GO on speed.** At ~7 t/s (½ of `qwen3.6-27b` 20 t/s, ⅓ of
`gemma-4-31b` dense 13.7 — the **slowest model benched on this rig**), and with a
mandatory thinking spiral inflating effective throughput further, it is disqualified
as a daily-driver / agentic model regardless of coding quality. Its only differentiating
axis is coding quality (LCB / HumanEval), but a full run would be ~1–2 rig-days at this
speed for a model already ruled out — **not worth the compute.** Coding-quality numbers
**deferred / not measured.**

**Revisit only if:** a faster path appears — a lighter quant that keeps the SWE quality,
a speculative-decoding draft model (LM Studio supports `--speculative-draft-*`), or a
smaller Kimi-Dev distillation. Until then, `qwen3.6-27b` (LCB 62 %) remains the coding-quality
reference and `gemma-4-26b-a4b@6bit` (LCB 80 %) the coding leader.

## DeepSeek-V4-Flash GGUF (IQ2_XS) — ✅ GO via llama.cpp (the runtime that MLX never could be, 2026-07-05)

**The headline: DeepSeek-V4-Flash runs cleanly on this rig for the first time.** The MLX
build (`deepseek-v4-flash-dq`) was blocked for weeks by the mlx-lm MLA live-buffer leak
(Metal `resource_limit` at ~11.3k tokens). The GGUF build (`teamblobfish/DeepSeek-V4-Flash-GGUF`,
IQ2_XS-XL, 81 GB, 2 shards) on **llama.cpp** uses a completely different Metal path and has
**no leak** — it sustained a **16,384-token single generation with memory dead-flat at
82.3 GB, 0 errors**. The MLX plan's own re-test hypothesis ("GGUF via llama.cpp, a different
Metal path") is confirmed GO.

### The working recipe (not LM Studio-native — see blockers)
Three gates, three fixes:
1. **Arch:** stock llama.cpp 2.23.1 → `unknown model architecture: 'deepseek4'`. **Fix:** upgrade
   the LM Studio GGUF runtime to **2.24.0** (beta channel; `lms runtime get --channel beta ...`).
2. **Repack crash:** even on 2.24.0, LM Studio-native load aborts on the first forward pass —
   `ggml_abort` in the CPU **repack** path (Q8_0 MoE `mul_mat_id`, ref llama.cpp PR #17869).
   `lms load` has no flag for it and the `LLAMA_ARG_REPACK` env is **not honored** by LM Studio's
   `LlamaV4::load` wrapper. **Fix:** run the standalone `llama-server` (LM Studio's own 2.24.0
   binary) with `--no-repack`.
3. **Metal OOM:** default `n_slots=4` overcommits KV. **Fix:** `-np 1`.

```bash
BIN=~/.lmstudio/extensions/backends/llama.cpp-mac-arm64-apple-metal-advsimd-2.24.0
M=~/.lmstudio/models/teamblobfish/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-IQ2_XS-XL-00001-of-00002.gguf
cd "$BIN" && ./llama-server -m "$M" -a deepseek-v4-flash-iq2xs \
  --no-repack -c 32768 -np 1 -ngl 999 --host 127.0.0.1 --port 1235
# harness: LMSTUDIO_URL=http://127.0.0.1:1235/v1
```

**LM Studio-native is BLOCKED** (no repack toggle; env ignored). **MLX-native is BLOCKED**
(`ValueError: Model type deepseek_v4 not supported` on mlx-llm 1.9.1 — LM Studio's MLX engine
never had the arch; the May-29 test used a standalone patched mlx-lm). GGUF-via-standalone is
the only working path on this stack.

### Non-thinking — a key property
**0 reasoning tokens on every generation** (output goes straight to the answer/code — verified
in raw JSONL). Unlike Kimi (`◁think▷` spiral) or MiniMax/Qwen3.6, its **effective throughput
= its raw throughput** — no reasoning tax. This is why ~10 t/s is usable.

### Cheap-signal ladder (IQ2_XS, standalone llama-server, ctx 32768, single-model)
| Signal | Score | Notes |
|---|---|---|
| Speed | **~10 t/s** | vs MLX-DQ's 26 t/s cold probe — but MLX never completed a bench; GGUF is stable. Compute-bound, GPU ~100 %. |
| Feasibility soak | ✅ 16,384 tok single gen | memory flat 82.3 GB, 0 leak/OOM/error — past MLX's ~11.3k death point |
| jdhodges (40) | **87.5 %** (35/40) | **overturns the MLX 12.5 % crash-floor** — DS4 *is* tool-calling capable (near coder-next 90 %). |
| Veerman (12) | **58.3 %** (7/12) | strong mechanics, weak agentic proactivity (p6/p8/p12 tool-mismatch, p7 spiral) — same shape as MiniMax. |
| HumanEval | **88 %** (88/100) | 0 trunc; ~90 % excluding 2 empty-response hiccups. Ties coder-next 89 % — strong for 2-bit. 3.1 h (verbose non-thinking). |
| LiveCodeBench v6 | **86 % partial (6/7)** ⏸ | **INCOMPLETE — stopped at 7/50** (runtime: some cases blow up to 11k tokens/~19 min). Finish overnight — see next steps. |
| MMLU | — | not run |

Occasional **empty-response hiccup** (~2–3 %: 0 tokens returned, counted as FAIL) — low-rate, non-systematic; watch it.

### Verdict + next steps
**GO — DeepSeek-V4-Flash is feasible and genuinely capable on the GGUF path**, and the session's
biggest runtime win. Quality clears the gate (tool-calling 87.5 %, HumanEval 88 %). Two steps
remain to finish the cheap-signal ladder:
1. **Finish LCB v6 overnight** — restart the server (recipe above), then either run the remaining
   43 (`bench2.py livecodebench --examples 50 --only 8,9,...,50 --max-tokens 32768`, then
   **manually merge** with the first 7 — bench2 writes a fresh summary, no auto-merge) OR re-run
   the full 50 fresh for a self-contained summary. Budget ~4–6 h (a few hard cases may hit the
   32 768 cap → ~55 min each). Partial so far: 7/50, 86 %, 0 trunc.
2. **MMLU (100)** after LCB.
Then regenerate charts and update `docs/local-llm-reference.md` if it earns a slot (it's the only
runnable model in the DeepSeek-V4 / large-MoE class on this rig).

Raw data: `benchmarks/runs/{toolcall_*,humaneval_*,livecodebench_*}_deepseek-v4-flash-iq2xs_*`,
`results/speed_probe/deepseek-v4-flash-iq2xs_*`. Plan: [`docs/benchmark-plans/2026-07-05-phase-5-new-arrivals.md`](../../../docs/benchmark-plans/2026-07-05-phase-5-new-arrivals.md).

## MiniMax-M2.5 GGUF (Q3_K_S) — ✅ GO, the MLX NO-GO overturned (2026-07-05)

The marquee Phase 5 experiment: the **MLX build (`mlx-community/minimax-m2.5`, 3-bit)
kernel-panicked the host ×3** in Apple's GPU driver → hard NO-GO. The MLX plan's own
re-test hypothesis was *"a different runtime (GGUF via llama.cpp, a different Metal
path)."* **This is that test — and it's a GO.** llama.cpp's Metal backend allocates GPU
buffers on a different code path than MLX; the panic **did not recur** across load,
probes, and a full sustained soak. The failure was MLX's allocation pattern, **not the
model**.

### Phase 0 feasibility soak — PASS (sole-model, ctx 32768, `--gpu max --parallel 1`)
`unsloth/minimax-m2.5`, Q3_K_S, 98.69 GB resident (estimate 97.91 GiB). LM Studio-native
load (bundled llama.cpp 2.23.1 recognizes `minimax-m2`) — **no fork, no repack flag, no
standalone server needed** (unlike DeepSeek-V4).

| Time | Step | Result |
|---|---|---|
| 18:11 | Load (ctx 32768) | ✅ clean, **no panic**; 98.69 GB resident |
| 18:12 | Probe 1 (trivial) | "4", coherent (153 reasoning tok) |
| 18:13 | Probe 2 (timed medium) | **36.2 t/s**, coherent hash-map explanation |
| 18:14–18:16 | **8k sustained soak** | 5038 tok, `finish=stop` (finished naturally), coherent **~4100-word essay**, **36.8 t/s sustained**, peak **121.9/128 GB**, swap flat 1.58 GB, **0 Metal errors, no panic** |
| 18:30 | Unload | clean → memory back to **12.5 GB baseline (no leak)** |

Pass criteria all green: 8k soak completes coherent, 0 `metal::malloc`, no kernel panic,
memory held steady (no upward trend / OOM), swap flat, host uptime unbroken.
Telemetry: `.bench-logs/minimax-gguf-feasibility-{macmon.jsonl,lmslog.txt}` (repo root).

### Reasoning tokens — parsed cleanly (unlike Kimi)
Card says "no explicit thinking tags," but it **does reason internally** (~145–810
reasoning tok/response, heavier on code). Crucially LM Studio parses them as **structured
`reasoning_tokens`**, so they don't pollute `content` the way Kimi's unparsed `◁think▷`
did. There's a reasoning tax on token count, but the content stays clean and 36.8 t/s is
genuinely usable — **5× Kimi's 7 t/s**, faster than `qwen3.6-27b`.

### Cheap-signal ladder (Q3_K_S, LM Studio :1234, ctx 32768, sole-model)
| Signal | Score | Notes |
|---|---|---|
| Speed (sustained) | **36.8 t/s** | held over the 2.3-min soak; tool-calls 28–31 t/s |
| Feasibility soak | ✅ 8k tok single gen | mem peak 121.9 GB, flat, 0 leak/OOM/panic |
| jdhodges (40) | **95 %** (38/40) | clears the ≥85 % gate; matches MLX pre-crash 97.5 %. 6.9 min, 28.3 t/s |
| Veerman (12) | **75 %** (9/12) | 3 tool_mismatch (p2/p6/p12); same band as base `qwen3.6-35b-a3b` (75 %) — agentic tune did **not** lift the holdout suite |
| HumanEval (100) | **94 %** (94/100) | ~73 min, 36 t/s, **1 trunc** (Q17 `largest_prime_factor` spiraled to the 32k cap → the only FAIL-by-truncation; true ceiling ~94–95 %). Beats DeepSeek-V4 88 %, matches MLX pre-crash 95.8 % — GGUF loses nothing. |
| LCB v6 (50) | **68 %** (34/50) | 32k cap, ~4.5 h. Difficulty split: **easy 15/15 (100 %)**, medium 16/23 (70 %), **hard 3/12 (25 %)**. **5 truncations** (all FAIL); true ceiling ~70–74 %. Above `qwen3.6-27b` (62 %), `kimi` (64 %), `coder-next` (56 %); below Gemma coding leaders (`gemma-4-26b-a4b@6bit` 80 %). Matches MLX-build partial (68 % raw). |
| Terminal-Bench 2.0 | ❌ **NO-GO (memory)** | see below — 98.69 GB model can't coexist with Docker on 128 GB |
| MMLU | — | **not run** (session stopped after tbench NO-GO) |

### Verdict + next steps
**GO — MiniMax-M2.5 is feasible AND fast on the GGUF path**, overturning the MLX NO-GO.
It clears the cheap-signal gate (jdhodges 95 % ≥ 85 %) and is a decisive positive result:
the MiniMax family is runnable on this rig via llama.cpp, and at 36.8 t/s it's a viable
daily-driver-class large MoE (not a cost-trap like Kimi). **HumanEval 94 %** (run
2026-07-05, ~73 min) confirms strong coding. The remaining knowledge tail (LCB v6 → MMLU,
`--max-tokens 32768`, sole-model) was **deferred** — MiniMax (98.69 GB) can't co-exist
with the 01:00 DeepSeek LCB job (82 GB); it earns the rest on a future sole-model session.
Then charts + a `docs/local-llm-reference.md` slot (top-tier local MoE candidate).

### Terminal-Bench 2.0 — ❌ NO-GO (memory coexistence, not capability)

Attempted the full 89-task Harbor run (`terminus-2` agent, Docker); **stopped after 46
trials, all errored, mean 0.0.** Every trial died with `Environment start timed out after
600.0 seconds` — **the Docker task containers can't start.**

**Root cause = memory, decisively.** The model holds **98.69 GB**; the OS + Docker
Desktop's Linux VM consume the rest, leaving **~3 GB free** (macmon showed 125 GB used
from the *first* trial, climbing to a 134 GB peak — over the 128 GB physical, into swap).
Terminal-Bench's amd64-emulated task images (many multi-GB) can't allocate/start in that
sliver → 600 s timeout, 100 % failure.

- **Not a concurrency bug.** Trials fired at an exact 10-min cadence (`-n 1` worked,
  sequential). The 28 lingering containers were **orphans** — Harbor doesn't tear down a
  container when its trial times out, so they accumulate and compound the exhaustion.
- **The first trial failed with free memory** → freeing more won't help enough: dropping
  ctx 64k→32k recovers only ~3.5 GB vs the 20–40 GB Docker needs.
- **Why 27b succeeded and this can't:** `qwen3.6-27b` is ~20 GB → ~100 GB free for Docker.
  A 98.69 GB model leaves ~3 GB. **Terminal-Bench requires a model that leaves Docker
  headroom; ≥~70 GB models are effectively locked out on a 128 GB rig.** Same *class* of
  operational NO-GO as Kimi's speed wall — a rig limit, not a model-quality verdict.
- Raw job data: `.bench-logs/tbench-runs/minimax-m2.5/` (46 `EnvironmentStartTimeoutError`).

### Context length — native **196,608 (192k)**, usable **~64k** on this rig (corrects the plan)

The Phase-5 plan's "65536 won't fit / 32768 is the ceiling" was inherited from the MLX
build and is **wrong for the GGUF**. Measured via `lms load --estimate-only` (no load) +
GGUF metadata:

- **Native trained cap:** `minimax-m2.context_length = 196608` (RoPE freq_base 5e6, no
  YaRN). Beyond needs RoPE scaling.
- **KV is cheap** (GQA, 48 heads / 8 KV heads, ~103 KB/token): footprint 32k→131k adds
  only ~9.7 GiB. Estimates: 32k=97.9, 64k=101.1, 96k=104.4, 192k=114.0 GiB.
- **Usable inference ceiling = 64,000 tokens; hard cliff at 64,512.** Swept empirically
  (load at ctx N → real inference): **32768, 40960, 49152, 57344, 59392, 61440, 63488,
  63744, 64000 all COMPUTE OK**; **64512, 65024, 65280, 65535, 65536 all return
  `{"error":"Compute error."}`** (the model *loads* fine at those — shows IDLE/98.69 GB —
  but every inference errors). Sharp wall in the 2^16 region → a Metal KV-buffer limit for
  `minimax-m2`, not a memory-fit issue (footprint at 64k is only 101 GiB). **Recommended
  operating ctx = 61,440 (60k)** — safe margin below the cliff, validated with a real
  2,693-token generation over the LAN. This is ~2× the 32k the benches ran at.
  (Earlier draft of this note said "usable = 32k" — WRONG; that was before the sweep. The
  original `Compute error` was seen only at 65536, which happens to be just past the cliff.)
  Peak-memory math (est + ~17 GB overhead): 60k→~124 GB, 96k→~129 (also over 128), 192k→~138.
- ⚠️ **`max_tokens` cap:** a request with `max_tokens=60000` returned **HTTP 400 Bad
  Request** (the LCB truncation-recovery rerun failed on this). LM Studio rejects very large
  `max_tokens`; the truncated LCB Q8/Q19 were **not** recovered.

### Terminal-Bench path forward — distributed (model rig + separate Docker host)

The tbench memory NO-GO above is **rig-local**, not fundamental: the model calls are cheap,
it's the Docker containers that need RAM. **Split them across two machines.** LM Studio on
the 128 GB rig serves the model on the LAN; a *second* Mac runs Docker + the terminus-2
agent, hitting the rig over the network. Neither competes for RAM.

- **Rig (model server) — set up & validated 2026-07-06:** firewall off; `lms server start
  --bind 0.0.0.0 --port 1234` (listens `*:1234`); LAN IP **192.168.68.123**; MiniMax loaded
  **at 61440 (60k)** — the safe max below the 64,512 compute cliff (see Context length).
  Verified: a 2,693-tok generation via `curl http://192.168.68.123:1234/v1/chat/completions`
  from the LAN returns coherent output, `finish=stop`. 48h TTL.
- **Docker host (other Apple-Silicon Mac):** ready-to-run driver at
  `.bench-logs/run-tbench-minimax-REMOTE.sh` — `OPENAI_API_BASE=http://192.168.68.123:1234/v1`,
  `--model openai/unsloth/minimax-m2.5`, `--environment-build-timeout 3.0` (amd64 emulation
  is slow), `-n` sized to that Mac's free Docker RAM, orphan-cleanup around the run. Caveat:
  task images are still amd64-emulated on Apple Silicon (slow starts) — an x86 Linux host
  would be strictly better, but free RAM is the thing that actually unblocks it.

Raw data: `benchmarks/runs/{toolcall_{jdhodges,veerman},humaneval,livecodebench}_unsloth_minimax-m2.5_*`,
`results/speed_probe/unsloth_minimax-m2.5_*`. Plan: [`docs/benchmark-plans/2026-07-05-phase-5-new-arrivals.md`](../../../docs/benchmark-plans/2026-07-05-phase-5-new-arrivals.md).
