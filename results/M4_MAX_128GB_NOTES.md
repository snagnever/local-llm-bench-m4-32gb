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

## Phase 3 #10 — DeepSeek V4 Flash (blocked, 2026-05-29)

Plan: [`docs/benchmark-plans/2026-05-29-deepseek-v4-flash-phase-3.md`](../../../docs/benchmark-plans/2026-05-29-deepseek-v4-flash-phase-3.md).
Status: **full sweep aborted at Step 3b (tool-calling jdhodges).** The
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
