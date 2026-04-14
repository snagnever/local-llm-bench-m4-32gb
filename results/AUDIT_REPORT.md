# Benchmark Audit Report (Historical)

**Status:** HISTORICAL — all bugs described here have been FIXED
**Original audit date:** 2026-04-11
**Bugs fixed in:** `scripts/bench2.py`
**MATH reruns completed:** 2026-04-11 to 2026-04-12
**Current canonical results:** see `FINAL_100Q_RESULTS.md`

---

## Summary

This document describes three bugs in the original `bench.py` harness that produced incorrect scores, particularly for MATH benchmarks. All bugs have been fixed in `bench2.py`, and the affected MATH runs have been rerun with corrected scores.

**Before/after MATH scores:**

| Model | Old (bugged) | New (corrected) | Change |
|-------|-------------|-----------------|--------|
| Qwen3-Coder | 74% | **78%** | +4pp |
| Gemma 4 | 59% | **82%** | **+23pp** |
| huihui-ref | 51% | **73%** | **+22pp** |

The corrected scores completely change the ranking — Gemma 4 went from *worst* at MATH to *best*.

---

## Bug 1: MATH `max_tokens=4096` hardcoded

**What was wrong:** `run_math()` in bench.py called `api_call(messages, max_tokens=4096)` while all other benchmarks used `max_tokens=MAX_TOKENS` (16384). This was a copy-paste error.

**Impact:** Thinking models (Gemma 4, huihui-ref) generate 200-4000+ reasoning tokens before producing their answer. With only 4096 total tokens, many responses were truncated before reaching `\boxed{answer}`, causing questions to be scored as wrong even when the model was reasoning correctly.

**Evidence from LM Studio logs (confirmed from server logs):**

| Model | MATH requests (100q run) | Truncated at 4096 | % affected |
|-------|-------------------------|-------------------|------------|
| Qwen3-Coder | 100 | **1** (1%) | barely affected — doesn't think |
| Gemma 4 | 99 | **25** (25%) | seriously affected |
| huihui-ref | 100 | **32** (32%) | worst affected |

Truncated entries had `finish_reason: "length"` and typically used ~4093 thinking tokens with only 3 visible tokens (i.e., no answer).

**Fix:** bench2.py uses `max_tokens=32768` for all benchmarks by default. Rerun results confirmed 0 truncations on MATH at 32768 for Gemma 4 and Qwen3-Coder, and only 11 for huihui-ref (these are genuine thinking spirals that need even more budget).

## Bug 2: Broken `\boxed{}` regex

**What was wrong:** The original regex `\\boxed\{([^}]+)\}` cannot handle nested braces. MATH answers commonly contain things like `\boxed{\frac{1}{2}}`, `\boxed{x^{2}}`, `\boxed{2^{n-1}}`.

**Example:** For `\boxed{\frac{1}{2}}`:
- Broken regex captures: `\frac{1` (stops at first `}`)
- Correct extraction: `\frac{1}{2}`

**Impact:** Systematically undercounted correct answers for ALL models on roughly 30-50% of MATH problems that had nested-brace answers.

**Fix:** bench2.py uses `extract_boxed()` with brace-depth counting. Also added LaTeX normalization so `\frac83` matches `\frac{8}{3}` and `\left(x\right)` matches `(x)`.

## Bug 3: HumanEval had fake hardcoded questions (earlier bench.py version)

**What was wrong:** An early version of bench.py had 20 hardcoded toy Python problems instead of loading the real `openai/openai_humaneval` dataset (164 problems).

**Impact on 100q runs:** NONE. This bug was caught and fixed before any 100q run was executed. The 100q HumanEval scores (Qwen3-Coder 94%, Gemma 4 99%, huihui 91%) are from the real dataset.

**Fix:** bench2.py loads `openai/openai_humaneval` and samples 100 real problems.

---

## Other Issues Also Fixed in bench2.py

### 1. No per-question data saved
**Old bench.py:** Only saved aggregate score to JSON. No way to verify individual answers, debug scoring, or audit the run.
**bench2.py:** Saves full per-question JSONL with prompt, response, token counts, timing, grading details, and system state. See METHODOLOGY.md for the full schema.

### 2. No timing data saved
**Old bench.py:** Per-question timing printed to stdout only, lost when terminal closed.
**bench2.py:** All timing data persisted in the per-question JSONL.

### 3. No raw response text saved
**Old bench.py:** Could not re-grade answers later if scoring logic changed.
**bench2.py:** Saves full raw response text in JSONL for later re-analysis.

### 4. No model/hardware metadata saved
**Old bench.py:** No record of which quant, context length, or hardware was used.
**bench2.py:** Summary JSON includes full model config (arch, quant, size, context_length) and hardware info (CPU, RAM, macOS version).

---

## Verdict By Benchmark (current state)

### MMLU (100q) — VALID, no bugs
| Model | Score | Confidence |
|-------|-------|------------|
| Qwen3-Coder | 67% | HIGH |
| Gemma 4 | 84% | HIGH |
| huihui-ref | 78% | HIGH |

### HumanEval (100q) — VALID, no bugs
| Model | Score | Confidence |
|-------|-------|------------|
| Qwen3-Coder | 94% | HIGH |
| Gemma 4 | 99% | HIGH |
| huihui-ref | 91% | HIGH |

### MATH (100q) — RERUN with fixes
| Model | Score | Confidence | Notes |
|-------|-------|------------|-------|
| Qwen3-Coder | 78% | HIGH | +4pp from regex fix (1 question still truncated) |
| Gemma 4 | 82% | HIGH | +23pp from max_tokens + regex fixes |
| huihui-ref | 73% | HIGH | +22pp from fixes, 11 questions spiral past 32768 tokens |

### DROP (100q) — VALID, no bugs
| Model | Score | Confidence |
|-------|-------|------------|
| Qwen3-Coder | 80% | HIGH |
| Gemma 4 | 89% | HIGH |
| huihui-ref | 89% | HIGH |

### GPQA (100q) — VALID where tested
| Model | Score | Confidence | Notes |
|-------|-------|------------|-------|
| Qwen3-Coder | 42% | HIGH | |
| Gemma 4 | 70%* | LOW (10q probe) | Full 100q in progress |
| huihui-ref | 54% | HIGH | |

---

## Data Sources Cross-Check

During the audit, we verified scores using three independent data sources:

1. **LM Studio server logs** at `~/.lmstudio/server-logs/2026-04/` — full request/response records with token counts and raw content. Used to confirm the 4096 truncation bug and count affected questions.

2. **Conversation transcript** from the Claude Code session that ran the original benchmarks — contained partial per-question output captured from stdout. Used to cross-check timing data.

3. **bench2.py rerun** — produced fresh, fully-logged data. This is now the canonical source.

All three sources agree on the timeline and number of truncated questions.

---

## Lessons Learned

1. **Never trust a script without reading it.** The `max_tokens=4096` hardcode was visible in one line of bench.py but went undetected for multiple benchmark runs.

2. **Log everything, or you can't debug anything.** We only figured out the 4096 bug because LM Studio had its own server logs. If those didn't exist, the score differences would have been unexplainable.

3. **Statistical results hide systematic bugs.** Gemma 4 at 59% MATH looked "about right" — thinking models aren't great at math, right? Wrong. It was 82% once the bug was removed.

4. **Different evaluators produce different scores.** Our early 20q runs used simple-evals, the 100q runs used bench.py, and the current runs use bench2.py. They are NOT directly comparable because scoring methods differ. The 20q vs 100q comparison in earlier reports was misleading.

5. **Save raw responses.** bench2.py saves full response text in the JSONL, so if scoring logic changes in the future (e.g., a better LaTeX normalizer), we can re-grade existing runs without re-running the model.
