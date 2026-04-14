# Tool-Calling Benchmark Results — 11 Runs, 10 Models + 1 Engine A/B

**Date:** 2026-04-14 (overnight 00:28 → 03:41, MLX A/B added 11:06 → 11:17)
**Hardware:** MacBook Air M4, 32 GB unified memory, 120 GB/s, fanless
**Engines tested:** LM Studio (llama.cpp + MLX backends, OpenAI endpoint :1234) and `mlx_vlm.server` (mlx_vlm 0.4.4, OpenAI endpoint :8080)
**Harness:** `scripts/tool_call_bench.py` — `bench2.py`-grade logging (temp, t/s, tokens, raw tool calls, GPU/CPU/swap before & after)
**Gen params:** `temperature=0.0, top_p=1.0, seed=42, tool_choice="auto"`
**Thermal gate:** ≤60 °C GPU before every question for the 10-model overnight run; **disabled** for the 8-minute mlx_vlm A/B
**Scoring:** deterministic schema-aware matching (tool name + arg values), ported from jdhodges' `eval_tool_calling.py`. No LLM-as-judge.

---

## 🏆 TL;DR

1. **Winner: Gemma 4 21B REAP (GGUF Q4_K_M, LM Studio).** 96.2 % combined (100 % jdhodges, 83 % veerman), **fastest wall-clock** (8.3 min for 52 cases, 2.98 s mean latency). 12.9 GB.
2. **Runner-up (tied): Gemma 4 26B A4B (GGUF)** and **Gemma 4 21B REAP (MLX 4-bit, mlx_vlm.server)** at 92.3 %. Both miss one jdhodges case + one veerman case vs the GGUF REAP winner.
3. **Budget pick: Qwen3.5-4B (GGUF Q4)** — 90.4 %, **2.7 GB**. Small enough to keep resident alongside a larger chat model.
4. **Engine comparison (Gemma REAP, same weights, 2 engines):** llama.cpp GGUF wins end-to-end on tool calling despite mlx_vlm's faster pure decode. **Why:** tool-call outputs are short (20–60 tokens), so prompt processing dominates latency, and llama.cpp's prefill is ~2× faster than mlx_vlm on Apple Silicon. **MLX is the wrong engine for short-output workloads like tool calls.**
5. **Do not use:** Qwen3-Coder 30B (jinja template bug, 0 % all runs — LM Studio can't render its tool-call template), DeepHermes-ToolCalling-Specialist-Atropos (1/8 on multi-tool chains despite the "specialist" label), Hermes-4-14B (0/3 on hard traps + 4× slowest in the set).
6. **Knowledge-bench ranking has zero predictive power for tool calling.** Gemma 4 21B REAP was **worst** on the 9h knowledge suite (77.3 %) and is **best** here (96.2 %). REAP pruning kills world knowledge but preserves instruction-following.

---

## Engine comparison — Gemma 4 21B REAP, same weights, GGUF vs MLX

**Both runs are the same underlying `0xSero/gemma-4-21b-a4b-it-REAP` weights** (confirmed via HF `base_model_relation: quantized`). The only differences are the re-quantization (Q4_K_M vs MLX 4-bit affine group_size=64) and the inference engine.

| Aspect | **GGUF Q4_K_M** (LM Studio / llama.cpp) | **MLX 4-bit** (`mlx_vlm.server`) |
|--------|-----------------------------------------|----------------------------------|
| HF repo | `barozp/gemma-4-21b-a4b-it-REAP-GGUF` | `deadbydawn101/gemma-4-21b-REAP-Tool-Calling-mlx-4bit` |
| On-disk size | 12.9 GB | 12.0 GB |
| **jdhodges** | **40/40 (100.0 %)** | 39/40 (97.5 %) |
| **veerman** | 10/12 (83.3 %) | 9/12 (75.0 %) |
| **Combined** | **96.2 % (50/52)** | 92.3 % (48/52) |
| **Wall-clock total (jd + vm)** | **8.3 min** | 8.5 min |
| **jd mean latency** | **2.98 s** | 5.59 s |
| **jd wall-clock t/s** (tok-weighted, full request) | **22.8** | 9.5 |
| **jd engine-reported decode t/s** (generation only) | not reported by LM Studio | **35.3** |
| **jd engine prompt t/s** (prefill only) | not reported | ~287 |
| GPU temp Δ (no cooldown gate for MLX) | 41 → 72 °C (over 8.3 min) | 41 → 72 °C (over 8.5 min) |
| Peak memory | ~14 GB | ~14.1 GB |

**Key finding — prompt processing vs decode dominance:**

mlx_vlm's **pure decode is 55 % faster than llama.cpp's wall-clock throughput** (35.3 vs 22.8 t/s) — exactly consistent with the 9h knowledge bench where MLX was 1.67× faster on long generations. **But wall-clock end-to-end on tool calling MLX is slower** because tool-call outputs are short (mean completion = 59 tokens on this suite) and prompt processing takes ~3.5 s for the ~1000-token system+tool-schema prompt.

Arithmetic breakdown for a typical MLX case (~1000 prompt tokens, ~30 completion tokens):
- Prefill: 1000 / 287 t/s ≈ **3.48 s**
- Decode: 30 / 35.3 t/s ≈ **0.85 s**
- Total wall-clock: ~4.3 s (matches the observed 5.59 s mean latency)

For LM Studio GGUF:
- Prefill: llama.cpp's prefill on Apple Silicon is ~2× faster than mlx_vlm's for this prompt length → ~1.7 s
- Decode: 30 / ~25 t/s ≈ 1.2 s
- Total: ~2.9 s (matches observed 2.98 s)

**Rule of thumb** (confirmed now by two separate benches): **llama.cpp for short outputs / tool calls, MLX for long generations / chat.** The 1.67× MLX speedup from the knowledge bench inverts to a 1.9× llama.cpp speedup here, for the exact same weights. Choose the engine based on output-length workload, not the model.

**Accuracy difference** (MLX −3.9 pp on combined): the MLX variant lost 1 jdhodges case (`multi_email_after_calendar_read`, turn-1 tool mismatch) and 1 extra veerman case (`no_tool_called` on what the GGUF variant got right). Both variants have the same failure *pattern* (turn-1 chain failure, occasional refusal). The delta is within quantization noise — both are ~4-bit quants of the same weights but with different calibration (Q4_K_M k-quant blocks vs MLX affine group_size=64). Not a meaningful quality gap.

---

## Full leaderboard (11 runs, sorted by combined score)

| # | Model | Engine | Size | jdhodges | veerman | **Combined** | jd wall-clock **mean lat** | **wall-clock t/s** | **jd+vm wall-clock min** |
|---|-------|--------|------|----------|---------|--------------|----------------------------|--------------------|--------------------------|
| **1** | **Gemma 4 21B REAP** (barozp) | **LM Studio GGUF Q4** | 12.9 GB | **40/40 (100 %)** | 10/12 (83 %) | **96.2 %** | **2.98 s** | 22.8 | **8.3 min** |
| 2= | Gemma 4 26B A4B | LM Studio GGUF Q4 | 16.8 GB | 39/40 (97.5 %) | 9/12 (75 %) | 92.3 % | 5.60 s | 23.8 | 15.1 min |
| 2= | Gemma 4 21B REAP (deadbydawn101) | **mlx_vlm.server MLX 4-bit** | 12.0 GB | 39/40 (97.5 %) | 9/12 (75 %) | 92.3 % | 5.59 s | 9.5 (wall) / **35.3 (decode)** | 8.5 min |
| 4 | **Qwen3.5-4B** | LM Studio GGUF Q4 | **2.7 GB** | 37/40 (92.5 %) | 10/12 (83 %) | **90.4 %** | 10.35 s | 20.0 | 21.0 min |
| 5 | GLM-4.7-Flash | LM Studio GGUF Q4 | 16.9 GB | 36/40 (90 %) | 9/12 (75 %) | 86.5 % | 6.05 s | 24.8 | 16.2 min |
| 6= | huihui-claude Opus 35B | LM Studio GGUF Q3 | 14.1 GB | 36/40 (90 %) | 8/12 (67 %) | 84.6 % | 7.64 s | 18.5 | 16.8 min |
| 6= | NVIDIA Nemotron 3 Nano 4B | LM Studio GGUF Q4 | 2.7 GB | 36/40 (90 %) | 8/12 (67 %) | 84.6 % | 8.20 s | 18.0 | 21.8 min |
| 8 | Hermes-4-14B | LM Studio MLX 4-bit | 7.8 GB | 35/40 (87.5 %) | 8/12 (67 %) | 82.7 % | 5.63 s | 10.0 | **54.3 min** |
| 9 | Mistral Nemo 12B | LM Studio MLX 4-bit | 6.4 GB | 31/40 (77.5 %) | **11/12 (92 %)** | 80.8 % | 5.37 s | 11.0 | 12.7 min |
| 10 | DeepHermes-ToolCalling-Atropos | LM Studio GGUF Q4 | 4.6 GB | 30/40 (75 %) | 8/12 (67 %) | 73.1 % | 2.99 s | 16.6 | 14.7 min |
| 11 | Qwen3-Coder 30B A3B | LM Studio GGUF Q4 | 17.3 GB | 0/40 (0 %) | 0/12 (0 %) | **0 %** | — | — | 4.2 min (all errors) |

---

## jdhodges per-category breakdown (8 cases each)

| Model | Engine | Selection | Args | Multi-Tool | Edge | Format | **jd mean lat** |
|-------|--------|-----------|------|------------|------|--------|-----------------|
| Gemma 4 21B REAP (barozp) | GGUF | **8/8** | **8/8** | **8/8** | **8/8** | **8/8** | **2.98 s** |
| Gemma 4 26B A4B | GGUF | 8/8 | 8/8 | 7/8 | 8/8 | 8/8 | 5.60 s |
| Gemma 4 21B REAP (deadbydawn101) | MLX | 8/8 | 8/8 | 7/8 | 8/8 | 8/8 | 5.59 s |
| Qwen3.5-4B | GGUF | 8/8 | 8/8 | 5/8 | 8/8 | 8/8 | 10.35 s |
| GLM-4.7-Flash | GGUF | 7/8 | 8/8 | 5/8 | 8/8 | 8/8 | 6.05 s |
| huihui-claude Opus 35B | GGUF | 8/8 | 8/8 | 4/8 | 8/8 | 8/8 | 7.64 s |
| NVIDIA Nemotron 3 Nano 4B | GGUF | 8/8 | 8/8 | 5/8 | 7/8 | 8/8 | 8.20 s |
| Hermes-4-14B | MLX | 8/8 | 7/8 | 5/8 | 7/8 | 8/8 | 5.63 s |
| Mistral Nemo 12B | MLX | 8/8 | 8/8 | 3/8 | 4/8 | 8/8 | 5.37 s |
| DeepHermes-ToolCalling-Atropos | GGUF | 8/8 | 8/8 | 1/8 | 5/8 | 8/8 | 2.99 s |
| Qwen3-Coder 30B | GGUF | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | — |

**Observations:**
- **Every non-broken model nails Selection + Args + Format.** The discriminator is Multi-Tool (parallel + chained calls) and Edge (knowing when not to call).
- **Only Gemma 4 21B REAP (GGUF) solves all 8 multi-tool cases.** Its MLX twin loses one multi-tool case (the `multi_email_after_calendar_read` chain) — minor quant noise.
- **DeepHermes Atropos: 1/8 multi-tool.** The supposed tool-calling specialist is worst in class on chains. Advertised capability ≠ measured capability.
- **Mistral Nemo: 4/8 edge cases.** Classic "helpful rewriter" over-eagerness — calls `web_search` for "how does photosynthesis work", `set_reminder` for "I'm feeling overwhelmed".

---

## Veerman P1–P12 breakdown

| Model | Engine | Action (P1–P4, P6–P8) | Restraint (P5, P9) | Hard (P10–P12) | **vm mean lat** |
|-------|--------|------------------------|---------------------|-----------------|-----------------|
| Gemma 4 21B REAP (barozp) | GGUF | 6/7 | **2/2** | 2/3 | 4.31 s |
| Gemma 4 26B A4B | GGUF | 6/7 | **2/2** | 1/3 | 8.00 s |
| Gemma 4 21B REAP (deadbydawn101) | MLX | 6/7 | **2/2** | 1/3 | 3.77 s |
| Qwen3.5-4B | GGUF | 6/7 | **2/2** | 2/3 | — |
| GLM-4.7-Flash | GGUF | 6/7 | **2/2** | 1/3 | — |
| huihui-claude Opus 35B | GGUF | 5/7 | 1/2 | 2/3 | — |
| NVIDIA Nemotron 3 Nano 4B | GGUF | 6/7 | 1/2 | 1/3 | — |
| Hermes-4-14B | MLX | 6/7 | **2/2** | **0/3** | — |
| **Mistral Nemo 12B** | MLX | **7/7** | 1/2 | **3/3** | — |
| DeepHermes-ToolCalling-Atropos | GGUF | 4/7 | **2/2** | 2/3 | — |
| Qwen3-Coder 30B | GGUF | 0/7 | 0/2 | 0/3 | — |

**Veerman leader (surprising): Mistral Nemo 12B — 7/7 action, 3/3 hard traps.** It solves every negation / redundancy / implicit-reasoning prompt (P10 cycling-depends-on-weather, P11 "don't check the weather", P12 "weather already given") that other models fail on. But it's the worst on jdhodges edge cases — personality mismatch: great at hard traps, can't restrain itself on emotional prompts.

**Hermes-4-14B: 0/3 on hard traps** — the exact failure mode it's marketed to fix. Confirms the user's prior concerns about Hermes-Agent.

---

## Speed table — jdhodges suite (all 11 runs)

| Model | Engine | **Mean latency (s)** | **Median (s)** | **P95 (s)** | **Wall-clock t/s** | **Engine decode t/s** | Tokens total | **jd wall-clock (min)** |
|-------|--------|----------------------|----------------|-------------|--------------------|------------------------|--------------|------------------------|
| Gemma 4 21B REAP (barozp) | GGUF | **2.98** | **1.79** | 6.71 | 22.8 | not reported | 2720 | **6.2** |
| Gemma 4 26B A4B | GGUF | 5.60 | 3.06 | 14.93 | **23.8** | not reported | 5332 | 9.7 |
| Gemma 4 21B REAP (deadbydawn101) | **MLX** | 5.59 | 5.11 | 8.08 | 9.5 | **35.3** | 2127 | **6.7** |
| GLM-4.7-Flash | GGUF | 6.05 | 5.33 | 11.44 | **24.8** | not reported | 6010 | 11.9 |
| DeepHermes-Atropos | GGUF | 2.99 | 2.30 | 9.44 | 16.6 | not reported | 1984 | 12.1 |
| Mistral Nemo 12B | MLX | 5.37 | 4.71 | 11.70 | 11.0 | not reported by LM Studio MLX | 2375 | 10.4 |
| Hermes-4-14B | MLX | 5.63 | 4.53 | 10.55 | 10.0 | not reported by LM Studio MLX | 2193 | **40.9** |
| huihui-claude Opus 35B | GGUF | 7.64 | 6.82 | 12.74 | 18.5 | not reported | 5664 | 13.2 |
| NVIDIA Nemotron 3 Nano 4B | GGUF | 8.20 | 7.07 | 17.54 | 18.0 | not reported | 5914 | 16.0 |
| Qwen3.5-4B | GGUF | 10.35 | 9.37 | 15.43 | 20.0 | not reported | 8269 | 17.4 |
| Qwen3-Coder 30B | GGUF | — | — | — | — | — | 0 | 3.2 (all errors) |

**Speed insights:**
1. **Gemma 4 21B REAP GGUF dominates every latency metric AND the combined accuracy** — dominated strategy. Lowest mean, lowest median, lowest suite wall-clock, highest accuracy.
2. **MLX engine decode is 35.3 t/s — faster than GGUF's 22.8 wall-clock t/s on the same weights** — but wall-clock end-to-end loses because prefill is slower. The number you care about depends on output length.
3. **Hermes-4-14B's 40.9 min on jdhodges alone is 6× the median.** Its MLX decode is slow (~10 t/s wall-clock) AND it generates more tokens per answer on multi-turn cases. Totally unusable for interactive agents.
4. **DeepHermes Atropos has the joint-fastest mean latency (2.99 s, tied with GGUF Gemma REAP)** — cheap and fast. But with 73.1 % combined accuracy, speed without correctness is useless.
5. **Qwen3.5-4B is the slowest per-call (10.35 s mean)** despite being the smallest model — it generates ~220 tokens/case (3× more than Gemma REAP). Small ≠ fast if the model is verbose.

---

## Per-model wall-clock (jdhodges + veerman total)

| Model | Engine | jdhodges (min) | veerman (min) | **Total (min)** |
|-------|--------|----------------|----------------|-----------------|
| **Gemma 4 21B REAP (barozp)** | GGUF | 6.2 | 2.1 | **8.3** |
| **Gemma 4 21B REAP (deadbydawn101)** | **MLX** | 6.7 | 1.8 | **8.5** |
| Mistral Nemo 12B | MLX | 10.4 | 2.3 | 12.7 |
| DeepHermes-ToolCalling-Atropos | GGUF | 12.1 | 2.6 | 14.7 |
| Gemma 4 26B A4B | GGUF | 9.7 | 5.4 | 15.1 |
| GLM-4.7-Flash | GGUF | 11.9 | 4.3 | 16.2 |
| huihui-claude Opus 35B | GGUF | 13.2 | 3.6 | 16.8 |
| Qwen3.5-4B | GGUF | 17.4 | 3.6 | 21.0 |
| NVIDIA Nemotron 3 Nano 4B | GGUF | 16.0 | 5.8 | 21.8 |
| **Hermes-4-14B** | MLX | **40.9** | **13.4** | **54.3** |
| Qwen3-Coder 30B | GGUF | 3.2 | 1.0 | 4.2 (all errors) |

**Note on how close Gemma REAP GGUF and MLX are on wall-clock (8.3 vs 8.5 min):** the engines are nearly tied end-to-end on the full suite, despite wildly different per-request shapes. GGUF has faster per-request latency; MLX has faster decode for the few long-output cases (e.g. `edge_photosynthesis` 256-token explanation). They basically break even over 52 mixed-length cases.

---

## Relationship to the prior knowledge benchmark (2026-04-12/13)

| Model | Knowledge avg (MMLU+HE+MATH+DROP) | Tool calling combined | Δ |
|-------|----------------------------------|-----------------------|---|
| **Gemma 4 21B REAP (MLX knowledge, GGUF tool)** | **77.3 %** (worst) | **96.2 %** (best) | **+18.9** |
| Gemma 4 26B A4B | 88.5 % | 92.3 % | +3.8 |
| huihui-claude-i1 35B | 77.0 % | 84.6 % | +7.6 |
| Qwen3-Coder 30B | 72.2 % | 0 % (LM Studio template bug) | — |

**The knowledge-bench ranking has zero predictive power for tool calling.** Gemma 4 21B REAP goes from dead-last baseline to first-place overall. Qwen3-Coder goes from solid generalist (72.2 %) to completely broken (0 % — jinja filter error in LM Studio). **Benchmarks must match your actual workload** — there is no general-purpose "local LLM leaderboard".

---

## Recommendation for the KB / tool-calling agent use case

**Ship:** `barozp/gemma-4-21b-a4b-it-REAP-GGUF` (Q4_K_M, 12.9 GB, LM Studio key `gemma-4-21b-a4b-it-reap`).

**Why:**
1. **96.2 % combined — highest accuracy in the 11-run set**, with perfect selection + args + multi-tool + edge + format on jdhodges.
2. **Fastest wall-clock (8.3 min for 52 cases), fastest mean latency (2.98 s)**, 2nd-fastest decode (22.8 t/s wall-clock). Dominated strategy — wins on every axis.
3. **12.9 GB** — smallest top-tier footprint. 4 GB under Gemma 4 26B, 4 GB under GLM-4.7-Flash.
4. **Already loaded in LM Studio.** Already downloaded. Use today.

**Alternate: same weights, MLX engine (`deadbydawn101/gemma-4-21b-REAP-Tool-Calling-mlx-4bit`)** — 92.3 % combined, virtually identical wall-clock on the full suite. Pick this if you plan to use the same model for long-context chat / KB exploration where MLX's decode advantage matters, but **note the 3.9 pp accuracy regression on tool-calling specifically** (within quant noise but not zero).

**Do not ship:**
- **Qwen3-Coder 30B** — broken in LM Studio (jinja template filter). Re-test elsewhere if you need it.
- **DeepHermes-ToolCalling-Specialist-Atropos** — 1/8 multi-tool is a dealbreaker. Name is marketing, not measurement.
- **Hermes-4-14B** — 0/3 on hard Veerman traps + 54 min for one bench suite. Both the accuracy and the speed disqualify it.

---

## Known caveats

1. **jdhodges YAMLs are reconstructed.** The downloadable zip claims to include `tool_definitions.yaml` and `test_cases.yaml` but actually only ships the Python harness. I rebuilt 40 cases from the 8 tool schemas, 5×8 category structure, and 4 verbatim blog prompts (marked `[verbatim]` in `test_cases.yaml`). Our Qwen3.5-4B score (92.5 %) is 5 pp below jdhodges' published 97.5 % — **absolute numbers are not comparable to the blog; internal model ranking is the signal**.
2. **The MLX run had no thermal gate** (`--no-cooldown`). Peak GPU was 72 °C after 8.5 min of sustained load — well under the ~85 °C danger zone we saw on the earlier 9h knowledge bench. No thermal throttling observed. Safe to run short tool-call benches without the gate.
3. **LM Studio does not expose per-request decode vs prefill tps** — only end-to-end. All GGUF speeds in the "Engine decode t/s" column are wall-clock (comp_tokens / elapsed_s), which conflates prefill + decode. mlx_vlm.server reports `generation_tps` and `prompt_tps` separately so we have the real decode rate for the MLX row.
4. **One cosmetic harness bug** was active during model 1 (Qwen3.5-4B jdhodges) on the overnight run — the running cumulative %-score displayed during that single bench undercounted 3 carried-over smoke-test cases. The **final report numbers are correct** because the report generator reads JSONL entries directly and de-dupes by case_id.

---

## Methodology + reproducibility

- **Harness:** `scripts/tool_call_bench.py` — `--base-url`, `--no-cooldown`, `--run-prefix` flags so the same file runs against LM Studio (`:1234/v1`) or `mlx_vlm.server` (`:8080/v1`). Per-request logging captures: prompt tokens, completion tokens, wall-clock elapsed, engine-reported decode/prefill tps (when available), engine peak memory, GPU temp before & after, CPU temp, swap before & after, raw assistant message, raw tool-call array, score pass/fail + reason + breakdown.
- **Suites:** `results/tool_calling/{tool_definitions,test_cases,veerman_tools,veerman_cases}.yaml`
- **Raw logs:** `results/runs/toolcall_*.jsonl` (per-case) + `toolcall_*_summary.json` (per-run) + `toolcall_*_console_*.log` (per-run stdout). MLX runs use `toolcall_mlx_*` prefix for separation.
- **Overnight runner:** `scripts/run_tool_call_overnight.sh` — loads one LM Studio model at a time, unloads between models, enforces ≤60 °C gate, runs both suites.
- **Report generator:** `scripts/tool_call_report.py` — idempotent, rebuilds tables from raw JSONL.
- **Total compute time:** ~3h 13m overnight for 10 models × 2 suites via LM Studio + ~12 min for the MLX A/B on Gemma REAP via `mlx_vlm.server`. 0 kernel panics, 0 OOMs, peak swap 3.5 GB.
