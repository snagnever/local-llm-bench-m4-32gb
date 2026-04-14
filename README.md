# local-llm-bench-m4

Benchmarks and findings for running small local LLMs on a **MacBook Air M4 32 GB** (fanless, 120 GB/s unified memory). Covers knowledge benchmarks, tool calling, inference engines (LM Studio vs MLX), REAP-pruned models, KV-cache quantization, and long-context behavior.

**Goal:** Find the best open-weight model that fits on this hardware — scored on both quality *and* speed.

**Current status (2026-04-14):** Multiple benchmarking passes complete across two workload classes:
- **Knowledge (MMLU, HumanEval, MATH, DROP, GPQA):** 3 LM Studio finalists at 100q each → see `results/FINAL_100Q_RESULTS.md`. Plus a 4th run on `deadbydawn101/gemma-4-21b-REAP-Tool-Calling-mlx-4bit` via `mlx_vlm` → `results/runs/mlx_*.jsonl`.
- **Tool calling (jdhodges 40 + Veerman 12):** 11 runs covering 10 models + 1 engine A/B (Gemma REAP GGUF vs MLX) → see `results/tool_calling_results.md`.

**Winner for daily use:** **Gemma 4 21B REAP** (0xSero REAP weights, GGUF Q4_K_M via LM Studio) — 96.2 % combined on tool calling and the fastest wall-clock in the set (8.3 min / 2.98 s mean latency). The same weights also run ~1.67× faster on the MLX engine for long-generation knowledge workloads.

---

## Directory Map

```
local-llm-bench-m4/
├── README.md                        ← you are here
├── LICENSE
├── .env.example                     ← template for LMSTUDIO_URL / optional API keys
│
├── scripts/                         ← benchmark harnesses and runners
│   ├── bench2.py                    ← CURRENT knowledge-bench harness (LM Studio)
│   ├── mlx_bench.py                 ← MLX knowledge-bench harness (mlx_vlm Python API)
│   ├── tool_call_bench.py           ← Tool-calling harness (LM Studio OR mlx_vlm.server via --base-url)
│   ├── tool_call_report.py          ← Aggregates tool-calling JSONL → tool_calling_results.md
│   ├── run_tool_call_overnight.sh   ← Overnight 10-model tool-calling runner
│   ├── run_mlx_full.sh              ← MLX knowledge-bench sweep (4 benches × 100q)
│   ├── lms.py                       ← LM Studio model loader / checker
│   ├── speed_probe.py               ← Fast 10q speed tests
│   ├── update_master_data.py        ← Aggregates runs/ → extracted/
│   ├── extract_final.py             ← Parses LM Studio server logs
│   ├── context_speed_bench.py       ← Context size × speed benchmark
│   ├── long_context_test.py         ← Needle-in-haystack long-context probe
│   ├── mlx_probe.py / probe.py      ← Quick probes for new models
│   ├── turboquant_test.py           ← KV-cache quantization (TurboQuant)
│   └── bench.py                     ← DEPRECATED (see results/AUDIT_REPORT.md)
│
├── results/                         ← benchmark results, data and analysis
│   ├── FINAL_100Q_RESULTS.md        ← knowledge-bench canonical results (3 LM Studio models × 5 benches)
│   ├── tool_calling_results.md      ← tool-calling canonical results (11 runs)
│   ├── AUDIT_REPORT.md              ← bug history (all fixed now)
│   ├── METHODOLOGY.md               ← how to run benchmarks correctly
│   ├── engine_comparison.md         ← LM Studio vs MLX head-to-head
│   ├── benchmark_analysis.ipynb     ← analysis notebook (pandas / matplotlib)
│   ├── benchmark_analysis_executed.ipynb
│   ├── pyproject.toml / uv.lock     ← notebook deps (reproduce with `cd results && uv sync`)
│   ├── charts/                      ← 16 PNG visualizations produced by the notebook
│   ├── runs/                        ← per-question JSONL logs (raw data)
│   ├── extracted/                   ← aggregated summaries (JSON / CSV)
│   ├── probes/                      ← small / ad-hoc probe results
│   ├── speed_probe/                 ← early 10q speed tests
│   ├── tool_calling/                ← tool-calling test YAMLs (jdhodges + Veerman)
│   └── third_party_comparisons/     ← cross-Mac data (M1 Max / M2 Pro / M3 Max) from local-llm-bench
│
├── notes/                           ← research notes (what informed the model shortlist)
│   ├── models/                      ← candidate model lists
│   ├── raw/                         ← web / Reddit research
│   └── twitter/                     ← X threads (0xSero, Karpathy)
│
├── docs/                            ← historical narrative documents
│   ├── FINAL_SYNTHESIS.md           ← pre-benchmark research synthesis (ARCHIVED)
│   ├── ACTION_PLAN.md               ← original plan (ARCHIVED, executed)
│   ├── WORKLOG_NIGHT.md             ← detailed inference-engine investigation timeline
│   └── NIGHT_FINDINGS.md            ← executive summary of night investigation
│
└── remote-setup/                    ← package for running the same harness on a Windows PC
    ├── README.md
    ├── requirements.txt
    └── scripts/
```

---

## Quick Start

### Requirements

- macOS with Apple Silicon (tested on M4 Air 32 GB)
- Python 3.11+
- [LM Studio](https://lmstudio.ai/) for the GGUF/llama.cpp path, **or** `mlx-vlm` / `mlx-lm` for the MLX path
- `uv` or `pip` for installing Python dependencies

```bash
# 1. Copy env template and fill in LMSTUDIO_URL if you want to hit a remote box
cp .env.example .env

# 2. (Optional) reproduce the analysis notebook environment
cd results && uv sync && cd ..
```

### Run a benchmark

```bash
# Pre-flight check (verifies model loaded, warmup, memory)
python3 scripts/lms.py check

# Run 100 questions on MATH against the currently loaded model
python3 scripts/bench2.py math --examples 100

# Run against a remote LM Studio instance (any reachable IP on your LAN)
LMSTUDIO_URL=http://<your-host>:1234/v1 python3 scripts/bench2.py math --examples 100

# Rerun only specific question numbers (e.g. ones that failed)
python3 scripts/bench2.py math --examples 100 --only 4,13,15,26

# MLX path (direct mlx_vlm, no server)
python3 scripts/mlx_bench.py math --examples 100 \
  --model deadbydawn101/gemma-4-21b-REAP-Tool-Calling-mlx-4bit
```

### View results

```bash
# Latest aggregate scores
cat results/extracted/master_summary.json

# Detailed per-question JSONL from a specific run
ls results/runs/

# Open the analysis notebook
jupyter notebook results/benchmark_analysis_executed.ipynb
```

---

## Key Findings

### Knowledge benchmarks (100q each, MacBook Air M4 32GB, LM Studio GGUF)

| Benchmark | Qwen3-Coder 30B-A3B | Gemma 4 26B-A4B | huihui-ref 35B-A3B |
|-----------|--------------------:|----------------:|-------------------:|
| MMLU | 67% | **84%** | 78% |
| HumanEval | 94% | **99%** | 91% |
| MATH | 78% | **82%** | 73% |
| DROP | 80% | **89%** | **89%** |
| GPQA | 42% | **64%** | 54% |
| **Time (total)** | **2.3h** | 24h | 15.5h |

- **Gemma 4 26B A4B** wins on quality across all 5 benchmarks (83.6% avg, 24h runtime)
- **Qwen3-Coder 30B** wins on speed — 10× faster — still 72.2% avg
- **huihui-ref 35B** middle ground, no single strength
- Full write-up with time/quality tradeoffs: `results/FINAL_100Q_RESULTS.md`

**Additional MLX engine run (2026-04-13):** `deadbydawn101/gemma-4-21b-REAP-Tool-Calling-mlx-4bit` via `mlx_vlm` — 77.3% avg across 4 benches (no GPQA). Same 0xSero REAP weights but MLX-quantized; the REAP pruning costs ~11pp vs base Gemma 4 26B on knowledge. Daily driver because of speed + tool-calling quality, not knowledge accuracy.

### Tool-calling benchmark (jdhodges 40 + Veerman 12, 11 runs 2026-04-14)

| # | Model | Engine | Size | **Combined** | jd mean lat | **wall-clock t/s** |
|---|-------|--------|------|--------------|-------------|--------------------|
| **1** | **Gemma 4 21B REAP** (0xSero/barozp) | LM Studio GGUF Q4 | 12.9 GB | **96.2%** | **2.98s** | 22.8 |
| 2= | Gemma 4 26B A4B | LM Studio GGUF Q4 | 16.8 GB | 92.3% | 5.60s | 23.8 |
| 2= | Gemma 4 21B REAP (0xSero/deadbydawn101) | **mlx_vlm.server MLX 4-bit** | 12.0 GB | 92.3% | 5.59s | 9.5 wall / **35.3 decode** |
| 4 | **Qwen3.5-4B** | LM Studio GGUF Q4 | **2.7 GB** | **90.4%** | 10.35s | 20.0 |
| 10 | DeepHermes-ToolCalling-Atropos | LM Studio GGUF Q4 | 4.6 GB | 73.1% | 2.99s | 16.6 |
| 11 | **Qwen3-Coder 30B A3B** | LM Studio GGUF Q4 | 17.3 GB | **0%** (broken) | — | — |

- **Gemma 4 21B REAP dominates tool calling** — perfect 40/40 jdhodges, best accuracy AND fastest wall-clock.
- **Knowledge-bench ranking has zero correlation with tool-calling ranking.** Gemma REAP was worst on knowledge (77.3%), best on tool calling (96.2%). Qwen3-Coder was a solid generalist on knowledge, completely broken on tool calling (jinja template bug in LM Studio).
- **Engine rule-of-thumb confirmed:** llama.cpp for short outputs (tool calls, prefill-dominated); MLX for long generations (chat, decode-dominated). Same Gemma REAP weights: llama.cpp 1.9× faster end-to-end on tool calls, MLX 1.67× faster end-to-end on knowledge.

Full write-up: `results/tool_calling_results.md`

---

## Project History

1. **April 7-8, 2026** — Research phase. Web/Reddit/Twitter research (`notes/raw/`), candidate models identified (`notes/models/`), initial action plan (`docs/ACTION_PLAN.md`).
2. **April 8-9, 2026** — Initial benchmark runs using `bench.py`. Several bugs discovered (`results/AUDIT_REPORT.md`).
3. **April 10-11, 2026** — Rewrote as `bench2.py` with full logging. MATH bugs fixed.
4. **April 11-12, 2026** — Full 100q reruns for all 3 models with corrected harness. Canonical results in `results/FINAL_100Q_RESULTS.md`.
5. **April 12-13, 2026** — Inference engine investigation (LM Studio vs `mlx_vlm` vs `mlx_vlm.server` vs `mlx-openai-server`), REAP model testing, KV quantization (TurboQuant), long-context performance, kernel-panic thermal investigation. See `docs/NIGHT_FINDINGS.md` + `docs/WORKLOG_NIGHT.md`. MLX Gemma REAP 100q knowledge bench (4 benches, ~9h) completed.
6. **April 14, 2026** — Tool-calling benchmark suite: built `tool_call_bench.py` harness (reuses `bench2.py` logging conventions, adds thermal gate + auto-skip), reconstructed jdhodges 40-case suite from blog + harness code, integrated Mike Veerman's 12-prompt suite. 10 LM Studio models run overnight (3h13m), plus engine A/B run (`mlx_vlm.server`) for Gemma REAP. Canonical results in `results/tool_calling_results.md`.

---

## Environment Variables

- `LMSTUDIO_URL` — override LM Studio API URL. Default: `http://127.0.0.1:1234/v1`. Use this to run benchmarks against a remote LM Studio instance (e.g. Windows PC with RTX 3080 Ti).

---

## Related Work

- **0xSero's REAP models** — `notes/twitter/0xSero_tweets.json`. 0xSero publishes pruned versions of frontier models (Gemma 4, Qwen 3.5, GLM 5, etc.) at reduced VRAM. Tested extensively here; `gemma-4-21b-REAP` is the daily driver.
- **Karpathy's autoresearch** — `notes/twitter/karpathy_autoresearch.json`. Autonomous agent loop that iterates on training code to find hyperparameter improvements.
- **[LiveBench](https://github.com/LiveBench/LiveBench)** — upstream LiveBench benchmark suite. Not used here (public data only goes to Nov 2024); referenced for methodology.
- **[OpenAI simple-evals](https://github.com/openai/simple-evals)** — OpenAI's reference eval harness (deprecated July 2025). Studied for methodology but not used for the actual runs.
- **[famstack-dev/local-llm-bench](https://github.com/famstack-dev/local-llm-bench)** — cross-Mac performance data (M1 Max / M2 Pro / M3 Max). A subset of the data this repo relied on lives under `results/third_party_comparisons/`.

## License

MIT — see [`LICENSE`](./LICENSE).
