# Reference Scores — Market & Open-Weight Models

Reference numbers for comparing local model results against frontier APIs and well-known open-weight models on the same 5 knowledge benchmarks the harness uses: **MMLU, HumanEval, MATH, DROP, GPQA**.

## How to read this table — comparability caveats

Our local harness runs **n=100, seed=42, temp=0**. Published reference scores below are almost always **full benchmark, often with few-shot prompting** (5-shot MMLU is standard, GPQA varies). That introduces two sources of difference vs. our local numbers:

1. **Sample noise**: ±5pp two-sigma at n=100 for binomial scoring. A model that scores 80% on the full set could easily land 75–85% on a 100-question seeded slice.
2. **Prompting / extraction**: providers report their own scoring, often with CoT or majority-vote at higher temperatures. Our harness uses temp=0 and deterministic letter/boxed extraction.

**Use these numbers for ranking, not for absolute deltas.** A 10pp gap is real; a 3pp gap is noise.

The Claude 4 Opus GPQA ≈ 85% number already cited in `FINAL_100Q_RESULTS.md` is the canonical kind of comparison to make: *"Gemma 4 26B-A4B at 64% sits between Claude 3.5 Sonnet (60%) and GPT-4o (54%) — a 16GB local model in API frontier territory."*

## Confidence legend

- **✓** — published in a model card, paper, or first-party announcement; cross-checked
- **~** — widely cited but I want it double-checked against the source before publishing
- **TBD** — not yet collected; needs lookup before comparison
- **—** — not reported by the provider on this benchmark (DROP especially)

---

## Frontier (closed-API)

| Model | MMLU | HumanEval | MATH | DROP (F1) | GPQA Diamond |
|---|---|---|---|---|---|
| **OpenAI** | | | | | |
| GPT-3.5 Turbo | 70.0 ✓ | 48.1 ✓ | 34.1 ✓ | 64.1 ✓ | 28.1 ~ |
| GPT-4 (original) | 86.4 ✓ | 67.0 ✓ | 52.9 ✓ | 80.9 ✓ | 39.5 ✓ |
| GPT-4 Turbo | 86.5 ✓ | ~85 ~ | 72.6 ~ | 86.0 ~ | 48.0 ~ |
| GPT-4o | 88.7 ✓ | 90.2 ✓ | 76.6 ✓ | 83.4 ~ | 53.6 ✓ |
| o1-preview | ~90 ~ | 92.4 ✓ | 85.5 ✓ | — | 73.3 ✓ |
| o1 | 92.3 ✓ | — | 94.8 ✓ | — | 78.0 ✓ |
| o3 / o3-mini | TBD | TBD | TBD | — | TBD |
| **Anthropic** | | | | | |
| Claude 3 Haiku | 75.2 ✓ | 75.9 ✓ | 38.9 ✓ | 78.4 ✓ | 33.3 ✓ |
| Claude 3 Sonnet | 79.0 ✓ | 73.0 ✓ | 40.5 ✓ | 78.9 ✓ | 40.4 ✓ |
| Claude 3 Opus | 86.8 ✓ | 84.9 ✓ | 60.1 ✓ | 83.1 ✓ | 50.4 ✓ |
| Claude 3.5 Sonnet | 88.7 ✓ | 92.0 ✓ | 71.1 ✓ | 87.1 ✓ | 59.4 ✓ |
| Claude 3.5 Haiku | ~77 ~ | ~88 ~ | ~69 ~ | ~83 ~ | ~42 ~ |
| Claude 4 Sonnet | TBD | ~93 ~ | TBD | — | ~75 ~ |
| Claude 4 Opus | TBD | TBD | TBD | — | **~85** ✓ (cited inline in FINAL_100Q_RESULTS) |
| Claude 4.5 / 4.6 / 4.7 Opus | TBD | TBD | TBD | — | TBD |
| **Google** | | | | | |
| Gemini 1.5 Pro | 81.9 ✓ | 84.1 ✓ | 67.7 ✓ | 78.9 ✓ | 46.2 ✓ |
| Gemini 1.5 Flash | 78.9 ✓ | 74.3 ✓ | 54.9 ✓ | — | 39.5 ~ |
| Gemini 2.0 Flash | ~77 ~ | ~89 ~ | ~83 ~ | — | ~62 ~ |
| Gemini 2.0 Flash Thinking | TBD | TBD | TBD | — | 73.3 ~ |
| Gemini 2.5 Pro | ~88 ~ | ~86 ~ | ~86 ~ | — | ~84 ~ |

## Open-weight (would-be local if we had the hardware)

| Model | MMLU | HumanEval | MATH | DROP | GPQA Diamond |
|---|---|---|---|---|---|
| Llama 3.1 8B Instruct | 69.4 ✓ | 72.6 ✓ | 51.9 ✓ | 59.5 ✓ | 30.4 ✓ |
| Llama 3.1 70B Instruct | 86.0 ✓ | 80.5 ✓ | 68.0 ✓ | 79.6 ✓ | 46.7 ✓ |
| Llama 3.1 405B Instruct | 88.6 ✓ | 89.0 ✓ | 73.8 ✓ | 84.8 ✓ | 51.1 ✓ |
| Llama 3.3 70B Instruct | ~86 ~ | ~88 ~ | ~77 ~ | — | ~50 ~ |
| Mistral Large 2 (123B) | 84.0 ✓ | 92.0 ✓ | ~70 ~ | — | ~48 ~ |
| Mixtral 8x22B | 77.8 ✓ | 76.2 ✓ | 41.8 ✓ | — | ~35 ~ |
| DeepSeek V3 | ~88 ~ | 89.0 ~ | 90.2 ✓ (MATH-500) | — | 59.1 ~ |
| DeepSeek R1 | ~90 ~ | TBD | 97.3 ✓ (MATH-500) | — | 71.5 ~ |
| Qwen 2.5 72B Instruct | 85.3 ✓ | 86.6 ✓ | 83.1 ✓ | — | 49.0 ~ |
| Qwen 2.5 Coder 32B | ~75 ~ | 92.7 ✓ | ~57 ~ | — | TBD |

## Locally measured — for direct comparison

From [FINAL_100Q_RESULTS.md](FINAL_100Q_RESULTS.md) (upstream M4 Air, n=100, seed=42, temp=0):

| Model | MMLU | HumanEval | MATH | DROP | GPQA |
|---|---|---|---|---|---|
| Qwen3-Coder 30B-A3B (Q4_K_M, 17GB) | 67 | 94 | 78 | 80 | 42 |
| Gemma 4 26B-A4B (Q4_K_M, 16GB) | **84** | **99** | **82** | 89 | **64** |
| huihui-claude-i1 35B (Q3_K_S, 14GB) | 78 | 91 | 73 | 89 | 54 |

From [benchmarks/runs/](../benchmarks/runs/) (this rig, M4 Max 128GB — Phase 1 in progress):

| Model | MMLU | HumanEval | MATH | DROP | GPQA |
|---|---|---|---|---|---|
| qwen/qwen3-coder-next (6-bit MLX, 65GB) | 76 | 89 | 84 | 83 | 37 |
| qwen3.6-27b (6-bit MLX) | 88 | 93 | 88 | 90 | 70 |
| qwen3.6-35b-a3b (6-bit MLX) | 83 | 87 | 89 | 89 | 65 |
| gemma-4-26b-a4b-it-mlx (4-bit MLX) | 78 | 98 | 80 | 79 | 47 |
| **deepseek-v4-flash-2bit-dq** (2-bit DQ MLX, ~96GB) | **44** | **48** | **47** | **71** | **24** |

Full measured set (all Phase 1/2 models × all benches, incl. LiveCodeBench / tool-calling /
Terminal-Bench) is plotted in [`results/charts/chart_m4max_phase1_scores.png`](charts/chart_m4max_phase1_scores.png)
— blank cells = not yet measured. DeepSeek-V4-Flash's low scores are the **2-bit DQ quality
floor** (see [`M4_MAX_128GB_NOTES.md`](M4_MAX_128GB_NOTES.md) Phase 3 #10, Addendum 2), not a
runtime issue; its Metal-OOM blocker is fixed and filed upstream (ml-explore/mlx-lm#1332,
Blaizzy/mlx-lm#25).

---

## Where Gemma 4 26B-A4B (current upstream knowledge winner) actually sits

Using the locally measured Gemma 4 26B-A4B numbers vs. published reference scores. Adjust ±5pp for the n=100 sample noise.

| Benchmark | Gemma 4 26B-A4B (local, 16GB) | Closest API peer | Frontier |
|---|---|---|---|
| MMLU | 84 | Mistral Large 2 (84), Llama 3.1 70B (86), Claude 3 Opus (87) | o1 (92) |
| HumanEval | **99** | *exceeds all standard refs* — likely contamination or sample luck | o1 (~92), Claude 3.5 Sonnet (92) |
| MATH | 82 | Qwen 2.5 72B (83), Gemini 2.0 Flash (~83) | o1 (95), DeepSeek R1 MATH-500 (97) |
| DROP | 89 | Claude 3.5 Sonnet (87), Llama 3.1 405B (85) | — |
| GPQA | 64 | between Claude 3.5 Sonnet (59) and GPT-4o (54), well below o1 (78) and Claude 4 Opus (~85) | o1 (78), Gemini 2.5 Pro (~84) |

**Takeaway:** a 16GB local MoE roughly tracks **Claude 3.5 Sonnet / Llama 3.1 70B-class quality on broad knowledge benches**, with notable strength on HumanEval (where local-quant contamination is plausible and worth a probe) and a clear gap from reasoning-tuned frontier models on GPQA / MATH-hard.

---

## Open work / what's missing

1. **Verify all `~` entries** against their primary source (model card, system card, or release blog) before quoting these numbers in a public report.
2. **Fill in Claude 4 / 4.5 / 4.6 / 4.7 Opus and Sonnet rows.** Knowledge-bench scores for the newest Anthropic models aren't all reported in places I've cross-checked. MMLU especially may not be reported (the family has been moving to harder evals).
3. **Note on DROP**: most providers post-2024 stopped reporting DROP — Anthropic only reports it through Claude 3.5, Google's Gemini cards typically omit it. Treat DROP comparison as "vs. the Claude 3 / Llama 3 era" not vs. current frontier.
4. **GPQA versions**: "Diamond" is the standard hard subset (198 questions). Some older papers cited GPQA Main (448 questions). Where I marked `~`, the version may differ from Diamond — needs check.
5. **Decide on a "thinking allowed" footnote**: o1, Gemini 2.5 Pro, DeepSeek R1 use extensive test-time compute. Our local harness limits max_tokens=32768 with no system-level CoT scaffolding. These models would score lower under our harness rules; published numbers represent best-effort settings.

## Sources to crosswalk before publishing

- OpenAI GPT-4 paper (arXiv 2303.08774), GPT-4o announcement post, o1 system card
- Anthropic Claude 3 / 3.5 / 4 model cards
- Google Gemini 1.5 / 2.0 / 2.5 technical reports
- Meta Llama 3.1 paper (arXiv 2407.21783), Llama 3.3 release notes
- Mistral Large 2 announcement
- DeepSeek V3 / R1 technical reports
- Qwen 2.5 technical report

Update the `✓` / `~` markers as each source is checked.
