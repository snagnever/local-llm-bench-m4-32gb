# Full Model Ranking - 32GB M4 Mac
**Date:** 2026-04-08 | **Budget:** ~14-22GB GGUF | **Engine:** LM Studio

Legend: **TESTED** = we ran benchmarks | **PROBED** = quick speed test only | NEW = untested candidate

---

## Master Table (ranked by composite expected quality)

| # | Model | Type | Total/Active | GGUF Size | Arena ELO | MMLU-Pro | GPQA | LiveCode | AIME | Status | Notes |
|---|-------|------|-------------|-----------|-----------|----------|------|----------|------|--------|-------|
| 1 | Qwen3.5-35B-A3B (base) | MoE | 35B/3B | 15.3 Q3 | 1400 | 85.3 | 84.2 | 74.6 | 89.0 | NEW | Pure base, no fine-tune |
| 2 | Gemma-4-26B-A4B-it | MoE | 25B/3.8B | 13.4 IQ4 | 1441 | 82.6 | 82.3 | 77.1 | 88.3 | NEW | Highest Arena ELO |
| 3 | Nemotron-Cascade-2-30B-A3B | MoE+Mamba | 32B/3B | ~20 Q3 | - | 79.8 | 76.1 | 87.2 | 90.9 | NEW | IMO/IOI/ICPC golds |
| 4 | HauhauCS Uncensored-Aggressive | MoE | 35B/3B | 15.3 Q3 | - | ~85* | ~84* | - | - | NEW | 815K dl, 1230 likes |
| 5 | Huihui-abliterated (plain) | MoE | 35B/3B | 15.3 Q3 | - | ~85* | ~84* | - | - | NEW | No Claude distill, 262K ctx |
| 6 | **huihui-claude-opus-abliterated-i1** | **MoE** | **35B/3B** | **15.3 Q3** | **-** | **90%** | **60%** | **-** | **-** | **TESTED** | **Our reference. 8K ctx** |
| 7 | GLM-4.7-Flash | MoE | 30B/3B | 14.6 Q3 | - | ~60 | 75.2 | 64.0 | 91.6 | NEW | Fastest inference |
| 8 | Qwen3-Coder-30B-A3B | MoE | 30B/3.3B | 13.3 Q3 | - | - | - | ~SWE51.6 | - | NEW | Coding specialist |
| 9 | Qwopus3.5-27B-v3 | Dense | 27B/27B | ~14 Q3 | - | - | - | HE:97.6 | - | PROBED | 15s for "2+2". Slow. |
| 10 | Qwen3.5-9B | Dense | 9B/9B | 5.7 Q4 | - | 82.5 | 81.7 | - | 91.3 | NEW | Beats 120B models! |
| 11 | Nemotron-3-Nano-30B-A3B | MoE+Mamba | 31B/3.5B | 20 Q3 | - | 78.3 | 73.0 | 68.3 | 89.1 | NEW | 1M context, Mamba hybrid |
| 12 | Qwen3-30B-A3B | MoE | 30B/3.3B | 14.7 Q3 | - | 80.9 | - | 66.0 | - | NEW | Predecessor to 3.5 |
| 13 | Huihui3.5-67B-A3B | MoE | 68B/3B | ~30? Q3 | - | - | - | - | - | NEW | 1 day old, may not fit |
| 14 | GLM-4.7-Flash-Heretic-NEO-CODE | MoE | 30B/3B | ~14 Q4 | - | - | - | - | - | NEW | Coding abliterated GLM |
| 15 | Phi-4-reasoning 14B | Dense | 14B/14B | 8.9 Q4 | - | 84.8 MMLU | - | - | - | NEW | Math/STEM specialist |
| 16 | Mistral Small 3.2 24B | Dense | 24B/24B | 13.1 Q4 | - | 81+ | - | - | - | NEW | Instruction following |
| 17 | Qwopus3.5-9B-v3 | Dense | 9B/9B | ~5.5 Q4 | - | - | - | HE:87.8 | - | NEW | Small Claude distill |
| 18 | Gemma-4 (dense variant) | Dense | 27B/27B | ~17 Q4 | ~1452 | 85.2 | 85.7 | - | 89.2 | TESTED | Slower than MoE on our HW |
| 19 | Qwopus3.5-27B-v3-i1 (dense) | Dense | 27B/27B | ~16 Q4 | - | - | - | - | - | PROBED | ~15s for "2+2". Too slow. |

*\* = estimated from base model; abliteration may degrade scores*

---

## Where Our Model (#6) Stands

```
                    MMLU-Pro  GPQA   LiveCode  AIME   Arena  GGUF
                    --------  ----   --------  ----   -----  ----
#1 Qwen3.5 base     85.3     84.2    74.6     89.0   1400   15.3
#2 Gemma-4 MoE      82.6     82.3    77.1     88.3   1441   13.4  <-- smallest
#3 Cascade-2        79.8     76.1    87.2     90.9    -     ~20   <-- code/math king
#4 HauhauCS         ~85*     ~84*     -        -      -     15.3  <-- most popular
#5 Huihui plain     ~85*     ~84*     -        -      -     15.3  <-- full 262K ctx
                    --------  ----   --------  ----   -----  ----
#6 OUR MODEL         90%↑     60%↓    -        -      -     15.3  <-- 8K ctx limit
                    --------  ----   --------  ----   -----  ----
#7 GLM-4.7-Flash     ~60      75.2    64.0     91.6   -     14.6  <-- fastest
#10 Qwen3.5-9B       82.5     81.7     -       91.3   -      5.7  <-- tiny giant
```

### Key Observations:

**Our model's MMLU 90% is suspicious.** It's HIGHER than the base model's official 85.3 MMLU-Pro. But note: we ran standard MMLU (not MMLU-Pro, which is harder). The 90% on standard MMLU is plausible and roughly in line.

**Our GPQA 60% is alarming.** The base model scores 84.2 on GPQA Diamond. Three possible explanations:
1. **8K context limit** - Claude distillation reduced context from 262K to 8K. GPQA questions with long passages may get truncated.
2. **Abliteration damage** - community reports intelligence degradation from abliteration
3. **Quantization at Q3** - aggressive quantization hurts specialized reasoning more than general knowledge

**The plain abliterated version (#5) and HauhauCS (#4) likely score closer to 84% on GPQA** because they preserve the full 262K context window and don't have the Claude LoRA context regression.

### Speed Comparison (on our 32GB M4):

| Model | Type | Expected Speed | Actual |
|-------|------|---------------|--------|
| MoE 35B-A3B models (#1,4,5,6) | MoE 3B active | ~5-10s per question | ~5s (tested) |
| Gemma-4-26B-A4B (#2) | MoE 3.8B active | ~5-8s per question | untested |
| GLM-4.7-Flash (#7) | MoE 3B active | ~3-6s per question (fastest arch) | untested |
| Nemotron Cascade (#3) | MoE+Mamba 3B | ~5-10s per question | untested |
| Dense 27B (#9,18,19) | Dense 27B active | ~15-20s per question | ~15s (tested) |
| Qwen3.5-9B (#10) | Dense 9B active | ~2-4s per question | untested |

---

## Recommended Testing Order

**Round 1 - Quick probe (2 min each):**
1. Gemma-4-26B-A4B (smallest, highest Arena ELO)
2. GLM-4.7-Flash (fastest expected)
3. Qwen3.5-35B-A3B base (test if distillation hurt quality)
4. HauhauCS Uncensored-Aggressive (most popular)
5. Qwen3.5-9B (tiny giant at 5.7GB)

**Round 2 - If promising, add:**
6. Nemotron-Cascade-2 (math/code beast)
7. Huihui plain abliterated (compare to our Claude version)
8. Qwen3-Coder-30B-A3B (coding specialist)

**Round 3 - Full benchmark on top 3-5 survivors**
