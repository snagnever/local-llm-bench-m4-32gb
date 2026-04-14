# Speed Probe Results: All 10 Models
**Date:** 2026-04-08 | **Hardware:** M4 32GB | **Engine:** LM Studio llama.cpp
**Test:** 3 questions (2+2, MMLU atmosphere, code second_largest)
**Params:** temp=0, max_tokens=1024, context=16384

## Results (sorted by total time)

| # | Model | GGUF | Total | 2+2 | MMLU | Code | Correct | Think? | GPU% | Swap |
|---|-------|------|-------|------|------|------|---------|--------|------|------|
| 1 | Qwen3-Coder-30B-A3B | 17GB Q4 | **2.9s** | 0.3s | 0.8s | 1.8s | 2/2 | No | 61% | 8.0GB |
| 2 | Devstral Small 2 24B | 13GB Q4 | **10.8s** | 0.8s | 1.8s | 8.2s | 2/2 | No | 90% | 5.8GB |
| 3 | Gemma 4 26B-A4B | 16GB Q4 | **25.9s** | 2.6s | 7.9s | 15.4s | 2/2 | Yes (efficient) | 95% | 4.2GB |
| 4 | huihui-claude-i1 (ref) | 14GB Q3 | 31.9s | 6.5s | 19.7s | 5.7s | 1/2 | Yes (moderate) | 95% | 4.2GB |
| 5 | GLM-4.7-Flash | 17GB Q4 | 59.1s | 1.9s | 15.8s | 41.4s | 2/2 | Yes (heavy) | 69% | 4.0GB |
| 6 | HauhauCS Uncensored | 16GB Q3 | 61.4s | 4.7s | 14.7s | 42.0s* | 2/2 | Yes (heavy) | 84% | 3.8GB |
| 7 | Qwen3.5-9B | 5.2GB Q4 | 62.9s | 8.4s | 28.5s | 26.0s | 2/2 | Yes (heavy) | 80% | 4.2GB |
| 8 | Huihui plain abliterated | 14GB Q3 | 63.9s | 4.9s | 19.8s | 39.2s* | 1/2 | Yes (heavy) | 97% | 3.8GB |
| 9 | Qwen3.5-35B base | 20GB Q4 | 72.8s | 6.0s | 17.4s | 49.3s* | 2/2 | Yes (heavy) | 69% | 4.1GB |
| 10 | Nemotron-Cascade-2 | 23GB Q4 | FAILED | 400 | 400 | 400 | 0/3 | - | 2% | 7.4GB |

*hit max_tokens (1024) - all thinking, answer truncated

## Key Findings

### Speed Tiers
- **Tier 1 (< 15s):** Qwen3-Coder (2.9s), Devstral (10.8s) - no thinking mode
- **Tier 2 (15-35s):** Gemma 4 (25.9s), huihui-claude ref (31.9s) - efficient thinking
- **Tier 3 (40-75s):** GLM, HauhauCS, Qwen3.5 variants - heavy thinking (500-1024 tokens)
- **FAILED:** Nemotron-Cascade-2 - too large for 32GB (23GB model + 7.4GB swap)

### Thinking Mode Impact
| Behavior | Models | Speed | Token Overhead |
|----------|--------|-------|---------------|
| No thinking | Qwen3-Coder, Devstral | 3-11s | 0 extra tokens |
| Efficient thinking | Gemma 4 | 26s | 50-500 reasoning tokens |
| Moderate thinking | huihui-claude | 32s | 93-341 tokens |
| Heavy thinking | Qwen3.5 family, GLM, HauhauCS | 59-73s | 348-1024 tokens |

### Eliminated
- **Nemotron-Cascade-2:** 23GB is too big. 400 errors, 2% GPU, massive swap. Eliminated.
- **Qwen3.5-35B base (22GB):** Borderline - 69% GPU, lots of thinking. Keep for quality test but memory is tight.

### Advancing to Quality Benchmark (7 models)
1. Qwen3-Coder-30B-A3B (fastest, but is it smart?)
2. Devstral Small 2 24B (fast, but quality?)
3. Gemma 4 26B-A4B (balanced speed + thinking)
4. huihui-claude-i1 (our reference)
5. GLM-4.7-Flash (slow but potentially high quality)
6. HauhauCS Uncensored (most popular, compare to Qwen3.5 family)
7. Qwen3.5-9B (tiny, but heavy thinker)

Huihui plain abliterated and Qwen3.5-35B base are borderline - same arch as HauhauCS with worse instruction following.
