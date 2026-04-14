# Inference Engine Comparison Results
**Date:** 2026-04-08
**Hardware:** Apple M4, 32GB unified memory, 120 GB/s bandwidth
**Tool:** scripts/engine_benchmark.py (5 questions: 2+2, MMLU, Code, Reasoning, Math)

---

## Test 1: Gemma 4 26B-A4B (MoE, 3.8B active)

GGUF: Q4_K_M 17.99GB | MLX: mlx-community 4-bit ~16GB

| Engine | Backend | Format | Total | 2+2 | MMLU | Code | Reason | Math |
|--------|---------|--------|-------|------|------|------|--------|------|
| **LM Studio** | llama.cpp | GGUF Q4_K_M | **54.3s** | 2.5s | 7.5s | 19.4s | 12.7s | 12.2s |
| llama-server | llama.cpp raw | GGUF Q4_K_M | 79.2s | 2.1s | 7.3s | 32.6s | 25.0s | 12.2s |
| Rapid-MLX | MLX | MLX 4-bit | 81.0s | 6.5s | 9.7s | 31.2s | 21.3s | 12.3s |
| mlx_lm.server | MLX | MLX 4-bit | ~80s* | 9.2s | 8.2s | 20.6s | - | - |
| Ollama 0.20 | llama.cpp** | GGUF Q4_K_M | 83.4s | 13.0s | 8.1s | 21.1s | 25.5s | 15.8s |

*partial test | **likely llama.cpp fallback, not MLX

**Notes:**
- All engines produced correct answers on all questions
- LM Studio 1.5x faster than all alternatives
- MLX model (27.6GB needed) exceeded recommended 21.8GB, causing memory pressure warnings
- Gemma 4 thinking mode: 54-510 reasoning tokens per question (LM Studio separates these)
- Ollama first query slow (13s) due to cold model load

---

## Test 2: Qwen3.5-9B (Dense, 9B params, ~6GB model)

GGUF: Q4_K_M 6.55GB | MLX: mlx-community 4-bit ~5-6GB

| Engine | Backend | Format | Total | 2+2 | MMLU | Code | Reason | Math |
|--------|---------|--------|-------|------|------|------|--------|------|
| **LM Studio** | llama.cpp | GGUF Q4_K_M | **345s** | 9.8s | 18.7s | 117s | timeout | 79s |
| llama-server | llama.cpp raw | GGUF Q4_K_M | 369s | 10.3s | 26.6s | timeout | timeout | 92s |
| Rapid-MLX | MLX | MLX 4-bit | 278s | 10.2s | 26.1s | timeout | timeout | crash |
| mlx_lm.server | MLX | MLX 4-bit | 345s | 10.8s | 27.9s | timeout | timeout | 66s |
| Ollama 0.20 | llama.cpp | GGUF | 412s | 13.5s | 38.8s | timeout | timeout | timeout |

**Notes:**
- Qwen3.5-9B generates massive thinking chains (1000-1800 tokens), causing 120s timeouts
- Code question: 1760 thinking tokens + 44 code tokens (LM Studio: 117s, others: timeout)
- On comparable short questions, LM Studio is ~30% faster than MLX engines
- Ollama consistently slowest (~45% slower than LM Studio)
- Rapid-MLX crashed on Math question (remote connection closed)

---

## Conclusions

1. **LM Studio (llama.cpp) is the fastest engine on base M4 (120 GB/s)**
   - 1.5x faster than raw llama-server (optimization overhead pays off)
   - 1.3-1.5x faster than MLX engines (bandwidth bottleneck equalizes compute advantage)
   - 1.5-1.8x faster than Ollama (Go wrapper + possible llama.cpp fallback)

2. **MLX speed advantage does NOT apply to base M4**
   - MLX advantage is real on M4 Pro/Max (273-546 GB/s) where compute matters more
   - On 120 GB/s, all engines are memory-bandwidth-bound
   - MLX also has larger model footprint (~7-13% bigger than GGUF)

3. **Model choice matters more than engine choice**
   - Gemma 4 26B-A4B: 54s total (efficient thinking, ~50-500 reasoning tokens)
   - Qwen3.5-9B: 345s total (excessive thinking, 1000-1800 reasoning tokens)
   - Despite being 3x smaller, Qwen3.5-9B is 6x slower due to thinking overhead

4. **Recommendation:** Use LM Studio + GGUF for all benchmarking on this hardware
