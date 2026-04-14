# Discovered Communities, Resources, and Tools
## For Local LLM Quality Benchmarking on Apple Silicon
## Compiled April 8, 2026

---

## 1. Community Forums & Discussion Platforms

### Reddit
- **r/LocalLLaMA**: 266,500+ members - THE central hub for local AI discussion
  - URL: https://reddit.com/r/LocalLLaMA
  - Most active discussions on model quality, quantization, hardware
- **r/homelab**: Hardware setup sharing
- **r/privacy**: Privacy-focused local AI deployments

### Discord Servers
- **LocalLLM Discord**: 4,437 members - https://discord.com/invite/localllm
- **Open WebUI Discord**: 33,204 members - https://discord.com/invite/5rJgQTnV4s
- **LM Studio Discord**: Available from https://lmstudio.ai/ (member count not confirmed)
- **Ollama Discord**: Referenced in community (not directly confirmed URL)
- **llama.cpp Discord**: Referenced in GitHub Discussion #250

### HuggingFace Discussion Pages (Most Active for Our Models)
- Qwen/Qwen3.5-35B-A3B: https://huggingface.co/Qwen/Qwen3.5-35B-A3B/discussions (43+ threads)
- google/gemma-4-26B-A4B-it: https://huggingface.co/google/gemma-4-26B-A4B-it/discussions
- nvidia/Nemotron-Cascade-2-30B-A3B: https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B/discussions (22+ threads)
- unsloth/Qwen3.5-35B-A3B-GGUF: https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/discussions (25+ threads)
- HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive: 27 discussions, 1,230+ likes
- huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated: 18 discussions

### Hacker News
- Active LLM comparison threads: https://news.ycombinator.com/item?id=47582482 (Ollama MLX)
- Gemma vs Qwen benchmarks: https://news.ycombinator.com/item?id=47616761

### GitHub Discussions
- llama.cpp: https://github.com/ggml-org/llama.cpp/discussions
  - TurboQuant Discussion #20969
  - Performance on Apple Silicon Discussion #4167
  - MLX framework Discussion #4345
- MLX Community Projects: https://github.com/ml-explore/mlx/discussions/654
- Cline (Rapid-MLX integration): https://github.com/cline/cline/discussions/9940

### NVIDIA Developer Forums
- Nemotron-Cascade-2 discussion: https://forums.developer.nvidia.com/t/nvidia-nemotron-cascade-2-30b-a3b-yet-another-model-to-test/364250

---

## 2. Blogs & Newsletters

### Technical Newsletters (Quantization-Focused)
- **The Kaitchup** (Benjamin Marie): https://kaitchup.substack.com/
  - Most comprehensive GGUF evaluation series for Qwen3.5
  - 750-998 sample evaluations across LiveCodeBench, GPQA Diamond, MMLU-Pro, Math500
  - The Kaitchup Index: leaderboard for quantized models by GPU tier
  - Key articles:
    - "Qwen3.5 Quantization: Similar Accuracy, More Thinking"
    - "Summary of Qwen3.5 GGUF Evaluations"
    - "Lessons from GGUF Evaluations: Ternary Qwen3.5, Bricked MiniMax"
    - "Qwen3.5 Medium Models: Dense vs MoE"

### Developer Blogs
- **Simon Willison**: https://simonwillison.net/
  - Tags: https://simonwillison.net/tags/local-llms/, /tags/mlx/, /tags/qwen/
  - Created llm-mlx plugin: https://github.com/simonw/llm-mlx
  - Covered Dan Woods' 397B on 48GB Mac experiment
  - Covered Google AI Edge Gallery for local Gemma 4 on iPhone

- **Julien Simon** (Medium): https://julsimon.medium.com/
  - "What to Buy for Local LLMs (April 2026)" - comprehensive hardware guide

- **Aayush Garg**: https://aayushgarg.dev/
  - "Notes on Qwen3.5 vs Gemma4 for Local Agentic Coding" (April 5, 2026)
  - RTX 4090 testing with real agent coding tasks

- **Stochastic Sandbox**: https://stochasticsandbox.com/
  - "The Stack: Apple Silicon Local LLM Servers for Running Agents" (March 28, 2026)
  - Comprehensive server comparison for agent workloads

- **MACGPU Blog**: https://macgpu.com/en/blog/
  - "2026 Mac Inference Framework Selection: vllm-mlx vs. Ollama vs. llama.cpp"
  - M4 Pro 64GB benchmarks with 32 concurrent requests

- **Yage.ai**: https://yage.ai/
  - "MLX: The Next Inference Engine for Apple Silicon" (March 31, 2026)

### AI News & Analysis Sites
- **Artificial Analysis**: https://artificialanalysis.ai/ - Model comparison tool
- **ai.rs**: https://ai.rs/ - "Gemma 4 vs Qwen 3.5 vs Llama 4: Updated Benchmarks, New Leader"
- **Botmonster Tech**: https://botmonster.com/ - Open model comparison 2026
- **Maniac.ai**: https://www.maniac.ai/ - Benchmark-by-size comparison
- **BenchLM.ai**: https://benchlm.ai/ - AI benchmark comparison tool
- **DEV Community**: Active local LLM community
  - "Gemma 4 After 24 Hours" analysis
  - "Ollama Just Got 93% Faster on Mac"
  - Local LLM inference complete guide 2026
- **bswen.com**: Abliterated vs regular Qwen3.5 comparison (March 10, 2026)

### Official Vendor Blogs
- **Unsloth**: https://unsloth.ai/blog/ and https://unsloth.ai/docs/
  - Dynamic 2.0 GGUFs documentation
  - Model-specific quantization guides and benchmarks
- **Ollama**: https://ollama.com/blog/
  - MLX integration announcement (March 31, 2026)
- **Apple ML Research**: https://machinelearning.apple.com/research/
  - "Exploring LLMs with MLX and Neural Accelerators in M5 GPU"
  - WWDC 2025: "Get started with MLX for Apple silicon"

---

## 3. Benchmarking Tools & Databases

### Apple Silicon Benchmark Databases
- **Silicon Score**: https://siliconscore.com/bench/
  - 309 benchmark rows, 28 Mac configs, 33 models
  - Per-model Apple Silicon field reports
  - Mac buying guide: https://siliconscore.com/guides/best-mac-for-local-llms/
  
- **SiliconBench**: https://siliconbench.radicchio.page/
  - Measured tok/s by model, chip, quantization
  - JSON/CSV downloads, real runs not estimates
  - M5 benchmarks added March 18, 2026

- **Apple Silicon Benchmark Explorer**: https://creativestrategies.com/wp-content/uploads/2026/03/apple_silicon_benchmark_explorer.html

- **benchmarks_llm_silicon** (GitHub): https://github.com/ivanfioravanti/benchmarks_llm_silicon
  - Community benchmark storage for MLX, llama.cpp, LM Studio, Ollama

### Benchmarking Software
- **MLXBench**: https://github.com/linusvwe/MLXBench
  - Compare Ollama, vLLM-MLX, Docker Model Runner
  - Measures TTFT, tok/s, e2e latency, inter-token latency, memory
  - Configurable via TOML
  
- **Anubis**: https://github.com/uncSoft/anubis-oss
  - Native macOS SwiftUI app
  - Real-time hardware telemetry
  - Any OpenAI-compatible endpoint

- **local-llm-bench**: https://github.com/famstack-dev/local-llm-bench
  - MLX vs llama.cpp scenario comparison on Apple Silicon

- **mlx_transformers_benchmark**: https://github.com/aukejw/mlx_transformers_benchmark
  - LLM benchmarking on Apple Silicon

- **lm-evaluation-harness**: https://github.com/EleutherAI/lm-evaluation-harness
  - 60+ standard academic benchmarks
  - Supports MPS backend, GGUF models, local-completions API
  - Backend for HuggingFace Open LLM Leaderboard

### Model Fit / Recommendation Tools
- **ModelFit / LLMfit**: https://modelfit.io/
  - Find best LLM for specific Mac hardware
- **Will It Run AI**: https://willitrunai.com/
  - Mac-specific model compatibility guide

---

## 4. Inference Engines & Frameworks

### Primary (Apple Silicon Optimized)
- **Ollama** (v0.19 with MLX): https://ollama.com/
  - Now MLX-powered on Apple Silicon (March 31, 2026)
  - 2.3x decode speedup over old llama.cpp backend
  - Best for: development, single-user, ease of use

- **Rapid-MLX**: https://github.com/raullenchai/Rapid-MLX
  - Fastest local AI engine for Apple Silicon (claimed)
  - 2-4.2x faster than Ollama on M3 Ultra
  - DeltaNet state snapshots for hybrid models
  - 100% tool calling, 17 parsers
  - `brew install raullenchai/rapid-mlx/rapid-mlx`

- **MLX-LM** (Apple official): https://github.com/ml-explore/mlx-lm
  - Most mature MLX inference tool
  - Supports inference + LoRA/QLoRA fine-tuning

- **vLLM-MLX**: https://macgpu.com/en/blog/2026-mac-inference-framework-vllm-mlx-ollama-llamacpp-benchmark.html
  - PagedAttention + continuous batching on Apple Silicon
  - Best for: high-concurrency (25+ req/s), production agent fleets

- **oMLX**: https://github.com/jundot/omlx
  - SSD-backed KV caching, survives server restarts
  - Menu bar app
  - Best for: long-running agents on Mac

- **LM Studio** (v0.4.6): https://lmstudio.ai/
  - GUI-based, supports both GGUF and MLX backends
  - Continuous batching for MLX (v0.4.2+)
  - Best for: non-technical users, model browsing

- **llama.cpp**: https://github.com/ggml-org/llama.cpp
  - Metal GPU acceleration
  - Foundational backbone, maximum control
  - GGUF format originator

### Model Format Providers
- **MLX Community** (HuggingFace): https://huggingface.co/mlx-community
  - 4,316 pre-converted MLX models
  - Most major models available within days of release

---

## 5. Quantization Resources

### Quantization Providers (by methodology)
- **Unsloth** (Dynamic 2.0): https://unsloth.ai/
  - Per-model layer-wise quantization optimization
  - Works on MoE and non-MoE
  - Lower KLD than standard imatrix and QAT

- **mudler/APEX**: https://github.com/mudler/apex-quant
  - MoE-aware mixed-precision quantization
  - 23 models in collection: https://huggingface.co/collections/mudler/apex-quants-gguf
  - Technical report available
  - Beats F16 perplexity at 38% smaller size

- **bartowski** (HuggingFace): Standard high-quality GGUF quants
  - https://huggingface.co/bartowski

- **mradermacher** (HuggingFace): Comprehensive GGUF coverage
  - https://huggingface.co/mradermacher

- **lmstudio-community**: LM Studio optimized GGUFs
  - https://huggingface.co/lmstudio-community

- **ggml-org**: Official llama.cpp GGUFs
  - https://huggingface.co/ggml-org

### Quantization Techniques Documented
| Technique | Source | Key Feature |
|-----------|--------|-------------|
| APEX | mudler | MoE-aware layer-wise precision gradient |
| Dynamic 2.0 | Unsloth | Per-model layer-wise optimization |
| Standard imatrix | Various | Wikipedia-calibrated importance matrix |
| NVFP4 | NVIDIA | 4-bit for coding/agent models |
| TurboQuant KV | llama.cpp | KV cache compression (13-14% prefill speedup) |
| QAT | Various | Quantization-aware training |
| TQ1_0 (Ternary) | llama.cpp | {-1,0,+1} weights, extreme compression |
| MXFP4 | Various | Mixed-format FP4 (found problematic for some MoE layers) |

---

## 6. Uncensored/Abliterated Model Providers

| Provider | Method | Key Model | Downloads/mo |
|----------|--------|-----------|-------------|
| HauhauCS | Proprietary (private) | Qwen3.5-35B-A3B-Uncensored-Aggressive | 815,445 |
| huihui-ai | Abliteration (remove-refusals-with-transformers) | Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated | 136,281 |
| huihui-ai | Abliteration | Huihui-Qwen3.5-35B-A3B-abliterated | 62,666 |
| Jackrong | Claude-4.6-Opus SFT distillation | Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled | N/A |
| dealignai | JANG method | Nemotron-Cascade-2-30B-A3B-UNCENSORED-JANG_2L | N/A |
| DavidAU | Heretic/NEO-CODE/Imatrix | GLM-4.7-Flash-Uncensored variants | N/A |

### Key abliteration resources:
- Original paper: "Refusal in Language Models Is Mediated by a Single Direction" (2024, arxiv.org/abs/2406.11717)
- Tutorial: https://huggingface.co/blog/mlabonne/abliteration
- Tool: https://github.com/Sumandora/remove-refusals-with-transformers
- Improved method: "Norm-Preserving Biprojected Abliteration" - https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration

---

## 7. Model Comparison & Analysis Tools

- **Artificial Analysis**: https://artificialanalysis.ai/models/comparisons/gemma-4-26b-a4b-vs-qwen3-5-35b-a3b
- **Arena AI / LMArena**: Human preference ELO rankings
- **Open LLM Leaderboard** (HuggingFace): Automated benchmarks
- **Open WebUI Leaderboard**: https://openwebui.com/leaderboard - 100+ LLMs ranked by real usage
- **Price Per Token**: https://pricepertoken.com/leaderboards/ - Rankings with community votes

---

## 8. Key Developer Profiles to Follow

| Person | Platform | Focus |
|--------|----------|-------|
| Benjamin Marie (@bnjmn_marie) | Substack (Kaitchup), X | GGUF evaluation, quantization |
| Simon Willison (@simonw) | Blog, GitHub, X | Local LLM tooling, MLX |
| mudler/Ettore Di Giacinto (@mudler_it) | GitHub, HuggingFace, X | APEX quantization, LocalAI |
| Julien Simon | Medium | Hardware buying guides |
| Daniel Han (danielhanchen) | HuggingFace, GitHub | Unsloth, Dynamic 2.0 |
| HauhauCS | HuggingFace | Uncensored models |
| huihui-ai | HuggingFace | Abliterated models |
| bartowski | HuggingFace | High-quality GGUF quantizations |
| Jackrong | HuggingFace | Claude distillation models |
| raullenchai | GitHub | Rapid-MLX |
| David Hendrickson (@TeksEdge) | X | APEX benchmark reporting |

---

## 9. Hardware-Specific Resources

### Apple Silicon
- WWDC 2025 MLX session: https://developer.apple.com/videos/play/wwdc2025/315/
- Apple ML Research MLX page: https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- "LLM in a Flash" technique for running 397B on 48GB Mac
- Google AI Edge Gallery: iPhone app for local Gemma 4

### Memory Bandwidth Reference
| Chip | Bandwidth | Typical tok/s (30B MoE) |
|------|-----------|------------------------|
| M1 | 68 GB/s | ~15-20 |
| M2 | 100 GB/s | ~20-30 |
| M3 | 100 GB/s | ~20-30 |
| M4 | 120 GB/s | ~25-35 |
| M4 Pro | 273 GB/s | ~50-60 |
| M4 Max | 546 GB/s | ~80-100 |
| M3 Ultra | 800 GB/s | ~100-120 |
| M5 (est.) | TBD | 19-27% faster than M4 |

### Mac Buying Guides
- Silicon Score: https://siliconscore.com/guides/best-mac-for-local-llms/
- Will It Run AI: https://willitrunai.com/blog/best-llm-for-mac-apple-silicon-2026
- ToolHalla: https://toolhalla.ai/blog/best-local-llm-mac-apple-silicon-2026
- Refurb.me: https://www.refurb.me/blog/best-mac-for-ai
- InsiderLLM: https://insiderllm.com/guides/best-local-llms-mac-2026/

---

## 10. Academic / Research Resources

- **"Benchmarking On-Device ML on Apple Silicon with MLX"**: https://arxiv.org/abs/2510.18921
- **"Native LLM and MLLM Inference at Scale on Apple Silicon"**: https://arxiv.org/html/2601.19139v2
- **"A Comparative Study of MLX, MLC-LLM, Ollama, llama.cpp"**: https://arxiv.org/pdf/2511.05502
- **APEX Technical Report**: https://github.com/mudler/apex-quant/blob/main/paper/APEX_Technical_Report.md
- **Nemotron-Cascade 2 Research Page**: https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/
