# Windows Transfer Package — Gemma 4 19B REAP Benchmarking

This package contains everything needed to run the same benchmark suite on a Windows PC with RTX 3080 Ti (12 GB VRAM).

**Goal:** Test 0xSero's pruned `gemma-4-19b-a4b-it-REAP` against our verified Gemma 4 26B-A4B scores to see if REAP preserves quality.

---

## What's in this package

```
remote-setup/
├── README.md                   ← you are here
├── requirements.txt             ← Python deps
├── scripts/
│   ├── bench2.py                ← patched: supports LMSTUDIO_URL env var, Windows psutil, nvidia-smi
│   └── lms.py                   ← patched: Windows compatible
└── runs/                        ← output will land here (created on first run)
```

**Note:** The scripts are patched versions of the Mac originals. They add:
- `LMSTUDIO_URL` env var support (for running over network from the Mac)
- `psutil` for cross-platform memory stats
- `nvidia-smi` for GPU temperature
- Windows-safe paths

---

## Setup Steps

### 1. Install Python 3.10+ for Windows

Download from https://www.python.org/downloads/windows/ — pick the "Windows installer (64-bit)" and **check "Add Python to PATH"** during install.

Verify:
```powershell
python --version
# Should show Python 3.10 or later
```

### 2. Install LM Studio for Windows

Download from https://lmstudio.ai/download — run the installer.

After install:
- Open LM Studio
- Go to **Settings** (gear icon) → **Developer** → enable "Show Developer Logs"
- Go to **Local Server** tab → check "Serve on Local Network" (so your Mac can connect later)
- Check default port is 1234

### 3. Install Python dependencies

Open PowerShell in this `remote-setup/` directory:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Download the model

**Recommended for 3080 Ti 12 GB VRAM:**
- Repo: `mradermacher/gemma-4-19b-a4b-it-REAP-i1-GGUF`
- File: `gemma-4-19b-a4b-it-REAP.i1-IQ4_XS.gguf` (9.5 GB)
- Why this one: fits in 12 GB VRAM with ~16K context headroom, uses importance-matrix calibration for better quality per GB than standard K-quants

**In LM Studio:**
1. Click the magnifying glass icon (Search)
2. Paste: `mradermacher/gemma-4-19b-a4b-it-REAP-i1-GGUF`
3. Find `i1-IQ4_XS.gguf` in the file list → click Download
4. Wait for ~9.5 GB download to finish

### 5. Load the model with proper settings

In LM Studio → Chat tab → select the model → click "Load":

**Load settings (important!):**
- **Context length:** 16384 (up to 32768 if you have headroom)
- **GPU offload:** Max (all layers — we want everything on GPU)
- **Flash attention:** Enabled (reduces KV cache size)
- **Keep model in RAM:** Enabled

After load, run a test chat ("What is 2+2?") to verify it works.

### 6. Verify the API is serving

In LM Studio → **Local Server** tab → Start Server (default port 1234).

From PowerShell:
```powershell
curl http://127.0.0.1:1234/v1/models
# Should return JSON with the loaded model id
```

If you want to test from your Mac, find this PC's IP:
```powershell
ipconfig
# Look for "IPv4 Address" under your WiFi/Ethernet adapter, e.g., 192.168.1.100
```

Then from Mac:
```bash
curl http://192.168.1.100:1234/v1/models
```

### 7. Run the pre-flight check

From this directory in PowerShell (with venv activated):
```powershell
python scripts/lms.py check
```

Expected output:
```
=== PRE-FLIGHT CHECK ===
Model:   mradermacher/gemma-4-19b-a4b-it-REAP-i1-GGUF
...
Warming up ...
  Response: "4"
  Warmup OK
PRE-FLIGHT: PASS
```

---

## Running the benchmarks

### Quick test (1-2 questions) — verify it works

```powershell
python scripts/bench2.py math --examples 100 --only 1,2
```

Should output per-question logs and a summary. Check that it prints `tok_s` and `OK/FAIL` and writes files to `runs/`.

### Full benchmark suite

Run each benchmark back-to-back. Each one takes ~15 min to ~2 hours depending on benchmark.

```powershell
# MMLU — fastest, ~15 min
python scripts/bench2.py mmlu --examples 100 2>&1 | Tee-Object -FilePath runs/gemma19b_mmlu_console.log

# HumanEval — ~30 min
python scripts/bench2.py humaneval --examples 100 2>&1 | Tee-Object -FilePath runs/gemma19b_humaneval_console.log

# DROP — ~30 min
python scripts/bench2.py drop --examples 100 2>&1 | Tee-Object -FilePath runs/gemma19b_drop_console.log

# MATH — longest, ~2-4 hours (heavy thinking)
python scripts/bench2.py math --examples 100 2>&1 | Tee-Object -FilePath runs/gemma19b_math_console.log

# GPQA — very long, ~3-10 hours (Gemma thinks a LOT on GPQA)
python scripts/bench2.py gpqa --examples 100 2>&1 | Tee-Object -FilePath runs/gemma19b_gpqa_console.log
```

**Between runs:** let GPU cool down to < 50°C. Check with:
```powershell
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits
```

---

## Running from Mac (remote mode)

If you want to keep all your scripts/data on the Mac and just use the Windows PC as a GPU server:

1. Make sure LM Studio is running with "Serve on Local Network" enabled
2. From your Mac:
   ```bash
   export LMSTUDIO_URL=http://192.168.1.100:1234/v1
   python3 scripts/bench2.py math --examples 100
   ```
3. Results are saved to the Mac's `results/runs/` — no need to copy anything back

**Note:** You'll need to patch the Mac's `bench2.py` to also read `LMSTUDIO_URL` from env (the Mac version hardcodes 127.0.0.1). Or just use the patched version from this transfer package and copy back any results.

---

## Expected Results

We're testing whether 0xSero's REAP pruning preserves quality. Compare these to our verified Mac results for **Gemma 4 26B-A4B (unpruned)**:

| Benchmark | Gemma 4 26B-A4B (our baseline) | Gemma 4 19B REAP (target) |
|-----------|-------------------------------:|--------------------------:|
| MMLU | 84% | ≥80% expected |
| HumanEval | 99% | ≥95% expected |
| MATH | 82% | ≥78% expected |
| DROP | 89% | ≥85% expected |
| GPQA | ~70% (10q probe) | ≥60% expected |

**The model is ~27% smaller than the original**, but 0xSero claims it "held up really well and actually gained accuracy on reasoning tasks". This benchmark run is a clean empirical test of that claim.

**Expected speed on RTX 3080 Ti 12GB:**
- Non-thinking tasks (MMLU, DROP): ~50-80 tok/s
- Thinking tasks (MATH, GPQA): ~40-60 tok/s
- 3-4x faster than your Mac M4 Air for the same model class

---

## Troubleshooting

**"nvidia-smi not found"**
- Install NVIDIA drivers from https://www.nvidia.com/download/index.aspx
- Or install CUDA Toolkit (includes nvidia-smi)

**"CUDA out of memory" when loading**
- Reduce context_length in LM Studio (try 8192 instead of 16384)
- Or pick a smaller quant: `i1-IQ3_M.gguf` (8.5 GB)

**API 400 errors on MATH / GPQA**
- Usually "Context size has been exceeded" — the model ran out of context mid-generation
- Increase context_length in LM Studio model config and reload

**Very slow inference**
- Check GPU offload is at max (all layers on GPU)
- Check flash attention is enabled
- Watch nvidia-smi to confirm the GPU is actually being used
- Check Windows isn't running heavy tasks on the GPU (Discord hardware accel, browser, etc.)

**psutil import error**
```powershell
pip install psutil
```

**How to stop a running benchmark**
- Ctrl+C in the terminal
- The JSONL file so far is preserved — you can resume with `--only` and a list of questions not yet run

---

## Data Files You'll Get

After a successful run:

```
runs/
├── math_{model}_{timestamp}.jsonl          ← 100 lines, one per question
├── math_{model}_{timestamp}_summary.json   ← aggregate score + metadata
├── gemma19b_math_console.log               ← full stdout (from Tee)
```

Each JSONL line has: question_num, prompt/completion/thinking tokens, elapsed time, tok/s, GPU temp, correct/wrong, expected/extracted answer, and the full raw model response (for later re-grading if needed).

To compare with Mac results, just share the JSONL file or the summary JSON with me and I'll cross-reference.
