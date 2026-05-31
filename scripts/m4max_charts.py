#!/usr/bin/env python3
"""
Generate Phase 1 progress charts for the M4 Max 128 GB fork.

Scans benchmarks/runs/*_summary.json (knowledge + tool-calling) and overlays
upstream's published M4 Air 32 GB numbers for comparison.

Outputs into results/charts/:
  - chart_m4max_phase1_scores.png   (model × benchmark accuracy)
  - chart_m4max_phase1_throughput.png (per-bench tok/s)
  - chart_m4max_phase1_time.png     (per-bench wall-clock minutes, log scale)

Run anytime. Safe to call mid-Phase-1: partial data is plotted as-is and
missing cells are left blank.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent
REPO = SCRIPT_DIR.parent
RUNS = REPO / "benchmarks" / "runs"
CHARTS = REPO / "results" / "charts"

# Upstream's published numbers (M4 Air 32 GB, GGUF Q4_K_M, LM Studio).
# Source: README.md "Knowledge benchmarks (100q each)" table, plus
# tool_calling_results.md combined column for jdhodges+Veerman.
UPSTREAM = {
    "Gemma 4 26B-A4B (upstream)": {
        "mmlu": 0.84, "humaneval": 0.99, "math": 0.82,
        "drop": 0.89, "gpqa": 0.64, "tool_combined": 0.923,
    },
    "Qwen3-Coder 30B-A3B (upstream)": {
        "mmlu": 0.67, "humaneval": 0.94, "math": 0.78,
        "drop": 0.80, "gpqa": 0.42, "tool_combined": 0.00,  # broken jinja template
    },
    "huihui-ref 35B-A3B (upstream)": {
        "mmlu": 0.78, "humaneval": 0.91, "math": 0.73,
        "drop": 0.89, "gpqa": 0.54,
    },
    "Gemma 4 21B REAP (upstream)": {
        "tool_combined": 0.962,
    },
}

# Benchmarks in canonical display order
BENCH_ORDER = ["humaneval", "mmlu", "math", "drop", "gpqa",
               "livecodebench", "tool_combined", "tbench"]
BENCH_LABEL = {
    "humaneval": "HumanEval", "mmlu": "MMLU", "math": "MATH",
    "drop": "DROP", "gpqa": "GPQA",
    "livecodebench": "LCB v6\n(n=50)",
    "tool_combined": "Tool calls\n(combined)",
    "tbench": "Terminal-\nBench 2.0",
}


def load_local_runs():
    """Walk benchmarks/runs/*_summary.json, return {model: {bench: (score, elapsed_s, tok_s)}}."""
    out: dict[str, dict[str, tuple[float, float, float]]] = {}

    # Knowledge benches (one summary per (bench, model) — keep most recent if multiple)
    for f in sorted(RUNS.glob("*_summary.json")):
        if f.name.startswith("toolcall_"):
            continue
        d = json.loads(f.read_text())
        model = d["model"]
        bench = d["benchmark"]
        score = d["score"]
        elapsed = d.get("elapsed_s", 0)
        # tok/s isn't in knowledge summaries — derive crude average from jsonl
        tok_s = _avg_tok_s(f.with_suffix("").name.replace("_summary", ""))
        out.setdefault(model, {})[bench] = (score, elapsed, tok_s)

    # Tool calling — combine jdhodges + veerman per model into one number
    tc_pairs: dict[str, dict[str, dict]] = {}
    for f in sorted(RUNS.glob("toolcall_*_summary.json")):
        d = json.loads(f.read_text())
        model = d["model"]
        suite = d["suite"]
        tc_pairs.setdefault(model, {})[suite] = d
    for model, suites in tc_pairs.items():
        jd = suites.get("jdhodges")
        ve = suites.get("veerman")
        if not (jd and ve):
            continue
        correct = jd["correct"] + ve["correct"]
        total = jd["total"] + ve["total"]
        elapsed = jd.get("elapsed_s", 0) + ve.get("elapsed_s", 0)
        tok_s = (jd.get("fresh_tok_weighted_tps", 0) + ve.get("fresh_tok_weighted_tps", 0)) / 2
        out.setdefault(model, {})["tool_combined"] = (correct / total, elapsed, tok_s)

    return out


def _avg_tok_s(run_name: str) -> float:
    """Average tok/s across all questions in a JSONL run."""
    p = RUNS / f"{run_name}.jsonl"
    if not p.exists():
        return 0.0
    rates = []
    with p.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                t = row.get("tok_s") or 0
                if t > 0:
                    rates.append(t)
            except json.JSONDecodeError:
                continue
    return sum(rates) / len(rates) if rates else 0.0


# Display name mapping (LM Studio model id → friendly label)
DISPLAY = {
    "qwen/qwen3-coder-next": "qwen3-coder-next 80B/3B (this rig, MLX 6-bit)",
    "qwen3.6-27b": "qwen3.6-27b dense (this rig, MLX 6-bit)",
    "qwen3.6-35b-a3b@6bit": "qwen3.6-35b-a3b (this rig, MLX 6-bit)",
    "qwen3.6-35b-a3b@8bit": "qwen3.6-35b-a3b (this rig, MLX 8-bit)",
    "gemma-4-26b-a4b-it-mlx@4bit": "Gemma 4 26B-A4B (this rig, MLX 4-bit)",
    "gemma-4-26b-a4b-it-mlx@6bit": "Gemma 4 26B-A4B (this rig, MLX 6-bit)",
    "gemma-4-31b-it-mlx": "Gemma 4 31B dense (this rig, MLX 8-bit)",
    "gemma-4-e4b-it-mlx": "Gemma 4 E4B (this rig, MLX 8-bit)",
    "/Users/vitor/.lmstudio/models/mlx-community/DeepSeek-V4-Flash-2bit-DQ":
        "DeepSeek-V4-Flash (this rig, MLX 2-bit DQ)",
}


def chart_scores(local: dict, out_path: Path) -> None:
    models_local = [m for m in DISPLAY if m in local]
    upstream_keys = list(UPSTREAM.keys())
    all_labels = [DISPLAY[m] for m in models_local] + upstream_keys
    n_models = len(all_labels)
    n_benches = len(BENCH_ORDER)

    width = 0.8 / n_models
    x = np.arange(n_benches)

    fig, ax = plt.subplots(figsize=(14, 6))
    palette = plt.get_cmap("tab10").colors

    for i, label in enumerate(all_labels):
        scores = []
        for b in BENCH_ORDER:
            if i < len(models_local):
                m = models_local[i]
                v = local[m].get(b)
                scores.append(v[0] * 100 if v else np.nan)
            else:
                v = UPSTREAM[label].get(b)
                scores.append(v * 100 if v is not None else np.nan)
        offset = (i - n_models / 2) * width + width / 2
        bars = ax.bar(x + offset, scores, width, label=label, color=palette[i % 10])
        for j, s in enumerate(scores):
            if not np.isnan(s):
                ax.annotate(f"{s:.0f}", xy=(x[j] + offset, s),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([BENCH_LABEL[b] for b in BENCH_ORDER])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Phase 1 — Mac Studio M4 Max 128 GB vs. upstream (M4 Air 32 GB)\nblank cells = not yet measured")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def chart_throughput(local: dict, out_path: Path) -> None:
    models_local = [m for m in DISPLAY if m in local]
    if not models_local:
        return
    benches = ["humaneval", "mmlu", "math", "drop", "gpqa",
               "livecodebench", "tool_combined", "tbench"]
    x = np.arange(len(benches))
    width = 0.8 / len(models_local)

    fig, ax = plt.subplots(figsize=(12, 5))
    palette = plt.get_cmap("Set2").colors
    for i, m in enumerate(models_local):
        rates = []
        for b in benches:
            v = local[m].get(b)
            rates.append(v[2] if v else np.nan)
        offset = (i - len(models_local) / 2) * width + width / 2
        ax.bar(x + offset, rates, width, label=DISPLAY[m], color=palette[i % 8])
        for j, r in enumerate(rates):
            if not np.isnan(r) and r > 0:
                ax.annotate(f"{r:.0f}", xy=(x[j] + offset, r),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([BENCH_LABEL[b] for b in benches])
    ax.set_ylabel("avg tok/s")
    ax.set_title("Phase 1 — throughput per benchmark (this rig only)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def chart_time(local: dict, out_path: Path) -> None:
    models_local = [m for m in DISPLAY if m in local]
    if not models_local:
        return
    benches = ["humaneval", "mmlu", "math", "drop", "gpqa",
               "livecodebench", "tool_combined", "tbench"]
    x = np.arange(len(benches))
    width = 0.8 / len(models_local)

    fig, ax = plt.subplots(figsize=(12, 5))
    palette = plt.get_cmap("Set1").colors
    for i, m in enumerate(models_local):
        mins = []
        for b in benches:
            v = local[m].get(b)
            mins.append((v[1] / 60) if v else np.nan)
        offset = (i - len(models_local) / 2) * width + width / 2
        ax.bar(x + offset, mins, width, label=DISPLAY[m], color=palette[i % 9])
        for j, mi in enumerate(mins):
            if not np.isnan(mi) and mi > 0:
                ax.annotate(f"{mi:.0f}m", xy=(x[j] + offset, mi),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([BENCH_LABEL[b] for b in benches])
    ax.set_ylabel("Wall-clock minutes (log scale)")
    ax.set_yscale("log")
    ax.set_title("Phase 1 — wall-clock per benchmark (this rig only)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8)
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    CHARTS.mkdir(parents=True, exist_ok=True)
    local = load_local_runs()
    print(f"Models found: {sorted(local.keys())}")
    for m, benches in local.items():
        print(f"  {m}: {sorted(benches.keys())}")
    chart_scores(local, CHARTS / "chart_m4max_phase1_scores.png")
    chart_throughput(local, CHARTS / "chart_m4max_phase1_throughput.png")
    chart_time(local, CHARTS / "chart_m4max_phase1_time.png")
    print(f"Charts written to {CHARTS}")


if __name__ == "__main__":
    main()
