#!/usr/bin/env python3
"""
Aggregate tool-calling benchmark runs into a master report.

Scans results/runs/toolcall_*.jsonl and the corresponding
_summary.json files, builds per-model + per-suite totals, category
breakdowns, speed stats, and writes a markdown report to
results/tool_calling_results.md.

Usage:
    python3 tool_call_report.py            # write report
    python3 tool_call_report.py --print    # also print to stdout
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

SCRIPT_DIR = Path(__file__).parent
REPO = SCRIPT_DIR.parent.parent
RUNS_DIR = REPO / "research" / "benchmarks" / "runs"
OUT_MD = REPO / "research" / "benchmarks" / "tool_calling_results.md"


def collect():
    """Return nested dict: results[model][suite] = {entries, summary}."""
    results = defaultdict(lambda: defaultdict(lambda: {"entries": [], "summary": None}))

    # Load every jsonl
    for path in sorted(RUNS_DIR.glob("toolcall_*.jsonl")):
        try:
            with open(path) as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    model = e.get("model") or e.get("model_slug")
                    suite = e.get("suite")
                    if not model or not suite:
                        continue
                    results[model][suite]["entries"].append(e)
        except Exception:
            continue

    # Load latest summaries
    for path in sorted(RUNS_DIR.glob("toolcall_*_summary.json")):
        try:
            s = json.loads(path.read_text())
            model = s.get("model") or s.get("model_slug")
            suite = s.get("suite")
            if not model or not suite:
                continue
            # Keep the summary with the highest `total` (most complete run)
            cur = results[model][suite]["summary"]
            if cur is None or s.get("total", 0) >= cur.get("total", 0):
                results[model][suite]["summary"] = s
        except Exception:
            continue

    return results


def dedupe_by_case_id(entries):
    """Keep the most recent entry per (case_id, suite, model)."""
    best = {}
    for e in entries:
        cid = e.get("case_id")
        if cid is None:
            continue
        ts = e.get("timestamp", "")
        if cid not in best or ts > best[cid].get("timestamp", ""):
            best[cid] = e
    return list(best.values())


def summarize_suite(entries):
    entries = dedupe_by_case_id(entries)
    total = len(entries)
    passed = sum(1 for e in entries if e.get("score_pass"))
    by_cat = defaultdict(lambda: {"pass": 0, "total": 0})
    latencies = []
    tok_per_sec = []
    engine_gen_tps = []   # engine-reported decode-only (mlx_vlm.server)
    engine_prompt_tps = []
    comp_tokens_total = 0
    elapsed_total = 0.0
    errors = 0
    error_types = defaultdict(int)
    for e in entries:
        cat = e.get("category") or "?"
        by_cat[cat]["total"] += 1
        if e.get("score_pass"):
            by_cat[cat]["pass"] += 1
        el = e.get("turn_1_elapsed_s") or 0
        ct = e.get("completion_tokens") or 0
        if el and ct:
            latencies.append(el)
            tok_per_sec.append(ct / el if el else 0)
            comp_tokens_total += ct
            elapsed_total += el
        egt = e.get("engine_generation_tps")
        if egt:
            engine_gen_tps.append(egt)
        ept = e.get("engine_prompt_tps")
        if ept:
            engine_prompt_tps.append(ept)
        if e.get("error"):
            errors += 1
            reason = e.get("score_reason") or "?"
            # Collapse to first colon for grouping
            key = reason.split(":")[0].strip() if reason else "error"
            error_types[key] += 1
        elif not e.get("score_pass"):
            reason = e.get("score_reason") or "?"
            key = reason.split(":")[0].strip() if reason else "fail"
            error_types[key] += 1
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0,
        "by_category": {k: dict(v) for k, v in by_cat.items()},
        "mean_latency_s": round(mean(latencies), 2) if latencies else 0,
        "median_latency_s": round(median(latencies), 2) if latencies else 0,
        "p95_latency_s": round(sorted(latencies)[int(len(latencies)*0.95)-1], 2) if latencies else 0,
        "mean_tok_s": round(mean(tok_per_sec), 1) if tok_per_sec else 0,
        "tok_weighted_tok_s": round(comp_tokens_total / elapsed_total, 1) if elapsed_total else 0,
        "engine_mean_gen_tps": round(mean(engine_gen_tps), 1) if engine_gen_tps else 0,
        "engine_mean_prompt_tps": round(mean(engine_prompt_tps), 1) if engine_prompt_tps else 0,
        "comp_tokens_total": comp_tokens_total,
        "elapsed_total_s": round(elapsed_total, 1),
        "errors": errors,
        "error_types": dict(error_types),
    }


def short_name(model):
    # Drop publisher prefix for display
    if "/" in model:
        return model.split("/", 1)[1]
    return model


def fmt_pct(p):
    return f"{p*100:.1f}%"


def fmt_cat(cat):
    if not cat or cat["total"] == 0:
        return "-"
    return f"{cat['pass']}/{cat['total']}"


def build_report(results):
    lines = []
    lines.append("# Tool-Calling Benchmark Results\n")
    lines.append("**Date:** 2026-04-14  \n")
    lines.append("**Hardware:** MacBook Air M4, 32GB, 120 GB/s, fanless  \n")
    lines.append("**Engine:** LM Studio (llama.cpp + MLX backends), OpenAI-compatible endpoint on :1234  \n")
    lines.append("**Harness:** `scripts/tool_call_bench.py` — single-turn unless `multi_turn` is set.  \n")
    lines.append("**Gen params:** `temperature=0.0, top_p=1.0, seed=42, tool_choice=auto`  \n")
    lines.append("**Thermal gate:** cooldown to ≤60°C GPU before each question (matches prior benches).  \n\n")

    lines.append("## Suites\n")
    lines.append("- **jdhodges** — 40 cases, 8 tool schemas, 5 categories (tool_selection / argument_accuracy / multi_tool / edge_cases / format_compliance). Test YAMLs reconstructed from the jdhodges 2026 blog post (original YAMLs not shipped in the downloadable zip). 4 prompts are verbatim from the blog; the other 36 follow the same tool schemas and category definitions.\n")
    lines.append("- **veerman** — 12 P1-P12 prompts with 3 tools (weather/files/meetings) from `github.com/MikeVeerman/tool-calling-benchmark`. Splits: action (P1-P4, P6-P8), restraint (P5, P9), hard (P10-P12 with negation / redundancy / implicit reasoning).\n\n")

    lines.append("## Headline results\n\n")
    lines.append("| Model | Size | jdhodges | veerman | Combined | jdhodges mean t/s | tok-wt t/s | Errors |\n")
    lines.append("|-------|------|----------|---------|----------|-------------------|------------|--------|\n")

    # Sort models by combined pass rate desc
    ranked = []
    for model, suites in results.items():
        jd = suites.get("jdhodges", {}).get("entries", [])
        vm = suites.get("veerman", {}).get("entries", [])
        jd_s = summarize_suite(jd) if jd else None
        vm_s = summarize_suite(vm) if vm else None
        total = 0
        passed = 0
        if jd_s:
            total += jd_s["total"]; passed += jd_s["passed"]
        if vm_s:
            total += vm_s["total"]; passed += vm_s["passed"]
        combined = passed / total if total else 0
        ranked.append((combined, model, jd_s, vm_s))
    ranked.sort(key=lambda x: -x[0])

    for combined, model, jd_s, vm_s in ranked:
        name = short_name(model)
        # Size lookup from a summary
        size = "?"
        for s_dict in results[model].values():
            if s_dict.get("summary"):
                hw = s_dict["summary"].get("hardware", {})
                break
        # Format cells
        jd_cell = f"{jd_s['passed']}/{jd_s['total']} ({fmt_pct(jd_s['pass_rate'])})" if jd_s else "-"
        vm_cell = f"{vm_s['passed']}/{vm_s['total']} ({fmt_pct(vm_s['pass_rate'])})" if vm_s else "-"
        c_cell = fmt_pct(combined)
        jd_tps = f"{jd_s['mean_tok_s']}" if jd_s else "-"
        jd_wtps = f"{jd_s['tok_weighted_tok_s']}" if jd_s else "-"
        errs_jd = jd_s["errors"] if jd_s else 0
        errs_vm = vm_s["errors"] if vm_s else 0
        lines.append(f"| {name} | {size} | {jd_cell} | {vm_cell} | **{c_cell}** | {jd_tps} | {jd_wtps} | {errs_jd + errs_vm} |\n")

    # Per-category breakdown
    lines.append("\n## jdhodges per-category breakdown\n\n")
    lines.append("| Model | Selection | Args | Multi-Tool | Edge | Format |\n")
    lines.append("|-------|-----------|------|------------|------|--------|\n")
    for combined, model, jd_s, vm_s in ranked:
        if not jd_s:
            continue
        cats = jd_s["by_category"]
        sel = fmt_cat(cats.get("tool_selection"))
        arg = fmt_cat(cats.get("argument_accuracy"))
        mt  = fmt_cat(cats.get("multi_tool"))
        ed  = fmt_cat(cats.get("edge_cases"))
        fm  = fmt_cat(cats.get("format_compliance"))
        lines.append(f"| {short_name(model)} | {sel} | {arg} | {mt} | {ed} | {fm} |\n")

    lines.append("\n## Veerman per-category breakdown\n\n")
    lines.append("| Model | Action (P1-P8) | Restraint (P5, P9) | Hard (P10-P12) |\n")
    lines.append("|-------|----------------|--------------------|----------------|\n")
    for combined, model, jd_s, vm_s in ranked:
        if not vm_s:
            continue
        cats = vm_s["by_category"]
        act = fmt_cat(cats.get("veerman_action"))
        res = fmt_cat(cats.get("veerman_restraint"))
        hrd = fmt_cat(cats.get("veerman_hard"))
        lines.append(f"| {short_name(model)} | {act} | {res} | {hrd} |\n")

    # Speed comparison
    lines.append("\n## Speed (jdhodges cases — typical short tool-call generations)\n\n")
    lines.append("| Model | mean latency (s) | median latency (s) | p95 latency (s) | mean t/s | tok-weighted t/s | tokens total |\n")
    lines.append("|-------|------------------|--------------------|----|----------|------------------|--------------|\n")
    for combined, model, jd_s, vm_s in ranked:
        if not jd_s:
            continue
        lines.append(f"| {short_name(model)} | {jd_s['mean_latency_s']} | {jd_s['median_latency_s']} | {jd_s['p95_latency_s']} | {jd_s['mean_tok_s']} | {jd_s['tok_weighted_tok_s']} | {jd_s['comp_tokens_total']} |\n")

    # Errors / failure reasons
    lines.append("\n## Failure reasons\n\n")
    for combined, model, jd_s, vm_s in ranked:
        total_failed = 0
        for s in (jd_s, vm_s):
            if s:
                total_failed += (s["total"] - s["passed"])
        if total_failed == 0:
            continue
        lines.append(f"### {short_name(model)}\n\n")
        for suite_name, s in [("jdhodges", jd_s), ("veerman", vm_s)]:
            if not s or not s["error_types"]:
                continue
            lines.append(f"**{suite_name}**: ")
            parts = [f"{k} ({v})" for k, v in sorted(s["error_types"].items(), key=lambda kv: -kv[1])]
            lines.append(", ".join(parts))
            lines.append("\n\n")

    lines.append("\n## Methodology notes\n\n")
    lines.append("- **Scoring** is deterministic schema-aware matching (tool name + arg values), ported from jdhodges' `eval_tool_calling.py`. No LLM-as-judge.\n")
    lines.append("- **Single-turn default**. A case with `multi_turn: true` runs a second turn where the model sees a stubbed tool result and must call the next tool in the chain.\n")
    lines.append("- **Edge cases** (tell-a-joke, how-photosynthesis-works, vague-request) pass iff the model does NOT call a tool.\n")
    lines.append("- **Veerman restraint prompts** P5 and P9 are the two prompts where the model should not call a tool (meta question + 'write a Python script').\n")
    lines.append("- **Thermal gate** waits for a valid GPU reading ≤60°C before each question; 'None' readings never count as ready.\n")
    lines.append("- Each model was loaded fresh via `lms load` (previous model unloaded first). Generation: `temperature=0.0, top_p=1.0, seed=42`.\n")

    return "".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()

    results = collect()
    report = build_report(results)
    OUT_MD.write_text(report)
    print(f"Wrote {OUT_MD} ({len(report)} bytes)")
    if args.print:
        print()
        print(report)
